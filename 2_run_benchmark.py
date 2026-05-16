"""Systematic benchmark: baseline training, fine-tuning, and comparison across 5 datasets.

Modes:
    baseline  — train models from scratch on all datasets
    finetune  — fine-tune pre-trained encoder on all datasets
    compare   — print comparison table from saved results

Usage:
    python 2_run_benchmark.py --mode baseline --model EEGLSTM --epochs 30
    python 2_run_benchmark.py --mode finetune --encoder Pretrained/multi_phase2_xxx/encoder.pt --adapter Pretrained/multi_phase2_xxx/adapter.pt
    python 2_run_benchmark.py --mode compare
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from model import MODEL_DICT, EEGLSTM, ChannelAdapter, MoELayer
from utils import (
    load_dataset_info, create_dataloaders,
    start_log, stop_log, write_summary_txt,
    resolve_device,
)
from trainer import Trainer

ALL_DATASETS = ["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"]


def _set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _run_single_train(dataset, model_name, args):
    """Train one model from scratch on one dataset. Returns metrics dict."""
    channels, num_classes, window_sec = load_dataset_info(dataset)
    time_points = int(window_sec * 200)

    tag = f"{dataset}_{model_name}_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("Results", "benchmark_baseline", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, args.batch_size, fold=args.fold
    )

    model_cls = MODEL_DICT[model_name]
    if model_name in ("SimpleLinear", "SimpleMLP"):
        model = model_cls(input_channels=channels, time_points=time_points, num_classes=num_classes)
    elif model_name == "EEGNet":
        model = model_cls(chans=channels, num_classes=num_classes, time_point=time_points)
    else:
        model = model_cls(chans=channels, num_classes=num_classes,
                          hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                          dropout=args.dropout)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, lr=args.lr, epochs=args.epochs,
        patience=args.patience, device=args.device,
        use_amp=args.amp, grad_accum_steps=args.grad_accum_steps,
    )

    print(f"\n{'='*50}\n  {dataset} | {model_name} | from scratch\n{'='*50}")
    history = trainer.train()

    trainer.save_predictions(run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    metrics = {
        "dataset": dataset, "model": model_name,
        "final_val_accuracy": history["val_accuracies"][-1],
        "best_val_accuracy": history["best_val_accuracy"],
        "best_epoch": history["best_epoch"],
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "val_accuracies": history["val_accuracies"],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _write_bench_summary(run_dir, dataset, "baseline", metrics, args)

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return metrics


def _run_single_finetune(dataset, args):
    """Fine-tune pre-trained encoder on one dataset. Returns metrics dict."""
    channels, num_classes, window_sec = load_dataset_info(dataset)
    time_points = int(window_sec * 200)

    tag = f"{dataset}_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join("Results", "benchmark_finetune", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, args.batch_size, fold=args.fold
    )

    encoder_state = torch.load(args.encoder, map_location="cpu")
    has_moe = any(key.startswith("moe.") for key in encoder_state)

    if args.adapter:
        adapter_state = torch.load(args.adapter, map_location="cpu")
        adapter = ChannelAdapter({dataset: channels}, unified_dim=args.unified_dim)
        adapter_key = dataset
        adapter.adapters[adapter_key].load_state_dict(
            {k.replace(f"adapters.{adapter_key}.", ""): v
             for k, v in adapter_state.items()
             if k.startswith(f"adapters.{adapter_key}")}
        )
        input_channels = args.unified_dim
    else:
        adapter = None
        input_channels = channels

    if has_moe:
        encoder = _build_moe_downstream(
            chans=input_channels, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, num_classes=num_classes,
            dropout=args.dropout,
        )
    else:
        encoder = EEGLSTM(
            chans=input_channels, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, num_classes=num_classes,
            dropout=args.dropout, bidirectional=True,
        )

    _load_matching_state(encoder, encoder_state)

    if adapter is not None:
        class FinetuneModel(torch.nn.Module):
            def __init__(self, adapter, encoder, dataset_name):
                super().__init__()
                self.adapter = adapter
                self.encoder = encoder
                self.dataset_name = dataset_name

            def forward(self, x):
                src = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                x = self.adapter(x, src)
                return self.encoder(x)

        model = FinetuneModel(adapter, encoder, dataset)
    else:
        model = encoder

    classifier_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": classifier_params, "lr": args.classifier_lr},
    ])

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, lr=args.classifier_lr, epochs=args.epochs,
        optimizer=optimizer, patience=args.patience, device=args.device,
        use_amp=args.amp, grad_accum_steps=args.grad_accum_steps,
    )

    print(f"\n{'='*50}\n  {dataset} | Finetune | encoder={args.encoder}\n{'='*50}")
    history = trainer.train()

    trainer.save_predictions(run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    metrics = {
        "dataset": dataset,
        "encoder": args.encoder, "adapter": args.adapter,
        "final_val_accuracy": history["val_accuracies"][-1],
        "best_val_accuracy": history["best_val_accuracy"],
        "best_epoch": history["best_epoch"],
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "val_accuracies": history["val_accuracies"],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _write_bench_summary(run_dir, dataset, "finetune", metrics, args)

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return metrics


def _write_bench_summary(run_dir, dataset, mode, metrics, args):
    """Write human-readable summary.txt for a single benchmark run."""
    if mode == "baseline":
        heading = "Baseline Training"
        cfg = [
            f"Dataset:      {dataset}",
            f"Model:        {metrics['model']}",
            f"LR:           {args.lr}",
            f"Epochs:       {args.epochs}",
            f"Batch size:   {args.batch_size}",
            f"Patience:     {args.patience}",
            f"Hidden dim:   {args.hidden_dim}",
            f"LSTM layers:  {args.num_layers}",
        ]
    else:
        heading = "Fine-tuning"
        cfg = [
            f"Dataset:      {dataset}",
            f"Encoder:      {metrics['encoder']}",
            f"Adapter:      {metrics.get('adapter', 'N/A')}",
            f"Encoder LR:   {args.encoder_lr}",
            f"Classifier LR: {args.classifier_lr}",
            f"Epochs:       {args.epochs}",
            f"Batch size:   {args.batch_size}",
            f"Patience:     {args.patience}",
            f"Hidden dim:   {args.hidden_dim}",
            f"LSTM layers:  {args.num_layers}",
        ]

    results = [
        f"Best val accuracy:   {metrics['best_val_accuracy']:.4f} (epoch {metrics['best_epoch']})",
        f"Final val accuracy:  {metrics['final_val_accuracy']:.4f}",
    ]

    sections = [
        (heading, cfg),
        ("Results", results),
        ("Output Files",
         [f"Model:        {run_dir}/model.pt",
          f"Predictions:  {run_dir}/predictions.txt",
          f"Metrics:      {run_dir}/metrics.json",
          f"Log:          {run_dir}/run.log"]),
    ]
    write_summary_txt(run_dir, sections)


def _build_moe_downstream(chans, hidden_dim, num_layers, num_classes, dropout,
                          moe_num_experts=4, moe_top_k=2):
    """Build a downstream LSTM -> MoE -> classifier model."""
    feat_dim = hidden_dim * 2

    class MoEDownstream(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=chans,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.moe = MoELayer(
                dim=feat_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                dropout=dropout,
            )
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            _, (h_n, _) = self.lstm(x)
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
            feat, _ = self.moe(feat)
            return self.classifier(feat)

    return MoEDownstream()


def _load_matching_state(module, state_dict):
    """Load only compatible tensors from a checkpoint into a module."""
    module_state = module.state_dict()
    matched = {
        key: value for key, value in state_dict.items()
        if key in module_state and module_state[key].shape == value.shape
    }
    module_state.update(matched)
    module.load_state_dict(module_state)


def _mode_baseline(args):
    """Run baseline training on all datasets."""
    datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS
    results = {}
    for ds in datasets:
        _set_seed(args.seed)
        try:
            m = _run_single_train(ds, args.model, args)
            results[ds] = {"best_val_accuracy": m["best_val_accuracy"]}
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            results[ds] = {"best_val_accuracy": None, "error": str(e)}

    _print_summary_table("Baseline", results)


def _mode_finetune(args):
    """Run fine-tuning on all datasets."""
    datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS
    results = {}
    for ds in datasets:
        _set_seed(args.seed)
        try:
            m = _run_single_finetune(ds, args)
            results[ds] = {"best_val_accuracy": m["best_val_accuracy"]}
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            results[ds] = {"best_val_accuracy": None, "error": str(e)}

    _print_summary_table("Finetune", results)


def _print_summary_table(title, results):
    print(f"\n{'='*60}")
    print(f"{title} Summary")
    print(f"{'Dataset':<10} {'Best Val Acc'}")
    print("-" * 30)
    for ds, r in results.items():
        acc = r["best_val_accuracy"]
        print(f"{ds:<10} {acc:.4f}" if acc is not None else f"{ds:<10} ERROR")


def _mode_compare(args):
    """Generate comparison table from saved benchmark results."""
    baseline_dir = os.path.join("Results", "benchmark_baseline")
    finetune_dir = os.path.join("Results", "benchmark_finetune")

    baselines = {}
    if os.path.isdir(baseline_dir):
        for d in os.listdir(baseline_dir):
            path = os.path.join(baseline_dir, d, "metrics.json")
            if os.path.isfile(path):
                with open(path) as f:
                    m = json.load(f)
                ds = m["dataset"]
                acc = m["best_val_accuracy"]
                if ds not in baselines or acc > baselines[ds]:
                    baselines[ds] = acc

    finetunes = {}
    if os.path.isdir(finetune_dir):
        for d in os.listdir(finetune_dir):
            path = os.path.join(finetune_dir, d, "metrics.json")
            if os.path.isfile(path):
                with open(path) as f:
                    m = json.load(f)
                ds = m["dataset"]
                acc = m["best_val_accuracy"]
                if ds not in finetunes or acc > finetunes[ds]:
                    finetunes[ds] = acc

    # Print to console
    header = f"{'Dataset':<10} {'Baseline':<12} {'Finetune':<12} {'Delta':<10}"
    sep = "-" * 50
    lines = [f"\n{'='*60}", "Benchmark Comparison", header, sep]

    for ds in ALL_DATASETS:
        b = baselines.get(ds)
        f = finetunes.get(ds)
        b_str = f"{b:.4f}" if b is not None else "N/A"
        f_str = f"{f:.4f}" if f is not None else "N/A"
        if b is not None and f is not None:
            delta_str = f"{f - b:+.4f}"
        else:
            delta_str = "N/A"
        lines.append(f"{ds:<10} {b_str:<12} {f_str:<12} {delta_str:<10}")

    for line in lines:
        print(line)

    # Write comparison report
    report_path = os.path.join("Results", "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Benchmark Comparison Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for line in lines:
            f.write(line + "\n")
        f.write(f"\nSource: {baseline_dir}\n")
        f.write(f"Source: {finetune_dir}\n")
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Systematic benchmark: baseline, finetune, compare across datasets"
    )
    parser.add_argument("--mode", type=str, choices=["baseline", "finetune", "compare"],
                        required=True)
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset list (default: all 5)")
    parser.add_argument("--model", type=str, default="EEGLSTM",
                        choices=list(MODEL_DICT.keys()))

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--encoder", type=str, default="Pretrained/multi_phase2_20260428_000128/encoder.pt",
                        help="Path to pre-trained encoder.pt")
    parser.add_argument("--adapter", type=str, default="Pretrained/multi_phase2_20260428_000128/adapter.pt",
                        help="Path to pre-trained adapter.pt (Phase 2)")
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--classifier_lr", type=float, default=5e-4)
    parser.add_argument("--unified_dim", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients for this many micro-batches")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training on CUDA")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device: cpu, cuda, cuda:0, etc. (auto-fallback if CUDA missing)")

    args = parser.parse_args()
    args.device = resolve_device(args.device)

    if args.mode == "baseline":
        _mode_baseline(args)
    elif args.mode == "finetune":
        if args.encoder is None:
            parser.error("--encoder is required for finetune mode")
        _mode_finetune(args)
    elif args.mode == "compare":
        _mode_compare(args)


if __name__ == "__main__":
    main()
