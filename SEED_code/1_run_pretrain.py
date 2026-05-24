"""SimCLR unsupervised pre-training and fine-tuning for EEG classification.

Phase 1: single-dataset pre-training + fine-tuning
    python 1_run_pretrain.py --phase 1 --action both --dataset MDD
    python 1_run_pretrain.py --phase 1 --action both --dataset MDD --use_moe

Phase 2: multi-dataset pre-training
    python 1_run_pretrain.py --phase 2 --action pretrain --epochs_pretrain 50
    python 1_run_pretrain.py --phase 2 --action pretrain --epochs_pretrain 50 --use_moe

Pre-trained weights saved to Pretrained/{tag}/, fine-tune results to Results/{tag}/.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from model import (
    SimCLREncoder, MoESimCLREncoder, EEGLSTM, MoELayer,
    GaussianNoise, ChannelDropout, TimeShift, Compose, SimCLRTransform,
    ChannelAdapter,
)
from model.channel_adapter import Phase2SimCLR, Phase2MoESimCLR
from utils import (
    load_dataset_info, create_dataloaders,
    create_pretrain_loaders, create_multi_pretrain_loaders,
    start_log, stop_log, write_summary_txt,
    resolve_device,
)
from pretrainer import Pretrainer
from trainer import Trainer


ALL_DATASETS = ["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"]


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_augmentations(args):
    transform = Compose(
        GaussianNoise(std=args.aug_std),
        ChannelDropout(p=args.aug_dropout),
        TimeShift(max_shift=args.aug_shift),
    )
    return SimCLRTransform(transform)


def _make_pretrain_model(args, channels):
    """Build SimCLREncoder or MoESimCLREncoder based on args."""
    common = dict(
        chans=channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        bidirectional=True,
    )
    if args.use_moe:
        common.update(
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            moe_expert_mult=args.moe_expert_mult,
        )
        return MoESimCLREncoder(**common)
    return SimCLREncoder(**common)


def _make_phase2_pretrain_model(args, dataset_channels):
    """Build Phase2SimCLR or Phase2MoESimCLR based on args."""
    common = dict(
        dataset_channels=dataset_channels,
        unified_dim=args.unified_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        bidirectional=True,
    )
    if args.use_moe:
        common.update(
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            moe_expert_mult=args.moe_expert_mult,
        )
        return Phase2MoESimCLR(**common)
    return Phase2SimCLR(**common)


def _build_moe_downstream(LSTM_and_moe_state, channels, hidden_dim, num_layers,
                          num_classes, dropout, moe_num_experts, moe_top_k):
    """Build a downstream model with MoE: LSTM → MoE → Classifier."""
    feat_dim = hidden_dim * 2  # bidirectional

    class MoEDownstream(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=channels, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.moe = MoELayer(
                dim=feat_dim, num_experts=moe_num_experts,
                top_k=moe_top_k, dropout=dropout,
            )
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            out, (h_n, c_n) = self.lstm(x)
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
            feat, _ = self.moe(feat)
            return self.classifier(feat)

    model = MoEDownstream()
    # Load LSTM + MoE weights from pre-trained state dict
    model_state = model.state_dict()
    pretrained = {k: v for k, v in LSTM_and_moe_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
    model_state.update(pretrained)
    model.load_state_dict(model_state)
    return model


def _write_pretrain_summary(run_dir, args, config, history, phase_label):
    """Write human-readable summary.txt for a pre-training run."""
    lines = []
    lines.append(f"Phase:        {phase_label}")
    lines.append(f"Dataset:      {config.get('dataset', config.get('datasets', 'N/A'))}")
    lines.append(f"MoE:          {'Yes' if args.use_moe else 'No'}")
    lines.append(f"Channels:     {config.get('channels', config.get('unified_dim', 'N/A'))}")
    lines.append(f"Hidden dim:   {config['hidden_dim']}")
    lines.append(f"LSTM layers:  {config['num_layers']}")
    lines.append(f"Projection:   {config['proj_dim']}")
    if args.use_moe:
        lines.append(f"MoE experts:  {config.get('moe_num_experts', 'N/A')}")
        lines.append(f"MoE top-k:    {config.get('moe_top_k', 'N/A')}")
        lines.append(f"Balance wt:   {config.get('balance_weight', 'N/A')}")
    lines.append(f"LR:           {config['lr']}")
    lines.append(f"Batch size:   {config['batch_size']}")
    lines.append(f"AMP:          {config.get('amp', False)}")
    lines.append(f"Grad accum:   {config.get('grad_accum_steps', 1)}")
    lines.append(f"Epochs:       {config['epochs']}")
    lines.append(f"Temperature:  {config['temperature']}")
    lines.append(f"Use all data: {config['use_all_data']}")
    lines.append(f"Aug noise:    {config.get('aug_std', 'N/A')}")
    lines.append(f"Aug ch drop:  {config.get('aug_dropout', 'N/A')}")
    lines.append(f"Aug shift:    {config.get('aug_shift', 'N/A')}")

    results = []
    results.append(f"Final pretrain loss:  {history['train_losses'][-1]:.4f}")
    results.append(f"Initial pretrain loss: {history['train_losses'][0]:.4f}")
    if history.get('balance_losses') and any(b > 0 for b in history['balance_losses']):
        results.append(f"Final balance loss:   {history['balance_losses'][-1]:.4f}")

    sections = [
        ("Configuration", lines),
        ("Pre-training Results", results),
        ("Output Files",
         [f"Encoder:      {run_dir}/encoder.pt",
          f"Config:       {run_dir}/config.json",
          f"Metrics:      {run_dir}/pretrain_metrics.json",
          f"Log:          {run_dir}/run.log"]),
    ]
    write_summary_txt(run_dir, sections)


def _write_finetune_summary(run_dir, args, config, metrics, phase_label):
    """Write human-readable summary.txt for a fine-tuning run."""
    lines = []
    lines.append(f"Phase:         {phase_label}")
    lines.append(f"Dataset:       {config['dataset']}")
    lines.append(f"MoE:           {'Yes' if args.use_moe else 'No'}")
    lines.append(f"Channels:      {config['channels']}")
    lines.append(f"Classes:       {config['num_classes']}")
    lines.append(f"Hidden dim:    {config['hidden_dim']}")
    lines.append(f"LSTM layers:   {config['num_layers']}")
    lines.append(f"Encoder LR:    {config['encoder_lr']}")
    lines.append(f"Classifier LR: {config['classifier_lr']}")
    lines.append(f"Epochs:        {config['epochs']}")
    lines.append(f"Batch size:    {config['batch_size']}")
    lines.append(f"AMP:           {config.get('amp', False)}")
    lines.append(f"Grad accum:    {config.get('grad_accum_steps', 1)}")
    lines.append(f"Patience:      {config['patience']}")
    lines.append(f"Pretrained:    {config.get('pretrained_encoder', 'N/A')}")

    results = []
    results.append(f"Best val accuracy:   {metrics['best_val_accuracy']:.4f} (epoch {metrics['best_epoch']})")
    results.append(f"Final val accuracy:  {metrics['final_val_accuracy']:.4f}")
    results.append(f"Train loss (final):  {metrics['train_losses'][-1]:.4f}")
    results.append(f"Val loss (final):    {metrics['val_losses'][-1]:.4f}")

    sections = [
        ("Configuration", lines),
        ("Fine-tuning Results", results),
        ("Output Files",
         [f"Model:        {run_dir}/model.pt",
          f"Predictions:  {run_dir}/predictions.txt",
          f"Config:       {run_dir}/config.json",
          f"Metrics:      {run_dir}/metrics.json",
          f"Log:          {run_dir}/run.log"]),
    ]
    write_summary_txt(run_dir, sections)


def _phase1_pretrain(args):
    """Single-dataset SimCLR pre-training."""
    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moe_suffix = "_moe" if args.use_moe else ""
    tag = f"{args.dataset}_phase1{moe_suffix}_{timestamp}"
    run_dir = os.path.join("Pretrained", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    model = _make_pretrain_model(args, channels)

    train_loader = create_pretrain_loaders(
        args.dataset, args.batch_size, use_all_data=args.use_all_data
    )
    transform = _build_augmentations(args)

    pretrainer = Pretrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        epochs=args.epochs_pretrain,
        temperature=args.temperature,
        transform=transform,
        is_phase2=False,
        balance_weight=args.balance_weight if args.use_moe else 0.0,
        device=args.device,
        use_amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
    )

    moe_tag = "+MoE" if args.use_moe else ""
    print(f"=== Phase 1 Pretrain{moe_tag} | Dataset: {args.dataset} ===")
    print(f"Channels: {channels}, Time points: {time_points}")
    print(f"Encoder: LSTM hidden={args.hidden_dim} layers={args.num_layers} proj={args.proj_dim}")
    if args.use_moe:
        print(f"MoE: {args.moe_num_experts} experts, top-{args.moe_top_k}, "
              f"balance_weight={args.balance_weight}")
    print(f"LR: {args.lr} | Batch: {args.batch_size} | Epochs: {args.epochs_pretrain}")
    print(f"Temperature: {args.temperature} | Use all data: {args.use_all_data}")
    print(f"Device: {args.device}")
    print("-" * 40)

    history = pretrainer.train()

    encoder_path = os.path.join(run_dir, "encoder.pt")
    pretrainer.save_checkpoint(encoder_path)

    config = {
        "phase": 1, "use_moe": args.use_moe,
        "dataset": args.dataset, "channels": channels,
        "time_points": time_points, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "proj_dim": args.proj_dim,
        "dropout": args.dropout, "lr": args.lr,
        "epochs": args.epochs_pretrain, "batch_size": args.batch_size,
        "temperature": args.temperature, "use_all_data": args.use_all_data,
        "aug_std": args.aug_std, "aug_dropout": args.aug_dropout,
        "aug_shift": args.aug_shift, "amp": args.amp,
        "grad_accum_steps": args.grad_accum_steps,
    }
    if args.use_moe:
        config.update(moe_num_experts=args.moe_num_experts,
                      moe_top_k=args.moe_top_k,
                      balance_weight=args.balance_weight)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "pretrain_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    _write_pretrain_summary(run_dir, args, config, history, "Phase 1")

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return run_dir


def _phase1_finetune(args):
    """Fine-tune pre-trained encoder on a single dataset with gradient decay."""
    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moe_suffix = "_moe" if args.use_moe else ""
    model_name = "EEGLSTM" if not args.use_moe else "EEGLSTM_MoE"
    tag = f"{args.dataset}_{model_name}_pretrained{moe_suffix}_{timestamp}"
    run_dir = os.path.join("Results", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    state = torch.load(args.pretrained_encoder, map_location="cpu")

    if args.use_moe:
        model = _build_moe_downstream(
            state, channels, args.hidden_dim, args.num_layers,
            num_classes, args.dropout,
            args.moe_num_experts, args.moe_top_k,
        )
    else:
        model = EEGLSTM(
            chans=channels, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, num_classes=num_classes,
            dropout=args.dropout, bidirectional=True,
        )
        model.lstm.load_state_dict(state)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size, fold=args.fold
    )

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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.classifier_lr,
        epochs=args.epochs_finetune,
        optimizer=optimizer,
        patience=args.patience,
        device=args.device,
        use_amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
    )

    moe_tag = " +MoE" if args.use_moe else ""
    print(f"=== Phase 1 Finetune{moe_tag} | Dataset: {args.dataset} ===")
    print(f"Encoder LR: {args.encoder_lr} | Classifier LR: {args.classifier_lr}")
    print(f"Epochs: {args.epochs_finetune} | Batch: {args.batch_size} "
          f"| Patience: {args.patience} | Device: {args.device}")
    print("-" * 40)

    history = trainer.train()

    trainer.save_predictions(run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    metrics = {
        "final_val_accuracy": history["val_accuracies"][-1],
        "best_val_accuracy": history["best_val_accuracy"],
        "best_epoch": history["best_epoch"],
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "val_accuracies": history["val_accuracies"],
        "pretrained_encoder": args.pretrained_encoder,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "phase": 1, "action": "finetune", "use_moe": args.use_moe,
        "dataset": args.dataset, "channels": channels,
        "num_classes": num_classes, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "dropout": args.dropout,
        "encoder_lr": args.encoder_lr, "classifier_lr": args.classifier_lr,
        "epochs": args.epochs_finetune, "batch_size": args.batch_size,
        "patience": args.patience, "amp": args.amp,
        "grad_accum_steps": args.grad_accum_steps,
        "pretrained_encoder": args.pretrained_encoder,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _write_finetune_summary(run_dir, args, config, metrics, "Phase 1")

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return run_dir


def _phase2_pretrain(args):
    """Multi-dataset SimCLR pre-training with channel adapters."""
    datasets = args.datasets.split(",") if args.datasets else ALL_DATASETS

    dataset_channels = {}
    for name in datasets:
        ch, _, _ = load_dataset_info(name)
        dataset_channels[name] = ch

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moe_suffix = "_moe" if args.use_moe else ""
    tag = f"multi_phase2{moe_suffix}_{timestamp}"
    run_dir = os.path.join("Pretrained", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    moe_tag = "+MoE" if args.use_moe else ""
    print(f"=== Phase 2 Pretrain{moe_tag} | Multi-dataset ===")
    print(f"Datasets: {datasets}")
    for name, ch in dataset_channels.items():
        print(f"  {name}: {ch} channels")
    print(f"Unified dim: {args.unified_dim}")
    if args.use_moe:
        print(f"MoE: {args.moe_num_experts} experts, top-{args.moe_top_k}")

    model = _make_phase2_pretrain_model(args, dataset_channels)

    train_loader, _ = create_multi_pretrain_loaders(
        datasets, args.batch_size, use_train_only=not args.use_all_data
    )
    transform = _build_augmentations(args)

    pretrainer = Pretrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        epochs=args.epochs_pretrain,
        temperature=args.temperature,
        transform=transform,
        is_phase2=True,
        balance_weight=args.balance_weight if args.use_moe else 0.0,
        device=args.device,
        use_amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
    )

    print(f"LR: {args.lr} | Batch: {args.batch_size} | Epochs: {args.epochs_pretrain}")
    print(f"Temperature: {args.temperature} | Device: {args.device}")
    print("-" * 40)

    history = pretrainer.train()

    torch.save(model.get_encoder_state_dict(), os.path.join(run_dir, "encoder.pt"))
    torch.save(model.get_adapter_state_dict(), os.path.join(run_dir, "adapter.pt"))

    config = {
        "phase": 2, "use_moe": args.use_moe,
        "datasets": datasets, "dataset_channels": dataset_channels,
        "unified_dim": args.unified_dim, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "proj_dim": args.proj_dim,
        "dropout": args.dropout, "lr": args.lr,
        "epochs": args.epochs_pretrain, "batch_size": args.batch_size,
        "temperature": args.temperature, "use_all_data": args.use_all_data,
        "aug_std": args.aug_std, "aug_dropout": args.aug_dropout,
        "aug_shift": args.aug_shift, "amp": args.amp,
        "grad_accum_steps": args.grad_accum_steps,
    }
    if args.use_moe:
        config.update(moe_num_experts=args.moe_num_experts,
                      moe_top_k=args.moe_top_k,
                      balance_weight=args.balance_weight)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "pretrain_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    _write_pretrain_summary(run_dir, args, config, history, "Phase 2")

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return run_dir


def _phase2_finetune(args):
    """Fine-tune Phase 2 pre-trained encoder on a single dataset."""
    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moe_suffix = "_moe" if args.use_moe else ""
    tag = f"{args.dataset}_EEGLSTM_phase2{moe_suffix}_{timestamp}"
    run_dir = os.path.join("Results", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    adapter_state = torch.load(args.pretrained_adapter, map_location="cpu")
    adapter = ChannelAdapter({args.dataset: channels}, unified_dim=args.unified_dim)
    adapter_key = args.dataset
    adapter.adapters[adapter_key].load_state_dict(
        {k.replace(f"adapters.{adapter_key}.", ""): v
         for k, v in adapter_state.items()
         if k.startswith(f"adapters.{adapter_key}")}
    )

    encoder_state = torch.load(args.pretrained_encoder, map_location="cpu")

    if args.use_moe:
        inner = _build_moe_downstream(
            encoder_state, args.unified_dim, args.hidden_dim,
            args.num_layers, num_classes, args.dropout,
            args.moe_num_experts, args.moe_top_k,
        )
    else:
        inner = EEGLSTM(
            chans=args.unified_dim, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, num_classes=num_classes,
            dropout=args.dropout, bidirectional=True,
        )
        inner.lstm.load_state_dict(encoder_state)

    class FinetuneModel(nn.Module):
        def __init__(self, adapter, encoder, dataset_name):
            super().__init__()
            self.adapter = adapter
            self.encoder = encoder
            self.dataset_name = dataset_name

        def forward(self, x):
            src = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self.adapter(x, src)
            return self.encoder(x)

    ft_model = FinetuneModel(adapter, inner, args.dataset)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size, fold=args.fold
    )

    classifier_params = []
    other_params = []
    for name, param in ft_model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": args.encoder_lr},
        {"params": classifier_params, "lr": args.classifier_lr},
    ])

    trainer = Trainer(
        model=ft_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.classifier_lr,
        epochs=args.epochs_finetune,
        optimizer=optimizer,
        patience=args.patience,
        device=args.device,
    )

    moe_tag = " +MoE" if args.use_moe else ""
    print(f"=== Phase 2 Finetune{moe_tag} | Dataset: {args.dataset} ===")
    print(f"Encoder LR: {args.encoder_lr} | Classifier LR: {args.classifier_lr}")
    print(f"Epochs: {args.epochs_finetune} | Batch: {args.batch_size} "
          f"| Patience: {args.patience}")
    print("-" * 40)

    history = trainer.train()

    trainer.save_predictions(run_dir)
    torch.save(ft_model.state_dict(), os.path.join(run_dir, "model.pt"))

    metrics = {
        "final_val_accuracy": history["val_accuracies"][-1],
        "best_val_accuracy": history["best_val_accuracy"],
        "best_epoch": history["best_epoch"],
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "val_accuracies": history["val_accuracies"],
        "pretrained_encoder": args.pretrained_encoder,
        "pretrained_adapter": args.pretrained_adapter,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "phase": 2, "action": "finetune", "use_moe": args.use_moe,
        "dataset": args.dataset, "channels": channels,
        "num_classes": num_classes, "unified_dim": args.unified_dim,
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "dropout": args.dropout, "encoder_lr": args.encoder_lr,
        "classifier_lr": args.classifier_lr,
        "epochs": args.epochs_finetune, "batch_size": args.batch_size,
        "patience": args.patience, "amp": args.amp,
        "grad_accum_steps": args.grad_accum_steps,
        "pretrained_encoder": args.pretrained_encoder,
        "pretrained_adapter": args.pretrained_adapter,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _write_finetune_summary(run_dir, args, config, metrics, "Phase 2")

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR unsupervised pre-training for EEG classification"
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2,
                        help="Phase 1: single-dataset; Phase 2: multi-dataset")
    parser.add_argument("--action", type=str, choices=["pretrain", "finetune", "both"],
                        default="finetune")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (Phase 1 or Phase 2 finetune target)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset list for Phase 2 pretrain (default: all 5)")

    # Pre-training hyperparams
    parser.add_argument("--lr", type=float, default=5e-4, help="Pre-training LR")
    parser.add_argument("--epochs_pretrain", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (larger for SimCLR)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients for this many micro-batches")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--use_all_data", action="store_true",
                        help="Include val.h5 and test_x_only.h5 in pre-training")

    # Fine-tuning hyperparams
    parser.add_argument("--epochs_finetune", type=int, default=30)
    parser.add_argument("--encoder_lr", type=float, default=5e-5,
                        help="Encoder LR during fine-tuning (1/10 of classifier)")
    parser.add_argument("--classifier_lr", type=float, default=5e-4,
                        help="Classifier LR during fine-tuning")
    parser.add_argument("--fold", type=int, default=None,
                        help="CV fold 1-5 (requires prepared folds)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unified_dim", type=int, default=64,
                        help="Phase 2 unified channel dimension")

    # MoE
    parser.add_argument("--use_moe", action="store_true",
                        help="Enable Mixture of Experts layer")
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--moe_expert_mult", type=int, default=4)
    parser.add_argument("--balance_weight", type=float, default=0.01,
                        help="MoE load balancing loss weight")

    # Augmentation
    parser.add_argument("--aug_std", type=float, default=0.05)
    parser.add_argument("--aug_dropout", type=float, default=0.1)
    parser.add_argument("--aug_shift", type=int, default=10)

    # Checkpoint paths for fine-tuning
    parser.add_argument("--pretrained_encoder", type=str, default="Pretrained/multi_phase2_moe_20260427_233324/encoder.pt",
                        help="Path to encoder.pt for fine-tuning")
    parser.add_argument("--pretrained_adapter", type=str, default="Pretrained/multi_phase2_moe_20260427_233324/adapter.pt",
                        help="Path to adapter.pt for Phase 2 fine-tuning")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device: cpu, cuda, cuda:0, cuda:3, etc. (auto-fallback if CUDA missing)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training on CUDA")

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    _set_seed(args.seed)

    if args.phase == 1:
        if args.action in ("pretrain", "both"):
            result_dir = _phase1_pretrain(args)
            if args.action == "both":
                args.pretrained_encoder = os.path.join(result_dir, "encoder.pt")
        if args.action in ("finetune", "both"):
            if args.pretrained_encoder is None:
                parser.error("--pretrained_encoder is required for finetune action")
            _phase1_finetune(args)

    elif args.phase == 2:
        if args.action in ("pretrain", "both"):
            result_dir = _phase2_pretrain(args)
            if args.action == "both":
                args.pretrained_encoder = os.path.join(result_dir, "encoder.pt")
                args.pretrained_adapter = os.path.join(result_dir, "adapter.pt")
        if args.action in ("finetune", "both"):
            if args.pretrained_encoder is None:
                parser.error("--pretrained_encoder is required for finetune action")
            if args.pretrained_adapter is None:
                parser.error("--pretrained_adapter is required for Phase 2 finetune")
            _phase2_finetune(args)


if __name__ == "__main__":
    main()
