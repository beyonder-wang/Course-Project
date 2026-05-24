"""Multi-band SimCLR pre-training + fine-tuning for EEG classification.

Pre-training: decomposes EEG into 5 frequency bands (delta/theta/alpha/beta/gamma),
  uses shared LSTM encoder with per-band projection heads and multi-head NT-Xent loss.
  Reference: DGNet (ICLR 2026) — ~25% relative improvement over standard SimCLR.

Fine-tuning: standard supervised training with EEGLSTM, loading pre-trained LSTM weights.

Usage:
    # Pre-train only
    python 3_run_multiband_simclr.py --action pretrain --dataset MDD --epochs_pretrain 50

    # Pre-train + auto fine-tune
    python 3_run_multiband_simclr.py --action both --dataset MDD --epochs_pretrain 50 --epochs_finetune 30

    # With MoE
    python 3_run_multiband_simclr.py --action both --dataset MDD --use_moe --moe_num_experts 4

    # Fine-tune only (provide encoder path)
    python 3_run_multiband_simclr.py --action finetune --dataset MDD --pretrained_encoder Pretrained/MDD_multiband_xxx/encoder.pt

Output:
    Pretrained/{tag}/  — encoder.pt, config.json, pretrain_metrics.json, summary.txt, run.log
    Results/{tag}/     — model.pt, predictions.txt, config.json, metrics.json, summary.txt, run.log
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from model import EEGLSTM, MoELayer
from model.augmentations import (
    GaussianNoise, ChannelDropout, TimeShift, Compose, SimCLRTransform,
)
from model.multiband_simclr import MultiBandSimCLREncoder, MultiBandMoESimCLREncoder
from utils import (
    load_dataset_info, create_dataloaders, create_pretrain_loaders,
    start_log, stop_log, write_summary_txt,
    resolve_device,
)
from pretrainer_multiband import MultiBandPretrainer
from trainer import Trainer


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


def _make_multiband_model(args, channels):
    """Build MultiBandSimCLREncoder or MultiBandMoESimCLREncoder."""
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
        return MultiBandMoESimCLREncoder(**common)
    return MultiBandSimCLREncoder(**common)


def _build_moe_downstream(pretrained_state, channels, hidden_dim, num_layers,
                          num_classes, dropout, moe_num_experts, moe_top_k):
    """Build downstream model with LSTM -> MoE -> Classifier, loading pre-trained weights."""
    feat_dim = hidden_dim * 2

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
    model_state = model.state_dict()
    matched = {k: v for k, v in pretrained_state.items()
               if k in model_state and model_state[k].shape == v.shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    return model


def _write_pretrain_summary(run_dir, args, config, history):
    lines = [
        f"Method:       Multi-band SimCLR",
        f"Dataset:      {config['dataset']}",
        f"MoE:          {'Yes' if args.use_moe else 'No'}",
        f"Channels:     {config['channels']}",
        f"Hidden dim:   {config['hidden_dim']}",
        f"LSTM layers:  {config['num_layers']}",
        f"Projection:   {config['proj_dim']}",
    ]
    if args.use_moe:
        lines += [
            f"MoE experts:  {config.get('moe_num_experts', 'N/A')}",
            f"MoE top-k:    {config.get('moe_top_k', 'N/A')}",
            f"Balance wt:   {config.get('balance_weight', 'N/A')}",
        ]
    lines += [
        f"LR:           {config['lr']}",
        f"Batch size:   {config['batch_size']}",
        f"Epochs:       {config['epochs']}",
        f"Temperature:  {config['temperature']}",
        f"Use all data: {config['use_all_data']}",
        f"Aug noise:    {config.get('aug_std', 'N/A')}",
        f"Aug ch drop:  {config.get('aug_dropout', 'N/A')}",
        f"Aug shift:    {config.get('aug_shift', 'N/A')}",
    ]
    results = [
        f"Final pretrain loss:  {history['train_losses'][-1]:.4f}",
        f"Initial pretrain loss: {history['train_losses'][0]:.4f}",
    ]
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


def _write_finetune_summary(run_dir, args, config, metrics):
    lines = [
        f"Method:         Multi-band SimCLR (fine-tune)",
        f"Dataset:       {config['dataset']}",
        f"MoE:           {'Yes' if args.use_moe else 'No'}",
        f"Channels:      {config['channels']}",
        f"Classes:       {config['num_classes']}",
        f"Hidden dim:    {config['hidden_dim']}",
        f"LSTM layers:   {config['num_layers']}",
        f"Encoder LR:    {config['encoder_lr']}",
        f"Classifier LR: {config['classifier_lr']}",
        f"Epochs:        {config['epochs']}",
        f"Batch size:    {config['batch_size']}",
        f"Patience:      {config['patience']}",
        f"Pretrained:    {config.get('pretrained_encoder', 'N/A')}",
    ]
    results = [
        f"Best val accuracy:   {metrics['best_val_accuracy']:.4f} (epoch {metrics['best_epoch']})",
        f"Final val accuracy:  {metrics['final_val_accuracy']:.4f}",
        f"Train loss (final):  {metrics['train_losses'][-1]:.4f}",
        f"Val loss (final):    {metrics['val_losses'][-1]:.4f}",
    ]
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


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def run_pretrain(args):
    """Multi-band SimCLR pre-training on a single dataset."""
    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moe_suffix = "_moe" if args.use_moe else ""
    tag = f"{args.dataset}_multiband{moe_suffix}_{timestamp}"
    run_dir = os.path.join("Pretrained", tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    model = _make_multiband_model(args, channels)

    train_loader = create_pretrain_loaders(
        args.dataset, args.batch_size, use_all_data=args.use_all_data
    )
    transform = _build_augmentations(args)

    pretrainer = MultiBandPretrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        epochs=args.epochs_pretrain,
        temperature=args.temperature,
        transform=transform,
        balance_weight=args.balance_weight if args.use_moe else 0.0,
        device=args.device,
    )

    moe_tag = " + MoE" if args.use_moe else ""
    print(f"=== Multi-Band SimCLR Pre-train{moe_tag} | Dataset: {args.dataset} ===")
    print(f"Channels: {channels}, Time points: {time_points}")
    print(f"Bands: delta/theta/alpha/beta/gamma")
    print(f"Encoder: LSTM hidden={args.hidden_dim} layers={args.num_layers} proj={args.proj_dim}")
    if args.use_moe:
        print(f"MoE: {args.moe_num_experts} experts, top-{args.moe_top_k}, "
              f"balance_weight={args.balance_weight}")
    print(f"LR: {args.lr} | Batch: {args.batch_size} | Epochs: {args.epochs_pretrain}")
    print(f"Temperature: {args.temperature} | Use all data: {args.use_all_data}")
    print(f"Device: {args.device}")
    print("-" * 50)

    history = pretrainer.train()

    encoder_path = os.path.join(run_dir, "encoder.pt")
    pretrainer.save_checkpoint(encoder_path)

    config = {
        "method": "multiband_simclr",
        "dataset": args.dataset,
        "channels": channels,
        "time_points": time_points,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "proj_dim": args.proj_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs_pretrain,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "use_all_data": args.use_all_data,
        "use_moe": args.use_moe,
        "aug_std": args.aug_std,
        "aug_dropout": args.aug_dropout,
        "aug_shift": args.aug_shift,
    }
    if args.use_moe:
        config.update(
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            balance_weight=args.balance_weight,
        )

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "pretrain_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    _write_pretrain_summary(run_dir, args, config, history)

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return run_dir


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def run_finetune(args):
    """Fine-tune multi-band pre-trained encoder on a single dataset."""
    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moe_suffix = "_moe" if args.use_moe else ""
    tag = f"{args.dataset}_EEGLSTM_multiband{moe_suffix}_{timestamp}"
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
    )

    moe_tag = " + MoE" if args.use_moe else ""
    print(f"=== Multi-Band SimCLR Fine-tune{moe_tag} | Dataset: {args.dataset} ===")
    print(f"Encoder LR: {args.encoder_lr} | Classifier LR: {args.classifier_lr}")
    print(f"Epochs: {args.epochs_finetune} | Batch: {args.batch_size} "
          f"| Patience: {args.patience} | Device: {args.device}")
    print("-" * 50)

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
        "method": "multiband_simclr",
        "action": "finetune",
        "dataset": args.dataset,
        "channels": channels,
        "num_classes": num_classes,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_moe": args.use_moe,
        "encoder_lr": args.encoder_lr,
        "classifier_lr": args.classifier_lr,
        "epochs": args.epochs_finetune,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "pretrained_encoder": args.pretrained_encoder,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _write_finetune_summary(run_dir, args, config, metrics)

    print(f"Saved to: {run_dir}/")
    stop_log(tee)
    return run_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-band SimCLR pre-training + fine-tuning for EEG classification"
    )
    parser.add_argument("--action", type=str, choices=["pretrain", "finetune", "both"],
                        default="both")
    parser.add_argument("--dataset", type=str, default="MDD",
                        choices=["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"])

    # Pre-training
    parser.add_argument("--lr", type=float, default=5e-4, help="Pre-training LR")
    parser.add_argument("--epochs_pretrain", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--use_all_data", action="store_true")

    # Fine-tuning
    parser.add_argument("--epochs_finetune", type=int, default=30)
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--classifier_lr", type=float, default=5e-4)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--patience", type=int, default=0)

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)

    # MoE
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--moe_expert_mult", type=int, default=4)
    parser.add_argument("--balance_weight", type=float, default=0.01)

    # Augmentation
    parser.add_argument("--aug_std", type=float, default=0.05)
    parser.add_argument("--aug_dropout", type=float, default=0.1)
    parser.add_argument("--aug_shift", type=int, default=10)

    # Checkpoint
    parser.add_argument("--pretrained_encoder", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    _set_seed(args.seed)

    if args.action in ("pretrain", "both"):
        result_dir = run_pretrain(args)
        if args.action == "both":
            args.pretrained_encoder = os.path.join(result_dir, "encoder.pt")

    if args.action in ("finetune", "both"):
        if args.pretrained_encoder is None:
            parser.error("--pretrained_encoder is required for finetune action")
        run_finetune(args)


if __name__ == "__main__":
    main()
