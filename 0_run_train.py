"""Launch script for EEG model training and test prediction.

Usage:
    python 0_run_train.py --dataset MDD --model SimpleMLP --epochs 20
    python 0_run_train.py --dataset SEED --model EEGNet --lr 1e-3 --epochs 50
    python 0_run_train.py --dataset SLEEP --model EEGLSTM --lr 5e-4 --epochs 30
    python 0_run_train.py --dataset MDD --model EEGNet --fold 1
    python 0_run_train.py --dataset MDD --model EEGNet --fold -1

Each run creates a timestamped subdirectory under Results/ containing:
    - predictions.txt   : predicted labels for the test set
    - model.pt          : trained model state_dict
    - config.json       : run configuration (dataset, model, hyperparams)
    - metrics.json      : final val accuracy and loss history

With --fold -1, creates a parent directory with sub-folders for each fold
and a cv_summary.json aggregating the 5-fold results.
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from model import (
    MODEL_DICT, GaussianNoise, ChannelDropout, TimeShift, Compose,
    EmotionDLHead, DomainAdversarialHead,
)
from utils import load_dataset_info, create_dataloaders, resolve_device, start_log, stop_log, write_summary_txt
from trainer import Trainer


def _write_supervised_summary(run_dir, config, metrics, fold_label):
    sections = [
        ("Configuration", [
            f"Dataset: {config['dataset']}",
            f"Model: {config['model']}",
            f"Fold: {fold_label}",
            f"Channels: {config['channels']}",
            f"Classes: {config['num_classes']}",
            f"Time points: {config['time_points']}",
            f"Device: {config['device']}",
            f"AMP: {config.get('amp', False)}",
            f"Grad accum steps: {config.get('grad_accum_steps', 1)}",
            f"Learning rate: {config['lr']}",
            f"Weight decay: {config['weight_decay']}",
            f"Batch size: {config['batch_size']}",
            f"Epochs: {config['epochs']}",
            f"Patience: {config['patience']}",
            f"Scheduler: {config['scheduler']}",
            f"Label smoothing: {config.get('label_smoothing', 0.0)}",
            f"Mixup alpha: {config.get('mixup_alpha', 0.0)}",
            f"EmotionDL alpha: {config.get('emotion_dl_alpha', 0.0)}",
            f"Subject adv weight: {config.get('subject_adv_weight', 0.0)}",
            f"Subject adv key: {config.get('subject_adv_key', 'subject_id')}",
        ]),
        ("Results", [
            f"Best val accuracy: {metrics['best_val_accuracy']:.4f} (epoch {metrics['best_epoch']})",
            f"Final val accuracy: {metrics['final_val_accuracy']:.4f}",
            f"Final train loss: {metrics['train_losses'][-1]:.4f}",
            f"Final val loss: {metrics['val_losses'][-1]:.4f}",
        ]),
        ("Output Files", [
            f"Model: {os.path.join(run_dir, 'model.pt')}",
            f"Predictions: {os.path.join(run_dir, 'predictions.txt')}",
            f"Metrics: {os.path.join(run_dir, 'metrics.json')}",
            f"Config: {os.path.join(run_dir, 'config.json')}",
            f"Log: {os.path.join(run_dir, 'run.log')}",
        ]),
    ]
    write_summary_txt(run_dir, sections)


def _make_run_config(args, channels, num_classes, time_points):
    config = dict(vars(args))
    config["channels"] = channels
    config["num_classes"] = num_classes
    config["time_points"] = time_points
    return config


def _needs_feature_outputs(args):
    return args.emotion_dl_alpha > 0 or args.subject_adv_weight > 0


def _resolve_metadata_cardinality(dataset, key):
    cardinalities = getattr(dataset, "metadata_cardinalities", None) or {}
    if key in cardinalities:
        return int(cardinalities[key])

    metadata = getattr(dataset, "metadata", None) or {}
    if key in metadata:
        return int(torch.unique(metadata[key]).numel())
    return None


def _train_fold(args, fold, run_dir, channels, num_classes, time_points):
    """Train on one fold (or original split if fold is None) and save results."""
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset,
        args.batch_size,
        fold=fold,
        standardize=args.standardize_inputs,
        sr_aug_times=args.sr_aug_times,
        sr_segments=args.sr_segments,
    )
    needs_features = _needs_feature_outputs(args)

    model_cls = MODEL_DICT[args.model]
    time_point_models = {"EEGNet", "EEGNet_SE", "EEGNet_SimAM", "EEGNet_SimAM_SE",
                         "EEGNet_KAN", "ShallowConvNet", "ATCNet", "FBCNet",
                         "EEGTCNet", "MICNN", "EEGConformer"}
    if args.model in ("SimpleLinear", "SimpleMLP", "DENet"):
        model = model_cls(
            input_channels=channels, time_points=time_points, num_classes=num_classes
        )
    elif args.model in time_point_models:
        model_kwargs = dict(chans=channels, num_classes=num_classes, time_point=time_points)
        if args.model == "ATCNet":
            from model.atcnet import ATCNET_PRESETS
            if args.atc_preset is not None:
                model_kwargs.update(ATCNET_PRESETS[args.atc_preset])
            else:
                model_kwargs["n_windows"] = args.atc_n_windows
                model_kwargs["F1"] = args.atc_f1
                model_kwargs["d_model"] = args.atc_d_model
                model_kwargs["dropout_conv"] = args.atc_dropout_conv
                model_kwargs["dropout_attn"] = args.atc_dropout_attn
                model_kwargs["dropout_tcn"] = args.atc_dropout_tcn
                model_kwargs["tcn_depth"] = args.atc_tcn_depth
        elif args.model == "EEGConformer":
            model_kwargs["dim"] = args.conf_dim
            model_kwargs["n_blocks"] = args.conf_blocks
            model_kwargs["n_head"] = args.conf_heads
            model_kwargs["kernel_size"] = args.conf_kernel
            model_kwargs["ff_expansion"] = args.conf_ff_expansion
            model_kwargs["patch_kernel"] = args.conf_patch_kernel
            model_kwargs["patch_stride"] = args.conf_patch_stride
            model_kwargs["dropout"] = args.conf_dropout
        model = model_cls(**model_kwargs)
    elif args.model == "SEEDGraphormer":
        model = model_cls(
            chans=channels,
            num_classes=num_classes,
            d_model=args.graphormer_dim,
            depth=args.graphormer_depth,
            num_heads=args.graphormer_heads,
            mlp_ratio=args.graphormer_mlp_ratio,
            dropout=args.graphormer_dropout,
            attn_dropout=args.graphormer_attn_dropout,
            top_k=args.graphormer_top_k,
            dyn_alpha=args.graphormer_dyn_alpha,
            return_features=needs_features,
        )
    elif args.model == "SEEDAsymNet":
        model = model_cls(
            chans=channels,
            num_classes=num_classes,
            hidden_dim=args.seedasym_hidden_dim,
            graph_layers=args.seedasym_graph_layers,
            asym_hidden=args.seedasym_asym_hidden,
            fusion_hidden=args.seedasym_fusion_hidden,
            dropout=args.seedasym_dropout,
            top_k=args.seedasym_top_k,
            dyn_alpha=args.seedasym_dyn_alpha,
            return_features=needs_features,
        )
    elif args.model == "SEEDBandGraphNet":
        model = model_cls(
            chans=channels,
            num_classes=num_classes,
            hidden_dim=args.bandgraph_hidden_dim,
            graph_layers=args.bandgraph_graph_layers,
            band_hidden=args.bandgraph_band_hidden,
            asym_hidden=args.bandgraph_asym_hidden,
            fusion_hidden=args.bandgraph_fusion_hidden,
            dropout=args.bandgraph_dropout,
            top_k=args.bandgraph_top_k,
            dyn_alpha=args.bandgraph_dyn_alpha,
            return_features=needs_features,
        )
    else:
        model = model_cls(chans=channels, num_classes=num_classes)

    if args.model == "RGNN":
        model.top_k = args.rgnn_top_k
        model.dyn_alpha = args.rgnn_dyn_alpha
        model.return_features = needs_features
    elif args.model in ("DGCNN", "DGCNN_RG"):
        if hasattr(model, "return_features"):
            model.return_features = needs_features

    optimizer_cls = torch.optim.AdamW if args.weight_decay > 0 else torch.optim.Adam
    emotion_head = None
    domain_head = None
    params = list(model.parameters())
    if args.emotion_dl_alpha > 0:
        feature_dim = getattr(model, "feature_dim", None)
        if feature_dim is None:
            raise ValueError(
                f"Model {args.model} does not expose feature_dim for EmotionDL."
            )
        emotion_head = EmotionDLHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=args.emotion_hidden_dim,
            dropout=args.emotion_dropout,
        )
        params.extend(list(emotion_head.parameters()))
    if args.subject_adv_weight > 0:
        feature_dim = getattr(model, "feature_dim", None)
        if feature_dim is None:
            raise ValueError(
                f"Model {args.model} does not expose feature_dim for subject-adversarial training."
            )
        num_domains = _resolve_metadata_cardinality(
            train_loader.dataset, args.subject_adv_key
        )
        if num_domains is None:
            raise ValueError(
                f"Dataset {args.dataset} does not provide metadata key "
                f"{args.subject_adv_key!r} in the training split. "
                "Rebuild the dataset/folds with metadata preserved, or disable "
                "subject-adversarial training."
            )
        if num_domains < 2:
            raise ValueError(
                f"Metadata key {args.subject_adv_key!r} has only {num_domains} domain(s); "
                "subject-adversarial training needs at least 2."
            )
        domain_head = DomainAdversarialHead(
            feature_dim=feature_dim,
            num_domains=num_domains,
            hidden_dim=args.subject_adv_hidden_dim,
            dropout=args.subject_adv_dropout,
            grl_lambda=args.subject_adv_grl_lambda,
        )
        params.extend(list(domain_head.parameters()))

    optimizer = optimizer_cls(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor,
            patience=args.plateau_patience, min_lr=args.plateau_min_lr
        )

    batch_transform = None
    transforms = []
    if args.aug_noise_std > 0:
        transforms.append(GaussianNoise(args.aug_noise_std))
    if args.aug_channel_dropout > 0:
        transforms.append(ChannelDropout(args.aug_channel_dropout))
    if args.aug_time_shift > 0:
        transforms.append(TimeShift(args.aug_time_shift))
    if transforms:
        batch_transform = Compose(*transforms)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smoothing=args.label_smoothing,
        batch_transform=batch_transform,
        mixup_alpha=args.mixup_alpha,
        patience=args.patience,
        device=args.device,
        use_amp=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        emotion_head=emotion_head,
        emotion_dl_alpha=args.emotion_dl_alpha,
        emotion_aux_weight=args.emotion_aux_weight,
        domain_head=domain_head,
        domain_adv_weight=args.subject_adv_weight,
        domain_target_key=args.subject_adv_key,
        grad_clip_norm=args.grad_clip_norm,
        warmup_epochs=args.warmup_epochs,
    )

    history = trainer.train()
    trainer.save_predictions(run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    metrics = {
        "final_val_accuracy": history["val_accuracies"][-1],
        "best_val_accuracy": max(history["val_accuracies"]),
        "best_epoch": history["val_accuracies"].index(max(history["val_accuracies"])) + 1,
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "val_accuracies": history["val_accuracies"],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _write_supervised_summary(
        run_dir,
        config=_make_run_config(args, channels, num_classes, time_points),
        metrics=metrics,
        fold_label=(fold if fold is not None else "original_split"),
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train EEG classification models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BCIC2A",
        choices=["MDD", "BCIC2A", "CHINESE", "SEED", "SEED_DE", "SEED_BYSUBJ", "SEED_SUB1_DE", "SEED_SUB1_DE_RANDOM", "SEED_SUB1_DE_TRIAL", "SEED_SUB1_DE_S23v1", "SEED_SUB1_DE_STRAT", "SLEEP"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EEGNet_KAN",
        choices=list(MODEL_DICT.keys()),
        help="Model architecture",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 disables)")
    parser.add_argument(
        "--output_root",
        type=str,
        default="Results",
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="CV fold (1-5), -1 for all 5 folds (requires prepare_folds.py). "
             "Omit to use original train/val split.",
    )
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, cuda:0, etc. (auto-fallback to CPU)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training on CUDA")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--standardize_inputs", action="store_true",
                        help="Standardize inputs with train-split channel/time statistics")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor for cross-entropy")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "cosine", "plateau"], help="Learning-rate scheduler")
    parser.add_argument("--plateau_factor", type=float, default=0.9,
                        help="ReduceLROnPlateau factor")
    parser.add_argument("--plateau_patience", type=int, default=20,
                        help="ReduceLROnPlateau patience in epochs")
    parser.add_argument("--plateau_min_lr", type=float, default=1e-4,
                        help="ReduceLROnPlateau minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (uses AdamW when > 0)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Beta alpha for mixup (0 disables)")
    parser.add_argument("--aug_noise_std", type=float, default=0.0,
                        help="Gaussian noise std for supervised augmentation")
    parser.add_argument("--aug_channel_dropout", type=float, default=0.0,
                        help="Channel dropout probability for supervised augmentation")
    parser.add_argument("--aug_time_shift", type=int, default=0,
                        help="Max circular time shift in samples for augmentation")
    parser.add_argument("--emotion_dl_alpha", type=float, default=0.0,
                        help="Blend factor for EmotionDL soft targets (0 disables)")
    parser.add_argument("--emotion_aux_weight", type=float, default=1.0,
                        help="Weight for the auxiliary EmotionDL classifier loss")
    parser.add_argument("--emotion_hidden_dim", type=int, default=128,
                        help="Hidden dimension for the EmotionDL auxiliary head")
    parser.add_argument("--emotion_dropout", type=float, default=0.1,
                        help="Dropout for the EmotionDL auxiliary head")
    parser.add_argument("--subject_adv_weight", type=float, default=0.0,
                        help="Weight for domain-adversarial loss on subject/session labels")
    parser.add_argument("--subject_adv_key", type=str, default="subject_id",
                        help="Metadata key used as domain target, e.g. subject_id or session_id")
    parser.add_argument("--subject_adv_hidden_dim", type=int, default=128,
                        help="Hidden dimension for the adversarial domain head")
    parser.add_argument("--subject_adv_dropout", type=float, default=0.1,
                        help="Dropout for the adversarial domain head")
    parser.add_argument("--subject_adv_grl_lambda", type=float, default=1.0,
                        help="Gradient-reversal strength for domain-adversarial training")
    parser.add_argument("--rgnn_top_k", type=int, default=8,
                        help="Top-k sparse neighbors retained by RGNN")
    parser.add_argument("--rgnn_dyn_alpha", type=float, default=0.15,
                        help="Weight of dynamic DE-similarity adjacency in RGNN")
    parser.add_argument("--graphormer_dim", type=int, default=128,
                        help="Token dimension for SEEDGraphormer")
    parser.add_argument("--graphormer_depth", type=int, default=6,
                        help="Transformer depth for SEEDGraphormer")
    parser.add_argument("--graphormer_heads", type=int, default=8,
                        help="Attention heads for SEEDGraphormer")
    parser.add_argument("--graphormer_mlp_ratio", type=int, default=4,
                        help="MLP ratio inside SEEDGraphormer transformer blocks")
    parser.add_argument("--graphormer_dropout", type=float, default=0.3,
                        help="Dropout for SEEDGraphormer")
    parser.add_argument("--graphormer_attn_dropout", type=float, default=0.1,
                        help="Attention dropout for SEEDGraphormer")
    parser.add_argument("--graphormer_top_k", type=int, default=12,
                        help="Top-k sparse neighbors retained by SEEDGraphormer")
    parser.add_argument("--graphormer_dyn_alpha", type=float, default=0.2,
                        help="Weight of dynamic DE-similarity adjacency in SEEDGraphormer")
    parser.add_argument("--seedasym_hidden_dim", type=int, default=64,
                        help="Graph hidden dimension for SEEDAsymNet")
    parser.add_argument("--seedasym_graph_layers", type=int, default=2,
                        help="Graph layer count for SEEDAsymNet")
    parser.add_argument("--seedasym_asym_hidden", type=int, default=256,
                        help="Asymmetry branch hidden size for SEEDAsymNet")
    parser.add_argument("--seedasym_fusion_hidden", type=int, default=256,
                        help="Fusion MLP hidden size for SEEDAsymNet")
    parser.add_argument("--seedasym_dropout", type=float, default=0.3,
                        help="Dropout for SEEDAsymNet")
    parser.add_argument("--seedasym_top_k", type=int, default=8,
                        help="Top-k sparse neighbors retained by SEEDAsymNet")
    parser.add_argument("--seedasym_dyn_alpha", type=float, default=0.15,
                        help="Weight of dynamic DE-similarity adjacency in SEEDAsymNet")
    parser.add_argument("--bandgraph_hidden_dim", type=int, default=48,
                        help="Per-band graph hidden dimension for SEEDBandGraphNet")
    parser.add_argument("--bandgraph_graph_layers", type=int, default=2,
                        help="Per-band graph layer count for SEEDBandGraphNet")
    parser.add_argument("--bandgraph_band_hidden", type=int, default=96,
                        help="Per-band fused embedding size for SEEDBandGraphNet")
    parser.add_argument("--bandgraph_asym_hidden", type=int, default=96,
                        help="Per-band asymmetry hidden size for SEEDBandGraphNet")
    parser.add_argument("--bandgraph_fusion_hidden", type=int, default=192,
                        help="Classifier hidden size for SEEDBandGraphNet")
    parser.add_argument("--bandgraph_dropout", type=float, default=0.3,
                        help="Dropout for SEEDBandGraphNet")
    parser.add_argument("--bandgraph_top_k", type=int, default=10,
                        help="Top-k sparse neighbors retained by SEEDBandGraphNet")
    parser.add_argument("--bandgraph_dyn_alpha", type=float, default=0.2,
                        help="Weight of dynamic DE-similarity adjacency in SEEDBandGraphNet")
    parser.add_argument("--atc_preset", type=str, default=None,
                        choices=["base", "large", "xl"],
                        help="ATCNet capacity preset (overrides individual --atc_* args)")
    parser.add_argument("--atc_n_windows", type=int, default=5,
                        help="Sliding-window count for ATCNet")
    parser.add_argument("--sr_aug_times", type=int, default=0,
                        help="Times to expand the train split with segment-reconstruction augmentation")
    parser.add_argument("--sr_segments", type=int, default=8,
                        help="Number of temporal segments for segment-reconstruction augmentation")
    parser.add_argument("--atc_f1", type=int, default=16,
                        help="ATCNet frontend temporal filter count")
    parser.add_argument("--atc_d_model", type=int, default=32,
                        help="ATCNet attention/TCN hidden size")
    parser.add_argument("--atc_dropout_conv", type=float, default=0.3,
                        help="ATCNet conv-block dropout")
    parser.add_argument("--atc_dropout_attn", type=float, default=0.5,
                        help="ATCNet attention dropout")
    parser.add_argument("--atc_dropout_tcn", type=float, default=0.3,
                        help="ATCNet TCN dropout")
    parser.add_argument("--atc_tcn_depth", type=int, default=2,
                        help="ATCNet TCN depth")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Number of linear LR warmup epochs (0 = no warmup)")
    parser.add_argument("--conf_dim", type=int, default=64,
                        help="EEGConformer model dimension")
    parser.add_argument("--conf_blocks", type=int, default=4,
                        help="EEGConformer number of Conformer blocks")
    parser.add_argument("--conf_heads", type=int, default=4,
                        help="EEGConformer attention heads per block")
    parser.add_argument("--conf_kernel", type=int, default=31,
                        help="EEGConformer depthwise conv kernel size")
    parser.add_argument("--conf_ff_expansion", type=int, default=4,
                        help="EEGConformer FFN expansion factor")
    parser.add_argument("--conf_patch_kernel", type=int, default=25,
                        help="EEGConformer patch embedding kernel size")
    parser.add_argument("--conf_patch_stride", type=int, default=10,
                        help="EEGConformer patch embedding stride")
    parser.add_argument("--conf_dropout", type=float, default=0.1,
                        help="EEGConformer dropout rate")

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Validate fold ---
    if args.fold is not None and args.fold not in (-1, 1, 2, 3, 4, 5):
        parser.error("--fold must be 1-5 (single fold) or -1 (all 5 folds)")

    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)

    # --- Create run directory ------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.fold == -1:
        tag = f"{args.dataset}_{args.model}_{timestamp}_allfolds"
    elif args.fold is not None:
        tag = f"{args.dataset}_{args.model}_{timestamp}_fold{args.fold}"
    else:
        tag = f"{args.dataset}_{args.model}_{timestamp}"
    run_dir = os.path.join(args.output_root, tag)
    os.makedirs(run_dir, exist_ok=True)
    tee = start_log(run_dir)

    # Save config
    config = _make_run_config(args, channels, num_classes, time_points)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"=== Dataset: {args.dataset} ===")
    print(f"Channels: {channels}, Classes: {num_classes}, Time points: {time_points}")
    print(f"Model: {args.model} | LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp} | Grad accum: {args.grad_accum_steps}")
    print("-" * 40)

    if args.fold == -1:
        # --- Auto-run all 5 folds --------------------------------------------
        all_metrics = []
        for k in range(1, 6):
            print(f"\n{'=' * 40}\n  Fold {k}/5\n{'=' * 40}")
            fold_dir = os.path.join(run_dir, f"fold_{k}")
            os.makedirs(fold_dir, exist_ok=True)
            m = _train_fold(args, k, fold_dir, channels, num_classes, time_points)
            all_metrics.append(m)

        # Aggregate
        final_accs = [m["final_val_accuracy"] for m in all_metrics]
        best_accs = [m["best_val_accuracy"] for m in all_metrics]
        cv_summary = {
            "per_fold": {
                f"fold_{k}": {
                    "final_val_accuracy": all_metrics[k - 1]["final_val_accuracy"],
                    "best_val_accuracy": all_metrics[k - 1]["best_val_accuracy"],
                    "best_epoch": all_metrics[k - 1]["best_epoch"],
                }
                for k in range(1, 6)
            },
            "mean_final_val_accuracy": float(np.mean(final_accs)),
            "std_final_val_accuracy": float(np.std(final_accs)),
            "mean_best_val_accuracy": float(np.mean(best_accs)),
            "std_best_val_accuracy": float(np.std(best_accs)),
        }
        with open(os.path.join(run_dir, "cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=2)
        write_summary_txt(run_dir, [
            ("Configuration", [
                f"Dataset: {config['dataset']}",
                f"Model: {config['model']}",
                "Fold: all_folds",
                f"Channels: {config['channels']}",
                f"Classes: {config['num_classes']}",
                f"Time points: {config['time_points']}",
                f"Device: {config['device']}",
                f"AMP: {config.get('amp', False)}",
                f"Grad accum steps: {config.get('grad_accum_steps', 1)}",
                f"Learning rate: {config['lr']}",
                f"Batch size: {config['batch_size']}",
                f"Epochs: {config['epochs']}",
                f"Label smoothing: {config.get('label_smoothing', 0.0)}",
                f"Mixup alpha: {config.get('mixup_alpha', 0.0)}",
                f"EmotionDL alpha: {config.get('emotion_dl_alpha', 0.0)}",
                f"Subject adv weight: {config.get('subject_adv_weight', 0.0)}",
                f"Subject adv key: {config.get('subject_adv_key', 'subject_id')}",
            ]),
            ("Cross-Validation Results", [
                f"Mean final val accuracy: {cv_summary['mean_final_val_accuracy']:.4f}",
                f"Std final val accuracy: {cv_summary['std_final_val_accuracy']:.4f}",
                f"Mean best val accuracy: {cv_summary['mean_best_val_accuracy']:.4f}",
                f"Std best val accuracy: {cv_summary['std_best_val_accuracy']:.4f}",
            ]),
            ("Output Files", [
                f"CV summary: {os.path.join(run_dir, 'cv_summary.json')}",
                f"Config: {os.path.join(run_dir, 'config.json')}",
                f"Log: {os.path.join(run_dir, 'run.log')}",
            ]),
        ])

        print(f"\n{'=' * 40}")
        print("CV Results (5 folds):")
        for k in range(1, 6):
            print(f"  Fold {k}:  final={final_accs[k - 1]:.4f}  best={best_accs[k - 1]:.4f}")
        print(f"  Mean ± std final:  {cv_summary['mean_final_val_accuracy']:.4f} ± {cv_summary['std_final_val_accuracy']:.4f}")
        print(f"  Mean ± std best:   {cv_summary['mean_best_val_accuracy']:.4f} ± {cv_summary['std_best_val_accuracy']:.4f}")

    else:
        # --- Single fold (or original split if fold is None) -----------------
        _train_fold(args, args.fold, run_dir, channels, num_classes, time_points)

    print(f"\nAll outputs saved to: {run_dir}/")
    stop_log(tee)


if __name__ == "__main__":
    main()
