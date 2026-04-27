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
from datetime import datetime

import numpy as np
import torch

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders
from trainer import Trainer


def _train_fold(args, fold, run_dir, channels, num_classes, time_points):
    """Train on one fold (or original split if fold is None) and save results."""
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size, fold=fold
    )

    model_cls = MODEL_DICT[args.model]
    if args.model in ("SimpleLinear", "SimpleMLP"):
        model = model_cls(
            input_channels=channels, time_points=time_points, num_classes=num_classes
        )
    elif args.model == "EEGNet":
        model = model_cls(
            chans=channels, num_classes=num_classes, time_point=time_points
        )
    else:
        model = model_cls(chans=channels, num_classes=num_classes)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        epochs=args.epochs,
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

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train EEG classification models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MDD",
        choices=["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EEGNet",
        choices=list(MODEL_DICT.keys()),
        help="Model architecture",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="CV fold (1-5), -1 for all 5 folds (requires prepare_folds.py). "
             "Omit to use original train/val split.",
    )

    args = parser.parse_args()

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
    run_dir = os.path.join("Results", tag)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["channels"] = channels
    config["num_classes"] = num_classes
    config["time_points"] = time_points
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"=== Dataset: {args.dataset} ===")
    print(f"Channels: {channels}, Classes: {num_classes}, Time points: {time_points}")
    print(f"Model: {args.model} | LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batch_size}")
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


if __name__ == "__main__":
    main()
