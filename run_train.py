"""Launch script for EEG model training and test prediction.

Usage:
    python run_train.py --dataset MDD --model SimpleMLP --epochs 20
    python run_train.py --dataset SEED --model EEGNet --lr 1e-3 --epochs 50
    python run_train.py --dataset SLEEP --model EEGLSTM --lr 5e-4 --epochs 30

Each run creates a timestamped subdirectory under Results/ containing:
    - predictions.txt   : predicted labels for the test set
    - model.pt          : trained model state_dict
    - config.json       : run configuration (dataset, model, hyperparams)
    - metrics.json      : final val accuracy and loss history
"""

import argparse
import json
import os
from datetime import datetime

import torch

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train EEG classification models")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="MDD",
        choices=["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
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
        choices=[1, 2, 3, 4, 5],
        help="CV fold (1-5). Requires running prepare_folds.py first.",
    )

    args = parser.parse_args()

    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)  # 200 Hz sampling

    # --- Create isolated run directory -----------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.dataset}_{args.model}_{timestamp}"
    if args.fold is not None:
        tag += f"_fold{args.fold}"
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

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size, fold=args.fold
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

    print(f"Model: {args.model} | LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("-" * 40)

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

    # Save model weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    # Save training metrics
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

    print(f"\nAll outputs saved to: {run_dir}/")


if __name__ == "__main__":
    main()
