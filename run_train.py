"""Launch script for EEG model training and test prediction.

Usage:
    python run_train.py --dataset MDD --model SimpleMLP --epochs 20
    python run_train.py --dataset SEED --model EEGNet --lr 1e-3 --epochs 50
    python run_train.py --dataset SLEEP --model EEGLSTM --lr 5e-4 --epochs 30
"""

import argparse

from model import MODEL_DICT
from utils import load_dataset_info, create_dataloaders
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train EEG classification models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MDD", "BCIC2A", "CHINESE", "SEED", "SLEEP"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_DICT.keys()),
        help="Model architecture",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    channels, num_classes, window_sec = load_dataset_info(args.dataset)
    time_points = int(window_sec * 200)  # 200 Hz sampling

    print(f"=== Dataset: {args.dataset} ===")
    print(f"Channels: {channels}, Classes: {num_classes}, Time points: {time_points}")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size
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

    trainer.train()
    trainer.save_predictions(args.dataset, output_dir="Results")


if __name__ == "__main__":
    main()
