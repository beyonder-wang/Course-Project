import json
import os
from torch.utils.data import DataLoader
from data.TEST_DATASET import TrainDataset, FoldDataset, TestDataset


def load_dataset_info(data_name):
    """Read dataset_info.json and return (channels, num_classes, window_sec)."""
    info_path = os.path.join("data", data_name, "dataset_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    channels = len(info["dataset"]["channels"])
    num_classes = info["dataset"]["num_labels"]
    window_sec = info["processing"]["window_sec"]

    return channels, num_classes, window_sec


def create_dataloaders(data_name, batch_size, fold=None):
    """Create train/val/test DataLoaders for a dataset.

    If *fold* is given (int, 1-based), uses the pre-computed CV split
    from all.h5 + fold_{fold}/*.npy. Requires running prepare_folds.py first.
    Otherwise uses the original train.h5 / val.h5 files.
    """
    if fold is not None:
        all_path = os.path.join("data", data_name, "all.h5")
        train_idx = os.path.join("data", data_name, f"fold_{fold}", "train_idx.npy")
        val_idx = os.path.join("data", data_name, f"fold_{fold}", "val_idx.npy")

        if not os.path.exists(all_path):
            raise FileNotFoundError(
                f"{all_path} not found. Run `python prepare_folds.py --dataset {data_name}` first."
            )
        train_ds = FoldDataset(all_path, train_idx)
        val_ds = FoldDataset(all_path, val_idx)
    else:
        train_path = os.path.join("data", data_name, "train.h5")
        val_path = os.path.join("data", data_name, "val.h5")
        train_ds = TrainDataset(train_path)
        val_ds = TrainDataset(val_path)

    test_path = os.path.join("data", data_name, "test_x_only.h5")
    test_ds = TestDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
