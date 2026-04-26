import json
import os
from torch.utils.data import DataLoader
from data.TEST_DATASET import TrainDataset, TestDataset


def load_dataset_info(data_name):
    """Read dataset_info.json and return (channels, num_classes, window_sec)."""
    info_path = os.path.join("data", data_name, "dataset_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    channels = len(info["dataset"]["channels"])
    num_classes = info["dataset"]["num_labels"]
    window_sec = info["processing"]["window_sec"]

    return channels, num_classes, window_sec


def create_dataloaders(data_name, batch_size):
    """Create train/val/test DataLoaders for a dataset."""
    train_path = os.path.join("data", data_name, "train.h5")
    val_path = os.path.join("data", data_name, "val.h5")
    test_path = os.path.join("data", data_name, "test_x_only.h5")

    train_ds = TrainDataset(train_path)
    val_ds = TrainDataset(val_path)
    test_ds = TestDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
