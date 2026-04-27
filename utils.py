import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from data.TEST_DATASET import TrainDataset, FoldDataset, TestDataset, UnlabeledDataset, MultiUnlabeledDataset


# --- Device utilities -----------------------------------------------------------

def resolve_device(device_str="cpu"):
    """Resolve device string, falling back to CPU if CUDA unavailable.

    Args:
        device_str: e.g. "cpu", "cuda", "cuda:0", "cuda:1", "auto"

    Returns:
        str: canonical device string (JSON-safe)
    """
    device_str = device_str.strip().lower()
    if device_str in ("", "auto"):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[WARNING] CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
            if idx >= torch.cuda.device_count():
                print(f"[WARNING] GPU {idx} not available ({torch.cuda.device_count()} GPUs found), using GPU 0")
                return "cuda:0"
        return device_str

    return "cpu"


# --- Logging & summary utilities ------------------------------------------------

class TeeLogger:
    """Duplicate stdout to a log file while preserving console output."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def start_log(run_dir):
    """Redirect stdout to both console and run_dir/run.log. Returns TeeLogger."""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    return tee


def stop_log(tee):
    """Restore original stdout and close log file."""
    tee.close()


def write_summary_txt(run_dir, sections):
    """Write a human-readable summary.txt from a list of (heading, lines) tuples.

    Args:
        run_dir: directory to write summary.txt into
        sections: list of (heading, lines) where lines is a list of str
    """
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        for heading, lines in sections:
            f.write(f"\n{heading}\n")
            f.write("-" * 40 + "\n")
            for line in lines:
                f.write(f"  {line}\n")
    return path


def _find_info_file(data_name):
    """Find dataset_info JSON file, handling variant filenames (e.g. BCIC2A)."""
    base = os.path.join("data", data_name)
    for name in ("dataset_info.json", "dataset_info_fixed.json"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No dataset_info.json found in {base}")


def load_dataset_info(data_name):
    """Read dataset_info.json and return (channels, num_classes, window_sec)."""
    info_path = _find_info_file(data_name)
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


# --- Pre-training data loaders -------------------------------------------------

def _pretrain_paths(dataset_name, use_all_data):
    """Build list of HDF5 paths for a single dataset's pre-training."""
    base = os.path.join("data", dataset_name)
    paths = [os.path.join(base, "train.h5")]
    if use_all_data:
        for fname in ("val.h5", "test_x_only.h5"):
            p = os.path.join(base, fname)
            if os.path.exists(p):
                paths.append(p)
    return paths


def create_pretrain_loaders(dataset_name, batch_size, use_all_data=True):
    """Phase 1: single-dataset unlabeled DataLoader for SimCLR pre-training.

    Args:
        dataset_name: e.g. "MDD"
        batch_size: batch size
        use_all_data: if True, include val.h5 and test_x_only.h5
    """
    paths = _pretrain_paths(dataset_name, use_all_data)
    ds = UnlabeledDataset(paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def create_multi_pretrain_loaders(dataset_names, batch_size, use_train_only=True):
    """Phase 2: multi-dataset unlabeled DataLoader for SimCLR pre-training.

    Args:
        dataset_names: list of dataset names, defaults to all 5
        batch_size: batch size
        use_train_only: if True, only use train.h5 from each dataset

    Returns:
        data_loader: DataLoader yielding (x, source_idx) batches
        dataset_channels: dict of {name: channel_count}
    """
    configs = []
    dataset_channels = {}
    for name in dataset_names:
        paths = [os.path.join("data", name, "train.h5")]
        if not use_train_only:
            for fname in ("val.h5", "test_x_only.h5"):
                p = os.path.join("data", name, fname)
                if os.path.exists(p):
                    paths.append(p)
        configs.append((name, paths))
        channels, _, _ = load_dataset_info(name)
        dataset_channels[name] = channels

    ds = MultiUnlabeledDataset(configs)
    sampler = PerSourceBatchSampler(ds, batch_size, shuffle=True)
    loader = DataLoader(ds, batch_sampler=sampler)
    return loader, dataset_channels


class PerSourceBatchSampler(Sampler):
    """Yields batches of global indices, each batch coming from a single source block.

    This avoids shape mismatches when different datasets have different channel counts.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build (start, end) ranges for each source block
        self.ranges = []
        start = 0
        for cum in dataset.cum_sizes:
            end = cum.item()
            self.ranges.append((start, end))
            start = end

    def __len__(self):
        return sum(
            (end - start + self.batch_size - 1) // self.batch_size
            for start, end in self.ranges
        )

    def __iter__(self):
        range_order = list(range(len(self.ranges)))
        if self.shuffle:
            np.random.shuffle(range_order)

        for ri in range_order:
            start, end = self.ranges[ri]
            indices = list(range(start, end))
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
