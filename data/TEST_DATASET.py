import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --- Train: reads X and y ---
class TrainDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)

        assert len(self.x) == len(self.y), "X and y length mismatch"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# --- CV Fold: reads all.h5 + index subset ---
class FoldDataset(Dataset):
    """Dataset that loads from merged all.h5 and indexes a specific fold subset."""

    def __init__(self, all_h5_path, index_npy_path):
        with h5py.File(all_h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)
        self.indices = np.load(index_npy_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.x[real_idx], self.y[real_idx]


# --- Test: reads X only ---
class TestDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# --- Unlabeled: loads X from multiple HDF5 files, discarding labels ---
class UnlabeledDataset(Dataset):
    """Loads X from one or more HDF5 files, concatenates, returns tensors without labels.

    Used for SimCLR pre-training (Phase 1: single dataset).
    """

    def __init__(self, h5_paths):
        xs = []
        for path in h5_paths:
            with h5py.File(path, "r") as f:
                xs.append(torch.tensor(f["X"][()], dtype=torch.float32))
        self.x = torch.cat(xs, dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# --- Multi-dataset unlabeled: returns sample + source index for channel routing ---
class MultiUnlabeledDataset(Dataset):
    """Loads X from multiple datasets, returning (x, source_idx) tuples.

    Stores tensors separately (different datasets have different channel counts).
    source_idx is used by ChannelAdapter to select the correct 1x1 Conv1d.
    """

    def __init__(self, dataset_configs):
        """
        Args:
            dataset_configs: list of (name, [h5_paths]) tuples, e.g.
                [("MDD", ["data/MDD/train.h5"]), ("SEED", ["data/SEED/train.h5"])]
        """
        self.xs = []
        self.source_idx = []
        sizes = []
        for src_idx, (name, paths) in enumerate(dataset_configs):
            for path in paths:
                with h5py.File(path, "r") as f:
                    data = torch.tensor(f["X"][()], dtype=torch.float32)
                    self.xs.append(data)
                    self.source_idx.append(src_idx)
                    sizes.append(len(data))

        self.cum_sizes = torch.tensor(sizes).cumsum(dim=0)
        self.total = self.cum_sizes[-1].item()
        self.dataset_names = [name for name, _ in dataset_configs]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Binary search to find which sub-tensor contains this global index
        block = int(torch.searchsorted(self.cum_sizes, idx, right=True).item())
        offset = self.cum_sizes[block - 1].item() if block > 0 else 0
        local_idx = idx - offset
        return self.xs[block][local_idx], self.source_idx[block]
