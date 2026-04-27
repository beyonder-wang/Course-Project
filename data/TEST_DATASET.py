import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- Train: 读 x 和 y ---
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


# --- CV Fold: 读 all.h5 + 索引子集 ---
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


# --- Test: 只读 x ---
class TestDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]