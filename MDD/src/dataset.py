import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class EEGH5Dataset(Dataset):
    def __init__(self, h5_path, mode='train', norm_stats=None, augment=False):
        with h5py.File(h5_path, 'r') as f:
            self.X = f['X'][:].astype(np.float32)
            if 'y' in f:
                self.y = f['y'][:].astype(np.int64)
            else:
                self.y = None

        self.mode = mode
        self.augment = augment and (mode == 'train')

        if mode == 'train' and norm_stats is None:
            self.mean = self.X.mean(axis=(0, 2), keepdims=True)  # (1, channels, 1)
            self.std = self.X.std(axis=(0, 2), keepdims=True) + 1e-8
        elif norm_stats is not None:
            self.mean = norm_stats['mean']
            self.std = norm_stats['std']
        else:
            self.mean = np.zeros((1, self.X.shape[1], 1), dtype=np.float32)
            self.std = np.ones((1, self.X.shape[1], 1), dtype=np.float32)

        self.X = (self.X - self.mean) / self.std

    def get_norm_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def save_norm_stats(self, path='outputs/norm_stats.npz'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load_norm_stats(path='outputs/norm_stats.npz'):
        data = np.load(path)
        return {'mean': data['mean'], 'std': data['std']}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        if self.augment:
            x = self._augment(x)

        x = torch.from_numpy(x)
        if self.y is not None:
            return x, self.y[idx]
        return x

    def _augment(self, x):
        # Gaussian noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.1, x.shape).astype(np.float32)

        # Time shift (circular)
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift, axis=-1)

        # Channel dropout
        if np.random.random() < 0.2:
            ch = np.random.randint(0, x.shape[0])
            x[ch] = 0.0

        # Amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        return x
