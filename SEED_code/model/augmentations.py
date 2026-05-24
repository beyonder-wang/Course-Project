import torch


_SEED_HEMISPHERE_PAIRS = [
    (0, 2), (3, 4), (5, 13), (6, 12), (7, 11), (8, 10), (14, 22), (15, 21),
    (16, 20), (17, 19), (23, 31), (24, 30), (25, 29), (26, 28), (32, 40),
    (33, 39), (34, 38), (35, 37), (41, 49), (42, 48), (43, 47), (44, 46),
    (50, 56), (51, 55), (52, 54), (58, 60), (57, 61),
]


class GaussianNoise:
    """Add Gaussian noise to EEG signal.

    Args:
        std: noise standard deviation relative to signal std (default 0.05)
    """

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class ChannelDropout:
    """Randomly zero out entire channels.

    Args:
        p: probability of dropping a channel (default 0.1)
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        if self.p <= 0:
            return x
        # x: (C, T) or (B, C, T)
        mask = torch.bernoulli(torch.full((x.shape[-2], 1), 1 - self.p, device=x.device))
        return x * mask


class TimeShift:
    """Circular shift along the time axis.

    Args:
        max_shift: maximum shift in time steps (default 10, ~50ms at 200Hz)
    """

    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, x):
        if self.max_shift <= 0:
            return x
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class Compose:
    """Chain multiple augmentations."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class AsymmetryJitter:
    """Perturb DE asymmetry by shifting hemispheric pairs in opposite directions."""

    def __init__(self, std=0.02, p=1.0):
        self.std = std
        self.p = p
        self.left_idx = torch.tensor([pair[0] for pair in _SEED_HEMISPHERE_PAIRS], dtype=torch.long)
        self.right_idx = torch.tensor([pair[1] for pair in _SEED_HEMISPHERE_PAIRS], dtype=torch.long)

    def __call__(self, x):
        if self.std <= 0 or self.p <= 0:
            return x
        if x.dim() not in (2, 3) or x.shape[-2] != 62 or x.shape[-1] != 5:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        out = x.clone()
        left_idx = self.left_idx.to(x.device)
        right_idx = self.right_idx.to(x.device)

        if x.dim() == 2:
            delta = torch.randn((left_idx.numel(), x.shape[-1]), device=x.device, dtype=x.dtype) * self.std
            out[left_idx] = out[left_idx] + delta
            out[right_idx] = out[right_idx] - delta
            return out

        delta = torch.randn(
            (x.shape[0], left_idx.numel(), x.shape[-1]),
            device=x.device,
            dtype=x.dtype,
        ) * self.std
        out[:, left_idx] = out[:, left_idx] + delta
        out[:, right_idx] = out[:, right_idx] - delta
        return out


class SimCLRTransform:
    """Apply the same augmentation pipeline twice to produce two views.
    Used by Pretrainer to generate positive pairs for SimCLR.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)
