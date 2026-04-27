import torch


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


class SimCLRTransform:
    """Apply the same augmentation pipeline twice to produce two views.
    Used by Pretrainer to generate positive pairs for SimCLR.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)
