"""FFT-based frequency band decomposition for EEG signals.

Splits raw EEG into standard clinical frequency bands:
  delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz),
  beta (13–30 Hz), gamma (30–75 Hz)

Uses torch.fft for GPU-compatible online filtering.
"""

import torch
import torch.nn as nn


class BandDecomposition(nn.Module):
    """Decompose EEG signal into frequency bands via FFT filtering.

    Args:
        bands: dict of {band_name: (low_freq, high_freq)} in Hz.
            Default: delta, theta, alpha, beta, gamma bands.
        fs: sampling frequency in Hz (default 200).
    """

    DEFAULT_BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 75.0),
    }

    def __init__(self, bands=None, fs=200):
        super().__init__()
        self.bands = bands if bands is not None else self.DEFAULT_BANDS
        self.band_names = list(self.bands.keys())
        self.fs = fs

    def _bandpass_mask(self, T, low, high, device):
        """Create a bandpass mask in frequency domain.

        Args:
            T: time points
            low, high: frequency range in Hz
            device: torch device

        Returns:
            (freq_bins,) boolean mask
        """
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs, device=device)
        return (freqs >= low) & (freqs <= high)

    def _filter(self, x, low, high):
        """Apply bandpass filter in frequency domain.

        Args:
            x: (B, C, T) time-domain signal
            low, high: frequency cutoff in Hz

        Returns:
            (B, C, T) filtered signal
        """
        X = torch.fft.rfft(x, dim=-1)
        mask = self._bandpass_mask(x.shape[-1], low, high, x.device)
        # Expand mask to match X: (B, C, freq_bins)
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(X)
        X_filtered = X * mask
        return torch.fft.irfft(X_filtered, n=x.shape[-1], dim=-1)

    def forward(self, x):
        """Decompose EEG into frequency bands.

        Args:
            x: (B, C, T) raw EEG signal

        Returns:
            dict of {band_name: (B, C, T)} filtered signals
        """
        return {
            name: self._filter(x, low, high)
            for name, (low, high) in self.bands.items()
        }

    def stack_bands(self, x):
        """Decompose and stack into a single tensor.

        Args:
            x: (B, C, T)

        Returns:
            (B * num_bands, C, T) stacked band signals
        """
        bands = self.forward(x)
        return torch.cat([bands[name] for name in self.band_names], dim=0)
