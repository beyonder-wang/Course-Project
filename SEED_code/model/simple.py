import torch
import torch.nn as nn
from .band_decomposition import BandDecomposition


class SimpleLinear(nn.Module):
    def __init__(self, input_channels, time_points, num_classes):
        super(SimpleLinear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_channels * time_points, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        time_points=200,
        hidden_dims=(256, 128),
        dropout=0.3,
    ):
        super().__init__()

        input_dim = input_channels * time_points

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.mlp(x)


class DENet(nn.Module):
    """SEED-friendly band-DE baseline computed directly from raw EEG.

    The model follows the classic SEED recipe:
    1. Split the raw waveform into standard EEG bands with FFT masking
    2. Compute per-channel differential-entropy style log-variance features
    3. Classify the resulting (channels x 5 bands) feature vector with an MLP
    """

    def __init__(
        self,
        input_channels,
        num_classes,
        time_points=200,
        hidden_dims=(256, 128),
        dropout=0.3,
        fs=200,
    ):
        super().__init__()
        del time_points  # Raw length is consumed by the internal DE extraction.

        input_dim = input_channels * 5
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))

        self.decomp = BandDecomposition(fs=fs)
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        band_signals = self.decomp(x)
        de_features = []
        for name in self.decomp.band_names:
            band_x = band_signals[name]
            # Differential entropy for Gaussian signals differs from log-variance
            # by a constant, so log-variance is sufficient for classification.
            band_var = band_x.var(dim=-1, unbiased=False)
            de_features.append(0.5 * torch.log(band_var + 1e-6))

        feats = torch.cat(de_features, dim=1)
        return self.classifier(feats)
