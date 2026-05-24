import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import butter, lfilter


# ============================================================
# Signal processing utilities for DE features
# ============================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def compute_DE(data, fs=200):
    """Compute differential entropy features. Input: (batch, channels, time) -> Output: (batch, channels*5)"""
    batch_size, chans, _ = data.shape
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    data_np = data.cpu().numpy() if torch.is_tensor(data) else data
    de_features = []
    for b in range(batch_size):
        channel_features = []
        for c in range(chans):
            sample = data_np[b, c, :]
            for _, (low, high) in bands.items():
                filtered = butter_bandpass_filter(sample, low, high, fs)
                variance = np.var(filtered)
                de = np.log(variance + 1e-6)
                channel_features.append(de)
        de_features.append(channel_features)
    return torch.tensor(de_features, dtype=torch.float32)


# ============================================================
# SE Attention Block
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================
# EEGNet (current version with SE attention)
# ============================================================
class EEGNet(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, F1=16, D=2, F2=32, dropout_rate=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.block1_temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(F1)
        )

        self.block1_spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.attention = SEBlock(F1 * D)

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, time_points)
            x = self.block1_temporal(dummy)
            x = self.block1_spatial(x)
            x = self.attention(x)
            x = self.block2(x)
            self.flatten_size = x.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1_temporal(x)
        x = self.block1_spatial(x)
        x = self.attention(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# EEGNet Old (best_model.pth uses this: F1=8, D=2, no SE, block1/block2 naming)
# ============================================================
class EEGNetOld(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, F1=8, D=2, F2=16, dropout_rate=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, time_points)
            x = self.block1(dummy)
            x = self.block2(x)
            self.flatten_size = x.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# EEGNet Hybrid (EEGNet + DE frequency features)
# Matches hybrid_best_model.pth key structure:
#   eegnet.block1_temporal, eegnet.block1_spatial, eegnet.attention, eegnet.block2, eegnet.classifier
#   de_branch (Linear+BN+ReLU+Dropout)
#   classifier (Linear+ReLU+Linear)
# ============================================================
class EEGNetHybrid(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, F1=16, D=2, F2=32, dropout_rate=0.5):
        super().__init__()
        self.eegnet = EEGNet(chans, num_classes, time_points, F1, D, F2, dropout_rate)

        de_input_dim = chans * 5  # 20 channels * 5 bands = 100
        self.de_branch = nn.Sequential(
            nn.Linear(de_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        combined_dim = self.eegnet.flatten_size + 64  # 192 + 64 = 256
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_raw):
        # Compute DE features
        x_de = compute_DE(x_raw).to(x_raw.device)

        # EEGNet feature extraction (without final classifier)
        x1 = x_raw.unsqueeze(1)
        x1 = self.eegnet.block1_temporal(x1)
        x1 = self.eegnet.block1_spatial(x1)
        x1 = self.eegnet.attention(x1)
        x1 = self.eegnet.block2(x1)
        feat_raw = x1.view(x1.size(0), -1)

        # DE branch
        feat_de = self.de_branch(x_de)

        # Fusion
        combined = torch.cat((feat_raw, feat_de), dim=1)
        return self.classifier(combined)


class TemporalCNN(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, dropout_rate=0.4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(chans, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.res1 = self._make_res_block(64, 64, 5, dropout_rate)
        self.res2 = self._make_res_block(64, 128, 5, dropout_rate)
        self.res3 = self._make_res_block(128, 128, 3, dropout_rate)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def _make_res_block(self, in_ch, out_ch, kernel_size, dropout):
        return ResBlock1D(in_ch, out_ch, kernel_size, dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class CNN_LSTM(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, dropout_rate=0.4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(chans, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True,
                           dropout=dropout_rate, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # (B, 128, T/4)
        x = x.permute(0, 2, 1)  # (B, T/4, 128)
        out, (h_n, _) = self.lstm(x)
        feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(feat)


class MultiScaleCNN(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, dropout_rate=0.4):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        b15 = self.branch15(x)
        x = torch.cat([b3, b5, b7, b15], dim=1)
        x = self.conv2(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def get_model(name, **kwargs):
    models = {
        'eegnet': EEGNet,
        'eegnet_old': EEGNetOld,
        'eegnet_hybrid': EEGNetHybrid,
        'temporalcnn': TemporalCNN,
        'cnn_lstm': CNN_LSTM,
        'multiscale': MultiScaleCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)


def get_all_model_classes():
    """Return list of (name, constructor) for checkpoint probing."""
    return [
        ('EEGNet', EEGNet),
        ('EEGNetOld', EEGNetOld),
        ('EEGNetHybrid', EEGNetHybrid),
        ('TemporalCNN', TemporalCNN),
        ('CNN_LSTM', CNN_LSTM),
        ('MultiScaleCNN', MultiScaleCNN),
    ]
