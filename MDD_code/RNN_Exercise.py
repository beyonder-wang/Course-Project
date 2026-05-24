import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, lfilter

# ============================================================
# 0. 信号处理工具函数 (微分熵 DE 相关)
# ============================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def compute_DE(data, fs=200):
    """
    计算微分熵特征
    输入 data shape: (batch, channels, time)
    输出 shape: (batch, channels * 5)
    """
    batch_size, chans, _ = data.shape
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    de_features = []
    # 为了速度，我们将张量转为 numpy 处理
    data_np = data.cpu().numpy() if torch.is_tensor(data) else data
    
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
# 1. 通道注意力模块 (Squeeze-and-Excitation Block)
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
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
# 2. 核心模型：EEGNet (带注意力机制)
# ============================================================
class EEGNet(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, F1=16, D=2, F2=32, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.block1_temporal = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        self.block1_spatial = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (chans, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.attention = SEBlock(self.F1 * self.D)

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding='same', groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, chans, time_points)
            x = self.block1_temporal(dummy_input)
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
# 3. 终极模型：时空频混合模型 EEGNet_Hybrid
# ============================================================
class EEGNet_Hybrid(nn.Module):
    def __init__(self, chans=20, num_classes=2, time_points=200, F1=16, D=2, F2=32):
        super(EEGNet_Hybrid, self).__init__()
        
        # 实例化内部 EEGNet 获取特征提取器
        self.eegnet_part = EEGNet(chans, num_classes, time_points, F1, D, F2)
        
        # 频域分支 (DE 特征)
        de_input_dim = chans * 5
        self.de_mlp = nn.Sequential(
            nn.Linear(de_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        
        # 融合分类器
        combined_dim = self.eegnet_part.flatten_size + 64
        self.final_classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_raw):
        # 1. 自动计算 DE 特征
        x_de = compute_DE(x_raw).to(x_raw.device)
        
        # 2. 提取波形特征
        x1 = self.eegnet_part.block1_temporal(x_raw.unsqueeze(1))
        x1 = self.eegnet_part.block1_spatial(x1)
        x1 = self.eegnet_part.attention(x1)
        x1 = self.eegnet_part.block2(x1)
        feat_raw = x1.view(x1.size(0), -1)
        
        # 3. 提取频域特征
        feat_de = self.de_mlp(x_de)
        
        # 4. 融合
        combined = torch.cat((feat_raw, feat_de), dim=1)
        return self.final_classifier(combined)

# ============================================================
# 4. 基础模型：SimpleRNN & LSTM
# ============================================================
class ExerciseEEGSimpleRNN(nn.Module):
    def __init__(self, chans=20, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3, bidirectional=True, grad_clip=1.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip
        self.rnn = nn.RNN(input_size=chans, hidden_size=hidden_dim, num_layers=num_layers, nonlinearity="relu",
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self._init_rnn_weights()
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def _init_rnn_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name: nn.init.orthogonal_(param)
            elif "weight_ih" in name: nn.init.xavier_uniform_(param)
            elif "bias" in name: nn.init.zeros_(param)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, h_n = self.rnn(x)
        feat = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.classifier(feat)

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

class ExerciseEEGLSTM(nn.Module):
    def __init__(self, chans=20, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3, bidirectional=True, grad_clip=1.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip
        self.lstm = nn.LSTM(input_size=chans, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, (h_n, c_n) = self.lstm(x)
        feat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1, :, :]
        return self.classifier(feat)

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)