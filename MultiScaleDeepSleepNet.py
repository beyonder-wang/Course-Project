import torch
import torch.nn as nn


class MultiScaleDeepSleepNet(nn.Module):
    def __init__(self, chans=6, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.5, grad_clip=1.0):
        super(MultiScaleDeepSleepNet, self).__init__()
        self.grad_clip = grad_clip

        # 分支 1: 小尺度 (捕捉细微波动)
        # 6000 -> 1000 -> 125 -> 125 -> 125 -> 31
        self.small_branch = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=32, stride=6, padding=24),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        # 分支 2: 大尺度 (捕捉宏观频率)
        # 6000 -> 120 -> 30 -> 30 -> 30 -> 15 (会进行上采样/池化对齐)
        self.large_branch = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=512, stride=50, padding=175),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=6, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=6, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 拼接后的通道数 64+64=128
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # 因为拼接了 avg_pool 和 max_pool (hidden_dim*2 + hidden_dim*2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),  # 改为 * 4
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        s = self.small_branch(x)
        l = self.large_branch(x)

        # 动态对齐时间步
        target_len = s.size(2)
        l = nn.functional.interpolate(l, size=target_len, mode='linear', align_corners=False)

        feat = torch.cat([s, l], dim=1)  # (B, 128, T)
        feat = feat.permute(0, 2, 1)  # (B, T, 128)

        # LSTM 处理
        # out shape: (Batch, Time, Hidden_Dim * 2)
        out, _ = self.lstm(feat)

        # --- 【优化关键点】：全局特征融合 ---
        # 不再只取 h_n，而是对整个时间轴取平均值和最大值
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        # 拼接平均和最大特征 (效果通常优于只取 Last State)
        final_feat = torch.cat([avg_pool, max_pool], dim=1)

        # 注意：如果这里拼接了两个，classifier 的输入维度要翻倍
        # 请确保修改 __init__ 中的 self.classifier 第一个 Linear 层为 (hidden_dim * 4)
        return self.classifier(final_feat)

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)