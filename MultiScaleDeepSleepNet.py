import torch
import torch.nn as nn

# ==========================================
# 优化方案 3：定义 1D 残差块 (Residual Block)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 快捷连接 (Shortcut)：如果输入输出通道不同，使用 1x1 卷积对齐维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差相加
        out = self.relu(out)
        return out


class MultiScaleDeepSleepNet(nn.Module):
    def __init__(self, chans=6, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.5, grad_clip=1.0):
        super(MultiScaleDeepSleepNet, self).__init__()
        self.grad_clip = grad_clip

        # ==========================================
        # 优化方案 4：数学对齐 - 小尺度分支 (最终 T=31)
        # ==========================================
        # L_in = 6000
        # (6000 + 44 - 50)/6 + 1 = 1000
        self.small_branch = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=50, stride=6, padding=22),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),                 # 1000 / 8 = 125
            nn.Dropout(dropout),
            ResidualBlock(32, 64, kernel_size=7, padding=3),       # 保持 125
            nn.MaxPool1d(kernel_size=4, stride=4)                  # floor(125 / 4) = 31
        )

        # ==========================================
        # 优化方案 4：数学对齐 - 大尺度分支 (最终 T=31)
        # ==========================================
        # L_in = 6000
        # (6000 + 550 - 400)/50 + 1 = 124
        self.large_branch = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=400, stride=50, padding=275),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),                 # 124 / 4 = 31
            nn.Dropout(dropout),
            ResidualBlock(32, 64, kernel_size=7, padding=3)        # 保持 31
            # 舍弃了最后的 MaxPool(2,2)，从而实现两条分支在物理时序长度上的完美对齐
        )

        # LSTM 序列建模
        self.lstm = nn.LSTM(
            input_size=128, # 拼接后通道数为 64 + 64 = 128
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # ==========================================
        # 优化方案 5：平滑梯度的宽过渡层分类器
        # ==========================================
        # 拼接 avg_pool 和 max_pool 后是 hidden_dim * 4 (512维)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),  # 更温和的降维 (512 -> 256)
            nn.BatchNorm1d(256),             # 稳定线性层输出方差
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 提取特征
        s = self.small_branch(x)  # Shape: (B, 64, 31)
        l = self.large_branch(x)  # Shape: (B, 64, 31)

        # 完美对齐，直接拼接！彻底移除了破坏波形的 F.interpolate
        feat = torch.cat([s, l], dim=1)  # Shape: (B, 128, 31)
        feat = feat.permute(0, 2, 1)     # Shape: (B, 31, 128)

        # LSTM 处理
        out, _ = self.lstm(feat)         # Shape: (B, 31, 256)

        # 全局特征融合 (均值 + 最大值)
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)

        final_feat = torch.cat([avg_pool, max_pool], dim=1) # Shape: (B, 512)

        return self.classifier(final_feat)

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)