import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        chans,
        num_classes,
        time_point=200,
        f1=8,
        d=2,
        pk1=4,
        pk2=8,
        dp=0.5,
        max_norm1=1,
        norm=nn.Identity(),
    ):
        super(EEGNet, self).__init__()
        f2 = f1 * d
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)
            ),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp),
        )

        self._apply_max_norm(self.block2[0], max_norm1)
        self.embed_dim = f2 * ((time_point // pk1) // pk2)
        self.norm = norm
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(
                    param.data, p=2, dim=0, maxnorm=max_norm
                )

    def forward(self, x):
        x = self.norm(x)
        x = self.block1(x.unsqueeze(dim=1))
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
