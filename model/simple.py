import torch
import torch.nn as nn


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
