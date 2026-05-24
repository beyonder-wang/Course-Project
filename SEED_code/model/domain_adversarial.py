"""Domain-adversarial utilities for cross-subject and cross-session EEG training."""

import torch
import torch.nn as nn


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def gradient_reverse(x, coeff=1.0):
    return _GradientReverseFn.apply(x, coeff)


class DomainAdversarialHead(nn.Module):
    """Predict domains from features while reversing encoder gradients."""

    def __init__(
        self,
        feature_dim,
        num_domains,
        hidden_dim=128,
        dropout=0.1,
        grl_lambda=1.0,
    ):
        super().__init__()
        if num_domains < 2:
            raise ValueError("DomainAdversarialHead needs at least 2 domains.")

        self.grl_lambda = float(grl_lambda)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features):
        reversed_features = gradient_reverse(features, self.grl_lambda)
        return self.classifier(reversed_features)
