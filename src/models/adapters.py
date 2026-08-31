"""Bottleneck adapter, in the style of CLIP-Adapter (Gao et al., 2021).

Rather than fine-tuning the whole encoder (Method A), Method B freezes CLIP
and inserts a tiny two-layer MLP after the projection head. The residual
mixing (`residual_ratio`) keeps most of the original CLIP feature, only
letting the adapter nudge it — which is what makes this cheap to train on a
handful of target examples without catastrophic forgetting of source
knowledge.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64, residual_ratio: float = 0.2):
        super().__init__()
        self.residual_ratio = residual_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, dim, bias=False),
        )
        # Near-identity init: adapter starts as a no-op and learns a
        # correction, rather than fighting the pretrained CLIP features
        # from a random starting point.
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        return self.residual_ratio * delta + (1 - self.residual_ratio) * x
