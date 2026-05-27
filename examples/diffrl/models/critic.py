"""MLP critic, config-compatible with DiffRL's `models.critic.CriticMLP`."""

import math

import torch
import torch.nn as nn

from .actor import _mlp


class CriticMLP(nn.Module):
    def __init__(self, obs_dim: int, units: list[int], activation: str = "elu"):
        super().__init__()
        self.net = _mlp(obs_dim, 1, units, activation)
        # DiffRL inits every Linear with orthogonal weights (gain sqrt(2)) and
        # zero bias. (No zero-init on the last layer — DiffRL doesn't do it; the
        # target-critic Polyak copy + grad clipping handle early instability.)
        gain = math.sqrt(2.0)
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
