"""MLP critic, config-compatible with DiffRL's `models.critic.CriticMLP`."""

import torch
import torch.nn as nn

from .actor import _activation, _mlp


class CriticMLP(nn.Module):
    def __init__(self, obs_dim: int, units: list[int], activation: str = "elu"):
        super().__init__()
        self.net = _mlp(obs_dim, 1, units, activation)
        # Zero-init last layer so V(s) ≈ 0 at start. This keeps the SHAC
        # bootstrap term ≈ 0 until the critic has been trained on a few
        # batches of real targets — otherwise the random initial critic acts
        # like a strong wrong gradient signal on the actor loss.
        last_linear = self.net[-1]
        with torch.no_grad():
            last_linear.weight.zero_()
            if last_linear.bias is not None:
                last_linear.bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
