"""Stochastic and deterministic MLP actors, ported to be config-compatible with
NVIDIA DiffRL's `models.actor.{ActorStochasticMLP,ActorDeterministicMLP}`."""

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"unknown activation: {name}")


def _mlp(in_dim: int, out_dim: int, units: list[int], activation: str) -> nn.Sequential:
    # Matches DiffRL: each hidden block is `Linear -> activation -> LayerNorm`,
    # final layer is a bare Linear (Identity output). The LayerNorm after every
    # hidden activation is load-bearing — without it the critic value-regression
    # is ill-conditioned across the wide return range and actor grads blow up.
    layers: list[nn.Module] = []
    prev = in_dim
    for u in units:
        layers.append(nn.Linear(prev, u))
        layers.append(_activation(activation))
        layers.append(nn.LayerNorm(u))
        prev = u
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ActorDeterministicMLP(nn.Module):
    """Deterministic MLP actor: action = tanh(MLP(obs))."""

    def __init__(self, obs_dim: int, act_dim: int, units: list[int], activation: str = "elu"):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, units, activation)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class ActorStochasticMLP(nn.Module):
    """Gaussian MLP actor. Mean comes from the MLP, log-std is a learned
    state-independent parameter. Returns tanh(mean + eps * std) when sampling,
    tanh(mean) when deterministic.

    DiffRL parameterizes log-std as a free parameter initialized to -1.0.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        units: list[int],
        activation: str = "elu",
        logstd_init: float = -1.0,
    ):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, units, activation)
        self.log_std = nn.Parameter(torch.full((act_dim,), logstd_init))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean = self.net(obs)
        if deterministic:
            return torch.tanh(mean)
        std = torch.exp(self.log_std)
        eps = torch.randn_like(mean)
        return torch.tanh(mean + eps * std)
