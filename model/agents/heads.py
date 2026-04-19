from __future__ import annotations

import torch
from torch import nn

from enviroment_bj.core import ACTION_ORDER

from .common import build_activation


class QHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), build_activation(activation)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, len(ACTION_ORDER)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        value_hidden_dim: int,
        advantage_hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        value_layers: list[nn.Module] = [nn.Linear(input_dim, value_hidden_dim), build_activation(activation)]
        advantage_layers: list[nn.Module] = [
            nn.Linear(input_dim, advantage_hidden_dim),
            build_activation(activation),
        ]
        if dropout > 0:
            value_layers.append(nn.Dropout(dropout))
            advantage_layers.append(nn.Dropout(dropout))
        value_layers.append(nn.Linear(value_hidden_dim, 1))
        advantage_layers.append(nn.Linear(advantage_hidden_dim, len(ACTION_ORDER)))
        self.value_stream = nn.Sequential(*value_layers)
        self.advantage_stream = nn.Sequential(*advantage_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values, value, advantage
