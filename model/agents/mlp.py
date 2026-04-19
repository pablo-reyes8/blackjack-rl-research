from __future__ import annotations

from typing import Sequence

from torch import nn

from .common import build_activation


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: str,
    use_layer_norm: bool,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(build_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)
