from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def apply_action_mask(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    fill_value = torch.finfo(q_values.dtype).min
    return q_values.masked_fill(~action_mask, fill_value)


def infer_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_encoded_batch_to_device(encoded: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, Mapping):
            moved[key] = {
                nested_key: nested_value.to(device) if isinstance(nested_value, torch.Tensor) else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            moved[key] = value
    return moved
