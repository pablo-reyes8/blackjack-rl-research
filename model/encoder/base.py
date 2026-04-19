from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch
from torch import nn

from .config import EncoderConfig


class BaseFeatureEncoder(nn.Module, ABC):
    output_dim: int

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        return self.encode(observation, table_rules)


class BaseBlackjackEncoder(nn.Module, ABC):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, response: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
