from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from enviroment_bj.core import ACTION_ORDER

from .base import BaseBlackjackEncoder
from .batching import pad_encoded_sequences, stack_encoded_steps
from .config import EncoderConfig
from .hand import BetEncoder, BettingContextEncoder, HandContextEncoder, HandFeatureEncoder, InsuranceContextEncoder, OtherHandsEncoder
from .history import DiscardSummaryEncoder, ExactShoeEncoder, ObservedCardsHistoryEncoder, TemporalFeatureEncoder
from .rules import RuleEncoder


class BlackjackObservationEncoder(BaseBlackjackEncoder):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(config or EncoderConfig.for_profile("table_realistic_default"))
        self.modules_by_name = nn.ModuleDict()
        self.module_dims: OrderedDict[str, int] = OrderedDict()
        self.module_slices: dict[str, tuple[int, int]] = {}
        self._build_modules()

    @classmethod
    def from_profile(cls, profile: str, **config_overrides: Any) -> BlackjackObservationEncoder:
        config = EncoderConfig.for_profile(profile)
        for key, value in config_overrides.items():
            setattr(config, key, value)
        config.__post_init__()
        return cls(config=config)

    @property
    def state_dim(self) -> int:
        return sum(self.module_dims.values()) + (len(ACTION_ORDER) if self.config.encode_action_mask_features else 0)

    def _build_modules(self) -> None:
        self._register_module("hand", HandFeatureEncoder(self.config))
        if self.config.encode_other_hands:
            self._register_module("other_hands", OtherHandsEncoder(self.config))
        self._register_module("hand_context", HandContextEncoder(self.config))
        self._register_module("insurance", InsuranceContextEncoder())
        if self.config.encode_betting_context:
            self._register_module("betting_context", BettingContextEncoder())
        if self.config.profile != "minimal_basic_strategy":
            self._register_module("bet", BetEncoder())
        if self.config.encode_rules:
            self._register_module("rules", RuleEncoder())
        if self.config.encode_observed_history:
            self._register_module("observed_history", ObservedCardsHistoryEncoder(self.config))
        if self.config.encode_discard_summary:
            self._register_module("discard_summary", DiscardSummaryEncoder(self.config))
        if self.config.encode_temporal:
            self._register_module("temporal", TemporalFeatureEncoder(self.config))
        if self.config.encode_exact_shoe:
            self._register_module("exact_shoe", ExactShoeEncoder())

        start = 0
        for name, dim in self.module_dims.items():
            self.module_slices[name] = (start, start + dim)
            start += dim
        if self.config.encode_action_mask_features:
            self.module_slices["mask_features"] = (start, start + len(ACTION_ORDER))

    def _register_module(self, name: str, module: nn.Module) -> None:
        self.modules_by_name[name] = module
        self.module_dims[name] = getattr(module, "output_dim")

    def _resolve_action_mask(self, response: Mapping[str, Any]) -> torch.Tensor:
        action_mask_value = response.get("action_mask", [0] * len(ACTION_ORDER))
        if isinstance(action_mask_value, torch.Tensor):
            return action_mask_value.detach().to(dtype=torch.bool)
        return torch.tensor(action_mask_value, dtype=torch.bool)

    def encode_state_only(self, response: Mapping[str, Any]) -> dict[str, Any]:
        observation = response.get("observation") or {}
        table_rules = response.get("table_rules") or {}
        module_tensors = [module(observation, table_rules).to(torch.float32) for module in self.modules_by_name.values()]
        state_vector = torch.cat(module_tensors, dim=0) if module_tensors else torch.zeros(0, dtype=torch.float32)
        action_mask = self._resolve_action_mask(response)

        if self.config.encode_action_mask_features:
            state_vector = torch.cat([state_vector, action_mask.to(torch.float32)], dim=0)

        return {
            "state_vector": state_vector,
            "action_mask": action_mask,
        }

    def forward(self, response: Mapping[str, Any]) -> dict[str, Any]:
        observation = response.get("observation") or {}
        table_rules = response.get("table_rules") or {}
        module_tensors: OrderedDict[str, torch.Tensor] = OrderedDict()

        for name, module in self.modules_by_name.items():
            tensor = module(observation, table_rules).to(torch.float32)
            module_tensors[name] = tensor

        state_vector = torch.cat(list(module_tensors.values()), dim=0) if module_tensors else torch.zeros(0, dtype=torch.float32)
        action_mask = self._resolve_action_mask(response)

        if self.config.encode_action_mask_features:
            state_vector = torch.cat([state_vector, action_mask.to(torch.float32)], dim=0)

        return {
            "state_vector": state_vector,
            "action_mask": action_mask,
            "module_tensors": dict(module_tensors),
            "metadata": {
                "profile": self.config.profile,
                "state_dim": int(state_vector.shape[0]),
                "module_dims": dict(self.module_dims),
                "module_slices": dict(self.module_slices),
                "observation_profile": observation.get("profile"),
                "observation_mode": observation.get("mode"),
                "decision_phase": observation.get("decision_phase"),
                "available_bet_multipliers": observation.get("available_bet_multipliers"),
            },
        }

    def encode_batch(self, responses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        encoded_steps = [self.forward(response) for response in responses]
        batch = stack_encoded_steps(encoded_steps)
        batch["metadata"]["state_dim"] = self.state_dim
        batch["metadata"]["module_dims"] = dict(self.module_dims)
        return batch

    def encode_sequence_batch(self, response_sequences: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        encoded_sequences = [[self.forward(response) for response in sequence] for sequence in response_sequences]
        batch = pad_encoded_sequences(encoded_sequences)
        batch["metadata"]["state_dim"] = self.state_dim
        batch["metadata"]["module_dims"] = dict(self.module_dims)
        return batch
