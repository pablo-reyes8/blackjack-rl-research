from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from enviroment_bj.core import ACTION_ORDER
from model.encoder import BlackjackObservationEncoder

from .common import apply_action_mask, infer_module_device, move_encoded_batch_to_device
from .config import AgentNetworkConfig
from .heads import DuelingQHead, QHead
from .mlp import build_mlp
from .recurrent import RecurrentBackbone


class BaseBlackjackQNetwork(nn.Module):
    def __init__(
        self,
        config: AgentNetworkConfig,
        encoder: BlackjackObservationEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder or BlackjackObservationEncoder(config.encoder)
        self.num_actions = len(ACTION_ORDER)

    @classmethod
    def from_profiles(
        cls,
        *,
        architecture: str,
        encoder_profile: str,
        **config_overrides: Any,
    ) -> BaseBlackjackQNetwork:
        config = AgentNetworkConfig.for_architecture(
            architecture,
            encoder_profile=encoder_profile,
            **config_overrides,
        )
        return cls(config=config)

    @property
    def state_dim(self) -> int:
        return self.encoder.state_dim

    def _device(self) -> torch.device:
        return infer_module_device(self)

    def _prepare_feedforward_batch(self, inputs: Any) -> dict[str, Any]:
        if isinstance(inputs, Mapping) and isinstance(inputs.get("state_vector"), torch.Tensor):
            encoded = dict(inputs)
        elif isinstance(inputs, Mapping):
            encoded = self.encoder(inputs)
        elif isinstance(inputs, Sequence):
            if not inputs:
                raise ValueError("inputs cannot be empty")
            if isinstance(inputs[0], Mapping) and isinstance(inputs[0].get("state_vector"), torch.Tensor):
                first_module_tensors = inputs[0].get("module_tensors", {})
                encoded = {
                    "state_vector": torch.stack([item["state_vector"] for item in inputs], dim=0),
                    "action_mask": torch.stack([item["action_mask"] for item in inputs], dim=0),
                    "module_tensors": {
                        key: torch.stack([item["module_tensors"][key] for item in inputs], dim=0)
                        for key in first_module_tensors
                    },
                    "metadata": {"batch_size": len(inputs)},
                }
            else:
                encoded = self.encoder.encode_batch(inputs)
        else:
            raise TypeError("Unsupported input type for feedforward network")

        encoded = move_encoded_batch_to_device(encoded, self._device())
        state_vector = encoded["state_vector"]
        action_mask = encoded["action_mask"]

        if state_vector.ndim == 1:
            state_vector = state_vector.unsqueeze(0)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)

        encoded["state_vector"] = state_vector
        encoded["action_mask"] = action_mask.to(torch.bool)
        return encoded

    def _prepare_sequence_batch(self, inputs: Any) -> dict[str, Any]:
        if isinstance(inputs, Mapping) and isinstance(inputs.get("state_vector"), torch.Tensor):
            encoded = dict(inputs)
        elif isinstance(inputs, Mapping):
            encoded = self.encoder(inputs)
        elif isinstance(inputs, Sequence):
            if not inputs:
                raise ValueError("inputs cannot be empty")
            if isinstance(inputs[0], Mapping):
                if isinstance(inputs[0].get("state_vector"), torch.Tensor):
                    first_module_tensors = inputs[0].get("module_tensors", {})
                    encoded = {
                        "state_vector": torch.stack([item["state_vector"] for item in inputs], dim=0).unsqueeze(0),
                        "action_mask": torch.stack([item["action_mask"] for item in inputs], dim=0).unsqueeze(0),
                        "padding_mask": torch.ones((1, len(inputs)), dtype=torch.bool),
                        "module_tensors": {
                            key: torch.stack([item["module_tensors"][key] for item in inputs], dim=0).unsqueeze(0)
                            for key in first_module_tensors
                        },
                        "metadata": {"batch_size": 1, "sequence_lengths": [len(inputs)]},
                    }
                else:
                    encoded = self.encoder.encode_sequence_batch([inputs])
            else:
                encoded = self.encoder.encode_sequence_batch(inputs)
        else:
            raise TypeError("Unsupported input type for recurrent network")

        encoded = move_encoded_batch_to_device(encoded, self._device())
        state_vector = encoded["state_vector"]
        action_mask = encoded["action_mask"]
        padding_mask = encoded.get("padding_mask")

        if state_vector.ndim == 1:
            state_vector = state_vector.unsqueeze(0).unsqueeze(0)
        elif state_vector.ndim == 2:
            state_vector = state_vector.unsqueeze(1)

        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0).unsqueeze(0)
        elif action_mask.ndim == 2:
            action_mask = action_mask.unsqueeze(1)

        if padding_mask is None:
            padding_mask = torch.ones(state_vector.shape[:2], dtype=torch.bool, device=self._device())
        elif padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)

        encoded["state_vector"] = state_vector
        encoded["action_mask"] = action_mask.to(torch.bool)
        encoded["padding_mask"] = padding_mask.to(torch.bool)
        return encoded


class FeedForwardDoubleDQN(BaseBlackjackQNetwork):
    @classmethod
    def from_profile(cls, encoder_profile: str = "minimal_basic_strategy", **config_overrides: Any) -> FeedForwardDoubleDQN:
        return cls(
            config=AgentNetworkConfig.for_architecture(
                "feedforward",
                encoder_profile=encoder_profile,
                **config_overrides,
            )
        )

    def __init__(
        self,
        config: AgentNetworkConfig | None = None,
        encoder: BlackjackObservationEncoder | None = None,
    ) -> None:
        network_config = config or AgentNetworkConfig.for_architecture(
            "feedforward",
            encoder_profile="minimal_basic_strategy",
        )
        super().__init__(network_config, encoder=encoder)
        self.backbone = build_mlp(
            self.state_dim,
            self.config.feedforward_hidden_dims,
            activation=self.config.activation,
            use_layer_norm=self.config.use_layer_norm,
            dropout=self.config.dropout,
        )
        hidden_dim = self.config.feedforward_hidden_dims[-1]
        self.q_head = nn.Linear(hidden_dim, self.num_actions)

    def forward(self, inputs: Any) -> dict[str, Any]:
        encoded = self._prepare_feedforward_batch(inputs)
        state_vector = encoded["state_vector"]
        hidden = self.backbone(state_vector)
        q_values = self.q_head(hidden)
        masked_q_values = apply_action_mask(q_values, encoded["action_mask"])
        return {
            "q_values": q_values,
            "masked_q_values": masked_q_values,
            "action_mask": encoded["action_mask"],
            "state_vector": state_vector,
            "backbone_output": hidden,
            "module_tensors": encoded.get("module_tensors", {}),
            "metadata": {
                "architecture": self.config.architecture,
                "encoder_profile": self.config.encoder.profile,
                "state_dim": self.state_dim,
                "batch_shape": tuple(q_values.shape),
                **(encoded.get("metadata") or {}),
            },
        }


class RecurrentDoubleDQN(BaseBlackjackQNetwork):
    @classmethod
    def from_profile(cls, encoder_profile: str = "table_realistic_default", **config_overrides: Any) -> RecurrentDoubleDQN:
        return cls(
            config=AgentNetworkConfig.for_architecture(
                "recurrent",
                encoder_profile=encoder_profile,
                **config_overrides,
            )
        )

    def __init__(
        self,
        config: AgentNetworkConfig | None = None,
        encoder: BlackjackObservationEncoder | None = None,
    ) -> None:
        network_config = config or AgentNetworkConfig.for_architecture(
            "recurrent",
            encoder_profile="table_realistic_default",
        )
        super().__init__(network_config, encoder=encoder)
        self.input_projection = build_mlp(
            self.state_dim,
            (self.config.projection_dim,),
            activation=self.config.activation,
            use_layer_norm=self.config.use_layer_norm,
            dropout=self.config.dropout,
        )
        self.recurrent_backbone = RecurrentBackbone(
            input_dim=self.config.projection_dim,
            hidden_dim=self.config.recurrent_hidden_dim,
            num_layers=self.config.recurrent_num_layers,
            recurrent_type=self.config.recurrent_type,
        )
        self.q_head = QHead(
            input_dim=self.config.recurrent_hidden_dim,
            hidden_dim=self.config.head_hidden_dim,
            activation=self.config.activation,
            dropout=self.config.dropout,
        )

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> Any:
        return self.recurrent_backbone.init_hidden(batch_size, device or self._device())

    def forward(self, inputs: Any, hidden_state: Any = None) -> dict[str, Any]:
        encoded = self._prepare_sequence_batch(inputs)
        state_vector = encoded["state_vector"]
        padding_mask = encoded["padding_mask"]
        projected = self.input_projection(state_vector)
        recurrent_output, next_hidden_state = self.recurrent_backbone(
            projected,
            padding_mask=padding_mask,
            hidden_state=hidden_state,
        )
        q_values = self.q_head(recurrent_output)
        masked_q_values = apply_action_mask(q_values, encoded["action_mask"])
        return {
            "q_values": q_values,
            "masked_q_values": masked_q_values,
            "action_mask": encoded["action_mask"],
            "padding_mask": padding_mask,
            "state_vector": state_vector,
            "projected_state": projected,
            "recurrent_output": recurrent_output,
            "hidden_state": next_hidden_state,
            "module_tensors": encoded.get("module_tensors", {}),
            "metadata": {
                "architecture": self.config.architecture,
                "encoder_profile": self.config.encoder.profile,
                "state_dim": self.state_dim,
                "sequence_shape": tuple(q_values.shape),
                **(encoded.get("metadata") or {}),
            },
        }

    def forward_step(self, inputs: Any, hidden_state: Any = None) -> dict[str, Any]:
        output = self.forward(inputs, hidden_state=hidden_state)
        return {
            **output,
            "q_values": output["q_values"].squeeze(1),
            "masked_q_values": output["masked_q_values"].squeeze(1),
            "action_mask": output["action_mask"].squeeze(1),
            "padding_mask": output["padding_mask"].squeeze(1),
            "state_vector": output["state_vector"].squeeze(1),
            "projected_state": output["projected_state"].squeeze(1),
            "recurrent_output": output["recurrent_output"].squeeze(1),
        }


class DuelingRecurrentDoubleDQN(BaseBlackjackQNetwork):
    @classmethod
    def from_profile(
        cls,
        encoder_profile: str = "table_realistic_default",
        **config_overrides: Any,
    ) -> DuelingRecurrentDoubleDQN:
        return cls(
            config=AgentNetworkConfig.for_architecture(
                "dueling_recurrent",
                encoder_profile=encoder_profile,
                **config_overrides,
            )
        )

    def __init__(
        self,
        config: AgentNetworkConfig | None = None,
        encoder: BlackjackObservationEncoder | None = None,
    ) -> None:
        network_config = config or AgentNetworkConfig.for_architecture(
            "dueling_recurrent",
            encoder_profile="table_realistic_default",
        )
        super().__init__(network_config, encoder=encoder)
        self.input_projection = build_mlp(
            self.state_dim,
            (self.config.projection_dim,),
            activation=self.config.activation,
            use_layer_norm=self.config.use_layer_norm,
            dropout=self.config.dropout,
        )
        self.recurrent_backbone = RecurrentBackbone(
            input_dim=self.config.projection_dim,
            hidden_dim=self.config.recurrent_hidden_dim,
            num_layers=self.config.recurrent_num_layers,
            recurrent_type=self.config.recurrent_type,
        )
        self.dueling_head = DuelingQHead(
            input_dim=self.config.recurrent_hidden_dim,
            value_hidden_dim=self.config.value_hidden_dim,
            advantage_hidden_dim=self.config.advantage_hidden_dim,
            activation=self.config.activation,
            dropout=self.config.dropout,
        )

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> Any:
        return self.recurrent_backbone.init_hidden(batch_size, device or self._device())

    def forward(self, inputs: Any, hidden_state: Any = None) -> dict[str, Any]:
        encoded = self._prepare_sequence_batch(inputs)
        state_vector = encoded["state_vector"]
        padding_mask = encoded["padding_mask"]
        projected = self.input_projection(state_vector)
        recurrent_output, next_hidden_state = self.recurrent_backbone(
            projected,
            padding_mask=padding_mask,
            hidden_state=hidden_state,
        )
        q_values, values, advantages = self.dueling_head(recurrent_output)
        masked_q_values = apply_action_mask(q_values, encoded["action_mask"])
        return {
            "q_values": q_values,
            "masked_q_values": masked_q_values,
            "state_value": values,
            "advantages": advantages,
            "action_mask": encoded["action_mask"],
            "padding_mask": padding_mask,
            "state_vector": state_vector,
            "projected_state": projected,
            "recurrent_output": recurrent_output,
            "hidden_state": next_hidden_state,
            "module_tensors": encoded.get("module_tensors", {}),
            "metadata": {
                "architecture": self.config.architecture,
                "encoder_profile": self.config.encoder.profile,
                "state_dim": self.state_dim,
                "sequence_shape": tuple(q_values.shape),
                **(encoded.get("metadata") or {}),
            },
        }

    def forward_step(self, inputs: Any, hidden_state: Any = None) -> dict[str, Any]:
        output = self.forward(inputs, hidden_state=hidden_state)
        return {
            **output,
            "q_values": output["q_values"].squeeze(1),
            "masked_q_values": output["masked_q_values"].squeeze(1),
            "state_value": output["state_value"].squeeze(1),
            "advantages": output["advantages"].squeeze(1),
            "action_mask": output["action_mask"].squeeze(1),
            "padding_mask": output["padding_mask"].squeeze(1),
            "state_vector": output["state_vector"].squeeze(1),
            "projected_state": output["projected_state"].squeeze(1),
            "recurrent_output": output["recurrent_output"].squeeze(1),
        }
