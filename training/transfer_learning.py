from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from enviroment_bj.core import hand_value, rank_value
from model.agents import AgentNetworkConfig, DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN
from model.agents.common import apply_action_mask
from model.encoder import EncoderConfig

from .checkpoints import load_checkpoint
from .config import DistillationConfig


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return dict(value)


def _build_encoder_config(data: Mapping[str, Any] | None) -> EncoderConfig:
    values = _ensure_mapping(data, context="model.encoder")
    profile = values.pop("profile", None)
    config = EncoderConfig.for_profile(profile) if profile else EncoderConfig()
    for key, value in values.items():
        setattr(config, key, value)
    config.__post_init__()
    return config


def _default_encoder_profile(architecture: str) -> str:
    if architecture == "feedforward":
        return "minimal_basic_strategy"
    return "table_realistic_default"


def build_model_from_config(data: Mapping[str, Any] | None) -> nn.Module:
    values = _ensure_mapping(data, context="model")
    architecture = str(values.pop("architecture", "feedforward"))
    encoder_data = values.pop("encoder", None)
    encoder_profile = values.pop("encoder_profile", None) or _default_encoder_profile(architecture)

    if encoder_data is not None:
        values["encoder"] = _build_encoder_config(encoder_data)
        config = AgentNetworkConfig(architecture=architecture, **values)
    else:
        config = AgentNetworkConfig.for_architecture(
            architecture,
            encoder_profile=encoder_profile,
            **values,
        )

    if config.architecture == "feedforward":
        return FeedForwardDoubleDQN(config=config)
    if config.architecture == "recurrent":
        return RecurrentDoubleDQN(config=config)
    if config.architecture == "dueling_recurrent":
        return DuelingRecurrentDoubleDQN(config=config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary")
    if "online_model_state_dict" not in payload:
        raise KeyError("Checkpoint payload is missing required key 'online_model_state_dict'")
    return payload


def _pad_expanded_input_weights(saved_tensor: torch.Tensor, current_tensor: torch.Tensor) -> torch.Tensor:
    padded = torch.zeros_like(current_tensor)
    padded[..., : saved_tensor.shape[-1]] = saved_tensor.to(dtype=current_tensor.dtype)
    return padded


def _can_pad_input_dim(saved_value: torch.Tensor, current_value: torch.Tensor) -> bool:
    return (
        saved_value.ndim >= 2
        and saved_value.ndim == current_value.ndim
        and saved_value.shape[:-1] == current_value.shape[:-1]
        and saved_value.shape[-1] < current_value.shape[-1]
    )


def _warm_start_model_state(
    model: nn.Module,
    saved_state: Mapping[str, Any],
    *,
    allow_input_dim_padding: bool,
    allow_partial: bool,
    verbose: bool,
) -> dict[str, Any]:
    current_state = model.state_dict()
    adapted_state: dict[str, Any] = {}
    loaded_keys: set[str] = set()
    report = {
        "loaded": [],
        "padded": [],
        "skipped": [],
        "missing": [],
        "unexpected": [],
    }

    for key, saved_value in saved_state.items():
        current_value = current_state.get(key)
        if current_value is None:
            report["unexpected"].append(key)
            continue

        if isinstance(saved_value, torch.Tensor) and isinstance(current_value, torch.Tensor):
            if saved_value.shape == current_value.shape:
                adapted_state[key] = saved_value.to(dtype=current_value.dtype)
                report["loaded"].append(key)
                loaded_keys.add(key)
                continue
            if allow_input_dim_padding and _can_pad_input_dim(saved_value, current_value):
                adapted_state[key] = _pad_expanded_input_weights(saved_value, current_value)
                report["padded"].append(key)
                loaded_keys.add(key)
                continue
            report["skipped"].append(key)
            continue

        if type(saved_value) is type(current_value):
            adapted_state[key] = saved_value
            report["loaded"].append(key)
            loaded_keys.add(key)
            continue

        report["skipped"].append(key)

    report["missing"] = sorted(key for key in current_state.keys() if key not in loaded_keys)

    if not allow_partial and (report["skipped"] or report["missing"] or report["unexpected"]):
        raise RuntimeError(
            "Warm start encountered incompatible state. "
            f"skipped={report['skipped']}, missing={report['missing']}, unexpected={report['unexpected']}"
        )

    incompatible = model.load_state_dict(adapted_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Warm start loaded unexpected keys into model: {incompatible.unexpected_keys}")

    if verbose:
        print(
            "Warm start summary: "
            f"loaded={len(report['loaded'])} | padded={len(report['padded'])} | "
            f"skipped={len(report['skipped'])} | missing={len(report['missing'])} | "
            f"unexpected={len(report['unexpected'])}"
        )

    return report


def warm_start_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    state_key: str = "online_model_state_dict",
    map_location: str | torch.device = "cpu",
    allow_input_dim_padding: bool = True,
    allow_partial: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    if state_key not in payload:
        raise KeyError(f"Checkpoint payload is missing requested state key '{state_key}'")
    saved_state = payload[state_key]
    if not isinstance(saved_state, Mapping):
        raise TypeError(f"Checkpoint state '{state_key}' must be a mapping")
    report = _warm_start_model_state(
        model,
        saved_state,
        allow_input_dim_padding=allow_input_dim_padding,
        allow_partial=allow_partial,
        verbose=verbose,
    )
    report["checkpoint_path"] = str(Path(checkpoint_path))
    report["state_key"] = state_key
    return report


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def freeze_playing_policy_parts(model: nn.Module) -> None:
    for attribute_name in (
        "module_gates",
        "backbone",
        "input_projection",
        "recurrent_backbone",
        "play_adapter",
        "play_head",
    ):
        module = getattr(model, attribute_name, None)
        if isinstance(module, nn.Module):
            freeze_module(module)


def _collect_trainable_parameters(module: nn.Module, seen: set[int]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        parameters.append(parameter)
        seen.add(id(parameter))
    return parameters


def build_optimizer_with_param_groups(
    model: nn.Module,
    *,
    backbone_lr: float,
    play_lr: float,
    bet_lr: float,
    default_lr: float,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    optimizer_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be 'adam' or 'adamw'")

    group_definitions = [
        (
            "backbone",
            [getattr(model, name, None) for name in ("module_gates", "backbone", "input_projection", "recurrent_backbone")],
            backbone_lr,
        ),
        ("play", [getattr(model, name, None) for name in ("play_adapter", "play_head")], play_lr),
        ("bet", [getattr(model, name, None) for name in ("bet_adapter", "bet_head")], bet_lr),
    ]

    seen: set[int] = set()
    param_groups: list[dict[str, Any]] = []
    for group_name, modules, lr in group_definitions:
        params: list[nn.Parameter] = []
        for module in modules:
            if isinstance(module, nn.Module):
                params.extend(_collect_trainable_parameters(module, seen))
        if params:
            param_groups.append(
                {
                    "name": group_name,
                    "params": params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

    default_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in seen
    ]
    if default_params:
        param_groups.append(
            {
                "name": "default",
                "params": default_params,
                "lr": default_lr,
                "weight_decay": weight_decay,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters are available to build optimizer param groups")

    return optimizer_cls(param_groups)


def adapt_response_to_minimal_basic_strategy(response: Mapping[str, Any]) -> dict[str, Any]:
    observation = _ensure_mapping(response.get("observation"), context="response.observation")
    teacher_observation = {
        "profile": "minimal_basic_strategy",
        "mode": "basic_strategy",
        "dealer_upcard": observation.get("dealer_upcard"),
        "dealer_upcard_value": observation.get("dealer_upcard_value"),
        "decision_phase": observation.get("decision_phase"),
        "available_bet_multipliers": list(observation.get("available_bet_multipliers") or []),
        "current_hand_total": observation.get("current_hand_total"),
        "current_hand_is_soft": observation.get("current_hand_is_soft"),
        "hand_context": deepcopy(observation.get("hand_context") or {}),
        "insurance_context": deepcopy(observation.get("insurance_context") or {}),
        "betting_context": deepcopy(observation.get("betting_context") or {}),
        "current_bet": observation.get("current_bet"),
    }

    current_hand_cards = observation.get("current_hand_cards")
    if current_hand_cards is not None and (
        teacher_observation["current_hand_total"] is None or teacher_observation["current_hand_is_soft"] is None
    ):
        total, is_soft = hand_value(current_hand_cards)
        teacher_observation["current_hand_total"] = total
        teacher_observation["current_hand_is_soft"] = is_soft

    if teacher_observation["dealer_upcard_value"] is None and teacher_observation["dealer_upcard"] is not None:
        teacher_observation["dealer_upcard_value"] = rank_value(str(teacher_observation["dealer_upcard"]))

    return {
        "observation": teacher_observation,
        "table_rules": deepcopy(response.get("table_rules") or {}),
        "action_mask": response.get("action_mask"),
    }


def encode_teacher_state(teacher_model: nn.Module, response: Mapping[str, Any]) -> dict[str, Any]:
    teacher_profile = getattr(getattr(teacher_model, "config", None), "encoder", None)
    teacher_profile_name = getattr(teacher_profile, "profile", None)
    teacher_response = (
        adapt_response_to_minimal_basic_strategy(response)
        if teacher_profile_name == "minimal_basic_strategy"
        else response
    )
    encoded = teacher_model.encoder.encode_state_only(teacher_response)
    return {
        "state_vector": encoded["state_vector"].detach().cpu(),
        "action_mask": encoded["action_mask"].detach().cpu(),
    }


def load_teacher_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise KeyError("Checkpoint payload is missing required key 'model_config'")
    teacher_model = build_model_from_config(model_config)
    _warm_start_model_state(
        teacher_model,
        payload["online_model_state_dict"],
        allow_input_dim_padding=True,
        allow_partial=False,
        verbose=False,
    )
    teacher_model.to(device)
    teacher_model.eval()
    freeze_module(teacher_model)
    return teacher_model


def distillation_weight(config: DistillationConfig, update_count: int) -> float:
    fraction = min(max(float(update_count), 0.0) / float(config.decay_steps), 1.0)
    return float(config.weight + fraction * (config.final_weight - config.weight))


def _resolve_distillation_slice(
    action_mask: torch.Tensor,
    *,
    playing_action_slice: slice,
    playing_only: bool,
) -> tuple[torch.Tensor, slice]:
    if playing_only:
        return action_mask[..., playing_action_slice], playing_action_slice
    return action_mask, slice(0, action_mask.shape[-1])


def _expand_valid_rows(valid_rows: torch.Tensor, target_ndim: int) -> torch.Tensor:
    expanded = valid_rows.to(torch.bool)
    while expanded.ndim < target_ndim:
        expanded = expanded.unsqueeze(-1)
    return expanded


def compute_q_distillation_loss(
    student_output: Mapping[str, torch.Tensor],
    teacher_output: Mapping[str, torch.Tensor],
    action_mask: torch.Tensor,
    *,
    playing_action_slice: slice,
    valid_rows: torch.Tensor | None = None,
    playing_only: bool = True,
) -> torch.Tensor:
    active_mask, action_slice = _resolve_distillation_slice(
        action_mask,
        playing_action_slice=playing_action_slice,
        playing_only=playing_only,
    )
    if valid_rows is not None:
        active_mask = active_mask & _expand_valid_rows(valid_rows, active_mask.ndim)

    if not active_mask.any():
        return student_output["q_values"].new_zeros(())

    student_q = student_output["q_values"][..., action_slice]
    teacher_q = teacher_output["q_values"][..., action_slice].detach()
    loss_values = (student_q - teacher_q) ** 2
    return loss_values[active_mask].mean()


def compute_policy_kl_distillation_loss(
    student_output: Mapping[str, torch.Tensor],
    teacher_output: Mapping[str, torch.Tensor],
    action_mask: torch.Tensor,
    *,
    playing_action_slice: slice,
    temperature: float,
    valid_rows: torch.Tensor | None = None,
    playing_only: bool = True,
) -> torch.Tensor:
    active_mask, action_slice = _resolve_distillation_slice(
        action_mask,
        playing_action_slice=playing_action_slice,
        playing_only=playing_only,
    )
    row_mask = active_mask.any(dim=-1)
    if valid_rows is not None:
        row_mask = row_mask & valid_rows.to(torch.bool)
    if not row_mask.any():
        return student_output["q_values"].new_zeros(())

    student_q = student_output["q_values"][..., action_slice]
    teacher_q = teacher_output["q_values"][..., action_slice].detach()
    masked_student_q = apply_action_mask(student_q, active_mask)[row_mask] / temperature
    masked_teacher_q = apply_action_mask(teacher_q, active_mask)[row_mask] / temperature
    teacher_probs = F.softmax(masked_teacher_q, dim=-1)
    student_log_probs = F.log_softmax(masked_student_q, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def compute_greedy_ce_distillation_loss(
    student_output: Mapping[str, torch.Tensor],
    teacher_output: Mapping[str, torch.Tensor],
    action_mask: torch.Tensor,
    *,
    playing_action_slice: slice,
    valid_rows: torch.Tensor | None = None,
    playing_only: bool = True,
) -> torch.Tensor:
    active_mask, action_slice = _resolve_distillation_slice(
        action_mask,
        playing_action_slice=playing_action_slice,
        playing_only=playing_only,
    )
    row_mask = active_mask.any(dim=-1)
    if valid_rows is not None:
        row_mask = row_mask & valid_rows.to(torch.bool)
    if not row_mask.any():
        return student_output["q_values"].new_zeros(())

    student_q = student_output["q_values"][..., action_slice]
    teacher_q = teacher_output["q_values"][..., action_slice].detach()
    masked_student_q = apply_action_mask(student_q, active_mask)[row_mask]
    masked_teacher_q = apply_action_mask(teacher_q, active_mask)[row_mask]
    teacher_actions = masked_teacher_q.argmax(dim=-1)
    return F.cross_entropy(masked_student_q, teacher_actions)


def compute_distillation_loss(
    student_output: Mapping[str, torch.Tensor],
    teacher_output: Mapping[str, torch.Tensor],
    action_mask: torch.Tensor,
    *,
    config: DistillationConfig,
    playing_action_slice: slice,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    if config.mode == "q_mse":
        return compute_q_distillation_loss(
            student_output,
            teacher_output,
            action_mask,
            playing_action_slice=playing_action_slice,
            valid_rows=valid_rows,
            playing_only=config.playing_only,
        )
    if config.mode == "policy_kl":
        return compute_policy_kl_distillation_loss(
            student_output,
            teacher_output,
            action_mask,
            playing_action_slice=playing_action_slice,
            temperature=config.temperature,
            valid_rows=valid_rows,
            playing_only=config.playing_only,
        )
    if config.mode == "greedy_ce":
        return compute_greedy_ce_distillation_loss(
            student_output,
            teacher_output,
            action_mask,
            playing_action_slice=playing_action_slice,
            valid_rows=valid_rows,
            playing_only=config.playing_only,
        )
    raise ValueError(f"Unsupported distillation mode: {config.mode}")
