from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .config import CheckpointConfig


def _serialize_config(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def _pad_expanded_input_weights(saved_tensor: torch.Tensor, current_tensor: torch.Tensor) -> torch.Tensor:
    padded = torch.zeros_like(current_tensor)
    padded[..., : saved_tensor.shape[-1]] = saved_tensor.to(dtype=current_tensor.dtype)
    return padded


def _adapt_model_state_dict_for_compatibility(model: Any, model_state_dict: dict[str, Any]) -> dict[str, Any]:
    current_state_dict = model.state_dict()
    adapted_state_dict = dict(model_state_dict)

    for key, saved_value in model_state_dict.items():
        current_value = current_state_dict.get(key)
        if not isinstance(saved_value, torch.Tensor) or not isinstance(current_value, torch.Tensor):
            continue
        if saved_value.shape == current_value.shape:
            continue

        can_pad_input_dim = (
            saved_value.ndim >= 2
            and saved_value.ndim == current_value.ndim
            and saved_value.shape[:-1] == current_value.shape[:-1]
            and saved_value.shape[-1] < current_value.shape[-1]
        )
        if can_pad_input_dim:
            # Older checkpoints can miss newly added encoder features on the first input projection.
            adapted_state_dict[key] = _pad_expanded_input_weights(saved_value, current_value)

    return adapted_state_dict


def _load_model_state(model: Any, model_state_dict: dict[str, Any], *, label: str) -> None:
    adapted_state_dict = _adapt_model_state_dict_for_compatibility(model, model_state_dict)
    try:
        incompatible = model.load_state_dict(adapted_state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load {label} state from checkpoint") from exc

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {label} state is incompatible. "
            f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}"
        )


def _adapt_optimizer_state_dict_for_compatibility(optimizer: Any, optimizer_state_dict: dict[str, Any]) -> dict[str, Any]:
    adapted_state_dict = deepcopy(optimizer_state_dict)
    saved_param_groups = adapted_state_dict.get("param_groups", [])
    saved_state = adapted_state_dict.get("state", {})

    for optimizer_group, saved_group in zip(optimizer.param_groups, saved_param_groups):
        for parameter, saved_param_id in zip(optimizer_group["params"], saved_group.get("params", [])):
            parameter_state = saved_state.get(saved_param_id)
            if not isinstance(parameter_state, dict):
                continue

            parameter_tensor = parameter.detach()
            for state_key, state_value in list(parameter_state.items()):
                if not isinstance(state_value, torch.Tensor):
                    continue
                if state_value.shape == parameter_tensor.shape:
                    continue

                can_pad_input_dim = (
                    state_value.ndim >= 2
                    and state_value.ndim == parameter_tensor.ndim
                    and state_value.shape[:-1] == parameter_tensor.shape[:-1]
                    and state_value.shape[-1] < parameter_tensor.shape[-1]
                )
                if can_pad_input_dim:
                    parameter_state[state_key] = _pad_expanded_input_weights(state_value, parameter_tensor)

    return adapted_state_dict


def load_checkpoint_into_trainer(trainer: Any, checkpoint: dict[str, Any]) -> None:
    _load_model_state(trainer.online_network, checkpoint["online_model_state_dict"], label="online model")
    _load_model_state(trainer.target_network, checkpoint["target_model_state_dict"], label="target model")

    try:
        trainer.optimizer.load_state_dict(
            _adapt_optimizer_state_dict_for_compatibility(trainer.optimizer, checkpoint["optimizer_state_dict"])
        )
    except ValueError as exc:
        raise RuntimeError("Failed to restore optimizer state from checkpoint") from exc

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        if trainer.scheduler is None:
            raise RuntimeError("Checkpoint includes scheduler state but the current trainer has no scheduler")
        trainer.scheduler.load_state_dict(scheduler_state)

    trainer.load_state_dict(checkpoint.get("trainer_state", {}))


class CheckpointManager:
    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self.directory = config.directory_path
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_metric_value: float | None = None

    def _build_payload(self, trainer: Any, *, extra_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "online_model_state_dict": trainer.online_network.state_dict(),
            "target_model_state_dict": trainer.target_network.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
            "trainer_state": trainer.state_dict(),
            "pipeline_config": _serialize_config(trainer.pipeline_config),
            "model_config": _serialize_config(trainer.online_network.config),
            "metrics": extra_metrics or {},
        }

    def save_latest(self, trainer: Any, *, metrics: dict[str, Any]) -> Path | None:
        if not self.config.save_latest:
            return None
        path = self.directory / "latest.pt"
        torch.save(self._build_payload(trainer, extra_metrics=metrics), path)
        return path

    def save_periodic(self, trainer: Any, *, metrics: dict[str, Any]) -> Path | None:
        if not self.config.save_periodic:
            return None
        path = self.directory / f"step_{trainer.update_count:08d}.pt"
        torch.save(self._build_payload(trainer, extra_metrics=metrics), path)
        return path

    def save_best(self, trainer: Any, *, metrics: dict[str, Any]) -> Path | None:
        if not self.config.save_best_eval:
            return None
        metric_value = metrics.get(self.config.best_metric_name)
        if metric_value is None:
            return None
        metric_value = float(metric_value)

        should_save = self.best_metric_value is None
        if self.best_metric_value is not None:
            if self.config.maximize_best_metric:
                should_save = metric_value > self.best_metric_value
            else:
                should_save = metric_value < self.best_metric_value

        if not should_save:
            return None

        self.best_metric_value = metric_value
        path = self.directory / "best_eval.pt"
        torch.save(self._build_payload(trainer, extra_metrics=metrics), path)
        return path
