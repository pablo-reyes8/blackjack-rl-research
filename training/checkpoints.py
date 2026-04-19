from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .config import CheckpointConfig


def _serialize_config(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config


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
