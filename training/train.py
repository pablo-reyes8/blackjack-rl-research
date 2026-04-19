from __future__ import annotations

from typing import Any

from .config import TrainingPipelineConfig
from .pipeline import build_trainer


def train_model(
    envs: Any,
    model: Any,
    *,
    pipeline_config: TrainingPipelineConfig | None = None,
    target_network: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    trainer = build_trainer(
        envs,
        model,
        pipeline_config=pipeline_config,
        target_network=target_network,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    result = trainer.train()
    result["trainer"] = trainer
    return result
