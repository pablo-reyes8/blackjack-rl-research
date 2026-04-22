from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoints import load_checkpoint, load_checkpoint_into_trainer
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
    resume: bool = False,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] | None = None
    resolved_pipeline_config = pipeline_config
    if resume:
        if resume_checkpoint_path is None:
            raise ValueError("resume_checkpoint_path is required when resume=True")
        checkpoint = load_checkpoint(resume_checkpoint_path)
        if resolved_pipeline_config is None:
            resolved_pipeline_config = TrainingPipelineConfig.from_dict(checkpoint["pipeline_config"])

    trainer = build_trainer(
        envs,
        model,
        pipeline_config=resolved_pipeline_config,
        target_network=target_network,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if checkpoint is not None:
        load_checkpoint_into_trainer(trainer, checkpoint)

    result = trainer.train()
    result["trainer"] = trainer
    return result
