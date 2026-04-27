from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoints import load_checkpoint, load_checkpoint_into_trainer
from .config import TrainingPipelineConfig
from .pipeline import build_trainer
from .transfer_learning import warm_start_model_from_checkpoint


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
    warm_start_checkpoint_path: str | Path | None = None,
    warm_start_state_key: str = "online_model_state_dict",
    warm_start_allow_input_dim_padding: bool = True,
    warm_start_allow_partial: bool = True,
    warm_start_verbose: bool = False,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] | None = None
    warm_start_report: dict[str, Any] | None = None
    resolved_pipeline_config = pipeline_config
    if resume:
        if resume_checkpoint_path is None:
            raise ValueError("resume_checkpoint_path is required when resume=True")
        checkpoint = load_checkpoint(resume_checkpoint_path)
        if resolved_pipeline_config is None:
            resolved_pipeline_config = TrainingPipelineConfig.from_dict(checkpoint["pipeline_config"])

    config_warm_start_path = None
    if resolved_pipeline_config is not None and resolved_pipeline_config.transfer.enabled:
        config_warm_start_path = resolved_pipeline_config.transfer.warm_start_checkpoint_path
    resolved_warm_start_path = warm_start_checkpoint_path or config_warm_start_path
    if resume and resolved_warm_start_path is not None:
        raise ValueError("warm start cannot be used together with resume=True")
    if resolved_warm_start_path is not None:
        warm_start_report = warm_start_model_from_checkpoint(
            model,
            resolved_warm_start_path,
            state_key=warm_start_state_key,
            allow_input_dim_padding=warm_start_allow_input_dim_padding,
            allow_partial=warm_start_allow_partial,
            verbose=warm_start_verbose,
        )

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
    result["warm_start_report"] = warm_start_report
    return result
