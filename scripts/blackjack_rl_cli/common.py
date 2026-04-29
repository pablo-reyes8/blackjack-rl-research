from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from loss import BellmanLossConfig
from model.agents import AgentNetworkConfig, DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN
from model.encoder import EncoderConfig
from training import (
    BettingAuxiliaryConfig,
    CheckpointConfig,
    CountAuxiliaryConfig,
    DistillationConfig,
    DualEpsilonConfig,
    EpsilonScheduleConfig,
    EvaluationConfig,
    NStepConfig,
    OptimizationConfig,
    PrintConfig,
    ReplayBufferConfig,
    TargetUpdateConfig,
    TransferLearningConfig,
    TrainerConfig,
    TrainingPipelineConfig,
)


def resolve_repo_path(path_like: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    anchor = ROOT_DIR if base_dir is None else base_dir
    return (anchor / path).resolve()


def _ensure_mapping(data: Any, *, context: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return dict(data)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Top-level YAML document in {path} must be a mapping")
    return dict(data)


def _load_yaml_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"Circular config inheritance detected: {chain}")

    raw = _load_yaml_document(resolved_path)
    extends = raw.pop("extends", [])
    if isinstance(extends, (str, Path)):
        extends_paths = [extends]
    elif isinstance(extends, list):
        extends_paths = extends
    elif extends in ({}, None):
        extends_paths = []
    else:
        raise TypeError("The 'extends' field must be a string or a list of strings")

    merged: dict[str, Any] = {}
    next_stack = (*stack, resolved_path)
    for parent in extends_paths:
        parent_path = resolve_repo_path(parent, base_dir=resolved_path.parent)
        merged = _deep_merge(merged, _load_yaml_with_extends(parent_path, stack=next_stack))
    return _deep_merge(merged, raw)


def load_experiment_config(path_like: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(path_like)
    config = _load_yaml_with_extends(path)
    metadata = _ensure_mapping(config.get("metadata"), context="metadata")
    metadata.setdefault("name", path.stem)
    metadata.setdefault("source_path", str(path.relative_to(ROOT_DIR)))
    config["metadata"] = metadata

    run = _ensure_mapping(config.get("run"), context="run")
    run.setdefault("num_envs", 1)
    config["run"] = run
    return config


def build_observation_config(data: Mapping[str, Any] | None = None) -> ObservationConfig:
    values = _ensure_mapping(data, context="environment.observation")
    profile = values.pop("profile", None)
    config = ObservationConfig.for_profile(profile) if profile else ObservationConfig()
    for key, value in values.items():
        setattr(config, key, value)
    config.__post_init__()
    return config


def build_start_state_config(data: Mapping[str, Any] | None = None) -> StartStateConfig:
    return StartStateConfig(**_ensure_mapping(data, context="start_state"))


def build_blackjack_config(data: Mapping[str, Any] | None = None) -> BlackjackConfig:
    values = _ensure_mapping(data, context="environment")
    observation_data = values.pop("observation", None)
    if observation_data is not None:
        values["observation"] = build_observation_config(observation_data)
    if "bet_multipliers" in values and isinstance(values["bet_multipliers"], list):
        values["bet_multipliers"] = tuple(values["bet_multipliers"])
    return BlackjackConfig(**values)


def build_encoder_config(data: Mapping[str, Any] | None = None) -> EncoderConfig:
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


def build_model_config(data: Mapping[str, Any] | None = None) -> AgentNetworkConfig:
    values = _ensure_mapping(data, context="model")
    architecture = str(values.pop("architecture", "feedforward"))
    encoder_data = values.pop("encoder", None)
    encoder_profile = values.pop("encoder_profile", None) or _default_encoder_profile(architecture)

    if encoder_data is not None:
        values["encoder"] = build_encoder_config(encoder_data)
        config = AgentNetworkConfig(architecture=architecture, **values)
    else:
        config = AgentNetworkConfig.for_architecture(
            architecture,
            encoder_profile=encoder_profile,
            **values,
        )
    return config


def build_model(data: Mapping[str, Any] | None = None) -> torch.nn.Module:
    config = build_model_config(data)
    if config.architecture == "feedforward":
        return FeedForwardDoubleDQN(config=config)
    if config.architecture == "recurrent":
        return RecurrentDoubleDQN(config=config)
    if config.architecture == "dueling_recurrent":
        return DuelingRecurrentDoubleDQN(config=config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")


def build_training_pipeline_config(data: Mapping[str, Any] | None = None) -> TrainingPipelineConfig:
    values = _ensure_mapping(data, context="training")

    trainer_data = _ensure_mapping(values.get("trainer"), context="training.trainer")
    loss_data = _ensure_mapping(trainer_data.pop("loss", None), context="training.trainer.loss")
    if loss_data:
        phase_weights_data = _ensure_mapping(loss_data.pop("phase_weights", None), context="training.trainer.loss.phase_weights")
        if phase_weights_data:
            from loss import LossPhaseWeightConfig

            loss_data["phase_weights"] = LossPhaseWeightConfig(**phase_weights_data)
        trainer_data["loss"] = BellmanLossConfig(**loss_data)

    replay_buffer = ReplayBufferConfig(**_ensure_mapping(values.get("replay_buffer"), context="training.replay_buffer"))
    epsilon_data = _ensure_mapping(values.get("epsilon"), context="training.epsilon")
    if not epsilon_data:
        epsilon = DualEpsilonConfig()
    elif "betting" in epsilon_data or "playing" in epsilon_data:
        epsilon = DualEpsilonConfig(
            betting=EpsilonScheduleConfig(**_ensure_mapping(epsilon_data.get("betting"), context="training.epsilon.betting")),
            playing=EpsilonScheduleConfig(**_ensure_mapping(epsilon_data.get("playing"), context="training.epsilon.playing")),
        )
    else:
        epsilon = EpsilonScheduleConfig(**epsilon_data)
    n_step = NStepConfig(**_ensure_mapping(values.get("n_step"), context="training.n_step"))
    optimization = OptimizationConfig(**_ensure_mapping(values.get("optimization"), context="training.optimization"))
    target_update = TargetUpdateConfig(**_ensure_mapping(values.get("target_update"), context="training.target_update"))
    evaluation = EvaluationConfig(**_ensure_mapping(values.get("evaluation"), context="training.evaluation"))
    checkpoints = CheckpointConfig(**_ensure_mapping(values.get("checkpoints"), context="training.checkpoints"))
    betting_auxiliary = BettingAuxiliaryConfig(
        **_ensure_mapping(values.get("betting_auxiliary"), context="training.betting_auxiliary")
    )
    count_auxiliary = CountAuxiliaryConfig(
        **_ensure_mapping(values.get("count_auxiliary"), context="training.count_auxiliary")
    )
    transfer_data = _ensure_mapping(values.get("transfer"), context="training.transfer")
    distillation = DistillationConfig(
        **_ensure_mapping(transfer_data.pop("distillation", None), context="training.transfer.distillation")
    )
    transfer = TransferLearningConfig(distillation=distillation, **transfer_data)
    prints = PrintConfig(**_ensure_mapping(values.get("prints"), context="training.prints"))

    return TrainingPipelineConfig(
        trainer=TrainerConfig(**trainer_data),
        replay_buffer=replay_buffer,
        epsilon=epsilon,
        n_step=n_step,
        optimization=optimization,
        target_update=target_update,
        evaluation=evaluation,
        checkpoints=checkpoints,
        betting_auxiliary=betting_auxiliary,
        count_auxiliary=count_auxiliary,
        transfer=transfer,
        prints=prints,
    )


def build_environments(
    environment_data: Mapping[str, Any] | None,
    start_state_data: Mapping[str, Any] | None,
    *,
    num_envs: int,
    base_seed: int,
) -> list[BlackjackEnvironment]:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    env_config = build_blackjack_config(environment_data)
    start_state = build_start_state_config(start_state_data)
    return [
        BlackjackEnvironment(
            config=deepcopy(env_config),
            seed=base_seed + index,
            start_state=deepcopy(start_state),
        )
        for index in range(num_envs)
    ]


def resolve_training_setup(experiment: Mapping[str, Any]) -> dict[str, Any]:
    spec = _ensure_mapping(experiment, context="experiment")
    metadata = _ensure_mapping(spec.get("metadata"), context="metadata")
    run = _ensure_mapping(spec.get("run"), context="run")
    pipeline_config = build_training_pipeline_config(spec.get("training"))
    num_envs = int(run.get("num_envs", 1))
    base_seed = int(run.get("base_seed", pipeline_config.trainer.seed))

    environment_config = build_blackjack_config(spec.get("environment"))
    start_state_config = build_start_state_config(spec.get("start_state"))
    model = build_model(spec.get("model"))
    envs = build_environments(
        asdict(environment_config),
        asdict(start_state_config),
        num_envs=num_envs,
        base_seed=base_seed,
    )

    return {
        "metadata": metadata,
        "run": run,
        "num_envs": num_envs,
        "base_seed": base_seed,
        "environment_config": environment_config,
        "start_state_config": start_state_config,
        "pipeline_config": pipeline_config,
        "model": model,
        "envs": envs,
    }


def summarize_setup(setup: Mapping[str, Any]) -> dict[str, Any]:
    model = setup["model"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return {
        "metadata": deepcopy(setup["metadata"]),
        "run": {
            **deepcopy(setup["run"]),
            "num_envs": int(setup["num_envs"]),
            "base_seed": int(setup["base_seed"]),
        },
        "environment": asdict(setup["environment_config"]),
        "start_state": asdict(setup["start_state_config"]),
        "model": asdict(model.config),
        "training": asdict(setup["pipeline_config"]),
        "derived": {
            "state_dim": int(model.state_dim),
            "num_actions": int(getattr(model, "num_actions", 0)),
            "num_bet_actions": int(getattr(model, "num_bet_actions", 0)),
            "num_play_actions": int(getattr(model, "num_play_actions", 0)),
            "parameter_count": int(parameter_count),
            "trainable_parameter_count": int(trainable_parameter_count),
        },
    }


def load_checkpoint_payload(path_like: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = resolve_repo_path(path_like)
    return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(payload: Mapping[str, Any]) -> torch.nn.Module:
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["online_model_state_dict"])
    return model


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def to_pretty_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, sort_keys=False)


def write_json_file(path_like: str | Path, payload: Any) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)
        handle.write("\n")
    return path
