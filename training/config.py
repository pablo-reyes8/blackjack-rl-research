from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loss import BellmanLossConfig, LossPhaseWeightConfig


@dataclass(slots=True)
class ReplayBufferConfig:
    capacity: int = 50_000
    batch_size: int = 64
    warmup_size: int = 1_000
    sequence_length: int = 8
    min_sequence_length: int = 2

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.batch_size <= 0 or self.warmup_size < 0:
            raise ValueError("capacity and batch_size must be positive, warmup_size must be non-negative")
        if self.sequence_length <= 0 or self.min_sequence_length <= 0:
            raise ValueError("sequence_length and min_sequence_length must be positive")
        if self.min_sequence_length > self.sequence_length:
            raise ValueError("min_sequence_length cannot be greater than sequence_length")


@dataclass(slots=True)
class EpsilonScheduleConfig:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 25_000
    evaluation_epsilon: float = 0.0

    def __post_init__(self) -> None:
        for name in ("start", "end", "evaluation_epsilon"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")


@dataclass(slots=True)
class DualEpsilonConfig:
    betting: EpsilonScheduleConfig = field(
        default_factory=lambda: EpsilonScheduleConfig(
            start=1.0,
            end=0.10,
            decay_steps=40_000,
            evaluation_epsilon=0.0,
        )
    )
    playing: EpsilonScheduleConfig = field(
        default_factory=lambda: EpsilonScheduleConfig(
            start=1.0,
            end=0.03,
            decay_steps=25_000,
            evaluation_epsilon=0.0,
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.betting, EpsilonScheduleConfig):
            raise TypeError("betting must be an EpsilonScheduleConfig instance")
        if not isinstance(self.playing, EpsilonScheduleConfig):
            raise TypeError("playing must be an EpsilonScheduleConfig instance")

    @classmethod
    def from_shared(cls, config: EpsilonScheduleConfig) -> DualEpsilonConfig:
        if not isinstance(config, EpsilonScheduleConfig):
            raise TypeError("config must be an EpsilonScheduleConfig instance")
        return cls(betting=deepcopy(config), playing=deepcopy(config))


@dataclass(slots=True)
class NStepConfig:
    enabled: bool = False
    n_steps: int = 3

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")


@dataclass(slots=True)
class OptimizationConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "none"
    scheduler_step_size: int = 1_000
    scheduler_gamma: float = 0.99
    gradient_clipping: bool = True
    max_grad_norm: float = 5.0

    def __post_init__(self) -> None:
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.scheduler not in {"none", "step"}:
            raise ValueError("scheduler must be 'none' or 'step'")
        if self.scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be positive")
        if not 0 < self.scheduler_gamma <= 1:
            raise ValueError("scheduler_gamma must be in (0, 1]")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class TargetUpdateConfig:
    mode: str = "hard"
    hard_update_interval: int = 250
    soft_tau: float = 0.005

    def __post_init__(self) -> None:
        if self.mode not in {"hard", "soft"}:
            raise ValueError("mode must be 'hard' or 'soft'")
        if self.hard_update_interval <= 0:
            raise ValueError("hard_update_interval must be positive")
        if not 0 < self.soft_tau <= 1:
            raise ValueError("soft_tau must be in (0, 1]")


@dataclass(slots=True)
class EvaluationConfig:
    enabled: bool = True
    every_n_epochs: int = 1
    num_rounds: int = 100
    max_decisions: int = 10_000

    def __post_init__(self) -> None:
        if self.every_n_epochs <= 0 or self.num_rounds <= 0 or self.max_decisions <= 0:
            raise ValueError("every_n_epochs, num_rounds, and max_decisions must be positive")


@dataclass(slots=True)
class CheckpointConfig:
    directory: str = "training_checkpoints"
    save_latest: bool = True
    save_best_eval: bool = True
    save_periodic: bool = True
    periodic_interval_updates: int = 500
    best_metric_name: str = "ev_per_1000_hands"
    maximize_best_metric: bool = True

    def __post_init__(self) -> None:
        if self.periodic_interval_updates <= 0:
            raise ValueError("periodic_interval_updates must be positive")

    @property
    def directory_path(self) -> Path:
        return Path(self.directory)


@dataclass(slots=True)
class DistillationConfig:
    enabled: bool = False
    weight: float = 0.0
    mode: str = "q_mse"
    temperature: float = 1.0
    playing_only: bool = True
    decay_steps: int = 50_000
    final_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.weight < 0 or self.final_weight < 0:
            raise ValueError("weight and final_weight must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.mode not in {"q_mse", "policy_kl", "greedy_ce"}:
            raise ValueError("mode must be 'q_mse', 'policy_kl', or 'greedy_ce'")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")


@dataclass(slots=True)
class BettingAuxiliaryConfig:
    enabled: bool = False
    mode: str = "count_proxy_ce"
    weight: float = 0.10
    final_weight: float = 0.02
    decay_steps: int = 50_000
    threshold_2x: float = 1.0
    threshold_3x: float = 2.0
    threshold_4x: float = 4.0
    min_observed_cards: int = 12
    betting_phase_only: bool = True
    bet_multipliers: tuple[int, ...] = (1, 2, 3, 4)

    def __post_init__(self) -> None:
        if self.mode != "count_proxy_ce":
            raise ValueError("Only mode='count_proxy_ce' is currently supported")
        if self.weight < 0 or self.final_weight < 0:
            raise ValueError("weight and final_weight must be non-negative")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if not (self.threshold_2x <= self.threshold_3x <= self.threshold_4x):
            raise ValueError("thresholds must satisfy threshold_2x <= threshold_3x <= threshold_4x")
        if self.min_observed_cards < 0:
            raise ValueError("min_observed_cards must be non-negative")
        self.bet_multipliers = tuple(int(multiplier) for multiplier in self.bet_multipliers)
        if self.enabled and self.bet_multipliers != (1, 2, 3, 4):
            raise ValueError("Betting auxiliary mode currently requires bet_multipliers=(1, 2, 3, 4)")


@dataclass(slots=True)
class TransferLearningConfig:
    enabled: bool = False
    teacher_checkpoint_path: str | None = None
    warm_start_checkpoint_path: str | None = None
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.distillation, DistillationConfig):
            raise TypeError("distillation must be a DistillationConfig instance")


@dataclass(slots=True)
class PrintConfig:
    enable: bool = True
    print_run_summary: bool = True
    print_warmup_interval: int = 200
    print_update_interval: int = 100
    print_collection_interval: int = 500
    print_epoch_header: bool = True
    print_epoch_summary: bool = True
    print_eval_summary: bool = True
    include_segment_details: bool = False

    def __post_init__(self) -> None:
        if self.print_warmup_interval <= 0 or self.print_update_interval <= 0 or self.print_collection_interval <= 0:
            raise ValueError("print intervals must be positive")


@dataclass(slots=True)
class TrainerConfig:
    total_epochs: int = 10
    env_steps_per_epoch: int = 1_000
    train_frequency: int = 4
    updates_per_train_step: int = 1
    max_updates_per_epoch: int | None = None
    device: str = "auto"
    seed: int = 7
    reset_hidden_on_round_end: bool = False
    sequence_end_on_done: bool = False
    flush_partial_sequences_at_epoch_end: bool = True
    loss: BellmanLossConfig = field(default_factory=BellmanLossConfig)

    def __post_init__(self) -> None:
        if self.total_epochs <= 0 or self.env_steps_per_epoch <= 0:
            raise ValueError("total_epochs and env_steps_per_epoch must be positive")
        if self.train_frequency <= 0 or self.updates_per_train_step <= 0:
            raise ValueError("train_frequency and updates_per_train_step must be positive")
        if self.max_updates_per_epoch is not None and self.max_updates_per_epoch <= 0:
            raise ValueError("max_updates_per_epoch must be positive when provided")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


@dataclass(slots=True)
class TrainingPipelineConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    epsilon: DualEpsilonConfig | EpsilonScheduleConfig = field(default_factory=DualEpsilonConfig)
    n_step: NStepConfig = field(default_factory=NStepConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    target_update: TargetUpdateConfig = field(default_factory=TargetUpdateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    betting_auxiliary: BettingAuxiliaryConfig = field(default_factory=BettingAuxiliaryConfig)
    transfer: TransferLearningConfig = field(default_factory=TransferLearningConfig)
    prints: PrintConfig = field(default_factory=PrintConfig)

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, EpsilonScheduleConfig):
            self.epsilon = DualEpsilonConfig.from_shared(self.epsilon)
        elif not isinstance(self.epsilon, DualEpsilonConfig):
            raise TypeError("epsilon must be an EpsilonScheduleConfig or DualEpsilonConfig instance")
        if not isinstance(self.n_step, NStepConfig):
            raise TypeError("n_step must be an NStepConfig instance")
        if not isinstance(self.betting_auxiliary, BettingAuxiliaryConfig):
            raise TypeError("betting_auxiliary must be a BettingAuxiliaryConfig instance")
        if not isinstance(self.transfer, TransferLearningConfig):
            raise TypeError("transfer must be a TransferLearningConfig instance")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPipelineConfig:
        trainer_config = dict(data.get("trainer", {}))
        loss_config = dict(trainer_config.get("loss", {}))
        phase_weights = LossPhaseWeightConfig(**loss_config.pop("phase_weights", {}))
        trainer_config["loss"] = BellmanLossConfig(phase_weights=phase_weights, **loss_config)

        epsilon_config = data.get("epsilon", {})
        betting_auxiliary_config = BettingAuxiliaryConfig(**data.get("betting_auxiliary", {}))
        transfer_config = dict(data.get("transfer", {}))
        distillation_config = DistillationConfig(**transfer_config.pop("distillation", {}))
        return cls(
            trainer=TrainerConfig(**trainer_config),
            replay_buffer=ReplayBufferConfig(**data.get("replay_buffer", {})),
            epsilon=DualEpsilonConfig(
                betting=EpsilonScheduleConfig(**epsilon_config.get("betting", {})),
                playing=EpsilonScheduleConfig(**epsilon_config.get("playing", {})),
            ),
            n_step=NStepConfig(**data.get("n_step", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
            target_update=TargetUpdateConfig(**data.get("target_update", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            checkpoints=CheckpointConfig(**data.get("checkpoints", {})),
            betting_auxiliary=betting_auxiliary_config,
            transfer=TransferLearningConfig(distillation=distillation_config, **transfer_config),
            prints=PrintConfig(**data.get("prints", {})),
        )
