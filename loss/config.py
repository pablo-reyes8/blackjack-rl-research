from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LossPhaseWeightConfig:
    enabled: bool = True
    betting_weight: float = 1.5
    playing_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.betting_weight <= 0 or self.playing_weight <= 0:
            raise ValueError("betting_weight and playing_weight must be positive")


@dataclass(slots=True)
class BellmanLossConfig:
    gamma: float = 0.99
    loss_type: str = "huber"
    validate_current_actions: bool = True
    validate_next_action_mask: bool = True
    allow_terminal_without_legal_next_action: bool = True
    phase_weights: LossPhaseWeightConfig = field(default_factory=LossPhaseWeightConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.loss_type not in {"huber", "mse"}:
            raise ValueError("loss_type must be 'huber' or 'mse'")
        if not isinstance(self.phase_weights, LossPhaseWeightConfig):
            raise TypeError("phase_weights must be a LossPhaseWeightConfig instance")
