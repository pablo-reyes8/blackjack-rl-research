from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BellmanLossConfig:
    gamma: float = 0.99
    loss_type: str = "huber"
    validate_current_actions: bool = True
    validate_next_action_mask: bool = True
    allow_terminal_without_legal_next_action: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.loss_type not in {"huber", "mse"}:
            raise ValueError("loss_type must be 'huber' or 'mse'")
