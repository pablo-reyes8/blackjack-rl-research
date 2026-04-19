from __future__ import annotations

from dataclasses import dataclass

from .config import EpsilonScheduleConfig


@dataclass(slots=True)
class EpsilonScheduler:
    config: EpsilonScheduleConfig
    step_count: int = 0

    def value(self) -> float:
        fraction = min(self.step_count / self.config.decay_steps, 1.0)
        return self.config.start + fraction * (self.config.end - self.config.start)

    def step(self, n: int = 1) -> float:
        self.step_count += n
        return self.value()

    def evaluation_value(self) -> float:
        return self.config.evaluation_epsilon

    def state_dict(self) -> dict[str, float | int]:
        return {"step_count": self.step_count, "current_epsilon": self.value()}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.step_count = int(state.get("step_count", 0))
