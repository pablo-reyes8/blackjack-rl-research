from __future__ import annotations

from dataclasses import dataclass, field

from .config import DualEpsilonConfig, EpsilonScheduleConfig


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


@dataclass(slots=True)
class DualEpsilonScheduler:
    config: DualEpsilonConfig
    betting: EpsilonScheduler = field(init=False)
    playing: EpsilonScheduler = field(init=False)

    def __post_init__(self) -> None:
        self.betting = EpsilonScheduler(self.config.betting)
        self.playing = EpsilonScheduler(self.config.playing)

    def _scheduler_for_phase(self, decision_phase: str | None) -> EpsilonScheduler:
        if decision_phase == "betting":
            return self.betting
        return self.playing

    def value(self, decision_phase: str | None) -> float:
        return self._scheduler_for_phase(decision_phase).value()

    def step(self, decision_phase: str | None, n: int = 1) -> float:
        return self._scheduler_for_phase(decision_phase).step(n)

    def evaluation_value(self, decision_phase: str | None) -> float:
        return self._scheduler_for_phase(decision_phase).evaluation_value()

    def current_values(self) -> dict[str, float]:
        return {
            "epsilon_betting": self.betting.value(),
            "epsilon_playing": self.playing.value(),
        }

    def state_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            "betting": self.betting.state_dict(),
            "playing": self.playing.state_dict(),
        }

    def load_state_dict(self, state: dict[str, dict[str, float | int]]) -> None:
        self.betting.load_state_dict(state.get("betting", {}))
        self.playing.load_state_dict(state.get("playing", {}))
