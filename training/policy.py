from __future__ import annotations

import random
from typing import Any, Sequence

import torch

from .config import DualEpsilonConfig, EpsilonScheduleConfig


def sample_random_legal_action(action_mask: torch.Tensor, rng: random.Random) -> int:
    legal_indices = torch.nonzero(action_mask, as_tuple=False).flatten().tolist()
    if not legal_indices:
        raise ValueError("No legal actions are available for epsilon-greedy selection")
    return int(rng.choice(legal_indices))


def select_epsilon_greedy_action(
    *,
    masked_q_values: torch.Tensor,
    action_mask: torch.Tensor,
    epsilon: float,
    rng: random.Random,
) -> tuple[int, bool]:
    if rng.random() < epsilon:
        return sample_random_legal_action(action_mask, rng), True
    return int(masked_q_values.argmax(dim=-1).item()), False


def action_name_from_index(action_index: int, action_order: Sequence[str]) -> str:
    return str(action_order[action_index])


def infer_decision_phase(response: dict[str, Any] | None) -> str:
    if not response:
        return "playing"

    observation = response.get("observation") or {}
    phase = observation.get("decision_phase")
    if isinstance(phase, str) and phase:
        return phase

    public_state = (response.get("info") or {}).get("public_state") or {}
    phase = public_state.get("decision_phase")
    if isinstance(phase, str) and phase:
        return phase
    return "playing"


def resolve_epsilon_value(
    epsilon: Any,
    *,
    decision_phase: str,
    evaluation: bool,
) -> float:
    if isinstance(epsilon, (int, float)):
        return float(epsilon)
    if isinstance(epsilon, EpsilonScheduleConfig):
        return float(epsilon.evaluation_epsilon if evaluation else epsilon.start)
    if isinstance(epsilon, DualEpsilonConfig):
        schedule = epsilon.betting if decision_phase == "betting" else epsilon.playing
        return float(schedule.evaluation_epsilon if evaluation else schedule.start)
    if evaluation and hasattr(epsilon, "evaluation_value"):
        return float(epsilon.evaluation_value(decision_phase))
    if not evaluation and hasattr(epsilon, "value"):
        return float(epsilon.value(decision_phase))
    raise TypeError("Unsupported epsilon specification")
