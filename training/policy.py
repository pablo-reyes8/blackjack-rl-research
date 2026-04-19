from __future__ import annotations

import random
from typing import Any, Sequence

import torch


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
