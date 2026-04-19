from __future__ import annotations

from typing import Any, Sequence

import torch

from enviroment_bj.core import hand_value

from .constants import CARD_RANKS, CARD_TO_INDEX, HAND_SETTLEMENT_VALUES, PUBLIC_ACTION_TO_INDEX


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def safe_bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def normalize_scalar(value: Any, scale: float, default: float = 0.0) -> float:
    if value is None:
        return default
    if scale == 0:
        return float(value)
    return float(value) / float(scale)


def one_hot(index: int | None, size: int) -> torch.Tensor:
    vector = torch.zeros(size, dtype=torch.float32)
    if index is not None and 0 <= index < size:
        vector[index] = 1.0
    return vector


def rank_one_hot(rank: str | None) -> torch.Tensor:
    return one_hot(CARD_TO_INDEX.get(rank), len(CARD_RANKS))


def cards_to_padded_rank_tensor(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    matrix = torch.zeros((max_cards, len(CARD_RANKS)), dtype=torch.float32)
    if cards is None:
        return matrix
    for index, card in enumerate(list(cards)[:max_cards]):
        matrix[index, CARD_TO_INDEX[card]] = 1.0
    return matrix


def cards_to_presence_mask(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    mask = torch.zeros(max_cards, dtype=torch.float32)
    if cards is None:
        return mask
    for index in range(min(len(cards), max_cards)):
        mask[index] = 1.0
    return mask


def card_total_features(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    if not cards:
        return torch.zeros(3, dtype=torch.float32)
    total, is_soft = hand_value(cards)
    return torch.tensor(
        [normalize_scalar(total, 21.0), safe_bool(is_soft), normalize_scalar(len(cards), max_cards)],
        dtype=torch.float32,
    )


def normalized_count_vector(counts: Sequence[float], *, normalize: bool) -> torch.Tensor:
    values = torch.tensor(list(counts), dtype=torch.float32)
    if not normalize:
        return values
    total = float(values.sum().item())
    if total <= 0:
        return values
    return values / total


def settlement_count_vector(settlements: Sequence[str] | None) -> torch.Tensor:
    values = torch.zeros(len(HAND_SETTLEMENT_VALUES), dtype=torch.float32)
    if settlements is None:
        return values
    for settlement in settlements:
        if settlement in HAND_SETTLEMENT_VALUES:
            values[HAND_SETTLEMENT_VALUES.index(settlement)] += 1.0
    return values


def action_tokens_to_tensor(tokens: Sequence[str] | None, max_actions: int) -> torch.Tensor:
    matrix = torch.zeros((max_actions, len(PUBLIC_ACTION_TO_INDEX)), dtype=torch.float32)
    mask = torch.zeros(max_actions, dtype=torch.float32)
    if not tokens:
        return torch.cat([matrix.flatten(), mask], dim=0)

    clipped_tokens = list(tokens)[-max_actions:]
    start_index = max_actions - len(clipped_tokens)
    for offset, token in enumerate(clipped_tokens):
        row = start_index + offset
        matrix[row, PUBLIC_ACTION_TO_INDEX.get(token, PUBLIC_ACTION_TO_INDEX["unk"])] = 1.0
        mask[row] = 1.0
    return torch.cat([matrix.flatten(), mask], dim=0)


def recent_cards_to_tensor(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    matrix = cards_to_padded_rank_tensor(list(cards)[-max_cards:] if cards else None, max_cards)
    mask = cards_to_presence_mask(list(cards)[-max_cards:] if cards else None, max_cards)
    return torch.cat([matrix.flatten(), mask], dim=0)


def ensure_tensor_dim(tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if tensor.numel() != expected_dim:
        raise ValueError(f"Tensor has {tensor.numel()} elements but expected {expected_dim}")
    return tensor
