from __future__ import annotations

from typing import Any, Sequence

import torch

from enviroment_bj.core import hand_value

from .constants import CARD_RANKS, CARD_TO_INDEX, HAND_SETTLEMENT_VALUES, PUBLIC_ACTION_TO_INDEX


CARD_ONE_HOT_TABLE = torch.eye(len(CARD_RANKS), dtype=torch.float32)
ZERO_CARD_ONE_HOT = torch.zeros(len(CARD_RANKS), dtype=torch.float32)
EMPTY_CARD_TOTAL_FEATURES = torch.zeros(3, dtype=torch.float32)
HAND_SETTLEMENT_TO_INDEX = {name: index for index, name in enumerate(HAND_SETTLEMENT_VALUES)}
ACTION_TOKEN_ONE_HOT_TABLE = torch.eye(len(PUBLIC_ACTION_TO_INDEX), dtype=torch.float32)
UNKNOWN_ACTION_TOKEN_INDEX = PUBLIC_ACTION_TO_INDEX["unk"]
_ONE_HOT_TABLES: dict[int, torch.Tensor] = {}
_ZERO_ONE_HOT_VECTORS: dict[int, torch.Tensor] = {}
_EMPTY_RECENT_CARDS_TENSORS: dict[int, torch.Tensor] = {}
_EMPTY_ACTION_TOKEN_TENSORS: dict[int, torch.Tensor] = {}


def _one_hot_table(size: int) -> torch.Tensor:
    table = _ONE_HOT_TABLES.get(size)
    if table is None:
        table = torch.eye(size, dtype=torch.float32)
        _ONE_HOT_TABLES[size] = table
        _ZERO_ONE_HOT_VECTORS[size] = torch.zeros(size, dtype=torch.float32)
    return table


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
    if index is None or not 0 <= index < size:
        _one_hot_table(size)
        return _ZERO_ONE_HOT_VECTORS[size]
    return _one_hot_table(size)[index]


def rank_one_hot(rank: str | None) -> torch.Tensor:
    index = CARD_TO_INDEX.get(rank)
    if index is None:
        return ZERO_CARD_ONE_HOT
    return CARD_ONE_HOT_TABLE[index]


def cards_to_padded_rank_tensor_and_mask(cards: Sequence[str] | None, max_cards: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.zeros((max_cards, len(CARD_RANKS)), dtype=torch.float32)
    mask = torch.zeros(max_cards, dtype=torch.float32)
    if not cards:
        return matrix, mask

    clipped_cards = cards[-max_cards:] if len(cards) > max_cards else cards
    card_count = len(clipped_cards)
    if card_count <= 0:
        return matrix, mask

    matrix[:card_count] = CARD_ONE_HOT_TABLE[[CARD_TO_INDEX[card] for card in clipped_cards]]
    mask[:card_count] = 1.0
    return matrix, mask


def cards_to_padded_rank_tensor(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    matrix, _ = cards_to_padded_rank_tensor_and_mask(cards, max_cards)
    return matrix


def cards_to_presence_mask(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    _, mask = cards_to_padded_rank_tensor_and_mask(cards, max_cards)
    return mask


def card_total_features(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    if not cards:
        return EMPTY_CARD_TOTAL_FEATURES
    total, is_soft = hand_value(cards)
    return torch.tensor(
        [normalize_scalar(total, 21.0), safe_bool(is_soft), normalize_scalar(len(cards), max_cards)],
        dtype=torch.float32,
    )


def normalized_count_vector(counts: Sequence[float], *, normalize: bool) -> torch.Tensor:
    total = float(sum(counts)) if normalize else 0.0
    values = torch.tensor(counts, dtype=torch.float32)
    if not normalize:
        return values
    if total <= 0:
        return values
    return values / total


def settlement_count_vector(settlements: Sequence[str] | None) -> torch.Tensor:
    values = torch.zeros(len(HAND_SETTLEMENT_VALUES), dtype=torch.float32)
    if settlements is None:
        return values
    for settlement in settlements:
        index = HAND_SETTLEMENT_TO_INDEX.get(settlement)
        if index is not None:
            values[index] += 1.0
    return values


def action_tokens_to_tensor(tokens: Sequence[str] | None, max_actions: int) -> torch.Tensor:
    vector = _EMPTY_ACTION_TOKEN_TENSORS.get(max_actions)
    if vector is None:
        vector = torch.zeros(max_actions * (len(PUBLIC_ACTION_TO_INDEX) + 1), dtype=torch.float32)
        _EMPTY_ACTION_TOKEN_TENSORS[max_actions] = vector
    if not tokens:
        return vector

    clipped_tokens = list(tokens)[-max_actions:]
    encoded = torch.zeros_like(vector)
    matrix = encoded[: max_actions * len(PUBLIC_ACTION_TO_INDEX)].view(max_actions, len(PUBLIC_ACTION_TO_INDEX))
    mask = encoded[max_actions * len(PUBLIC_ACTION_TO_INDEX) :]
    start_index = max_actions - len(clipped_tokens)
    token_indices = [PUBLIC_ACTION_TO_INDEX.get(token, UNKNOWN_ACTION_TOKEN_INDEX) for token in clipped_tokens]
    matrix[start_index : start_index + len(token_indices)] = ACTION_TOKEN_ONE_HOT_TABLE[token_indices]
    mask[start_index : start_index + len(token_indices)] = 1.0
    return encoded


def recent_cards_to_tensor(cards: Sequence[str] | None, max_cards: int) -> torch.Tensor:
    vector = _EMPTY_RECENT_CARDS_TENSORS.get(max_cards)
    if vector is None:
        vector = torch.zeros(max_cards * (len(CARD_RANKS) + 1), dtype=torch.float32)
        _EMPTY_RECENT_CARDS_TENSORS[max_cards] = vector
    if not cards:
        return vector

    encoded = torch.zeros_like(vector)
    matrix = encoded[: max_cards * len(CARD_RANKS)].view(max_cards, len(CARD_RANKS))
    mask = encoded[max_cards * len(CARD_RANKS) :]
    padded_matrix, padded_mask = cards_to_padded_rank_tensor_and_mask(cards, max_cards)
    matrix.copy_(padded_matrix)
    mask.copy_(padded_mask)
    return encoded


def ensure_tensor_dim(tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if tensor.numel() != expected_dim:
        raise ValueError(f"Tensor has {tensor.numel()} elements but expected {expected_dim}")
    return tensor
