from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .base import BaseFeatureEncoder
from .config import EncoderConfig
from .constants import CARD_RANKS, HAND_SETTLEMENT_VALUES, PROGRESS_BUCKET_VALUES, PUBLIC_ACTION_TOKENS
from .utils import (
    action_tokens_to_tensor,
    normalized_count_vector,
    one_hot,
    recent_cards_to_tensor,
    settlement_count_vector,
    normalize_scalar,
    safe_bool,
    safe_float,
)


PROGRESS_BUCKET_TO_INDEX = {value: index for index, value in enumerate(PROGRESS_BUCKET_VALUES)}


class ObservedCardsHistoryEncoder(BaseFeatureEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.history_encoding == "rank_counts":
            self.output_dim = len(CARD_RANKS) + 1
        elif config.history_encoding == "low_neutral_high":
            self.output_dim = 4
        else:
            self.output_dim = config.max_recent_cards * len(CARD_RANKS) + config.max_recent_cards + 1

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        history = observation.get("observed_cards_history")
        if self.config.history_encoding == "rank_counts":
            counts = [safe_float((history or {}).get(rank)) for rank in CARD_RANKS]
            vector = normalized_count_vector(counts, normalize=self.config.normalize_counts)
            total = torch.tensor([normalize_scalar(sum(counts), 52.0 * 8.0)], dtype=torch.float32)
            return torch.cat([vector, total], dim=0)

        if self.config.history_encoding == "low_neutral_high":
            counts = [safe_float((history or {}).get(name)) for name in ("low", "neutral", "high")]
            vector = normalized_count_vector(counts, normalize=self.config.normalize_counts)
            total = torch.tensor([normalize_scalar(sum(counts), 52.0 * 8.0)], dtype=torch.float32)
            return torch.cat([vector, total], dim=0)

        cards = (history or {}).get("cards", [])
        count_tensor = recent_cards_to_tensor(cards, self.config.max_recent_cards)
        total = torch.tensor([normalize_scalar(len(cards), float(self.config.max_recent_cards))], dtype=torch.float32)
        return torch.cat([count_tensor, total], dim=0)


class DiscardSummaryEncoder(BaseFeatureEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dim = 1 + 3 + config.max_recent_discard_cards * len(CARD_RANKS) + config.max_recent_discard_cards

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        summary = observation.get("discard_summary") or {}
        observed_count = torch.tensor(
            [normalize_scalar(summary.get("observed_cards_count"), 52.0 * 8.0)],
            dtype=torch.float32,
        )
        by_group = summary.get("by_group") or {}
        group_values = [safe_float(by_group.get(name)) for name in ("low", "neutral", "high")]
        group_tensor = normalized_count_vector(group_values, normalize=self.config.normalize_counts)
        recent_cards = recent_cards_to_tensor(summary.get("recent_cards", []), self.config.max_recent_discard_cards)
        return torch.cat([observed_count, group_tensor, recent_cards], dim=0)


class TemporalFeatureEncoder(BaseFeatureEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.base_dim = 7 + 4 + (5 + len(HAND_SETTLEMENT_VALUES)) + 7
        self.recent_actions_dim = (
            config.max_recent_actions * len(PUBLIC_ACTION_TOKENS) + config.max_recent_actions
            if config.encode_recent_actions
            else 0
        )
        self.output_dim = self.base_dim + self.recent_actions_dim

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        temporal = observation.get("temporal_context") or {}
        numeric = torch.tensor(
            [
                normalize_scalar(temporal.get("shuffle_count"), 100.0),
                normalize_scalar(temporal.get("rounds_played_total"), 1000.0),
                normalize_scalar(temporal.get("rounds_since_shuffle"), 100.0),
                normalize_scalar(temporal.get("player_hands_seen_since_shuffle"), 200.0),
                normalize_scalar(temporal.get("dealer_hands_seen_since_shuffle"), 200.0),
                normalize_scalar(temporal.get("player_hands_seen_total"), 1000.0),
                normalize_scalar(temporal.get("dealer_hands_seen_total"), 1000.0),
            ],
            dtype=torch.float32,
        )

        estimated_progress = temporal.get("estimated_shoe_progress") or {}
        bucket_index = PROGRESS_BUCKET_TO_INDEX.get(estimated_progress.get("bucket"))
        progress_tensor = torch.cat(
            [
                torch.tensor([safe_float(estimated_progress.get("fraction_used"))], dtype=torch.float32),
                one_hot(bucket_index, len(PROGRESS_BUCKET_VALUES)),
            ],
            dim=0,
        )

        outcome = temporal.get("last_round_outcome") or {}
        outcome_tensor = torch.cat(
            [
                torch.tensor(
                    [
                        safe_bool(bool(outcome)),
                        normalize_scalar(outcome.get("reward"), 10.0),
                        normalize_scalar(outcome.get("insurance_reward"), 2.0),
                        normalize_scalar(outcome.get("dealer_total"), 21.0),
                        safe_bool(outcome.get("dealer_has_blackjack")),
                    ],
                    dtype=torch.float32,
                ),
                settlement_count_vector(outcome.get("hand_settlements")),
            ],
            dim=0,
        )

        observed_shuffle_tensor = torch.tensor(
            [
                safe_bool(temporal.get("observed_shuffle_reset")),
                safe_bool(temporal.get("has_observed_shuffle_reference")),
                normalize_scalar(temporal.get("hands_since_observed_shuffle"), 100.0),
                normalize_scalar(temporal.get("observed_cards_since_shuffle"), 52.0 * 8.0),
                safe_float(temporal.get("low_fraction_since_shuffle")),
                safe_float(temporal.get("high_fraction_since_shuffle")),
                safe_float(temporal.get("high_minus_low_balance")),
            ],
            dtype=torch.float32,
        )

        tensors = [numeric, progress_tensor, outcome_tensor, observed_shuffle_tensor]
        if self.config.encode_recent_actions:
            recent_actions = temporal.get("recent_actions") or []
            tokens = [action.get("token", "unk") for action in recent_actions]
            tensors.append(action_tokens_to_tensor(tokens, self.config.max_recent_actions))

        return torch.cat(tensors, dim=0)


class ExactShoeEncoder(BaseFeatureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = len(CARD_RANKS)

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        composition = observation.get("exact_shoe_composition") or {}
        counts = [safe_float(composition.get(rank)) for rank in CARD_RANKS]
        return normalized_count_vector(counts, normalize=True)
