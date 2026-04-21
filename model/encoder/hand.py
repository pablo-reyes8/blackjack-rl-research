from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .base import BaseFeatureEncoder
from .config import EncoderConfig
from .constants import AVAILABLE_BET_MULTIPLIERS
from .utils import (
    card_total_features,
    cards_to_padded_rank_tensor,
    cards_to_presence_mask,
    normalize_scalar,
    rank_one_hot,
    safe_bool,
    safe_float,
)


class HandFeatureEncoder(BaseFeatureEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.profile == "minimal_basic_strategy":
            self.output_dim = 16
        else:
            self.output_dim = (
                config.max_current_hand_cards * 13
                + config.max_current_hand_cards
                + 3
                + 13
                + 1
            )

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        dealer_rank = observation.get("dealer_upcard")
        dealer_one_hot = rank_one_hot(dealer_rank)
        dealer_value = torch.tensor(
            [normalize_scalar(observation.get("dealer_upcard_value"), 11.0)],
            dtype=torch.float32,
        )

        if self.config.profile == "minimal_basic_strategy":
            values = torch.tensor(
                [
                    normalize_scalar(observation.get("current_hand_total"), 21.0),
                    safe_bool(observation.get("current_hand_is_soft")),
                ],
                dtype=torch.float32,
            )
            return torch.cat([values, dealer_one_hot, dealer_value], dim=0)

        current_hand_cards = observation.get("current_hand_cards")
        if current_hand_cards is None and observation.get("current_hand_total") is not None:
            total_features = torch.tensor(
                [
                    normalize_scalar(observation.get("current_hand_total"), 21.0),
                    safe_bool(observation.get("current_hand_is_soft")),
                    0.0,
                ],
                dtype=torch.float32,
            )
        else:
            total_features = card_total_features(current_hand_cards, self.config.max_current_hand_cards)

        card_matrix = cards_to_padded_rank_tensor(current_hand_cards, self.config.max_current_hand_cards).flatten()
        card_mask = cards_to_presence_mask(current_hand_cards, self.config.max_current_hand_cards)
        return torch.cat([card_matrix, card_mask, total_features, dealer_one_hot, dealer_value], dim=0)


class OtherHandsEncoder(BaseFeatureEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.per_hand_dim = config.max_cards_per_hand * 13 + config.max_cards_per_hand + 7
        self.output_dim = config.max_other_hands * self.per_hand_dim

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        hands = observation.get("other_player_hands_visible") or []
        base_bet = safe_float(table_rules.get("base_bet"), default=1.0) or 1.0
        encoded_hands: list[torch.Tensor] = []

        for hand in list(hands)[: self.config.max_other_hands]:
            cards = hand.get("cards")
            card_matrix = cards_to_padded_rank_tensor(cards, self.config.max_cards_per_hand).flatten()
            card_mask = cards_to_presence_mask(cards, self.config.max_cards_per_hand)
            totals = card_total_features(cards, self.config.max_cards_per_hand)
            flags = torch.tensor(
                [
                    normalize_scalar(hand.get("bet"), base_bet),
                    safe_bool(hand.get("from_split")),
                    safe_bool(hand.get("split_aces")),
                    safe_bool(hand.get("closed")),
                ],
                dtype=torch.float32,
            )
            encoded_hands.append(torch.cat([card_matrix, card_mask, totals, flags], dim=0))

        while len(encoded_hands) < self.config.max_other_hands:
            encoded_hands.append(torch.zeros(self.per_hand_dim, dtype=torch.float32))

        return torch.cat(encoded_hands, dim=0)


class HandContextEncoder(BaseFeatureEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dim = 5

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        context = observation.get("hand_context") or {}
        return torch.tensor(
            [
                normalize_scalar(context.get("current_hand_index"), max(1, self.config.max_other_hands)),
                normalize_scalar(context.get("n_player_hands"), self.config.max_other_hands + 1),
                safe_bool(context.get("from_split")),
                safe_bool(context.get("split_aces")),
                safe_bool(context.get("first_decision_on_hand")),
            ],
            dtype=torch.float32,
        )


class InsuranceContextEncoder(BaseFeatureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 2

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        context = observation.get("insurance_context") or {}
        base_bet = safe_float(table_rules.get("base_bet"), default=1.0) or 1.0
        return torch.tensor(
            [
                safe_bool(context.get("insurance_offer_active")),
                normalize_scalar(context.get("insurance_bet"), base_bet),
            ],
            dtype=torch.float32,
        )


class BettingContextEncoder(BaseFeatureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 2 + 1 + len(AVAILABLE_BET_MULTIPLIERS) + 1 + 1

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        decision_phase = observation.get("decision_phase")
        betting_context = observation.get("betting_context") or {}
        available_multipliers = set(observation.get("available_bet_multipliers") or betting_context.get("available_bet_multipliers") or [])
        base_bet = safe_float(table_rules.get("base_bet"), default=1.0) or 1.0
        current_bet = betting_context.get("current_bet", observation.get("current_bet"))

        return torch.tensor(
            [
                safe_bool(decision_phase == "betting"),
                safe_bool(decision_phase == "playing"),
                safe_bool(betting_context.get("can_place_bet")),
                *[safe_bool(multiplier in available_multipliers) for multiplier in AVAILABLE_BET_MULTIPLIERS],
                safe_bool(current_bet is not None),
                normalize_scalar(current_bet, base_bet),
            ],
            dtype=torch.float32,
        )


class BetEncoder(BaseFeatureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 1

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        base_bet = safe_float(table_rules.get("base_bet"), default=1.0) or 1.0
        return torch.tensor([normalize_scalar(observation.get("current_bet"), base_bet)], dtype=torch.float32)
