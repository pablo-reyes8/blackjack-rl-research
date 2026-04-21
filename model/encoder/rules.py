from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import BaseFeatureEncoder
from .constants import DOUBLE_ALLOWED_ON_VALUES, SPLIT_RULE_VALUES
from .utils import normalize_scalar, one_hot, safe_bool


class RuleEncoder(BaseFeatureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 22

    def encode(self, observation: Mapping[str, Any], table_rules: Mapping[str, Any]) -> torch.Tensor:
        bool_values = torch.tensor(
            [
                safe_bool(table_rules.get("dealer_hits_soft_17")),
                safe_bool(table_rules.get("dealer_peeks_for_blackjack")),
                safe_bool(table_rules.get("double_after_split_allowed")),
                safe_bool(table_rules.get("double_split_aces_allowed")),
                safe_bool(table_rules.get("resplit_aces_allowed")),
                safe_bool(table_rules.get("hit_split_aces_allowed")),
                safe_bool(table_rules.get("surrender_allowed")),
                safe_bool(table_rules.get("insurance_allowed")),
                safe_bool(table_rules.get("six_card_charlie_enabled")),
                safe_bool(table_rules.get("use_cut_card")),
            ],
            dtype=torch.float32,
        )

        double_one_hot = one_hot(
            DOUBLE_ALLOWED_ON_VALUES.index(table_rules["double_allowed_on"])
            if table_rules.get("double_allowed_on") in DOUBLE_ALLOWED_ON_VALUES
            else None,
            len(DOUBLE_ALLOWED_ON_VALUES),
        )
        split_one_hot = one_hot(
            SPLIT_RULE_VALUES.index(table_rules["split_rule"])
            if table_rules.get("split_rule") in SPLIT_RULE_VALUES
            else None,
            len(SPLIT_RULE_VALUES),
        )

        continuous_values = torch.tensor(
            [
                normalize_scalar(table_rules.get("blackjack_payout"), 2.0),
                normalize_scalar(table_rules.get("base_bet"), 10.0),
                normalize_scalar(table_rules.get("max_hands_after_split"), 8.0),
                normalize_scalar(table_rules.get("max_split_depth_per_hand"), 4.0),
                normalize_scalar(len(table_rules.get("bet_multipliers") or []), 4.0),
                normalize_scalar(table_rules.get("n_decks"), 8.0),
                normalize_scalar(table_rules.get("shoe_penetration"), 1.0),
            ],
            dtype=torch.float32,
        )
        return torch.cat([bool_values, double_one_hot, split_one_hot, continuous_values], dim=0)
