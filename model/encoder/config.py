from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EncoderConfig:
    profile: str = "table_realistic_default"
    encode_rules: bool = True
    encode_other_hands: bool = True
    encode_temporal: bool = True
    encode_observed_history: bool = True
    encode_discard_summary: bool = True
    encode_recent_actions: bool = False
    encode_exact_shoe: bool = False
    encode_action_mask_features: bool = False
    card_encoding: str = "one_hot_rank"
    history_encoding: str = "rank_counts"
    normalize_counts: bool = True
    use_visible_table_rules_only: bool = True
    max_current_hand_cards: int = 12
    max_cards_per_hand: int = 12
    max_other_hands: int = 4
    max_recent_actions: int = 5
    max_recent_cards: int = 20
    max_recent_discard_cards: int = 10

    def __post_init__(self) -> None:
        if self.profile not in {
            "minimal_basic_strategy",
            "table_realistic_default",
            "table_realistic_unknown_progress",
            "fully_observable_sim",
        }:
            raise ValueError(
                "profile must be 'minimal_basic_strategy', 'table_realistic_default', 'table_realistic_unknown_progress', or 'fully_observable_sim'"
            )
        if self.card_encoding != "one_hot_rank":
            raise ValueError("Only 'one_hot_rank' card encoding is currently supported")
        if self.history_encoding not in {"rank_counts", "low_neutral_high", "recent_cards_sequence"}:
            raise ValueError(
                "history_encoding must be 'rank_counts', 'low_neutral_high', or 'recent_cards_sequence'"
            )
        for field_name in (
            "max_current_hand_cards",
            "max_cards_per_hand",
            "max_other_hands",
            "max_recent_actions",
            "max_recent_cards",
            "max_recent_discard_cards",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    @classmethod
    def for_profile(cls, profile: str) -> EncoderConfig:
        if profile == "minimal_basic_strategy":
            return cls(
                profile=profile,
                encode_rules=True,
                encode_other_hands=False,
                encode_temporal=False,
                encode_observed_history=False,
                encode_discard_summary=False,
                encode_recent_actions=False,
                encode_exact_shoe=False,
                encode_action_mask_features=False,
                history_encoding="rank_counts",
                use_visible_table_rules_only=True,
            )

        if profile == "table_realistic_default":
            return cls(
                profile=profile,
                encode_rules=True,
                encode_other_hands=True,
                encode_temporal=True,
                encode_observed_history=True,
                encode_discard_summary=True,
                encode_recent_actions=False,
                encode_exact_shoe=False,
                encode_action_mask_features=False,
                history_encoding="rank_counts",
                use_visible_table_rules_only=True,
                max_recent_actions=8,
            )

        if profile == "table_realistic_unknown_progress":
            return cls(
                profile=profile,
                encode_rules=True,
                encode_other_hands=True,
                encode_temporal=True,
                encode_observed_history=True,
                encode_discard_summary=True,
                encode_recent_actions=False,
                encode_exact_shoe=False,
                encode_action_mask_features=False,
                history_encoding="rank_counts",
                use_visible_table_rules_only=True,
                max_recent_actions=8,
            )

        if profile == "fully_observable_sim":
            return cls(
                profile=profile,
                encode_rules=True,
                encode_other_hands=True,
                encode_temporal=True,
                encode_observed_history=True,
                encode_discard_summary=True,
                encode_recent_actions=True,
                encode_exact_shoe=True,
                encode_action_mask_features=False,
                history_encoding="rank_counts",
                use_visible_table_rules_only=False,
                max_recent_actions=10,
            )

        raise ValueError(
            "Unsupported encoder profile. Use 'minimal_basic_strategy', 'table_realistic_default', 'table_realistic_unknown_progress', or 'fully_observable_sim'"
        )
