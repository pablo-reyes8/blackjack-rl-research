from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StartStateConfig:
    mode: str = "fresh_shoe"
    min_burned_rounds: int = 0
    max_burned_rounds: int = 0
    clear_visible_histories_after_burn: bool = True
    hide_reshuffle_progress_from_observation: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"fresh_shoe", "unknown_progress"}:
            raise ValueError("mode must be 'fresh_shoe' or 'unknown_progress'")
        if self.min_burned_rounds < 0:
            raise ValueError("min_burned_rounds must be non-negative")
        if self.max_burned_rounds < self.min_burned_rounds:
            raise ValueError("max_burned_rounds must be greater than or equal to min_burned_rounds")


@dataclass(slots=True)
class ObservationConfig:
    profile: str = "table_realistic_default"
    obs_include_table_rules: bool = True
    obs_include_visible_rules_only: bool = True
    obs_include_hidden_rules: bool = False
    obs_current_hand_mode: str = "table_raw"
    obs_include_other_player_hands: bool = True
    obs_include_current_bet: bool = True
    obs_include_hand_context: bool = True
    obs_include_insurance_context: bool = True
    obs_include_temporal_context: bool = True
    obs_include_hands_since_shuffle: bool = True
    obs_include_estimated_shoe_progress: bool = True
    obs_include_last_hand_outcome: bool = False
    obs_include_recent_actions: bool = False
    obs_recent_actions_window: int = 5
    obs_include_observed_cards_history: bool = True
    obs_observed_cards_mode: str = "rank_counts"
    obs_recent_cards_window: int = 20
    obs_reset_history_on_shuffle: bool = True
    obs_include_exact_shoe_composition: bool = False
    obs_include_discard_summary: bool = True
    obs_include_n_decks: bool = False
    obs_include_shoe_penetration_rule: bool = False

    def __post_init__(self) -> None:
        if self.obs_current_hand_mode not in {"basic_strategy", "table_raw"}:
            raise ValueError("obs_current_hand_mode must be 'basic_strategy' or 'table_raw'")
        if self.obs_observed_cards_mode not in {
            "rank_counts",
            "low_neutral_high",
            "recent_cards_sequence",
        }:
            raise ValueError(
                "obs_observed_cards_mode must be 'rank_counts', 'low_neutral_high', or 'recent_cards_sequence'"
            )
        if self.obs_recent_actions_window <= 0:
            raise ValueError("obs_recent_actions_window must be positive")
        if self.obs_recent_cards_window <= 0:
            raise ValueError("obs_recent_cards_window must be positive")

    @classmethod
    def for_profile(cls, profile: str) -> ObservationConfig:
        if profile == "table_realistic_default":
            return cls(profile=profile)

        if profile == "table_realistic_unknown_progress":
            return cls(
                profile=profile,
                obs_include_table_rules=True,
                obs_include_visible_rules_only=True,
                obs_include_hidden_rules=False,
                obs_current_hand_mode="table_raw",
                obs_include_other_player_hands=True,
                obs_include_current_bet=True,
                obs_include_hand_context=True,
                obs_include_insurance_context=True,
                obs_include_temporal_context=True,
                obs_include_hands_since_shuffle=False,
                obs_include_estimated_shoe_progress=False,
                obs_include_last_hand_outcome=False,
                obs_include_recent_actions=False,
                obs_include_observed_cards_history=True,
                obs_observed_cards_mode="rank_counts",
                obs_recent_cards_window=20,
                obs_reset_history_on_shuffle=True,
                obs_include_exact_shoe_composition=False,
                obs_include_discard_summary=True,
                obs_include_n_decks=False,
                obs_include_shoe_penetration_rule=False,
            )

        if profile == "fully_observable_sim":
            return cls(
                profile=profile,
                obs_include_table_rules=True,
                obs_include_visible_rules_only=False,
                obs_include_hidden_rules=True,
                obs_current_hand_mode="table_raw",
                obs_include_other_player_hands=True,
                obs_include_current_bet=True,
                obs_include_hand_context=True,
                obs_include_insurance_context=True,
                obs_include_temporal_context=True,
                obs_include_hands_since_shuffle=True,
                obs_include_estimated_shoe_progress=True,
                obs_include_last_hand_outcome=True,
                obs_include_recent_actions=True,
                obs_recent_actions_window=10,
                obs_include_observed_cards_history=True,
                obs_observed_cards_mode="rank_counts",
                obs_recent_cards_window=20,
                obs_reset_history_on_shuffle=True,
                obs_include_exact_shoe_composition=True,
                obs_include_discard_summary=True,
                obs_include_n_decks=True,
                obs_include_shoe_penetration_rule=True,
            )

        if profile == "minimal_basic_strategy":
            return cls(
                profile=profile,
                obs_include_table_rules=True,
                obs_include_visible_rules_only=True,
                obs_include_hidden_rules=False,
                obs_current_hand_mode="basic_strategy",
                obs_include_other_player_hands=False,
                obs_include_current_bet=False,
                obs_include_hand_context=True,
                obs_include_insurance_context=True,
                obs_include_temporal_context=False,
                obs_include_hands_since_shuffle=False,
                obs_include_estimated_shoe_progress=False,
                obs_include_last_hand_outcome=False,
                obs_include_recent_actions=False,
                obs_include_observed_cards_history=False,
                obs_include_discard_summary=False,
                obs_include_n_decks=False,
                obs_include_shoe_penetration_rule=False,
            )

        raise ValueError(
            "Unsupported observation profile. Use 'table_realistic_default', 'table_realistic_unknown_progress', 'fully_observable_sim', or 'minimal_basic_strategy'"
        )


def _default_observation_config() -> ObservationConfig:
    return ObservationConfig.for_profile("table_realistic_default")


@dataclass(slots=True)
class BlackjackConfig:
    n_decks: int = 6
    shoe_penetration: float = 0.8
    dealer_hits_soft_17: bool = False
    blackjack_payout: float = 1.5
    dealer_peeks_for_blackjack: bool = True
    double_allowed_on: str = "any_two_cards"
    double_after_split_allowed: bool = True
    split_rule: str = "same_value"
    max_hands_after_split: int = 4
    resplit_aces_allowed: bool = True
    hit_split_aces_allowed: bool = False
    surrender_allowed: bool = True
    insurance_allowed: bool = True
    base_bet: float = 1.0
    strict_shoe_validation: bool = False
    observation: ObservationConfig = field(default_factory=_default_observation_config)
    observation_mode: str | None = None
    expose_shoe_composition: bool = False

    def __post_init__(self) -> None:
        if self.n_decks <= 0:
            raise ValueError("n_decks must be positive")
        if not 0 < self.shoe_penetration <= 1:
            raise ValueError("shoe_penetration must be in (0, 1]")
        if self.blackjack_payout <= 0:
            raise ValueError("blackjack_payout must be positive")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive")
        if self.max_hands_after_split < 2:
            raise ValueError("max_hands_after_split must be at least 2")
        if self.split_rule not in {"same_rank", "same_value"}:
            raise ValueError("split_rule must be 'same_rank' or 'same_value'")
        if self.double_allowed_on not in {
            "any_two_cards",
            "hard_9_10_11",
            "hard_10_11",
        }:
            raise ValueError(
                "double_allowed_on must be one of: any_two_cards, hard_9_10_11, hard_10_11"
            )
        if not isinstance(self.observation, ObservationConfig):
            raise TypeError("observation must be an ObservationConfig instance")

        if self.observation_mode is not None:
            self.observation.obs_current_hand_mode = self.observation_mode
            self.observation.__post_init__()

        if self.expose_shoe_composition:
            self.observation.obs_include_exact_shoe_composition = True
