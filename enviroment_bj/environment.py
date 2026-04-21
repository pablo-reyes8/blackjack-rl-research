from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict
import random
from typing import Any

from .config import BlackjackConfig, ObservationConfig, StartStateConfig
from .core import (
    ACTION_ORDER,
    BET_ACTION_MULTIPLIERS,
    CARD_RANKS,
    HandState,
    Shoe,
    coerce_action_name,
    hand_value,
    is_natural_blackjack,
    rank_value,
    split_value,
    validate_shoe_cards,
)


VISIBLE_TABLE_RULES = {
    "dealer_hits_soft_17",
    "blackjack_payout",
    "double_allowed_on",
    "double_after_split_allowed",
    "double_split_aces_allowed",
    "split_rule",
    "max_hands_after_split",
    "max_split_depth_per_hand",
    "resplit_aces_allowed",
    "hit_split_aces_allowed",
    "surrender_allowed",
    "insurance_allowed",
    "six_card_charlie_enabled",
    "base_bet",
    "bet_multipliers",
    "use_cut_card",
}
HIDDEN_TABLE_RULES = {"dealer_peeks_for_blackjack", "n_decks", "shoe_penetration", "reward_mode"}
LOW_RANKS = {"2", "3", "4", "5", "6"}
NEUTRAL_RANKS = {"7", "8", "9"}
HIGH_RANKS = {"10", "J", "Q", "K", "A"}


class BlackjackEnvironment:
    def __init__(
        self,
        config: BlackjackConfig | None = None,
        seed: int | None = None,
        start_state: StartStateConfig | None = None,
    ) -> None:
        self.config = config or BlackjackConfig()
        self.start_state = start_state or StartStateConfig()
        self._rng = random.Random(seed)
        self.shoe = Shoe(
            n_decks=self.config.n_decks,
            penetration=self.config.shoe_penetration,
            rng=self._rng,
            use_cut_card=self.config.use_cut_card,
        )
        self._build_bet_action_cache()
        self._build_table_rule_caches()
        self.round_index = 0
        self._reshuffle_pending = False
        self._start_state_prepared = False
        self._manual_shoe_loaded = False
        self.enable_transition_recording = True
        self.compact_response_mode = False
        self._initialize_temporal_state()
        self._reset_round_state()

    def set_runtime_options(
        self,
        *,
        enable_transition_recording: bool | None = None,
        compact_response_mode: bool | None = None,
    ) -> None:
        if enable_transition_recording is not None:
            self.enable_transition_recording = bool(enable_transition_recording)
        if compact_response_mode is not None:
            self.compact_response_mode = bool(compact_response_mode)

    def _initialize_temporal_state(self) -> None:
        self.shuffle_count = 0
        self.total_rounds_played = 0
        self.total_player_hands_seen = 0
        self.total_dealer_hands_seen = 0
        self.rounds_since_shuffle = 0
        self.player_hands_seen_since_shuffle = 0
        self.dealer_hands_seen_since_shuffle = 0
        self.observed_cards_history: list[dict[str, Any]] = []
        self.public_action_history: list[dict[str, Any]] = []
        self.last_round_outcome: dict[str, Any] | None = None
        self.last_shuffle_reason: str | None = None
        self.hidden_burned_rounds = 0
        self.hidden_burned_cards = 0
        self.hidden_burned_reshuffles = 0
        self._reset_observed_card_caches()
        self._start_new_shoe_tracking(reason="initial_shoe", record_action=False)

    def _build_bet_action_cache(self) -> None:
        self._available_bet_multipliers = tuple(self.config.bet_multipliers)
        self._available_bet_multipliers_list = list(self._available_bet_multipliers)
        self._bet_amount_by_action = {
            f"bet_{multiplier}x": self.config.base_bet * float(multiplier)
            for multiplier in self._available_bet_multipliers
        }

    def _build_table_rule_caches(self) -> None:
        full_rules = {
            "n_decks": self.config.n_decks,
            "shoe_penetration": self.config.shoe_penetration,
            "dealer_hits_soft_17": self.config.dealer_hits_soft_17,
            "blackjack_payout": self.config.blackjack_payout,
            "dealer_peeks_for_blackjack": self.config.dealer_peeks_for_blackjack,
            "double_allowed_on": self.config.double_allowed_on,
            "double_after_split_allowed": self.config.double_after_split_allowed,
            "double_split_aces_allowed": self.config.double_split_aces_allowed,
            "split_rule": self.config.split_rule,
            "max_hands_after_split": self.config.max_hands_after_split,
            "max_split_depth_per_hand": self.config.max_split_depth_per_hand,
            "resplit_aces_allowed": self.config.resplit_aces_allowed,
            "hit_split_aces_allowed": self.config.hit_split_aces_allowed,
            "surrender_allowed": self.config.surrender_allowed,
            "insurance_allowed": self.config.insurance_allowed,
            "six_card_charlie_enabled": self.config.six_card_charlie_enabled,
            "base_bet": self.config.base_bet,
            "bet_multipliers": self._available_bet_multipliers,
            "use_cut_card": self.config.use_cut_card,
            "reward_mode": "round_end",
        }
        self._cached_full_table_rules = full_rules

        obs_cfg = self.config.observation
        if not obs_cfg.obs_include_table_rules:
            self._cached_visible_table_rules = {}
            return

        if obs_cfg.obs_include_visible_rules_only:
            selected = {name: full_rules[name] for name in VISIBLE_TABLE_RULES}
        else:
            selected = dict(full_rules)

        if obs_cfg.obs_include_hidden_rules:
            for name in HIDDEN_TABLE_RULES:
                selected[name] = full_rules[name]

        if obs_cfg.obs_include_n_decks:
            selected["n_decks"] = full_rules["n_decks"]

        if obs_cfg.obs_include_shoe_penetration_rule:
            selected["shoe_penetration"] = full_rules["shoe_penetration"]

        self._cached_visible_table_rules = selected

    def _reset_observed_card_caches(self) -> None:
        recent_window = max(self.config.observation.obs_recent_cards_window, 10)
        self._observed_rank_counts = {rank: 0 for rank in CARD_RANKS}
        self._observed_group_counts = {"low": 0, "neutral": 0, "high": 0}
        self._observed_cards_count = 0
        self._recent_observed_cards: deque[str] = deque(maxlen=recent_window)

    def _reset_round_state(self) -> None:
        self.player_hands: list[HandState] = []
        self.dealer_cards: list[str] = []
        self.current_hand_index: int | None = None
        self.decision_phase = "betting"
        self.round_over = False
        self.round_reward = 0.0
        self.insurance_bet = 0.0
        self.insurance_reward = 0.0
        self.insurance_offer_active = False
        self.dealer_revealed = False
        self.dealer_peeked = False
        self.dealer_has_blackjack: bool | None = None
        self._reshuffled_on_last_reset = False
        self.transition_log: list[dict[str, Any]] = []
        self.last_transition: dict[str, Any] | None = None
        self._dealer_hole_observed_this_round = False

    def load_shoe(
        self,
        cards: list[str],
        total_cards: int | None = None,
        *,
        strict: bool | None = None,
    ) -> None:
        normalized_total = len(cards) if total_cards is None else total_cards
        strict_validation = self.config.strict_shoe_validation if strict is None else strict
        validate_shoe_cards(
            cards,
            n_decks=self.config.n_decks,
            total_cards=normalized_total,
            strict=strict_validation,
        )
        self.shoe.force_order(cards, total_cards=normalized_total)
        self._reshuffle_pending = False
        self._start_new_shoe_tracking(reason="manual_load", record_action=False)
        self._manual_shoe_loaded = True
        self._start_state_prepared = True
        self.hidden_burned_rounds = 0
        self.hidden_burned_cards = 0
        self.hidden_burned_reshuffles = 0

    def reset(self) -> dict[str, Any]:
        self._reset_round_state()
        public_actions_added: list[dict[str, Any]] = []

        self._prepare_episode_start()

        if self._reshuffle_pending or self.shoe.should_reshuffle() or self.shoe.remaining_cards < 4:
            self.shoe.shuffle()
            self._reshuffle_pending = False
            self._reshuffled_on_last_reset = True
            self._start_new_shoe_tracking(reason="auto_reshuffle", record_action=True, public_actions_added=public_actions_added)

        if self.shoe.remaining_cards < 4:
            raise RuntimeError("The shoe does not contain enough cards to start a round.")

        self.round_index += 1
        self.total_rounds_played += 1
        self.rounds_since_shuffle += 1
        self._record_public_action(
            public_actions_added,
            actor="table",
            action="reset_to_betting",
            available_bet_multipliers=list(self._available_bet_multipliers),
        )

        response = self._build_response(action_name="reset_to_betting", reward=0.0)
        self._record_transition(
            action_name="reset_to_betting",
            observation_before=None,
            public_state_before=None,
            action_mask_before=None,
            action_mask_by_name_before=None,
            response=response,
            drawn_cards=[],
            public_actions_added=public_actions_added,
        )
        return response

    def _prepare_episode_start(self) -> None:
        if self._start_state_prepared or self._manual_shoe_loaded:
            return

        self._reshuffle_pending = False
        if self.start_state.mode == "fresh_shoe":
            self.last_shuffle_reason = "episode_fresh_shoe"
            self._start_state_prepared = True
            return

        burn_rounds = self._rng.randint(self.start_state.min_burned_rounds, self.start_state.max_burned_rounds)
        self.last_shuffle_reason = "episode_unknown_progress"
        self._burn_hidden_rounds(burn_rounds)
        if self.start_state.clear_visible_histories_after_burn:
            self._clear_public_histories_after_hidden_burn()
        self._start_state_prepared = True

    def _deal_round_after_bet(
        self,
        *,
        action_name: str,
        initial_bet: float,
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        if self.decision_phase != "betting":
            raise RuntimeError("The round is not waiting for a bet.")
        if self.shoe.remaining_cards < 4:
            raise RuntimeError("The shoe does not contain enough cards to deal after the bet.")

        self.total_player_hands_seen += 1
        self.player_hands_seen_since_shuffle += 1
        self.total_dealer_hands_seen += 1
        self.dealer_hands_seen_since_shuffle += 1

        bet_multiplier = BET_ACTION_MULTIPLIERS[action_name]
        self._record_public_action(
            public_actions_added,
            actor="player",
            action=action_name,
            multiplier=bet_multiplier,
            amount=initial_bet,
        )

        player_first = self._draw_card("player", drawn_cards, hand_index=0, visible=True, public_source="player")
        dealer_upcard = self._draw_card(
            "dealer_upcard",
            drawn_cards,
            visible=True,
            public_source="dealer_upcard",
        )
        player_second = self._draw_card("player", drawn_cards, hand_index=0, visible=True, public_source="player")
        dealer_hole = self._draw_card("dealer_hole", drawn_cards, visible=False)

        self.player_hands = [HandState(cards=[player_first, player_second], bet=initial_bet)]
        self.dealer_cards = [dealer_upcard, dealer_hole]
        self.current_hand_index = 0
        self.decision_phase = "playing"

        self._record_public_action(
            public_actions_added,
            actor="table",
            action="deal_round",
            player_hands=1,
            dealer_upcard=dealer_upcard,
            initial_bet=initial_bet,
            bet_multiplier=bet_multiplier,
        )
        self._apply_initial_table_rules(drawn_cards, public_actions_added)

    def _burn_hidden_rounds(self, n_rounds: int) -> None:
        for _ in range(n_rounds):
            self._play_hidden_round()
            self.hidden_burned_rounds += 1
            if self.shoe.should_reshuffle() or self.shoe.remaining_cards < 4:
                self.shoe.shuffle()
                self._reshuffle_pending = False
                self.hidden_burned_reshuffles += 1

    def _play_hidden_round(self) -> None:
        if self.shoe.remaining_cards < 4:
            self.shoe.shuffle()
            self._reshuffle_pending = False
            self.hidden_burned_reshuffles += 1

        player_cards = [self.shoe.draw()]
        dealer_cards = [self.shoe.draw()]
        player_cards.append(self.shoe.draw())
        dealer_cards.append(self.shoe.draw())
        self.hidden_burned_cards += 4

        player_blackjack = is_natural_blackjack(player_cards)
        dealer_blackjack = is_natural_blackjack(dealer_cards)

        if not player_blackjack and not dealer_blackjack:
            while hand_value(player_cards)[0] < 17:
                player_cards.append(self.shoe.draw())
                self.hidden_burned_cards += 1
                if hand_value(player_cards)[0] > 21:
                    break

        if not player_blackjack and hand_value(player_cards)[0] <= 21 and not dealer_blackjack:
            while True:
                dealer_total, dealer_soft = hand_value(dealer_cards)
                if dealer_total < 17 or (
                    dealer_total == 17 and dealer_soft and self.config.dealer_hits_soft_17
                ):
                    dealer_cards.append(self.shoe.draw())
                    self.hidden_burned_cards += 1
                    continue
                break

    def _clear_public_histories_after_hidden_burn(self) -> None:
        self.observed_cards_history = []
        self.public_action_history = []
        self._reset_observed_card_caches()
        self.last_round_outcome = None
        self.total_rounds_played = 0
        self.total_player_hands_seen = 0
        self.total_dealer_hands_seen = 0
        self.rounds_since_shuffle = 0
        self.player_hands_seen_since_shuffle = 0
        self.dealer_hands_seen_since_shuffle = 0
        self.shuffle_count = 0
        self.last_shuffle_reason = None
        self.transition_log = []
        self.last_transition = None

    def _hide_reshuffle_progress_from_observation(self) -> bool:
        return (
            self.start_state.mode == "unknown_progress"
            and self.start_state.hide_reshuffle_progress_from_observation
        )

    def step(self, action: Any) -> dict[str, Any]:
        if self.round_over:
            raise RuntimeError("The round is over. Call reset() to deal the next hand.")

        action_name = coerce_action_name(action)
        action_mask_by_name_before = self.legal_actions()
        if self.enable_transition_recording:
            observation_before = self.get_agent_observation()
            public_state_before = self.get_public_state()
            action_mask_before = [int(action_mask_by_name_before[name]) for name in ACTION_ORDER]
        else:
            observation_before = None
            public_state_before = None
            action_mask_before = None

        if not action_mask_by_name_before.get(action_name, False):
            raise ValueError(f"Illegal action '{action_name}' for the current state")

        drawn_cards: list[dict[str, Any]] = []
        public_actions_added: list[dict[str, Any]] = []

        if self.decision_phase == "betting":
            initial_bet = self._bet_amount_by_action.get(action_name)
            if initial_bet is None:
                raise ValueError(f"Unsupported betting action '{action_name}'")

            self._deal_round_after_bet(
                action_name=action_name,
                initial_bet=initial_bet,
                drawn_cards=drawn_cards,
                public_actions_added=public_actions_added,
            )
            reward = self.round_reward if self.round_over else 0.0
            response = self._build_response(action_name=action_name, reward=reward)
            self._record_transition(
                action_name=action_name,
                observation_before=observation_before,
                public_state_before=public_state_before,
                action_mask_before=action_mask_before,
                action_mask_by_name_before=action_mask_by_name_before,
                response=response,
                drawn_cards=drawn_cards,
                public_actions_added=public_actions_added,
            )
            return response

        if self.decision_phase != "playing":
            raise RuntimeError(f"Unsupported decision phase '{self.decision_phase}'")

        current_hand = self._current_hand()

        if self.insurance_offer_active and action_name != "insurance":
            self.insurance_offer_active = False
            if self.dealer_peeked and self.dealer_has_blackjack:
                self._finalize_round(play_dealer=False, drawn_cards=drawn_cards, public_actions_added=public_actions_added)
                response = self._build_response(action_name=action_name, reward=self.round_reward)
                self._record_transition(
                    action_name=action_name,
                    observation_before=observation_before,
                    public_state_before=public_state_before,
                    action_mask_before=action_mask_before,
                    action_mask_by_name_before=action_mask_by_name_before,
                    response=response,
                    drawn_cards=drawn_cards,
                    public_actions_added=public_actions_added,
                )
                return response
            if current_hand is not None and current_hand.is_blackjack():
                self._finalize_round(play_dealer=False, drawn_cards=drawn_cards, public_actions_added=public_actions_added)
                response = self._build_response(action_name=action_name, reward=self.round_reward)
                self._record_transition(
                    action_name=action_name,
                    observation_before=observation_before,
                    public_state_before=public_state_before,
                    action_mask_before=action_mask_before,
                    action_mask_by_name_before=action_mask_by_name_before,
                    response=response,
                    drawn_cards=drawn_cards,
                    public_actions_added=public_actions_added,
                )
                return response

        if action_name == "insurance":
            if current_hand is None:
                raise RuntimeError("No active hand is available.")
            self.insurance_bet = current_hand.bet * 0.5
            self.insurance_offer_active = False
            self._record_public_action(public_actions_added, actor="player", action="insurance", bet=self.insurance_bet)

            if self.dealer_peeked and self.dealer_has_blackjack:
                self._finalize_round(play_dealer=False, drawn_cards=drawn_cards, public_actions_added=public_actions_added)
                response = self._build_response(action_name=action_name, reward=self.round_reward)
                self._record_transition(
                    action_name=action_name,
                    observation_before=observation_before,
                    public_state_before=public_state_before,
                    action_mask_before=action_mask_before,
                    action_mask_by_name_before=action_mask_by_name_before,
                    response=response,
                    drawn_cards=drawn_cards,
                    public_actions_added=public_actions_added,
                )
                return response

            current_hand = self._current_hand()
            if current_hand is not None and current_hand.is_blackjack():
                self._finalize_round(play_dealer=False, drawn_cards=drawn_cards, public_actions_added=public_actions_added)
                response = self._build_response(action_name=action_name, reward=self.round_reward)
                self._record_transition(
                    action_name=action_name,
                    observation_before=observation_before,
                    public_state_before=public_state_before,
                    action_mask_before=action_mask_before,
                    action_mask_by_name_before=action_mask_by_name_before,
                    response=response,
                    drawn_cards=drawn_cards,
                    public_actions_added=public_actions_added,
                )
                return response

            response = self._build_response(action_name=action_name, reward=0.0)
            self._record_transition(
                action_name=action_name,
                observation_before=observation_before,
                public_state_before=public_state_before,
                action_mask_before=action_mask_before,
                action_mask_by_name_before=action_mask_by_name_before,
                response=response,
                drawn_cards=drawn_cards,
                public_actions_added=public_actions_added,
            )
            return response

        if current_hand is None:
            raise RuntimeError("No active hand is available.")

        if action_name == "stand":
            current_hand.action_count += 1
            current_hand.closed = True
            current_hand.close_reason = "stand"
            self._record_public_action(
                public_actions_added,
                actor="player",
                action="stand",
                hand_index=self.current_hand_index,
            )
            self._advance_round_flow(drawn_cards, public_actions_added)
        elif action_name == "hit":
            current_hand.action_count += 1
            card = self._draw_card(
                "player",
                drawn_cards,
                hand_index=self.current_hand_index,
                visible=True,
                public_source="player",
            )
            current_hand.cards.append(card)
            self._record_public_action(
                public_actions_added,
                actor="player",
                action="hit",
                hand_index=self.current_hand_index,
                card=card,
            )
            if self._is_six_card_charlie(current_hand):
                current_hand.closed = True
                current_hand.close_reason = "six_card_charlie"
                self._advance_round_flow(drawn_cards, public_actions_added)
            elif current_hand.is_bust():
                current_hand.closed = True
                current_hand.close_reason = "bust"
                self._advance_round_flow(drawn_cards, public_actions_added)
        elif action_name == "double":
            current_hand.action_count += 1
            current_hand.bet *= 2
            current_hand.doubled = True
            card = self._draw_card(
                "player",
                drawn_cards,
                hand_index=self.current_hand_index,
                visible=True,
                public_source="player",
            )
            current_hand.cards.append(card)
            current_hand.closed = True
            current_hand.close_reason = "bust" if current_hand.is_bust() else "double"
            self._record_public_action(
                public_actions_added,
                actor="player",
                action="double",
                hand_index=self.current_hand_index,
                card=card,
                bet=current_hand.bet,
            )
            self._advance_round_flow(drawn_cards, public_actions_added)
        elif action_name == "split":
            self._split_current_hand(drawn_cards, public_actions_added)
        elif action_name == "surrender":
            current_hand.action_count += 1
            current_hand.surrendered = True
            current_hand.closed = True
            current_hand.close_reason = "surrender"
            self._record_public_action(
                public_actions_added,
                actor="player",
                action="surrender",
                hand_index=self.current_hand_index,
            )
            self._advance_round_flow(drawn_cards, public_actions_added)
        else:
            raise ValueError(f"Unsupported action '{action_name}'")

        reward = self.round_reward if self.round_over else 0.0
        response = self._build_response(action_name=action_name, reward=reward)
        self._record_transition(
            action_name=action_name,
            observation_before=observation_before,
            public_state_before=public_state_before,
            action_mask_before=action_mask_before,
            action_mask_by_name_before=action_mask_by_name_before,
            response=response,
            drawn_cards=drawn_cards,
            public_actions_added=public_actions_added,
        )
        return response

    def action_mask(self) -> list[int]:
        legal = self.legal_actions()
        return [int(legal[name]) for name in ACTION_ORDER]

    def legal_actions(self) -> dict[str, bool]:
        legal = {name: False for name in ACTION_ORDER}
        if self.round_over:
            return legal

        if self.decision_phase == "betting":
            for action_name in self._bet_amount_by_action:
                legal[action_name] = True
            return legal

        if self.decision_phase != "playing":
            raise RuntimeError(f"Unsupported decision phase '{self.decision_phase}'")

        hand = self._current_hand()
        if hand is None:
            return legal

        if self.insurance_offer_active and self.insurance_bet == 0:
            legal["insurance"] = True

        if hand.is_blackjack():
            legal["stand"] = True
            return legal

        legal["stand"] = True
        legal["hit"] = self._can_hit(hand)
        legal["double"] = self._can_double(hand)
        legal["split"] = self._can_split(hand)
        legal["surrender"] = self._can_surrender(hand)
        return legal

    def get_agent_observation(self, mode: str | None = None) -> dict[str, Any]:
        obs_cfg = self.config.observation
        selected_mode = obs_cfg.obs_current_hand_mode if mode is None else mode
        if selected_mode not in {"basic_strategy", "table_raw"}:
            raise ValueError("observation mode must be 'basic_strategy' or 'table_raw'")

        hand = self._current_hand()
        observation: dict[str, Any] = {
            "profile": obs_cfg.profile,
            "mode": selected_mode,
            "dealer_upcard": self.dealer_cards[0] if self.dealer_cards else None,
        }

        if obs_cfg.obs_include_decision_phase:
            observation["decision_phase"] = self.decision_phase

        if obs_cfg.obs_include_available_bet_multipliers:
            observation["available_bet_multipliers"] = list(self._available_bet_multipliers)

        if self.dealer_cards:
            observation["dealer_upcard_value"] = rank_value(self.dealer_cards[0])
        else:
            observation["dealer_upcard_value"] = None

        if selected_mode == "basic_strategy":
            observation["current_hand_total"] = hand.total() if hand is not None else None
            observation["current_hand_is_soft"] = hand.is_soft() if hand is not None else None
        else:
            observation["current_hand_cards"] = list(hand.cards) if hand is not None else None
            if obs_cfg.obs_include_other_player_hands:
                observation["other_player_hands_visible"] = self._serialize_other_player_hands()

        if obs_cfg.obs_include_current_bet:
            observation["current_bet"] = hand.bet if hand is not None else None

        if obs_cfg.obs_include_betting_context:
            observation["betting_context"] = {
                "decision_phase": self.decision_phase,
                "available_bet_multipliers": list(self._available_bet_multipliers),
                "can_place_bet": self.decision_phase == "betting" and not self.round_over,
                "current_bet": hand.bet if hand is not None else None,
            }

        if obs_cfg.obs_include_hand_context:
            observation["hand_context"] = {
                "current_hand_index": self.current_hand_index,
                "n_player_hands": len(self.player_hands),
                "from_split": hand.from_split if hand is not None else None,
                "split_aces": hand.split_aces if hand is not None else None,
                "first_decision_on_hand": (
                    hand is not None and hand.action_count == 0 and len(hand.cards) == 2 and not hand.closed
                ),
            }

        if obs_cfg.obs_include_insurance_context:
            observation["insurance_context"] = {
                "insurance_offer_active": self.insurance_offer_active,
                "insurance_bet": self.insurance_bet,
            }

        if obs_cfg.obs_include_temporal_context:
            observation["temporal_context"] = self.get_temporal_features()

        if obs_cfg.obs_include_observed_cards_history:
            observation["observed_cards_history"] = self.get_observed_cards_summary()

        if obs_cfg.obs_include_discard_summary:
            observation["discard_summary"] = self.get_discard_summary()

        if obs_cfg.obs_include_exact_shoe_composition:
            observation["exact_shoe_composition"] = self.shoe.composition()

        return observation

    def get_temporal_features(self) -> dict[str, Any]:
        obs_cfg = self.config.observation
        features: dict[str, Any] = {
            "shuffle_count": self.shuffle_count,
            "rounds_played_total": self.total_rounds_played,
        }

        if obs_cfg.obs_include_hands_since_shuffle:
            features.update(
                {
                    "rounds_since_shuffle": self.rounds_since_shuffle,
                    "player_hands_seen_since_shuffle": self.player_hands_seen_since_shuffle,
                    "dealer_hands_seen_since_shuffle": self.dealer_hands_seen_since_shuffle,
                    "player_hands_seen_total": self.total_player_hands_seen,
                    "dealer_hands_seen_total": self.total_dealer_hands_seen,
                }
            )

        if obs_cfg.obs_include_estimated_shoe_progress and not self._hide_reshuffle_progress_from_observation():
            features["estimated_shoe_progress"] = self._build_estimated_shoe_progress()

        if obs_cfg.obs_include_last_hand_outcome:
            features["last_round_outcome"] = deepcopy(self.last_round_outcome)

        if obs_cfg.obs_include_recent_actions:
            features["recent_actions"] = self._get_recent_public_actions(obs_cfg.obs_recent_actions_window)

        return features

    def get_observed_cards_summary(self, mode: str | None = None) -> dict[str, Any]:
        obs_cfg = self.config.observation
        selected_mode = obs_cfg.obs_observed_cards_mode if mode is None else mode

        if selected_mode == "rank_counts":
            return dict(self._observed_rank_counts)

        if selected_mode == "low_neutral_high":
            return dict(self._observed_group_counts)

        if selected_mode == "recent_cards_sequence":
            return {
                "window": obs_cfg.obs_recent_cards_window,
                "cards": list(self._recent_observed_cards)[-obs_cfg.obs_recent_cards_window :],
            }

        raise ValueError("Unsupported observed cards mode")

    def get_discard_summary(self) -> dict[str, Any]:
        return {
            "observed_cards_count": self._observed_cards_count,
            "by_group": dict(self._observed_group_counts),
            "recent_cards": list(self._recent_observed_cards),
        }

    def get_table_rules(self) -> dict[str, Any]:
        return self.get_visible_table_rules()

    def get_visible_table_rules(self) -> dict[str, Any]:
        return dict(self._cached_visible_table_rules)

    def get_full_table_rules(self) -> dict[str, Any]:
        return dict(self._cached_full_table_rules)

    def get_public_state(self) -> dict[str, Any]:
        current_hand = self._current_hand()
        return {
            "round_index": self.round_index,
            "decision_phase": self.decision_phase,
            "round_over": self.round_over,
            "round_reward": self.round_reward,
            "current_bet": current_hand.bet if current_hand is not None else None,
            "available_bet_multipliers": list(self._available_bet_multipliers),
            "current_hand_index": self.current_hand_index,
            "current_hand": self._serialize_hand(current_hand, self.current_hand_index) if current_hand is not None else None,
            "player_hands": [self._serialize_hand(hand, index) for index, hand in enumerate(self.player_hands)],
            "dealer": self._serialize_public_dealer(),
            "insurance": {
                "offer_active": self.insurance_offer_active,
                "bet": self.insurance_bet,
                "reward": self.insurance_reward,
            },
            "shoe": {
                "remaining_cards": self.shoe.remaining_cards,
                "cards_used": self.shoe.total_cards - self.shoe.remaining_cards,
                "penetration_used": (
                    (self.shoe.total_cards - self.shoe.remaining_cards) / self.shoe.total_cards
                    if self.shoe.total_cards
                    else 0.0
                ),
                "cut_card_enabled": self.config.use_cut_card,
                "cut_card_reached": self.shoe.cut_card_reached,
                "last_hand_before_reshuffle": self.config.use_cut_card
                and (self.shoe.cut_card_reached or self._reshuffle_pending),
                "reshuffle_pending": self._reshuffle_pending,
                "reshuffled_on_reset": self._reshuffled_on_last_reset,
            },
            "table_rules_visible": self.get_visible_table_rules(),
            "history": {
                "rounds_since_shuffle": self.rounds_since_shuffle,
                "player_hands_seen_since_shuffle": self.player_hands_seen_since_shuffle,
                "dealer_hands_seen_since_shuffle": self.dealer_hands_seen_since_shuffle,
                "last_round_outcome": deepcopy(self.last_round_outcome),
                "recent_actions": self._get_recent_public_actions(10),
                "observed_cards_count": len(self.observed_cards_history),
                "discard_summary": self.get_discard_summary(),
            },
        }

    def get_debug_state(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "decision_phase": self.decision_phase,
            "round_over": self.round_over,
            "current_hand_index": self.current_hand_index,
            "round_reward": self.round_reward,
            "available_bet_multipliers": list(self._available_bet_multipliers),
            "table_rules": self.get_full_table_rules(),
            "visible_table_rules": self.get_visible_table_rules(),
            "observation_config": asdict(self.config.observation),
            "start_state": asdict(self.start_state),
            "dealer": {
                "cards": list(self.dealer_cards),
                "upcard": self.dealer_cards[0] if self.dealer_cards else None,
                "hole_card": self.dealer_cards[1] if len(self.dealer_cards) >= 2 else None,
                "revealed": self.dealer_revealed,
                "peek_checked": self.dealer_peeked,
                "has_blackjack": self.dealer_has_blackjack,
            },
            "player_hands": [self._serialize_hand(hand, index) for index, hand in enumerate(self.player_hands)],
            "insurance": {
                "offer_active": self.insurance_offer_active,
                "bet": self.insurance_bet,
                "reward": self.insurance_reward,
            },
            "shoe": {
                "cards": list(self.shoe.cards),
                "remaining_cards": self.shoe.remaining_cards,
                "total_cards": self.shoe.total_cards,
                "standard_total_cards": self.shoe.standard_total_cards,
                "cut_card_enabled": self.config.use_cut_card,
                "cut_card_reached": self.shoe.cut_card_reached,
                "reshuffle_pending": self._reshuffle_pending,
                "shuffle_count": self.shuffle_count,
                "last_shuffle_reason": self.last_shuffle_reason,
            },
            "history": {
                "rounds_played_total": self.total_rounds_played,
                "rounds_since_shuffle": self.rounds_since_shuffle,
                "player_hands_seen_total": self.total_player_hands_seen,
                "player_hands_seen_since_shuffle": self.player_hands_seen_since_shuffle,
                "dealer_hands_seen_total": self.total_dealer_hands_seen,
                "dealer_hands_seen_since_shuffle": self.dealer_hands_seen_since_shuffle,
                "last_round_outcome": deepcopy(self.last_round_outcome),
                "observed_cards_history": deepcopy(self.observed_cards_history),
                "public_action_history": deepcopy(self.public_action_history),
                "hidden_burned_rounds": self.hidden_burned_rounds,
                "hidden_burned_cards": self.hidden_burned_cards,
                "hidden_burned_reshuffles": self.hidden_burned_reshuffles,
            },
        }

    def get_transition_log(self) -> list[dict[str, Any]]:
        return deepcopy(self.transition_log)

    def _serialize_tracking_hand(self, hand: HandState | None) -> dict[str, Any] | None:
        if hand is None:
            return None
        return {
            "cards": list(hand.cards),
            "is_soft": hand.is_soft(),
            "from_split": hand.from_split,
        }

    def _build_compact_public_state(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "decision_phase": self.decision_phase,
            "current_hand": self._serialize_tracking_hand(self._current_hand()),
            "insurance": {"offer_active": self.insurance_offer_active},
            "hand_count": len(self.player_hands),
        }

    def _build_response(self, *, action_name: str | None, reward: float) -> dict[str, Any]:
        action_mask_by_name = self.legal_actions()
        action_mask = [int(action_mask_by_name[name]) for name in ACTION_ORDER]
        observation = self.get_agent_observation()

        if self.compact_response_mode and not self.enable_transition_recording:
            return {
                "observation": observation,
                "table_rules": self._cached_visible_table_rules,
                "action_mask": action_mask,
                "reward": reward,
                "done": self.round_over,
                "terminated": self.round_over,
                "truncated": False,
                "info": {
                    "action": action_name,
                    "round_index": self.round_index,
                    "decision_phase": self.decision_phase,
                    "round_over": self.round_over,
                    "round_reward": self.round_reward,
                    "insurance_reward": self.insurance_reward,
                    "hand_rewards": [hand.reward for hand in self.player_hands],
                    "hand_settlements": [hand.settlement for hand in self.player_hands],
                    "hand_count": len(self.player_hands),
                    "public_state": self._build_compact_public_state(),
                },
            }

        info = {
            "action": action_name,
            "round_index": self.round_index,
            "decision_phase": self.decision_phase,
            "round_over": self.round_over,
            "round_reward": self.round_reward,
            "insurance_reward": self.insurance_reward,
            "hand_rewards": [hand.reward for hand in self.player_hands],
            "hand_settlements": [hand.settlement for hand in self.player_hands],
            "public_state": self.get_public_state(),
        }
        return {
            "observation": observation,
            "table_rules": self.get_visible_table_rules(),
            "action_mask": action_mask,
            "action_mask_by_name": action_mask_by_name,
            "legal_actions": [name for name, allowed in action_mask_by_name.items() if allowed],
            "reward": reward,
            "done": self.round_over,
            "terminated": self.round_over,
            "truncated": False,
            "info": info,
        }

    def _record_transition(
        self,
        *,
        action_name: str,
        observation_before: dict[str, Any] | None,
        public_state_before: dict[str, Any] | None,
        action_mask_before: list[int] | None,
        action_mask_by_name_before: dict[str, bool] | None,
        response: dict[str, Any],
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        if not self.enable_transition_recording:
            response["info"]["transition_log_length"] = 0
            return
        transition = {
            "step_index": len(self.transition_log),
            "action": action_name,
            "observation_before": deepcopy(observation_before),
            "public_state_before": deepcopy(public_state_before),
            "action_mask_before": deepcopy(action_mask_before),
            "action_mask_by_name_before": deepcopy(action_mask_by_name_before),
            "drawn_cards": deepcopy(drawn_cards),
            "public_actions_added": deepcopy(public_actions_added),
            "observation_after": deepcopy(response["observation"]),
            "public_state_after": deepcopy(response["info"]["public_state"]),
            "table_rules_after": deepcopy(response["table_rules"]),
            "action_mask_after": deepcopy(response["action_mask"]),
            "action_mask_by_name_after": deepcopy(response["action_mask_by_name"]),
            "closed_hand_indices": [index for index, hand in enumerate(self.player_hands) if hand.closed],
            "hand_rewards": deepcopy(response["info"]["hand_rewards"]),
            "reward": response["reward"],
            "round_reward": self.round_reward,
            "hand_settlements": deepcopy(response["info"]["hand_settlements"]),
            "done": self.round_over,
        }
        self.last_transition = transition
        self.transition_log.append(transition)
        response["info"]["last_transition"] = deepcopy(transition)
        response["info"]["transition_log_length"] = len(self.transition_log)

    def _draw_card(
        self,
        recipient: str,
        drawn_cards: list[dict[str, Any]],
        *,
        hand_index: int | None = None,
        visible: bool,
        public_source: str | None = None,
    ) -> str:
        card = self.shoe.draw()
        event: dict[str, Any] = {"recipient": recipient, "card": card, "visible": visible}
        if hand_index is not None:
            event["hand_index"] = hand_index
        drawn_cards.append(event)

        if visible:
            self._observe_public_card(card, source=public_source or recipient, hand_index=hand_index)
        return card

    def _observe_public_card(self, card: str, *, source: str, hand_index: int | None = None) -> None:
        event: dict[str, Any] = {"card": card, "source": source, "round_index": self.round_index}
        if hand_index is not None:
            event["hand_index"] = hand_index
        self.observed_cards_history.append(event)
        self._observed_rank_counts[card] += 1
        self._observed_cards_count += 1
        self._recent_observed_cards.append(card)
        if card in LOW_RANKS:
            self._observed_group_counts["low"] += 1
        elif card in NEUTRAL_RANKS:
            self._observed_group_counts["neutral"] += 1
        else:
            self._observed_group_counts["high"] += 1

    def _record_public_action(
        self,
        public_actions_added: list[dict[str, Any]] | None,
        *,
        actor: str,
        action: str,
        **details: Any,
    ) -> dict[str, Any]:
        event = {
            "actor": actor,
            "action": action,
            "token": f"{actor}:{action}",
            "round_index": self.round_index,
        }
        event.update(details)
        self.public_action_history.append(event)
        if public_actions_added is not None:
            public_actions_added.append(event)
        return event

    def _start_new_shoe_tracking(
        self,
        *,
        reason: str,
        record_action: bool,
        public_actions_added: list[dict[str, Any]] | None = None,
    ) -> None:
        self.shuffle_count += 1
        self.rounds_since_shuffle = 0
        self.player_hands_seen_since_shuffle = 0
        self.dealer_hands_seen_since_shuffle = 0
        self.last_shuffle_reason = reason

        if self.config.observation.obs_reset_history_on_shuffle:
            self.observed_cards_history = []
            self._reset_observed_card_caches()

        if record_action:
            self._record_public_action(public_actions_added, actor="table", action="reshuffle", reason=reason)

    def _apply_initial_table_rules(
        self,
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        upcard = self.dealer_cards[0]
        self.dealer_peeked = self.config.dealer_peeks_for_blackjack and (
            upcard == "A" or split_value(upcard) == 10
        )

        if self.dealer_peeked:
            self.dealer_has_blackjack = is_natural_blackjack(self.dealer_cards)

        if upcard == "A" and self.config.insurance_allowed:
            self.insurance_offer_active = True
            self._record_public_action(public_actions_added, actor="dealer", action="offer_insurance")

        current_hand = self._current_hand()
        if current_hand is None:
            return

        if current_hand.is_blackjack() and not self.insurance_offer_active:
            self._finalize_round(play_dealer=False, drawn_cards=drawn_cards, public_actions_added=public_actions_added)
            return

        if self.dealer_peeked and self.dealer_has_blackjack and not self.insurance_offer_active:
            self._finalize_round(play_dealer=False, drawn_cards=drawn_cards, public_actions_added=public_actions_added)

    def _current_hand(self) -> HandState | None:
        if self.current_hand_index is None:
            return None
        if self.current_hand_index < 0 or self.current_hand_index >= len(self.player_hands):
            return None
        return self.player_hands[self.current_hand_index]

    def _is_six_card_charlie(self, hand: HandState) -> bool:
        return self.config.six_card_charlie_enabled and len(hand.cards) >= 6 and hand.total() <= 21

    def _hand_needs_dealer_resolution(self, hand: HandState) -> bool:
        return not hand.is_bust() and not hand.surrendered and not self._is_six_card_charlie(hand)

    def _can_hit(self, hand: HandState) -> bool:
        if hand.closed or hand.total() >= 21:
            return False
        if hand.split_aces and not self.config.hit_split_aces_allowed:
            return False
        return True

    def _can_double(self, hand: HandState) -> bool:
        if hand.closed or hand.action_count > 0 or len(hand.cards) != 2:
            return False
        if hand.from_split and not self.config.double_after_split_allowed:
            return False
        if hand.split_aces and not self.config.double_split_aces_allowed:
            return False

        total, is_soft = hand_value(hand.cards)
        if self.config.double_allowed_on == "any_two_cards":
            return total < 21
        if self.config.double_allowed_on == "hard_9_10_11":
            return not is_soft and total in {9, 10, 11}
        if self.config.double_allowed_on == "hard_10_11":
            return not is_soft and total in {10, 11}
        return False

    def _can_split(self, hand: HandState) -> bool:
        if hand.closed or hand.action_count > 0 or len(hand.cards) != 2:
            return False
        if len(self.player_hands) >= self.config.max_hands_after_split:
            return False
        if (
            self.config.max_split_depth_per_hand is not None
            and hand.split_depth >= self.config.max_split_depth_per_hand
        ):
            return False
        if hand.split_aces and not self.config.resplit_aces_allowed:
            return False

        left, right = hand.cards
        if self.config.split_rule == "same_rank":
            return left == right
        return split_value(left) == split_value(right)

    def _can_surrender(self, hand: HandState) -> bool:
        if not self.config.surrender_allowed:
            return False
        if hand.closed or hand.from_split or hand.action_count > 0 or len(hand.cards) != 2:
            return False
        return not hand.is_blackjack()

    def _split_current_hand(
        self,
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        hand = self._current_hand()
        if hand is None:
            raise RuntimeError("No active hand is available to split.")

        left, right = hand.cards
        is_split_aces = left == "A" and right == "A"
        active_index = self.current_hand_index

        first_card = self._draw_card(
            "player_split",
            drawn_cards,
            hand_index=active_index,
            visible=True,
            public_source="player_split",
        )
        second_card = self._draw_card(
            "player_split",
            drawn_cards,
            hand_index=active_index + 1 if active_index is not None else None,
            visible=True,
            public_source="player_split",
        )

        first_hand = HandState(
            cards=[left, first_card],
            bet=hand.bet,
            from_split=True,
            split_depth=hand.split_depth + 1,
            split_aces=is_split_aces,
        )
        second_hand = HandState(
            cards=[right, second_card],
            bet=hand.bet,
            from_split=True,
            split_depth=hand.split_depth + 1,
            split_aces=is_split_aces,
        )

        self.player_hands[self.current_hand_index : self.current_hand_index + 1] = [first_hand, second_hand]
        self.total_player_hands_seen += 1
        self.player_hands_seen_since_shuffle += 1

        self._record_public_action(
            public_actions_added,
            actor="player",
            action="split",
            hand_index=active_index,
            created_hands=2,
        )

        self._auto_close_locked_split_hand(first_hand)
        self._auto_close_locked_split_hand(second_hand)
        self._advance_round_flow(drawn_cards, public_actions_added)

    def _auto_close_locked_split_hand(self, hand: HandState) -> None:
        if not hand.split_aces or self.config.hit_split_aces_allowed:
            return
        if self._can_split(hand):
            return
        hand.closed = True
        hand.close_reason = "split_aces_locked"

    def _advance_round_flow(
        self,
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        current_hand = self._current_hand()
        if current_hand is not None and not current_hand.closed:
            return

        next_index = self._find_next_open_hand()
        if next_index is not None:
            self.current_hand_index = next_index
            next_hand = self._current_hand()
            if next_hand is not None:
                self._auto_close_locked_split_hand(next_hand)
            if next_hand is not None and next_hand.closed:
                self._advance_round_flow(drawn_cards, public_actions_added)
            return

        self.current_hand_index = None
        self._finalize_round(
            play_dealer=self._should_play_dealer(),
            drawn_cards=drawn_cards,
            public_actions_added=public_actions_added,
        )

    def _find_next_open_hand(self) -> int | None:
        if self.current_hand_index is None:
            return None
        for index in range(self.current_hand_index + 1, len(self.player_hands)):
            if not self.player_hands[index].closed:
                return index
        return None

    def _should_play_dealer(self) -> bool:
        return any(self._hand_needs_dealer_resolution(hand) for hand in self.player_hands)

    def _finalize_round(
        self,
        *,
        play_dealer: bool,
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        self.insurance_offer_active = False
        self.dealer_revealed = True
        self._reveal_dealer_hole(public_actions_added)

        if play_dealer:
            self._play_dealer(drawn_cards, public_actions_added)

        dealer_total, _ = hand_value(self.dealer_cards)
        dealer_blackjack = is_natural_blackjack(self.dealer_cards)
        self.dealer_has_blackjack = dealer_blackjack

        round_reward = 0.0
        for hand in self.player_hands:
            hand_reward, settlement = self._settle_hand(hand, dealer_total, dealer_blackjack)
            hand.reward = hand_reward
            hand.settlement = settlement
            round_reward += hand_reward

        if self.insurance_bet:
            self.insurance_reward = self.insurance_bet * 2 if dealer_blackjack else -self.insurance_bet
            round_reward += self.insurance_reward
        else:
            self.insurance_reward = 0.0

        self.round_reward = round_reward
        self.round_over = True
        self._reshuffle_pending = self.shoe.should_reshuffle()
        self.last_round_outcome = {
            "round_index": self.round_index,
            "reward": self.round_reward,
            "insurance_reward": self.insurance_reward,
            "dealer_total": dealer_total,
            "dealer_has_blackjack": dealer_blackjack,
            "dealer_cards": list(self.dealer_cards),
            "hand_rewards": [hand.reward for hand in self.player_hands],
            "hand_settlements": [hand.settlement for hand in self.player_hands],
        }
        self._record_public_action(
            public_actions_added,
            actor="table",
            action="settle_round",
            reward=self.round_reward,
            hand_settlements=[hand.settlement for hand in self.player_hands],
        )

    def _reveal_dealer_hole(self, public_actions_added: list[dict[str, Any]]) -> None:
        if self._dealer_hole_observed_this_round or len(self.dealer_cards) < 2:
            return
        hole_card = self.dealer_cards[1]
        self._observe_public_card(hole_card, source="dealer_hole")
        self._dealer_hole_observed_this_round = True
        self._record_public_action(public_actions_added, actor="dealer", action="reveal_hole", card=hole_card)

    def _play_dealer(
        self,
        drawn_cards: list[dict[str, Any]],
        public_actions_added: list[dict[str, Any]],
    ) -> None:
        if is_natural_blackjack(self.dealer_cards):
            return

        while True:
            total, is_soft = hand_value(self.dealer_cards)
            if total < 17:
                card = self._draw_card("dealer", drawn_cards, visible=True, public_source="dealer")
                self.dealer_cards.append(card)
                self._record_public_action(public_actions_added, actor="dealer", action="hit", card=card)
                continue
            if total == 17 and is_soft and self.config.dealer_hits_soft_17:
                card = self._draw_card("dealer", drawn_cards, visible=True, public_source="dealer")
                self.dealer_cards.append(card)
                self._record_public_action(public_actions_added, actor="dealer", action="hit", card=card)
                continue
            self._record_public_action(public_actions_added, actor="dealer", action="stand", total=total)
            return

    def _settle_hand(self, hand: HandState, dealer_total: int, dealer_blackjack: bool) -> tuple[float, str]:
        if hand.surrendered:
            return -(hand.bet / 2), "surrender"
        if hand.is_bust():
            return -hand.bet, "loss"
        if self._is_six_card_charlie(hand):
            return hand.bet, "six_card_charlie"
        if dealer_blackjack:
            if hand.is_blackjack():
                return 0.0, "push"
            return -hand.bet, "loss"
        if hand.is_blackjack():
            return hand.bet * self.config.blackjack_payout, "blackjack"
        if dealer_total > 21:
            return hand.bet, "win"

        player_total = hand.total()
        if player_total > dealer_total:
            return hand.bet, "win"
        if player_total < dealer_total:
            return -hand.bet, "loss"
        return 0.0, "push"

    def _serialize_hand(self, hand: HandState | None, index: int | None) -> dict[str, Any] | None:
        if hand is None:
            return None

        total, is_soft = hand_value(hand.cards)
        return {
            "index": index,
            "cards": list(hand.cards),
            "total": total,
            "is_soft": is_soft,
            "is_blackjack": hand.is_blackjack(),
            "is_bust": hand.is_bust(),
            "is_six_card_charlie": self._is_six_card_charlie(hand),
            "bet": hand.bet,
            "doubled": hand.doubled,
            "from_split": hand.from_split,
            "split_depth": hand.split_depth,
            "split_aces": hand.split_aces,
            "closed": hand.closed,
            "surrendered": hand.surrendered,
            "action_count": hand.action_count,
            "close_reason": hand.close_reason,
            "settlement": hand.settlement,
            "reward": hand.reward,
        }

    def _serialize_other_player_hands(self) -> list[dict[str, Any]]:
        visible_hands: list[dict[str, Any]] = []
        for index, hand in enumerate(self.player_hands):
            if index == self.current_hand_index:
                continue
            visible_hands.append(
                {
                    "index": index,
                    "cards": list(hand.cards),
                    "bet": hand.bet,
                    "from_split": hand.from_split,
                    "split_aces": hand.split_aces,
                    "closed": hand.closed,
                }
            )
        return visible_hands

    def _serialize_public_dealer(self) -> dict[str, Any]:
        if not self.dealer_cards:
            return {
                "upcard": None,
                "cards": [],
                "hole_card_hidden": False,
                "visible_total": None,
                "visible_is_soft": None,
                "peek_checked": self.dealer_peeked,
                "has_blackjack": self.dealer_has_blackjack if self.round_over else None,
            }

        revealed = self.dealer_revealed or self.round_over
        visible_cards = list(self.dealer_cards if revealed else self.dealer_cards[:1])
        visible_total, visible_soft = hand_value(visible_cards)
        payload: dict[str, Any] = {
            "upcard": self.dealer_cards[0] if self.dealer_cards else None,
            "cards": visible_cards,
            "hole_card_hidden": not revealed,
            "visible_total": visible_total,
            "visible_is_soft": visible_soft,
            "peek_checked": self.dealer_peeked,
            "has_blackjack": self.dealer_has_blackjack if revealed else None,
        }
        if revealed and self.dealer_cards:
            total, is_soft = hand_value(self.dealer_cards)
            payload["total"] = total
            payload["is_soft"] = is_soft
        return payload

    def _get_recent_public_actions(self, window: int) -> list[dict[str, Any]]:
        return deepcopy(self.public_action_history[-window:])

    def _build_estimated_shoe_progress(self) -> dict[str, Any]:
        if not self.shoe.total_cards:
            return {"fraction_used": 0.0, "bucket": "early"}

        fraction_used = (self.shoe.total_cards - self.shoe.remaining_cards) / self.shoe.total_cards
        if fraction_used < 0.33:
            bucket = "early"
        elif fraction_used < 0.66:
            bucket = "mid"
        else:
            bucket = "late"
        return {"fraction_used": fraction_used, "bucket": bucket}

    def _low_neutral_high_counts(self, cards: list[str]) -> dict[str, int]:
        counts = {"low": 0, "neutral": 0, "high": 0}
        for card in cards:
            if card in LOW_RANKS:
                counts["low"] += 1
            elif card in NEUTRAL_RANKS:
                counts["neutral"] += 1
            elif card in HIGH_RANKS:
                counts["high"] += 1
        return counts
