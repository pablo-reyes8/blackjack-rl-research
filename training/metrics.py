from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from enviroment_bj.core import ACTION_ORDER, BET_ACTION_ORDER, PLAYING_ACTION_ORDER, split_value


@dataclass(slots=True)
class ScalarMetricAccumulator:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.totals[key] += float(value)
                self.counts[key] += 1

    def summary(self) -> dict[str, float]:
        return {
            key: self.totals[key] / self.counts[key]
            for key in self.totals
            if self.counts[key] > 0
        }


@dataclass(slots=True)
class BehaviorMetricsTracker:
    action_counts: Counter[str] = field(default_factory=Counter)
    phase_action_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    random_action_count: int = 0
    greedy_action_count: int = 0
    total_decisions: int = 0
    random_action_count_by_phase: Counter[str] = field(default_factory=Counter)
    greedy_action_count_by_phase: Counter[str] = field(default_factory=Counter)
    decisions_by_phase: Counter[str] = field(default_factory=Counter)
    total_rounds: int = 0
    total_hands: int = 0
    total_reward: float = 0.0
    total_hand_reward: float = 0.0
    total_insurance_reward: float = 0.0
    reward_by_phase: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    settlement_counts: Counter[str] = field(default_factory=Counter)
    situation_counts: Counter[str] = field(default_factory=Counter)
    action_by_situation: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    table_counts: Counter[str] = field(default_factory=Counter)
    pending_bets_by_round: dict[tuple[str, int], str] = field(default_factory=dict)
    bet_reward_totals: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    bet_round_counts: Counter[str] = field(default_factory=Counter)
    hand_close_reason_counts: Counter[str] = field(default_factory=Counter)
    round_reward_samples: list[float] = field(default_factory=list)

    def _decision_phase(self, response: dict[str, Any]) -> str:
        observation = response.get("observation") or {}
        phase = observation.get("decision_phase")
        if isinstance(phase, str) and phase:
            return phase
        public_state = (response.get("info") or {}).get("public_state") or {}
        phase = public_state.get("decision_phase")
        return phase if isinstance(phase, str) and phase else "playing"

    def record_decision(
        self,
        response: dict[str, Any],
        action_name: str,
        *,
        was_random: bool,
        table_key: str,
        env_key: str,
    ) -> None:
        self.action_counts[action_name] += 1
        self.total_decisions += 1
        self.table_counts[table_key] += 1
        phase = self._decision_phase(response)
        self.phase_action_counts[phase][action_name] += 1
        self.decisions_by_phase[phase] += 1
        if was_random:
            self.random_action_count += 1
            self.random_action_count_by_phase[phase] += 1
        else:
            self.greedy_action_count += 1
            self.greedy_action_count_by_phase[phase] += 1

        info = response["info"]
        public_state = info.get("public_state") or {}
        current_hand = public_state.get("current_hand")
        insurance_offer = bool(public_state.get("insurance", {}).get("offer_active", False))

        if phase == "betting" and action_name in BET_ACTION_ORDER:
            self.pending_bets_by_round[(env_key, int(info.get("round_index", public_state.get("round_index", 0))))] = action_name

        if current_hand:
            cards = current_hand.get("cards", [])
            is_soft = current_hand.get("is_soft", False)
            from_split = current_hand.get("from_split", False)
            is_pair = len(cards) == 2 and split_value(cards[0]) == split_value(cards[1])

            hand_type = "soft_hands" if is_soft else "hard_hands"
            self.situation_counts[hand_type] += 1
            self.action_by_situation[hand_type][action_name] += 1
            if is_pair:
                self.situation_counts["pairs"] += 1
                self.action_by_situation["pairs"][action_name] += 1
            if from_split:
                self.situation_counts["post_split_states"] += 1
                self.action_by_situation["post_split_states"][action_name] += 1

        if insurance_offer:
            self.situation_counts["insurance_offers"] += 1
            self.action_by_situation["insurance_offers"][action_name] += 1

    def record_round_result(self, response: dict[str, Any], *, env_key: str) -> None:
        if not response["done"]:
            return
        info = response["info"]
        public_state = info.get("public_state") or {}
        settlements = info.get("hand_settlements", [])
        hand_rewards = info.get("hand_rewards", [])
        hand_close_reasons = info.get("hand_close_reasons", [])
        insurance_reward = float(info.get("insurance_reward", 0.0))
        hand_count = int(info.get("hand_count", public_state.get("hand_count", len(public_state.get("player_hands", [])))))

        self.total_rounds += 1
        self.total_hands += hand_count
        self.total_reward += float(response["reward"])
        self.round_reward_samples.append(float(response["reward"]))
        self.total_hand_reward += float(sum(hand_rewards))
        self.total_insurance_reward += insurance_reward
        self.reward_by_phase["betting"] += float(response["reward"])
        self.reward_by_phase["playing"] += float(response["reward"])
        for settlement in settlements:
            if settlement is not None:
                self.settlement_counts[settlement] += 1
        for close_reason in hand_close_reasons:
            if close_reason is not None:
                self.hand_close_reason_counts[close_reason] += 1

        round_key = (env_key, int(info.get("round_index", public_state.get("round_index", 0))))
        bet_action = self.pending_bets_by_round.pop(round_key, None)
        if bet_action is not None:
            self.bet_reward_totals[bet_action] += float(response["reward"])
            self.bet_round_counts[bet_action] += 1

    def summary(self) -> dict[str, Any]:
        total_hands = max(self.total_hands, 1)
        total_rounds = max(self.total_rounds, 1)
        total_decisions = max(self.total_decisions, 1)
        total_betting_decisions = max(self.decisions_by_phase["betting"], 1)
        total_playing_decisions = max(self.decisions_by_phase["playing"], 1)

        action_frequency = {
            action: self.action_counts[action] / total_decisions
            for action in ACTION_ORDER
        }
        bet_action_frequencies = {
            action: self.phase_action_counts["betting"][action] / total_betting_decisions
            for action in BET_ACTION_ORDER
        }
        play_action_frequencies = {
            action: self.phase_action_counts["playing"][action] / total_playing_decisions
            for action in PLAYING_ACTION_ORDER
        }
        win_like = self.settlement_counts["win"] + self.settlement_counts["blackjack"]
        round_reward_mean = self.total_reward / total_rounds
        round_reward_variance = 0.0
        if self.round_reward_samples:
            round_reward_variance = sum(
                (reward - round_reward_mean) ** 2 for reward in self.round_reward_samples
            ) / len(self.round_reward_samples)
        return {
            "reward_per_round": self.total_reward / total_rounds,
            "reward_per_hand": self.total_reward / total_hands,
            "ev_per_1000_hands": 1000.0 * self.total_reward / total_hands,
            "round_reward_std": sqrt(round_reward_variance),
            "win_rate": win_like / total_hands,
            "push_rate": self.settlement_counts["push"] / total_hands,
            "loss_rate": self.settlement_counts["loss"] / total_hands,
            "surrender_rate": self.settlement_counts["surrender"] / total_hands,
            "blackjack_rate": self.settlement_counts["blackjack"] / total_hands,
            "bust_rate": self.hand_close_reason_counts["bust"] / total_hands,
            "action_frequencies": action_frequency,
            "bet_action_frequencies": bet_action_frequencies,
            "play_action_frequencies": play_action_frequencies,
            "random_action_fraction": self.random_action_count / total_decisions,
            "greedy_action_fraction": self.greedy_action_count / total_decisions,
            "random_action_fraction_betting": self.random_action_count_by_phase["betting"] / total_betting_decisions,
            "random_action_fraction_playing": self.random_action_count_by_phase["playing"] / total_playing_decisions,
            "greedy_action_fraction_betting": self.greedy_action_count_by_phase["betting"] / total_betting_decisions,
            "greedy_action_fraction_playing": self.greedy_action_count_by_phase["playing"] / total_playing_decisions,
            "betting_decisions": float(self.decisions_by_phase["betting"]),
            "playing_decisions": float(self.decisions_by_phase["playing"]),
            "bet_reward_per_round": self.reward_by_phase["betting"] / total_rounds,
            "play_reward_per_round": self.reward_by_phase["playing"] / total_rounds,
            "bet_ev_per_1000_rounds_by_action": {
                action: 1000.0 * self.bet_reward_totals[action] / max(self.bet_round_counts[action], 1)
                for action in BET_ACTION_ORDER
            },
            "bet_round_fraction_by_action": {
                action: self.bet_round_counts[action] / total_rounds
                for action in BET_ACTION_ORDER
            },
            "conservative_bet_fraction": self.bet_round_counts["bet_1x"] / total_rounds,
            "aggressive_bet_fraction": (self.bet_round_counts["bet_3x"] + self.bet_round_counts["bet_4x"]) / total_rounds,
            "rounds_completed": float(self.total_rounds),
            "hands_completed": float(self.total_hands),
            "insurance_reward_total": self.total_insurance_reward,
            "situation_counts": dict(self.situation_counts),
            "action_by_situation": {key: dict(counter) for key, counter in self.action_by_situation.items()},
            "table_counts": dict(self.table_counts),
        }
