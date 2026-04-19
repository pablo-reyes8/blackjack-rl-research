from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from enviroment_bj.core import ACTION_ORDER, split_value


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
    random_action_count: int = 0
    greedy_action_count: int = 0
    total_decisions: int = 0
    total_rounds: int = 0
    total_hands: int = 0
    total_reward: float = 0.0
    total_hand_reward: float = 0.0
    total_insurance_reward: float = 0.0
    settlement_counts: Counter[str] = field(default_factory=Counter)
    situation_counts: Counter[str] = field(default_factory=Counter)
    action_by_situation: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    table_counts: Counter[str] = field(default_factory=Counter)

    def record_decision(self, response: dict[str, Any], action_name: str, *, was_random: bool, table_key: str) -> None:
        self.action_counts[action_name] += 1
        self.total_decisions += 1
        self.table_counts[table_key] += 1
        if was_random:
            self.random_action_count += 1
        else:
            self.greedy_action_count += 1

        public_state = response["info"]["public_state"]
        current_hand = public_state.get("current_hand")
        insurance_offer = public_state.get("insurance", {}).get("offer_active", False)

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

    def record_round_result(self, response: dict[str, Any]) -> None:
        if not response["done"]:
            return
        info = response["info"]
        public_state = info["public_state"]
        settlements = info.get("hand_settlements", [])
        hand_rewards = info.get("hand_rewards", [])
        insurance_reward = float(info.get("insurance_reward", 0.0))
        hand_count = len(public_state.get("player_hands", []))

        self.total_rounds += 1
        self.total_hands += hand_count
        self.total_reward += float(response["reward"])
        self.total_hand_reward += float(sum(hand_rewards))
        self.total_insurance_reward += insurance_reward
        for settlement in settlements:
            if settlement is not None:
                self.settlement_counts[settlement] += 1

    def summary(self) -> dict[str, Any]:
        total_hands = max(self.total_hands, 1)
        total_rounds = max(self.total_rounds, 1)
        total_decisions = max(self.total_decisions, 1)

        action_frequency = {
            action: self.action_counts[action] / total_decisions
            for action in ACTION_ORDER
        }
        win_like = self.settlement_counts["win"] + self.settlement_counts["blackjack"]
        return {
            "reward_per_round": self.total_reward / total_rounds,
            "reward_per_hand": self.total_reward / total_hands,
            "ev_per_1000_hands": 1000.0 * self.total_reward / total_hands,
            "win_rate": win_like / total_hands,
            "push_rate": self.settlement_counts["push"] / total_hands,
            "loss_rate": self.settlement_counts["loss"] / total_hands,
            "surrender_rate": self.settlement_counts["surrender"] / total_hands,
            "action_frequencies": action_frequency,
            "random_action_fraction": self.random_action_count / total_decisions,
            "greedy_action_fraction": self.greedy_action_count / total_decisions,
            "rounds_completed": float(self.total_rounds),
            "hands_completed": float(self.total_hands),
            "insurance_reward_total": self.total_insurance_reward,
            "situation_counts": dict(self.situation_counts),
            "action_by_situation": {key: dict(counter) for key, counter in self.action_by_situation.items()},
            "table_counts": dict(self.table_counts),
        }
