from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any, Mapping

import torch

from enviroment_bj.core import BET_ACTION_ORDER

from .config import ObservedEVRankingConfig
from .count_utils import COUNT_BUCKET_NAMES, map_count_proxy_to_bucket


class EVBucketActionTable:
    def __init__(
        self,
        bucket_names: tuple[str, ...] = COUNT_BUCKET_NAMES,
        action_names: tuple[str, ...] = BET_ACTION_ORDER,
    ) -> None:
        self.bucket_names = tuple(bucket_names)
        self.action_names = tuple(action_names)
        self.reward_totals: defaultdict[tuple[str, str], float] = defaultdict(float)
        self.reward_squares: defaultdict[tuple[str, str], float] = defaultdict(float)
        self.counts: Counter[tuple[str, str]] = Counter()

    def update(self, bucket: str, action: str, reward: float) -> None:
        if bucket not in self.bucket_names or action not in self.action_names:
            return
        key = (bucket, action)
        value = float(reward)
        self.reward_totals[key] += value
        self.reward_squares[key] += value * value
        self.counts[key] += 1

    def action_stats(
        self,
        bucket: str,
        action: str,
        *,
        compute_confidence_intervals: bool = True,
        confidence_z: float = 1.96,
    ) -> dict[str, float]:
        key = (bucket, action)
        n = int(self.counts[key])
        if n <= 0:
            return {
                "n": 0.0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "ev_per_1000": 0.0,
                "se": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
            }

        mean = self.reward_totals[key] / float(n)
        variance = max(self.reward_squares[key] / float(n) - mean * mean, 0.0)
        std = sqrt(variance)
        se = std / sqrt(float(n)) if n > 0 else 0.0
        ci_low = 1000.0 * (mean - confidence_z * se) if compute_confidence_intervals else 0.0
        ci_high = 1000.0 * (mean + confidence_z * se) if compute_confidence_intervals else 0.0
        return {
            "n": float(n),
            "mean_reward": float(mean),
            "std_reward": float(std),
            "ev_per_1000": float(1000.0 * mean),
            "se": float(se),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    def summary(
        self,
        *,
        compute_confidence_intervals: bool = True,
        confidence_z: float = 1.96,
    ) -> dict[str, dict[str, dict[str, float]]]:
        return {
            bucket: {
                action: self.action_stats(
                    bucket,
                    action,
                    compute_confidence_intervals=compute_confidence_intervals,
                    confidence_z=confidence_z,
                )
                for action in self.action_names
            }
            for bucket in self.bucket_names
        }

    def get_preferred_pairs(
        self,
        *,
        min_samples: int,
        min_ev_gap_per_round: float,
        compare_against_1x_only: bool,
        allowed_actions: tuple[str, ...] | None = None,
    ) -> dict[str, list[tuple[str, str, float]]]:
        actions = tuple(action for action in (allowed_actions or self.action_names) if action in self.action_names)
        pairs_by_bucket: dict[str, list[tuple[str, str, float]]] = {}
        for bucket in self.bucket_names:
            candidates: list[tuple[str, float]] = []
            for action in actions:
                count = int(self.counts[(bucket, action)])
                if count < min_samples:
                    continue
                candidates.append((action, self.reward_totals[(bucket, action)] / float(count)))

            pairs: list[tuple[str, str, float]] = []
            if compare_against_1x_only:
                base = next((item for item in candidates if item[0] == "bet_1x"), None)
                if base is not None:
                    base_action, base_mean = base
                    for action, mean in candidates:
                        if action == base_action:
                            continue
                        gap = mean - base_mean
                        if abs(gap) < min_ev_gap_per_round:
                            continue
                        good, bad = (action, base_action) if gap > 0 else (base_action, action)
                        pairs.append((good, bad, abs(float(gap))))
            else:
                for good_action, good_mean in candidates:
                    for bad_action, bad_mean in candidates:
                        if good_action == bad_action:
                            continue
                        gap = good_mean - bad_mean
                        if gap >= min_ev_gap_per_round:
                            pairs.append((good_action, bad_action, float(gap)))

            if pairs:
                pairs_by_bucket[bucket] = pairs
        return pairs_by_bucket


def observed_ev_ranking_weight(config: ObservedEVRankingConfig, update_count: int) -> float:
    fraction = min(max(float(update_count), 0.0) / float(config.decay_steps), 1.0)
    return float(config.weight + fraction * (config.final_weight - config.weight))


def compute_observed_ev_ranking_loss(
    student_output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    *,
    config: ObservedEVRankingConfig,
    ev_table: EVBucketActionTable,
    bet_action_names: tuple[str, ...] = BET_ACTION_ORDER,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    bet_q = student_output["bet_q_values"]
    auxiliary = batch.get("betting_auxiliary")
    if auxiliary is None:
        raise ValueError("betting_auxiliary is required when observed EV ranking is enabled")

    true_count_proxy = auxiliary["true_count_proxy"].to(device=bet_q.device, dtype=torch.float32)
    observed_cards = auxiliary["observed_cards"].to(device=bet_q.device, dtype=torch.long)
    bucket_idx = map_count_proxy_to_bucket(
        true_count_proxy,
        threshold_medium=config.threshold_medium,
        threshold_high=config.threshold_high,
        threshold_very_high=config.threshold_very_high,
    )
    valid = observed_cards >= int(config.min_observed_cards)
    if valid_rows is not None:
        valid = valid & valid_rows.to(device=bet_q.device, dtype=torch.bool)

    pairs_by_bucket = ev_table.get_preferred_pairs(
        min_samples=config.min_bucket_action_samples,
        min_ev_gap_per_round=config.min_ev_gap_per_round,
        compare_against_1x_only=config.compare_against_1x_only,
        allowed_actions=bet_action_names,
    )
    if not pairs_by_bucket:
        return bet_q.new_zeros(())

    losses: list[torch.Tensor] = []
    pair_count = 0
    for bucket_index, bucket_name in enumerate(COUNT_BUCKET_NAMES):
        pairs = pairs_by_bucket.get(bucket_name)
        if not pairs:
            continue
        rows = valid & (bucket_idx == bucket_index)
        if not bool(rows.any().item()):
            continue
        for good_action, bad_action, _ev_gap in pairs:
            if pair_count >= config.max_pairs_per_batch:
                break
            if good_action not in bet_action_names or bad_action not in bet_action_names:
                continue
            good_idx = bet_action_names.index(good_action)
            bad_idx = bet_action_names.index(bad_action)
            pair_loss = torch.relu(config.margin - (bet_q[..., good_idx][rows] - bet_q[..., bad_idx][rows]))
            losses.append(pair_loss.mean())
            pair_count += 1

    if not losses:
        return bet_q.new_zeros(())
    return torch.stack(losses).mean()
