from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from enviroment_bj.core import BET_ACTION_ORDER, CARD_RANKS
from model.agents.common import apply_action_mask

from .config import BettingAuxiliaryConfig


HI_LO_WEIGHTS = {
    "2": 1.0,
    "3": 1.0,
    "4": 1.0,
    "5": 1.0,
    "6": 1.0,
    "7": 0.0,
    "8": 0.0,
    "9": 0.0,
    "10": -1.0,
    "J": -1.0,
    "Q": -1.0,
    "K": -1.0,
    "A": -1.0,
}
HI_LO_VECTOR = torch.tensor([HI_LO_WEIGHTS[rank] for rank in CARD_RANKS], dtype=torch.float32)
TEN_SLOT_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10")
COUNT_PROXY_BUCKET_NAMES = ("low", "medium", "high", "very_high")


def _sequence_to_rank_count_mapping(values: Sequence[Any]) -> dict[str, float] | None:
    if len(values) == len(CARD_RANKS):
        return {rank: float(values[index]) for index, rank in enumerate(CARD_RANKS)}
    if len(values) == len(TEN_SLOT_RANKS):
        return {rank: float(values[index]) for index, rank in enumerate(TEN_SLOT_RANKS)}
    return None


def _cards_to_rank_counts(cards: Sequence[Any]) -> dict[str, float]:
    counts = {rank: 0.0 for rank in CARD_RANKS}
    for card in cards:
        normalized = str(card).strip().upper()
        if normalized in counts:
            counts[normalized] += 1.0
    return counts


def try_parse_rank_counts(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _sequence_to_rank_count_mapping(value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, Mapping):
        if "rank_counts" in value:
            parsed = try_parse_rank_counts(value.get("rank_counts"))
            if parsed is not None:
                return parsed
        if "cards" in value and isinstance(value.get("cards"), Sequence):
            return _cards_to_rank_counts(value.get("cards") or [])
        if "recent_cards" in value and isinstance(value.get("recent_cards"), Sequence):
            return _cards_to_rank_counts(value.get("recent_cards") or [])
        if any(rank in value for rank in CARD_RANKS):
            return {rank: float(value.get(rank, 0.0)) for rank in CARD_RANKS}
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _sequence_to_rank_count_mapping(list(value))
    return None


def try_parse_group_counts(value: Any) -> dict[str, float] | None:
    if value is None or not isinstance(value, Mapping):
        return None
    if "by_group" in value:
        parsed = try_parse_group_counts(value.get("by_group"))
        if parsed is not None:
            return parsed
    if all(name in value for name in ("low", "neutral", "high")):
        return {
            "low": float(value.get("low", 0.0)),
            "neutral": float(value.get("neutral", 0.0)),
            "high": float(value.get("high", 0.0)),
        }
    return None


def extract_rank_counts_from_response(response: Mapping[str, Any]) -> dict[str, float] | None:
    observation = response.get("observation") or {}
    candidate_keys = (
        "observed_cards_history",
        "observed_rank_counts",
        "observed_cards_rank_counts",
        "rank_counts",
    )
    for key in candidate_keys:
        parsed = try_parse_rank_counts(observation.get(key))
        if parsed is not None:
            return parsed
    parsed = try_parse_rank_counts(observation.get("discard_summary"))
    if parsed is not None:
        return parsed
    return None


def extract_group_counts_from_response(response: Mapping[str, Any]) -> dict[str, float] | None:
    observation = response.get("observation") or {}
    candidate_keys = (
        "observed_cards_history",
        "discard_summary",
    )
    for key in candidate_keys:
        parsed = try_parse_group_counts(observation.get(key))
        if parsed is not None:
            return parsed
    return None


def compute_observed_hi_lo_proxy_from_response(
    response: Mapping[str, Any],
    *,
    n_decks: int = 8,
) -> dict[str, float | int]:
    running_count = 0.0
    observed_cards = 0

    rank_counts = extract_rank_counts_from_response(response)
    if rank_counts is not None:
        running_count = sum(float(rank_counts.get(rank, 0.0)) * HI_LO_WEIGHTS[rank] for rank in CARD_RANKS)
        observed_cards = int(round(sum(float(rank_counts.get(rank, 0.0)) for rank in CARD_RANKS)))
    else:
        group_counts = extract_group_counts_from_response(response)
        if group_counts is not None:
            running_count = float(group_counts.get("low", 0.0)) - float(group_counts.get("high", 0.0))
            observed_cards = int(
                round(
                    float(group_counts.get("low", 0.0))
                    + float(group_counts.get("neutral", 0.0))
                    + float(group_counts.get("high", 0.0))
                )
            )

    total_cards = max(int(n_decks), 1) * 52
    estimated_cards_remaining = max(total_cards - observed_cards, 1)
    estimated_decks_seen = observed_cards / 52.0
    estimated_decks_remaining = estimated_cards_remaining / 52.0
    true_count_proxy = running_count / max(estimated_decks_remaining, 0.25)

    return {
        "running_count": float(running_count),
        "observed_cards": int(observed_cards),
        "estimated_decks_seen": float(estimated_decks_seen),
        "estimated_decks_remaining": float(estimated_decks_remaining),
        "true_count_proxy": float(true_count_proxy),
    }


def betting_auxiliary_weight(config: BettingAuxiliaryConfig, update_count: int) -> float:
    fraction = min(max(float(update_count), 0.0) / float(config.decay_steps), 1.0)
    return float(config.weight + fraction * (config.final_weight - config.weight))


def map_count_proxy_to_bet_target(
    true_count_proxy: torch.Tensor,
    *,
    threshold_2x: float,
    threshold_3x: float,
    threshold_4x: float,
) -> torch.Tensor:
    target = torch.zeros_like(true_count_proxy, dtype=torch.long)
    target = torch.where(true_count_proxy >= threshold_2x, torch.ones_like(target), target)
    target = torch.where(true_count_proxy >= threshold_3x, torch.full_like(target, 2), target)
    target = torch.where(true_count_proxy >= threshold_4x, torch.full_like(target, 3), target)
    return target


def clamp_target_to_legal_bet(target: torch.Tensor, bet_mask: torch.Tensor) -> torch.Tensor:
    flat_target = target.to(torch.long).reshape(-1)
    flat_mask = bet_mask.to(torch.bool).reshape(-1, bet_mask.shape[-1])
    resolved_target = flat_target.clone()

    for row_index in range(flat_mask.shape[0]):
        row_target = int(flat_target[row_index].item())
        row_mask = flat_mask[row_index]
        if row_mask[row_target]:
            continue
        legal_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
        if legal_indices.numel() == 0:
            resolved_target[row_index] = 0
            continue
        not_above_target = legal_indices[legal_indices <= row_target]
        resolved_target[row_index] = (
            int(not_above_target.max().item())
            if not_above_target.numel() > 0
            else int(legal_indices[0].item())
        )

    return resolved_target.reshape(target.shape)


def compute_betting_count_proxy_ce_loss(
    student_output: Mapping[str, torch.Tensor],
    action_mask: torch.Tensor,
    betting_auxiliary: Mapping[str, torch.Tensor],
    *,
    config: BettingAuxiliaryConfig,
    betting_action_slice: slice,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    q_values = student_output["q_values"]
    bet_q_values = q_values[..., betting_action_slice]
    bet_mask = action_mask[..., betting_action_slice].to(torch.bool)

    true_count_proxy = betting_auxiliary["true_count_proxy"].to(device=q_values.device, dtype=torch.float32)
    observed_cards = betting_auxiliary["observed_cards"].to(device=q_values.device, dtype=torch.long)

    target = map_count_proxy_to_bet_target(
        true_count_proxy,
        threshold_2x=config.threshold_2x,
        threshold_3x=config.threshold_3x,
        threshold_4x=config.threshold_4x,
    )
    target = clamp_target_to_legal_bet(target, bet_mask)

    valid = observed_cards >= int(config.min_observed_cards)
    valid = valid & bet_mask.any(dim=-1)
    if valid_rows is not None:
        valid = valid & valid_rows.to(device=q_values.device, dtype=torch.bool)

    if not bool(valid.any().item()):
        return q_values.new_zeros(())

    masked_bet_q = apply_action_mask(bet_q_values, bet_mask)
    return F.cross_entropy(masked_bet_q[valid], target[valid])


def count_proxy_bucket_name(true_count_proxy: float, config: BettingAuxiliaryConfig) -> str:
    if true_count_proxy < config.threshold_2x:
        return "low"
    if true_count_proxy < config.threshold_3x:
        return "medium"
    if true_count_proxy < config.threshold_4x:
        return "high"
    return "very_high"


@dataclass(slots=True)
class BettingAuxiliaryEvaluationTracker:
    proxy_values: list[float] = field(default_factory=list)
    target_counts: Counter[str] = field(default_factory=Counter)
    bucket_state_counts: Counter[str] = field(default_factory=Counter)
    bucket_q_totals: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    bucket_q_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    bucket_greedy_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    bucket_margin_totals: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    bucket_margin_counts: Counter[str] = field(default_factory=Counter)

    def record(
        self,
        *,
        q_values: torch.Tensor,
        action_mask: torch.Tensor,
        proxy_info: Mapping[str, float | int],
        config: BettingAuxiliaryConfig,
    ) -> None:
        observed_cards = int(proxy_info.get("observed_cards", 0))
        if observed_cards < int(config.min_observed_cards):
            return

        bet_q_values = q_values[..., : len(BET_ACTION_ORDER)]
        bet_mask = action_mask[..., : len(BET_ACTION_ORDER)].to(torch.bool)
        if not bool(bet_mask.any().item()):
            return

        true_count_proxy = float(proxy_info.get("true_count_proxy", 0.0))
        bucket = count_proxy_bucket_name(true_count_proxy, config)
        target = map_count_proxy_to_bet_target(
            torch.tensor([true_count_proxy], dtype=torch.float32, device=bet_q_values.device),
            threshold_2x=config.threshold_2x,
            threshold_3x=config.threshold_3x,
            threshold_4x=config.threshold_4x,
        )
        target = clamp_target_to_legal_bet(target, bet_mask.unsqueeze(0))
        target_action = BET_ACTION_ORDER[int(target.item())]

        self.proxy_values.append(true_count_proxy)
        self.target_counts[target_action] += 1
        self.bucket_state_counts[bucket] += 1

        masked_bet_q = apply_action_mask(bet_q_values.unsqueeze(0), bet_mask.unsqueeze(0)).squeeze(0)
        greedy_index = int(masked_bet_q.argmax(dim=-1).item())
        self.bucket_greedy_counts[bucket][BET_ACTION_ORDER[greedy_index]] += 1

        q_bet_1x: float | None = None
        best_aggressive_q: float | None = None
        for index, action_name in enumerate(BET_ACTION_ORDER):
            if not bool(bet_mask[index].item()):
                continue
            q_value = float(bet_q_values[index].item())
            self.bucket_q_totals[bucket][action_name] += q_value
            self.bucket_q_counts[bucket][action_name] += 1
            if action_name == "bet_1x":
                q_bet_1x = q_value
            elif best_aggressive_q is None or q_value > best_aggressive_q:
                best_aggressive_q = q_value

        if q_bet_1x is not None and best_aggressive_q is not None:
            self.bucket_margin_totals[bucket] += best_aggressive_q - q_bet_1x
            self.bucket_margin_counts[bucket] += 1

    def summary(self) -> dict[str, Any]:
        total_states = len(self.proxy_values)
        if total_states <= 0:
            return {
                "count_proxy_valid_states": 0.0,
                "count_proxy_mean": 0.0,
                "count_proxy_p10": 0.0,
                "count_proxy_p50": 0.0,
                "count_proxy_p90": 0.0,
                "count_proxy_target_bet_distribution": {action: 0.0 for action in BET_ACTION_ORDER},
                "greedy_bet_distribution_by_count_bucket": {},
                "mean_margin_by_count_bucket": {},
                "count_proxy_bucket_stats": {},
            }

        proxy_tensor = torch.tensor(self.proxy_values, dtype=torch.float32)
        bucket_stats: dict[str, dict[str, Any]] = {}
        greedy_distributions: dict[str, dict[str, float]] = {}
        margin_by_bucket: dict[str, float | None] = {}

        for bucket in COUNT_PROXY_BUCKET_NAMES:
            bucket_count = int(self.bucket_state_counts[bucket])
            if bucket_count <= 0:
                continue
            stats: dict[str, Any] = {"n_states": float(bucket_count)}
            for action_name in BET_ACTION_ORDER:
                action_count = int(self.bucket_q_counts[bucket][action_name])
                stats[f"mean_q_{action_name}"] = (
                    self.bucket_q_totals[bucket][action_name] / action_count
                    if action_count > 0
                    else None
                )
                stats[f"greedy_{action_name}_frac"] = self.bucket_greedy_counts[bucket][action_name] / bucket_count

            margin_count = int(self.bucket_margin_counts[bucket])
            margin_value = (
                self.bucket_margin_totals[bucket] / margin_count
                if margin_count > 0
                else None
            )
            stats["mean_margin_best_aggressive_vs_1x"] = margin_value
            bucket_stats[bucket] = stats
            greedy_distributions[bucket] = {
                action_name: self.bucket_greedy_counts[bucket][action_name] / bucket_count
                for action_name in BET_ACTION_ORDER
            }
            margin_by_bucket[bucket] = margin_value

        return {
            "count_proxy_valid_states": float(total_states),
            "count_proxy_mean": float(proxy_tensor.mean().item()),
            "count_proxy_p10": float(torch.quantile(proxy_tensor, 0.10).item()),
            "count_proxy_p50": float(torch.quantile(proxy_tensor, 0.50).item()),
            "count_proxy_p90": float(torch.quantile(proxy_tensor, 0.90).item()),
            "count_proxy_target_bet_distribution": {
                action_name: self.target_counts[action_name] / total_states
                for action_name in BET_ACTION_ORDER
            },
            "greedy_bet_distribution_by_count_bucket": greedy_distributions,
            "mean_margin_by_count_bucket": margin_by_bucket,
            "count_proxy_bucket_stats": bucket_stats,
        }
