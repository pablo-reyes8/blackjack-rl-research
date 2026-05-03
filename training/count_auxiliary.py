from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .config import CountAuxiliaryConfig
from .count_utils import COUNT_BUCKET_NAMES as COUNT_PROXY_BUCKET_NAMES, map_count_proxy_to_bucket


def count_auxiliary_weight(config: CountAuxiliaryConfig, update_count: int) -> float:
    fraction = min(max(float(update_count), 0.0) / float(config.decay_steps), 1.0)
    return float(config.weight + fraction * (config.final_weight - config.weight))


def map_count_proxy_to_count_bucket(
    true_count_proxy: torch.Tensor,
    *,
    threshold_medium: float,
    threshold_high: float,
    threshold_very_high: float,
) -> torch.Tensor:
    return map_count_proxy_to_bucket(
        true_count_proxy,
        threshold_medium=threshold_medium,
        threshold_high=threshold_high,
        threshold_very_high=threshold_very_high,
    )


def compute_count_bucket_ce_loss(
    student_output: Mapping[str, torch.Tensor],
    count_auxiliary: Mapping[str, torch.Tensor],
    *,
    config: CountAuxiliaryConfig,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = student_output.get("count_bucket_logits")
    if logits is None:
        raise ValueError("count_bucket_logits missing from model output while count auxiliary is enabled")

    true_count_proxy = count_auxiliary["true_count_proxy"].to(device=logits.device, dtype=torch.float32)
    observed_cards = count_auxiliary["observed_cards"].to(device=logits.device, dtype=torch.long)

    target = map_count_proxy_to_count_bucket(
        true_count_proxy,
        threshold_medium=config.threshold_medium,
        threshold_high=config.threshold_high,
        threshold_very_high=config.threshold_very_high,
    )

    valid = observed_cards >= int(config.min_observed_cards)
    if valid_rows is not None:
        valid = valid & valid_rows.to(device=logits.device, dtype=torch.bool)

    if not bool(valid.any().item()):
        return logits.new_zeros(())

    class_weight_tensor = None
    if config.class_weights is not None:
        class_weight_tensor = torch.tensor(
            config.class_weights,
            dtype=logits.dtype,
            device=logits.device,
        )

    return F.cross_entropy(logits[valid], target[valid], weight=class_weight_tensor)


@dataclass(slots=True)
class CountAuxiliaryEvaluationTracker:
    target_counts: Counter[str] = field(default_factory=Counter)
    pred_counts: Counter[str] = field(default_factory=Counter)
    confusion_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {bucket: Counter() for bucket in COUNT_PROXY_BUCKET_NAMES}
    )
    valid_states: int = 0
    correct_predictions: int = 0

    def record(
        self,
        *,
        count_bucket_logits: torch.Tensor | None,
        proxy_info: Mapping[str, float | int],
        config: CountAuxiliaryConfig,
    ) -> None:
        if count_bucket_logits is None:
            return

        observed_cards = int(proxy_info.get("observed_cards", 0))
        if observed_cards < int(config.min_observed_cards):
            return

        target_index = int(
            map_count_proxy_to_count_bucket(
                torch.tensor([float(proxy_info.get("true_count_proxy", 0.0))], dtype=torch.float32, device=count_bucket_logits.device),
                threshold_medium=config.threshold_medium,
                threshold_high=config.threshold_high,
                threshold_very_high=config.threshold_very_high,
            ).item()
        )
        pred_index = int(count_bucket_logits.argmax(dim=-1).item())
        target_name = COUNT_PROXY_BUCKET_NAMES[target_index]
        pred_name = COUNT_PROXY_BUCKET_NAMES[pred_index]

        self.valid_states += 1
        self.correct_predictions += int(target_index == pred_index)
        self.target_counts[target_name] += 1
        self.pred_counts[pred_name] += 1
        self.confusion_counts[target_name][pred_name] += 1

    def summary(self) -> dict[str, Any]:
        if self.valid_states <= 0:
            return {
                "count_aux_valid_states": 0.0,
                "count_aux_accuracy": 0.0,
                "count_aux_target_distribution": {bucket: 0.0 for bucket in COUNT_PROXY_BUCKET_NAMES},
                "count_aux_pred_distribution": {bucket: 0.0 for bucket in COUNT_PROXY_BUCKET_NAMES},
                "count_aux_confusion_matrix": {
                    target: {pred: 0.0 for pred in COUNT_PROXY_BUCKET_NAMES}
                    for target in COUNT_PROXY_BUCKET_NAMES
                },
            }

        return {
            "count_aux_valid_states": float(self.valid_states),
            "count_aux_accuracy": float(self.correct_predictions) / float(self.valid_states),
            "count_aux_target_distribution": {
                bucket: float(self.target_counts[bucket]) / float(self.valid_states)
                for bucket in COUNT_PROXY_BUCKET_NAMES
            },
            "count_aux_pred_distribution": {
                bucket: float(self.pred_counts[bucket]) / float(self.valid_states)
                for bucket in COUNT_PROXY_BUCKET_NAMES
            },
            "count_aux_confusion_matrix": {
                target: {
                    pred: float(self.confusion_counts[target][pred]) / max(float(self.target_counts[target]), 1.0)
                    for pred in COUNT_PROXY_BUCKET_NAMES
                }
                for target in COUNT_PROXY_BUCKET_NAMES
            },
        }
