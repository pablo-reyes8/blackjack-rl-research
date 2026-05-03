from __future__ import annotations

import torch


COUNT_BUCKET_NAMES = ("low", "medium", "high", "very_high")


def map_count_proxy_to_bucket(
    true_count_proxy: torch.Tensor,
    *,
    threshold_medium: float,
    threshold_high: float,
    threshold_very_high: float,
) -> torch.Tensor:
    bucket = torch.zeros_like(true_count_proxy, dtype=torch.long)
    bucket = torch.where(true_count_proxy >= threshold_medium, torch.ones_like(bucket), bucket)
    bucket = torch.where(true_count_proxy >= threshold_high, torch.full_like(bucket, 2), bucket)
    bucket = torch.where(true_count_proxy >= threshold_very_high, torch.full_like(bucket, 3), bucket)
    return bucket


def count_proxy_bucket_name(
    true_count_proxy: float,
    *,
    threshold_medium: float,
    threshold_high: float,
    threshold_very_high: float,
) -> str:
    if true_count_proxy < threshold_medium:
        return "low"
    if true_count_proxy < threshold_high:
        return "medium"
    if true_count_proxy < threshold_very_high:
        return "high"
    return "very_high"
