from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from model.agents.common import apply_action_mask

from .config import BellmanLossConfig


def build_td_criterion(loss_type: str) -> nn.Module:
    if loss_type == "huber":
        return nn.SmoothL1Loss(reduction="none")
    if loss_type == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"Unsupported loss type: {loss_type}")


def ensure_bool_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype.is_floating_point or mask.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        return mask != 0
    raise TypeError(f"{name} must be a bool-compatible tensor")


def validate_action_shapes(
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    action_mask: torch.Tensor,
    next_action_mask: torch.Tensor | None = None,
) -> None:
    if reward.shape != done.shape:
        raise ValueError("reward and done must have the same shape")
    if action.shape != reward.shape:
        raise ValueError("action, reward, and done must have the same shape")
    if action_mask.shape[:-1] != action.shape:
        raise ValueError("action_mask must have shape action.shape + [num_actions]")
    if next_action_mask is not None and next_action_mask.shape[:-1] != action.shape:
        raise ValueError("next_action_mask must have shape action.shape + [num_actions]")


def gather_q_values(q_values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return q_values.gather(dim=-1, index=action.long().unsqueeze(-1)).squeeze(-1)


def validate_current_action_legality(
    action: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    valid_rows: torch.Tensor | None = None,
) -> None:
    selected_is_legal = gather_q_values(action_mask.to(torch.int64), action).to(torch.bool)
    if valid_rows is not None:
        illegal_rows = valid_rows & ~selected_is_legal
    else:
        illegal_rows = ~selected_is_legal
    if illegal_rows.any():
        raise ValueError("The batch contains actions that are illegal under the provided action_mask")


def compute_double_dqn_next_values(
    *,
    next_online_q_values: torch.Tensor,
    next_target_q_values: torch.Tensor,
    next_action_mask: torch.Tensor,
    done: torch.Tensor,
    config: BellmanLossConfig,
    valid_rows: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    next_action_mask = ensure_bool_mask(next_action_mask, name="next_action_mask")
    done_mask = ensure_bool_mask(done, name="done")
    if valid_rows is None:
        valid_rows = torch.ones_like(done_mask, dtype=torch.bool)
    else:
        valid_rows = ensure_bool_mask(valid_rows, name="valid_rows")

    next_has_legal = next_action_mask.any(dim=-1)
    nonterminal_rows = valid_rows & ~done_mask
    illegal_nonterminal = nonterminal_rows & ~next_has_legal

    if config.validate_next_action_mask and illegal_nonterminal.any():
        raise ValueError("Non-terminal next states must have at least one legal action in next_action_mask")

    if config.validate_next_action_mask and not config.allow_terminal_without_legal_next_action:
        illegal_terminal = valid_rows & done_mask & ~next_has_legal
        if illegal_terminal.any():
            raise ValueError("Terminal rows without legal next actions are not allowed by this config")

    next_online_masked_q = apply_action_mask(next_online_q_values, next_action_mask)
    next_action = next_online_masked_q.argmax(dim=-1)
    next_q = gather_q_values(next_target_q_values, next_action)
    bootstrap_mask = (valid_rows & ~done_mask).to(next_q.dtype)
    next_q = next_q * bootstrap_mask
    return {
        "next_action": next_action,
        "next_q": next_q,
        "bootstrap_mask": bootstrap_mask,
        "next_has_legal": next_has_legal,
        "masked_next_online_q": next_online_masked_q,
    }


def build_metrics(
    *,
    loss: torch.Tensor,
    q_pred: torch.Tensor,
    target: torch.Tensor,
    td_error: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    next_has_legal: torch.Tensor,
    valid_rows: torch.Tensor,
) -> dict[str, float]:
    valid_rows = ensure_bool_mask(valid_rows, name="valid_rows")
    if not valid_rows.any():
        raise ValueError("No valid rows are available to compute metrics")

    valid_q_pred = q_pred[valid_rows]
    valid_target = target[valid_rows]
    valid_td_error = td_error[valid_rows]
    valid_reward = reward[valid_rows]
    valid_done = ensure_bool_mask(done, name="done")[valid_rows]
    valid_next_has_legal = ensure_bool_mask(next_has_legal, name="next_has_legal")[valid_rows]

    return {
        "loss": float(loss.detach().item()),
        "mean_q_pred": float(valid_q_pred.detach().mean().item()),
        "mean_target": float(valid_target.detach().mean().item()),
        "mean_reward": float(valid_reward.detach().mean().item()),
        "mean_abs_td_error": float(valid_td_error.detach().abs().mean().item()),
        "max_abs_td_error": float(valid_td_error.detach().abs().max().item()),
        "terminal_fraction": float(valid_done.float().mean().item()),
        "next_legal_fraction": float(valid_next_has_legal.float().mean().item()),
        "num_valid_rows": float(valid_rows.sum().item()),
    }


def resolve_mask(batch: Mapping[str, Any], key: str, fallback_output: Mapping[str, Any] | None = None) -> torch.Tensor:
    if key in batch:
        return ensure_bool_mask(batch[key], name=key)
    if fallback_output is None or "action_mask" not in fallback_output:
        raise KeyError(f"Batch is missing required key '{key}' and no fallback mask is available")
    return ensure_bool_mask(fallback_output["action_mask"], name=key)
