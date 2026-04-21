from __future__ import annotations

from typing import Any, Mapping

import torch

from .common import (
    build_metrics,
    build_phase_metrics,
    build_phase_weight_tensor,
    build_td_criterion,
    compute_double_dqn_next_values,
    gather_q_values,
    resolve_n_steps_tensor,
    resolve_mask,
    validate_action_shapes,
    validate_current_action_legality,
)
from .config import BellmanLossConfig


def compute_double_dqn_targets_feedforward(
    online_network: Any,
    target_network: Any,
    batch: Mapping[str, Any],
    gamma: float,
    config: BellmanLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    loss_config = config or BellmanLossConfig(gamma=gamma)
    reward = batch["reward"].to(torch.float32)
    done = batch["done"]
    n_steps = resolve_n_steps_tensor(batch, reward)

    with torch.no_grad():
        next_online_output = online_network(batch["next_state"])
        next_target_output = target_network(batch["next_state"])
        next_action_mask = resolve_mask(batch, "next_action_mask", next_online_output)
        target_info = compute_double_dqn_next_values(
            next_online_q_values=next_online_output["q_values"],
            next_target_q_values=next_target_output["q_values"],
            next_action_mask=next_action_mask,
            done=done,
            config=loss_config,
        )
        discount = torch.pow(torch.full_like(n_steps, loss_config.gamma), n_steps)
        target = reward + (discount * target_info["next_q"])

    return {
        "target": target,
        "next_action": target_info["next_action"],
        "next_q": target_info["next_q"],
        "next_has_legal": target_info["next_has_legal"],
        "bootstrap_mask": target_info["bootstrap_mask"],
        "n_steps": n_steps,
    }


def compute_td_loss_feedforward(
    online_network: Any,
    target_network: Any,
    batch: Mapping[str, Any],
    gamma: float,
    loss_type: str = "huber",
    config: BellmanLossConfig | None = None,
) -> dict[str, Any]:
    loss_config = config or BellmanLossConfig(gamma=gamma, loss_type=loss_type)
    criterion = build_td_criterion(loss_config.loss_type)
    action = batch["action"]
    reward = batch["reward"].to(torch.float32)
    done = batch["done"]

    current_output = online_network(batch["state"])
    action_mask = resolve_mask(batch, "action_mask", current_output)
    validate_action_shapes(action, reward, done, action_mask)

    if loss_config.validate_current_actions:
        validate_current_action_legality(action, action_mask)

    q_pred = gather_q_values(current_output["q_values"], action)
    target_info = compute_double_dqn_targets_feedforward(
        online_network=online_network,
        target_network=target_network,
        batch=batch,
        gamma=loss_config.gamma,
        config=loss_config,
    )
    target = target_info["target"]
    td_error = q_pred - target
    loss_per_sample = criterion(q_pred, target)
    phase_weights = build_phase_weight_tensor(action, loss_config)
    if loss_config.phase_weights.enabled:
        weighted_loss_per_sample = loss_per_sample * phase_weights
        loss = weighted_loss_per_sample.mean()
    else:
        weighted_loss_per_sample = loss_per_sample
        loss = loss_per_sample.mean()

    metrics = build_metrics(
        loss=loss,
        q_pred=q_pred,
        target=target,
        td_error=td_error,
        reward=reward,
        done=done,
        next_has_legal=target_info["next_has_legal"],
        valid_rows=torch.ones_like(done, dtype=torch.bool),
    )
    metrics.update(
        build_phase_metrics(
            action=action,
            loss_values=loss_per_sample,
            td_error=td_error,
            valid_rows=torch.ones_like(done, dtype=torch.bool),
        )
    )
    metrics["mean_n_steps"] = float(target_info["n_steps"].detach().mean().item())
    metrics["mean_phase_weight"] = float(phase_weights.detach().mean().item())

    return {
        "loss": loss,
        "loss_per_sample": loss_per_sample,
        "weighted_loss_per_sample": weighted_loss_per_sample,
        "q_pred": q_pred,
        "target": target,
        "td_error": td_error,
        "next_action": target_info["next_action"],
        "next_q": target_info["next_q"],
        "next_has_legal": target_info["next_has_legal"],
        "current_output": current_output,
        "metrics": metrics,
    }
