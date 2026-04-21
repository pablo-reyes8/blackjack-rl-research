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


def compute_double_dqn_targets_recurrent(
    online_network: Any,
    target_network: Any,
    batch: Mapping[str, Any],
    gamma: float,
    config: BellmanLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    loss_config = config or BellmanLossConfig(gamma=gamma)
    reward = batch["reward"].to(torch.float32)
    done = batch["done"]
    padding_mask = batch["padding_mask"]
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
            valid_rows=padding_mask,
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


def compute_td_loss_recurrent(
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
    padding_mask = batch["padding_mask"]

    current_output = online_network(batch["state"])
    action_mask = resolve_mask(batch, "action_mask", current_output)
    validate_action_shapes(action, reward, done, action_mask)

    if padding_mask.shape != action.shape:
        raise ValueError("padding_mask must have the same shape as action, reward, and done")
    if not padding_mask.any():
        raise ValueError("padding_mask does not contain any valid steps")

    if loss_config.validate_current_actions:
        validate_current_action_legality(action, action_mask, valid_rows=padding_mask)

    q_pred = gather_q_values(current_output["q_values"], action)
    target_info = compute_double_dqn_targets_recurrent(
        online_network=online_network,
        target_network=target_network,
        batch=batch,
        gamma=loss_config.gamma,
        config=loss_config,
    )
    target = target_info["target"]
    td_error = q_pred - target
    loss_per_timestep = criterion(q_pred, target)

    valid_steps = padding_mask.to(loss_per_timestep.dtype)
    num_valid_steps = valid_steps.sum()
    if num_valid_steps.item() <= 0:
        raise ValueError("No valid timesteps are available for recurrent TD loss")
    phase_weights = build_phase_weight_tensor(action, loss_config)
    if loss_config.phase_weights.enabled:
        weighted_loss_per_timestep = loss_per_timestep * phase_weights
        loss = (weighted_loss_per_timestep * valid_steps).sum() / num_valid_steps
    else:
        weighted_loss_per_timestep = loss_per_timestep
        loss = (loss_per_timestep * valid_steps).sum() / num_valid_steps

    metrics = build_metrics(
        loss=loss,
        q_pred=q_pred,
        target=target,
        td_error=td_error,
        reward=reward,
        done=done,
        next_has_legal=target_info["next_has_legal"],
        valid_rows=padding_mask,
    )
    metrics.update(
        build_phase_metrics(
            action=action,
            loss_values=loss_per_timestep,
            td_error=td_error,
            valid_rows=padding_mask,
        )
    )
    metrics["mean_n_steps"] = float(target_info["n_steps"][padding_mask].detach().mean().item())
    metrics["mean_phase_weight"] = float(phase_weights[padding_mask].detach().mean().item())
    metrics["num_valid_steps"] = float(num_valid_steps.item())

    return {
        "loss": loss,
        "loss_per_timestep": loss_per_timestep,
        "weighted_loss_per_timestep": weighted_loss_per_timestep,
        "q_pred": q_pred,
        "target": target,
        "td_error": td_error,
        "next_action": target_info["next_action"],
        "next_q": target_info["next_q"],
        "next_has_legal": target_info["next_has_legal"],
        "num_valid_steps": num_valid_steps,
        "current_output": current_output,
        "metrics": metrics,
    }
