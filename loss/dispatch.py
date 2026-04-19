from __future__ import annotations

from typing import Any, Mapping

from .config import BellmanLossConfig
from .feedforward import compute_double_dqn_targets_feedforward, compute_td_loss_feedforward
from .recurrent import compute_double_dqn_targets_recurrent, compute_td_loss_recurrent


def compute_double_dqn_targets(
    online_network: Any,
    target_network: Any,
    batch: Mapping[str, Any],
    gamma: float,
    config: BellmanLossConfig | None = None,
) -> dict[str, Any]:
    action = batch["action"]
    if action.ndim == 1:
        return compute_double_dqn_targets_feedforward(online_network, target_network, batch, gamma, config=config)
    if action.ndim == 2:
        return compute_double_dqn_targets_recurrent(online_network, target_network, batch, gamma, config=config)
    raise ValueError("action tensor must be rank 1 (feedforward) or rank 2 (recurrent)")


def compute_td_loss(
    online_network: Any,
    target_network: Any,
    batch: Mapping[str, Any],
    gamma: float,
    loss_type: str = "huber",
    config: BellmanLossConfig | None = None,
) -> dict[str, Any]:
    action = batch["action"]
    if action.ndim == 1:
        return compute_td_loss_feedforward(
            online_network,
            target_network,
            batch,
            gamma,
            loss_type=loss_type,
            config=config,
        )
    if action.ndim == 2:
        return compute_td_loss_recurrent(
            online_network,
            target_network,
            batch,
            gamma,
            loss_type=loss_type,
            config=config,
        )
    raise ValueError("action tensor must be rank 1 (feedforward) or rank 2 (recurrent)")
