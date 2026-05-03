from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from torch import nn

from loss import compute_td_loss

from .betting_auxiliary import betting_auxiliary_weight, compute_betting_count_proxy_ce_loss
from .config import (
    BettingAuxiliaryConfig,
    CountAuxiliaryConfig,
    DistillationConfig,
    ObservedEVRankingConfig,
    OptimizationConfig,
    TargetUpdateConfig,
)
from .count_auxiliary import compute_count_bucket_ce_loss, count_auxiliary_weight
from .ev_calibration import EVBucketActionTable, compute_observed_ev_ranking_loss, observed_ev_ranking_weight
from .transfer_learning import compute_distillation_loss, distillation_weight


def _infer_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_training_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_training_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_training_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_training_batch_to_device(value, device) for value in batch)
    return batch


def build_optimizer(model: nn.Module, config: OptimizationConfig) -> torch.optim.Optimizer:
    optimizer_cls = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    return optimizer_cls(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if config.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )


def compute_gradient_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def hard_update_target(online_network: nn.Module, target_network: nn.Module) -> None:
    target_network.load_state_dict(online_network.state_dict())


def soft_update_target(online_network: nn.Module, target_network: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target_network.parameters(), online_network.parameters()):
            target_parameter.data.mul_(1.0 - tau)
            target_parameter.data.add_(tau * online_parameter.data)


def maybe_update_target(
    online_network: nn.Module,
    target_network: nn.Module,
    update_count: int,
    config: TargetUpdateConfig,
) -> bool:
    if config.mode == "soft":
        soft_update_target(online_network, target_network, config.soft_tau)
        return True
    if update_count % config.hard_update_interval == 0:
        hard_update_target(online_network, target_network)
        return True
    return False


def train_gradient_step(
    *,
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    loss_config: Any,
    optimization_config: OptimizationConfig,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    teacher_model: nn.Module | None = None,
    distillation_config: DistillationConfig | None = None,
    betting_auxiliary_config: BettingAuxiliaryConfig | None = None,
    count_auxiliary_config: CountAuxiliaryConfig | None = None,
    observed_ev_ranking_config: ObservedEVRankingConfig | None = None,
    ev_bucket_action_table: EVBucketActionTable | None = None,
    update_count: int = 0,
) -> dict[str, Any]:
    start_time = perf_counter()
    online_network.train()
    target_network.eval()
    batch = move_training_batch_to_device(batch, _infer_model_device(online_network))

    optimizer.zero_grad(set_to_none=True)
    loss_info = compute_td_loss(
        online_network,
        target_network,
        batch,
        gamma=loss_config.gamma,
        loss_type=loss_config.loss_type,
        config=loss_config,
    )
    td_loss = loss_info["loss"]
    total_loss = td_loss
    distill_loss = td_loss.new_zeros(())
    lambda_distill = 0.0
    bet_aux_loss = td_loss.new_zeros(())
    lambda_bet_aux = 0.0
    count_aux_loss = td_loss.new_zeros(())
    lambda_count_aux = 0.0
    ev_rank_loss = td_loss.new_zeros(())
    lambda_ev_rank = 0.0
    teacher_output: dict[str, Any] | None = None
    if teacher_model is not None and distillation_config is not None and distillation_config.enabled:
        teacher_state = batch.get("teacher_state")
        if teacher_state is None:
            raise ValueError("teacher_state is required in the batch when distillation is enabled")
        teacher_model.eval()
        with torch.no_grad():
            teacher_output = teacher_model(teacher_state)
        lambda_distill = distillation_weight(distillation_config, update_count)
        distill_loss = compute_distillation_loss(
            student_output=loss_info["current_output"],
            teacher_output=teacher_output,
            action_mask=batch["action_mask"],
            config=distillation_config,
            playing_action_slice=online_network.play_action_slice,
            valid_rows=batch.get("padding_mask"),
        )
        total_loss = td_loss + (distill_loss * lambda_distill)

    if betting_auxiliary_config is not None and betting_auxiliary_config.enabled:
        betting_auxiliary = batch.get("betting_auxiliary")
        if betting_auxiliary is None:
            raise ValueError("betting_auxiliary is required in the batch when betting auxiliary loss is enabled")
        lambda_bet_aux = betting_auxiliary_weight(betting_auxiliary_config, update_count)
        bet_aux_loss = compute_betting_count_proxy_ce_loss(
            student_output=loss_info["current_output"],
            action_mask=batch["action_mask"],
            betting_auxiliary=betting_auxiliary,
            config=betting_auxiliary_config,
            betting_action_slice=online_network.bet_action_slice,
            valid_rows=batch.get("padding_mask"),
        )
        total_loss = total_loss + (bet_aux_loss * lambda_bet_aux)

    if count_auxiliary_config is not None and count_auxiliary_config.enabled:
        count_auxiliary = batch.get("betting_auxiliary")
        if count_auxiliary is None:
            raise ValueError("betting_auxiliary is required in the batch when count auxiliary loss is enabled")
        lambda_count_aux = count_auxiliary_weight(count_auxiliary_config, update_count)
        count_aux_loss = compute_count_bucket_ce_loss(
            student_output=loss_info["current_output"],
            count_auxiliary=count_auxiliary,
            config=count_auxiliary_config,
            valid_rows=batch.get("padding_mask"),
        )
        total_loss = total_loss + (count_aux_loss * lambda_count_aux)

    if observed_ev_ranking_config is not None and observed_ev_ranking_config.enabled:
        if ev_bucket_action_table is None:
            raise ValueError("ev_bucket_action_table is required when observed EV ranking is enabled")
        lambda_ev_rank = observed_ev_ranking_weight(observed_ev_ranking_config, update_count)
        ev_rank_loss = compute_observed_ev_ranking_loss(
            student_output=loss_info["current_output"],
            batch=batch,
            config=observed_ev_ranking_config,
            ev_table=ev_bucket_action_table,
            bet_action_names=tuple(online_network.bet_action_names),
            valid_rows=batch.get("padding_mask"),
        )
        total_loss = total_loss + (ev_rank_loss * lambda_ev_rank)

    total_loss.backward()

    if optimization_config.gradient_clipping:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), optimization_config.max_grad_norm).item()
        )
    else:
        grad_norm = float(compute_gradient_norm(online_network.parameters()))

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    elapsed = perf_counter() - start_time
    learning_rate = float(optimizer.param_groups[0]["lr"])
    metrics = dict(loss_info["metrics"])
    metrics.update(
        {
            "td_loss": float(td_loss.detach().item()),
            "distillation_loss": float(distill_loss.detach().item()),
            "distillation_weight": float(lambda_distill),
            "bet_aux_loss": float(bet_aux_loss.detach().item()),
            "bet_aux_weight": float(lambda_bet_aux),
            "count_aux_loss": float(count_aux_loss.detach().item()),
            "count_aux_weight": float(lambda_count_aux),
            "ev_rank_loss": float(ev_rank_loss.detach().item()),
            "ev_rank_weight": float(lambda_ev_rank),
            "total_loss": float(total_loss.detach().item()),
            "loss": float(total_loss.detach().item()),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "update_time_sec": elapsed,
        }
    )

    return {
        **loss_info,
        "loss": total_loss,
        "td_loss": td_loss,
        "distillation_loss": distill_loss,
        "bet_aux_loss": bet_aux_loss,
        "count_aux_loss": count_aux_loss,
        "ev_rank_loss": ev_rank_loss,
        "teacher_output": teacher_output,
        "metrics": metrics,
    }
