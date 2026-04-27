from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from torch import nn

from loss import compute_td_loss

from .config import DistillationConfig, OptimizationConfig, TargetUpdateConfig
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
        "teacher_output": teacher_output,
        "metrics": metrics,
    }
