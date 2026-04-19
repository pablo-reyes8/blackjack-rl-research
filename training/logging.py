from __future__ import annotations

from typing import Any

from .config import PrintConfig


class TrainingLogger:
    def __init__(self, config: PrintConfig) -> None:
        self.config = config

    def _print(self, message: str) -> None:
        if self.config.enable:
            print(message)

    def log_warmup(self, *, buffer_size: int, target_size: int) -> None:
        self._print(f"[Warmup] buffer={buffer_size}/{target_size}")

    def log_update(self, *, epoch: int, env_steps: int, update: int, metrics: dict[str, Any]) -> None:
        if not self.config.enable or update % self.config.print_update_interval != 0:
            return
        self._print(
            "[Update] "
            f"epoch={epoch} env_steps={env_steps} update={update} "
            f"loss={metrics.get('loss', 0.0):.6f} "
            f"q={metrics.get('mean_q_pred', 0.0):.4f} "
            f"target={metrics.get('mean_target', 0.0):.4f} "
            f"td={metrics.get('mean_abs_td_error', 0.0):.4f} "
            f"grad_norm={metrics.get('grad_norm', 0.0):.4f} "
            f"epsilon={metrics.get('epsilon', 0.0):.4f} "
            f"buffer={int(metrics.get('buffer_size', 0))}"
        )

    def log_collection(self, *, epoch: int, env_steps: int, metrics: dict[str, Any]) -> None:
        if not self.config.enable or env_steps % self.config.print_collection_interval != 0:
            return
        self._print(
            "[Collect] "
            f"epoch={epoch} env_steps={env_steps} rounds={int(metrics.get('rounds_completed', 0))} "
            f"hands={int(metrics.get('hands_completed', 0))} "
            f"reward/round={metrics.get('reward_per_round', 0.0):.4f} "
            f"EV/1000={metrics.get('ev_per_1000_hands', 0.0):.2f}"
        )

    def log_epoch_summary(self, *, epoch: int, summary: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_epoch_summary:
            return
        self._print(
            "[Epoch] "
            f"epoch={epoch} env_steps={int(summary.get('env_steps', 0))} "
            f"updates={int(summary.get('updates', 0))} "
            f"buffer={int(summary.get('buffer_size', 0))} "
            f"epsilon={summary.get('epsilon', 0.0):.4f} "
            f"loss={summary.get('loss', 0.0):.6f} "
            f"grad_norm={summary.get('grad_norm', 0.0):.4f} "
            f"reward/round={summary.get('reward_per_round', 0.0):.4f} "
            f"EV/1000={summary.get('ev_per_1000_hands', 0.0):.2f}"
        )
        if self.config.include_segment_details:
            self._print(f"[Epoch Details] situations={summary.get('situation_counts', {})}")
            self._print(f"[Epoch Details] actions={summary.get('action_frequencies', {})}")

    def log_evaluation(self, *, epoch: int, metrics: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_eval_summary:
            return
        self._print(
            "[Eval] "
            f"epoch={epoch} rounds={int(metrics.get('rounds_completed', 0))} "
            f"reward/round={metrics.get('reward_per_round', 0.0):.4f} "
            f"EV/1000={metrics.get('ev_per_1000_hands', 0.0):.2f} "
            f"win={metrics.get('win_rate', 0.0):.4f} "
            f"push={metrics.get('push_rate', 0.0):.4f} "
            f"loss={metrics.get('loss_rate', 0.0):.4f}"
        )

    def log_checkpoint(self, *, kind: str, path: str) -> None:
        self._print(f"[Checkpoint] kind={kind} path={path}")
