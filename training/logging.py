from __future__ import annotations

from math import floor
from typing import Any

from .config import PrintConfig


class TrainingLogger:
    SEPARATOR = "-" * 92

    def __init__(self, config: PrintConfig) -> None:
        self.config = config
        self._last_warmup_bucket = 0
        self._last_warmup_size = 0

    def _print(self, message: str) -> None:
        if self.config.enable:
            print(message)

    def log_warmup(self, *, buffer_size: int, target_size: int, force: bool = False) -> None:
        if not self.config.enable:
            return
        bucket = buffer_size // self.config.print_warmup_interval
        should_print = force or buffer_size >= target_size or (bucket > self._last_warmup_bucket and buffer_size > 0)
        if should_print and buffer_size == self._last_warmup_size:
            should_print = False
        if not should_print:
            return
        self._last_warmup_bucket = bucket
        self._last_warmup_size = buffer_size
        self._print(f"[Warmup] buffer {buffer_size}/{target_size}")

    def log_epoch_start(self, *, epoch: int, total_epochs: int) -> None:
        if not self.config.enable or not self.config.print_epoch_header:
            return
        self._print(f"\n=== Epoch {epoch}/{total_epochs} ===")

    def log_run_summary(self, *, summary: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_run_summary:
            return

        self._print(self.SEPARATOR)
        self._print(
            "Blackjack RL run | "
            f"arch: {summary.get('architecture')} | "
            f"recurrent: {summary.get('recurrent_type')} | "
            f"encoder: {summary.get('encoder_profile')} | "
            f"obs: {summary.get('observation_profile')} | "
            f"start: {summary.get('start_state_mode')}"
        )
        self._print(
            "Device: "
            f"{summary.get('device')} | "
            f"epochs: {int(summary.get('total_epochs', 0))} | "
            f"envs: {int(summary.get('num_envs', 0))} | "
            f"steps/epoch: {int(summary.get('nominal_env_steps_per_epoch', 0))} | "
            f"updates/epoch~: {int(summary.get('estimated_updates_per_epoch', 0))} | "
            f"params: {int(summary.get('parameter_count', 0)):,}"
        )
        self._print(
            "Optim: "
            f"{summary.get('optimizer')} | lr: {summary.get('learning_rate', 0.0):.2e} | "
            f"loss: {summary.get('loss_type')} | gamma: {summary.get('gamma', 0.0):.4f} | "
            f"grad_clip: {summary.get('gradient_clipping')}({summary.get('max_grad_norm', 0.0):.2f})"
        )
        self._print(
            "Replay: "
            f"warmup: {int(summary.get('warmup_size', 0))} | "
            f"capacity: {int(summary.get('buffer_capacity', 0))} | "
            f"batch: {int(summary.get('batch_size', 0))} | "
            f"seq_len: {int(summary.get('sequence_length', 0))} | "
            f"min_seq_len: {int(summary.get('min_sequence_length', 0))}"
        )
        self._print(
            "Explore/Target: "
            f"eps {summary.get('epsilon_start', 0.0):.3f}->{summary.get('epsilon_end', 0.0):.3f} "
            f"(decay {int(summary.get('epsilon_decay_steps', 0))}) | "
            f"target {summary.get('target_update_mode')} | "
            f"interval {int(summary.get('target_hard_interval', 0))} | "
            f"tau {summary.get('target_soft_tau', 0.0):.4f}"
        )
        self._print(
            "Eval/CKPT: "
            f"eval_rounds: {int(summary.get('eval_rounds', 0))} | "
            f"eval_decisions: {int(summary.get('eval_max_decisions', 0))} | "
            f"checkpoints: {summary.get('checkpoint_dir')}"
        )
        self._print(
            "Environment: "
            f"decks={int(summary.get('n_decks', 0))} | pen={summary.get('shoe_penetration', 0.0):.2f} | "
            f"S17={not bool(summary.get('dealer_hits_soft_17', False))} | "
            f"payout={summary.get('blackjack_payout', 0.0):.2f} | "
            f"double={summary.get('double_allowed_on')} | split={summary.get('split_rule')} | "
            f"DAS={summary.get('double_after_split_allowed')}"
        )
        self._print(self.SEPARATOR)

    def log_update(
        self,
        *,
        update_in_epoch: int,
        total_updates_in_epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        if not self.config.enable or update_in_epoch % self.config.print_update_interval != 0:
            return
        updates_per_sec = 0.0
        if metrics.get("update_time_sec", 0.0):
            updates_per_sec = 1.0 / metrics["update_time_sec"]
        self._print(
            f"[train step {update_in_epoch}/{total_updates_in_epoch}] "
            f"loss {metrics.get('loss', 0.0):.6f} | "
            f"q {metrics.get('mean_q_pred', 0.0):.4f} | "
            f"target {metrics.get('mean_target', 0.0):.4f} | "
            f"td {metrics.get('mean_abs_td_error', 0.0):.4f} | "
            f"grad {metrics.get('grad_norm', 0.0):.4f} | "
            f"eps {metrics.get('epsilon', 0.0):.4f} | "
            f"lr {metrics.get('learning_rate', 0.0):.2e} | "
            f"upd/s {updates_per_sec:.1f} | "
            f"buffer {int(metrics.get('buffer_size', 0))}"
        )

    def log_collection(
        self,
        *,
        env_step_in_epoch: int,
        total_env_steps_in_epoch: int,
        buffer_size: int,
        warmup_target: int,
        metrics: dict[str, Any],
    ) -> None:
        if not self.config.enable or env_step_in_epoch % self.config.print_collection_interval != 0:
            return
        self._print(
            f"[collect {env_step_in_epoch}/{total_env_steps_in_epoch}] "
            f"buffer {buffer_size}/{warmup_target} | "
            f"rounds {int(metrics.get('rounds_completed', 0))} | "
            f"hands {int(metrics.get('hands_completed', 0))} | "
            f"reward/round {metrics.get('reward_per_round', 0.0):.4f} | "
            f"EV/1000 {metrics.get('ev_per_1000_hands', 0.0):.2f}"
        )

    def log_epoch_summary(self, *, summary: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_epoch_summary:
            return
        self._print(
            "[Train] "
            f"loss {summary.get('loss', 0.0):.6f} | "
            f"q {summary.get('mean_q_pred', 0.0):.4f} | "
            f"target {summary.get('mean_target', 0.0):.4f} | "
            f"td {summary.get('mean_abs_td_error', 0.0):.4f} | "
            f"grad {summary.get('grad_norm', 0.0):.4f} | "
            f"reward/round {summary.get('reward_per_round', 0.0):.4f} | "
            f"EV/1000 {summary.get('ev_per_1000_hands', 0.0):.2f} | "
            f"eps {summary.get('epsilon', 0.0):.4f} | "
            f"lr {summary.get('learning_rate', 0.0):.2e}"
        )
        if self.config.include_segment_details:
            self._print(f"[Epoch Details] situations={summary.get('situation_counts', {})}")
            self._print(f"[Epoch Details] actions={summary.get('action_frequencies', {})}")

    def log_evaluation(self, *, metrics: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_eval_summary:
            return
        self._print(
            "[Val]   "
            f"reward/round {metrics.get('reward_per_round', 0.0):.4f} | "
            f"EV/1000 {metrics.get('ev_per_1000_hands', 0.0):.2f} | "
            f"win {metrics.get('win_rate', 0.0):.4f} | "
            f"push {metrics.get('push_rate', 0.0):.4f} | "
            f"loss {metrics.get('loss_rate', 0.0):.4f} | "
            f"surrender {metrics.get('surrender_rate', 0.0):.4f}"
        )

    def log_checkpoint(self, *, kind: str, path: str, metric_name: str | None = None, metric_value: float | None = None) -> None:
        if kind == "best_eval":
            if metric_name is not None and metric_value is not None:
                self._print(f"Best saved to {path} ({metric_name} {metric_value:.4f})")
                return
            self._print(f"Best saved to {path}")
            return
        if kind == "latest":
            self._print(f"Latest checkpoint saved to {path}")
            return
        if kind == "periodic":
            self._print(f"Periodic checkpoint saved to {path}")
            return
        self._print(f"[Checkpoint] kind={kind} path={path}")

    def log_epoch_time(self, *, epoch_time_sec: float) -> None:
        if not self.config.enable:
            return
        self._print(f"Epoch time: {epoch_time_sec / 60.0:.2f} min")
