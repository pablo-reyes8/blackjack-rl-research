from __future__ import annotations

from typing import Any

from enviroment_bj.core import BET_ACTION_ORDER, PLAYING_ACTION_ORDER

from .config import PrintConfig


class TrainingLogger:
    SEPARATOR = "=" * 96
    BLOCK_SEPARATOR = "-" * 96

    def __init__(self, config: PrintConfig) -> None:
        self.config = config
        self._last_warmup_bucket = 0
        self._last_warmup_size = 0

    def _print(self, message: str) -> None:
        if self.config.enable:
            print(message)

    def _print_block(self, title: str, lines: list[str], *, strong: bool = False) -> None:
        separator = self.SEPARATOR if strong else self.BLOCK_SEPARATOR
        self._print(separator)
        self._print(title)
        for line in lines:
            self._print(line)

    def should_log_collection(self, env_step_in_epoch: int) -> bool:
        return self.config.enable and env_step_in_epoch % self.config.print_collection_interval == 0

    def should_log_update(self, update_in_epoch: int) -> bool:
        return self.config.enable and update_in_epoch % self.config.print_update_interval == 0

    def _format_action_distribution(self, metrics: dict[str, Any], key: str, actions: tuple[str, ...]) -> str:
        frequencies = metrics.get(key) or {}
        if not isinstance(frequencies, dict):
            return "n/a"
        return " ".join(f"{action}:{float(frequencies.get(action, 0.0)):.2f}" for action in actions)

    def _format_bet_ev(self, metrics: dict[str, Any]) -> str:
        values = metrics.get("bet_ev_per_1000_rounds_by_action") or {}
        if not isinstance(values, dict):
            return "n/a"
        return " ".join(f"{action}:{float(values.get(action, 0.0)):+.1f}" for action in BET_ACTION_ORDER)

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
        self._print_block(
            "BLACKJACK RL RUN",
            [
                "  Model      : "
                f"arch={summary.get('architecture')} | recurrent={summary.get('recurrent_type')} | "
                f"encoder={summary.get('encoder_profile')} | obs={summary.get('observation_profile')} | "
                f"start={summary.get('start_state_mode')}",
                "  Runtime    : "
                f"device={summary.get('device')} | epochs={int(summary.get('total_epochs', 0))} | "
                f"envs={int(summary.get('num_envs', 0))} | steps/epoch={int(summary.get('nominal_env_steps_per_epoch', 0))} | "
                f"updates/epoch~={int(summary.get('estimated_updates_per_epoch', 0))} | params={int(summary.get('parameter_count', 0)):,}",
                "  Optim      : "
                f"optimizer={summary.get('optimizer')} | lr={summary.get('learning_rate', 0.0):.2e} | "
                f"loss={summary.get('loss_type')} | gamma={summary.get('gamma', 0.0):.4f} | "
                f"grad_clip={summary.get('gradient_clipping')}({summary.get('max_grad_norm', 0.0):.2f})",
                "  Replay     : "
                f"warmup={int(summary.get('warmup_size', 0))} | capacity={int(summary.get('buffer_capacity', 0))} | "
                f"batch={int(summary.get('batch_size', 0))} | seq_len={int(summary.get('sequence_length', 0))} | "
                f"min_seq_len={int(summary.get('min_sequence_length', 0))}",
                "  Explore    : "
                f"eps_bet={summary.get('epsilon_betting_start', 0.0):.3f}->{summary.get('epsilon_betting_end', 0.0):.3f} "
                f"(decay {int(summary.get('epsilon_betting_decay_steps', 0))}) | "
                f"eps_play={summary.get('epsilon_playing_start', 0.0):.3f}->{summary.get('epsilon_playing_end', 0.0):.3f} "
                f"(decay {int(summary.get('epsilon_playing_decay_steps', 0))}) | "
                f"target={summary.get('target_update_mode')} | interval={int(summary.get('target_hard_interval', 0))} | "
                f"tau={summary.get('target_soft_tau', 0.0):.4f}",
                "  Extras     : "
                f"n_step={summary.get('n_step_enabled', False)}({int(summary.get('n_step_size', 1))}) | "
                f"phase_loss_w={summary.get('phase_loss_weights_enabled', False)} "
                f"(bet {summary.get('betting_loss_weight', 0.0):.2f}, play {summary.get('playing_loss_weight', 0.0):.2f}) | "
                f"phase_adapters={summary.get('use_phase_adapters', False)} | module_gating={summary.get('use_module_gating', False)}",
                "  Transfer   : "
                f"enabled={summary.get('transfer_enabled', False)} | warm_start={summary.get('warm_start_checkpoint_path') or 'none'} | "
                f"teacher={summary.get('teacher_checkpoint_path') or 'none'} | distill={summary.get('distillation_enabled', False)} "
                f"({summary.get('distillation_mode')}, {summary.get('distillation_weight', 0.0):.3f}->{summary.get('distillation_final_weight', 0.0):.3f})",
                "  Eval / CKPT: "
                f"eval_rounds={int(summary.get('eval_rounds', 0))} | eval_decisions={int(summary.get('eval_max_decisions', 0))} | "
                f"checkpoints={summary.get('checkpoint_dir')}",
                "  Table      : "
                f"decks={int(summary.get('n_decks', 0))} | pen={summary.get('shoe_penetration', 0.0):.2f} | "
                f"S17={not bool(summary.get('dealer_hits_soft_17', False))} | payout={summary.get('blackjack_payout', 0.0):.2f} | "
                f"double={summary.get('double_allowed_on')} | split={summary.get('split_rule')} | DAS={summary.get('double_after_split_allowed')}",
            ],
            strong=True,
        )

    def log_update(
        self,
        *,
        update_in_epoch: int,
        total_updates_in_epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        if not self.should_log_update(update_in_epoch):
            return
        updates_per_sec = 0.0
        if metrics.get("update_time_sec", 0.0):
            updates_per_sec = 1.0 / metrics["update_time_sec"]
        self._print_block(
            f"TRAIN STEP {update_in_epoch}/{total_updates_in_epoch}",
            [
                "  Core   : "
                f"loss={metrics.get('loss', 0.0):.6f} | q={metrics.get('mean_q_pred', 0.0):.4f} | "
                f"target={metrics.get('mean_target', 0.0):.4f} | td={metrics.get('mean_abs_td_error', 0.0):.4f}",
                "  Phase  : "
                f"loss_bet={metrics.get('loss_betting', 0.0):.6f} | loss_play={metrics.get('loss_playing', 0.0):.6f} | "
                f"td_bet={metrics.get('mean_abs_td_error_betting', 0.0):.4f} | td_play={metrics.get('mean_abs_td_error_playing', 0.0):.4f}",
                "  Optim  : "
                f"grad={metrics.get('grad_norm', 0.0):.4f} | lr={metrics.get('learning_rate', 0.0):.2e} | "
                f"upd/s={updates_per_sec:.1f} | buffer={int(metrics.get('buffer_size', 0))}",
                "  Policy : "
                f"eps_bet={metrics.get('epsilon_betting', 0.0):.4f} | eps_play={metrics.get('epsilon_playing', 0.0):.4f} | "
                f"n_step={metrics.get('mean_n_steps', 1.0):.2f} | phase_w={metrics.get('mean_phase_weight', 1.0):.2f} | "
                f"distill={metrics.get('distillation_loss', 0.0):.6f} @ {metrics.get('distillation_weight', 0.0):.3f}",
            ],
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
        if not self.should_log_collection(env_step_in_epoch):
            return
        self._print_block(
            f"COLLECT {env_step_in_epoch}/{total_env_steps_in_epoch}",
            [
                "  Progress: "
                f"buffer={buffer_size}/{warmup_target} | rounds={int(metrics.get('rounds_completed', 0))} | "
                f"hands={int(metrics.get('hands_completed', 0))}",
                "  Reward  : "
                f"reward/round={metrics.get('reward_per_round', 0.0):.4f} | EV/1000={metrics.get('ev_per_1000_hands', 0.0):.2f} | "
                f"round_std={metrics.get('round_reward_std', 0.0):.4f}",
                "  Explore : "
                f"bet_random={metrics.get('random_action_fraction_betting', 0.0):.3f} | bet_greedy={metrics.get('greedy_action_fraction_betting', 0.0):.3f} | "
                f"play_random={metrics.get('random_action_fraction_playing', 0.0):.3f} | play_greedy={metrics.get('greedy_action_fraction_playing', 0.0):.3f}",
                "  Betting : " + self._format_action_distribution(metrics, 'bet_action_frequencies', BET_ACTION_ORDER),
                "  Playing : " + self._format_action_distribution(metrics, 'play_action_frequencies', PLAYING_ACTION_ORDER),
            ],
        )

    def log_epoch_summary(self, *, summary: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_epoch_summary:
            return
        self._print_block(
            "TRAIN EPOCH SUMMARY",
            [
                "  Core    : "
                f"loss={summary.get('loss', 0.0):.6f} | q={summary.get('mean_q_pred', 0.0):.4f} | target={summary.get('mean_target', 0.0):.4f} | td={summary.get('mean_abs_td_error', 0.0):.4f}",
                "  Reward  : "
                f"reward/round={summary.get('reward_per_round', 0.0):.4f} | EV/1000={summary.get('ev_per_1000_hands', 0.0):.2f} | round_std={summary.get('round_reward_std', 0.0):.4f}",
                "  Policy  : "
                f"eps_bet={summary.get('epsilon_betting', 0.0):.4f} | eps_play={summary.get('epsilon_playing', 0.0):.4f} | lr={summary.get('learning_rate', 0.0):.2e} | grad={summary.get('grad_norm', 0.0):.4f}",
                "  Outcomes: "
                f"win={summary.get('win_rate', 0.0):.4f} | push={summary.get('push_rate', 0.0):.4f} | loss={summary.get('loss_rate', 0.0):.4f} | blackjack={summary.get('blackjack_rate', 0.0):.4f} | bust={summary.get('bust_rate', 0.0):.4f}",
                "  Betting : "
                f"decisions={int(summary.get('betting_decisions', 0))} | random={summary.get('random_action_fraction_betting', 0.0):.3f} | "
                f"1x_frac={summary.get('conservative_bet_fraction', 0.0):.3f} | agg_frac={summary.get('aggressive_bet_fraction', 0.0):.3f}",
                "           " + self._format_action_distribution(summary, 'bet_action_frequencies', BET_ACTION_ORDER),
                "           bet_EV " + self._format_bet_ev(summary),
                "  Playing : "
                f"decisions={int(summary.get('playing_decisions', 0))} | random={summary.get('random_action_fraction_playing', 0.0):.3f} | "
                f"loss_bet={summary.get('loss_betting', 0.0):.6f} | loss_play={summary.get('loss_playing', 0.0):.6f}",
                "           " + self._format_action_distribution(summary, 'play_action_frequencies', PLAYING_ACTION_ORDER),
            ],
            strong=True,
        )
        if self.config.include_segment_details:
            self._print(f"[Epoch Details] situations={summary.get('situation_counts', {})}")
            self._print(f"[Epoch Details] actions={summary.get('action_frequencies', {})}")

    def log_evaluation(self, *, metrics: dict[str, Any]) -> None:
        if not self.config.enable or not self.config.print_eval_summary:
            return
        self._print_block(
            "VAL",
            [
                "  Reward  : "
                f"reward/round={metrics.get('reward_per_round', 0.0):.4f} | EV/1000={metrics.get('ev_per_1000_hands', 0.0):.2f} | round_std={metrics.get('round_reward_std', 0.0):.4f}",
                "  Outcomes: "
                f"win={metrics.get('win_rate', 0.0):.4f} | push={metrics.get('push_rate', 0.0):.4f} | loss={metrics.get('loss_rate', 0.0):.4f} | "
                f"blackjack={metrics.get('blackjack_rate', 0.0):.4f} | bust={metrics.get('bust_rate', 0.0):.4f}",
                "  Betting : "
                f"1x_frac={metrics.get('conservative_bet_fraction', 0.0):.3f} | agg_frac={metrics.get('aggressive_bet_fraction', 0.0):.3f} | "
                f"bet_random={metrics.get('random_action_fraction_betting', 0.0):.3f}",
                "           " + self._format_action_distribution(metrics, 'bet_action_frequencies', BET_ACTION_ORDER),
                "           bet_EV " + self._format_bet_ev(metrics),
                "  Playing : "
                f"play_random={metrics.get('random_action_fraction_playing', 0.0):.3f} | insurance_reward={metrics.get('insurance_reward_total', 0.0):.4f}",
                "           " + self._format_action_distribution(metrics, 'play_action_frequencies', PLAYING_ACTION_ORDER),
            ],
            strong=True,
        )

    def log_train_val_comparison(self, *, train_metrics: dict[str, Any], eval_metrics: dict[str, Any] | None) -> None:
        if not self.config.enable or eval_metrics is None:
            return
        self._print_block(
            "TRAIN vs VAL",
            [
                "  Gap     : "
                f"EV_gap={eval_metrics.get('ev_per_1000_hands', 0.0) - train_metrics.get('ev_per_1000_hands', 0.0):+.2f} | "
                f"reward_gap={eval_metrics.get('reward_per_round', 0.0) - train_metrics.get('reward_per_round', 0.0):+.4f} | "
                f"std_gap={eval_metrics.get('round_reward_std', 0.0) - train_metrics.get('round_reward_std', 0.0):+.4f}",
                "  Betting : "
                f"1x train={train_metrics.get('conservative_bet_fraction', 0.0):.3f} / val={eval_metrics.get('conservative_bet_fraction', 0.0):.3f} | "
                f"agg train={train_metrics.get('aggressive_bet_fraction', 0.0):.3f} / val={eval_metrics.get('aggressive_bet_fraction', 0.0):.3f}",
                "  Playing : "
                f"double train={float((train_metrics.get('play_action_frequencies') or {}).get('double', 0.0)):.3f} / "
                f"val={float((eval_metrics.get('play_action_frequencies') or {}).get('double', 0.0)):.3f} | "
                f"insurance train={float((train_metrics.get('play_action_frequencies') or {}).get('insurance', 0.0)):.3f} / "
                f"val={float((eval_metrics.get('play_action_frequencies') or {}).get('insurance', 0.0)):.3f}",
            ],
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
