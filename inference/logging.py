from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from training.config import PrintConfig
from training.logging import TrainingLogger


class InferenceLogger:
    SEPARATOR = "=" * 96
    BLOCK_SEPARATOR = "-" * 96

    def __init__(self, *, enable: bool = True) -> None:
        self.enable = enable
        self._training_logger = TrainingLogger(
            PrintConfig(
                enable=enable,
                print_run_summary=False,
                print_epoch_header=False,
                print_epoch_summary=False,
                print_eval_summary=True,
            )
        )

    def _print(self, message: str) -> None:
        if self.enable:
            print(message)

    def _print_block(self, title: str, lines: list[str], *, strong: bool = False) -> None:
        separator = self.SEPARATOR if strong else self.BLOCK_SEPARATOR
        self._print(separator)
        self._print(title)
        for line in lines:
            self._print(line)

    def log_header(
        self,
        *,
        checkpoint_path: str | Path,
        stage_name: str,
        device: str,
        eval_rounds: int,
        eval_max_decisions: int,
        base_seed: int,
        model_config: Mapping[str, Any],
        bet_multipliers: tuple[int, ...],
        penetrations: tuple[float, ...],
        start_mode: str,
        min_burned_rounds: int,
        max_burned_rounds: int,
    ) -> None:
        encoder = model_config.get("encoder") or {}
        self._print_block(
            "BLACKJACK CHECKPOINT EVALUATION",
            [
                "  Checkpoint: "
                f"name={stage_name} | path={Path(checkpoint_path)}",
                "  Model     : "
                f"arch={model_config.get('architecture')} | recurrent={model_config.get('recurrent_type', 'none')} | "
                f"encoder={encoder.get('profile')}",
                "  Runtime   : "
                f"device={device} | rounds={int(eval_rounds)} | max_decisions={int(eval_max_decisions)} | seed={int(base_seed)}",
                "  Table     : "
                f"start={start_mode} | burned={int(min_burned_rounds)}-{int(max_burned_rounds)} | "
                f"bets={bet_multipliers} | pens={penetrations}",
            ],
            strong=True,
        )

    def log_model_start(
        self,
        *,
        index: int,
        total: int,
        stage_name: str,
        checkpoint_path: str | Path,
    ) -> None:
        self._print_block(
            "MODEL RUN",
            [
                f"  Slot      : {index}/{total}",
                f"  Name      : {stage_name}",
                f"  Checkpoint: {Path(checkpoint_path)}",
            ],
            strong=True,
        )

    def log_progress(
        self,
        *,
        stage_name: str,
        rounds_completed: int,
        eval_rounds: int,
        decisions: int,
        metrics: dict[str, Any],
    ) -> None:
        self._print_block(
            f"PROGRESS {stage_name}",
            [
                "  Runtime   : "
                f"rounds={int(rounds_completed)}/{int(eval_rounds)} | decisions={int(decisions)}",
            ],
        )
        self._training_logger.log_evaluation(metrics=metrics)

    def log_evaluation(self, *, metrics: dict[str, Any]) -> None:
        self._training_logger.log_evaluation(metrics=metrics)

    def log_result_summary(self, *, result: Mapping[str, Any]) -> None:
        self._print_block(
            "EVALUATION SUMMARY",
            [
                "  Core      : "
                f"EV/1000={result.get('ev_per_1000_hands', 0.0):+.2f} | "
                f"reward/round={result.get('reward_per_round', 0.0):+.4f} | "
                f"round_std={result.get('round_std', 0.0):.4f}",
                "  Betting   : "
                f"1x={result.get('bet_1x_frac', 0.0):.3f} | "
                f"2x={result.get('bet_2x_frac', 0.0):.3f} | "
                f"3x={result.get('bet_3x_frac', 0.0):.3f} | "
                f"4x={result.get('bet_4x_frac', 0.0):.3f} | "
                f"agg={result.get('agg_frac', 0.0):.3f}",
                "  Safety    : "
                f"win={result.get('win_frac', 0.0):.4f} | push={result.get('push_frac', 0.0):.4f} | "
                f"loss={result.get('loss_frac', 0.0):.4f} | bust={result.get('bust_frac', 0.0):.4f}",
            ],
        )

    def log_comparison(self, *, results: list[Mapping[str, Any]]) -> None:
        if not results:
            return
        ordered = sorted(results, key=lambda item: float(item.get("ev_per_1000_hands", float("-inf"))), reverse=True)
        lines = []
        for index, item in enumerate(ordered, start=1):
            lines.append(
                f"  {index:>2}. {item.get('stage_name', Path(str(item.get('checkpoint_path', 'unknown'))).stem)} | "
                f"EV/1000={float(item.get('ev_per_1000_hands', 0.0)):+.2f} | "
                f"std={float(item.get('round_std', 0.0)):.4f} | "
                f"agg={float(item.get('agg_frac', 0.0)):.3f} | "
                f"very_high_margin={float(((item.get('bet_aux') or {}).get('mean_margin_by_count_bucket') or {}).get('very_high') or 0.0):+.4f}"
            )
        self._print_block("MODEL COMPARISON", lines, strong=True)
