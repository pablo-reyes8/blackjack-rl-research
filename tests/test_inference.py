from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.final_wrapper import comparison_models, compare_blackjack_checkpoints, evaluate_blackjack_checkpoint
from scripts.blackjack_rl_cli import infer as infer_cli


CHECKPOINT_PATH = Path("outputs/models/KEEP_04C_unknown_betting_feedforward_best_eval.pt")


@unittest.skipUnless(CHECKPOINT_PATH.exists(), "Expected evaluation checkpoint is missing from outputs/models")
class BlackjackInferenceTests(unittest.TestCase):
    def test_evaluate_blackjack_checkpoint_runs_on_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = evaluate_blackjack_checkpoint(
                checkpoint_path=CHECKPOINT_PATH,
                stage_name="test_eval_04c",
                output_root=tmp_dir,
                bet_multipliers=(1, 2, 3, 4),
                penetrations=(0.75,),
                eval_rounds=4,
                eval_max_decisions=64,
                device="cpu",
                print_summary=False,
            )

        self.assertEqual(result["stage_name"], "test_eval_04c")
        self.assertIn("ev_per_1000_hands", result)
        self.assertIn("bet_q", result)
        self.assertIn("playing", result)

    def test_compare_blackjack_checkpoints_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = compare_blackjack_checkpoints(
                [
                    {
                        "name": "cmp_04c",
                        "checkpoint_path": str(CHECKPOINT_PATH),
                        "bet_multipliers": (1, 2, 3, 4),
                        "penetrations": (0.75,),
                        "architecture": "feedforward",
                        "feedforward_hidden_dims": (256, 256, 128),
                        "use_layer_norm": False,
                        "use_phase_adapters": False,
                        "use_module_gating": False,
                    }
                ],
                eval_rounds=4,
                eval_max_decisions=64,
                device="cpu",
                progress_every_n_rounds=2,
                print_summary=False,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["stage_name"], "cmp_04c")

    def test_comparison_models_alias_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = comparison_models(
                [
                    {
                        "name": "cmp_alias_04c",
                        "checkpoint_path": str(CHECKPOINT_PATH),
                        "bet_multipliers": (1, 2, 3, 4),
                    }
                ],
                eval_rounds=4,
                eval_max_decisions=64,
                device="cpu",
                progress_every_n_rounds=2,
                print_summary=False,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["stage_name"], "cmp_alias_04c")

    def test_infer_cli_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "blackjack-infer",
                    "--checkpoint",
                    str(CHECKPOINT_PATH),
                    "--stage-name",
                    "cli_eval_04c",
                    "--output-root",
                    tmp_dir,
                    "--bet-multipliers",
                    "1,2,3,4",
                    "--eval-rounds",
                    "4",
                    "--max-decisions",
                    "64",
                    "--progress-every-rounds",
                    "2",
                    "--device",
                    "cpu",
                    "--architecture",
                    "feedforward",
                    "--feedforward-hidden-dims",
                    "256,256,128",
                    "--quiet",
                ],
            ):
                with redirect_stdout(stdout):
                    infer_cli.main()

        output = stdout.getvalue()
        self.assertIn('"stage_name": "cli_eval_04c"', output)
        self.assertIn('"checkpoint_path":', output)


if __name__ == "__main__":
    unittest.main()
