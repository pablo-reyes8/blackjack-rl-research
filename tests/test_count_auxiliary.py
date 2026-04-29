from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from model.agents import FeedForwardDoubleDQN
from training.config import CountAuxiliaryConfig
from training.count_auxiliary import compute_count_bucket_ce_loss, map_count_proxy_to_count_bucket
from training.final_wrapper import run_blackjack_stage


class CountAuxiliaryTests(unittest.TestCase):
    def test_map_count_proxy_to_count_bucket(self) -> None:
        target = map_count_proxy_to_count_bucket(
            torch.tensor([-1.0, 1.0, 2.5, 5.0], dtype=torch.float32),
            threshold_medium=1.0,
            threshold_high=2.0,
            threshold_very_high=4.0,
        )
        self.assertTrue(torch.equal(target, torch.tensor([0, 1, 2, 3], dtype=torch.long)))

    def test_compute_count_bucket_ce_loss_ignores_rows_below_min_observed_cards(self) -> None:
        config = CountAuxiliaryConfig(enabled=True, min_observed_cards=12)
        student_output = {
            "count_bucket_logits": torch.tensor(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0],
                ],
                dtype=torch.float32,
            )
        }
        auxiliary = {
            "true_count_proxy": torch.tensor([0.0, 5.0], dtype=torch.float32),
            "observed_cards": torch.tensor([4, 20], dtype=torch.long),
        }

        loss = compute_count_bucket_ce_loss(
            student_output=student_output,
            count_auxiliary=auxiliary,
            config=config,
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_feedforward_model_emits_count_bucket_logits_when_enabled(self) -> None:
        model = FeedForwardDoubleDQN.from_profile(
            "minimal_basic_strategy",
            use_count_auxiliary_head=True,
            count_auxiliary_hidden_dim=32,
            count_auxiliary_num_buckets=4,
        )
        output = model(
            {
                "state_vector": torch.zeros(model.state_dim, dtype=torch.float32),
                "action_mask": torch.ones(model.num_actions, dtype=torch.bool),
                "module_tensors": {},
            }
        )

        self.assertIn("count_bucket_logits", output)
        self.assertEqual(tuple(output["count_bucket_logits"].shape), (1, 4))

    def test_run_blackjack_stage_exposes_count_auxiliary_and_auto_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_blackjack_stage(
                stage_name="count_aux_mode",
                output_root=Path(tmp_dir),
                run_training=False,
                enable_prints=False,
                include_observed_history=True,
                include_temporal_context=True,
                count_auxiliary_enabled=True,
            )

        self.assertTrue(result["pipeline_config"].count_auxiliary.enabled)
        self.assertTrue(result["model"].config.use_count_auxiliary_head)


if __name__ == "__main__":
    unittest.main()
