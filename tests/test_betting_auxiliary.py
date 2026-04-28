from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from training import BettingAuxiliaryConfig
from training.betting_auxiliary import compute_betting_count_proxy_ce_loss, compute_observed_hi_lo_proxy_from_response
from training.final_wrapper import run_blackjack_stage
from training.replay_buffer import FeedForwardReplayBuffer
from training.config import ReplayBufferConfig


class BettingAuxiliaryTests(unittest.TestCase):
    def test_compute_observed_hi_lo_proxy_from_rank_counts(self) -> None:
        response = {
            "observation": {
                "observed_cards_history": {
                    "A": 1,
                    "2": 2,
                    "3": 1,
                    "4": 0,
                    "5": 1,
                    "6": 1,
                    "7": 0,
                    "8": 0,
                    "9": 1,
                    "10": 1,
                    "J": 0,
                    "Q": 0,
                    "K": 1,
                }
            }
        }

        proxy = compute_observed_hi_lo_proxy_from_response(response, n_decks=8)

        self.assertEqual(proxy["observed_cards"], 9)
        self.assertAlmostEqual(float(proxy["running_count"]), 2.0, places=6)
        self.assertAlmostEqual(float(proxy["estimated_decks_seen"]), 9.0 / 52.0, places=6)
        self.assertGreater(float(proxy["true_count_proxy"]), 0.0)

    def test_compute_betting_count_proxy_ce_loss_ignores_rows_below_min_observed_cards(self) -> None:
        config = BettingAuxiliaryConfig(enabled=True, min_observed_cards=12)
        student_output = {
            "q_values": torch.tensor(
                [
                    [0.2, 0.1, 0.0, -0.1, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                    [0.1, 0.4, 0.2, -0.2, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                ],
                dtype=torch.float32,
            )
        }
        action_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        auxiliary = {
            "true_count_proxy": torch.tensor([3.5, 2.5], dtype=torch.float32),
            "observed_cards": torch.tensor([4, 20], dtype=torch.long),
        }

        loss = compute_betting_count_proxy_ce_loss(
            student_output=student_output,
            action_mask=action_mask,
            betting_auxiliary=auxiliary,
            config=config,
            betting_action_slice=slice(0, 4),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_compute_betting_count_proxy_ce_loss_supports_class_weights(self) -> None:
        student_output = {
            "q_values": torch.tensor(
                [
                    [2.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                    [0.0, 0.0, 0.0, 2.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                ],
                dtype=torch.float32,
            )
        }
        action_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        auxiliary = {
            "true_count_proxy": torch.tensor([0.0, 5.0], dtype=torch.float32),
            "observed_cards": torch.tensor([20, 20], dtype=torch.long),
        }

        unweighted = compute_betting_count_proxy_ce_loss(
            student_output=student_output,
            action_mask=action_mask,
            betting_auxiliary=auxiliary,
            config=BettingAuxiliaryConfig(enabled=True, min_observed_cards=12, class_weights=None),
            betting_action_slice=slice(0, 4),
        )
        weighted = compute_betting_count_proxy_ce_loss(
            student_output=student_output,
            action_mask=action_mask,
            betting_auxiliary=auxiliary,
            config=BettingAuxiliaryConfig(
                enabled=True,
                min_observed_cards=12,
                class_weights=(0.25, 1.0, 1.5, 2.0),
            ),
            betting_action_slice=slice(0, 4),
        )

        self.assertNotEqual(float(weighted.item()), float(unweighted.item()))

    def test_feedforward_replay_buffer_samples_betting_auxiliary(self) -> None:
        buffer = FeedForwardReplayBuffer(ReplayBufferConfig(capacity=16, batch_size=2, warmup_size=0))
        for index in range(2):
            state = {
                "state_vector": torch.tensor([float(index), 1.0], dtype=torch.float32),
                "action_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
                "betting_auxiliary": {
                    "true_count_proxy": torch.tensor(float(index), dtype=torch.float32),
                    "observed_cards": torch.tensor(12 + index, dtype=torch.long),
                },
            }
            next_state = {
                "state_vector": torch.tensor([float(index + 1), 2.0], dtype=torch.float32),
                "action_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
                "betting_auxiliary": {
                    "true_count_proxy": torch.tensor(float(index + 1), dtype=torch.float32),
                    "observed_cards": torch.tensor(13 + index, dtype=torch.long),
                },
            }
            buffer.add(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": index,
                    "reward": float(index),
                    "done": False,
                    "n_steps": 1,
                    "action_mask": state["action_mask"],
                    "next_action_mask": next_state["action_mask"],
                }
            )

        batch = buffer.sample()

        self.assertIn("betting_auxiliary", batch)
        self.assertEqual(tuple(batch["betting_auxiliary"]["true_count_proxy"].shape), (2,))
        self.assertEqual(tuple(batch["betting_auxiliary"]["observed_cards"].shape), (2,))

    def test_run_blackjack_stage_validates_auxiliary_mode_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                run_blackjack_stage(
                    stage_name="invalid_aux_mode",
                    output_root=Path(tmp_dir),
                    run_training=False,
                    enable_prints=False,
                    axu_loss_bet=True,
                    include_observed_history=True,
                    bet_multipliers=(1, 2),
                )

    def test_run_blackjack_stage_exposes_auxiliary_mode_in_pipeline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_blackjack_stage(
                stage_name="valid_aux_mode",
                output_root=Path(tmp_dir),
                run_training=False,
                enable_prints=False,
                include_observed_history=True,
                include_temporal_context=True,
                bet_multipliers=(1, 2, 3, 4),
                axu_loss_bet=True,
            )

        self.assertTrue(result["pipeline_config"].betting_auxiliary.enabled)
        self.assertEqual(result["pipeline_config"].betting_auxiliary.bet_multipliers, (1, 2, 3, 4))

    def test_run_blackjack_stage_accepts_auxiliary_class_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_blackjack_stage(
                stage_name="weighted_aux_mode",
                output_root=Path(tmp_dir),
                run_training=False,
                enable_prints=False,
                include_observed_history=True,
                bet_multipliers=(1, 2, 3, 4),
                axu_loss_bet=True,
                betting_auxiliary_class_weights=(0.25, 1.0, 1.5, 2.0),
            )

        self.assertEqual(
            result["pipeline_config"].betting_auxiliary.class_weights,
            (0.25, 1.0, 1.5, 2.0),
        )


if __name__ == "__main__":
    unittest.main()
