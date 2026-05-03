from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from training import BettingAuxiliaryConfig
from training.betting_auxiliary import (
    compute_betting_count_proxy_ce_loss,
    compute_observed_hi_lo_proxy_from_response,
    map_count_proxy_to_bet_target,
)
from training.config import ObservedEVRankingConfig
from training.ev_calibration import EVBucketActionTable, compute_observed_ev_ranking_loss
from training.final_wrapper import run_blackjack_stage
from training.replay_buffer import FeedForwardReplayBuffer
from training.config import ReplayBufferConfig


class BettingAuxiliaryTests(unittest.TestCase):
    def test_map_count_proxy_to_bet_target_supports_flexible_spreads(self) -> None:
        proxy = torch.tensor([-1.0, 1.5, 3.0, 10.0], dtype=torch.float32)

        cases = {
            (1, 2, 3, 4): [0, 1, 2, 3],
            (1, 2, 3): [0, 1, 2, 2],
            (1, 2): [0, 1, 1, 1],
            (1,): [0, 0, 0, 0],
        }
        for multipliers, expected in cases.items():
            with self.subTest(multipliers=multipliers):
                target = map_count_proxy_to_bet_target(
                    proxy,
                    threshold_2x=1.0,
                    threshold_3x=2.0,
                    threshold_4x=4.0,
                    bet_multipliers=multipliers,
                )
                self.assertEqual(target.tolist(), expected)

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

    def test_compute_betting_count_proxy_ce_loss_supports_three_class_spread(self) -> None:
        config = BettingAuxiliaryConfig(
            enabled=True,
            bet_multipliers=(1, 2, 3),
            class_weights=(0.5, 1.0, 1.5),
            min_observed_cards=0,
        )
        student_output = {
            "q_values": torch.tensor(
                [
                    [0.2, 0.1, 0.0, -5.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                    [0.0, 0.2, 0.4, -5.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                ],
                dtype=torch.float32,
            )
        }
        action_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        auxiliary = {
            "true_count_proxy": torch.tensor([1.5, 10.0], dtype=torch.float32),
            "observed_cards": torch.tensor([20, 20], dtype=torch.long),
        }

        loss = compute_betting_count_proxy_ce_loss(
            student_output=student_output,
            action_mask=action_mask,
            betting_auxiliary=auxiliary,
            config=config,
            betting_action_slice=slice(0, 4),
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_ev_bucket_action_table_summarizes_and_ranks_pairs(self) -> None:
        table = EVBucketActionTable()
        table.update("low", "bet_1x", -1.0)
        table.update("low", "bet_1x", 1.0)
        table.update("high", "bet_1x", -0.05)
        table.update("high", "bet_2x", 0.02)
        table.update("high", "bet_3x", -0.10)
        for _ in range(99):
            table.update("high", "bet_1x", -0.05)
            table.update("high", "bet_2x", 0.02)
            table.update("high", "bet_3x", -0.10)

        summary = table.summary()
        self.assertEqual(summary["low"]["bet_1x"]["n"], 2.0)
        self.assertAlmostEqual(summary["low"]["bet_1x"]["mean_reward"], 0.0)
        self.assertAlmostEqual(summary["low"]["bet_1x"]["ev_per_1000"], 0.0)
        self.assertGreaterEqual(summary["low"]["bet_1x"]["std_reward"], 0.0)

        pairs_vs_1x = table.get_preferred_pairs(
            min_samples=30,
            min_ev_gap_per_round=0.005,
            compare_against_1x_only=True,
            allowed_actions=("bet_1x", "bet_2x", "bet_3x"),
        )
        self.assertEqual([pair[:2] for pair in pairs_vs_1x["high"]], [("bet_2x", "bet_1x"), ("bet_1x", "bet_3x")])

        all_pairs = table.get_preferred_pairs(
            min_samples=30,
            min_ev_gap_per_round=0.005,
            compare_against_1x_only=False,
            allowed_actions=("bet_1x", "bet_2x", "bet_3x"),
        )
        all_pair_names = [pair[:2] for pair in all_pairs["high"]]
        self.assertIn(("bet_2x", "bet_1x"), all_pair_names)
        self.assertIn(("bet_2x", "bet_3x"), all_pair_names)
        self.assertIn(("bet_1x", "bet_3x"), all_pair_names)

    def test_observed_ev_ranking_loss_is_finite_and_decreases_when_ranking_is_satisfied(self) -> None:
        table = EVBucketActionTable()
        for _ in range(30):
            table.update("high", "bet_1x", -0.05)
            table.update("high", "bet_2x", 0.05)
        config = ObservedEVRankingConfig(
            enabled=True,
            weight=0.003,
            min_bucket_action_samples=30,
            min_observed_cards=0,
            margin=0.05,
            compare_against_1x_only=True,
        )
        batch = {
            "betting_auxiliary": {
                "true_count_proxy": torch.tensor([2.5, 3.0], dtype=torch.float32),
                "observed_cards": torch.tensor([20, 20], dtype=torch.long),
            }
        }
        worse_output = {"bet_q_values": torch.tensor([[0.3, 0.1, 0.0, 0.0], [0.2, 0.1, 0.0, 0.0]])}
        better_output = {"bet_q_values": torch.tensor([[0.1, 0.3, 0.0, 0.0], [0.1, 0.25, 0.0, 0.0]])}

        worse_loss = compute_observed_ev_ranking_loss(
            worse_output,
            batch,
            config=config,
            ev_table=table,
        )
        better_loss = compute_observed_ev_ranking_loss(
            better_output,
            batch,
            config=config,
            ev_table=table,
        )

        self.assertTrue(torch.isfinite(worse_loss).item())
        self.assertGreater(float(worse_loss.item()), float(better_loss.item()))

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

    def test_run_blackjack_stage_accepts_flexible_auxiliary_spreads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_blackjack_stage(
                stage_name="flex_aux_mode",
                output_root=Path(tmp_dir),
                run_training=False,
                enable_prints=False,
                axu_loss_bet=True,
                include_observed_history=True,
                bet_multipliers=(1, 2),
                betting_auxiliary_class_weights=(0.5, 1.0),
            )

        self.assertEqual(result["pipeline_config"].betting_auxiliary.bet_multipliers, (1, 2))
        self.assertEqual(result["pipeline_config"].betting_auxiliary.class_weights, (0.5, 1.0))

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
