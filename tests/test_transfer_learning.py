from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import torch

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig
from enviroment_bj.core import hand_value
from model.agents import FeedForwardDoubleDQN
from training import (
    CheckpointConfig,
    DistillationConfig,
    EpsilonScheduleConfig,
    EvaluationConfig,
    OptimizationConfig,
    PrintConfig,
    ReplayBufferConfig,
    TargetUpdateConfig,
    TrainerConfig,
    TrainingPipelineConfig,
    TransferLearningConfig,
    adapt_response_to_minimal_basic_strategy,
    build_optimizer_with_param_groups,
    compute_q_distillation_loss,
    freeze_playing_policy_parts,
    train_model,
)


class TransferLearningTests(unittest.TestCase):
    def make_env(self, observation_profile: str) -> BlackjackEnvironment:
        return BlackjackEnvironment(
            config=BlackjackConfig(
                n_decks=1,
                shoe_penetration=1.0,
                observation=ObservationConfig.for_profile(observation_profile),
            ),
            seed=17,
        )

    def make_pipeline_config(self, checkpoint_dir: Path) -> TrainingPipelineConfig:
        return TrainingPipelineConfig(
            trainer=TrainerConfig(
                total_epochs=1,
                env_steps_per_epoch=18,
                train_frequency=1,
                updates_per_train_step=1,
                max_updates_per_epoch=4,
                device="cpu",
                seed=23,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=256,
                batch_size=4,
                warmup_size=8,
                sequence_length=8,
                min_sequence_length=2,
            ),
            epsilon=EpsilonScheduleConfig(start=0.8, end=0.1, decay_steps=50, evaluation_epsilon=0.0),
            optimization=OptimizationConfig(
                optimizer="adam",
                learning_rate=1e-3,
                gradient_clipping=True,
                max_grad_norm=5.0,
            ),
            target_update=TargetUpdateConfig(mode="hard", hard_update_interval=2, soft_tau=0.01),
            evaluation=EvaluationConfig(enabled=True, every_n_epochs=1, num_rounds=4, max_decisions=200),
            checkpoints=CheckpointConfig(
                directory=str(checkpoint_dir),
                save_latest=True,
                save_best_eval=True,
                save_periodic=False,
                periodic_interval_updates=100,
            ),
            prints=PrintConfig(enable=False),
        )

    def test_adapt_response_to_minimal_basic_strategy_recovers_table_raw_hand_totals(self) -> None:
        env = self.make_env("table_realistic_default")
        env.reset()
        response = env.step("bet_1x")

        adapted = adapt_response_to_minimal_basic_strategy(response)
        cards = response["observation"]["current_hand_cards"]
        total, is_soft = hand_value(cards)

        self.assertEqual(adapted["observation"]["profile"], "minimal_basic_strategy")
        self.assertEqual(adapted["observation"]["mode"], "basic_strategy")
        self.assertEqual(adapted["observation"]["current_hand_total"], total)
        self.assertEqual(adapted["observation"]["current_hand_is_soft"], is_soft)
        self.assertEqual(adapted["observation"]["dealer_upcard"], response["observation"]["dealer_upcard"])

    def test_compute_q_distillation_loss_uses_only_legal_playing_actions(self) -> None:
        student_output = {
            "q_values": torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    [1.0, 2.0, 3.0, 4.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                ]
            )
        }
        teacher_output = {
            "q_values": torch.tensor(
                [
                    [9.0, 9.0, 9.0, 9.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
                    [9.0, 9.0, 9.0, 9.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                ]
            )
        }
        action_mask = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
            ],
            dtype=torch.bool,
        )

        loss = compute_q_distillation_loss(
            student_output,
            teacher_output,
            action_mask,
            playing_action_slice=slice(4, 10),
        )

        expected = torch.tensor([(20.0 - 18.0) ** 2, (21.0 - 19.0) ** 2, (24.0 - 22.0) ** 2], dtype=torch.float32).mean()
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)

    def test_freeze_playing_policy_parts_keeps_bet_head_trainable_and_builds_param_groups(self) -> None:
        model = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")

        freeze_playing_policy_parts(model)
        optimizer = build_optimizer_with_param_groups(
            model,
            backbone_lr=1e-5,
            play_lr=1e-5,
            bet_lr=3e-4,
            default_lr=1e-4,
            optimizer_name="adamw",
        )

        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.play_head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.bet_head.parameters()))
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["bet"])
        self.assertAlmostEqual(float(optimizer.param_groups[0]["lr"]), 3e-4, places=12)

    def test_train_model_runs_transfer_with_teacher_distillation_on_lightweight_feedforward_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            teacher_dir = Path(tmp_dir) / "teacher"
            transfer_dir = Path(tmp_dir) / "transfer"

            teacher_env = self.make_env("minimal_basic_strategy")
            teacher_model = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")
            teacher_config = self.make_pipeline_config(teacher_dir)
            teacher_result = train_model(teacher_env, teacher_model, pipeline_config=teacher_config)
            teacher_checkpoint = teacher_dir / "best_eval.pt"

            student_env = self.make_env("table_realistic_default")
            student_model = FeedForwardDoubleDQN.from_profile("table_realistic_default")
            student_config = self.make_pipeline_config(transfer_dir)
            student_config.transfer = TransferLearningConfig(
                enabled=True,
                teacher_checkpoint_path=str(teacher_checkpoint),
                warm_start_checkpoint_path=str(teacher_checkpoint),
                distillation=DistillationConfig(
                    enabled=True,
                    weight=0.5,
                    final_weight=0.25,
                    mode="q_mse",
                    decay_steps=10,
                    playing_only=True,
                ),
            )

            result = train_model(student_env, student_model, pipeline_config=student_config)
            trainer = result["trainer"]
            summary = result["history"][0]

            self.assertIsNotNone(result["warm_start_report"])
            self.assertIn("backbone.0.weight", result["warm_start_report"]["padded"])
            self.assertIsNotNone(trainer.teacher_model)
            self.assertIn("teacher_state_vector", trainer.replay_buffer.storage[0]["state"])
            self.assertTrue(math.isfinite(summary["distillation_loss"]))
            self.assertGreater(summary["distillation_weight"], 0.0)
            self.assertTrue((transfer_dir / "latest.pt").exists())
            self.assertTrue((transfer_dir / "best_eval.pt").exists())
            self.assertGreater(trainer.update_count, 0)
            self.assertGreaterEqual(len(teacher_result["history"]), 1)


if __name__ == "__main__":
    unittest.main()
