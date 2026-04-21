from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import torch

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from model.agents import DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN
from training import (
    CheckpointConfig,
    DualEpsilonConfig,
    EpsilonScheduleConfig,
    EvaluationConfig,
    NStepConfig,
    OptimizationConfig,
    PrintConfig,
    ReplayBufferConfig,
    TargetUpdateConfig,
    TrainerConfig,
    TrainingPipelineConfig,
    build_trainer,
    train_model,
    train_one_epoch,
)
from training.epsilon import DualEpsilonScheduler
from training.step import move_training_batch_to_device


class BlackjackTrainingPipelineTests(unittest.TestCase):
    def make_env(
        self,
        *,
        observation_profile: str,
        start_state: StartStateConfig | None = None,
        **config_overrides: object,
    ) -> BlackjackEnvironment:
        observation = ObservationConfig.for_profile(observation_profile)
        config_kwargs = {
            "n_decks": 1,
            "shoe_penetration": 1.0,
            "observation": observation,
        }
        config_kwargs.update(config_overrides)
        return BlackjackEnvironment(config=BlackjackConfig(**config_kwargs), seed=11, start_state=start_state)

    def make_pipeline_config(self, checkpoint_dir: Path, *, recurrent: bool) -> TrainingPipelineConfig:
        return TrainingPipelineConfig(
            trainer=TrainerConfig(
                total_epochs=1,
                env_steps_per_epoch=18 if not recurrent else 20,
                train_frequency=1 if not recurrent else 2,
                updates_per_train_step=1,
                max_updates_per_epoch=4,
                device="cpu",
                seed=13,
                reset_hidden_on_round_end=False,
                sequence_end_on_done=False,
                flush_partial_sequences_at_epoch_end=True,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=256,
                batch_size=4 if not recurrent else 1,
                warmup_size=8 if not recurrent else 2,
                sequence_length=4 if recurrent else 8,
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
                save_periodic=True,
                periodic_interval_updates=2,
            ),
            prints=PrintConfig(enable=False),
        )

    def assert_finite_model(self, model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_move_training_batch_to_device_moves_nested_tensors(self) -> None:
        device = torch.device("cpu")
        batch = {
            "action": torch.tensor([1, 2], dtype=torch.long),
            "reward": torch.tensor([0.5, -1.0], dtype=torch.float32),
            "nested": {
                "padding_mask": torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
                "states": [torch.tensor([1.0]), {"mask": torch.tensor([0, 1], dtype=torch.bool)}],
            },
            "python_only": [{"foo": "bar"}],
        }

        moved = move_training_batch_to_device(batch, device)

        self.assertEqual(moved["action"].device.type, "cpu")
        self.assertEqual(moved["reward"].device.type, "cpu")
        self.assertEqual(moved["nested"]["padding_mask"].device.type, "cpu")
        self.assertEqual(moved["nested"]["states"][0].device.type, "cpu")
        self.assertEqual(moved["nested"]["states"][1]["mask"].device.type, "cpu")
        self.assertEqual(moved["python_only"][0]["foo"], "bar")

    def test_dual_epsilon_scheduler_steps_betting_and_playing_independently(self) -> None:
        scheduler = DualEpsilonScheduler(
            DualEpsilonConfig(
                betting=EpsilonScheduleConfig(start=1.0, end=0.2, decay_steps=10, evaluation_epsilon=0.05),
                playing=EpsilonScheduleConfig(start=0.8, end=0.1, decay_steps=4, evaluation_epsilon=0.0),
            )
        )

        scheduler.step("betting", 2)
        scheduler.step("playing", 1)

        self.assertAlmostEqual(scheduler.value("betting"), 0.84, places=6)
        self.assertAlmostEqual(scheduler.value("playing"), 0.625, places=6)
        self.assertAlmostEqual(scheduler.evaluation_value("betting"), 0.05, places=6)
        self.assertAlmostEqual(scheduler.evaluation_value("playing"), 0.0, places=6)

    def test_trainer_builds_discounted_n_step_transition_when_enabled(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        model = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.make_pipeline_config(Path(tmp_dir), recurrent=False)
            config.n_step = NStepConfig(enabled=True, n_steps=3)
            config.trainer.loss.gamma = 0.5
            trainer = build_trainer(env, model, pipeline_config=config)

            transition = trainer._build_n_step_transition(
                [
                    {"state": {"id": 0}, "next_state": {"id": 1}, "action": 0, "reward": 1.0, "done": False, "action_mask": torch.ones(10, dtype=torch.bool), "next_action_mask": torch.ones(10, dtype=torch.bool)},
                    {"state": {"id": 1}, "next_state": {"id": 2}, "action": 4, "reward": 2.0, "done": False, "action_mask": torch.ones(10, dtype=torch.bool), "next_action_mask": torch.ones(10, dtype=torch.bool)},
                    {"state": {"id": 2}, "next_state": {"id": 3}, "action": 5, "reward": 3.0, "done": True, "action_mask": torch.ones(10, dtype=torch.bool), "next_action_mask": torch.zeros(10, dtype=torch.bool)},
                ]
            )

            self.assertEqual(transition["state"], {"id": 0})
            self.assertEqual(transition["next_state"], {"id": 3})
            self.assertEqual(transition["n_steps"], 3)
            self.assertTrue(transition["done"])
            self.assertAlmostEqual(float(transition["reward"]), 2.75, places=6)

    def test_train_model_runs_feedforward_pipeline_and_saves_checkpoints(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        model = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.make_pipeline_config(Path(tmp_dir), recurrent=False)
            result = train_model(env, model, pipeline_config=config)
            trainer = result["trainer"]

            self.assertEqual(len(result["history"]), 1)
            self.assertGreater(trainer.update_count, 0)
            self.assertGreaterEqual(len(trainer.replay_buffer), config.replay_buffer.warmup_size)
            self.assertTrue(math.isfinite(result["history"][0]["loss"]))
            self.assertTrue(math.isfinite(result["history"][0]["grad_norm"]))
            self.assertIsNotNone(result["history"][0]["eval"])
            self.assertTrue((Path(tmp_dir) / "latest.pt").exists())
            self.assertTrue((Path(tmp_dir) / "best_eval.pt").exists())
            self.assertTrue(any(path.name.startswith("step_") for path in Path(tmp_dir).glob("*.pt")))
            self.assert_finite_model(trainer.online_network)

    def test_replay_buffer_stores_compact_encoded_states_instead_of_raw_responses(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        model = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.make_pipeline_config(Path(tmp_dir), recurrent=False)
            config.replay_buffer.warmup_size = 4
            trainer = build_trainer(env, model, pipeline_config=config)
            trainer.warmup()

            transition = trainer.replay_buffer.storage[0]
            self.assertIn("state_vector", transition["state"])
            self.assertIn("action_mask", transition["state"])
            self.assertNotIn("observation", transition["state"])
            self.assertNotIn("info", transition["state"])
            self.assertEqual(transition["state"]["state_vector"].device.type, "cpu")
            self.assertEqual(transition["next_state"]["state_vector"].device.type, "cpu")

    def test_train_one_epoch_runs_recurrent_pipeline_with_gru(self) -> None:
        env = self.make_env(observation_profile="table_realistic_default")
        model = RecurrentDoubleDQN.from_profile("table_realistic_default", recurrent_type="gru")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.make_pipeline_config(Path(tmp_dir), recurrent=True)
            trainer = build_trainer(env, model, pipeline_config=config)
            summary = train_one_epoch(trainer)

            self.assertGreater(trainer.update_count, 0)
            self.assertGreaterEqual(len(trainer.replay_buffer), config.replay_buffer.warmup_size)
            self.assertTrue(math.isfinite(summary["loss"]))
            self.assertTrue(math.isfinite(summary["grad_norm"]))
            self.assertGreater(summary["updates_this_epoch"], 0)
            self.assertIsNotNone(summary["eval"])
            self.assertGreater(summary["eval"]["rounds_completed"], 0)
            self.assertTrue((Path(tmp_dir) / "latest.pt").exists())
            self.assert_finite_model(trainer.online_network)

    def test_train_model_runs_dueling_recurrent_unknown_progress_pipeline(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_unknown_progress",
            start_state=StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=2,
                max_burned_rounds=2,
                hide_reshuffle_progress_from_observation=True,
            ),
        )
        model = DuelingRecurrentDoubleDQN.from_profile(
            "table_realistic_unknown_progress",
            recurrent_type="lstm",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.make_pipeline_config(Path(tmp_dir), recurrent=True)
            result = train_model(env, model, pipeline_config=config)
            trainer = result["trainer"]
            summary = result["history"][0]

            self.assertGreater(trainer.update_count, 0)
            self.assertTrue(math.isfinite(summary["loss"]))
            self.assertTrue(math.isfinite(summary["grad_norm"]))
            self.assertTrue(math.isfinite(summary["reward_per_round"]))
            self.assertIsNotNone(summary["eval"])
            self.assertTrue(math.isfinite(summary["eval"]["ev_per_1000_hands"]))
            self.assertTrue((Path(tmp_dir) / "latest.pt").exists())
            self.assertTrue((Path(tmp_dir) / "best_eval.pt").exists())
            self.assert_finite_model(trainer.online_network)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_train_one_epoch_runs_on_cuda_without_device_mismatch(self) -> None:
        env = self.make_env(observation_profile="table_realistic_default")
        model = RecurrentDoubleDQN.from_profile("table_realistic_default", recurrent_type="lstm")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self.make_pipeline_config(Path(tmp_dir), recurrent=True)
            config.trainer.device = "cuda"
            config.trainer.env_steps_per_epoch = 8
            config.trainer.max_updates_per_epoch = 2
            config.replay_buffer.batch_size = 1
            config.replay_buffer.warmup_size = 2
            trainer = build_trainer(env, model, pipeline_config=config)
            summary = train_one_epoch(trainer)

            self.assertEqual(next(trainer.online_network.parameters()).device.type, "cuda")
            self.assertGreater(trainer.update_count, 0)
            self.assertTrue(math.isfinite(summary["loss"]))
            self.assertTrue(math.isfinite(summary["grad_norm"]))
            self.assert_finite_model(trainer.online_network)


if __name__ == "__main__":
    unittest.main()
