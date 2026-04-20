from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from training.config import PrintConfig
from training.logging import TrainingLogger


class TrainingLoggingTests(unittest.TestCase):
    def test_warmup_logging_uses_interval_instead_of_printing_every_step(self) -> None:
        logger = TrainingLogger(
            PrintConfig(
                enable=True,
                print_warmup_interval=3,
                print_update_interval=10,
                print_collection_interval=10,
            )
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            for size in range(1, 8):
                logger.log_warmup(buffer_size=size, target_size=10)

        output = buffer.getvalue().strip().splitlines()
        self.assertEqual(output, ["[Warmup] buffer 3/10", "[Warmup] buffer 6/10"])

    def test_logger_prints_epoch_update_train_and_eval_formats(self) -> None:
        logger = TrainingLogger(
            PrintConfig(
                enable=True,
                print_warmup_interval=10,
                print_update_interval=2,
                print_collection_interval=5,
            )
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            logger.log_run_summary(
                summary={
                    "architecture": "recurrent",
                    "recurrent_type": "lstm",
                    "encoder_profile": "table_realistic_default",
                    "observation_profile": "table_realistic_default",
                    "start_state_mode": "fresh_shoe",
                    "device": "cpu",
                    "total_epochs": 5,
                    "num_envs": 1,
                    "nominal_env_steps_per_epoch": 2000,
                    "estimated_updates_per_epoch": 500,
                    "parameter_count": 123456,
                    "optimizer": "adam",
                    "learning_rate": 3e-4,
                    "loss_type": "huber",
                    "gamma": 0.99,
                    "gradient_clipping": True,
                    "max_grad_norm": 5.0,
                    "warmup_size": 2000,
                    "buffer_capacity": 100000,
                    "batch_size": 32,
                    "sequence_length": 16,
                    "min_sequence_length": 4,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.05,
                    "epsilon_decay_steps": 100000,
                    "target_update_mode": "hard",
                    "target_hard_interval": 1000,
                    "target_soft_tau": 0.005,
                    "eval_rounds": 500,
                    "eval_max_decisions": 50000,
                    "checkpoint_dir": "checkpoints/run_a",
                    "n_decks": 6,
                    "shoe_penetration": 0.8,
                    "dealer_hits_soft_17": False,
                    "blackjack_payout": 1.5,
                    "double_allowed_on": "any_two_cards",
                    "split_rule": "same_value",
                    "double_after_split_allowed": True,
                }
            )
            logger.log_epoch_start(epoch=1, total_epochs=5)
            logger.log_update(
                update_in_epoch=2,
                total_updates_in_epoch=8,
                metrics={
                    "loss": 0.123456,
                    "mean_q_pred": 0.5,
                    "mean_target": 0.7,
                    "mean_abs_td_error": 0.2,
                    "grad_norm": 1.5,
                    "epsilon": 0.9,
                    "learning_rate": 3e-4,
                    "update_time_sec": 0.25,
                    "buffer_size": 200,
                },
            )
            logger.log_epoch_summary(
                summary={
                    "loss": 0.12,
                    "mean_q_pred": 0.51,
                    "mean_target": 0.72,
                    "mean_abs_td_error": 0.21,
                    "grad_norm": 1.2,
                    "reward_per_round": 0.03,
                    "ev_per_1000_hands": 12.0,
                    "epsilon": 0.8,
                    "learning_rate": 3e-4,
                }
            )
            logger.log_evaluation(
                metrics={
                    "reward_per_round": 0.05,
                    "ev_per_1000_hands": 20.0,
                    "win_rate": 0.42,
                    "push_rate": 0.08,
                    "loss_rate": 0.50,
                    "surrender_rate": 0.01,
                }
            )
            logger.log_checkpoint(
                kind="best_eval",
                path="checkpoints/best_eval.pt",
                metric_name="ev_per_1000_hands",
                metric_value=20.0,
            )
            logger.log_epoch_time(epoch_time_sec=90.0)

        output = buffer.getvalue()
        self.assertIn("Blackjack RL run | arch: recurrent | recurrent: lstm", output)
        self.assertIn("Device: cpu | epochs: 5 | envs: 1 | steps/epoch: 2000 | updates/epoch~: 500", output)
        self.assertIn("=== Epoch 1/5 ===", output)
        self.assertIn("[train step 2/8]", output)
        self.assertIn("[Train] loss 0.120000", output)
        self.assertIn("[Val]   reward/round 0.0500", output)
        self.assertIn("Best saved to checkpoints/best_eval.pt (ev_per_1000_hands 20.0000)", output)
        self.assertIn("Epoch time: 1.50 min", output)


if __name__ == "__main__":
    unittest.main()
