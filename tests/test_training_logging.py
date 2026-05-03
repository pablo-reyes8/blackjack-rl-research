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
                    "epsilon_betting_start": 1.0,
                    "epsilon_betting_end": 0.10,
                    "epsilon_betting_decay_steps": 120000,
                    "epsilon_playing_start": 1.0,
                    "epsilon_playing_end": 0.03,
                    "epsilon_playing_decay_steps": 80000,
                    "target_update_mode": "hard",
                    "target_hard_interval": 1000,
                    "target_soft_tau": 0.005,
                    "eval_rounds": 500,
                    "eval_max_decisions": 50000,
                    "checkpoint_dir": "checkpoints/run_a",
                    "n_step_enabled": True,
                    "n_step_size": 3,
                    "phase_loss_weights_enabled": True,
                    "betting_loss_weight": 1.5,
                    "playing_loss_weight": 1.0,
                    "use_module_gating": False,
                    "use_phase_adapters": False,
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
                    "mean_abs_td_error_betting": 0.3,
                    "mean_abs_td_error_playing": 0.1,
                    "grad_norm": 1.5,
                    "epsilon_betting": 0.9,
                    "epsilon_playing": 0.7,
                    "loss_betting": 0.2,
                    "loss_playing": 0.1,
                    "mean_n_steps": 1.5,
                    "mean_phase_weight": 1.2,
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
                    "round_reward_std": 1.25,
                    "win_rate": 0.40,
                    "push_rate": 0.08,
                    "loss_rate": 0.52,
                    "blackjack_rate": 0.05,
                    "bust_rate": 0.18,
                    "epsilon_betting": 0.8,
                    "epsilon_playing": 0.6,
                    "betting_decisions": 10,
                    "playing_decisions": 30,
                    "random_action_fraction_betting": 0.4,
                    "random_action_fraction_playing": 0.2,
                    "loss_betting": 0.22,
                    "loss_playing": 0.11,
                    "conservative_bet_fraction": 0.50,
                    "aggressive_bet_fraction": 0.20,
                    "bet_action_frequencies": {"bet_1x": 0.5, "bet_2x": 0.3, "bet_3x": 0.1, "bet_4x": 0.1},
                    "play_action_frequencies": {"stand": 0.4, "hit": 0.3, "double": 0.1, "split": 0.1, "surrender": 0.05, "insurance": 0.05},
                    "bet_ev_per_1000_rounds_by_action": {"bet_1x": 10.0, "bet_2x": 20.0, "bet_3x": -5.0, "bet_4x": -15.0},
                    "learning_rate": 3e-4,
                }
            )
            logger.log_evaluation(
                metrics={
                    "reward_per_round": 0.05,
                    "ev_per_1000_hands": 20.0,
                    "round_reward_std": 1.75,
                    "win_rate": 0.42,
                    "push_rate": 0.08,
                    "loss_rate": 0.50,
                    "surrender_rate": 0.01,
                    "blackjack_rate": 0.04,
                    "bust_rate": 0.15,
                    "random_action_fraction_betting": 0.1,
                    "random_action_fraction_playing": 0.05,
                    "conservative_bet_fraction": 0.60,
                    "aggressive_bet_fraction": 0.20,
                    "insurance_reward_total": 1.5,
                    "available_bet_multipliers": [1, 2, 3, 4],
                    "bet_action_frequencies": {"bet_1x": 0.6, "bet_2x": 0.2, "bet_3x": 0.1, "bet_4x": 0.1},
                    "play_action_frequencies": {"stand": 0.45, "hit": 0.25, "double": 0.1, "split": 0.1, "surrender": 0.05, "insurance": 0.05},
                    "bet_ev_per_1000_rounds_by_action": {"bet_1x": 15.0, "bet_2x": 30.0, "bet_3x": 5.0, "bet_4x": -10.0},
                    "ev_calibration_min_samples_to_report": 10,
                    "ev_by_count_bucket_and_bet": {
                        "high": {
                            "bet_1x": {"n": 20, "ev_per_1000": -30.0},
                            "bet_2x": {"n": 25, "ev_per_1000": 12.5},
                            "bet_3x": {"n": 4, "ev_per_1000": 80.0},
                            "bet_4x": {"n": 0, "ev_per_1000": 0.0},
                        }
                    },
                    "mean_q_bet_1x": 0.12,
                    "mean_q_bet_2x": 0.03,
                    "mean_q_bet_3x": -0.04,
                    "mean_q_bet_4x": -0.08,
                    "mean_margin_best_aggressive_vs_1x": -0.09,
                    "count_proxy_valid_states": 120,
                    "count_proxy_mean": 1.35,
                    "count_proxy_p10": -0.75,
                    "count_proxy_p50": 1.10,
                    "count_proxy_p90": 3.85,
                    "count_proxy_target_bet_distribution": {"bet_1x": 0.45, "bet_2x": 0.25, "bet_3x": 0.20, "bet_4x": 0.10},
                    "count_proxy_bucket_stats": {
                        "high": {
                            "n_states": 18.0,
                            "mean_q_bet_1x": -0.10,
                            "mean_q_bet_2x": 0.02,
                            "mean_q_bet_3x": 0.08,
                            "mean_q_bet_4x": 0.01,
                            "greedy_bet_1x_frac": 0.20,
                            "greedy_bet_2x_frac": 0.25,
                            "greedy_bet_3x_frac": 0.45,
                            "greedy_bet_4x_frac": 0.10,
                            "mean_margin_best_aggressive_vs_1x": 0.18,
                        }
                    },
                }
            )
            logger.log_train_val_comparison(
                train_metrics={
                    "ev_per_1000_hands": 12.0,
                    "reward_per_round": 0.03,
                    "round_reward_std": 1.25,
                    "conservative_bet_fraction": 0.50,
                    "aggressive_bet_fraction": 0.20,
                    "play_action_frequencies": {"double": 0.10, "insurance": 0.05},
                },
                eval_metrics={
                    "ev_per_1000_hands": 20.0,
                    "reward_per_round": 0.05,
                    "round_reward_std": 1.75,
                    "conservative_bet_fraction": 0.60,
                    "aggressive_bet_fraction": 0.20,
                    "play_action_frequencies": {"double": 0.12, "insurance": 0.02},
                },
            )
            logger.log_checkpoint(
                kind="best_eval",
                path="checkpoints/best_eval.pt",
                metric_name="ev_per_1000_hands",
                metric_value=20.0,
            )
            logger.log_epoch_time(epoch_time_sec=90.0)

        output = buffer.getvalue()
        self.assertIn("BLACKJACK RL RUN", output)
        self.assertIn("arch=recurrent | recurrent=lstm", output)
        self.assertIn("device=cpu | epochs=5 | envs=1 | steps/epoch=2000", output)
        self.assertIn("eps_bet=1.000->0.100", output)
        self.assertIn("eps_play=1.000->0.030", output)
        self.assertIn("=== Epoch 1/5 ===", output)
        self.assertIn("TRAIN STEP 2/8", output)
        self.assertIn("loss_bet=0.200000 | loss_play=0.100000", output)
        self.assertIn("TRAIN EPOCH SUMMARY", output)
        self.assertIn("bet_1x:0.50 bet_2x:0.30 bet_3x:0.10 bet_4x:0.10", output)
        self.assertIn("VAL", output)
        self.assertIn("blackjack=0.0400 | bust=0.1500", output)
        self.assertIn("mean_q_bet_1x=+0.1200", output)
        self.assertIn("mean_q_bet_4x=-0.0800", output)
        self.assertIn("mean_margin_best_aggressive_vs_1x=-0.0900", output)
        self.assertIn("EV Buckets", output)
        self.assertIn("high      bet_1x:-30.0(n=20) bet_2x:+12.5(n=25)", output)
        self.assertIn("proxy_mean=+1.350", output)
        self.assertIn("target bet_1x:0.45 bet_2x:0.25 bet_3x:0.20 bet_4x:0.10", output)
        self.assertIn("high      n=18", output)
        self.assertIn("margin=+0.1800", output)
        self.assertIn("stand:0.45 hit:0.25 double:0.10 split:0.10 surrender:0.05 insurance:0.05", output)
        self.assertIn("TRAIN vs VAL", output)
        self.assertIn("EV_gap=+8.00", output)
        self.assertIn("Best saved to checkpoints/best_eval.pt (ev_per_1000_hands 20.0000)", output)
        self.assertIn("Epoch time: 1.50 min", output)

    def test_logger_hides_bet_q_diagnostics_when_single_bet_multiplier(self) -> None:
        logger = TrainingLogger(PrintConfig(enable=True))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            logger.log_evaluation(
                metrics={
                    "reward_per_round": 0.01,
                    "ev_per_1000_hands": 5.0,
                    "round_reward_std": 1.0,
                    "win_rate": 0.4,
                    "push_rate": 0.1,
                    "loss_rate": 0.5,
                    "blackjack_rate": 0.03,
                    "bust_rate": 0.15,
                    "random_action_fraction_betting": 0.0,
                    "random_action_fraction_playing": 0.0,
                    "conservative_bet_fraction": 1.0,
                    "aggressive_bet_fraction": 0.0,
                    "insurance_reward_total": 0.0,
                    "available_bet_multipliers": [1],
                    "bet_action_frequencies": {"bet_1x": 1.0, "bet_2x": 0.0, "bet_3x": 0.0, "bet_4x": 0.0},
                    "play_action_frequencies": {"stand": 0.5, "hit": 0.3, "double": 0.1, "split": 0.05, "surrender": 0.05, "insurance": 0.0},
                    "bet_ev_per_1000_rounds_by_action": {"bet_1x": 5.0, "bet_2x": 0.0, "bet_3x": 0.0, "bet_4x": 0.0},
                    "mean_q_bet_1x": 0.2,
                    "mean_margin_best_aggressive_vs_1x": 0.0,
                }
            )

        output = buffer.getvalue()
        self.assertNotIn("mean_q_bet_1x", output)
        self.assertNotIn("mean_margin_best_aggressive_vs_1x", output)


if __name__ == "__main__":
    unittest.main()
