from __future__ import annotations

import unittest

import torch

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from enviroment_bj.core import ACTION_ORDER
from model.encoder import BlackjackObservationEncoder, EncoderConfig


class BlackjackEncoderTests(unittest.TestCase):
    def make_env(
        self,
        *,
        observation_profile: str,
        observation_overrides: dict[str, object] | None = None,
        start_state: StartStateConfig | None = None,
        **config_overrides: object,
    ) -> BlackjackEnvironment:
        observation = ObservationConfig.for_profile(observation_profile)
        for key, value in (observation_overrides or {}).items():
            setattr(observation, key, value)
        observation.__post_init__()

        config_kwargs = {
            "n_decks": 1,
            "shoe_penetration": 1.0,
            "observation": observation,
        }
        config_kwargs.update(config_overrides)
        return BlackjackEnvironment(config=BlackjackConfig(**config_kwargs), seed=11, start_state=start_state)

    def test_minimal_encoder_returns_single_step_tensors_with_expected_shapes(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        env.load_shoe(["10", "6", "7", "10"], total_cards=4)
        response = env.reset()

        encoder = BlackjackObservationEncoder.from_profile("minimal_basic_strategy")
        encoded = encoder(response)

        self.assertEqual(encoded["state_vector"].dtype, torch.float32)
        self.assertEqual(encoded["action_mask"].dtype, torch.bool)
        self.assertEqual(tuple(encoded["state_vector"].shape), (encoder.state_dim,))
        self.assertEqual(tuple(encoded["action_mask"].shape), (len(ACTION_ORDER),))
        self.assertEqual(set(encoded["module_tensors"].keys()), {"hand", "hand_context", "insurance", "betting_context", "rules"})
        self.assertEqual(encoded["metadata"]["profile"], "minimal_basic_strategy")
        self.assertEqual(encoded["metadata"]["observation_profile"], "minimal_basic_strategy")
        self.assertEqual(encoded["metadata"]["decision_phase"], "betting")
        self.assertEqual(encoded["metadata"]["available_bet_multipliers"], [1, 2, 3, 4])

    def test_betting_context_encoder_distinguishes_betting_and_playing_phase(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        env.load_shoe(["10", "6", "7", "10"], total_cards=4)

        betting = env.reset()
        playing = env.step("bet_2x")
        encoder = BlackjackObservationEncoder.from_profile("minimal_basic_strategy")

        betting_encoded = encoder(betting)
        playing_encoded = encoder(playing)
        betting_tensor = betting_encoded["module_tensors"]["betting_context"]
        playing_tensor = playing_encoded["module_tensors"]["betting_context"]

        self.assertEqual(tuple(betting_tensor.shape), (9,))
        self.assertEqual(float(betting_tensor[0].item()), 1.0)
        self.assertEqual(float(betting_tensor[1].item()), 0.0)
        self.assertEqual(float(betting_tensor[2].item()), 1.0)
        self.assertEqual(float(betting_tensor[-2].item()), 0.0)
        self.assertEqual(float(playing_tensor[0].item()), 0.0)
        self.assertEqual(float(playing_tensor[1].item()), 1.0)
        self.assertEqual(float(playing_tensor[2].item()), 0.0)
        self.assertEqual(float(playing_tensor[-2].item()), 1.0)
        self.assertEqual(float(playing_tensor[-1].item()), 2.0)

    def test_realistic_encoder_encodes_history_temporal_and_other_hands(self) -> None:
        env = self.make_env(observation_profile="table_realistic_default")
        env.load_shoe(["8", "6", "8", "10", "3", "K", "2", "10"], total_cards=8)
        encoder = BlackjackObservationEncoder.from_profile("table_realistic_default")

        start = env.reset()
        env.step("bet_1x")
        split_response = env.step("split")
        encoded = encoder(split_response)

        self.assertEqual(encoded["metadata"]["observation_profile"], "table_realistic_default")
        self.assertIn("other_hands", encoded["module_tensors"])
        self.assertIn("betting_context", encoded["module_tensors"])
        self.assertIn("observed_history", encoded["module_tensors"])
        self.assertIn("discard_summary", encoded["module_tensors"])
        self.assertIn("temporal", encoded["module_tensors"])
        self.assertEqual(tuple(encoded["state_vector"].shape), (encoder.state_dim,))
        self.assertTrue(torch.count_nonzero(encoded["module_tensors"]["other_hands"]).item() > 0)
        self.assertTrue(torch.count_nonzero(encoded["module_tensors"]["observed_history"]).item() > 0)
        self.assertTrue(torch.equal(encoded["action_mask"], torch.tensor(split_response["action_mask"], dtype=torch.bool)))
        self.assertEqual(start["observation"]["profile"], "table_realistic_default")

    def test_full_encoder_includes_exact_shoe_and_hidden_rules(self) -> None:
        env = self.make_env(observation_profile="fully_observable_sim")
        env.load_shoe(["10", "6", "7", "10", "5", "2"], total_cards=6)
        env.reset()
        response = env.step("bet_1x")
        encoder = BlackjackObservationEncoder.from_profile("fully_observable_sim")

        encoded = encoder(response)

        self.assertIn("exact_shoe", encoded["module_tensors"])
        self.assertIn("n_decks", response["table_rules"])
        self.assertIn("dealer_peeks_for_blackjack", response["table_rules"])
        self.assertEqual(tuple(encoded["module_tensors"]["exact_shoe"].shape), (13,))
        self.assertEqual(tuple(encoded["module_tensors"]["rules"].shape), (22,))
        self.assertAlmostEqual(float(encoded["module_tensors"]["exact_shoe"].sum().item()), 1.0, places=5)

    def test_unknown_progress_encoder_profile_handles_hidden_start_without_shape_changes(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_unknown_progress",
            start_state=StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=4,
                max_burned_rounds=4,
                hide_reshuffle_progress_from_observation=True,
            ),
        )
        env.reset()
        response = env.step("bet_1x")
        encoder = BlackjackObservationEncoder.from_profile("table_realistic_unknown_progress")

        encoded = encoder(response)

        self.assertEqual(encoded["metadata"]["profile"], "table_realistic_unknown_progress")
        self.assertEqual(encoded["metadata"]["observation_profile"], "table_realistic_unknown_progress")
        self.assertEqual(tuple(encoded["state_vector"].shape), (encoder.state_dim,))
        self.assertIn("temporal", encoded["module_tensors"])
        self.assertIn("observed_history", encoded["module_tensors"])
        self.assertIn("discard_summary", encoded["module_tensors"])
        self.assertTrue(torch.count_nonzero(encoded["module_tensors"]["temporal"]).item() > 0)
        self.assertNotIn("estimated_shoe_progress", response["observation"]["temporal_context"])

    def test_temporal_encoder_includes_observed_shuffle_signal(self) -> None:
        env = self.make_env(observation_profile="table_realistic_default", visible_shoe_change=True)
        env.load_shoe(["10", "6", "7", "10", "10", "9", "5", "2"], total_cards=8)
        encoder = BlackjackObservationEncoder.from_profile("table_realistic_default")

        before = env.reset()
        env.mark_observed_shuffle_reset()
        after = env.step("bet_1x")

        before_encoded = encoder(before)
        after_encoded = encoder(after)
        before_temporal = before_encoded["module_tensors"]["temporal"]
        after_temporal = after_encoded["module_tensors"]["temporal"]

        self.assertEqual(float(before_temporal[-3].item()), 0.0)
        self.assertEqual(float(before_temporal[-2].item()), 0.0)
        self.assertEqual(float(before_temporal[-1].item()), 0.0)
        self.assertEqual(float(after_temporal[-3].item()), 1.0)
        self.assertEqual(float(after_temporal[-2].item()), 1.0)
        self.assertEqual(float(after_temporal[-1].item()), 0.0)

    def test_encode_batch_stacks_responses_into_bxd(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        env.load_shoe(["10", "6", "7", "10", "9", "5", "2", "10"], total_cards=8)
        first = env.reset()
        second = env.step("bet_1x")

        encoder = BlackjackObservationEncoder.from_profile("minimal_basic_strategy")
        batch = encoder.encode_batch([first, second])

        self.assertEqual(tuple(batch["state_vector"].shape), (2, encoder.state_dim))
        self.assertEqual(tuple(batch["action_mask"].shape), (2, len(ACTION_ORDER)))
        self.assertEqual(tuple(batch["module_tensors"]["hand"].shape), (2, encoder.module_dims["hand"]))
        self.assertEqual(batch["metadata"]["batch_size"], 2)

    def test_encode_sequence_batch_pads_variable_length_sequences(self) -> None:
        encoder = BlackjackObservationEncoder.from_profile("minimal_basic_strategy")

        env_a = self.make_env(observation_profile="minimal_basic_strategy")
        env_a.load_shoe(["10", "6", "7", "10", "9", "5", "2", "10"], total_cards=8)
        seq_a = [env_a.reset(), env_a.step("bet_1x"), env_a.step("stand")]

        env_b = self.make_env(observation_profile="minimal_basic_strategy")
        env_b.load_shoe(["9", "7", "7", "10"], total_cards=4)
        seq_b = [env_b.reset(), env_b.step("bet_1x")]

        batch = encoder.encode_sequence_batch([seq_a, seq_b])

        self.assertEqual(tuple(batch["state_vector"].shape), (2, 3, encoder.state_dim))
        self.assertEqual(tuple(batch["action_mask"].shape), (2, 3, len(ACTION_ORDER)))
        self.assertEqual(tuple(batch["padding_mask"].shape), (2, 3))
        self.assertTrue(batch["padding_mask"][0, 0].item())
        self.assertTrue(batch["padding_mask"][0, 1].item())
        self.assertTrue(batch["padding_mask"][0, 2].item())
        self.assertTrue(batch["padding_mask"][1, 0].item())
        self.assertTrue(batch["padding_mask"][1, 1].item())
        self.assertFalse(batch["padding_mask"][1, 2].item())
        self.assertEqual(batch["metadata"]["sequence_lengths"], [3, 2])

    def test_encoder_state_dim_is_stable_across_round_progression(self) -> None:
        env = self.make_env(
            observation_profile="table_realistic_default",
            observation_overrides={"obs_include_recent_actions": True, "obs_recent_actions_window": 8},
        )
        env.load_shoe(
            ["10", "6", "7", "10", "10", "9", "5", "2", "10", "K", "8"],
            total_cards=11,
        )
        encoder = BlackjackObservationEncoder.from_profile(
            "table_realistic_default",
            encode_recent_actions=True,
            max_recent_actions=8,
        )

        start = env.reset()
        env.step("bet_1x")
        end = env.step("stand")
        next_round = env.reset()

        encoded_start = encoder(start)
        encoded_end = encoder(end)
        encoded_next = encoder(next_round)

        self.assertEqual(tuple(encoded_start["state_vector"].shape), (encoder.state_dim,))
        self.assertEqual(tuple(encoded_end["state_vector"].shape), (encoder.state_dim,))
        self.assertEqual(tuple(encoded_next["state_vector"].shape), (encoder.state_dim,))
        self.assertIn("temporal", encoded_next["module_tensors"])
        self.assertTrue(torch.count_nonzero(encoded_next["module_tensors"]["temporal"]).item() > 0)


if __name__ == "__main__":
    unittest.main()
