from __future__ import annotations

import unittest

import torch

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from enviroment_bj.core import ACTION_ORDER
from model.agents import DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN


class BlackjackAgentArchitectureTests(unittest.TestCase):
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

    def assert_has_gradients(self, model: torch.nn.Module) -> None:
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(
            any(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients)
        )

    def build_realistic_sequences(self) -> list[list[dict]]:
        env_a = self.make_env(observation_profile="table_realistic_default")
        env_a.load_shoe(["8", "6", "8", "10", "3", "K", "2", "10"], total_cards=8)
        seq_a = [env_a.reset(), env_a.step("bet_1x"), env_a.step("split"), env_a.step("double"), env_a.step("stand")]

        env_b = self.make_env(observation_profile="table_realistic_default")
        env_b.load_shoe(["10", "6", "7", "10", "10"], total_cards=5)
        seq_b = [env_b.reset(), env_b.step("bet_1x"), env_b.step("stand")]
        return [seq_a, seq_b]

    def build_unknown_progress_sequences(self) -> list[list[dict]]:
        start_state = StartStateConfig(
            mode="unknown_progress",
            min_burned_rounds=3,
            max_burned_rounds=3,
            hide_reshuffle_progress_from_observation=True,
        )

        env_a = self.make_env(
            observation_profile="table_realistic_unknown_progress",
            start_state=start_state,
        )
        seq_a = [env_a.reset(), env_a.step("bet_1x"), env_a.step("stand")]

        env_b = self.make_env(
            observation_profile="table_realistic_unknown_progress",
            start_state=StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=2,
                max_burned_rounds=2,
                hide_reshuffle_progress_from_observation=True,
            ),
        )
        seq_b = [env_b.reset(), env_b.step("bet_1x"), env_b.step("hit")]
        return [seq_a, seq_b]

    def test_feedforward_double_dqn_forward_and_backward(self) -> None:
        env = self.make_env(observation_profile="minimal_basic_strategy")
        env.load_shoe(["10", "6", "7", "10"], total_cards=4)
        response = env.reset()

        model = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")
        model.train()
        output = model(response)

        self.assertEqual(tuple(output["q_values"].shape), (1, len(ACTION_ORDER)))
        self.assertEqual(tuple(output["bet_q_values"].shape), (1, model.num_bet_actions))
        self.assertEqual(tuple(output["play_q_values"].shape), (1, model.num_play_actions))
        self.assertEqual(tuple(output["masked_q_values"].shape), (1, len(ACTION_ORDER)))
        self.assertEqual(tuple(output["state_vector"].shape), (1, model.state_dim))
        self.assertTrue(torch.equal(output["q_values"][:, model.bet_action_slice], output["bet_q_values"]))
        self.assertTrue(torch.equal(output["q_values"][:, model.play_action_slice], output["play_q_values"]))
        self.assertTrue(torch.equal(output["action_mask"], torch.tensor(response["action_mask"], dtype=torch.bool).unsqueeze(0)))
        illegal_mask = ~output["action_mask"]
        self.assertTrue((output["masked_q_values"][illegal_mask] < -1e20).all().item())

        loss = output["q_values"].mean()
        loss.backward()
        self.assert_has_gradients(model)

    def test_feedforward_supports_optional_phase_adapters_and_module_gating(self) -> None:
        env = self.make_env(observation_profile="table_realistic_default")
        env.load_shoe(["10", "6", "7", "10"], total_cards=4)
        response = env.reset()

        model = FeedForwardDoubleDQN.from_profile(
            "table_realistic_default",
            use_phase_adapters=True,
            use_module_gating=True,
        )
        output = model(response)

        self.assertEqual(tuple(output["q_values"].shape), (1, len(ACTION_ORDER)))
        self.assertEqual(tuple(output["bet_q_values"].shape), (1, model.num_bet_actions))
        self.assertEqual(tuple(output["play_q_values"].shape), (1, model.num_play_actions))

    def test_recurrent_double_dqn_gru_forward_and_backward_with_padding(self) -> None:
        sequences = self.build_realistic_sequences()
        model = RecurrentDoubleDQN.from_profile("table_realistic_default", recurrent_type="gru")
        model.train()

        output = model(sequences)

        self.assertEqual(tuple(output["q_values"].shape[:2]), tuple(output["padding_mask"].shape))
        self.assertEqual(output["q_values"].shape[-1], len(ACTION_ORDER))
        self.assertEqual(output["bet_q_values"].shape[-1], model.num_bet_actions)
        self.assertEqual(output["play_q_values"].shape[-1], model.num_play_actions)
        self.assertEqual(tuple(output["action_mask"].shape), tuple(output["q_values"].shape))
        self.assertEqual(tuple(output["state_vector"].shape[-2:]), (output["q_values"].shape[1], model.state_dim))
        self.assertEqual(tuple(output["hidden_state"].shape), (1, 2, model.config.recurrent_hidden_dim))
        self.assertTrue(torch.equal(output["q_values"][..., model.bet_action_slice], output["bet_q_values"]))
        self.assertTrue(torch.equal(output["q_values"][..., model.play_action_slice], output["play_q_values"]))
        illegal_mask = ~output["action_mask"]
        self.assertTrue((output["masked_q_values"][illegal_mask] < -1e20).all().item())

        valid_steps = output["padding_mask"].unsqueeze(-1).to(output["q_values"].dtype)
        loss = (output["q_values"] * valid_steps).sum() / valid_steps.sum().clamp_min(1.0)
        loss.backward()
        self.assert_has_gradients(model)

    def test_dueling_recurrent_double_dqn_lstm_forward_step_and_backward(self) -> None:
        sequences = self.build_unknown_progress_sequences()
        model = DuelingRecurrentDoubleDQN.from_profile(
            "table_realistic_unknown_progress",
            recurrent_type="lstm",
        )
        model.train()

        batch_output = model(sequences)
        self.assertEqual(tuple(batch_output["q_values"].shape[:2]), tuple(batch_output["padding_mask"].shape))
        self.assertEqual(batch_output["q_values"].shape[-1], len(ACTION_ORDER))
        self.assertEqual(batch_output["bet_q_values"].shape[-1], model.num_bet_actions)
        self.assertEqual(batch_output["play_q_values"].shape[-1], model.num_play_actions)
        self.assertEqual(batch_output["state_value"].shape[-1], 1)
        self.assertEqual(batch_output["bet_state_value"].shape[-1], 1)
        self.assertEqual(batch_output["play_state_value"].shape[-1], 1)
        self.assertEqual(tuple(batch_output["advantages"].shape), tuple(batch_output["q_values"].shape))
        self.assertEqual(tuple(batch_output["bet_advantages"].shape[:-1]), tuple(batch_output["q_values"].shape[:-1]))
        self.assertEqual(tuple(batch_output["play_advantages"].shape[:-1]), tuple(batch_output["q_values"].shape[:-1]))
        self.assertTrue(torch.equal(batch_output["q_values"][..., model.bet_action_slice], batch_output["bet_q_values"]))
        self.assertTrue(torch.equal(batch_output["q_values"][..., model.play_action_slice], batch_output["play_q_values"]))
        self.assertTrue(torch.equal(batch_output["advantages"][..., model.bet_action_slice], batch_output["bet_advantages"]))
        self.assertTrue(torch.equal(batch_output["advantages"][..., model.play_action_slice], batch_output["play_advantages"]))
        self.assertIsInstance(batch_output["hidden_state"], tuple)
        self.assertEqual(len(batch_output["hidden_state"]), 2)

        single_env = self.make_env(
            observation_profile="table_realistic_unknown_progress",
            start_state=StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=1,
                max_burned_rounds=1,
                hide_reshuffle_progress_from_observation=True,
            ),
        )
        step_response = single_env.reset()
        step_output = model.forward_step(step_response, hidden_state=model.init_hidden(batch_size=1))

        self.assertEqual(tuple(step_output["q_values"].shape), (1, len(ACTION_ORDER)))
        self.assertEqual(tuple(step_output["bet_q_values"].shape), (1, model.num_bet_actions))
        self.assertEqual(tuple(step_output["play_q_values"].shape), (1, model.num_play_actions))
        self.assertEqual(tuple(step_output["state_value"].shape), (1, 1))
        self.assertEqual(tuple(step_output["bet_state_value"].shape), (1, 1))
        self.assertEqual(tuple(step_output["play_state_value"].shape), (1, 1))
        self.assertEqual(tuple(step_output["advantages"].shape), (1, len(ACTION_ORDER)))
        self.assertTrue(torch.equal(step_output["q_values"][:, model.bet_action_slice], step_output["bet_q_values"]))
        self.assertTrue(torch.equal(step_output["q_values"][:, model.play_action_slice], step_output["play_q_values"]))
        self.assertNotIn("estimated_shoe_progress", step_response["observation"]["temporal_context"])

        valid_steps = batch_output["padding_mask"].unsqueeze(-1).to(batch_output["q_values"].dtype)
        loss = (batch_output["q_values"] * valid_steps).sum() / valid_steps.sum().clamp_min(1.0)
        loss.backward()
        self.assert_has_gradients(model)


if __name__ == "__main__":
    unittest.main()
