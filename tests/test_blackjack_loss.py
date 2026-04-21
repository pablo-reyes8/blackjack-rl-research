from __future__ import annotations

import unittest

import torch
from torch import nn

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from enviroment_bj.core import ACTION_ORDER
from loss import (
    BellmanLossConfig,
    LossPhaseWeightConfig,
    compute_double_dqn_targets_feedforward,
    compute_td_loss_feedforward,
    compute_td_loss_recurrent,
)
from model.agents import DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN


class FixedFeedForwardNet(nn.Module):
    def __init__(self, outputs_by_phase: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.outputs_by_phase = {key: value.to(torch.float32) for key, value in outputs_by_phase.items()}

    def forward(self, inputs: dict) -> dict[str, torch.Tensor]:
        q_values = self.outputs_by_phase[inputs["phase"]]
        return {
            "q_values": q_values,
            "masked_q_values": q_values,
            "action_mask": torch.ones_like(q_values, dtype=torch.bool),
        }


class FixedRecurrentNet(nn.Module):
    def __init__(self, outputs_by_phase: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.outputs_by_phase = {key: value.to(torch.float32) for key, value in outputs_by_phase.items()}

    def forward(self, inputs: dict) -> dict[str, torch.Tensor]:
        q_values = self.outputs_by_phase[inputs["phase"]]
        return {
            "q_values": q_values,
            "masked_q_values": q_values,
            "action_mask": torch.ones_like(q_values, dtype=torch.bool),
            "padding_mask": torch.ones(q_values.shape[:-1], dtype=torch.bool),
        }


class BlackjackBellmanLossTests(unittest.TestCase):
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

    def assert_only_online_has_gradients(self, online_network: nn.Module, target_network: nn.Module) -> None:
        self.assertTrue(any(parameter.grad is not None for parameter in online_network.parameters() if parameter.requires_grad))
        self.assertTrue(all(parameter.grad is None for parameter in target_network.parameters() if parameter.requires_grad))

    def build_feedforward_transition_batch(self) -> dict[str, object]:
        env_a = self.make_env(observation_profile="minimal_basic_strategy")
        env_a.load_shoe(["10", "6", "7", "10", "10"], total_cards=5)
        env_a.reset()
        state_a = env_a.step("bet_1x")
        next_a = env_a.step("stand")

        env_b = self.make_env(observation_profile="minimal_basic_strategy")
        env_b.load_shoe(["10", "9", "6", "7"], total_cards=4)
        env_b.reset()
        state_b = env_b.step("bet_1x")
        next_b = env_b.step("surrender")

        return {
            "state": [state_a, state_b],
            "next_state": [next_a, next_b],
            "action": torch.tensor([ACTION_ORDER.index("stand"), ACTION_ORDER.index("surrender")], dtype=torch.long),
            "reward": torch.tensor([next_a["reward"], next_b["reward"]], dtype=torch.float32),
            "done": torch.tensor([next_a["done"], next_b["done"]], dtype=torch.bool),
        }

    def build_recurrent_transition_batch(self, *, unknown_progress: bool = False) -> dict[str, object]:
        profile = "table_realistic_unknown_progress" if unknown_progress else "table_realistic_default"
        start_state = None
        if unknown_progress:
            start_state = StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=2,
                max_burned_rounds=2,
                hide_reshuffle_progress_from_observation=True,
            )

        env_a = self.make_env(observation_profile=profile, start_state=start_state)
        if not unknown_progress:
            env_a.load_shoe(["8", "6", "8", "10", "3", "K", "2", "10"], total_cards=8)
        a0 = env_a.reset()
        a1 = env_a.step("bet_1x")
        seq_a = []
        if not unknown_progress:
            a2 = env_a.step("split")
            a3 = env_a.step("double")
            a4 = env_a.step("stand")
            seq_a = [
                {
                    "state": a0,
                    "next_state": a1,
                    "action": ACTION_ORDER.index("bet_1x"),
                    "reward": a1["reward"],
                    "done": a1["done"],
                },
                {
                    "state": a1,
                    "next_state": a2,
                    "action": ACTION_ORDER.index("split"),
                    "reward": a2["reward"],
                    "done": a2["done"],
                },
                {
                    "state": a2,
                    "next_state": a3,
                    "action": ACTION_ORDER.index("double"),
                    "reward": a3["reward"],
                    "done": a3["done"],
                },
                {
                    "state": a3,
                    "next_state": a4,
                    "action": ACTION_ORDER.index("stand"),
                    "reward": a4["reward"],
                    "done": a4["done"],
                },
            ]
        else:
            a2 = env_a.step("stand")
            seq_a = [
                {
                    "state": a0,
                    "next_state": a1,
                    "action": ACTION_ORDER.index("bet_1x"),
                    "reward": a1["reward"],
                    "done": a1["done"],
                },
                {
                    "state": a1,
                    "next_state": a2,
                    "action": ACTION_ORDER.index("stand"),
                    "reward": a2["reward"],
                    "done": a2["done"],
                }
            ]

        env_b = self.make_env(
            observation_profile=profile,
            start_state=StartStateConfig(
                mode="unknown_progress",
                min_burned_rounds=1,
                max_burned_rounds=1,
                hide_reshuffle_progress_from_observation=True,
            )
            if unknown_progress
            else None,
        )
        if not unknown_progress:
            env_b.load_shoe(["10", "6", "7", "10", "10"], total_cards=5)
        b0 = env_b.reset()
        b1 = env_b.step("bet_1x")
        b2 = env_b.step("stand")
        seq_b = [
            {
                "state": b0,
                "next_state": b1,
                "action": ACTION_ORDER.index("bet_1x"),
                "reward": b1["reward"],
                "done": b1["done"],
            },
            {
                "state": b1,
                "next_state": b2,
                "action": ACTION_ORDER.index("stand"),
                "reward": b2["reward"],
                "done": b2["done"],
            }
        ]

        sequences = [seq_a, seq_b]
        max_len = max(len(sequence) for sequence in sequences)
        action = torch.zeros((len(sequences), max_len), dtype=torch.long)
        reward = torch.zeros((len(sequences), max_len), dtype=torch.float32)
        done = torch.ones((len(sequences), max_len), dtype=torch.bool)
        padding_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)

        state_batch: list[list[dict]] = []
        next_state_batch: list[list[dict]] = []
        for batch_index, sequence in enumerate(sequences):
            state_batch.append([item["state"] for item in sequence])
            next_state_batch.append([item["next_state"] for item in sequence])
            for time_index, item in enumerate(sequence):
                action[batch_index, time_index] = item["action"]
                reward[batch_index, time_index] = item["reward"]
                done[batch_index, time_index] = item["done"]
                padding_mask[batch_index, time_index] = True

        return {
            "state": state_batch,
            "next_state": next_state_batch,
            "action": action,
            "reward": reward,
            "done": done,
            "padding_mask": padding_mask,
        }

    def test_feedforward_targets_use_only_legal_next_actions(self) -> None:
        num_actions = len(ACTION_ORDER)
        online_network = FixedFeedForwardNet(
            {
                "state": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]]),
                "next": torch.tensor([[0.0, 100.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]]),
            }
        )
        target_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((1, num_actions), dtype=torch.float32),
                "next": torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0]]),
            }
        )
        batch = {
            "state": {"phase": "state"},
            "next_state": {"phase": "next"},
            "action": torch.tensor([2], dtype=torch.long),
            "reward": torch.tensor([1.0], dtype=torch.float32),
            "done": torch.tensor([False], dtype=torch.bool),
            "action_mask": torch.ones((1, num_actions), dtype=torch.bool),
            "next_action_mask": torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        }

        target_info = compute_double_dqn_targets_feedforward(online_network, target_network, batch, gamma=0.5)

        self.assertEqual(int(target_info["next_action"].item()), 2)
        self.assertAlmostEqual(float(target_info["next_q"].item()), 30.0, places=6)
        self.assertAlmostEqual(float(target_info["target"].item()), 16.0, places=6)

    def test_feedforward_terminal_transition_without_legal_next_action_is_finite(self) -> None:
        num_actions = len(ACTION_ORDER)
        online_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((1, num_actions), dtype=torch.float32),
                "next": torch.tensor([[1000.0, -1000.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0]]),
            }
        )
        target_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((1, num_actions), dtype=torch.float32),
                "next": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]]),
            }
        )
        batch = {
            "state": {"phase": "state"},
            "next_state": {"phase": "next"},
            "action": torch.tensor([0], dtype=torch.long),
            "reward": torch.tensor([1_000_000.0], dtype=torch.float32),
            "done": torch.tensor([True], dtype=torch.bool),
            "action_mask": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
            "next_action_mask": torch.zeros((1, num_actions), dtype=torch.bool),
        }

        loss_info = compute_td_loss_feedforward(online_network, target_network, batch, gamma=0.99)

        self.assertTrue(torch.isfinite(loss_info["loss"]).item())
        self.assertTrue(torch.isfinite(loss_info["target"]).all().item())
        self.assertEqual(float(loss_info["target"].item()), 1_000_000.0)

    def test_feedforward_targets_support_n_step_discounting(self) -> None:
        num_actions = len(ACTION_ORDER)
        online_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((1, num_actions), dtype=torch.float32),
                "next": torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            }
        )
        target_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((1, num_actions), dtype=torch.float32),
                "next": torch.tensor([[0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            }
        )
        batch = {
            "state": {"phase": "state"},
            "next_state": {"phase": "next"},
            "action": torch.tensor([ACTION_ORDER.index("bet_1x")], dtype=torch.long),
            "reward": torch.tensor([2.0], dtype=torch.float32),
            "done": torch.tensor([False], dtype=torch.bool),
            "n_steps": torch.tensor([3.0], dtype=torch.float32),
            "action_mask": torch.ones((1, num_actions), dtype=torch.bool),
            "next_action_mask": torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        }

        target_info = compute_double_dqn_targets_feedforward(online_network, target_network, batch, gamma=0.5)

        self.assertAlmostEqual(float(target_info["target"].item()), 2.875, places=6)

    def test_feedforward_loss_supports_phase_weighting(self) -> None:
        num_actions = len(ACTION_ORDER)
        online_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((2, num_actions), dtype=torch.float32),
                "next": torch.zeros((2, num_actions), dtype=torch.float32),
            }
        )
        target_network = FixedFeedForwardNet(
            {
                "state": torch.zeros((2, num_actions), dtype=torch.float32),
                "next": torch.zeros((2, num_actions), dtype=torch.float32),
            }
        )
        batch = {
            "state": {"phase": "state"},
            "next_state": {"phase": "next"},
            "action": torch.tensor([ACTION_ORDER.index("bet_1x"), ACTION_ORDER.index("stand")], dtype=torch.long),
            "reward": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "done": torch.tensor([True, True], dtype=torch.bool),
            "action_mask": torch.ones((2, num_actions), dtype=torch.bool),
            "next_action_mask": torch.zeros((2, num_actions), dtype=torch.bool),
        }

        unweighted = compute_td_loss_feedforward(
            online_network,
            target_network,
            batch,
            gamma=0.99,
            config=BellmanLossConfig(
                gamma=0.99,
                phase_weights=LossPhaseWeightConfig(enabled=False, betting_weight=1.5, playing_weight=1.0),
            ),
        )
        weighted = compute_td_loss_feedforward(
            online_network,
            target_network,
            batch,
            gamma=0.99,
            config=BellmanLossConfig(
                gamma=0.99,
                phase_weights=LossPhaseWeightConfig(enabled=True, betting_weight=1.5, playing_weight=1.0),
            ),
        )

        self.assertAlmostEqual(float(unweighted["loss"].item()), 0.5, places=6)
        self.assertAlmostEqual(float(weighted["loss"].item()), 0.625, places=6)
        self.assertGreater(weighted["metrics"]["loss_betting"], 0.0)
        self.assertGreater(weighted["metrics"]["loss_playing"], 0.0)

    def test_recurrent_loss_ignores_padding(self) -> None:
        num_actions = len(ACTION_ORDER)
        zero_q = torch.zeros((2, 3, num_actions), dtype=torch.float32)
        online_network = FixedRecurrentNet({"state": zero_q, "next": zero_q})
        target_network = FixedRecurrentNet({"state": zero_q, "next": zero_q})
        batch = {
            "state": {"phase": "state"},
            "next_state": {"phase": "next"},
            "action": torch.zeros((2, 3), dtype=torch.long),
            "reward": torch.tensor([[1.0, 2.0, 999.0], [3.0, 999.0, 999.0]], dtype=torch.float32),
            "done": torch.ones((2, 3), dtype=torch.bool),
            "padding_mask": torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool),
            "action_mask": torch.tensor(
                [
                    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                ],
                dtype=torch.bool,
            ),
            "next_action_mask": torch.zeros((2, 3, num_actions), dtype=torch.bool),
        }

        loss_info = compute_td_loss_recurrent(online_network, target_network, batch, gamma=0.99)

        self.assertAlmostEqual(float(loss_info["loss"].item()), 1.5, places=6)
        self.assertEqual(float(loss_info["num_valid_steps"].item()), 3.0)
        self.assertTrue(torch.isfinite(loss_info["loss_per_timestep"]).all().item())

    def test_recurrent_nonterminal_without_legal_next_action_raises(self) -> None:
        num_actions = len(ACTION_ORDER)
        zero_q = torch.zeros((1, 2, num_actions), dtype=torch.float32)
        online_network = FixedRecurrentNet({"state": zero_q, "next": zero_q})
        target_network = FixedRecurrentNet({"state": zero_q, "next": zero_q})
        batch = {
            "state": {"phase": "state"},
            "next_state": {"phase": "next"},
            "action": torch.zeros((1, 2), dtype=torch.long),
            "reward": torch.zeros((1, 2), dtype=torch.float32),
            "done": torch.tensor([[False, True]], dtype=torch.bool),
            "padding_mask": torch.tensor([[True, True]], dtype=torch.bool),
            "action_mask": torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),
            "next_action_mask": torch.zeros((1, 2, num_actions), dtype=torch.bool),
        }

        with self.assertRaises(ValueError):
            compute_td_loss_recurrent(online_network, target_network, batch, gamma=0.99)

    def test_feedforward_bellman_loss_backward_with_real_network(self) -> None:
        batch = self.build_feedforward_transition_batch()
        online_network = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")
        target_network = FeedForwardDoubleDQN.from_profile("minimal_basic_strategy")

        loss_info = compute_td_loss_feedforward(online_network, target_network, batch, gamma=0.99)

        self.assertFalse(loss_info["target"].requires_grad)
        self.assertTrue(loss_info["loss"].requires_grad)
        self.assertEqual(tuple(loss_info["q_pred"].shape), tuple(batch["reward"].shape))
        self.assertEqual(tuple(loss_info["target"].shape), tuple(batch["reward"].shape))
        self.assertTrue(torch.isfinite(loss_info["loss"]).item())

        loss_info["loss"].backward()
        self.assert_only_online_has_gradients(online_network, target_network)

    def test_recurrent_bellman_loss_backward_with_real_network(self) -> None:
        batch = self.build_recurrent_transition_batch(unknown_progress=False)
        online_network = RecurrentDoubleDQN.from_profile("table_realistic_default", recurrent_type="gru")
        target_network = RecurrentDoubleDQN.from_profile("table_realistic_default", recurrent_type="gru")

        loss_info = compute_td_loss_recurrent(online_network, target_network, batch, gamma=0.99)

        self.assertFalse(loss_info["target"].requires_grad)
        self.assertTrue(loss_info["loss"].requires_grad)
        self.assertEqual(tuple(loss_info["q_pred"].shape), tuple(batch["reward"].shape))
        self.assertEqual(tuple(loss_info["target"].shape), tuple(batch["reward"].shape))
        self.assertGreater(float(loss_info["num_valid_steps"].item()), 0.0)
        self.assertTrue(torch.isfinite(loss_info["loss"]).item())

        loss_info["loss"].backward()
        self.assert_only_online_has_gradients(online_network, target_network)

    def test_dueling_recurrent_bellman_loss_backward_with_real_network(self) -> None:
        batch = self.build_recurrent_transition_batch(unknown_progress=True)
        online_network = DuelingRecurrentDoubleDQN.from_profile(
            "table_realistic_unknown_progress",
            recurrent_type="lstm",
        )
        target_network = DuelingRecurrentDoubleDQN.from_profile(
            "table_realistic_unknown_progress",
            recurrent_type="lstm",
        )

        loss_info = compute_td_loss_recurrent(online_network, target_network, batch, gamma=0.99)

        self.assertFalse(loss_info["target"].requires_grad)
        self.assertTrue(loss_info["loss"].requires_grad)
        self.assertTrue(torch.isfinite(loss_info["loss"]).item())
        self.assertGreater(float(loss_info["num_valid_steps"].item()), 0.0)

        loss_info["loss"].backward()
        self.assert_only_online_has_gradients(online_network, target_network)


if __name__ == "__main__":
    unittest.main()
