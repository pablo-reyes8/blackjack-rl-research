from __future__ import annotations

from collections import deque
import random
from typing import Any, Sequence

import torch

from enviroment_bj.core import ACTION_ORDER

from .config import ReplayBufferConfig


class FeedForwardReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.storage: deque[dict[str, Any]] = deque(maxlen=config.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: dict[str, Any]) -> None:
        self.storage.append(transition)

    def can_sample(self) -> bool:
        return len(self.storage) >= self.config.batch_size

    def sample(self) -> dict[str, Any]:
        if not self.can_sample():
            raise ValueError("Replay buffer does not contain enough transitions to sample a batch")
        transitions = self.rng.sample(list(self.storage), self.config.batch_size)
        batch = {
            "state": [transition["state"] for transition in transitions],
            "next_state": [transition["next_state"] for transition in transitions],
            "action": torch.tensor([transition["action"] for transition in transitions], dtype=torch.long),
            "reward": torch.tensor([transition["reward"] for transition in transitions], dtype=torch.float32),
            "done": torch.tensor([transition["done"] for transition in transitions], dtype=torch.bool),
            "n_steps": torch.tensor([transition.get("n_steps", 1) for transition in transitions], dtype=torch.float32),
            "action_mask": torch.stack([transition["action_mask"] for transition in transitions], dim=0).to(torch.bool),
            "next_action_mask": torch.stack(
                [transition["next_action_mask"] for transition in transitions],
                dim=0,
            ).to(torch.bool),
        }
        if transitions and "teacher_state_vector" in transitions[0]["state"]:
            batch["teacher_state"] = {
                "state_vector": torch.stack(
                    [transition["state"]["teacher_state_vector"] for transition in transitions],
                    dim=0,
                ).to(torch.float32),
                "action_mask": torch.stack(
                    [transition["state"]["teacher_action_mask"] for transition in transitions],
                    dim=0,
                ).to(torch.bool),
                "module_tensors": {},
                "metadata": {"batch_size": len(transitions)},
            }
        if transitions and "betting_auxiliary" in transitions[0]["state"]:
            batch["betting_auxiliary"] = {
                "true_count_proxy": torch.stack(
                    [transition["state"]["betting_auxiliary"]["true_count_proxy"] for transition in transitions],
                    dim=0,
                ).to(torch.float32),
                "observed_cards": torch.stack(
                    [transition["state"]["betting_auxiliary"]["observed_cards"] for transition in transitions],
                    dim=0,
                ).to(torch.long),
            }
        return batch


class RecurrentReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.storage: deque[dict[str, Any]] = deque(maxlen=config.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, sequence: dict[str, Any]) -> None:
        if len(sequence["action"]) < self.config.min_sequence_length and not any(sequence.get("done", [])):
            return
        self.storage.append(sequence)

    def can_sample(self) -> bool:
        return len(self.storage) >= self.config.batch_size

    def sample(self) -> dict[str, Any]:
        if not self.can_sample():
            raise ValueError("Recurrent replay buffer does not contain enough sequences to sample a batch")

        sequences = self.rng.sample(list(self.storage), self.config.batch_size)
        max_len = min(max(len(sequence["action"]) for sequence in sequences), self.config.sequence_length)
        batch_size = len(sequences)
        num_actions = len(ACTION_ORDER)
        state_dim = sequences[0]["state"][0]["state_vector"].shape[0]

        action = torch.zeros((batch_size, max_len), dtype=torch.long)
        reward = torch.zeros((batch_size, max_len), dtype=torch.float32)
        done = torch.ones((batch_size, max_len), dtype=torch.bool)
        n_steps = torch.ones((batch_size, max_len), dtype=torch.float32)
        padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        state_vector = torch.zeros((batch_size, max_len, state_dim), dtype=torch.float32)
        next_state_vector = torch.zeros((batch_size, max_len, state_dim), dtype=torch.float32)
        action_mask = torch.zeros((batch_size, max_len, num_actions), dtype=torch.bool)
        next_action_mask = torch.zeros((batch_size, max_len, num_actions), dtype=torch.bool)
        sequence_lengths: list[int] = []
        has_teacher_state = bool(sequences and "teacher_state_vector" in sequences[0]["state"][0])
        teacher_state_dim = 0
        teacher_state_vector = None
        teacher_action_mask = None
        if has_teacher_state:
            teacher_state_dim = sequences[0]["state"][0]["teacher_state_vector"].shape[0]
            teacher_state_vector = torch.zeros((batch_size, max_len, teacher_state_dim), dtype=torch.float32)
            teacher_action_mask = torch.zeros((batch_size, max_len, num_actions), dtype=torch.bool)
        has_betting_auxiliary = bool(sequences and "betting_auxiliary" in sequences[0]["state"][0])
        true_count_proxy = None
        observed_cards = None
        if has_betting_auxiliary:
            true_count_proxy = torch.zeros((batch_size, max_len), dtype=torch.float32)
            observed_cards = torch.zeros((batch_size, max_len), dtype=torch.long)

        for batch_index, sequence in enumerate(sequences):
            truncated_len = min(len(sequence["action"]), max_len)
            sequence_lengths.append(truncated_len)

            action[batch_index, :truncated_len] = torch.tensor(sequence["action"][:truncated_len], dtype=torch.long)
            reward[batch_index, :truncated_len] = torch.tensor(sequence["reward"][:truncated_len], dtype=torch.float32)
            done[batch_index, :truncated_len] = torch.tensor(sequence["done"][:truncated_len], dtype=torch.bool)
            n_steps[batch_index, :truncated_len] = torch.tensor(sequence.get("n_steps", [1] * truncated_len)[:truncated_len], dtype=torch.float32)
            padding_mask[batch_index, :truncated_len] = True
            state_vector[batch_index, :truncated_len] = torch.stack(
                [step["state_vector"] for step in sequence["state"][:truncated_len]],
                dim=0,
            ).to(torch.float32)
            next_state_vector[batch_index, :truncated_len] = torch.stack(
                [step["state_vector"] for step in sequence["next_state"][:truncated_len]],
                dim=0,
            ).to(torch.float32)
            action_mask[batch_index, :truncated_len] = torch.stack(sequence["action_mask"][:truncated_len], dim=0).to(torch.bool)
            next_action_mask[batch_index, :truncated_len] = torch.stack(
                sequence["next_action_mask"][:truncated_len],
                dim=0,
            ).to(torch.bool)
            if has_teacher_state and teacher_state_vector is not None and teacher_action_mask is not None:
                teacher_state_vector[batch_index, :truncated_len] = torch.stack(
                    [step["teacher_state_vector"] for step in sequence["state"][:truncated_len]],
                    dim=0,
                ).to(torch.float32)
                teacher_action_mask[batch_index, :truncated_len] = torch.stack(
                    [step["teacher_action_mask"] for step in sequence["state"][:truncated_len]],
                    dim=0,
                ).to(torch.bool)
            if has_betting_auxiliary and true_count_proxy is not None and observed_cards is not None:
                true_count_proxy[batch_index, :truncated_len] = torch.stack(
                    [step["betting_auxiliary"]["true_count_proxy"] for step in sequence["state"][:truncated_len]],
                    dim=0,
                ).to(torch.float32)
                observed_cards[batch_index, :truncated_len] = torch.stack(
                    [step["betting_auxiliary"]["observed_cards"] for step in sequence["state"][:truncated_len]],
                    dim=0,
                ).to(torch.long)

        batch = {
            "state": {
                "state_vector": state_vector,
                "action_mask": action_mask,
                "padding_mask": padding_mask,
                "module_tensors": {},
                "metadata": {"batch_size": batch_size, "sequence_lengths": sequence_lengths},
            },
            "next_state": {
                "state_vector": next_state_vector,
                "action_mask": next_action_mask,
                "padding_mask": padding_mask,
                "module_tensors": {},
                "metadata": {"batch_size": batch_size, "sequence_lengths": sequence_lengths},
            },
            "action": action,
            "reward": reward,
            "done": done,
            "n_steps": n_steps,
            "padding_mask": padding_mask,
            "action_mask": action_mask,
            "next_action_mask": next_action_mask,
        }
        if has_teacher_state and teacher_state_vector is not None and teacher_action_mask is not None:
            batch["teacher_state"] = {
                "state_vector": teacher_state_vector,
                "action_mask": teacher_action_mask,
                "padding_mask": padding_mask,
                "module_tensors": {},
                "metadata": {"batch_size": batch_size, "sequence_lengths": sequence_lengths},
            }
        if has_betting_auxiliary and true_count_proxy is not None and observed_cards is not None:
            batch["betting_auxiliary"] = {
                "true_count_proxy": true_count_proxy,
                "observed_cards": observed_cards,
            }
        return batch
