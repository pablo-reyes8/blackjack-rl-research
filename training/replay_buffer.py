from __future__ import annotations

from collections import deque
from copy import deepcopy
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
        self.storage.append(deepcopy(transition))

    def can_sample(self) -> bool:
        return len(self.storage) >= self.config.batch_size

    def sample(self) -> dict[str, Any]:
        if not self.can_sample():
            raise ValueError("Replay buffer does not contain enough transitions to sample a batch")
        transitions = self.rng.sample(list(self.storage), self.config.batch_size)
        return {
            "state": [transition["state"] for transition in transitions],
            "next_state": [transition["next_state"] for transition in transitions],
            "action": torch.tensor([transition["action"] for transition in transitions], dtype=torch.long),
            "reward": torch.tensor([transition["reward"] for transition in transitions], dtype=torch.float32),
            "done": torch.tensor([transition["done"] for transition in transitions], dtype=torch.bool),
            "action_mask": torch.tensor([transition["action_mask"] for transition in transitions], dtype=torch.bool),
            "next_action_mask": torch.tensor(
                [transition["next_action_mask"] for transition in transitions],
                dtype=torch.bool,
            ),
        }


class RecurrentReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.storage: deque[dict[str, Any]] = deque(maxlen=config.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, sequence: dict[str, Any]) -> None:
        if len(sequence["action"]) < self.config.min_sequence_length:
            return
        self.storage.append(deepcopy(sequence))

    def can_sample(self) -> bool:
        return len(self.storage) >= self.config.batch_size

    def sample(self) -> dict[str, Any]:
        if not self.can_sample():
            raise ValueError("Recurrent replay buffer does not contain enough sequences to sample a batch")

        sequences = self.rng.sample(list(self.storage), self.config.batch_size)
        max_len = min(max(len(sequence["action"]) for sequence in sequences), self.config.sequence_length)
        batch_size = len(sequences)
        num_actions = len(ACTION_ORDER)

        action = torch.zeros((batch_size, max_len), dtype=torch.long)
        reward = torch.zeros((batch_size, max_len), dtype=torch.float32)
        done = torch.ones((batch_size, max_len), dtype=torch.bool)
        padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        action_mask = torch.zeros((batch_size, max_len, num_actions), dtype=torch.bool)
        next_action_mask = torch.zeros((batch_size, max_len, num_actions), dtype=torch.bool)
        states: list[list[dict[str, Any]]] = []
        next_states: list[list[dict[str, Any]]] = []

        for batch_index, sequence in enumerate(sequences):
            truncated_len = min(len(sequence["action"]), max_len)
            states.append(sequence["state"][:truncated_len])
            next_states.append(sequence["next_state"][:truncated_len])

            action[batch_index, :truncated_len] = torch.tensor(sequence["action"][:truncated_len], dtype=torch.long)
            reward[batch_index, :truncated_len] = torch.tensor(sequence["reward"][:truncated_len], dtype=torch.float32)
            done[batch_index, :truncated_len] = torch.tensor(sequence["done"][:truncated_len], dtype=torch.bool)
            padding_mask[batch_index, :truncated_len] = True
            action_mask[batch_index, :truncated_len] = torch.tensor(
                sequence["action_mask"][:truncated_len],
                dtype=torch.bool,
            )
            next_action_mask[batch_index, :truncated_len] = torch.tensor(
                sequence["next_action_mask"][:truncated_len],
                dtype=torch.bool,
            )

        return {
            "state": states,
            "next_state": next_states,
            "action": action,
            "reward": reward,
            "done": done,
            "padding_mask": padding_mask,
            "action_mask": action_mask,
            "next_action_mask": next_action_mask,
        }
