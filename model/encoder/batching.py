from __future__ import annotations

from typing import Any, Sequence

import torch


def stack_encoded_steps(encoded_steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not encoded_steps:
        raise ValueError("encoded_steps cannot be empty")

    module_keys = list(encoded_steps[0]["module_tensors"].keys())
    return {
        "state_vector": torch.stack([step["state_vector"] for step in encoded_steps], dim=0),
        "action_mask": torch.stack([step["action_mask"] for step in encoded_steps], dim=0),
        "module_tensors": {
            key: torch.stack([step["module_tensors"][key] for step in encoded_steps], dim=0)
            for key in module_keys
        },
        "metadata": {
            "batch_size": len(encoded_steps),
            "items": [step["metadata"] for step in encoded_steps],
        },
    }


def pad_encoded_sequences(encoded_sequences: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    if not encoded_sequences:
        raise ValueError("encoded_sequences cannot be empty")

    if not encoded_sequences[0]:
        raise ValueError("encoded_sequences cannot contain empty sequences")

    batch_size = len(encoded_sequences)
    max_length = max(len(sequence) for sequence in encoded_sequences)
    state_dim = encoded_sequences[0][0]["state_vector"].shape[0]
    action_dim = encoded_sequences[0][0]["action_mask"].shape[0]
    module_keys = list(encoded_sequences[0][0]["module_tensors"].keys())

    state_batch = torch.zeros((batch_size, max_length, state_dim), dtype=torch.float32)
    action_batch = torch.zeros((batch_size, max_length, action_dim), dtype=torch.bool)
    padding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    module_batches = {
        key: torch.zeros(
            (batch_size, max_length, encoded_sequences[0][0]["module_tensors"][key].shape[0]),
            dtype=torch.float32,
        )
        for key in module_keys
    }

    metadata_items: list[list[dict[str, Any]]] = []

    for batch_index, sequence in enumerate(encoded_sequences):
        metadata_items.append([])
        for time_index, step in enumerate(sequence):
            state_batch[batch_index, time_index] = step["state_vector"]
            action_batch[batch_index, time_index] = step["action_mask"]
            padding_mask[batch_index, time_index] = True
            metadata_items[batch_index].append(step["metadata"])
            for key in module_keys:
                module_batches[key][batch_index, time_index] = step["module_tensors"][key]

    return {
        "state_vector": state_batch,
        "action_mask": action_batch,
        "padding_mask": padding_mask,
        "module_tensors": module_batches,
        "metadata": {
            "batch_size": batch_size,
            "max_sequence_length": max_length,
            "sequence_lengths": [len(sequence) for sequence in encoded_sequences],
            "items": metadata_items,
        },
    }
