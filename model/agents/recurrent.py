from __future__ import annotations

from typing import Any

import torch
from torch import nn


class RecurrentBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        recurrent_type: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.recurrent_type = recurrent_type
        recurrent_cls = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.recurrent = recurrent_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Any:
        shape = (self.num_layers, batch_size, self.hidden_dim)
        if self.recurrent_type == "gru":
            return torch.zeros(shape, dtype=torch.float32, device=device)
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        if padding_mask is None:
            return self.recurrent(sequence, hidden_state)

        lengths = padding_mask.to(torch.int64).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden_state = self.recurrent(packed, hidden_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        return output, hidden_state
