from __future__ import annotations

from typing import Any


def train_one_epoch(trainer: Any) -> dict[str, Any]:
    return trainer.train_one_epoch()
