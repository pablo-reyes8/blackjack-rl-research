from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BlackjackConfig, StartStateConfig
from .environment import BlackjackEnvironment


class BlackjackJSONWrapper:
    def __init__(
        self,
        config: BlackjackConfig | None = None,
        seed: int | None = None,
        start_state: StartStateConfig | None = None,
    ) -> None:
        self.environment = BlackjackEnvironment(config=config, seed=seed, start_state=start_state)

    def reset(self) -> dict[str, Any]:
        return self.environment.reset()

    def step(self, payload: Any) -> dict[str, Any]:
        return self.environment.step(payload)

    def step_from_json(self, payload: str) -> dict[str, Any]:
        return self.step(json.loads(payload))

    def step_from_file(self, file_path: str | Path) -> dict[str, Any]:
        content = Path(file_path).read_text(encoding="utf-8")
        return self.step_from_json(content)
