from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from enviroment_bj import BlackjackEnvironment


def normalize_envs(envs: BlackjackEnvironment | Sequence[BlackjackEnvironment]) -> list[BlackjackEnvironment]:
    if isinstance(envs, BlackjackEnvironment):
        return [envs]
    normalized = list(envs)
    if not normalized:
        raise ValueError("At least one environment instance is required")
    return normalized


def clone_environment(environment: BlackjackEnvironment, *, seed: int) -> BlackjackEnvironment:
    return BlackjackEnvironment(
        config=deepcopy(environment.config),
        seed=seed,
        start_state=deepcopy(environment.start_state),
    )


def clone_environments(envs: Sequence[BlackjackEnvironment], *, seed_offset: int = 10_000) -> list[BlackjackEnvironment]:
    return [clone_environment(env, seed=seed_offset + index) for index, env in enumerate(envs)]
