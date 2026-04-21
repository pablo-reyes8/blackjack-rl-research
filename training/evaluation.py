from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import torch

from enviroment_bj.core import ACTION_ORDER

from .metrics import BehaviorMetricsTracker
from .policy import action_name_from_index, infer_decision_phase, resolve_epsilon_value, select_epsilon_greedy_action


@dataclass(slots=True)
class EvaluationEnvState:
    env: Any
    response: dict[str, Any] | None = None
    hidden_state: Any = None


def evaluate_policy(
    *,
    envs: list[Any],
    model: Any,
    epsilon: float,
    num_rounds: int,
    max_decisions: int,
    rng: random.Random,
    reset_hidden_on_round_end: bool,
) -> dict[str, Any]:
    model.eval()
    tracker = BehaviorMetricsTracker()
    env_states = [EvaluationEnvState(env=env) for env in envs]
    is_recurrent = model.config.architecture != "feedforward"
    decisions = 0

    with torch.no_grad():
        while tracker.total_rounds < num_rounds and decisions < max_decisions:
            for env_state in env_states:
                if tracker.total_rounds >= num_rounds or decisions >= max_decisions:
                    break
                while env_state.response is None or env_state.response["done"]:
                    if env_state.response is not None and env_state.response["done"]:
                        if is_recurrent and reset_hidden_on_round_end:
                            env_state.hidden_state = model.init_hidden(batch_size=1)
                    env_state.response = env_state.env.reset()
                    if is_recurrent and env_state.hidden_state is None:
                        env_state.hidden_state = model.init_hidden(batch_size=1)

                if is_recurrent:
                    model_output = model.forward_step(env_state.response, hidden_state=env_state.hidden_state)
                    env_state.hidden_state = model_output["hidden_state"]
                    action_mask = model_output["action_mask"].squeeze(0)
                    masked_q_values = model_output["masked_q_values"].squeeze(0)
                else:
                    model_output = model(env_state.response)
                    action_mask = model_output["action_mask"].squeeze(0)
                    masked_q_values = model_output["masked_q_values"].squeeze(0)

                decision_phase = infer_decision_phase(env_state.response)
                epsilon_value = resolve_epsilon_value(epsilon, decision_phase=decision_phase, evaluation=True)

                action_index, was_random = select_epsilon_greedy_action(
                    masked_q_values=masked_q_values,
                    action_mask=action_mask,
                    epsilon=epsilon_value,
                    rng=rng,
                )
                action_name = action_name_from_index(action_index, ACTION_ORDER)
                try:
                    next_response = env_state.env.step(action_name)
                except RuntimeError as exc:
                    if "shoe is empty" not in str(exc).lower() and "round is over" not in str(exc).lower():
                        raise
                    env_state.response = None
                    continue

                table_key = f"{env_state.env.start_state.mode}|{env_state.env.config.observation.profile}"
                tracker.record_decision(
                    env_state.response,
                    action_name,
                    was_random=was_random,
                    table_key=table_key,
                    env_key=str(id(env_state.env)),
                )
                tracker.record_round_result(next_response, env_key=str(id(env_state.env)))
                decisions += 1

                if next_response["done"]:
                    env_state.response = None
                    if is_recurrent and reset_hidden_on_round_end:
                        env_state.hidden_state = model.init_hidden(batch_size=1)
                else:
                    env_state.response = next_response

    summary = tracker.summary()
    summary.update({"evaluation_decisions": float(decisions)})
    return summary
