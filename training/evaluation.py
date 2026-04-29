from __future__ import annotations

from dataclasses import dataclass, field
import random
from collections import defaultdict
from typing import Any, Callable

import torch

from enviroment_bj.core import ACTION_ORDER, BET_ACTION_ORDER

from .betting_auxiliary import BettingAuxiliaryEvaluationTracker, compute_observed_hi_lo_proxy_from_response
from .config import BettingAuxiliaryConfig
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
    betting_auxiliary_config: BettingAuxiliaryConfig | None = None,
    progress_every_n_rounds: int | None = None,
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    model.eval()
    tracker = BehaviorMetricsTracker()
    env_states = [EvaluationEnvState(env=env) for env in envs]
    is_recurrent = model.config.architecture != "feedforward"
    decisions = 0
    bet_q_totals: dict[str, float] = defaultdict(float)
    bet_q_counts: dict[str, int] = defaultdict(int)
    best_aggressive_margin_total = 0.0
    best_aggressive_margin_count = 0
    betting_auxiliary_tracker = (
        BettingAuxiliaryEvaluationTracker()
        if betting_auxiliary_config is not None and betting_auxiliary_config.enabled
        else None
    )
    available_bet_multipliers = tuple(
        int(multiplier)
        for multiplier in getattr(getattr(envs[0], "config", None), "bet_multipliers", ())
    ) if envs else ()
    next_progress_round = (
        int(progress_every_n_rounds)
        if progress_every_n_rounds is not None and int(progress_every_n_rounds) > 0
        else None
    )

    def build_summary() -> dict[str, Any]:
        summary = tracker.summary()
        summary.update(
            {
                "evaluation_decisions": float(decisions),
                "available_bet_multipliers": list(available_bet_multipliers),
                "mean_margin_best_aggressive_vs_1x": (
                    best_aggressive_margin_total / best_aggressive_margin_count
                    if best_aggressive_margin_count > 0
                    else 0.0
                ),
            }
        )
        for action_name in BET_ACTION_ORDER:
            summary[f"mean_q_{action_name}"] = (
                bet_q_totals[action_name] / bet_q_counts[action_name]
                if bet_q_counts[action_name] > 0
                else None
            )
        if betting_auxiliary_tracker is not None:
            summary.update(betting_auxiliary_tracker.summary())
        return summary

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
                    q_values = model_output["q_values"].squeeze(0)
                    masked_q_values = model_output["masked_q_values"].squeeze(0)
                else:
                    model_output = model(env_state.response)
                    action_mask = model_output["action_mask"].squeeze(0)
                    q_values = model_output["q_values"].squeeze(0)
                    masked_q_values = model_output["masked_q_values"].squeeze(0)

                decision_phase = infer_decision_phase(env_state.response)
                epsilon_value = resolve_epsilon_value(epsilon, decision_phase=decision_phase, evaluation=True)

                if decision_phase == "betting":
                    best_aggressive_q: float | None = None
                    q_bet_1x: float | None = None
                    for bet_index, action_name in enumerate(BET_ACTION_ORDER):
                        if not bool(action_mask[bet_index].item()):
                            continue
                        q_value = float(q_values[bet_index].item())
                        bet_q_totals[action_name] += q_value
                        bet_q_counts[action_name] += 1
                        if action_name == "bet_1x":
                            q_bet_1x = q_value
                        elif action_name in {"bet_2x", "bet_3x", "bet_4x"}:
                            if best_aggressive_q is None or q_value > best_aggressive_q:
                                best_aggressive_q = q_value
                    if q_bet_1x is not None and best_aggressive_q is not None:
                        best_aggressive_margin_total += best_aggressive_q - q_bet_1x
                        best_aggressive_margin_count += 1
                    if betting_auxiliary_tracker is not None and betting_auxiliary_config is not None:
                        proxy_info = compute_observed_hi_lo_proxy_from_response(
                            env_state.response,
                            n_decks=int(getattr(env_state.env.config, "n_decks", 8)),
                        )
                        betting_auxiliary_tracker.record(
                            q_values=q_values,
                            action_mask=action_mask,
                            proxy_info=proxy_info,
                            config=betting_auxiliary_config,
                        )

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
                    current_rounds = int(tracker.total_rounds)
                    if (
                        progress_callback is not None
                        and next_progress_round is not None
                        and current_rounds >= next_progress_round
                    ):
                        progress_callback(build_summary(), current_rounds, decisions)
                        while next_progress_round is not None and current_rounds >= next_progress_round:
                            next_progress_round += int(progress_every_n_rounds or 0)
                    env_state.response = None
                    if is_recurrent and reset_hidden_on_round_end:
                        env_state.hidden_state = model.init_hidden(batch_size=1)
                else:
                    env_state.response = next_response

    return build_summary()
