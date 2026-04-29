from __future__ import annotations

from collections.abc import Mapping
from contextlib import redirect_stdout
from copy import deepcopy
import io
from itertools import product
from pathlib import Path
import random
from typing import Any

import torch

from scripts.blackjack_rl_cli.common import load_checkpoint_payload, resolve_device, to_pretty_json, write_json_file
from training.evaluation import evaluate_policy
from training.final_wrapper import run_blackjack_stage

from .logging import InferenceLogger


def _coerce_bet_multipliers(value: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if value is None:
        return (1, 2, 3, 4)
    return tuple(int(multiplier) for multiplier in value)


def _coerce_penetrations(value: list[float] | tuple[float, ...] | None) -> tuple[float, ...]:
    if value is None:
        return (0.75,)
    return tuple(float(item) for item in value)


def _resolve_model_and_encoder_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    model_config = deepcopy(dict(payload.get("model_config") or {}))
    encoder_config = deepcopy(dict(model_config.get("encoder") or {}))

    return {
        "architecture": model_config.get("architecture", "feedforward"),
        "activation": model_config.get("activation", "relu"),
        "use_layer_norm": bool(model_config.get("use_layer_norm", False)),
        "dropout": float(model_config.get("dropout", 0.0)),
        "feedforward_hidden_dims": tuple(model_config.get("feedforward_hidden_dims", (256, 256, 128))),
        "projection_dim": int(model_config.get("projection_dim", 256)),
        "recurrent_hidden_dim": int(model_config.get("recurrent_hidden_dim", 256)),
        "recurrent_num_layers": int(model_config.get("recurrent_num_layers", 1)),
        "recurrent_type": model_config.get("recurrent_type", "gru"),
        "head_hidden_dim": int(model_config.get("head_hidden_dim", 128)),
        "value_hidden_dim": int(model_config.get("value_hidden_dim", 128)),
        "advantage_hidden_dim": int(model_config.get("advantage_hidden_dim", 128)),
        "use_phase_adapters": bool(model_config.get("use_phase_adapters", False)),
        "use_module_gating": bool(model_config.get("use_module_gating", False)),
        "observation_profile": encoder_config.get("profile", "table_realistic_unknown_progress"),
        "encoder_profile": encoder_config.get("profile", "table_realistic_unknown_progress"),
        "include_observed_history": bool(encoder_config.get("encode_observed_history", False)),
        "include_discard_summary": bool(encoder_config.get("encode_discard_summary", False)),
        "include_temporal_context": bool(encoder_config.get("encode_temporal", False)),
        "include_recent_actions": bool(encoder_config.get("encode_recent_actions", False)),
        "encode_rules": bool(encoder_config.get("encode_rules", True)),
        "encode_betting_context": bool(encoder_config.get("encode_betting_context", True)),
        "encode_other_hands": bool(encoder_config.get("encode_other_hands", True)),
        "encode_exact_shoe": bool(encoder_config.get("encode_exact_shoe", False)),
        "encode_action_mask_features": bool(encoder_config.get("encode_action_mask_features", False)),
        "history_encoding": encoder_config.get("history_encoding", "rank_counts"),
        "normalize_counts": bool(encoder_config.get("normalize_counts", True)),
        "use_visible_table_rules_only": bool(encoder_config.get("use_visible_table_rules_only", True)),
        "max_current_hand_cards": int(encoder_config.get("max_current_hand_cards", 12)),
        "max_cards_per_hand": int(encoder_config.get("max_cards_per_hand", 12)),
        "max_other_hands": int(encoder_config.get("max_other_hands", 4)),
        "max_recent_actions": int(encoder_config.get("max_recent_actions", 5)),
        "max_recent_cards": int(encoder_config.get("max_recent_cards", 20)),
        "max_recent_discard_cards": int(encoder_config.get("max_recent_discard_cards", 10)),
    }


def _apply_explicit_model_overrides(
    model_and_encoder_args: Mapping[str, Any],
    *,
    architecture: str | None,
    feedforward_hidden_dims: tuple[int, ...] | list[int] | None,
    use_layer_norm: bool | None,
    use_phase_adapters: bool | None,
    use_module_gating: bool | None,
) -> dict[str, Any]:
    resolved = dict(model_and_encoder_args)
    if architecture is not None:
        resolved["architecture"] = str(architecture)
    if feedforward_hidden_dims is not None:
        resolved["feedforward_hidden_dims"] = tuple(int(dim) for dim in feedforward_hidden_dims)
    if use_layer_norm is not None:
        resolved["use_layer_norm"] = bool(use_layer_norm)
    if use_phase_adapters is not None:
        resolved["use_phase_adapters"] = bool(use_phase_adapters)
    if use_module_gating is not None:
        resolved["use_module_gating"] = bool(use_module_gating)
    return resolved


def _model_config_for_logging(model_and_encoder_args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "architecture": model_and_encoder_args.get("architecture"),
        "recurrent_type": model_and_encoder_args.get("recurrent_type", "none"),
        "feedforward_hidden_dims": model_and_encoder_args.get("feedforward_hidden_dims"),
        "use_layer_norm": model_and_encoder_args.get("use_layer_norm", False),
        "use_phase_adapters": model_and_encoder_args.get("use_phase_adapters", False),
        "use_module_gating": model_and_encoder_args.get("use_module_gating", False),
        "encoder": {
            "profile": model_and_encoder_args.get("encoder_profile"),
        },
    }


def _expected_checkpoint_state_dim(payload: Mapping[str, Any]) -> int | None:
    state_dict = (
        payload.get("online_model_state_dict")
        or payload.get("model_state_dict")
        or payload
    )
    if not isinstance(state_dict, Mapping):
        return None

    for key, value in state_dict.items():
        if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[1])
    return None


def _probe_state_dim(
    *,
    output_root: str | Path,
    stage_name: str,
    device: str,
    start_mode: str,
    min_burned_rounds: int,
    max_burned_rounds: int,
    bet_multipliers: tuple[int, ...],
    penetrations: tuple[float, ...],
    base_seed: int,
    n_decks: int,
    shoe_penetration: float,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
    double_allowed_on: str,
    double_after_split_allowed: bool,
    split_rule: str,
    surrender_allowed: bool,
    insurance_allowed: bool,
    six_card_charlie_enabled: bool,
    blackjack_overrides: Mapping[str, Any] | None,
    observation_overrides: Mapping[str, Any] | None,
    start_state_overrides: Mapping[str, Any] | None,
    model_and_encoder_args: Mapping[str, Any],
    betting_auxiliary_args: Mapping[str, Any],
) -> int:
    with redirect_stdout(io.StringIO()):
        probe_betting_auxiliary_args = dict(betting_auxiliary_args)
        probe_betting_auxiliary_args["axu_loss_bet"] = False
        setup = run_blackjack_stage(
            stage_name=f"{stage_name}_probe",
            output_root=output_root,
            run_training=False,
            enable_prints=False,
            print_run_summary=False,
            start_mode=start_mode,
            min_burned_rounds=min_burned_rounds,
            max_burned_rounds=max_burned_rounds,
            bet_multipliers=bet_multipliers,
            penetrations=penetrations,
            base_seed=base_seed,
            n_decks=n_decks,
            shoe_penetration=shoe_penetration,
            dealer_hits_soft_17=dealer_hits_soft_17,
            blackjack_payout=blackjack_payout,
            double_allowed_on=double_allowed_on,
            double_after_split_allowed=double_after_split_allowed,
            split_rule=split_rule,
            surrender_allowed=surrender_allowed,
            insurance_allowed=insurance_allowed,
            six_card_charlie_enabled=six_card_charlie_enabled,
            eval_rounds=1,
            eval_max_decisions=1,
            device=device,
            total_epochs=1,
            env_steps_per_epoch=1,
            checkpoint_directory=Path(output_root) / f"{stage_name}_probe",
            blackjack_overrides=blackjack_overrides,
            observation_overrides=observation_overrides,
            start_state_overrides=start_state_overrides,
            **model_and_encoder_args,
            **probe_betting_auxiliary_args,
        )
    return int(setup["model"].state_dim)


def _resolve_model_and_encoder_args_for_checkpoint(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path,
    stage_name: str,
    device: str,
    start_mode: str,
    min_burned_rounds: int,
    max_burned_rounds: int,
    bet_multipliers: tuple[int, ...],
    penetrations: tuple[float, ...],
    base_seed: int,
    n_decks: int,
    shoe_penetration: float,
    dealer_hits_soft_17: bool,
    blackjack_payout: float,
    double_allowed_on: str,
    double_after_split_allowed: bool,
    split_rule: str,
    surrender_allowed: bool,
    insurance_allowed: bool,
    six_card_charlie_enabled: bool,
    blackjack_overrides: Mapping[str, Any] | None,
    observation_overrides: Mapping[str, Any] | None,
    start_state_overrides: Mapping[str, Any] | None,
    betting_auxiliary_args: Mapping[str, Any],
    architecture: str | None,
    feedforward_hidden_dims: tuple[int, ...] | list[int] | None,
    use_layer_norm: bool | None,
    use_phase_adapters: bool | None,
    use_module_gating: bool | None,
) -> dict[str, Any]:
    model_and_encoder_args = _apply_explicit_model_overrides(
        _resolve_model_and_encoder_args(payload),
        architecture=architecture,
        feedforward_hidden_dims=feedforward_hidden_dims,
        use_layer_norm=use_layer_norm,
        use_phase_adapters=use_phase_adapters,
        use_module_gating=use_module_gating,
    )
    expected_state_dim = _expected_checkpoint_state_dim(payload)
    if expected_state_dim is None:
        return model_and_encoder_args

    current_state_dim = _probe_state_dim(
        output_root=output_root,
        stage_name=stage_name,
        device=device,
        start_mode=start_mode,
        min_burned_rounds=min_burned_rounds,
        max_burned_rounds=max_burned_rounds,
        bet_multipliers=bet_multipliers,
        penetrations=penetrations,
        base_seed=base_seed,
        n_decks=n_decks,
        shoe_penetration=shoe_penetration,
        dealer_hits_soft_17=dealer_hits_soft_17,
        blackjack_payout=blackjack_payout,
        double_allowed_on=double_allowed_on,
        double_after_split_allowed=double_after_split_allowed,
        split_rule=split_rule,
        surrender_allowed=surrender_allowed,
        insurance_allowed=insurance_allowed,
        six_card_charlie_enabled=six_card_charlie_enabled,
        blackjack_overrides=blackjack_overrides,
        observation_overrides=observation_overrides,
        start_state_overrides=start_state_overrides,
        model_and_encoder_args=model_and_encoder_args,
        betting_auxiliary_args=betting_auxiliary_args,
    )
    if current_state_dim == expected_state_dim:
        return model_and_encoder_args

    # Historical checkpoints in outputs/models were saved with stale encoder metadata.
    # Probe the small grid of legacy observation toggles and keep the combination that
    # reproduces the input dimensionality expected by the checkpoint weights.
    for include_observed_history, include_discard_summary, include_temporal_context, include_recent_actions in product(
        (False, True),
        repeat=4,
    ):
        candidate_args = dict(model_and_encoder_args)
        candidate_args.update(
            include_observed_history=include_observed_history,
            include_discard_summary=include_discard_summary,
            include_temporal_context=include_temporal_context,
            include_recent_actions=include_recent_actions,
        )
        candidate_state_dim = _probe_state_dim(
            output_root=output_root,
            stage_name=stage_name,
            device=device,
            start_mode=start_mode,
            min_burned_rounds=min_burned_rounds,
            max_burned_rounds=max_burned_rounds,
            bet_multipliers=bet_multipliers,
            penetrations=penetrations,
            base_seed=base_seed,
            n_decks=n_decks,
            shoe_penetration=shoe_penetration,
            dealer_hits_soft_17=dealer_hits_soft_17,
            blackjack_payout=blackjack_payout,
            double_allowed_on=double_allowed_on,
            double_after_split_allowed=double_after_split_allowed,
            split_rule=split_rule,
            surrender_allowed=surrender_allowed,
            insurance_allowed=insurance_allowed,
            six_card_charlie_enabled=six_card_charlie_enabled,
            blackjack_overrides=blackjack_overrides,
            observation_overrides=observation_overrides,
            start_state_overrides=start_state_overrides,
            model_and_encoder_args=candidate_args,
            betting_auxiliary_args=betting_auxiliary_args,
        )
        if candidate_state_dim == expected_state_dim:
            return candidate_args

    raise ValueError(
        "Could not reconstruct checkpoint observation features for inference. "
        f"Expected state_dim={expected_state_dim}, but the current wrapper rebuilds state_dim={current_state_dim}."
    )


def _resolve_betting_auxiliary_args(
    payload: Mapping[str, Any],
    *,
    axu_loss_bet: bool | None,
    betting_auxiliary_threshold_2x: float | None,
    betting_auxiliary_threshold_3x: float | None,
    betting_auxiliary_threshold_4x: float | None,
    betting_auxiliary_min_observed_cards: int | None,
    betting_auxiliary_class_weights: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    betting_auxiliary_config = deepcopy(dict((payload.get("pipeline_config") or {}).get("betting_auxiliary") or {}))
    enabled = bool(betting_auxiliary_config.get("enabled", False)) if axu_loss_bet is None else bool(axu_loss_bet)
    return {
        "axu_loss_bet": enabled,
        "betting_auxiliary_weight": 0.0,
        "betting_auxiliary_final_weight": 0.0,
        "betting_auxiliary_decay_steps": int(betting_auxiliary_config.get("decay_steps", 50_000)),
        "betting_auxiliary_threshold_2x": (
            float(betting_auxiliary_config.get("threshold_2x", 1.0))
            if betting_auxiliary_threshold_2x is None
            else float(betting_auxiliary_threshold_2x)
        ),
        "betting_auxiliary_threshold_3x": (
            float(betting_auxiliary_config.get("threshold_3x", 2.0))
            if betting_auxiliary_threshold_3x is None
            else float(betting_auxiliary_threshold_3x)
        ),
        "betting_auxiliary_threshold_4x": (
            float(betting_auxiliary_config.get("threshold_4x", 4.0))
            if betting_auxiliary_threshold_4x is None
            else float(betting_auxiliary_threshold_4x)
        ),
        "betting_auxiliary_min_observed_cards": (
            int(betting_auxiliary_config.get("min_observed_cards", 12))
            if betting_auxiliary_min_observed_cards is None
            else int(betting_auxiliary_min_observed_cards)
        ),
        "betting_auxiliary_class_weights": (
            tuple(float(weight) for weight in betting_auxiliary_config.get("class_weights"))
            if betting_auxiliary_class_weights is None and betting_auxiliary_config.get("class_weights") is not None
            else betting_auxiliary_class_weights
        ),
    }


def load_checkpoint_weights_for_eval(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)

    if "online_model_state_dict" in checkpoint:
        state_dict = checkpoint["online_model_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights are incompatible with the reconstructed model. "
            "Pass explicit model overrides such as architecture/feedforward_hidden_dims/"
            "use_layer_norm/use_phase_adapters/use_module_gating if needed."
        ) from exc
    if incompatible.missing_keys:
        print("Missing keys:", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        print("Unexpected keys:", incompatible.unexpected_keys)

    model.to(device)
    model.eval()
    return model


def _build_result_payload(
    *,
    checkpoint_path: str | Path,
    stage_name: str,
    metrics: Mapping[str, Any],
    device: str,
    eval_rounds: int,
    eval_max_decisions: int,
    setup: Mapping[str, Any],
) -> dict[str, Any]:
    bet_freq = dict(metrics.get("bet_action_frequencies") or {})
    play_freq = dict(metrics.get("play_action_frequencies") or {})
    result = {
        "stage_name": stage_name,
        "checkpoint_path": str(Path(checkpoint_path)),
        "device": str(device),
        "eval_rounds": int(eval_rounds),
        "eval_max_decisions": int(eval_max_decisions),
        "ev_per_1000_hands": float(metrics.get("ev_per_1000_hands", 0.0)),
        "reward_per_round": float(metrics.get("reward_per_round", 0.0)),
        "round_std": float(metrics.get("round_reward_std", 0.0)),
        "win_frac": float(metrics.get("win_rate", 0.0)),
        "push_frac": float(metrics.get("push_rate", 0.0)),
        "loss_frac": float(metrics.get("loss_rate", 0.0)),
        "blackjack_frac": float(metrics.get("blackjack_rate", 0.0)),
        "bust_frac": float(metrics.get("bust_rate", 0.0)),
        "bet_1x_frac": float(bet_freq.get("bet_1x", 0.0)),
        "bet_2x_frac": float(bet_freq.get("bet_2x", 0.0)),
        "bet_3x_frac": float(bet_freq.get("bet_3x", 0.0)),
        "bet_4x_frac": float(bet_freq.get("bet_4x", 0.0)),
        "agg_frac": float(metrics.get("aggressive_bet_fraction", 0.0)),
        "bet_ev": dict(metrics.get("bet_ev_per_1000_rounds_by_action") or {}),
        "bet_q": {
            "mean_q_bet_1x": metrics.get("mean_q_bet_1x"),
            "mean_q_bet_2x": metrics.get("mean_q_bet_2x"),
            "mean_q_bet_3x": metrics.get("mean_q_bet_3x"),
            "mean_q_bet_4x": metrics.get("mean_q_bet_4x"),
            "mean_margin_best_aggressive_vs_1x": metrics.get("mean_margin_best_aggressive_vs_1x"),
        },
        "bet_aux": {
            "count_proxy_mean": metrics.get("count_proxy_mean"),
            "count_proxy_p10": metrics.get("count_proxy_p10"),
            "count_proxy_p50": metrics.get("count_proxy_p50"),
            "count_proxy_p90": metrics.get("count_proxy_p90"),
            "target_bet_distribution": dict(metrics.get("count_proxy_target_bet_distribution") or {}),
            "greedy_bet_distribution_by_count_bucket": dict(metrics.get("greedy_bet_distribution_by_count_bucket") or {}),
            "mean_margin_by_count_bucket": dict(metrics.get("mean_margin_by_count_bucket") or {}),
            "count_proxy_bucket_stats": dict(metrics.get("count_proxy_bucket_stats") or {}),
        },
        "playing": {
            "random_action_fraction_playing": float(metrics.get("random_action_fraction_playing", 0.0)),
            "play_action_frequencies": play_freq,
        },
        "metrics": dict(metrics),
        "setup_summary": dict(setup.get("summary") or {}),
    }
    return result


def evaluate_blackjack_checkpoint(
    *,
    checkpoint_path: str | Path,
    stage_name: str | None = None,
    output_root: str | Path = Path("notebooks") / "inference",
    start_mode: str = "unknown_progress",
    min_burned_rounds: int = 10,
    max_burned_rounds: int = 60,
    bet_multipliers: tuple[int, ...] | list[int] | None = None,
    penetrations: list[float] | tuple[float, ...] | None = None,
    base_seed: int = 777,
    n_decks: int = 8,
    shoe_penetration: float = 0.75,
    dealer_hits_soft_17: bool = False,
    blackjack_payout: float = 1.5,
    double_allowed_on: str = "any_two_cards",
    double_after_split_allowed: bool = True,
    split_rule: str = "same_value",
    surrender_allowed: bool = False,
    insurance_allowed: bool = False,
    six_card_charlie_enabled: bool = False,
    eval_rounds: int = 50_000,
    eval_max_decisions: int = 600_000,
    device: str = "cpu",
    architecture: str | None = None,
    feedforward_hidden_dims: tuple[int, ...] | list[int] | None = None,
    use_layer_norm: bool | None = None,
    use_phase_adapters: bool | None = None,
    use_module_gating: bool | None = None,
    axu_loss_bet: bool | None = None,
    betting_auxiliary_threshold_2x: float | None = None,
    betting_auxiliary_threshold_3x: float | None = None,
    betting_auxiliary_threshold_4x: float | None = None,
    betting_auxiliary_min_observed_cards: int | None = None,
    betting_auxiliary_class_weights: tuple[float, float, float, float] | None = None,
    progress_every_n_rounds: int | None = None,
    print_summary: bool = True,
    summary_path: str | Path | None = None,
    blackjack_overrides: Mapping[str, Any] | None = None,
    observation_overrides: Mapping[str, Any] | None = None,
    start_state_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    resolved_stage_name = stage_name or Path(checkpoint_path).stem
    resolved_bet_multipliers = _coerce_bet_multipliers(bet_multipliers)
    resolved_penetrations = _coerce_penetrations(penetrations)
    resolved_device = resolve_device(device)

    betting_auxiliary_args = _resolve_betting_auxiliary_args(
        payload,
        axu_loss_bet=axu_loss_bet,
        betting_auxiliary_threshold_2x=betting_auxiliary_threshold_2x,
        betting_auxiliary_threshold_3x=betting_auxiliary_threshold_3x,
        betting_auxiliary_threshold_4x=betting_auxiliary_threshold_4x,
        betting_auxiliary_min_observed_cards=betting_auxiliary_min_observed_cards,
        betting_auxiliary_class_weights=betting_auxiliary_class_weights,
    )
    model_and_encoder_args = _resolve_model_and_encoder_args_for_checkpoint(
        payload,
        output_root=output_root,
        stage_name=resolved_stage_name,
        device=str(resolved_device),
        start_mode=start_mode,
        min_burned_rounds=min_burned_rounds,
        max_burned_rounds=max_burned_rounds,
        bet_multipliers=resolved_bet_multipliers,
        penetrations=resolved_penetrations,
        base_seed=base_seed,
        n_decks=n_decks,
        shoe_penetration=shoe_penetration,
        dealer_hits_soft_17=dealer_hits_soft_17,
        blackjack_payout=blackjack_payout,
        double_allowed_on=double_allowed_on,
        double_after_split_allowed=double_after_split_allowed,
        split_rule=split_rule,
        surrender_allowed=surrender_allowed,
        insurance_allowed=insurance_allowed,
        six_card_charlie_enabled=six_card_charlie_enabled,
        blackjack_overrides=blackjack_overrides,
        observation_overrides=observation_overrides,
        start_state_overrides=start_state_overrides,
        betting_auxiliary_args=betting_auxiliary_args,
        architecture=architecture,
        feedforward_hidden_dims=feedforward_hidden_dims,
        use_layer_norm=use_layer_norm,
        use_phase_adapters=use_phase_adapters,
        use_module_gating=use_module_gating,
    )

    with redirect_stdout(io.StringIO()):
        setup = run_blackjack_stage(
            stage_name=resolved_stage_name,
            output_root=output_root,
            run_training=False,
            enable_prints=False,
            print_run_summary=False,
            start_mode=start_mode,
            min_burned_rounds=min_burned_rounds,
            max_burned_rounds=max_burned_rounds,
            bet_multipliers=resolved_bet_multipliers,
            penetrations=resolved_penetrations,
            base_seed=base_seed,
            n_decks=n_decks,
            shoe_penetration=shoe_penetration,
            dealer_hits_soft_17=dealer_hits_soft_17,
            blackjack_payout=blackjack_payout,
            double_allowed_on=double_allowed_on,
            double_after_split_allowed=double_after_split_allowed,
            split_rule=split_rule,
            surrender_allowed=surrender_allowed,
            insurance_allowed=insurance_allowed,
            six_card_charlie_enabled=six_card_charlie_enabled,
            eval_rounds=eval_rounds,
            eval_max_decisions=eval_max_decisions,
            device=str(resolved_device),
            total_epochs=1,
            env_steps_per_epoch=1,
            checkpoint_directory=Path(output_root) / resolved_stage_name,
            blackjack_overrides=blackjack_overrides,
            observation_overrides=observation_overrides,
            start_state_overrides=start_state_overrides,
            **model_and_encoder_args,
            **betting_auxiliary_args,
        )

    logger = InferenceLogger(enable=print_summary)
    model = load_checkpoint_weights_for_eval(setup["model"], checkpoint_path, device=resolved_device)
    pipeline_config = setup["pipeline_config"]
    logger.log_header(
        checkpoint_path=checkpoint_path,
        stage_name=resolved_stage_name,
        device=str(resolved_device),
        eval_rounds=eval_rounds,
        eval_max_decisions=eval_max_decisions,
        base_seed=base_seed,
        model_config=_model_config_for_logging(model_and_encoder_args),
        bet_multipliers=resolved_bet_multipliers,
        penetrations=resolved_penetrations,
        start_mode=start_mode,
        min_burned_rounds=min_burned_rounds,
        max_burned_rounds=max_burned_rounds,
    )
    metrics = evaluate_policy(
        envs=setup["envs"],
        model=model,
        epsilon=deepcopy(pipeline_config.epsilon),
        num_rounds=eval_rounds,
        max_decisions=eval_max_decisions,
        rng=random.Random(base_seed),
        reset_hidden_on_round_end=pipeline_config.trainer.reset_hidden_on_round_end,
        betting_auxiliary_config=pipeline_config.betting_auxiliary,
        progress_every_n_rounds=progress_every_n_rounds,
        progress_callback=(
            (
                lambda progress_metrics, rounds_completed, decisions: logger.log_progress(
                    stage_name=resolved_stage_name,
                    rounds_completed=rounds_completed,
                    eval_rounds=eval_rounds,
                    decisions=decisions,
                    metrics=progress_metrics,
                )
            )
            if print_summary and progress_every_n_rounds is not None and int(progress_every_n_rounds) > 0
            else None
        ),
    )
    logger.log_evaluation(metrics=metrics)

    result = _build_result_payload(
        checkpoint_path=checkpoint_path,
        stage_name=resolved_stage_name,
        metrics=metrics,
        device=str(resolved_device),
        eval_rounds=eval_rounds,
        eval_max_decisions=eval_max_decisions,
        setup=setup,
    )
    logger.log_result_summary(result=result)

    if summary_path is not None:
        saved_path = write_json_file(summary_path, result)
        result["summary_path"] = str(saved_path)

    return result


def compare_blackjack_checkpoints(
    checkpoint_specs: list[dict[str, Any]],
    *,
    eval_rounds: int = 50_000,
    eval_max_decisions: int = 600_000,
    device: str = "cpu",
    progress_every_n_rounds: int | None = 2_500,
    print_summary: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    logger = InferenceLogger(enable=print_summary)
    for index, spec in enumerate(checkpoint_specs, start=1):
        stage_name = spec.get("name") or Path(str(spec["checkpoint_path"])).stem
        logger.log_model_start(
            index=index,
            total=len(checkpoint_specs),
            stage_name=stage_name,
            checkpoint_path=spec["checkpoint_path"],
        )
        result = evaluate_blackjack_checkpoint(
            checkpoint_path=spec["checkpoint_path"],
            stage_name=stage_name,
            bet_multipliers=spec.get("bet_multipliers"),
            penetrations=spec.get("penetrations"),
            start_mode=spec.get("start_mode", "unknown_progress"),
            min_burned_rounds=int(spec.get("min_burned_rounds", 10)),
            max_burned_rounds=int(spec.get("max_burned_rounds", 60)),
            base_seed=int(spec.get("base_seed", 777)),
            eval_rounds=eval_rounds,
            eval_max_decisions=eval_max_decisions,
            device=device,
            architecture=spec.get("architecture"),
            feedforward_hidden_dims=spec.get("feedforward_hidden_dims"),
            use_layer_norm=spec.get("use_layer_norm"),
            use_phase_adapters=spec.get("use_phase_adapters"),
            use_module_gating=spec.get("use_module_gating"),
            axu_loss_bet=spec.get("axu_loss_bet"),
            betting_auxiliary_threshold_2x=spec.get("betting_auxiliary_threshold_2x"),
            betting_auxiliary_threshold_3x=spec.get("betting_auxiliary_threshold_3x"),
            betting_auxiliary_threshold_4x=spec.get("betting_auxiliary_threshold_4x"),
            betting_auxiliary_min_observed_cards=spec.get("betting_auxiliary_min_observed_cards"),
            betting_auxiliary_class_weights=spec.get("betting_auxiliary_class_weights"),
            progress_every_n_rounds=spec.get("progress_every_n_rounds", progress_every_n_rounds),
            print_summary=print_summary,
        )
        results.append(result)

    logger.log_comparison(results=results)
    return results


def comparison_models(
    checkpoint_specs: list[dict[str, Any]],
    *,
    eval_rounds: int = 50_000,
    eval_max_decisions: int = 600_000,
    device: str = "cpu",
    progress_every_n_rounds: int | None = 2_500,
    print_summary: bool = True,
) -> list[dict[str, Any]]:
    return compare_blackjack_checkpoints(
        checkpoint_specs,
        eval_rounds=eval_rounds,
        eval_max_decisions=eval_max_decisions,
        device=device,
        progress_every_n_rounds=progress_every_n_rounds,
        print_summary=print_summary,
    )


def comparision_models(
    checkpoint_specs: list[dict[str, Any]],
    *,
    eval_rounds: int = 50_000,
    eval_max_decisions: int = 600_000,
    device: str = "cpu",
    progress_every_n_rounds: int | None = 2_500,
    print_summary: bool = True,
) -> list[dict[str, Any]]:
    return comparison_models(
        checkpoint_specs,
        eval_rounds=eval_rounds,
        eval_max_decisions=eval_max_decisions,
        device=device,
        progress_every_n_rounds=progress_every_n_rounds,
        print_summary=print_summary,
    )


def inference_with_comparison(**kwargs: Any) -> dict[str, Any]:
    return evaluate_blackjack_checkpoint(**kwargs)


def inference_with_comparision(**kwargs: Any) -> dict[str, Any]:
    return evaluate_blackjack_checkpoint(**kwargs)


def format_inference_result_json(result: Mapping[str, Any]) -> str:
    return to_pretty_json(result)
