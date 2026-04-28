from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from loss import BellmanLossConfig, LossPhaseWeightConfig
from model.agents import AgentNetworkConfig, DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN
from model.encoder import BlackjackObservationEncoder, EncoderConfig

from training import (
    CheckpointConfig,
    DistillationConfig,
    DualEpsilonConfig,
    EpsilonScheduleConfig,
    EvaluationConfig,
    NStepConfig,
    OptimizationConfig,
    PrintConfig,
    ReplayBufferConfig,
    TargetUpdateConfig,
    TrainerConfig,
    TrainingPipelineConfig,
    TransferLearningConfig,
    build_optimizer_with_param_groups,
    freeze_playing_policy_parts,
    train_model,
)


MODEL_CLASS_BY_ARCHITECTURE = {
    "feedforward": FeedForwardDoubleDQN,
    "recurrent": RecurrentDoubleDQN,
    "dueling_recurrent": DuelingRecurrentDoubleDQN,
}


def _apply_overrides(config: Any, overrides: Mapping[str, Any] | None) -> Any:
    if not overrides:
        return config
    for key, value in overrides.items():
        setattr(config, key, value)
    post_init = getattr(config, "__post_init__", None)
    if callable(post_init):
        post_init()
    return config


def _build_start_state_config(
    *,
    start_mode: str,
    min_burned_rounds: int,
    max_burned_rounds: int,
    clear_visible_histories_after_burn: bool,
    hide_reshuffle_progress_from_observation: bool | None,
    start_state_overrides: Mapping[str, Any] | None,
) -> StartStateConfig:
    resolved_hide_progress = (
        start_mode == "unknown_progress"
        if hide_reshuffle_progress_from_observation is None
        else hide_reshuffle_progress_from_observation
    )
    config = StartStateConfig(
        mode=start_mode,
        min_burned_rounds=min_burned_rounds,
        max_burned_rounds=max_burned_rounds,
        clear_visible_histories_after_burn=clear_visible_histories_after_burn,
        hide_reshuffle_progress_from_observation=resolved_hide_progress,
    )
    return _apply_overrides(config, start_state_overrides)


def _build_observation_config(
    *,
    observation_profile: str,
    observation_mode: str | None,
    include_observed_history: bool,
    include_discard_summary: bool,
    include_temporal_context: bool,
    include_recent_actions: bool,
    obs_include_table_rules: bool | None,
    obs_include_visible_rules_only: bool | None,
    obs_include_hidden_rules: bool | None,
    obs_include_decision_phase: bool | None,
    obs_include_available_bet_multipliers: bool | None,
    obs_include_other_player_hands: bool | None,
    obs_include_current_bet: bool | None,
    obs_include_betting_context: bool | None,
    obs_include_hand_context: bool | None,
    obs_include_insurance_context: bool | None,
    obs_include_hands_since_shuffle: bool | None,
    obs_include_estimated_shoe_progress: bool | None,
    obs_include_last_hand_outcome: bool | None,
    obs_recent_actions_window: int | None,
    obs_observed_cards_mode: str | None,
    obs_recent_cards_window: int | None,
    obs_reset_history_on_shuffle: bool | None,
    obs_include_exact_shoe_composition: bool | None,
    obs_include_n_decks: bool | None,
    obs_include_shoe_penetration_rule: bool | None,
    observation_overrides: Mapping[str, Any] | None,
) -> ObservationConfig:
    config = ObservationConfig.for_profile(observation_profile)

    config.obs_include_temporal_context = include_temporal_context
    config.obs_include_observed_cards_history = include_observed_history
    config.obs_include_discard_summary = include_discard_summary
    config.obs_include_recent_actions = include_recent_actions

    explicit_values = {
        "obs_current_hand_mode": observation_mode,
        "obs_include_table_rules": obs_include_table_rules,
        "obs_include_visible_rules_only": obs_include_visible_rules_only,
        "obs_include_hidden_rules": obs_include_hidden_rules,
        "obs_include_decision_phase": obs_include_decision_phase,
        "obs_include_available_bet_multipliers": obs_include_available_bet_multipliers,
        "obs_include_other_player_hands": obs_include_other_player_hands,
        "obs_include_current_bet": obs_include_current_bet,
        "obs_include_betting_context": obs_include_betting_context,
        "obs_include_hand_context": obs_include_hand_context,
        "obs_include_insurance_context": obs_include_insurance_context,
        "obs_include_hands_since_shuffle": obs_include_hands_since_shuffle,
        "obs_include_estimated_shoe_progress": obs_include_estimated_shoe_progress,
        "obs_include_last_hand_outcome": obs_include_last_hand_outcome,
        "obs_recent_actions_window": obs_recent_actions_window,
        "obs_observed_cards_mode": obs_observed_cards_mode,
        "obs_recent_cards_window": obs_recent_cards_window,
        "obs_reset_history_on_shuffle": obs_reset_history_on_shuffle,
        "obs_include_exact_shoe_composition": obs_include_exact_shoe_composition,
        "obs_include_n_decks": obs_include_n_decks,
        "obs_include_shoe_penetration_rule": obs_include_shoe_penetration_rule,
    }
    for key, value in explicit_values.items():
        if value is not None:
            setattr(config, key, value)

    return _apply_overrides(config, observation_overrides)


def _build_encoder_config(
    *,
    encoder_profile: str,
    include_observed_history: bool,
    include_discard_summary: bool,
    include_temporal_context: bool,
    include_recent_actions: bool,
    encode_rules: bool | None,
    encode_betting_context: bool | None,
    encode_other_hands: bool | None,
    encode_exact_shoe: bool | None,
    encode_action_mask_features: bool | None,
    history_encoding: str | None,
    normalize_counts: bool | None,
    use_visible_table_rules_only: bool | None,
    max_current_hand_cards: int | None,
    max_cards_per_hand: int | None,
    max_other_hands: int | None,
    max_recent_actions: int | None,
    max_recent_cards: int | None,
    max_recent_discard_cards: int | None,
    encoder_overrides: Mapping[str, Any] | None,
) -> EncoderConfig:
    config = EncoderConfig.for_profile(encoder_profile)
    config.encode_temporal = include_temporal_context
    config.encode_observed_history = include_observed_history
    config.encode_discard_summary = include_discard_summary
    config.encode_recent_actions = include_recent_actions

    explicit_values = {
        "encode_rules": encode_rules,
        "encode_betting_context": encode_betting_context,
        "encode_other_hands": encode_other_hands,
        "encode_exact_shoe": encode_exact_shoe,
        "encode_action_mask_features": encode_action_mask_features,
        "history_encoding": history_encoding,
        "normalize_counts": normalize_counts,
        "use_visible_table_rules_only": use_visible_table_rules_only,
        "max_current_hand_cards": max_current_hand_cards,
        "max_cards_per_hand": max_cards_per_hand,
        "max_other_hands": max_other_hands,
        "max_recent_actions": max_recent_actions,
        "max_recent_cards": max_recent_cards,
        "max_recent_discard_cards": max_recent_discard_cards,
    }
    for key, value in explicit_values.items():
        if value is not None:
            setattr(config, key, value)

    return _apply_overrides(config, encoder_overrides)


def _build_model(
    *,
    architecture: str,
    encoder: BlackjackObservationEncoder,
    encoder_profile: str,
    activation: str,
    use_layer_norm: bool,
    dropout: float,
    feedforward_hidden_dims: tuple[int, ...],
    projection_dim: int,
    recurrent_hidden_dim: int,
    recurrent_num_layers: int,
    recurrent_type: str,
    head_hidden_dim: int,
    value_hidden_dim: int,
    advantage_hidden_dim: int,
    use_phase_adapters: bool,
    use_module_gating: bool,
    model_overrides: Mapping[str, Any] | None,
) -> Any:
    if architecture not in MODEL_CLASS_BY_ARCHITECTURE:
        raise ValueError("architecture must be 'feedforward', 'recurrent', or 'dueling_recurrent'")

    model_config = AgentNetworkConfig.for_architecture(
        architecture=architecture,
        encoder_profile=encoder_profile,
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        feedforward_hidden_dims=feedforward_hidden_dims,
        projection_dim=projection_dim,
        recurrent_hidden_dim=recurrent_hidden_dim,
        recurrent_num_layers=recurrent_num_layers,
        recurrent_type=recurrent_type,
        head_hidden_dim=head_hidden_dim,
        value_hidden_dim=value_hidden_dim,
        advantage_hidden_dim=advantage_hidden_dim,
        use_phase_adapters=use_phase_adapters,
        use_module_gating=use_module_gating,
    )
    _apply_overrides(model_config, model_overrides)

    model_class = MODEL_CLASS_BY_ARCHITECTURE[architecture]
    return model_class(config=model_config, encoder=encoder)


def _resolve_env_penetrations(*, shoe_penetration: float, penetrations: list[float] | tuple[float, ...] | None, num_envs: int) -> list[float]:
    if penetrations is not None:
        return [float(value) for value in penetrations]
    return [float(shoe_penetration)] * num_envs


def run_blackjack_transfer_stage(
    *,
    stage_name: str,
    output_root: str | Path = Path("notebooks") / "training_checkpoints",

    # Execution mode
    run_training: bool = True,
    resume: bool = False,
    resume_checkpoint_path: str | Path | None = None,

    # Transfer learning
    transfer_enabled: bool | None = None,
    warm_start_checkpoint_path: str | Path | None = None,
    teacher_checkpoint_path: str | Path | None = None,
    warm_start_state_key: str = "online_model_state_dict",
    warm_start_allow_input_dim_padding: bool = True,
    warm_start_allow_partial: bool = True,
    warm_start_verbose: bool = False,
    distillation_enabled: bool = True,
    distillation_mode: str = "q_mse",
    distillation_weight: float = 0.20,
    distillation_final_weight: float = 0.05,
    distillation_decay_steps: int = 50_000,
    distillation_temperature: float = 1.0,
    distillation_playing_only: bool = True,

    # Environment fanout
    base_seed: int = 44,
    num_envs: int = 1,
    penetrations: list[float] | tuple[float, ...] | None = None,

    # Observation shortcuts
    observation_profile: str = "table_realistic_unknown_progress",
    observation_mode: str | None = "table_raw",
    include_observed_history: bool = False,
    include_discard_summary: bool = False,
    include_temporal_context: bool = False,
    include_recent_actions: bool = False,

    # Start state
    start_mode: str = "fresh_shoe",
    min_burned_rounds: int = 0,
    max_burned_rounds: int = 0,
    clear_visible_histories_after_burn: bool = True,
    hide_reshuffle_progress_from_observation: bool | None = None,

    # Table / game rules
    n_decks: int = 8,
    shoe_penetration: float = 0.75,
    use_cut_card: bool = True,
    visible_shoe_change: bool = True,
    exogenous_cards: bool = False,
    simulate_exogenous_visible_cards: bool = False,
    exogenous_visible_cards_mode: str = "disabled",
    dealer_hits_soft_17: bool = False,
    blackjack_payout: float = 1.5,
    dealer_peeks_for_blackjack: bool = True,
    double_allowed_on: str = "any_two_cards",
    double_after_split_allowed: bool = True,
    double_split_aces_allowed: bool = False,
    split_rule: str = "same_value",
    max_hands_after_split: int = 2,
    max_split_depth_per_hand: int | None = 1,
    resplit_aces_allowed: bool = False,
    hit_split_aces_allowed: bool = False,
    surrender_allowed: bool = False,
    insurance_allowed: bool = False,
    six_card_charlie_enabled: bool = False,
    base_bet: float = 1.0,
    bet_multipliers: tuple[int, ...] = (1,),
    strict_shoe_validation: bool = False,
    expose_shoe_composition: bool = False,

    # Observation fine-tuning
    obs_include_table_rules: bool | None = None,
    obs_include_visible_rules_only: bool | None = None,
    obs_include_hidden_rules: bool | None = None,
    obs_include_decision_phase: bool | None = None,
    obs_include_available_bet_multipliers: bool | None = None,
    obs_include_other_player_hands: bool | None = None,
    obs_include_current_bet: bool | None = None,
    obs_include_betting_context: bool | None = None,
    obs_include_hand_context: bool | None = None,
    obs_include_insurance_context: bool | None = None,
    obs_include_hands_since_shuffle: bool | None = None,
    obs_include_estimated_shoe_progress: bool | None = None,
    obs_include_last_hand_outcome: bool | None = None,
    obs_recent_actions_window: int | None = None,
    obs_observed_cards_mode: str | None = None,
    obs_recent_cards_window: int | None = None,
    obs_reset_history_on_shuffle: bool | None = None,
    obs_include_exact_shoe_composition: bool | None = None,
    obs_include_n_decks: bool | None = None,
    obs_include_shoe_penetration_rule: bool | None = None,

    # Encoder
    encoder_profile: str | None = None,
    encode_rules: bool | None = None,
    encode_betting_context: bool | None = None,
    encode_other_hands: bool | None = None,
    encode_exact_shoe: bool | None = None,
    encode_action_mask_features: bool | None = None,
    history_encoding: str | None = None,
    normalize_counts: bool | None = None,
    use_visible_table_rules_only: bool | None = None,
    max_current_hand_cards: int | None = None,
    max_cards_per_hand: int | None = None,
    max_other_hands: int | None = None,
    max_recent_actions: int | None = None,
    max_recent_cards: int | None = None,
    max_recent_discard_cards: int | None = None,

    # Model
    architecture: str = "feedforward",
    activation: str = "relu",
    use_layer_norm: bool = False,
    dropout: float = 0.0,
    feedforward_hidden_dims: tuple[int, ...] = (256, 256, 128),
    projection_dim: int = 256,
    recurrent_hidden_dim: int = 256,
    recurrent_num_layers: int = 1,
    recurrent_type: str = "gru",
    head_hidden_dim: int = 128,
    value_hidden_dim: int = 128,
    advantage_hidden_dim: int = 128,
    use_phase_adapters: bool = False,
    use_module_gating: bool = False,
    freeze_playing_parts: bool = False,
    use_optimizer_param_groups: bool = False,
    backbone_lr: float = 1e-5,
    play_lr: float = 1e-5,
    bet_lr: float = 3e-4,
    default_lr: float = 1e-4,
    param_group_optimizer_name: str = "adamw",

    # Loss
    gamma: float = 0.99,
    loss_type: str = "huber",
    validate_current_actions: bool = True,
    validate_next_action_mask: bool = True,
    allow_terminal_without_legal_next_action: bool = True,
    phase_weights_enabled: bool = True,
    betting_loss_weight: float = 0.25,
    playing_loss_weight: float = 1.50,

    # Epsilon
    betting_epsilon_start: float = 0.20,
    betting_epsilon_end: float = 0.02,
    betting_epsilon_decay_steps: int = 40_000,
    betting_evaluation_epsilon: float = 0.0,
    playing_epsilon_start: float = 0.18,
    playing_epsilon_end: float = 0.05,
    playing_epsilon_decay_steps: int = 80_000,
    playing_evaluation_epsilon: float = 0.0,

    # N-step
    n_step_enabled: bool = True,
    n_steps: int = 3,

    # Trainer
    total_epochs: int = 25,
    env_steps_per_epoch: int = 4_500,
    train_frequency: int = 4,
    updates_per_train_step: int = 1,
    max_updates_per_epoch: int | None = None,
    device: str = "cpu",
    reset_hidden_on_round_end: bool = False,
    sequence_end_on_done: bool = False,
    flush_partial_sequences_at_epoch_end: bool = True,

    # Replay
    replay_capacity: int = 120_000,
    batch_size: int = 128,
    warmup_size: int = 12_000,
    sequence_length: int = 8,
    min_sequence_length: int = 2,

    # Optimization
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_name: str = "step",
    scheduler_step_size: int = 25_000,
    scheduler_gamma: float = 0.97,
    gradient_clipping: bool = True,
    max_grad_norm: float = 5.0,

    # Target update
    target_update_mode: str = "soft",
    target_hard_interval: int = 1_000,
    target_soft_tau: float = 0.005,

    # Evaluation
    evaluation_enabled: bool = True,
    eval_every_n_epochs: int = 1,
    eval_rounds: int = 2_500,
    eval_max_decisions: int = 25_000,

    # Checkpoints
    checkpoint_directory: str | Path | None = None,
    save_latest: bool = True,
    save_best_eval: bool = True,
    save_periodic: bool = True,
    periodic_interval_updates: int = 2_500,
    best_metric_name: str = "ev_per_1000_hands",
    maximize_best_metric: bool = True,

    # Prints
    enable_prints: bool = True,
    print_run_summary: bool = True,
    print_warmup_interval: int = 1_000,
    print_update_interval: int = 200,
    print_collection_interval: int = 1_000,
    print_epoch_header: bool = True,
    print_epoch_summary: bool = True,
    print_eval_summary: bool = True,
    include_segment_details: bool = False,

    # Escape hatches
    observation_overrides: Mapping[str, Any] | None = None,
    start_state_overrides: Mapping[str, Any] | None = None,
    blackjack_overrides: Mapping[str, Any] | None = None,
    encoder_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    replay_buffer_overrides: Mapping[str, Any] | None = None,
    optimization_overrides: Mapping[str, Any] | None = None,
    target_update_overrides: Mapping[str, Any] | None = None,
    evaluation_overrides: Mapping[str, Any] | None = None,
    checkpoint_overrides: Mapping[str, Any] | None = None,
    print_overrides: Mapping[str, Any] | None = None,
    trainer_overrides: Mapping[str, Any] | None = None,
    transfer_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Wrapper de alto nivel para stages de Blackjack RL.

    La idea es que puedas cambiar desde aquí:
    - arquitectura del modelo (`feedforward`, `recurrent`, `dueling_recurrent`)
    - perfiles de observación y encoder
    - reglas de mesa y `bet_multipliers`
    - epsilons de betting y playing
    - loss y phase weights
    - transfer learning, teacher y warm start
    - freezing de playing y optimizer con param groups

    Si algún ajuste fino no está expuesto de forma explícita, usa los `*_overrides`.
    Esos mappings se aplican al final y tienen prioridad.
    """

    resolved_output_root = Path(output_root)
    resolved_encoder_profile = encoder_profile or observation_profile
    resolved_transfer_enabled = (
        bool(warm_start_checkpoint_path or teacher_checkpoint_path)
        if transfer_enabled is None
        else transfer_enabled
    )

    observation_config = _build_observation_config(
        observation_profile=observation_profile,
        observation_mode=observation_mode,
        include_observed_history=include_observed_history,
        include_discard_summary=include_discard_summary,
        include_temporal_context=include_temporal_context,
        include_recent_actions=include_recent_actions,
        obs_include_table_rules=obs_include_table_rules,
        obs_include_visible_rules_only=obs_include_visible_rules_only,
        obs_include_hidden_rules=obs_include_hidden_rules,
        obs_include_decision_phase=obs_include_decision_phase,
        obs_include_available_bet_multipliers=obs_include_available_bet_multipliers,
        obs_include_other_player_hands=obs_include_other_player_hands,
        obs_include_current_bet=obs_include_current_bet,
        obs_include_betting_context=obs_include_betting_context,
        obs_include_hand_context=obs_include_hand_context,
        obs_include_insurance_context=obs_include_insurance_context,
        obs_include_hands_since_shuffle=obs_include_hands_since_shuffle,
        obs_include_estimated_shoe_progress=obs_include_estimated_shoe_progress,
        obs_include_last_hand_outcome=obs_include_last_hand_outcome,
        obs_recent_actions_window=obs_recent_actions_window,
        obs_observed_cards_mode=obs_observed_cards_mode,
        obs_recent_cards_window=obs_recent_cards_window,
        obs_reset_history_on_shuffle=obs_reset_history_on_shuffle,
        obs_include_exact_shoe_composition=obs_include_exact_shoe_composition,
        obs_include_n_decks=obs_include_n_decks,
        obs_include_shoe_penetration_rule=obs_include_shoe_penetration_rule,
        observation_overrides=observation_overrides,
    )

    start_state_config = _build_start_state_config(
        start_mode=start_mode,
        min_burned_rounds=min_burned_rounds,
        max_burned_rounds=max_burned_rounds,
        clear_visible_histories_after_burn=clear_visible_histories_after_burn,
        hide_reshuffle_progress_from_observation=hide_reshuffle_progress_from_observation,
        start_state_overrides=start_state_overrides,
    )

    blackjack_config = BlackjackConfig(
        n_decks=n_decks,
        shoe_penetration=shoe_penetration,
        use_cut_card=use_cut_card,
        visible_shoe_change=visible_shoe_change,
        exogenous_cards=exogenous_cards,
        simulate_exogenous_visible_cards=simulate_exogenous_visible_cards,
        exogenous_visible_cards_mode=exogenous_visible_cards_mode,
        dealer_hits_soft_17=dealer_hits_soft_17,
        blackjack_payout=blackjack_payout,
        dealer_peeks_for_blackjack=dealer_peeks_for_blackjack,
        double_allowed_on=double_allowed_on,
        double_after_split_allowed=double_after_split_allowed,
        double_split_aces_allowed=double_split_aces_allowed,
        split_rule=split_rule,
        max_hands_after_split=max_hands_after_split,
        max_split_depth_per_hand=max_split_depth_per_hand,
        resplit_aces_allowed=resplit_aces_allowed,
        hit_split_aces_allowed=hit_split_aces_allowed,
        surrender_allowed=surrender_allowed,
        insurance_allowed=insurance_allowed,
        six_card_charlie_enabled=six_card_charlie_enabled,
        base_bet=base_bet,
        bet_multipliers=bet_multipliers,
        strict_shoe_validation=strict_shoe_validation,
        observation=observation_config,
        observation_mode=None,
        expose_shoe_composition=expose_shoe_composition,
    )
    _apply_overrides(blackjack_config, blackjack_overrides)

    envs: list[BlackjackEnvironment] = []
    resolved_penetrations = _resolve_env_penetrations(
        shoe_penetration=blackjack_config.shoe_penetration,
        penetrations=penetrations,
        num_envs=num_envs,
    )
    for env_index, penetration in enumerate(resolved_penetrations):
        env_config = deepcopy(blackjack_config)
        env_config.shoe_penetration = penetration
        envs.append(
            BlackjackEnvironment(
                config=env_config,
                seed=base_seed + env_index,
                start_state=deepcopy(start_state_config),
            )
        )

    encoder_config = _build_encoder_config(
        encoder_profile=resolved_encoder_profile,
        include_observed_history=include_observed_history,
        include_discard_summary=include_discard_summary,
        include_temporal_context=include_temporal_context,
        include_recent_actions=include_recent_actions,
        encode_rules=encode_rules,
        encode_betting_context=encode_betting_context,
        encode_other_hands=encode_other_hands,
        encode_exact_shoe=encode_exact_shoe,
        encode_action_mask_features=encode_action_mask_features,
        history_encoding=history_encoding,
        normalize_counts=normalize_counts,
        use_visible_table_rules_only=use_visible_table_rules_only,
        max_current_hand_cards=max_current_hand_cards,
        max_cards_per_hand=max_cards_per_hand,
        max_other_hands=max_other_hands,
        max_recent_actions=max_recent_actions,
        max_recent_cards=max_recent_cards,
        max_recent_discard_cards=max_recent_discard_cards,
        encoder_overrides=encoder_overrides,
    )
    encoder = BlackjackObservationEncoder(config=encoder_config)

    model = _build_model(
        architecture=architecture,
        encoder=encoder,
        encoder_profile=encoder_config.profile,
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        feedforward_hidden_dims=feedforward_hidden_dims,
        projection_dim=projection_dim,
        recurrent_hidden_dim=recurrent_hidden_dim,
        recurrent_num_layers=recurrent_num_layers,
        recurrent_type=recurrent_type,
        head_hidden_dim=head_hidden_dim,
        value_hidden_dim=value_hidden_dim,
        advantage_hidden_dim=advantage_hidden_dim,
        use_phase_adapters=use_phase_adapters,
        use_module_gating=use_module_gating,
        model_overrides=model_overrides,
    )

    if freeze_playing_parts:
        freeze_playing_policy_parts(model)

    loss_config = BellmanLossConfig(
        gamma=gamma,
        loss_type=loss_type,
        validate_current_actions=validate_current_actions,
        validate_next_action_mask=validate_next_action_mask,
        allow_terminal_without_legal_next_action=allow_terminal_without_legal_next_action,
        phase_weights=LossPhaseWeightConfig(
            enabled=phase_weights_enabled,
            betting_weight=betting_loss_weight,
            playing_weight=playing_loss_weight,
        ),
    )

    epsilon_config = DualEpsilonConfig(
        betting=EpsilonScheduleConfig(
            start=betting_epsilon_start,
            end=betting_epsilon_end,
            decay_steps=betting_epsilon_decay_steps,
            evaluation_epsilon=betting_evaluation_epsilon,
        ),
        playing=EpsilonScheduleConfig(
            start=playing_epsilon_start,
            end=playing_epsilon_end,
            decay_steps=playing_epsilon_decay_steps,
            evaluation_epsilon=playing_evaluation_epsilon,
        ),
    )

    replay_buffer_config = ReplayBufferConfig(
        capacity=replay_capacity,
        batch_size=batch_size,
        warmup_size=warmup_size,
        sequence_length=sequence_length,
        min_sequence_length=min_sequence_length,
    )
    _apply_overrides(replay_buffer_config, replay_buffer_overrides)

    optimization_config = OptimizationConfig(
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler_name,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        gradient_clipping=gradient_clipping,
        max_grad_norm=max_grad_norm,
    )
    _apply_overrides(optimization_config, optimization_overrides)

    target_update_config = TargetUpdateConfig(
        mode=target_update_mode,
        hard_update_interval=target_hard_interval,
        soft_tau=target_soft_tau,
    )
    _apply_overrides(target_update_config, target_update_overrides)

    evaluation_config = EvaluationConfig(
        enabled=evaluation_enabled,
        every_n_epochs=eval_every_n_epochs,
        num_rounds=eval_rounds,
        max_decisions=eval_max_decisions,
    )
    _apply_overrides(evaluation_config, evaluation_overrides)

    resolved_checkpoint_directory = Path(checkpoint_directory) if checkpoint_directory is not None else resolved_output_root / stage_name
    checkpoint_config = CheckpointConfig(
        directory=str(resolved_checkpoint_directory),
        save_latest=save_latest,
        save_best_eval=save_best_eval,
        save_periodic=save_periodic,
        periodic_interval_updates=periodic_interval_updates,
        best_metric_name=best_metric_name,
        maximize_best_metric=maximize_best_metric,
    )
    _apply_overrides(checkpoint_config, checkpoint_overrides)

    print_config = PrintConfig(
        enable=enable_prints,
        print_run_summary=print_run_summary,
        print_warmup_interval=print_warmup_interval,
        print_update_interval=print_update_interval,
        print_collection_interval=print_collection_interval,
        print_epoch_header=print_epoch_header,
        print_epoch_summary=print_epoch_summary,
        print_eval_summary=print_eval_summary,
        include_segment_details=include_segment_details,
    )
    _apply_overrides(print_config, print_overrides)

    trainer_config = TrainerConfig(
        total_epochs=total_epochs,
        env_steps_per_epoch=env_steps_per_epoch,
        train_frequency=train_frequency,
        updates_per_train_step=updates_per_train_step,
        max_updates_per_epoch=max_updates_per_epoch,
        device=device,
        seed=base_seed,
        reset_hidden_on_round_end=reset_hidden_on_round_end,
        sequence_end_on_done=sequence_end_on_done,
        flush_partial_sequences_at_epoch_end=flush_partial_sequences_at_epoch_end,
        loss=loss_config,
    )
    _apply_overrides(trainer_config, trainer_overrides)

    transfer_config = TransferLearningConfig(
        enabled=resolved_transfer_enabled,
        warm_start_checkpoint_path=str(warm_start_checkpoint_path) if warm_start_checkpoint_path is not None else None,
        teacher_checkpoint_path=str(teacher_checkpoint_path) if teacher_checkpoint_path is not None else None,
        distillation=DistillationConfig(
            enabled=distillation_enabled and teacher_checkpoint_path is not None,
            mode=distillation_mode,
            weight=distillation_weight,
            final_weight=distillation_final_weight,
            decay_steps=distillation_decay_steps,
            temperature=distillation_temperature,
            playing_only=distillation_playing_only,
        ),
    )
    _apply_overrides(transfer_config, transfer_overrides)

    n_step_config = NStepConfig(enabled=n_step_enabled, n_steps=n_steps)

    pipeline_config = TrainingPipelineConfig(
        trainer=trainer_config,
        replay_buffer=replay_buffer_config,
        epsilon=epsilon_config,
        n_step=n_step_config,
        optimization=optimization_config,
        target_update=target_update_config,
        evaluation=evaluation_config,
        checkpoints=checkpoint_config,
        transfer=transfer_config,
        prints=print_config,
    )

    optimizer = None
    if use_optimizer_param_groups:
        optimizer = build_optimizer_with_param_groups(
            model,
            backbone_lr=backbone_lr,
            play_lr=play_lr,
            bet_lr=bet_lr,
            default_lr=default_lr,
            weight_decay=optimization_config.weight_decay,
            optimizer_name=param_group_optimizer_name,
        )

    summary = {
        "stage_name": stage_name,
        "architecture": architecture,
        "observation_profile": observation_config.profile,
        "encoder_profile": encoder_config.profile,
        "start_mode": start_state_config.mode,
        "num_envs": len(envs),
        "penetrations": resolved_penetrations,
        "checkpoint_dir": str(checkpoint_config.directory_path),
        "resume": resume,
        "resume_checkpoint_path": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        "transfer_enabled": transfer_config.enabled,
        "warm_start_checkpoint_path": transfer_config.warm_start_checkpoint_path,
        "teacher_checkpoint_path": transfer_config.teacher_checkpoint_path,
        "distillation_enabled": transfer_config.distillation.enabled,
        "bet_multipliers": blackjack_config.bet_multipliers,
        "state_dim": model.state_dim,
        "use_optimizer_param_groups": use_optimizer_param_groups,
        "freeze_playing_parts": freeze_playing_parts,
    }

    print("=" * 96)
    print("BLACKJACK HIGH-LEVEL WRAPPER")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 96)

    result = {
        "envs": envs,
        "model": model,
        "optimizer": optimizer,
        "pipeline_config": pipeline_config,
        "summary": summary,
        "blackjack_config": blackjack_config,
        "observation_config": observation_config,
        "encoder_config": encoder_config,
        "start_state_config": start_state_config,
    }
    if not run_training:
        return result

    training_result = train_model(
        envs=envs,
        model=model,
        pipeline_config=pipeline_config,
        optimizer=optimizer,
        resume=resume,
        resume_checkpoint_path=resume_checkpoint_path,
        warm_start_checkpoint_path=warm_start_checkpoint_path,
        warm_start_state_key=warm_start_state_key,
        warm_start_allow_input_dim_padding=warm_start_allow_input_dim_padding,
        warm_start_allow_partial=warm_start_allow_partial,
        warm_start_verbose=warm_start_verbose,
    )
    training_result.update(result)
    training_result["stage_summary"] = summary
    return training_result


run_blackjack_stage = run_blackjack_transfer_stage


__all__ = [
    "run_blackjack_stage",
    "run_blackjack_transfer_stage",
]
