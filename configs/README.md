# Config Guide

This folder contains YAML experiment presets for the blackjack RL stack.

The presets are designed to be:
- reproducible
- easy to override
- friendly to CLI workflows

Most experiment files extend `configs/experiments/base.yaml`, then override only the pieces that matter for a specific run.

## File Layout

- `experiments/base.yaml`
  Shared defaults for environment, model, and training.
- `experiments/experiment.template.yaml`
  A copyable full template with the current schema spelled out explicitly.
- `experiments/*.yaml`
  Named presets for common runs such as smoke tests, feedforward baselines, and recurrent experiments.

## Top-Level Sections

Each experiment YAML can contain these sections:

- `metadata`
- `run`
- `start_state`
- `environment`
- `model`
- `training`

## `metadata`

Human-readable information about the preset.

Typical fields:
- `name`: short experiment name
- `description`: what this preset is trying to do

## `run`

Runtime settings for orchestration.

Common fields:
- `num_envs`: how many parallel environments to build
- `base_seed`: seed used as the starting point for per-environment seeds

If `num_envs` is greater than 1, environments are usually seeded as `base_seed + env_index`.

## `start_state`

Controls how each episode begins before the agent sees the first betting decision.

Fields:
- `mode`: `fresh_shoe` or `unknown_progress`
- `min_burned_rounds`: minimum hidden rounds to burn before the visible episode starts
- `max_burned_rounds`: maximum hidden rounds to burn before the visible episode starts
- `clear_visible_histories_after_burn`: whether visible histories should be wiped after hidden burn-in
- `hide_reshuffle_progress_from_observation`: whether to hide reshuffle progress details from the observation

Use `fresh_shoe` for clean debugging and deterministic tests.

Use `unknown_progress` when you want a more realistic partial-information setup where the shoe may already be in progress.

## `environment`

This config block controls blackjack rules, betting, shoe behavior, and observation settings.

Core table fields:
- `n_decks`
- `shoe_penetration`
- `use_cut_card`
- `dealer_hits_soft_17`
- `blackjack_payout`
- `dealer_peeks_for_blackjack`
- `double_allowed_on`
- `double_after_split_allowed`
- `double_split_aces_allowed`
- `split_rule`
- `max_hands_after_split`
- `max_split_depth_per_hand`
- `resplit_aces_allowed`
- `hit_split_aces_allowed`
- `surrender_allowed`
- `insurance_allowed`
- `six_card_charlie_enabled`

Betting fields:
- `base_bet`
- `bet_multipliers`

Validation and debugging fields:
- `strict_shoe_validation`
- `observation_mode`
- `expose_shoe_composition`

Nested block:
- `observation`

### `environment.observation`

This block controls what the agent is allowed to see.

Important fields:
- `profile`
- `obs_include_table_rules`
- `obs_include_visible_rules_only`
- `obs_include_hidden_rules`
- `obs_include_decision_phase`
- `obs_include_available_bet_multipliers`
- `obs_current_hand_mode`
- `obs_include_other_player_hands`
- `obs_include_current_bet`
- `obs_include_betting_context`
- `obs_include_hand_context`
- `obs_include_insurance_context`
- `obs_include_temporal_context`
- `obs_include_hands_since_shuffle`
- `obs_include_estimated_shoe_progress`
- `obs_include_last_hand_outcome`
- `obs_include_recent_actions`
- `obs_recent_actions_window`
- `obs_include_observed_cards_history`
- `obs_observed_cards_mode`
- `obs_recent_cards_window`
- `obs_reset_history_on_shuffle`
- `obs_include_exact_shoe_composition`
- `obs_include_discard_summary`
- `obs_include_n_decks`
- `obs_include_shoe_penetration_rule`

Recommended profiles:
- `minimal_basic_strategy`
  Small, compact observation for feedforward baselines.
- `table_realistic_default`
  Good default for partial observability with visible context.
- `table_realistic_unknown_progress`
  Similar to the default realistic setting, but meant for hidden shoe progress.
- `fully_observable_sim`
  Research-oriented setting with stronger visibility and exact shoe information.

## `model`

Controls the Q-network architecture.

Main fields:
- `architecture`: `feedforward`, `recurrent`, or `dueling_recurrent`
- `encoder_profile`
- `activation`
- `use_layer_norm`
- `dropout`
- `feedforward_hidden_dims`
- `projection_dim`
- `recurrent_hidden_dim`
- `recurrent_num_layers`
- `recurrent_type`
- `head_hidden_dim`
- `value_hidden_dim`
- `advantage_hidden_dim`
- `use_phase_adapters`
- `use_module_gating`

Notes:
- `feedforward_hidden_dims` matters mainly for `feedforward`.
- `head_hidden_dim` matters mainly for plain `recurrent`.
- `value_hidden_dim` and `advantage_hidden_dim` matter mainly for `dueling_recurrent`.
- `use_phase_adapters` helps the model separate betting and playing behavior.
- `use_module_gating` lets the network learn how strongly to use encoded feature groups.

## `training`

This section groups all trainer and optimization settings.

Subsections:
- `trainer`
- `replay_buffer`
- `n_step`
- `epsilon`
- `optimization`
- `target_update`
- `evaluation`
- `checkpoints`
- `prints`

### `training.trainer`

Main loop settings.

Fields:
- `total_epochs`
- `env_steps_per_epoch`
- `train_frequency`
- `updates_per_train_step`
- `max_updates_per_epoch`
- `device`
- `seed`
- `reset_hidden_on_round_end`
- `sequence_end_on_done`
- `flush_partial_sequences_at_epoch_end`
- `loss`

### `training.trainer.loss`

Bellman loss behavior.

Fields:
- `gamma`
- `loss_type`
- `validate_current_actions`
- `validate_next_action_mask`
- `allow_terminal_without_legal_next_action`
- `phase_weights`

### `training.trainer.loss.phase_weights`

Lets betting and playing transitions contribute differently to TD loss.

Fields:
- `enabled`
- `betting_weight`
- `playing_weight`

This is useful because betting decisions are less frequent and may otherwise be underweighted.

### `training.replay_buffer`

Replay storage and sampling.

Fields:
- `capacity`
- `batch_size`
- `warmup_size`
- `sequence_length`
- `min_sequence_length`

For recurrent models, `sequence_length` and `min_sequence_length` matter directly.

### `training.n_step`

Optional n-step returns.

Fields:
- `enabled`
- `n_steps`

### `training.epsilon`

Exploration schedule.

You can use either:
- one shared epsilon schedule, or
- separate schedules for betting and playing

Shared form:

```yaml
epsilon:
  start: 1.0
  end: 0.1
  decay_steps: 10000
  evaluation_epsilon: 0.0
```

Dual form:

```yaml
epsilon:
  betting:
    start: 1.0
    end: 0.10
    decay_steps: 40000
    evaluation_epsilon: 0.0
  playing:
    start: 1.0
    end: 0.03
    decay_steps: 25000
    evaluation_epsilon: 0.0
```

The dual form is usually better for the current two-phase action space.

### `training.optimization`

Optimizer and scheduler settings.

Fields:
- `optimizer`
- `learning_rate`
- `weight_decay`
- `scheduler`
- `scheduler_step_size`
- `scheduler_gamma`
- `gradient_clipping`
- `max_grad_norm`

### `training.target_update`

Target network synchronization.

Fields:
- `mode`: `hard` or `soft`
- `hard_update_interval`
- `soft_tau`

### `training.evaluation`

Evaluation cadence and limits.

Fields:
- `enabled`
- `every_n_epochs`
- `num_rounds`
- `max_decisions`

### `training.checkpoints`

Checkpoint saving behavior.

Fields:
- `directory`
- `save_latest`
- `save_best_eval`
- `save_periodic`
- `periodic_interval_updates`
- `best_metric_name`
- `maximize_best_metric`

### `training.prints`

Console logging and verbosity.

Fields:
- `enable`
- `print_run_summary`
- `print_warmup_interval`
- `print_update_interval`
- `print_collection_interval`
- `print_epoch_header`
- `print_epoch_summary`
- `print_eval_summary`
- `include_segment_details`

## Inheritance with `extends`

Presets can inherit from one or more YAML files.

Example:

```yaml
extends: base.yaml

metadata:
  name: my-custom-run

environment:
  observation:
    profile: fully_observable_sim
```

The loader deep-merges child values over parent values.

## Practical Advice

- Start from `smoke-test.yaml` when validating code paths quickly.
- Start from `feedforward-basic.yaml` for cheap compact baselines.
- Start from `recurrent-table-default.yaml` for realistic partial-observation training.
- Use `dueling-unknown-progress.yaml` when you want hidden-progress difficulty.
- Use `fully-observable-sim.yaml` for research-style upper-bound experiments.

If you want a full explicit template, copy `experiments/experiment.template.yaml`.
