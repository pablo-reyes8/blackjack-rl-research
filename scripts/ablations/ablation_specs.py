from __future__ import annotations

from copy import deepcopy
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


BASE_SPEC: dict[str, Any] = {
    "num_envs": 4,
    "seed": 101,
    "start_state": {
        "mode": "fresh_shoe",
        "min_burned_rounds": 0,
        "max_burned_rounds": 0,
        "clear_visible_histories_after_burn": True,
        "hide_reshuffle_progress_from_observation": False,
    },
    "environment": {
        "n_decks": 6,
        "shoe_penetration": 0.8,
        "dealer_hits_soft_17": False,
        "blackjack_payout": 1.5,
        "dealer_peeks_for_blackjack": True,
        "double_allowed_on": "any_two_cards",
        "double_after_split_allowed": True,
        "split_rule": "same_value",
        "max_hands_after_split": 4,
        "resplit_aces_allowed": True,
        "hit_split_aces_allowed": False,
        "surrender_allowed": True,
        "insurance_allowed": True,
        "base_bet": 1.0,
        "strict_shoe_validation": False,
        "expose_shoe_composition": False,
        "observation_profile": "table_realistic_default",
        "observation_overrides": {},
    },
    "model": {
        "architecture": "recurrent",
        "encoder_profile": "table_realistic_default",
        "activation": "relu",
        "use_layer_norm": True,
        "dropout": 0.0,
        "feedforward_hidden_dims": [256, 256],
        "projection_dim": 256,
        "recurrent_hidden_dim": 256,
        "recurrent_num_layers": 1,
        "recurrent_type": "gru",
        "head_hidden_dim": 128,
        "value_hidden_dim": 128,
        "advantage_hidden_dim": 128,
    },
    "training": {
        "trainer": {
            "total_epochs": 5,
            "env_steps_per_epoch": 512,
            "train_frequency": 2,
            "updates_per_train_step": 1,
            "max_updates_per_epoch": None,
            "device": "auto",
            "seed": 101,
            "reset_hidden_on_round_end": False,
            "sequence_end_on_done": False,
            "flush_partial_sequences_at_epoch_end": True,
            "loss": {
                "gamma": 0.99,
                "loss_type": "huber",
                "validate_current_actions": True,
                "validate_next_action_mask": True,
                "allow_terminal_without_legal_next_action": True,
            },
        },
        "replay_buffer": {
            "capacity": 50_000,
            "batch_size": 16,
            "warmup_size": 128,
            "sequence_length": 8,
            "min_sequence_length": 2,
        },
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_steps": 25_000,
            "evaluation_epsilon": 0.0,
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "scheduler": "none",
            "scheduler_step_size": 1_000,
            "scheduler_gamma": 0.99,
            "gradient_clipping": True,
            "max_grad_norm": 5.0,
        },
        "target_update": {
            "mode": "hard",
            "hard_update_interval": 250,
            "soft_tau": 0.005,
        },
        "evaluation": {
            "enabled": True,
            "every_n_epochs": 1,
            "num_rounds": 100,
            "max_decisions": 5_000,
        },
        "checkpoints": {
            "save_latest": True,
            "save_best_eval": True,
            "save_periodic": True,
            "periodic_interval_updates": 500,
            "best_metric_name": "ev_per_1000_hands",
            "maximize_best_metric": True,
        },
        "prints": {
            "enable": True,
            "print_run_summary": True,
            "print_warmup_interval": 200,
            "print_update_interval": 100,
            "print_collection_interval": 500,
            "print_epoch_header": True,
            "print_epoch_summary": True,
            "print_eval_summary": True,
            "include_segment_details": False,
        },
    },
}


def _make_spec(override: dict[str, Any]) -> dict[str, Any]:
    spec = _deep_merge(BASE_SPEC, override)
    spec["training"]["trainer"]["seed"] = spec["seed"]
    return spec


AB_1 = _make_spec(
    {
        "id": "ab_1",
        "slug": "feedforward_mse_minimal",
        "entrypoint": "ab_1_feedforward_mse.py",
        "title": "Feedforward + MSE + minimal observation",
        "description": "Removes recurrence and temporal context to test how much performance depends on sequence memory and a robust TD loss.",
        "changes": [
            "Feedforward Double DQN instead of a recurrent policy.",
            "Minimal basic-strategy-style observation profile.",
            "MSE Bellman loss instead of Huber.",
            "Hard target updates with standard Adam optimization.",
        ],
        "seed": 111,
        "environment": {
            "observation_profile": "minimal_basic_strategy",
        },
        "model": {
            "architecture": "feedforward",
            "encoder_profile": "minimal_basic_strategy",
            "use_layer_norm": False,
            "feedforward_hidden_dims": [256, 256],
        },
        "training": {
            "trainer": {
                "train_frequency": 1,
                "loss": {"loss_type": "mse"},
            },
            "replay_buffer": {
                "batch_size": 64,
                "warmup_size": 256,
            },
        },
    }
)

AB_2 = _make_spec(
    {
        "id": "ab_2",
        "slug": "gru_huber_realistic",
        "entrypoint": "ab_2_gru_huber.py",
        "title": "GRU + Huber + realistic table context",
        "description": "Sequence-aware baseline under realistic partial observability. This is the reference point for the rest of the ablation suite.",
        "changes": [
            "GRU-based recurrent Double DQN.",
            "Realistic partial-observation encoder profile.",
            "Huber Bellman loss for robust TD updates.",
            "Hard target updates with standard Adam optimization.",
        ],
        "seed": 222,
    }
)

AB_3 = _make_spec(
    {
        "id": "ab_3",
        "slug": "lstm_huber_realistic",
        "entrypoint": "ab_3_lstm_huber.py",
        "title": "LSTM + Huber + realistic table context",
        "description": "Controlled GRU-vs-LSTM comparison under the same observation regime and loss, isolating recurrent cell choice.",
        "changes": [
            "LSTM recurrent backbone instead of GRU.",
            "Same realistic partial-observation profile as the recurrent baseline.",
            "Huber Bellman loss for a controlled recurrent-cell comparison.",
            "Hard target updates with Adam to keep the comparison focused.",
        ],
        "seed": 333,
        "model": {
            "recurrent_type": "lstm",
        },
    }
)

AB_4 = _make_spec(
    {
        "id": "ab_4",
        "slug": "dueling_gru_soft_target",
        "entrypoint": "ab_4_dueling_gru_soft.py",
        "title": "Dueling GRU + soft targets + AdamW",
        "description": "Tests whether a stronger value/advantage decomposition plus smoother target tracking improves stability under realistic partial observability.",
        "changes": [
            "Dueling recurrent GRU architecture.",
            "Soft target updates instead of periodic hard copies.",
            "AdamW with weight decay for extra regularization.",
            "Light dropout to stress-test generalization stability.",
        ],
        "seed": 444,
        "model": {
            "architecture": "dueling_recurrent",
            "dropout": 0.1,
        },
        "training": {
            "optimization": {
                "optimizer": "adamw",
                "weight_decay": 0.0001,
            },
            "target_update": {
                "mode": "soft",
                "soft_tau": 0.01,
            },
        },
    }
)

AB_5 = _make_spec(
    {
        "id": "ab_5",
        "slug": "gru_mse_unknown_progress",
        "entrypoint": "ab_5_unknown_progress_mse.py",
        "title": "GRU + MSE + unknown shoe progress",
        "description": "Pushes the agent into a harder uncertainty regime where shoe progress is hidden, while also swapping to MSE and softer target updates.",
        "changes": [
            "Unknown-progress start state with hidden reshuffle progress.",
            "Table-realistic unknown-progress observation profile.",
            "MSE loss with soft target updates.",
            "Longer replay sequences with AdamW to handle stronger temporal uncertainty.",
        ],
        "seed": 555,
        "start_state": {
            "mode": "unknown_progress",
            "min_burned_rounds": 3,
            "max_burned_rounds": 12,
            "hide_reshuffle_progress_from_observation": True,
        },
        "environment": {
            "observation_profile": "table_realistic_unknown_progress",
        },
        "model": {
            "encoder_profile": "table_realistic_unknown_progress",
        },
        "training": {
            "trainer": {
                "loss": {"loss_type": "mse"},
            },
            "replay_buffer": {
                "batch_size": 12,
                "sequence_length": 12,
                "min_sequence_length": 4,
            },
            "optimization": {
                "optimizer": "adamw",
                "weight_decay": 0.0001,
            },
            "target_update": {
                "mode": "soft",
                "soft_tau": 0.01,
            },
        },
    }
)

AB_6 = _make_spec(
    {
        "id": "ab_6",
        "slug": "dueling_lstm_fully_observable",
        "entrypoint": "ab_6_fully_observable_dueling.py",
        "title": "Dueling LSTM + fully observable simulator",
        "description": "Upper-bound style ablation that gives the agent the richest observation regime together with a dueling LSTM backbone and softer optimization choices.",
        "changes": [
            "Dueling recurrent LSTM architecture.",
            "Fully observable simulator profile with exact shoe composition.",
            "Soft target updates with AdamW regularization.",
            "Slightly longer evaluation to compare stronger-information settings.",
        ],
        "seed": 666,
        "environment": {
            "observation_profile": "fully_observable_sim",
            "expose_shoe_composition": True,
        },
        "model": {
            "architecture": "dueling_recurrent",
            "encoder_profile": "fully_observable_sim",
            "recurrent_type": "lstm",
            "dropout": 0.1,
        },
        "training": {
            "epsilon": {
                "end": 0.02,
            },
            "optimization": {
                "optimizer": "adamw",
                "weight_decay": 0.0001,
            },
            "target_update": {
                "mode": "soft",
                "soft_tau": 0.01,
            },
            "evaluation": {
                "num_rounds": 150,
                "max_decisions": 7_500,
            },
        },
    }
)


ABLATION_SPECS: list[dict[str, Any]] = [AB_1, AB_2, AB_3, AB_4, AB_5, AB_6]


def get_ablation_spec(ablation_id: str) -> dict[str, Any]:
    for spec in ABLATION_SPECS:
        if spec["id"] == ablation_id:
            return deepcopy(spec)
    raise KeyError(f"Unknown ablation id: {ablation_id}")
