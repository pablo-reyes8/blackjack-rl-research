from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.evaluation import evaluate_policy

from scripts.blackjack_rl_cli.common import (
    build_model_from_checkpoint,
    build_training_pipeline_config,
    load_checkpoint_payload,
    load_experiment_config,
    resolve_device,
    resolve_training_setup,
    summarize_setup,
    to_pretty_json,
    write_json_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved blackjack RL checkpoint.")
    parser.add_argument("--experiment-config", required=True, help="Path to the experiment YAML used for environment setup.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file (.pt).")
    parser.add_argument("--num-envs", type=int, help="Override run.num_envs for evaluation.")
    parser.add_argument("--num-rounds", type=int, help="Override evaluation.num_rounds.")
    parser.add_argument("--max-decisions", type=int, help="Override evaluation.max_decisions.")
    parser.add_argument("--epsilon", type=float, help="Override both betting and playing evaluation epsilon.")
    parser.add_argument("--betting-epsilon", type=float, help="Override evaluation epsilon for the betting phase only.")
    parser.add_argument("--playing-epsilon", type=float, help="Override evaluation epsilon for the playing phase only.")
    parser.add_argument("--seed", type=int, help="Random seed for evaluation sampling.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device used for model evaluation.")
    parser.add_argument("--summary-path", help="Optional JSON file for evaluation output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    experiment = load_experiment_config(args.experiment_config)
    if args.num_envs is not None:
        experiment.setdefault("run", {})["num_envs"] = args.num_envs

    setup = resolve_training_setup(experiment)
    payload = load_checkpoint_payload(args.checkpoint, map_location="cpu")
    checkpoint_pipeline = build_training_pipeline_config(payload.get("pipeline_config"))

    device = resolve_device(args.device)
    model = build_model_from_checkpoint(payload)
    model.to(device)

    evaluation_seed = args.seed if args.seed is not None else checkpoint_pipeline.trainer.seed + 1000
    num_rounds = args.num_rounds if args.num_rounds is not None else checkpoint_pipeline.evaluation.num_rounds
    max_decisions = args.max_decisions if args.max_decisions is not None else checkpoint_pipeline.evaluation.max_decisions
    epsilon = deepcopy(checkpoint_pipeline.epsilon)
    if args.epsilon is not None:
        epsilon.betting.evaluation_epsilon = args.epsilon
        epsilon.playing.evaluation_epsilon = args.epsilon
    if args.betting_epsilon is not None:
        epsilon.betting.evaluation_epsilon = args.betting_epsilon
    if args.playing_epsilon is not None:
        epsilon.playing.evaluation_epsilon = args.playing_epsilon

    metrics = evaluate_policy(
        envs=setup["envs"],
        model=model,
        epsilon=epsilon,
        num_rounds=num_rounds,
        max_decisions=max_decisions,
        rng=random.Random(evaluation_seed),
        reset_hidden_on_round_end=checkpoint_pipeline.trainer.reset_hidden_on_round_end,
        betting_auxiliary_config=checkpoint_pipeline.betting_auxiliary,
        count_auxiliary_config=checkpoint_pipeline.count_auxiliary,
        ev_calibration_config=checkpoint_pipeline.ev_calibration_diagnostics,
    )

    result = {
        "experiment": summarize_setup(setup),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "evaluation": {
            "seed": evaluation_seed,
            "epsilon": {
                "betting": epsilon.betting.evaluation_epsilon,
                "playing": epsilon.playing.evaluation_epsilon,
            },
            "num_rounds": num_rounds,
            "max_decisions": max_decisions,
            "metrics": metrics,
        },
    }

    if args.summary_path:
        saved_path = write_json_file(args.summary_path, result)
        result["evaluation"]["summary_path"] = str(saved_path)

    print(to_pretty_json(result))


if __name__ == "__main__":
    main()
