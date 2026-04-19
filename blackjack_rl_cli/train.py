from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training import train_model

from blackjack_rl_cli.common import load_experiment_config, resolve_training_setup, summarize_setup, to_pretty_json, write_json_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a blackjack RL experiment from a YAML preset.")
    parser.add_argument("--experiment-config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--output-dir", help="Override training.checkpoints.directory.")
    parser.add_argument("--num-envs", type=int, help="Override run.num_envs.")
    parser.add_argument("--seed", type=int, help="Override training.trainer.seed.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved configuration before training.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the config and exit without training.")
    parser.add_argument("--summary-path", help="Optional JSON file for the resolved run summary.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    experiment = load_experiment_config(args.experiment_config)
    if args.output_dir is not None:
        experiment.setdefault("training", {}).setdefault("checkpoints", {})["directory"] = args.output_dir
    if args.num_envs is not None:
        experiment.setdefault("run", {})["num_envs"] = args.num_envs
    if args.seed is not None:
        experiment.setdefault("training", {}).setdefault("trainer", {})["seed"] = args.seed

    setup = resolve_training_setup(experiment)
    setup_summary = summarize_setup(setup)

    if args.print_config or args.dry_run:
        print(to_pretty_json(setup_summary))
    if args.dry_run:
        return

    result = train_model(setup["envs"], setup["model"], pipeline_config=setup["pipeline_config"])
    summary_path = args.summary_path or str(Path(result["checkpoint_dir"]) / "run_summary.json")

    payload = {
        "experiment": setup_summary,
        "result": {
            "history": result["history"],
            "best_eval_metrics": result["best_eval_metrics"],
            "state": result["state"],
            "checkpoint_dir": result["checkpoint_dir"],
        },
    }
    saved_path = write_json_file(summary_path, payload)
    payload["result"]["summary_path"] = str(saved_path)
    print(to_pretty_json(payload))


if __name__ == "__main__":
    main()
