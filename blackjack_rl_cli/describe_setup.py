from __future__ import annotations

import argparse

from blackjack_rl_cli.common import load_experiment_config, resolve_training_setup, summarize_setup, to_pretty_json, write_json_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a resolved summary of a blackjack RL experiment config.")
    parser.add_argument("--experiment-config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--summary-path", help="Optional JSON file for the resolved setup summary.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup = resolve_training_setup(load_experiment_config(args.experiment_config))
    summary = summarize_setup(setup)
    if args.summary_path:
        saved_path = write_json_file(args.summary_path, summary)
        summary["saved_path"] = str(saved_path)
    print(to_pretty_json(summary))


if __name__ == "__main__":
    main()
