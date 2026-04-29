from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.final_wrapper import evaluate_blackjack_checkpoint, format_inference_result_json


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run greedy inference for a trained blackjack checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file (.pt).")
    parser.add_argument("--stage-name", help="Optional friendly name for this evaluation run.")
    parser.add_argument("--output-root", default="notebooks/inference", help="Directory anchor for generated summaries.")
    parser.add_argument("--bet-multipliers", type=_parse_int_tuple, help="Comma-separated bet multipliers, e.g. 1,2,3,4")
    parser.add_argument("--penetrations", type=_parse_float_tuple, help="Comma-separated shoe penetrations, e.g. 0.75,0.85")
    parser.add_argument("--start-mode", default="unknown_progress", help="Evaluation start-state mode.")
    parser.add_argument("--min-burned-rounds", type=int, default=10, help="Minimum burned rounds when using hidden-progress starts.")
    parser.add_argument("--max-burned-rounds", type=int, default=60, help="Maximum burned rounds when using hidden-progress starts.")
    parser.add_argument("--eval-rounds", type=int, default=50_000, help="Number of rounds to evaluate.")
    parser.add_argument("--max-decisions", type=int, default=600_000, help="Maximum number of decisions during evaluation.")
    parser.add_argument("--progress-every-rounds", type=int, help="Print intermediate evaluation stats every N completed rounds.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for evaluation.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"], help="Device used for model evaluation.")
    parser.add_argument("--architecture", choices=["feedforward", "recurrent", "dueling_recurrent"], help="Explicit model architecture override.")
    parser.add_argument("--feedforward-hidden-dims", type=_parse_int_tuple, help="Comma-separated feedforward hidden dims, e.g. 256,256,128")
    parser.add_argument("--use-layer-norm", action="store_true", help="Enable layer norm when reconstructing the model.")
    parser.add_argument("--use-phase-adapters", action="store_true", help="Enable phase adapters when reconstructing the model.")
    parser.add_argument("--use-module-gating", action="store_true", help="Enable module gating when reconstructing the model.")
    parser.add_argument("--summary-path", help="Optional JSON file path for the inference summary.")
    parser.add_argument("--quiet", action="store_true", help="Disable human-readable evaluation prints.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_blackjack_checkpoint(
        checkpoint_path=args.checkpoint,
        stage_name=args.stage_name,
        output_root=args.output_root,
        bet_multipliers=args.bet_multipliers,
        penetrations=args.penetrations,
        start_mode=args.start_mode,
        min_burned_rounds=args.min_burned_rounds,
        max_burned_rounds=args.max_burned_rounds,
        base_seed=args.seed,
        eval_rounds=args.eval_rounds,
        eval_max_decisions=args.max_decisions,
        device=args.device,
        architecture=args.architecture,
        feedforward_hidden_dims=args.feedforward_hidden_dims,
        use_layer_norm=(True if args.use_layer_norm else None),
        use_phase_adapters=(True if args.use_phase_adapters else None),
        use_module_gating=(True if args.use_module_gating else None),
        progress_every_n_rounds=args.progress_every_rounds,
        summary_path=args.summary_path,
        print_summary=not args.quiet,
    )
    print(format_inference_result_json(result))


if __name__ == "__main__":
    main()
