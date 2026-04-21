from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


def _find_project_root(start: Path) -> Path:
    required = ("enviroment_bj", "loss", "model", "training")
    for candidate in (start, *start.parents):
        if all((candidate / name).exists() for name in required):
            return candidate
    raise RuntimeError("Could not locate the project root from the ablation scripts directory")


ABLATIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(ABLATIONS_DIR)

if str(ABLATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATIONS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation_specs import ABLATION_SPECS, get_ablation_spec  # noqa: E402
from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig  # noqa: E402
from loss import BellmanLossConfig  # noqa: E402
from model.agents import AgentNetworkConfig, DuelingRecurrentDoubleDQN, FeedForwardDoubleDQN, RecurrentDoubleDQN  # noqa: E402
from training import (  # noqa: E402
    CheckpointConfig,
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
    train_model,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)
        handle.write("\n")
    return path


def _deepcopy_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(spec)


def _build_observation(profile: str, overrides: dict[str, Any] | None = None) -> ObservationConfig:
    observation = ObservationConfig.for_profile(profile)
    for key, value in (overrides or {}).items():
        setattr(observation, key, value)
    observation.__post_init__()
    return observation


def _build_environment_config(spec: dict[str, Any]) -> BlackjackConfig:
    environment = deepcopy(spec["environment"])
    observation_profile = environment.pop("observation_profile")
    observation_overrides = environment.pop("observation_overrides", {})
    return BlackjackConfig(
        observation=_build_observation(observation_profile, observation_overrides),
        **environment,
    )


def _build_start_state(spec: dict[str, Any]) -> StartStateConfig:
    return StartStateConfig(**deepcopy(spec["start_state"]))


def _build_model(spec: dict[str, Any]) -> Any:
    model_data = deepcopy(spec["model"])
    architecture = model_data.pop("architecture")
    encoder_profile = model_data.pop("encoder_profile")
    config = AgentNetworkConfig.for_architecture(architecture, encoder_profile=encoder_profile, **model_data)

    if architecture == "feedforward":
        return FeedForwardDoubleDQN(config=config)
    if architecture == "recurrent":
        return RecurrentDoubleDQN(config=config)
    if architecture == "dueling_recurrent":
        return DuelingRecurrentDoubleDQN(config=config)
    raise ValueError(f"Unsupported architecture: {architecture}")


def _build_environments(spec: dict[str, Any]) -> list[BlackjackEnvironment]:
    environment_config = _build_environment_config(spec)
    start_state = _build_start_state(spec)
    seed = int(spec["seed"])
    num_envs = int(spec["num_envs"])
    return [
        BlackjackEnvironment(
            config=deepcopy(environment_config),
            seed=seed + index,
            start_state=deepcopy(start_state),
        )
        for index in range(num_envs)
    ]


def _build_pipeline_config(spec: dict[str, Any], output_dir: Path) -> TrainingPipelineConfig:
    training = deepcopy(spec["training"])
    trainer_data = training["trainer"]
    replay_buffer_data = training["replay_buffer"]
    epsilon_data = training["epsilon"]
    optimization_data = training["optimization"]
    target_update_data = training["target_update"]
    evaluation_data = training["evaluation"]
    checkpoint_data = training["checkpoints"]
    print_data = training["prints"]
    loss_data = trainer_data.pop("loss")
    phase_weights_data = loss_data.pop("phase_weights", None)
    if phase_weights_data is not None:
        from loss import LossPhaseWeightConfig

        loss_data["phase_weights"] = LossPhaseWeightConfig(**phase_weights_data)

    checkpoint_data["directory"] = str(output_dir)

    if not epsilon_data:
        epsilon = DualEpsilonConfig()
    elif "betting" in epsilon_data or "playing" in epsilon_data:
        epsilon = DualEpsilonConfig(
            betting=EpsilonScheduleConfig(**epsilon_data.get("betting", {})),
            playing=EpsilonScheduleConfig(**epsilon_data.get("playing", {})),
        )
    else:
        epsilon = EpsilonScheduleConfig(**epsilon_data)

    return TrainingPipelineConfig(
        trainer=TrainerConfig(loss=BellmanLossConfig(**loss_data), **trainer_data),
        replay_buffer=ReplayBufferConfig(**replay_buffer_data),
        epsilon=epsilon,
        n_step=NStepConfig(**training.get("n_step", {})),
        optimization=OptimizationConfig(**optimization_data),
        target_update=TargetUpdateConfig(**target_update_data),
        evaluation=EvaluationConfig(**evaluation_data),
        checkpoints=CheckpointConfig(**checkpoint_data),
        prints=PrintConfig(**print_data),
    )


def _resolve_runtime_spec(spec: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    resolved = _deepcopy_spec(spec)
    resolved["training"]["trainer"]["seed"] = resolved["seed"]

    if getattr(args, "epochs", None) is not None:
        resolved["training"]["trainer"]["total_epochs"] = args.epochs
    if getattr(args, "env_steps_per_epoch", None) is not None:
        resolved["training"]["trainer"]["env_steps_per_epoch"] = args.env_steps_per_epoch
    if getattr(args, "num_envs", None) is not None:
        resolved["num_envs"] = args.num_envs
    if getattr(args, "seed", None) is not None:
        resolved["seed"] = args.seed
        resolved["training"]["trainer"]["seed"] = args.seed
    if getattr(args, "device", None) is not None:
        resolved["training"]["trainer"]["device"] = args.device
    if getattr(args, "eval_rounds", None) is not None:
        resolved["training"]["evaluation"]["num_rounds"] = args.eval_rounds
    if getattr(args, "quiet", False):
        resolved["training"]["prints"]["enable"] = False
        resolved["training"]["prints"]["print_run_summary"] = False
        resolved["training"]["prints"]["print_epoch_header"] = False
        resolved["training"]["prints"]["print_epoch_summary"] = False
        resolved["training"]["prints"]["print_eval_summary"] = False

    output_dir = ABLATIONS_DIR / resolved["id"]
    resolved["output_dir"] = str(output_dir)
    return resolved, output_dir


def build_single_ablation_parser(spec: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {spec['id']}: {spec['title']}")
    parser.add_argument("--epochs", type=int, help="Override trainer.total_epochs.")
    parser.add_argument("--env-steps-per-epoch", type=int, help="Override trainer.env_steps_per_epoch.")
    parser.add_argument("--num-envs", type=int, help="Override the number of parallel environments.")
    parser.add_argument("--seed", type=int, help="Override the base seed for the ablation.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override trainer.device.")
    parser.add_argument("--eval-rounds", type=int, help="Override evaluation.num_rounds.")
    parser.add_argument("--quiet", action="store_true", help="Disable trainer printouts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved ablation setup without training.")
    return parser


def build_run_all_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run every ablation experiment in scripts/ablations.")
    parser.add_argument("--epochs", type=int, help="Override trainer.total_epochs for every ablation.")
    parser.add_argument("--env-steps-per-epoch", type=int, help="Override trainer.env_steps_per_epoch for every ablation.")
    parser.add_argument("--num-envs", type=int, help="Override the number of parallel environments for every ablation.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override trainer.device for every ablation.")
    parser.add_argument("--eval-rounds", type=int, help="Override evaluation.num_rounds for every ablation.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Add a constant offset to every ablation seed.")
    parser.add_argument("--only", nargs="*", choices=[spec["id"] for spec in ABLATION_SPECS], help="Run only a subset of ablations.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later ablations if one fails.")
    parser.add_argument("--quiet", action="store_true", help="Disable trainer printouts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved setups without training.")
    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare finished ablation runs from scripts/ablations/ab_*/run_summary.json.")
    parser.add_argument("--metric", default="ev_per_1000_hands", help="Metric key used to rank ablations.")
    parser.add_argument("--lower-is-better", action="store_true", help="Sort ascending instead of descending.")
    parser.add_argument("--summary-name", default="run_summary.json", help="Summary file name expected inside each ablation directory.")
    return parser


def build_ablation_payload(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    environment_config = _build_environment_config(spec)
    start_state = _build_start_state(spec)
    model = _build_model(spec)
    pipeline_config = _build_pipeline_config(spec, output_dir)
    return {
        "id": spec["id"],
        "slug": spec["slug"],
        "title": spec["title"],
        "description": spec["description"],
        "changes": spec["changes"],
        "num_envs": spec["num_envs"],
        "seed": spec["seed"],
        "entrypoint": spec["entrypoint"],
        "output_dir": str(output_dir),
        "environment": environment_config,
        "start_state": start_state,
        "model": model.config,
        "training": pipeline_config,
        "derived": {
            "state_dim": model.state_dim,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    }


def run_ablation(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved, output_dir = _resolve_runtime_spec(spec, args)
    payload = build_ablation_payload(resolved, output_dir)

    if getattr(args, "dry_run", False):
        print(json.dumps(_to_jsonable(payload), indent=2))
        return {"ablation": payload, "result": None}

    output_dir.mkdir(parents=True, exist_ok=True)
    envs = _build_environments(resolved)
    model = _build_model(resolved)
    pipeline_config = _build_pipeline_config(resolved, output_dir)
    result = train_model(envs, model, pipeline_config=pipeline_config)

    final_epoch_metrics = result["history"][-1] if result["history"] else None
    summary = {
        "ablation": payload,
        "result": {
            "history": result["history"],
            "best_eval_metrics": result["best_eval_metrics"],
            "final_epoch_metrics": final_epoch_metrics,
            "state": result["state"],
            "checkpoint_dir": str(output_dir),
            "summary_path": str(output_dir / "run_summary.json"),
            "completed_at": _timestamp(),
        },
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(_to_jsonable(summary), indent=2))
    return summary


def run_single_ablation_cli(ablation_id: str) -> None:
    spec = get_ablation_spec(ablation_id)
    args = build_single_ablation_parser(spec).parse_args()
    run_ablation(spec, args)


def _extract_comparison_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    best_eval = summary.get("result", {}).get("best_eval_metrics")
    if best_eval:
        return best_eval
    final_epoch = summary.get("result", {}).get("final_epoch_metrics") or {}
    if isinstance(final_epoch.get("eval"), dict):
        return final_epoch["eval"]
    return final_epoch


def collect_comparison_payload(metric: str, *, lower_is_better: bool = False, summary_name: str = "run_summary.json") -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for spec in ABLATION_SPECS:
        summary_path = ABLATIONS_DIR / spec["id"] / summary_name
        if not summary_path.exists():
            missing.append({"id": spec["id"], "title": spec["title"], "expected_summary": str(summary_path)})
            continue

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        metrics = _extract_comparison_metrics(summary)
        records.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "metric": metric,
                "metric_value": metrics.get(metric),
                "best_eval_metrics": metrics,
                "checkpoint_dir": summary.get("result", {}).get("checkpoint_dir"),
                "summary_path": str(summary_path),
            }
        )

    available = [record for record in records if record["metric_value"] is not None]
    available.sort(key=lambda item: item["metric_value"], reverse=not lower_is_better)

    return {
        "created_at": _timestamp(),
        "metric": metric,
        "lower_is_better": lower_is_better,
        "winner": available[0] if available else None,
        "ranked_results": available,
        "results_with_missing_metric": [record for record in records if record["metric_value"] is None],
        "missing_runs": missing,
    }


def print_comparison_payload(payload: dict[str, Any]) -> None:
    print(f"Comparison metric: {payload['metric']}")
    if payload["winner"] is not None:
        print(f"Best ablation: {payload['winner']['id']} ({payload['winner']['title']}) -> {payload['winner']['metric_value']}")
    else:
        print("Best ablation: none available")

    for index, record in enumerate(payload["ranked_results"], start=1):
        print(f"{index}. {record['id']} | {record['metric_value']} | {record['title']}")

    if payload["missing_runs"]:
        print("Missing runs:")
        for missing in payload["missing_runs"]:
            print(f"- {missing['id']} ({missing['expected_summary']})")


def run_all_ablation_cli() -> None:
    args = build_run_all_parser().parse_args()
    selected_ids = set(args.only) if args.only else {spec["id"] for spec in ABLATION_SPECS}
    run_payloads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for base_spec in ABLATION_SPECS:
        if base_spec["id"] not in selected_ids:
            continue

        spec = _deepcopy_spec(base_spec)
        if args.seed_offset:
            spec["seed"] = int(spec["seed"]) + int(args.seed_offset)

        try:
            run_payloads.append(run_ablation(spec, args))
        except Exception as exc:
            failures.append({"id": spec["id"], "error": str(exc)})
            if not args.continue_on_error:
                raise

    comparison = collect_comparison_payload("ev_per_1000_hands")
    _write_json(
        ABLATIONS_DIR / "run_all_summary.json",
        {
            "created_at": _timestamp(),
            "selected_ids": sorted(selected_ids),
            "dry_run": bool(args.dry_run),
            "completed_runs": run_payloads,
            "failures": failures,
            "comparison": comparison,
        },
    )
    _write_json(ABLATIONS_DIR / "ablation_comparison.json", comparison)
    print_comparison_payload(comparison)


def run_compare_cli() -> None:
    args = build_compare_parser().parse_args()
    payload = collect_comparison_payload(
        args.metric,
        lower_is_better=args.lower_is_better,
        summary_name=args.summary_name,
    )
    _write_json(ABLATIONS_DIR / "ablation_comparison.json", payload)
    print_comparison_payload(payload)
