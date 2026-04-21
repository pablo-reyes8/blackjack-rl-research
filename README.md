# Blackjack RL

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

Advanced reinforcement learning research environment for blackjack with partial observability, configurable table rules, explicit bet-sizing and play phases, recurrent agents, checkpointed training, and reproducible experiment presets.

This is not a toy "learn one round" blackjack project. The codebase already models hidden shoe state, reshuffle dynamics, cut-card behavior, variable observation profiles, explicit betting before play, replay buffers for recurrent training, and multiple Double DQN variants.

The stack is intentionally built without Gymnasium or other RL framework abstractions. Environment dynamics, observation encoding, replay buffers, training loops, evaluation, checkpointing, and agent implementations are developed in pure PyTorch and native Python to keep the full decision pipeline transparent, customizable, and research-friendly.


## Table of Contents

- [Why this repository stands out](#why-this-repository-stands-out)
- [Current status](#current-status)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Experiment presets](#experiment-presets)
- [CLI workflows](#cli-workflows)
- [Ablation CLIs](#ablation-clis)
- [Python API](#python-api)
- [Notebook workflow](#notebook-workflow)
- [Configuration system](#configuration-system)
- [Checkpoints and outputs](#checkpoints-and-outputs)
- [Docker](#docker)
- [Testing and CI](#testing-and-ci)
- [Engineering practices](#engineering-practices)
- [Limitations and next steps](#limitations-and-next-steps)
- [License](#license)

## Why this repository stands out

- Blackjack is treated as a sequential decision-making problem under uncertainty, not as a single isolated hand.
- The environment supports realistic rules such as split depth limits, doubles, surrender, insurance, multi-deck shoes, dealer peek behavior, cut-card reshuffle logic, and optional six-card charlie.
- Betting is part of the learning problem now: every round starts in a betting phase and the agent must learn both how much to bet and how to play the hand.
- Observation design is part of the research surface: from compact basic-strategy-style inputs to simulator-level fully observable settings.
- The project already includes feedforward, recurrent, and dueling recurrent Double DQN agents with separate betting and playing heads over a shared trunk.
- The training stack includes evaluation, replay buffers, dual epsilon scheduling by phase, optional n-step returns, optional phase-weighted loss, target-network updates, and checkpoint management.
- The repository is now packaged for a clean first public push: `README`, `pyproject.toml`, Docker, GitHub Actions, YAML presets, requirements files, scripts, license, and repo hygiene files.

## Current status

- The core blackjack environment is already seriously implemented, configurable, and covered by unit and smoke tests.
- The environment now models two explicit decision stages per round:

| Stage | What the agent decides |
| --- | --- |
| `betting` | `bet_1x`, `bet_2x`, `bet_3x`, `bet_4x` |
| `playing` | `stand`, `hit`, `double`, `split`, `surrender`, `insurance` |

- Table-rule coverage available today:

| Rule family | Current support |
| --- | --- |
| Shoe / dealing | multi-deck shoes, hidden shoe progress starts, reshuffle tracking, cut-card mode, realistic reset-to-betting flow |
| Core table rules | S17/H17, blackjack payout, dealer peek, insurance, surrender, split rules, DAS, split-aces restrictions |
| Extended rules | optional six-card charlie, configurable max split depth per hand, configurable bet multipliers |

- Encoder profiles are implemented for multiple observability regimes and now encode explicit betting context and decision phase.
- Agent architectures available today:

| Architecture | Intended use |
| --- | --- |
| `feedforward` | Compact state, faster smoke runs, low-memory baselines |
| `recurrent` | Partial observability with temporal credit assignment |
| `dueling_recurrent` | Stronger sequence model for richer or harder table settings |

- Training pipeline status today:

| Capability | Current state |
| --- | --- |
| Replay | feedforward and recurrent replay buffers |
| Exploration | dual epsilon by phase: betting and playing |
| Loss | standard Double DQN targets, optional phase weighting, optional n-step support |
| Network bias | separate betting and playing heads, plus phase adapters and module gating |
| Monitoring | epoch, update, and eval prints with total, betting, and playing metrics |
| Reproducibility | YAML presets, checkpointing, CLI describe/train/evaluate workflows |

- Observation and encoder profiles available today:

| Profile | Description |
| --- | --- |
| `minimal_basic_strategy` | Minimal state close to hand/rule features |
| `table_realistic_default` | Partial observability with realistic visible context |
| `table_realistic_unknown_progress` | Hidden shoe progress and stronger uncertainty |
| `fully_observable_sim` | Research-only simulator view with exact shoe information |



## Repository layout

```text
blackjack-rl/
├── configs/experiments/        # YAML presets for reproducible runs
├── enviroment_bj/              # Blackjack environment, wrapper, text game, rules
├── loss/                       # Bellman target and TD loss implementations
├── model/                      # Encoders and Q-network agents
├── notebooks/                  # Interactive exploration notebooks
├── scripts/                    # Final CLIs for describe/train/evaluate workflows
├── scripts/ablations/          # Self-contained ablation runners and comparison tools
├── tests/                      # Unit and smoke tests
├── training/                   # Replay buffers, evaluation, trainer, checkpoints
├── .github/workflows/ci.yml    # GitHub Actions CI
├── Dockerfile                  # CPU-friendly container image
├── pyproject.toml              # Packaging and console entry points
└── README.md
```

## Installation

### Option 1: standard local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

### Option 2: minimal runtime dependencies only

```bash
pip install -r requirements.txt
```

## Quick start

### 1. Inspect a preset before training

```bash
blackjack-describe --experiment-config configs/experiments/smoke-test.yaml
```

If you prefer direct script execution instead of installed console commands:

```bash
python scripts/blackjack_rl_cli/describe_setup.py --experiment-config configs/experiments/smoke-test.yaml
```

### 2. Run a smoke training job

```bash
blackjack-train --experiment-config configs/experiments/smoke-test.yaml
```

This preset is intentionally small and is used by CI as a fast end-to-end validation run.

Training output is phase-aware now. During a run the trainer prints, in a compact format:

- betting epsilon and playing epsilon
- loss and TD error split by phase
- betting action frequencies and bet EV-style summaries
- playing action frequencies
- total reward and EV summaries for train and evaluation

### 3. Evaluate a checkpoint

```bash
blackjack-evaluate \
  --experiment-config configs/experiments/smoke-test.yaml \
  --checkpoint outputs/smoke-test/latest.pt
```

## Experiment presets

The repository ships with a small but useful preset catalog.

| Preset | Main idea |
| --- | --- |
| `configs/experiments/smoke-test.yaml` | Fast CI and local sanity check |
| `configs/experiments/feedforward-basic.yaml` | Feedforward baseline on compact observations |
| `configs/experiments/recurrent-table-default.yaml` | GRU-based recurrent training on realistic table observations |
| `configs/experiments/dueling-unknown-progress.yaml` | LSTM dueling recurrent agent under hidden shoe progress |
| `configs/experiments/fully-observable-sim.yaml` | Stronger research preset using simulator-level visibility |
| `configs/experiments/experiment.template.yaml` | Copyable template for custom experiments |

## CLI workflows

### Describe a setup

Use this to inspect the fully resolved environment, model, and training config before spending time on a run.

```bash
blackjack-describe --experiment-config configs/experiments/recurrent-table-default.yaml
```

### Train with a preset

```bash
blackjack-train --experiment-config configs/experiments/feedforward-basic.yaml
```

Direct script execution:

```bash
python scripts/blackjack_rl_cli/train.py --experiment-config configs/experiments/feedforward-basic.yaml
```

### Override the number of environments or the output directory

```bash
blackjack-train \
  --experiment-config configs/experiments/recurrent-table-default.yaml \
  --num-envs 8 \
  --output-dir outputs/recurrent-table-default-x8
```

### Dry-run a config without training

```bash
blackjack-train \
  --experiment-config configs/experiments/dueling-unknown-progress.yaml \
  --print-config \
  --dry-run
```

### Evaluate a saved checkpoint with custom evaluation settings

```bash
blackjack-evaluate \
  --experiment-config configs/experiments/fully-observable-sim.yaml \
  --checkpoint outputs/fully-observable-sim/latest.pt \
  --num-rounds 500 \
  --max-decisions 20000 \
  --device auto
```

Phase-specific evaluation override example:

```bash
blackjack-evaluate \
  --experiment-config configs/experiments/recurrent-table-default.yaml \
  --checkpoint outputs/recurrent-table-default/latest.pt \
  --betting-epsilon 0.05 \
  --playing-epsilon 0.00
```


## Python API

You can use the project directly from Python without going through the CLIs.

```python
from enviroment_bj import BlackjackConfig, BlackjackEnvironment, ObservationConfig, StartStateConfig
from model.agents import RecurrentDoubleDQN
from training import TrainingPipelineConfig, train_model

observation = ObservationConfig.for_profile("table_realistic_default")
environment = BlackjackEnvironment(
    config=BlackjackConfig(
        n_decks=6,
        shoe_penetration=0.8,
        observation=observation,
    ),
    seed=7,
    start_state=StartStateConfig(mode="fresh_shoe"),
)

model = RecurrentDoubleDQN.from_profile("table_realistic_default", recurrent_type="gru")
pipeline_config = TrainingPipelineConfig()

result = train_model(environment, model, pipeline_config=pipeline_config)
print(result["checkpoint_dir"])
```

### Example: environment-only interaction

```python
from enviroment_bj import BlackjackJSONWrapper

game = BlackjackJSONWrapper(seed=7)
response = game.reset()

while not response["done"]:
    action = response["legal_actions"][0]
    response = game.step({"action": action})
```

## Ablation CLIs

The repository includes a self-contained ablation suite under `scripts/ablations/`.

All ablations now inherit the current phase-aware training stack unless they explicitly override it:

- separate betting and playing heads
- dual epsilon exploration by phase
- phase adapters and module gating enabled by default
- phase-weighted TD loss enabled by default
- optional `n_step` support available in the config, but disabled by default in the shipped ablation suite

Design goals of this folder:

- each ablation has its own runnable CLI
- outputs stay inside `scripts/ablations/`
- every ablation writes checkpoints and summaries into its own `ab_*` directory
- the folder is intentionally lightweight and portable, so moving it closer to the project root should require minimal edits

When you run an ablation script, it creates its output directory on demand:

- `scripts/ablations/ab_1/`
- `scripts/ablations/ab_2/`
- `scripts/ablations/ab_3/`
- `scripts/ablations/ab_4/`
- `scripts/ablations/ab_5/`
- `scripts/ablations/ab_6/`

Each directory stores the checkpoints plus a `run_summary.json` file that can later be consumed by the comparison CLI.

### Ablation matrix

| Ablation | Script | Main setup |
| --- | --- | --- |
| `ab_1` | `python scripts/ablations/ab_1_feedforward_mse.py` | Feedforward Double DQN, minimal observation profile, MSE loss, hard targets, current phase-aware defaults |
| `ab_2` | `python scripts/ablations/ab_2_gru_huber.py` | GRU recurrent Double DQN, realistic partial observation, Huber loss, hard targets, current phase-aware defaults |
| `ab_3` | `python scripts/ablations/ab_3_lstm_huber.py` | LSTM recurrent Double DQN, same realistic partial observation, Huber loss, hard targets |
| `ab_4` | `python scripts/ablations/ab_4_dueling_gru_soft.py` | Dueling GRU, realistic partial observation, AdamW, dropout, soft targets |
| `ab_5` | `python scripts/ablations/ab_5_unknown_progress_mse.py` | GRU, unknown shoe progress, MSE loss, longer replay sequences, soft targets |
| `ab_6` | `python scripts/ablations/ab_6_fully_observable_dueling.py` | Dueling LSTM, fully observable simulator profile, AdamW, softer exploration targets |

### What each ablation is testing

| Ablation | Main question |
| --- | --- |
| `ab_1` | How much performance survives when both recurrence and temporal table context are removed while keeping the betting-plus-play pipeline intact? |
| `ab_2` | What does a strong recurrent baseline look like under realistic table observations with the current phase-aware stack? |
| `ab_3` | GRU vs LSTM under the same realistic partially observable regime and same betting/play pipeline |
| `ab_4` | Does dueling decomposition plus soft targets improve stability under realistic table play? |
| `ab_5` | How robust is the agent when shoe progress is hidden and training uses a harsher MSE objective? |
| `ab_6` | What happens when a stronger model receives near upper-bound simulator visibility and slightly more aggressive phase-specific exploration? |

### Run one ablation

```bash
python scripts/ablations/ab_1_feedforward_mse.py
python scripts/ablations/ab_2_gru_huber.py
python scripts/ablations/ab_3_lstm_huber.py
python scripts/ablations/ab_4_dueling_gru_soft.py
python scripts/ablations/ab_5_unknown_progress_mse.py
python scripts/ablations/ab_6_fully_observable_dueling.py
```

Useful runtime overrides supported by every ablation CLI:

```bash
python scripts/ablations/ab_2_gru_huber.py --epochs 8 --env-steps-per-epoch 1024 --num-envs 8 --device auto
python scripts/ablations/ab_5_unknown_progress_mse.py --quiet
python scripts/ablations/ab_3_lstm_huber.py --dry-run
```

### Run the full ablation suite

```bash
python scripts/ablations/run_all.py
```

Optional examples:

```bash
python scripts/ablations/run_all.py --epochs 8 --env-steps-per-epoch 1024
python scripts/ablations/run_all.py --only ab_2 ab_3 ab_4
python scripts/ablations/run_all.py --seed-offset 1000 --continue-on-error
```

`run_all.py` also refreshes the comparison artifact at the end of the run.

### Compare finished ablations

```bash
python scripts/ablations/compare.py
```

By default, the comparison ranks ablations by `ev_per_1000_hands` using the best available evaluation metrics from each `run_summary.json`.

You can also compare using another metric. Dot-path access is supported for nested metrics:

```bash
python scripts/ablations/compare.py --metric reward_per_round
python scripts/ablations/compare.py --metric loss_rate --lower-is-better
python scripts/ablations/compare.py --metric bet_action_frequencies.bet_2x
```

The comparison summary is written to:

```text
scripts/ablations/ablation_comparison.json
```

## Notebook workflow

The repository already includes exploratory notebooks under `notebooks/`.

| Notebook | Typical use |
| --- | --- |
| `try_blackjack.ipynb` | Inspect environment behavior |
| `try_encoder.ipynb` | Inspect observation encoding and state vectors |
| `try_agents.ipynb` | Test model outputs and architecture behavior |
| `try_training.ipynb` | Experiment with the training loop interactively |
| `pipeline_settings.ipynb` | Explore pipeline settings and variants |

Recommended notebook setup:

```bash
pip install -r requirements-dev.txt
pip install -e .
jupyter lab
```

## Configuration system

Experiment presets live in `configs/experiments/` and are YAML-based.

Each experiment can define:

- `metadata`: human-readable name and description
- `run`: script-level settings such as `num_envs`
- `start_state`: how episodes start, including hidden burned rounds
- `environment`: full blackjack rules and observation profile
- `model`: agent architecture and network hyperparameters
- `training`: replay buffer, optimization, dual epsilon schedule, evaluation, checkpointing, and print settings

The config loader supports inheritance through `extends`, so presets can share a common base while overriding only what changes.

Example excerpt:

```yaml
extends: base.yaml

metadata:
  name: recurrent-table-default

run:
  num_envs: 4

environment:
  observation:
    profile: table_realistic_default

model:
  architecture: recurrent
  encoder_profile: table_realistic_default
  recurrent_type: gru

training:
  epsilon:
    betting:
      start: 1.0
      end: 0.10
      decay_steps: 40000
    playing:
      start: 1.0
      end: 0.03
      decay_steps: 25000
```

Important configuration notes:

- `training.epsilon` supports both a legacy single schedule and a dual `betting` / `playing` schedule.
- `training.n_step` is supported but remains optional.
- phase-weighted TD loss is supported and enabled by default.
- agent phase adapters and module gating are supported and enabled by default.

## Checkpoints and outputs

Training writes checkpoints to the directory defined in `training.checkpoints.directory`.

Typical files:

- `latest.pt`: most recent checkpoint
- `best_eval.pt`: best checkpoint according to the configured evaluation metric
- `step_XXXXXXXX.pt`: periodic snapshots when enabled
- `run_summary.json`: JSON summary produced by the training CLI

Checkpoint payloads include:

- model weights
- optimizer state
- scheduler state when present
- serialized model config
- serialized pipeline config
- trainer state and metrics

## Docker

Build the image:

```bash
docker build -t blackjack-rl .
```

Run the default smoke training job:

```bash
docker run --rm blackjack-rl
```

Run a different preset by overriding the command:

```bash
docker run --rm blackjack-rl \
  blackjack-train --experiment-config configs/experiments/feedforward-basic.yaml
```

## Testing and CI

Run the test suite locally:

```bash
python -m pytest tests -q
```

GitHub Actions is configured in `.github/workflows/ci.yml` and currently does the following:

- installs dependencies
- installs the package in editable mode
- runs the full test suite
- resolves the smoke preset with the describe CLI
- runs a smoke training job through the final training CLI

Dependabot is also configured for both Python dependencies and GitHub Actions updates.

## Engineering practices

This repository now includes the baseline pieces expected from a serious public ML repo:

- `pyproject.toml` for packaging and console entry points
- `requirements.txt` and `requirements-dev.txt`
- Docker support
- GitHub Actions CI
- YAML experiment presets with inheritance
- MIT license
- `.gitignore`, `.dockerignore`, and `.editorconfig`
- contributor guidance in `CONTRIBUTING.md`

## Limitations and next steps

What is already strong:

- environment fidelity and rule configurability
- encoder flexibility
- explicit bet-sizing plus hand-play modeling
- recurrent RL training pipeline with phase-aware exploration and monitoring
- reproducible preset-driven workflows

What is still missing or intentionally left lightweight:

- no published benchmark table yet
- no experiment tracking backend integration
- no hyperparameter sweep orchestration
- no pre-trained model zoo
- no prioritized replay yet
- no distributed or multi-node training support

If this becomes a public research repo with active iteration, the next natural additions would be benchmark reports, release notes, experiment tracking, prioritized replay, and curated result cards.

## License

This project is released under the MIT License. See `LICENSE` for details.
