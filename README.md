<div align="center">

# ♠️ Blackjack RL: Sequential Decision-Making Under Uncertainty

![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-research%20prototype-orange?style=for-the-badge)

<br>

**A pure-PyTorch research environment designed to push Blackjack beyond a toy problem.** *This project models Blackjack as a rigorous Partially Observable Markov Decision Process (POMDP), featuring dynamic bet-sizing, hidden shoe states, recurrent agents, and fully transparent, reproducible experiment pipelines.*

</div>

<br>



Traditional Reinforcement Learning implementations of Blackjack often reduce the game to a single-step, fully observable hand. **This repository breaks that mold.** The codebase natively supports hidden shoe state, reshuffle dynamics, cut-card behavior, variable observation profiles, and explicit betting phases before play.

The stack is intentionally built from the ground up—bypassing high-level abstractions like Gymnasium. Environment dynamics, observation encoding, replay buffers, training loops, evaluation, checkpointing, and agent architectures are developed in pure PyTorch and native Python. The goal: keeping the full decision pipeline transparent, customizable, and research-ready.

---

## 📑 Table of Contents

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

---

## Why this repository stands out

- **Beyond a single hand:** Blackjack is treated as a continuous, sequential decision-making problem under true uncertainty.
- **Integrated Betting Optimization:** Betting is no longer assumed; it is part of the learning objective. Every round begins in a betting phase, requiring the agent to learn *how much* to risk alongside *how* to play.
- **Advanced Table Dynamics:** Supports realistic casino constraints including split depth limits, doubles, surrender, insurance, multi-deck shoes, dealer peek behavior, cut-card reshuffle logic, and optional six-card charlie.
- **Observation as a Research Surface:** Seamlessly switch between compact basic-strategy-style inputs and simulator-level, fully observable states.
- **Sophisticated Agent Architectures:** Out-of-the-box support for feedforward, recurrent, and dueling recurrent Double DQN agents, utilizing separate betting and playing heads over a shared representational trunk.
- **Robust Training Pipeline:** Features dual epsilon scheduling by phase, replay buffers, optional n-step returns, phase-weighted loss, target-network updates, and strict checkpoint management.
- **Production-Ready Hygiene:** Packaged for a clean first public push: `README`, `pyproject.toml`, Docker, GitHub Actions, YAML presets, requirements files, scripts, license, and repo hygiene files.

---

## Current status & Capabilities

The core blackjack environment is fully implemented, bypassing standard abstractions to natively model the game as a two-stage decision process: **Betting** (`1x`, `2x`, `3x`, `4x`) and **Playing** (`stand`, `hit`, `double`, `split`, `surrender`, `insurance`). 

**🃏 Environment & Rule Engine**
* **Shoe Dynamics:** Multi-deck shoes, hidden progress starts, reshuffle tracking, cut-card mode, and realistic reset-to-betting flows.
* **Table Rules:** S17/H17, blackjack payouts, dealer peek, insurance, surrender, standard/DAS splits, and split-aces restrictions.
* **Extensions:** Configurable max split depth, variable bet multipliers, and optional six-card charlie.

**🧠 Agents & Observability**
* **Architectures:** `feedforward` (fast baselines), `recurrent` (handling partial observability), and `dueling_recurrent` (advanced sequence modeling).
* **Observation Profiles:** Ranging from `minimal_basic_strategy` (compact features), to `table_realistic` (partial visibility with or without shoe progress), up to a `fully_observable_sim` (research-only God-view for theoretical bounds).

**⚡ Training Pipeline**
* **Phase-Aware Optimization:** Separate betting/playing heads, module gating, and dual epsilon exploration schedules.
* **Loss & Replay:** Standard Double DQN targets, optional $n$-step returns, phase-weighted TD loss, and temporal replay buffers.
* **Reproducibility:** Driven by YAML presets, strict checkpointing, and dedicated CLI workflows (`describe`, `train`, `evaluate`).

---

## 📂 Repository layout

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

---

## ⚙️ Installation

### Option 1: Standard local setup (Recommended)

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

### Option 2: Minimal runtime dependencies only

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick start

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

---

## 🧪 Experiment presets

The repository ships with a small but useful preset catalog.

| Preset | Main idea |
| :--- | :--- |
| `configs/experiments/smoke-test.yaml` | Fast CI and local sanity check |
| `configs/experiments/feedforward-basic.yaml` | Feedforward baseline on compact observations |
| `configs/experiments/recurrent-table-default.yaml` | GRU-based recurrent training on realistic table observations |
| `configs/experiments/dueling-unknown-progress.yaml` | LSTM dueling recurrent agent under hidden shoe progress |
| `configs/experiments/fully-observable-sim.yaml` | Stronger research preset using simulator-level visibility |
| `configs/experiments/experiment.template.yaml` | Copyable template for custom experiments |

---

## 💻 CLI workflows

### Describe a setup
Use this to inspect the fully resolved environment, model, and training config before spending time on a run.
```bash
blackjack-describe --experiment-config configs/experiments/recurrent-table-default.yaml
```

### Train with a preset
```bash
blackjack-train --experiment-config configs/experiments/feedforward-basic.yaml
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

---

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

---

## 🔬 Ablation CLIs

The repository includes a self-contained ablation suite under `scripts/ablations/`.

All ablations inherit the current phase-aware training stack unless explicitly overridden:
- separate betting and playing heads
- dual epsilon exploration by phase
- phase adapters and module gating enabled by default
- phase-weighted TD loss enabled by default
- optional `n_step` support available in the config, but disabled by default in the shipped ablation suite

**Design goals of this folder:**
- each ablation has its own runnable CLI
- outputs stay inside `scripts/ablations/`
- every ablation writes checkpoints and summaries into its own `ab_*` directory
- the folder is intentionally lightweight and portable

### Ablation matrix

| Ablation | Main setup |
| :---: | :--- |
| **`ab_1`** | Feedforward Double DQN, minimal observation profile, MSE loss, hard targets, phase-aware defaults |
| **`ab_2`** | GRU recurrent Double DQN, realistic partial observation, Huber loss, hard targets, phase-aware defaults |
| **`ab_3`** | LSTM recurrent Double DQN, same realistic partial observation, Huber loss, hard targets |
| **`ab_4`** | Dueling GRU, realistic partial observation, AdamW, dropout, soft targets |
| **`ab_5`** | GRU, unknown shoe progress, MSE loss, longer replay sequences, soft targets |
| **`ab_6`** | Dueling LSTM, fully observable simulator profile, AdamW, softer exploration targets |

### Run one ablation
```bash
python scripts/ablations/ab_1_feedforward_mse.py
```
*(Supports runtime overrides like `--epochs 8 --num-envs 8 --device auto`)*

### Run the full ablation suite
```bash
python scripts/ablations/run_all.py
```

### Compare finished ablations
```bash
python scripts/ablations/compare.py
```
*(By default, ranks by `ev_per_1000_hands`. Supports custom metric targeting via `--metric`)*

---

## Notebook workflow

The repository already includes exploratory notebooks under `notebooks/`.

| Notebook | Typical use |
| :--- | :--- |
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

---

## 🛠️ Configuration system

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
```


---

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

---

## Testing and CI

Run the test suite locally:
```bash
python -m pytest tests -q
```

GitHub Actions is configured in `.github/workflows/ci.yml` and currently installs dependencies, runs the test suite, resolves the smoke preset, and runs a full smoke training job.



---

## Limitations and next steps

## Limitations and next steps

**What is already strong:**
- High-fidelity blackjack environment with broad rule configurability
- Realistic partial-observation setups, including unknown shoe progress
- Flexible encoder with modular feature blocks for hand state, rules, temporal context, observed card history, discard summary, and visible shoe-change signals
- Explicit two-phase decision modeling: bet sizing and hand play
- Recurrent Double/Dueling DQN pipeline with phase-aware exploration
- Phase-aware loss weighting for betting vs. playing
- Optional phase adapters and module gating in the network
- N-step training support
- Resume-from-checkpoint workflow for iterative training
- Support for exogenous visible-card events, including both training heuristics and future manual injection from a visual scraper
- Reproducible preset-driven workflows and checkpointed experimentation
- Expanded monitoring with train/validation behavior gaps and richer betting/playing diagnostics

**What is still missing or intentionally left lightweight:**
- No published benchmark table yet across standard blackjack settings or ablations
- No benchmark against basic strategy, card-counting baselines, or simple betting heuristics yet
- No experiment tracking backend integration (e.g. W&B)
- No hyperparameter sweep orchestration
- No pre-trained model zoo
- No prioritized replay yet
- No explicit calibration layer for betting confidence / risk control
- No published study yet on robustness across different table populations, shoe penetrations, and exogenous-card profiles

---

## License

This project is released under the MIT License. See `LICENSE` for details.

