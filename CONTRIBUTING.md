# Contributing

## Setup

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

## Development workflow

1. Keep changes focused and reproducible.
2. Add or update tests for behavioral changes.
3. Prefer YAML experiment presets for new training setups.
4. Document user-facing changes in `README.md` when relevant.

## Test command

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Useful local checks

```bash
blackjack-describe --experiment-config configs/experiments/smoke-test.yaml
blackjack-train --experiment-config configs/experiments/smoke-test.yaml
```

## Pull requests

1. Explain the motivation, not only the code diff.
2. Include the exact config or command used for experiments.
3. Mention whether results are smoke tests, short validation runs, or longer research runs.
