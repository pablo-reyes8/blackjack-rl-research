from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from blackjack_rl_cli.common import load_experiment_config, resolve_training_setup, summarize_setup
from training import train_model


class BlackjackCliConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.experiments_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"

    def test_all_experiment_presets_resolve_to_valid_setups(self) -> None:
        config_paths = sorted(
            path
            for path in self.experiments_dir.glob("*.yaml")
            if path.name not in {"experiment.template.yaml"}
        )

        self.assertGreaterEqual(len(config_paths), 5)

        for config_path in config_paths:
            experiment = load_experiment_config(config_path)
            setup = resolve_training_setup(experiment)
            summary = summarize_setup(setup)

            self.assertGreaterEqual(summary["run"]["num_envs"], 1)
            self.assertGreater(summary["derived"]["state_dim"], 0)
            self.assertGreater(summary["derived"]["parameter_count"], 0)
            self.assertIn(summary["model"]["architecture"], {"feedforward", "recurrent", "dueling_recurrent"})

    def test_smoke_preset_runs_end_to_end_from_yaml(self) -> None:
        experiment = load_experiment_config(self.experiments_dir / "smoke-test.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            experiment.setdefault("training", {}).setdefault("checkpoints", {})["directory"] = tmp_dir
            setup = resolve_training_setup(experiment)
            result = train_model(setup["envs"], setup["model"], pipeline_config=setup["pipeline_config"])

            self.assertEqual(len(result["history"]), 1)
            self.assertTrue((Path(tmp_dir) / "latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
