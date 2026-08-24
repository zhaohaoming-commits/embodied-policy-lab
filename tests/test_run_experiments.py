import unittest
from pathlib import Path

from embodied_policy.run_experiments import aggregate_results, prepare_run_config
from embodied_policy.sweep_replan import aggregate_by_replan_interval


class SeedSweepTest(unittest.TestCase):
    def test_prepared_run_changes_only_training_seed_and_output(self) -> None:
        base = {
            "seed": 7,
            "output_dir": "outputs/base",
            "data": {"split_seed": 7},
            "eval": {"seed": 7},
        }
        run = prepare_run_config(base, "pickcube_state_delta", 17, Path("outputs/sweep"))
        self.assertEqual(run["seed"], 17)
        self.assertEqual(run["data"]["split_seed"], 7)
        self.assertEqual(run["eval"]["seed"], 7)
        self.assertEqual(
            Path(run["output_dir"]), Path("outputs/sweep/pickcube_state_delta_train_seed_17")
        )
        self.assertEqual(base["seed"], 7)

    def test_aggregate_reports_sample_standard_deviation(self) -> None:
        rows = aggregate_results(
            [
                {"config_name": "transformer", "model_type": "chunk", "success_rate": 0.8,
                 "mean_steps": 20.0},
                {"config_name": "transformer", "model_type": "chunk", "success_rate": 1.0,
                 "mean_steps": 30.0},
            ]
        )
        self.assertEqual(rows[0]["runs"], 2)
        self.assertAlmostEqual(rows[0]["success_rate_mean"], 0.9)
        self.assertAlmostEqual(rows[0]["success_rate_std"], 0.141421356, places=7)
        self.assertAlmostEqual(rows[0]["mean_steps_mean"], 25.0)
        self.assertAlmostEqual(rows[0]["mean_steps_std"], 7.071067812, places=7)

    def test_replan_aggregate_groups_by_interval(self) -> None:
        rows = aggregate_by_replan_interval(
            [
                {"replan_interval": 2, "success_rate": 0.7, "mean_steps": 30.0},
                {"replan_interval": 1, "success_rate": 0.9, "mean_steps": 20.0},
                {"replan_interval": 2, "success_rate": 0.9, "mean_steps": 25.0},
            ]
        )
        self.assertEqual([row["replan_interval"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["runs"], 1)
        self.assertAlmostEqual(rows[1]["success_rate_mean"], 0.8)
        self.assertAlmostEqual(rows[1]["success_rate_std"], 0.141421356, places=7)
