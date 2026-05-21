import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_3_artifacts.py"


def load_v0_3_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_3", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV03Tests(unittest.TestCase):
    def test_fixed_methods_exclude_optional_eq8(self):
        module = load_v0_3_module()

        methods = [method for method, _config in module.fixed_method_configs(20)]

        self.assertEqual(
            [
                "no-coordination",
                "validated-selective-conflict",
                "arac-lite-v0",
                "arac-lite-v0.1-disable-fast",
            ],
            methods,
        )
        self.assertNotIn("eq8-correct", methods)

    def test_robustness_rows_pair_candidate_against_baselines(self):
        module = load_v0_3_module()
        run_rows = []
        for seed, candidate, v0 in [(1, 9.0, 10.0), (2, 11.0, 10.0), (3, 10.0, 10.0)]:
            run_rows.extend(
                [
                    {"problem": "E4", "tfes": 5000, "seed": seed, "method": "arac-lite-v0.1-disable-fast", "best_error": candidate},
                    {"problem": "E4", "tfes": 5000, "seed": seed, "method": "arac-lite-v0", "best_error": v0},
                    {"problem": "E4", "tfes": 5000, "seed": seed, "method": "validated-selective-conflict", "best_error": v0},
                    {"problem": "E4", "tfes": 5000, "seed": seed, "method": "no-coordination", "best_error": v0},
                ]
            )

        rows = module.build_robustness_rows(run_rows)
        row = next(item for item in rows if item["baseline"] == "arac-lite-v0" and item["tfes"] == 5000)

        self.assertEqual(3, row["paired_runs"])
        self.assertEqual(1, row["wins"])
        self.assertEqual(1, row["losses"])
        self.assertEqual(1, row["ties"])
        self.assertAlmostEqual(2.0 / 3.0, row["non_worse_rate"])

    def test_phase_bucket_and_pass_dynamics(self):
        module = load_v0_3_module()
        rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 0,
                "var_id": 2,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": -1.0,
                "best_improvement_after_action": False,
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 2,
                "var_id": 2,
                "action_candidate": "Disable",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": -0.5,
                "best_improvement_after_action": False,
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 3,
                "var_id": 2,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": True,
                "validation_delta": 2.0,
                "best_improvement_after_action": True,
            },
        ]

        dynamics = module.build_pass_action_dynamics(rows)
        by_key = {(row["pass_id"], row["action"]): row for row in dynamics}

        self.assertEqual("early", by_key[(0, "Fusion")]["phase"])
        self.assertEqual("middle", by_key[(2, "Disable")]["phase"])
        self.assertEqual("late", by_key[(3, "Fusion")]["phase"])
        self.assertEqual(1, by_key[(2, "Disable")]["disable_false_negative_count"])
        self.assertAlmostEqual(1.0, by_key[(3, "Fusion")]["fusion_accept_rate"])

    def test_phase_signal_lines_weight_accepts_across_tfes(self):
        module = load_v0_3_module()
        rows = [
            {
                "problem": "S6",
                "tfes": 5000,
                "method": "arac-lite-v0.1-disable-fast",
                "phase": "early",
                "action": "Fusion",
                "action_count": 2,
                "validation_attempt_count": 2,
                "validation_accept_count": 1,
                "action_delta_mean": -2.0,
            },
            {
                "problem": "S6",
                "tfes": 20000,
                "method": "arac-lite-v0.1-disable-fast",
                "phase": "early",
                "action": "Fusion",
                "action_count": 8,
                "validation_attempt_count": 8,
                "validation_accept_count": 1,
                "action_delta_mean": -4.0,
            },
            {
                "problem": "S6",
                "tfes": 20000,
                "method": "arac-lite-v0.1-disable-fast",
                "phase": "late",
                "action": "Fusion",
                "action_count": 2,
                "validation_attempt_count": 2,
                "validation_accept_count": 1,
                "action_delta_mean": -1.0,
            },
        ]

        lines = module.phase_signal_lines(rows)
        s6_line = next(line for line in lines if line.startswith("- S6:"))

        self.assertIn("early Fusion accept=2.000000e-01", s6_line)
        self.assertIn("(n=10)", s6_line)
        self.assertIn("late=5.000000e-01", s6_line)


if __name__ == "__main__":
    unittest.main()
