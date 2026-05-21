import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_2_artifacts.py"


def load_v0_2_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV02Tests(unittest.TestCase):
    def test_fixed_methods_freeze_disable_fast_without_sweep_variants(self):
        module = load_v0_2_module()

        methods = [method for method, _config in module.fixed_method_configs(20)]

        self.assertEqual(
            [
                "no-coordination",
                "eq8-correct",
                "validated-selective-conflict",
                "arac-lite-v0",
                "arac-lite-v0.1-disable-fast",
            ],
            methods,
        )
        self.assertNotIn("arac-lite-v0.1-balanced", methods)
        self.assertNotIn("arac-lite-v0.1-fusion-strict", methods)

        variant = module.disable_fast_variant()
        self.assertEqual(1, variant.overrides["arac_lite_history_min_attempts"])
        self.assertEqual(1, variant.overrides["arac_lite_disable_reject_streak"])
        self.assertEqual(0.0, variant.overrides["arac_lite_disable_accept_rate_threshold"])
        self.assertEqual(0.0, variant.overrides["arac_lite_disable_mean_delta_threshold"])

    def test_summary_keeps_eq8_validated_and_v0_gaps(self):
        module = load_v0_2_module()
        rows = [
            {"problem": "E4", "tfes": 10000, "method": "no-coordination", "best_error": 100.0, "final_error": 100.0, "fusion_count": 0, "freeze_count": 0, "disable_count": 0},
            {"problem": "E4", "tfes": 10000, "method": "eq8-correct", "best_error": 120.0, "final_error": 120.0, "fusion_count": 3, "freeze_count": 0, "disable_count": 0},
            {"problem": "E4", "tfes": 10000, "method": "validated-selective-conflict", "best_error": 110.0, "final_error": 110.0, "fusion_count": 2, "freeze_count": 0, "disable_count": 0},
            {"problem": "E4", "tfes": 10000, "method": "arac-lite-v0", "best_error": 105.0, "final_error": 105.0, "fusion_count": 1, "freeze_count": 2, "disable_count": 3},
            {"problem": "E4", "tfes": 10000, "method": "arac-lite-v0.1-disable-fast", "best_error": 103.0, "final_error": 103.0, "fusion_count": 1, "freeze_count": 1, "disable_count": 4},
        ]

        summary = module.summarize_runs(rows)
        candidate = next(row for row in summary if row["method"] == "arac-lite-v0.1-disable-fast")

        self.assertAlmostEqual(0.03, candidate["gap_vs_no_coordination"])
        self.assertAlmostEqual(-17.0 / 120.0, candidate["gap_vs_eq8_correct"])
        self.assertAlmostEqual(-7.0 / 110.0, candidate["gap_vs_validated"])
        self.assertAlmostEqual(-2.0 / 105.0, candidate["gap_vs_arac_lite_v0"])

    def test_generalization_judgment_requires_heldout_fusion(self):
        module = load_v0_2_module()
        summary = [
            {"problem": "E4", "method": "arac-lite-v0.1-disable-fast", "gap_vs_arac_lite_v0": -0.01, "gap_vs_validated": 0.01, "gap_vs_no_coordination": 0.02},
            {"problem": "S4", "method": "arac-lite-v0.1-disable-fast", "gap_vs_arac_lite_v0": -0.02, "gap_vs_validated": 0.02, "gap_vs_no_coordination": 0.03},
            {"problem": "S6", "method": "arac-lite-v0.1-disable-fast", "gap_vs_arac_lite_v0": 0.01, "gap_vs_validated": 0.03, "gap_vs_no_coordination": 0.04},
        ]
        action_rows = [
            {"problem": "E4", "method": "arac-lite-v0.1-disable-fast", "action_candidate": "Fusion", "action_count": 2},
            {"problem": "S4", "method": "arac-lite-v0.1-disable-fast", "action_candidate": "Disable", "action_count": 3, "disable_false_negative_rate": 0.0},
        ]

        judgment = module.generalization_judgment(summary, action_rows)

        self.assertTrue(judgment["stable"])
        self.assertEqual(2, judgment["held_v0_wins"])
        self.assertEqual(2, judgment["held_fusion_count"])


if __name__ == "__main__":
    unittest.main()
