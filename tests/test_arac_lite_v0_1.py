import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_1_artifacts.py"


def load_v0_1_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_1", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV01AttributionTests(unittest.TestCase):
    def test_action_attribution_reports_accept_false_negative_and_oracle_regret(self):
        module = load_v0_1_module()
        rows = [
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-rule",
                "pass_id": 0,
                "var_id": 7,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": -2.0,
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": "oracle-proxy",
                "pass_id": 0,
                "var_id": 7,
                "action_candidate": "Disable",
                "validation_attempted": True,
                "validation_accepted": True,
                "validation_delta": 0.0,
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-rule",
                "pass_id": 1,
                "var_id": 8,
                "action_candidate": "Disable",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": -1.0,
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-rule",
                "pass_id": 2,
                "var_id": 8,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": True,
                "validation_delta": 3.0,
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-rule",
                "pass_id": 3,
                "var_id": 9,
                "action_candidate": "Freeze",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": -4.0,
            },
        ]

        attribution = module.build_action_attribution_rows(rows)
        by_key = {
            (row["problem"], row["method"], row["action_candidate"]): row
            for row in attribution
        }

        fusion = by_key[("R6", "arac-lite-rule", "Fusion")]
        disable = by_key[("R6", "arac-lite-rule", "Disable")]
        freeze = by_key[("R6", "arac-lite-rule", "Freeze")]

        self.assertEqual(2, fusion["action_count"])
        self.assertAlmostEqual(0.5, fusion["fusion_accept_rate"])
        self.assertAlmostEqual(0.5, fusion["fusion_reject_rate"])
        self.assertAlmostEqual(0.5, fusion["action_delta_mean"])
        self.assertAlmostEqual(1.0, fusion["action_regret_vs_oracle"])
        self.assertEqual(1, freeze["freeze_saved_bad_update_count"])
        self.assertEqual(1, disable["disable_false_negative_count"])
        self.assertAlmostEqual(1.0, disable["disable_false_negative_rate"])

    def test_threshold_variants_focus_on_disable_fusion_and_balanced_split(self):
        module = load_v0_1_module()

        variants = module.build_threshold_variants()
        labels = [variant.label for variant in variants]

        self.assertIn("arac-lite-v0", labels)
        self.assertIn("arac-lite-v0.1-disable-fast", labels)
        self.assertIn("arac-lite-v0.1-fusion-strict", labels)
        self.assertIn("arac-lite-v0.1-balanced", labels)

        by_label = {variant.label: variant for variant in variants}
        disable_fast = by_label["arac-lite-v0.1-disable-fast"]
        fusion_strict = by_label["arac-lite-v0.1-fusion-strict"]
        balanced = by_label["arac-lite-v0.1-balanced"]

        self.assertLess(disable_fast.overrides["arac_lite_disable_reject_streak"], 2)
        self.assertGreater(fusion_strict.overrides["shared_variable_selective_min_positive_proposals"], 2)
        self.assertLess(
            fusion_strict.overrides["shared_variable_selective_max_proposal_std_ratio"],
            0.00125,
        )
        self.assertLessEqual(balanced.overrides["arac_lite_history_min_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
