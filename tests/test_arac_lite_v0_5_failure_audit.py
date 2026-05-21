import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_5_failure_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("v0_5_failure_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV05FailureAuditTests(unittest.TestCase):
    def test_offline_candidate_rebuilds_disable_then_future_accepted_fusion(self):
        module = load_module()
        rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.DISABLE_FAST_METHOD,
                "pass_id": 0,
                "var_id": 7,
                "action_candidate": "Fusion",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "-1.0",
                "proposal_support": "2",
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.DISABLE_FAST_METHOD,
                "pass_id": 1,
                "var_id": 7,
                "action_candidate": "Disable",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "0.0",
                "proposal_support": "2",
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.DISABLE_FAST_METHOD,
                "pass_id": 2,
                "var_id": 7,
                "action_candidate": "Fusion",
                "validation_attempted": "True",
                "validation_accepted": "True",
                "validation_delta": "3.0",
                "proposal_support": "2",
            },
        ]

        candidates = module.build_offline_candidate_rows(rows, {"S6"}, {10000})

        self.assertEqual(1, len(candidates))
        self.assertEqual("middle", candidates[0]["phase"])
        self.assertEqual(2, candidates[0]["future_accepted_fusion_pass_id"])

    def test_match_rows_count_same_phase_var_probe_and_positive_delta(self):
        module = load_module()
        offline = [
            {
                "problem": "S6",
                "seed": "1",
                "tfes": "10000",
                "phase": "middle",
                "var_id": "7",
                "group_pair": "1+2",
            }
        ]
        online = [
            {
                "problem": "S6",
                "seed": "1",
                "tfes": "10000",
                "phase": "middle",
                "var_id": "7",
                "group_pair": "1+2",
                "validation_delta": "4.0",
            },
            {
                "problem": "S6",
                "seed": "1",
                "tfes": "10000",
                "phase": "late",
                "var_id": "7",
                "group_pair": "1+2",
                "validation_delta": "5.0",
            },
        ]

        rows = module.build_match_rows(offline, online)
        middle = next(row for row in rows if row["phase"] == "middle")

        self.assertEqual(1, middle["offline_candidate_count"])
        self.assertEqual(1, middle["matched_probe_count"])
        self.assertEqual(1, middle["matched_positive_delta_count"])
        self.assertAlmostEqual(1.0, middle["match_rate"])
        self.assertEqual(1, middle["strict_group_pair_matched_probe_count"])

    def test_validation_attribution_groups_global_delta_by_pass(self):
        module = load_module()
        rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.V05_METHOD,
                "pass_id": 4,
                "var_id": 1,
                "action_candidate": "Fusion",
                "action_reason": "probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_probe_phase": "middle",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "-2.0",
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.V05_METHOD,
                "pass_id": 4,
                "var_id": 2,
                "action_candidate": "Disable",
                "action_reason": "history_disabled",
                "arac_probe_candidate": "False",
                "arac_probe_phase": "middle",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "-2.0",
            },
        ]

        attribution = module.build_validation_delta_attribution_rows(rows, {"S6"}, {10000})

        self.assertEqual(1, len(attribution))
        self.assertEqual(1, attribution[0]["num_probe_relations"])
        self.assertEqual(1, attribution[0]["num_disable_relations"])
        self.assertEqual(2, attribution[0]["relation_count"])
        self.assertGreater(attribution[0]["action_mix_entropy"], 0.0)
        self.assertEqual(-2.0, attribution[0]["global_delta"])


if __name__ == "__main__":
    unittest.main()
