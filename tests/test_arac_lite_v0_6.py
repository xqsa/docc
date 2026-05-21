import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
V06_SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_6_targeted_probe_artifacts.py"


def load_hcc_es_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_info_config():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from HCC.info_aware_nda import InfoAwareNDAConfig

    return InfoAwareNDAConfig


def load_v0_6_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_6", V06_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relation_history(mean_delta=1.0):
    return {
        "attempt_count": 3,
        "accept_count": 1,
        "delta_sum": float(mean_delta) * 3.0,
        "positive_delta_count": 3 if mean_delta > 0.0 else 0,
        "reject_streak": 3,
        "recent_attempts": [{"accepted": False, "delta": float(mean_delta)}] * 3,
    }


class AracLiteV06ConfigTests(unittest.TestCase):
    def test_targeted_probe_config_normalizes_signature_thresholds(self):
        InfoAwareNDAConfig = load_info_config()
        config = InfoAwareNDAConfig(
            arac_lite_targeted_probe_enabled=True,
            arac_lite_targeted_probe_phase="bad",
            arac_lite_targeted_probe_min_support=0,
            arac_lite_targeted_probe_min_proposal_std_ratio=-1.0,
            arac_lite_targeted_probe_max_proposal_std_ratio=0.0001,
            arac_lite_targeted_probe_min_relation_delta=0.0,
        ).normalized()

        self.assertTrue(config.arac_lite_targeted_probe_enabled)
        self.assertEqual("middle", config.arac_lite_targeted_probe_phase)
        self.assertEqual(1, config.arac_lite_targeted_probe_min_support)
        self.assertEqual(0.0, config.arac_lite_targeted_probe_min_proposal_std_ratio)
        self.assertGreaterEqual(
            config.arac_lite_targeted_probe_max_proposal_std_ratio,
            config.arac_lite_targeted_probe_min_proposal_std_ratio,
        )


class AracLiteV06TargetedProbeGateTests(unittest.TestCase):
    def test_targeted_probe_uses_signature_score_before_overlap_order(self):
        hcc_es = load_hcc_es_module()
        best = np.zeros(4, dtype=float)
        proposal_rows = [
            {"var_id": 1, "proposal_value": 0.10, "delta": 1.0, "fitness_before": 10.0, "fitness_after": 9.0, "group_id": 0},
            {"var_id": 1, "proposal_value": 0.11, "delta": 2.0, "fitness_before": 10.0, "fitness_after": 8.0, "group_id": 1},
            {"var_id": 2, "proposal_value": 0.10, "delta": 1.0, "fitness_before": 10.0, "fitness_after": 9.0, "group_id": 0},
            {"var_id": 2, "proposal_value": 0.50, "delta": 2.0, "fitness_before": 10.0, "fitness_after": 8.0, "group_id": 1},
        ]
        hypergraph = {"overlap_vars": [1, 2], "var_to_groups": {1: [0, 1], 2: [0, 1]}}

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(4, dtype=float) * 100.0,
            arac_relation_history={1: relation_history(1.0), 2: relation_history(1.0)},
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=2,
            arac_lite_probe_max_per_pass=1,
            arac_lite_probe_min_phase="middle",
            arac_lite_targeted_probe_enabled=True,
            arac_lite_targeted_probe_phase="middle",
            arac_lite_targeted_probe_min_support=2,
            arac_lite_targeted_probe_min_proposal_std_ratio=0.001,
            arac_lite_targeted_probe_max_proposal_std_ratio=0.01,
            arac_lite_targeted_probe_min_relation_delta=0.0,
        )

        by_var = {int(row["var_id"]): row for row in rows}
        self.assertEqual("Disable", by_var[1]["action_candidate"])
        self.assertEqual("Fusion", by_var[2]["action_candidate"])
        self.assertTrue(by_var[2]["arac_probe_candidate"])
        self.assertTrue(by_var[2]["arac_targeted_probe_candidate"])
        self.assertEqual("targeted_probe_fusion_candidate", by_var[2]["action_reason"])

    def test_targeted_probe_can_probe_proposal_conflict_freeze_candidate(self):
        hcc_es = load_hcc_es_module()
        best = np.zeros(3, dtype=float)
        proposal_rows = [
            {"var_id": 1, "proposal_value": 0.10, "delta": 1.0, "fitness_before": 10.0, "fitness_after": 9.0, "group_id": 0},
            {"var_id": 1, "proposal_value": 0.50, "delta": 2.0, "fitness_before": 10.0, "fitness_after": 8.0, "group_id": 1},
        ]
        hypergraph = {"overlap_vars": [1], "var_to_groups": {1: [0, 1]}}
        positive_history = {
            "attempt_count": 3,
            "accept_count": 3,
            "delta_sum": 6.0,
            "positive_delta_count": 3,
            "reject_streak": 0,
            "recent_attempts": [{"accepted": True, "delta": 2.0}] * 3,
        }

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(3, dtype=float) * 100.0,
            selective_max_proposal_std_ratio=0.001,
            arac_relation_history={1: positive_history},
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=2,
            arac_lite_probe_max_per_pass=1,
            arac_lite_probe_min_phase="middle",
            arac_lite_targeted_probe_enabled=True,
            arac_lite_targeted_probe_phase="middle",
            arac_lite_targeted_probe_min_support=2,
            arac_lite_targeted_probe_min_proposal_std_ratio=0.001,
            arac_lite_targeted_probe_max_proposal_std_ratio=0.01,
            arac_lite_targeted_probe_min_relation_delta=0.0,
        )

        self.assertEqual("Fusion", rows[0]["action_candidate"])
        self.assertTrue(str(rows[0]["action_reason"]).startswith("targeted_probe_fusion"))
        self.assertTrue(rows[0]["arac_targeted_probe_candidate"])
        self.assertEqual("targeted_probe_selected", rows[0]["arac_targeted_probe_reason"])

    def test_targeted_probe_blocks_nonpositive_relation_delta_even_with_high_accept(self):
        hcc_es = load_hcc_es_module()
        best = np.zeros(3, dtype=float)
        proposal_rows = [
            {"var_id": 1, "proposal_value": 0.10, "delta": 1.0, "fitness_before": 10.0, "fitness_after": 9.0, "group_id": 0},
            {"var_id": 1, "proposal_value": 0.50, "delta": 2.0, "fitness_before": 10.0, "fitness_after": 8.0, "group_id": 1},
        ]
        hypergraph = {"overlap_vars": [1], "var_to_groups": {1: [0, 1]}}

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(3, dtype=float) * 100.0,
            arac_relation_history={
                1: {
                    "attempt_count": 3,
                    "accept_count": 3,
                    "delta_sum": -3.0,
                    "positive_delta_count": 0,
                    "reject_streak": 3,
                    "recent_attempts": [{"accepted": True, "delta": -1.0}] * 3,
                }
            },
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=2,
            arac_lite_probe_max_per_pass=1,
            arac_lite_probe_min_phase="middle",
            arac_lite_targeted_probe_enabled=True,
            arac_lite_targeted_probe_phase="middle",
            arac_lite_targeted_probe_min_support=2,
            arac_lite_targeted_probe_min_proposal_std_ratio=0.001,
            arac_lite_targeted_probe_max_proposal_std_ratio=0.01,
            arac_lite_targeted_probe_min_relation_delta=0.0,
        )

        self.assertEqual("Disable", rows[0]["action_candidate"])
        self.assertFalse(rows[0]["arac_targeted_probe_candidate"])
        self.assertEqual("targeted_rolling_delta_nonpositive", rows[0]["arac_targeted_probe_reason"])


class AracLiteV06ArtifactTests(unittest.TestCase):
    def test_targeted_probe_variant_requires_stable_positive_history(self):
        module = load_v0_6_module()
        overrides = module.targeted_probe_variant().overrides

        self.assertEqual(2, overrides["arac_lite_targeted_probe_min_relation_attempts"])
        self.assertEqual(1.0, overrides["arac_lite_targeted_probe_min_accept_rate"])
        self.assertEqual(1.0, overrides["arac_lite_targeted_probe_min_positive_delta_rate"])

    def test_offline_candidates_are_filtered_to_requested_seeds(self):
        module = load_v0_6_module()
        args = SimpleNamespace(problems=["S6"], seeds=[1], tfes=[10000])
        rows = [
            {"problem": "S6", "seed": 1, "tfes": 10000, "phase": "middle", "var_id": 42},
            {"problem": "S6", "seed": 7, "tfes": 10000, "phase": "middle", "var_id": 77},
            {"problem": "E6", "seed": 1, "tfes": 10000, "phase": "middle", "var_id": 13},
        ]

        filtered = module.filter_offline_candidates_for_args(rows, args)

        self.assertEqual([42], [row["var_id"] for row in filtered])

    def test_targeted_probe_metrics_count_signature_selection_and_offline_match(self):
        module = load_v0_6_module()
        relation_rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.CANDIDATE_METHOD,
                "pass_id": 6,
                "var_id": 42,
                "phase": "middle",
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_attempted": "True",
                "validation_accepted": "True",
                "validation_delta": "5.0",
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": module.CANDIDATE_METHOD,
                "pass_id": 6,
                "var_id": 8,
                "phase": "middle",
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "-2.0",
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": module.CANDIDATE_METHOD,
                "pass_id": 7,
                "var_id": 77,
                "phase": "middle",
                "action_candidate": "Disable",
                "arac_targeted_probe_candidate": "False",
                "arac_targeted_probe_signature_matched": "True",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "-1.0",
            },
        ]
        offline_candidates = [
            {"problem": "S6", "seed": 1, "tfes": 10000, "phase": "middle", "var_id": 42},
            {"problem": "S6", "seed": 1, "tfes": 10000, "phase": "middle", "var_id": 77},
        ]

        metrics = module.build_targeted_probe_metric_rows(relation_rows, offline_candidates)
        by_problem = {row["problem"]: row for row in metrics}

        self.assertEqual(2, by_problem["S6"]["targeted_candidate_count"])
        self.assertEqual(1, by_problem["S6"]["targeted_probe_count"])
        self.assertEqual(1, by_problem["S6"]["matched_probe_count"])
        self.assertEqual(2, by_problem["S6"]["matched_offline_candidate_count"])
        self.assertAlmostEqual(5.0, by_problem["S6"]["targeted_probe_delta_mean"])
        self.assertEqual(1, by_problem["R6"]["R6_bad_probe_count"])


if __name__ == "__main__":
    unittest.main()
