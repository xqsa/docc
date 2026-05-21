import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_5_probe_artifacts.py"


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


def load_v0_5_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_5", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV05ConfigTests(unittest.TestCase):
    def test_probe_config_normalizes_budget_and_phase(self):
        InfoAwareNDAConfig = load_info_config()
        config = InfoAwareNDAConfig(
            arac_lite_probe_enabled=True,
            arac_lite_probe_every_n_pass=0,
            arac_lite_probe_max_per_pass=-3,
            arac_lite_probe_min_phase="bad",
            arac_lite_probe_recovery_min_attempts=0,
            arac_lite_probe_recovery_accept_rate_threshold=2.0,
        ).normalized()

        self.assertTrue(config.arac_lite_probe_enabled)
        self.assertEqual(1, config.arac_lite_probe_every_n_pass)
        self.assertEqual(1, config.arac_lite_probe_max_per_pass)
        self.assertEqual("middle", config.arac_lite_probe_min_phase)
        self.assertEqual(1, config.arac_lite_probe_recovery_min_attempts)
        self.assertEqual(1.0, config.arac_lite_probe_recovery_accept_rate_threshold)


class AracLiteV05ProbeGateTests(unittest.TestCase):
    def test_middle_phase_history_disabled_relation_can_be_low_frequency_probe(self):
        hcc_es = load_hcc_es_module()
        best = np.array([0.0, 0.0, 0.0], dtype=float)
        proposal_rows = [
            {
                "var_id": 1,
                "proposal_value": 0.10,
                "delta": 1.0,
                "fitness_before": 10.0,
                "fitness_after": 9.0,
                "group_id": 0,
            },
            {
                "var_id": 1,
                "proposal_value": 0.11,
                "delta": 2.0,
                "fitness_before": 10.0,
                "fitness_after": 8.0,
                "group_id": 1,
            },
        ]
        hypergraph = {"overlap_vars": [1], "var_to_groups": {1: [0, 1]}}
        history = {
            1: {
                "attempt_count": 3,
                "accept_count": 0,
                "delta_sum": -3.0,
                "reject_streak": 3,
                "recent_attempts": [{"accepted": False, "delta": -1.0}] * 3,
            }
        }

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(3, dtype=float) * 100.0,
            arac_relation_history=history,
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_recovery_enabled=True,
            arac_lite_recovery_phase="middle",
            arac_lite_recovery_min_attempts=20,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=2,
            arac_lite_probe_max_per_pass=1,
            arac_lite_probe_min_phase="middle",
        )

        self.assertEqual("Fusion", rows[0]["action_candidate"])
        self.assertEqual("probe_fusion_candidate", rows[0]["action_reason"])
        self.assertTrue(rows[0]["arac_probe_candidate"])
        self.assertEqual("probe_selected", rows[0]["arac_probe_reason"])
        self.assertTrue(rows[0]["gate_passed"])

    def test_early_phase_does_not_probe_history_disabled_relation(self):
        hcc_es = load_hcc_es_module()
        best = np.array([0.0, 0.0, 0.0], dtype=float)
        proposal_rows = [
            {
                "var_id": 1,
                "proposal_value": 0.10,
                "delta": 1.0,
                "fitness_before": 10.0,
                "fitness_after": 9.0,
                "group_id": 0,
            },
            {
                "var_id": 1,
                "proposal_value": 0.11,
                "delta": 2.0,
                "fitness_before": 10.0,
                "fitness_after": 8.0,
                "group_id": 1,
            },
        ]
        hypergraph = {"overlap_vars": [1], "var_to_groups": {1: [0, 1]}}
        history = {1: {"attempt_count": 3, "accept_count": 0, "delta_sum": -3.0, "reject_streak": 3}}

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(3, dtype=float) * 100.0,
            arac_relation_history=history,
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="early",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=2,
            arac_lite_probe_max_per_pass=1,
            arac_lite_probe_min_phase="middle",
        )

        self.assertEqual("Disable", rows[0]["action_candidate"])
        self.assertFalse(rows[0]["arac_probe_candidate"])
        self.assertEqual("phase_not_probeable", rows[0]["arac_probe_reason"])

    def test_probe_conflict_freeze_is_not_counted_as_probe_candidate(self):
        hcc_es = load_hcc_es_module()
        best = np.array([0.0, 0.0, 0.0], dtype=float)
        proposal_rows = [
            {
                "var_id": 1,
                "proposal_value": 0.10,
                "delta": 1.0,
                "fitness_before": 10.0,
                "fitness_after": 9.0,
                "group_id": 0,
            },
            {
                "var_id": 1,
                "proposal_value": 10.0,
                "delta": 2.0,
                "fitness_before": 10.0,
                "fitness_after": 8.0,
                "group_id": 1,
            },
        ]
        hypergraph = {"overlap_vars": [1], "var_to_groups": {1: [0, 1]}}
        history = {1: {"attempt_count": 3, "accept_count": 0, "delta_sum": -3.0, "reject_streak": 3}}

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(3, dtype=float) * 100.0,
            arac_relation_history=history,
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=2,
            arac_lite_probe_max_per_pass=1,
            arac_lite_probe_min_phase="middle",
        )

        self.assertEqual("Freeze", rows[0]["action_candidate"])
        self.assertEqual("probe_proposal_conflict", rows[0]["action_reason"])
        self.assertFalse(rows[0]["arac_probe_candidate"])
        self.assertEqual("probe_proposal_conflict", rows[0]["arac_probe_reason"])

    def test_probe_history_can_drive_recovery_only_with_positive_delta_mean(self):
        hcc_es = load_hcc_es_module()
        positive_history = {
            "probe_attempt_count": 5,
            "probe_accept_count": 3,
            "probe_delta_sum": 4.0,
            "probe_recent_attempts": [
                {"accepted": True, "delta": 2.0},
                {"accepted": False, "delta": -1.0},
                {"accepted": True, "delta": 1.0},
                {"accepted": False, "delta": -0.5},
                {"accepted": True, "delta": 2.5},
            ],
        }
        negative_history = {
            **positive_history,
            "probe_delta_sum": -4.0,
            "probe_recent_attempts": [{"accepted": True, "delta": -1.0}] * 5,
        }

        positive = hcc_es.should_recover_arac_relation(
            positive_history,
            phase="middle",
            recovery_enabled=True,
            recovery_min_attempts=20,
            recovery_accept_rate_threshold=0.25,
            recovery_delta_threshold=0.0,
            recovery_positive_delta_rate_threshold=0.25,
            probe_recovery_enabled=True,
            probe_recovery_min_attempts=5,
            probe_recovery_accept_rate_threshold=0.3,
            probe_recovery_delta_threshold=0.0,
        )
        negative = hcc_es.should_recover_arac_relation(
            negative_history,
            phase="middle",
            recovery_enabled=True,
            recovery_min_attempts=20,
            recovery_accept_rate_threshold=0.25,
            recovery_delta_threshold=0.0,
            recovery_positive_delta_rate_threshold=0.25,
            probe_recovery_enabled=True,
            probe_recovery_min_attempts=5,
            probe_recovery_accept_rate_threshold=0.3,
            probe_recovery_delta_threshold=0.0,
        )

        self.assertTrue(positive["should_recover"])
        self.assertEqual("probe_recovery_signal_passed", positive["recovery_reason"])
        self.assertFalse(negative["should_recover"])
        self.assertEqual("probe_delta_nonpositive", negative["recovery_reason"])


class AracLiteV05ArtifactTests(unittest.TestCase):
    def test_probe_metrics_count_probe_acceptance_and_bad_recovery(self):
        module = load_v0_5_module()
        rows = [
            {
                "problem": "S6",
                "method": "arac-lite-v0.5-low-frequency-probe",
                "pass_id": 4,
                "arac_recovery_phase": "middle",
                "action_candidate": "Fusion",
                "action_reason": "probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "validation_attempted": "True",
                "validation_accepted": "True",
                "validation_delta": "2.0",
            },
            {
                "problem": "R6",
                "method": "arac-lite-v0.5-low-frequency-probe",
                "pass_id": 4,
                "arac_recovery_phase": "middle",
                "action_candidate": "Fusion",
                "action_reason": "probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "validation_attempted": "True",
                "validation_accepted": "False",
                "validation_delta": "-3.0",
            },
            {
                "problem": "R6",
                "method": "arac-lite-v0.5-low-frequency-probe",
                "pass_id": 4,
                "arac_recovery_phase": "middle",
                "action_candidate": "Freeze",
                "action_reason": "probe_proposal_conflict",
                "arac_probe_candidate": "False",
                "validation_attempted": "False",
                "validation_accepted": "False",
                "validation_delta": "0.0",
            },
            {
                "problem": "S6",
                "method": "arac-lite-v0.5-low-frequency-probe",
                "pass_id": 5,
                "arac_recovery_phase": "middle",
                "action_candidate": "Fusion",
                "action_reason": "recovery_fusion_supported",
                "arac_recovery_candidate": "True",
                "validation_attempted": "True",
                "validation_accepted": "True",
                "validation_delta": "1.0",
            },
        ]

        metrics = module.build_probe_metric_rows(rows)
        by_problem = {row["problem"]: row for row in metrics}

        self.assertEqual(1, by_problem["S6"]["probe_count"])
        self.assertEqual(1, by_problem["S6"]["probe_accept_count"])
        self.assertEqual(1, by_problem["S6"]["recovered_fusion_count"])
        self.assertEqual(0, by_problem["S6"]["bad_recovery_count"])
        self.assertEqual(1, by_problem["S6"]["S6_middle_recovery_count"])
        self.assertEqual(1, by_problem["R6"]["R6_bad_probe_count"])

    def test_report_includes_probe_metrics(self):
        module = load_v0_5_module()
        args = SimpleNamespace(problems=["S6"], seeds=[1], tfes=[10000], cc_pass_group_fes=20)
        probe_rows = [
            {
                "problem": "S6",
                "probe_count": 2,
                "probe_accept_count": 1,
                "probe_accept_rate": 0.5,
                "probe_delta_mean": 1.5,
                "recovered_fusion_count": 1,
                "bad_recovery_count": 0,
                "extra_fe_ratio": 0.01,
                "S6_middle_recovery_count": 1,
                "R6_bad_probe_count": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            old_report_path = module.REPORT_PATH
            module.REPORT_PATH = Path(temp_dir) / "report.md"
            try:
                module.write_report([], [], probe_rows, args)
                report = module.REPORT_PATH.read_text(encoding="utf-8")
            finally:
                module.REPORT_PATH = old_report_path

    def test_report_marks_failed_benefit_acceptance(self):
        module = load_v0_5_module()
        args = SimpleNamespace(problems=["S6", "R6"], seeds=[1], tfes=[10000], cc_pass_group_fes=20)
        summary_rows = [
            {
                "problem": "S6",
                "tfes": 10000,
                "method": module.CANDIDATE_METHOD,
                "best_mean": 10.0,
                "gap_vs_no_coordination": 0.008,
                "gap_vs_validated": 0.007,
                "gap_vs_disable_fast": 0.004,
                "fusion_count": 1,
                "freeze_count": 0,
                "disable_count": 0,
                "fusion_validation_accept_rate": 0.2,
            },
            {
                "problem": "R6",
                "tfes": 10000,
                "method": module.CANDIDATE_METHOD,
                "best_mean": 20.0,
                "gap_vs_no_coordination": 0.009,
                "gap_vs_validated": -0.004,
                "gap_vs_disable_fast": 0.007,
                "fusion_count": 1,
                "freeze_count": 0,
                "disable_count": 0,
                "fusion_validation_accept_rate": 0.6,
            },
        ]
        probe_rows = [
            {
                "problem": "S6",
                "probe_count": 10,
                "probe_accept_count": 4,
                "probe_accept_rate": 0.4,
                "probe_delta_mean": -1.0,
                "recovered_fusion_count": 1,
                "bad_recovery_count": 1,
                "extra_fe_ratio": 0.001,
                "S6_middle_recovery_count": 0,
                "R6_bad_probe_count": 0,
            },
            {
                "problem": "R6",
                "probe_count": 10,
                "probe_accept_count": 6,
                "probe_accept_rate": 0.6,
                "probe_delta_mean": -1.0,
                "recovered_fusion_count": 1,
                "bad_recovery_count": 0,
                "extra_fe_ratio": 0.001,
                "S6_middle_recovery_count": 0,
                "R6_bad_probe_count": 4,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            old_report_path = module.REPORT_PATH
            module.REPORT_PATH = Path(temp_dir) / "report.md"
            try:
                module.write_report([], summary_rows, probe_rows, args)
                report = module.REPORT_PATH.read_text(encoding="utf-8")
                self.assertIn("收益验收：不通过", report)
                self.assertIn("probe 额外 FE 成本达标", report)
                self.assertIn("S6 未改善", report)
                self.assertIn("R6 被拉坏", report)
            finally:
                module.REPORT_PATH = old_report_path

        self.assertIn("## Probe Metrics", report)
        self.assertIn("S6", report)
        self.assertIn("recovered_fusion", report)


if __name__ == "__main__":
    unittest.main()
