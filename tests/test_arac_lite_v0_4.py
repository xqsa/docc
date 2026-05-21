import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_4_recovery_artifacts.py"


def load_hcc_es_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v0_4_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_4", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV04RecoveryGateTests(unittest.TestCase):
    def test_recovery_gate_requires_middle_or_late_positive_rolling_signal(self):
        hcc_es = load_hcc_es_module()
        strong_history = {
            "attempt_count": 21,
            "accept_count": 18,
            "delta_sum": 12.0,
            "reject_streak": 1,
            "recent_attempts": [{"accepted": True, "delta": 1.0}] * 20,
        }

        early = hcc_es.should_recover_arac_relation(
            strong_history,
            phase="early",
            recovery_enabled=True,
            recovery_min_attempts=20,
            recovery_accept_rate_threshold=0.25,
            recovery_delta_threshold=0.0,
            recovery_positive_delta_rate_threshold=0.25,
        )
        middle = hcc_es.should_recover_arac_relation(
            strong_history,
            phase="middle",
            recovery_enabled=True,
            recovery_min_attempts=20,
            recovery_accept_rate_threshold=0.25,
            recovery_delta_threshold=0.0,
            recovery_positive_delta_rate_threshold=0.25,
        )
        negative_delta = hcc_es.should_recover_arac_relation(
            {
                **strong_history,
                "recent_attempts": [{"accepted": True, "delta": -1.0}] * 20,
            },
            phase="late",
            recovery_enabled=True,
            recovery_min_attempts=20,
            recovery_accept_rate_threshold=0.25,
            recovery_delta_threshold=0.0,
            recovery_positive_delta_rate_threshold=0.25,
        )

        self.assertFalse(early["should_recover"])
        self.assertEqual("phase_not_recoverable", early["recovery_reason"])
        self.assertTrue(middle["should_recover"])
        self.assertEqual("recovery_signal_passed", middle["recovery_reason"])
        self.assertFalse(negative_delta["should_recover"])
        self.assertEqual("rolling_delta_nonpositive", negative_delta["recovery_reason"])

    def test_recovery_gate_bypasses_disable_fast_only_when_signal_is_strong(self):
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
                "attempt_count": 21,
                "accept_count": 18,
                "delta_sum": 12.0,
                "reject_streak": 1,
                "recent_attempts": [{"accepted": True, "delta": 1.0}] * 20,
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
            arac_lite_recovery_accept_rate_threshold=0.25,
            arac_lite_recovery_delta_threshold=0.0,
            arac_lite_recovery_positive_delta_rate_threshold=0.25,
        )

        self.assertEqual("Fusion", rows[0]["action_candidate"])
        self.assertEqual("recovery_fusion_supported", rows[0]["action_reason"])
        self.assertTrue(rows[0]["gate_passed"])
        self.assertTrue(rows[0]["arac_recovery_candidate"])


class AracLiteV04OfflineAuditTests(unittest.TestCase):
    def test_offline_recovery_audit_separates_accept_rate_from_positive_delta(self):
        module = load_v0_4_module()
        rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 0,
                "var_id": 7,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": -1.0,
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 1,
                "var_id": 7,
                "action_candidate": "Disable",
                "validation_attempted": True,
                "validation_accepted": False,
                "validation_delta": 0.0,
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 2,
                "var_id": 7,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": True,
                "validation_delta": 2.0,
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 10000,
                "method": "arac-lite-v0.1-disable-fast",
                "pass_id": 0,
                "var_id": 8,
                "action_candidate": "Fusion",
                "validation_attempted": True,
                "validation_accepted": True,
                "validation_delta": -5.0,
            },
        ]

        audit_rows = module.build_offline_recovery_audit_rows(rows)
        by_key = {
            (row["problem"], int(row["tfes"]), row["phase"]): row
            for row in audit_rows
        }

        s6_middle = by_key[("S6", 10000, "middle")]
        r6_early = by_key[("R6", 10000, "early")]

        self.assertEqual(1, s6_middle["disable_false_negative_count"])
        self.assertEqual(1, s6_middle["recovery_candidate_count"])
        self.assertAlmostEqual(1.0, s6_middle["disable_false_negative_rate"])
        self.assertAlmostEqual(1.0, r6_early["fusion_accept_rate"])
        self.assertAlmostEqual(-5.0, r6_early["fusion_delta_mean"])
        self.assertAlmostEqual(0.0, r6_early["fusion_positive_delta_rate"])


class AracLiteV04ReportTests(unittest.TestCase):
    def test_report_records_online_recovery_candidates_and_blocking_reasons(self):
        module = load_v0_4_module()
        args = SimpleNamespace(
            problems=["S6"],
            seeds=[1],
            tfes=[10000],
            cc_pass_group_fes=20,
        )
        relation_rows = [
            {
                "problem": "S6",
                "method": "arac-lite-v0.4-recovery",
                "arac_recovery_candidate": "True",
                "arac_recovery_reason": "recovery_signal_passed",
            },
            {
                "problem": "S6",
                "method": "arac-lite-v0.4-recovery",
                "arac_recovery_candidate": "False",
                "arac_recovery_reason": "rolling_attempts_below_min",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            old_report_path = module.REPORT_PATH
            module.REPORT_PATH = Path(temp_dir) / "report.md"
            try:
                module.write_report(
                    offline_rows=[],
                    run_rows=[{"status": "ok"}],
                    summary_rows=[],
                    robustness_rows=[],
                    args=args,
                    relation_rows=relation_rows,
                )
                report = module.REPORT_PATH.read_text(encoding="utf-8")
            finally:
                module.REPORT_PATH = old_report_path

        self.assertIn("## Online Recovery Diagnostics", report)
        self.assertIn("V0.4 recovery candidate rows: 1", report)
        self.assertIn("rolling_attempts_below_min=1", report)


if __name__ == "__main__":
    unittest.main()
