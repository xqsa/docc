import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from dataclasses import replace


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT_PATH = REPO_ROOT / "scripts" / "final" / "run_arac_lite_3e6_gate.py"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("arac_lite_final_3e6_gate", GATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteFinalGateConfigTests(unittest.TestCase):
    def test_load_final_config_freezes_cc_pass_group_fes(self):
        module = load_gate_module()

        config = module.load_final_config(cc_pass_group_fes=20)

        self.assertTrue(config.enable)
        self.assertTrue(config.arac_lite_recovery_enabled)
        self.assertTrue(config.arac_lite_probe_enabled)
        self.assertEqual(20, config.cc_pass_group_fes)
        self.assertEqual(3, config.cc_min_passes)

    def test_build_case_tasks_is_single_final_method_matrix(self):
        module = load_gate_module()
        args = SimpleNamespace(problems=["S6", "R6"], seeds=[1, 2], tfes=3000000, cc_pass_group_fes=20)

        tasks = module.build_case_tasks(args)

        self.assertEqual(4, len(tasks))
        self.assertEqual({module.FINAL_METHOD}, {task["method"] for task in tasks})
        self.assertEqual({3000000}, {task["tfes"] for task in tasks})
        self.assertEqual({20}, {task["cc_pass_group_fes"] for task in tasks})


class AracLiteFinalGateResumeTests(unittest.TestCase):
    def test_resume_only_skips_ok_case_rows(self):
        module = load_gate_module()
        tasks = [
            {"problem": "S6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "S6", "seed": 2, "tfes": 3000000, "method": module.FINAL_METHOD},
        ]
        existing_rows = [
            {
                "problem": "S6",
                "seed": "1",
                "tfes": "3000000",
                "method": module.FINAL_METHOD,
                "status": "ok",
            },
            {
                "problem": "S6",
                "seed": "2",
                "tfes": "3000000",
                "method": module.FINAL_METHOD,
                "status": "error:RuntimeError:boom",
            },
        ]

        pending = module.filter_completed_tasks(tasks, module.completed_case_keys(existing_rows))

        self.assertEqual([tasks[1]], pending)

    def test_parse_args_supports_summarize_only(self):
        module = load_gate_module()
        old_argv = module.sys.argv
        try:
            module.sys.argv = ["run_arac_lite_3e6_gate.py", "--summarize-only"]
            args = module.parse_args()
        finally:
            module.sys.argv = old_argv

        self.assertTrue(args.summarize_only)

    def test_parse_args_supports_fast_gate_prefix_and_longest_first(self):
        module = load_gate_module()
        old_argv = module.sys.argv
        try:
            module.sys.argv = [
                "run_arac_lite_3e6_gate.py",
                "--fast-gate",
                "--artifact-prefix",
                "arac_lite_final_3e6_fast_gate",
                "--reuse-default-cache",
                "--schedule",
                "longest-first",
            ]
            args = module.parse_args()
        finally:
            module.sys.argv = old_argv

        self.assertTrue(args.fast_gate)
        self.assertEqual("arac_lite_final_3e6_fast_gate", args.artifact_prefix)
        self.assertTrue(args.reuse_default_cache)
        self.assertEqual("longest-first", args.schedule)

    def test_fast_gate_config_disables_heavy_traces_without_changing_method_rules(self):
        module = load_gate_module()

        normal = module.load_final_config(cc_pass_group_fes=20, fast_gate=False)
        fast = module.load_final_config(cc_pass_group_fes=20, fast_gate=True)

        self.assertTrue(normal.save_diagnostics)
        self.assertTrue(normal.save_shared_variable_trace)
        self.assertTrue(normal.validation_trace_enabled)
        self.assertFalse(fast.save_diagnostics)
        self.assertFalse(fast.save_shared_variable_trace)
        self.assertTrue(fast.validation_trace_enabled)
        self.assertFalse(fast.save_overlap_blend_trace)
        self.assertFalse(fast.selector_trace_enabled)
        self.assertEqual(normal.shared_variable_coordination_mode, fast.shared_variable_coordination_mode)
        self.assertEqual(normal.arac_lite_recovery_enabled, fast.arac_lite_recovery_enabled)
        self.assertEqual(normal.arac_lite_probe_enabled, fast.arac_lite_probe_enabled)
        self.assertEqual(normal.arac_lite_targeted_probe_enabled, fast.arac_lite_targeted_probe_enabled)
        self.assertEqual(normal.validation_max_extra_fe_ratio, fast.validation_max_extra_fe_ratio)

    def test_artifact_paths_can_be_rebound_by_prefix(self):
        module = load_gate_module()

        module.configure_artifact_paths("arac_lite_final_3e6_fast_gate")

        self.assertEqual(
            module.ARTIFACTS_ROOT / "arac_lite_final_3e6_fast_gate_run_details.csv",
            module.RUN_DETAILS_PATH,
        )
        self.assertEqual(
            module.ARTIFACTS_ROOT / "arac_lite_final_3e6_fast_gate_cases",
            module.CASE_ROOT,
        )
        self.assertEqual(
            module.CASE_ROOT / "relation_action_audit",
            module.CASE_RELATION_AUDIT_ROOT,
        )

        module.configure_artifact_paths(module.DEFAULT_ARTIFACT_PREFIX)

    def test_longest_first_schedule_prioritizes_slowest_problem_families(self):
        module = load_gate_module()
        tasks = [
            {"problem": "E4", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "S4", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "R6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "E6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
        ]

        scheduled = module.schedule_tasks(tasks, "longest-first")

        self.assertEqual(["E6", "R6", "E4", "S4"], [task["problem"] for task in scheduled])

    def test_balanced_schedule_interleaves_problem_families(self):
        module = load_gate_module()
        tasks = [
            {"problem": "R6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "R6", "seed": 2, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "S6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "S6", "seed": 2, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "S4", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "S4", "seed": 2, "tfes": 3000000, "method": module.FINAL_METHOD},
        ]

        scheduled = module.schedule_tasks(tasks, "balanced")

        self.assertEqual(
            ["R6", "S6", "S4", "R6", "S6", "S4"],
            [task["problem"] for task in scheduled],
        )

    def test_fast_gate_run_case_uses_lightweight_metadata_when_relation_rows_are_absent(self):
        module = load_gate_module()
        calls = {}

        def fake_run_one_case(method, config, problem, seed, tfes):
            calls["trace_flags"] = {
                "save_diagnostics": config.save_diagnostics,
                "save_shared_variable_trace": config.save_shared_variable_trace,
                "validation_trace_enabled": config.validation_trace_enabled,
            }
            return (
                {
                    "problem": problem,
                    "seed": seed,
                    "tfes": tfes,
                    "method": method,
                    "coordination_mode": module.v01.hcc_es.resolve_shared_variable_coordination_mode(config),
                    "cc_pass_group_fes": config.cc_pass_group_fes,
                    "cc_pass_count": 3,
                    "best_error": 1.0,
                    "final_error": 2.0,
                    "fe_used": 3000000,
                    "runtime": 10.0,
                    "fusion_count": 0,
                    "freeze_count": 0,
                    "disable_count": 0,
                    "fusion_validation_attempt_count": 0,
                    "fusion_validation_accept_count": 0,
                    "fusion_validation_accept_rate": 0.0,
                    "validation_attempt_count": 0,
                    "validation_accept_count": 0,
                    "validation_reject_count": 0,
                    "validation_accept_rate": 0.0,
                    "validation_extra_fe_ratio": 0.0,
                    "relation_history_size": 4,
                    "status": "ok",
                },
                [],
            )

        task = {
            "problem": "S6",
            "seed": 1,
            "tfes": 3000000,
            "method": module.FINAL_METHOD,
            "cc_pass_group_fes": 20,
            "fast_gate": True,
        }
        old_run_one_case = module.v01.run_one_case
        old_case_root = module.CASE_ROOT
        old_case_run_root = module.CASE_RUN_DETAILS_ROOT
        old_case_relation_root = module.CASE_RELATION_AUDIT_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                module.v01.run_one_case = fake_run_one_case
                module.CASE_ROOT = Path(tmpdir)
                module.CASE_RUN_DETAILS_ROOT = Path(tmpdir) / "run_details"
                module.CASE_RELATION_AUDIT_ROOT = Path(tmpdir) / "relation_action_audit"

                row = module.run_case_task(task)
            finally:
                module.v01.run_one_case = old_run_one_case
                module.CASE_ROOT = old_case_root
                module.CASE_RUN_DETAILS_ROOT = old_case_run_root
                module.CASE_RELATION_AUDIT_ROOT = old_case_relation_root

        self.assertEqual("ok", row["status"])
        self.assertEqual(3000000, row["total_FEs"])
        self.assertEqual(0.0, row["extra_FE_ratio"])
        self.assertFalse(calls["trace_flags"]["save_diagnostics"])
        self.assertFalse(calls["trace_flags"]["save_shared_variable_trace"])
        self.assertTrue(calls["trace_flags"]["validation_trace_enabled"])

    def test_run_tasks_does_not_return_relation_rows_to_parent(self):
        module = load_gate_module()
        tasks = [
            {"problem": "S6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
            {"problem": "R6", "seed": 1, "tfes": 3000000, "method": module.FINAL_METHOD},
        ]

        def fake_run(task):
            return {
                "problem": task["problem"],
                "seed": task["seed"],
                "tfes": task["tfes"],
                "method": task["method"],
                "status": "ok",
            }

        rows = module.run_tasks(tasks, workers=1, run_case=fake_run)

        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["status"] == "ok" for row in rows))


class AracLiteFinalGateMetricsTests(unittest.TestCase):
    def test_load_paper_reference_csv_accepts_expected_columns(self):
        module = load_gate_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "refs.csv"
            path.write_text(
                "problem,paper_HCC_mean,paper_HCC_std\n"
                "E4,2.26E7,3.04E5\n"
                "S6,6.65E4,7.36E2\n",
                encoding="utf-8",
            )

            refs = module.load_paper_references(path)

        self.assertAlmostEqual(2.26e7, refs["E4"]["paper_HCC_mean"])
        self.assertAlmostEqual(3.04e5, refs["E4"]["paper_HCC_std"])
        self.assertAlmostEqual(6.65e4, refs["S6"]["paper_HCC_mean"])

    def test_summary_uses_missing_paper_reference_without_fabricating_gap(self):
        module = load_gate_module()
        run_rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "best_error": 80.0,
                "final_error": 90.0,
                "runtime": 10.0,
                "fe_used": 3000000,
                "validation_extra_fe_ratio": 0.001,
                "fusion_count": 2,
                "freeze_count": 3,
                "disable_count": 4,
                "validation_accept_rate": 0.5,
                "cc_pass_count": 6,
                "relation_history_size": 7,
                "status": "ok",
            },
            {
                "problem": "S6",
                "seed": 2,
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "best_error": 100.0,
                "final_error": 110.0,
                "runtime": 12.0,
                "fe_used": 3000000,
                "validation_extra_fe_ratio": 0.003,
                "fusion_count": 4,
                "freeze_count": 5,
                "disable_count": 6,
                "validation_accept_rate": 0.75,
                "cc_pass_count": 8,
                "relation_history_size": 9,
                "status": "ok",
            },
        ]

        rows = module.build_summary_rows(run_rows, paper_refs={})
        row = rows[0]

        self.assertEqual("S6", row["problem"])
        self.assertEqual(2, row["runs"])
        self.assertAlmostEqual(90.0, row["ARAC_best_mean"])
        self.assertAlmostEqual(10.0, row["ARAC_best_std"])
        self.assertAlmostEqual(90.0, row["ARAC_best_median"])
        self.assertEqual("", row["paper_HCC_mean"])
        self.assertEqual("", row["paper_HCC_std"])
        self.assertEqual("", row["gap_vs_paper_HCC"])
        self.assertEqual("reference_missing", row["improved_or_not"])
        self.assertAlmostEqual(0.002, row["extra_FE_ratio_mean"])
        self.assertEqual(6, row["Fusion_count"])
        self.assertEqual(8, row["Freeze_count"])
        self.assertEqual(10, row["Disable_count"])

    def test_summary_calculates_paper_gap_when_reference_is_present(self):
        module = load_gate_module()
        run_rows = [
            {
                "problem": "E6",
                "seed": 1,
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "best_error": 80.0,
                "final_error": 80.0,
                "runtime": 1.0,
                "fe_used": 3000000,
                "validation_extra_fe_ratio": 0.0,
                "fusion_count": 1,
                "freeze_count": 0,
                "disable_count": 0,
                "validation_accept_rate": 1.0,
                "cc_pass_count": 1,
                "relation_history_size": 1,
                "status": "ok",
            }
        ]

        rows = module.build_summary_rows(run_rows, paper_refs={"E6": {"paper_HCC_mean": 100.0, "paper_HCC_std": 5.0}})

        self.assertAlmostEqual(-0.2, rows[0]["gap_vs_paper_HCC"])
        self.assertEqual("improved", rows[0]["improved_or_not"])

    def test_probe_metrics_count_targeted_recovered_and_bad_probe(self):
        module = load_gate_module()
        relation_rows = [
            {
                "problem": "S6",
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_accepted": "True",
                "validation_delta": "3.0",
            },
            {
                "problem": "R6",
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_accepted": "False",
                "validation_delta": "-2.0",
            },
        ]
        summary_rows = [
            {"problem": "S6", "tfes": 3000000, "extra_FE_ratio_mean": 0.001},
            {"problem": "R6", "tfes": 3000000, "extra_FE_ratio_mean": 0.002},
        ]

        rows = module.attach_extra_fe_ratio(module.build_probe_metric_rows(relation_rows), summary_rows)
        by_problem = {row["problem"]: row for row in rows}

        self.assertEqual(1, by_problem["S6"]["targeted_probe_count"])
        self.assertEqual(1, by_problem["S6"]["matched_probe_count"])
        self.assertEqual(1, by_problem["S6"]["recovered_fusion_count"])
        self.assertEqual(0, by_problem["S6"]["bad_probe_count"])
        self.assertAlmostEqual(3.0, by_problem["S6"]["probe_delta_mean"])
        self.assertEqual(1, by_problem["R6"]["bad_probe_count"])
        self.assertAlmostEqual(0.002, by_problem["R6"]["extra_fe_ratio"])

    def test_action_and_probe_metrics_accept_streaming_iterators(self):
        module = load_gate_module()
        relation_rows = [
            {
                "problem": "S6",
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "action_candidate": "Fusion",
                "validation_attempted": "True",
                "validation_accepted": "True",
                "validation_delta": "2.0",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
            },
            {
                "problem": "S6",
                "tfes": 3000000,
                "method": module.FINAL_METHOD,
                "action_candidate": "Disable",
                "validation_attempted": "False",
                "validation_accepted": "False",
                "validation_delta": "-1.0",
            },
        ]

        action_rows = module.build_action_distribution_rows(iter(relation_rows))
        probe_rows = module.build_probe_metric_rows(iter(relation_rows))

        action_by_name = {row["action"]: row for row in action_rows}
        self.assertEqual(1, action_by_name["Fusion"]["action_count"])
        self.assertEqual(1, action_by_name["Disable"]["action_count"])
        self.assertAlmostEqual(0.5, action_by_name["Fusion"]["action_share"])
        self.assertEqual(1, probe_rows[0]["targeted_probe_count"])
        self.assertEqual(1, probe_rows[0]["matched_probe_count"])


if __name__ == "__main__":
    unittest.main()
