import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
V07_SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_7_generalization.py"


def load_v0_7_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_7", V07_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AracLiteV07MethodMatrixTests(unittest.TestCase):
    def test_fixed_method_configs_freezes_v0_6_and_adds_delta_gate_ablation(self):
        module = load_v0_7_module()

        methods = [name for name, _ in module.fixed_method_configs(cc_pass_group_fes=20)]

        self.assertEqual(
            [
                "no-coordination",
                "validated-selective-conflict",
                "arac-lite-v0.1-disable-fast",
                "arac-lite-v0.6-targeted-probe",
                "arac-lite-v0.6-no-delta-hard-block",
            ],
            methods,
        )

    def test_no_delta_hard_block_only_relaxes_delta_gates(self):
        module = load_v0_7_module()

        candidate = module.targeted_probe_variant().overrides
        ablation = module.no_delta_hard_block_variant().overrides

        self.assertEqual(0.0, candidate["arac_lite_targeted_probe_min_relation_delta"])
        self.assertEqual(1.0, candidate["arac_lite_targeted_probe_min_positive_delta_rate"])
        self.assertLess(ablation["arac_lite_targeted_probe_min_relation_delta"], 0.0)
        self.assertEqual(0.0, ablation["arac_lite_targeted_probe_min_positive_delta_rate"])
        for key, value in candidate.items():
            if key in {
                "arac_lite_targeted_probe_min_relation_delta",
                "arac_lite_targeted_probe_min_positive_delta_rate",
            }:
                continue
            self.assertEqual(value, ablation[key], key)


class AracLiteV07MetricTests(unittest.TestCase):
    def test_build_case_tasks_and_completed_keys_support_resume(self):
        module = load_v0_7_module()
        args = SimpleNamespace(
            problems=["S6"],
            seeds=[1, 2],
            tfes=[5000],
            cc_pass_group_fes=20,
        )
        existing_rows = [
            {
                "problem": "S6",
                "seed": "1",
                "tfes": "5000",
                "method": module.CANDIDATE_METHOD,
                "status": "ok",
            },
            {
                "problem": "S6",
                "seed": "2",
                "tfes": "5000",
                "method": module.CANDIDATE_METHOD,
                "status": "error:RuntimeError:boom",
            },
        ]

        all_tasks = module.build_case_tasks(args)
        pending = module.filter_completed_tasks(all_tasks, module.completed_case_keys(existing_rows))

        self.assertEqual(10, len(all_tasks))
        self.assertEqual(9, len(pending))
        self.assertNotIn(("S6", 5000, 1, module.CANDIDATE_METHOD), {module.task_key(task) for task in pending})
        self.assertIn(("S6", 5000, 2, module.CANDIDATE_METHOD), {module.task_key(task) for task in pending})

    def test_run_sweep_uses_executor_when_workers_gt_one(self):
        module = load_v0_7_module()
        tasks = [
            {"method": "m1", "config": object(), "problem": "S6", "seed": 1, "tfes": 5000},
            {"method": "m2", "config": object(), "problem": "R6", "seed": 1, "tfes": 5000},
        ]
        calls = []

        def fake_run(task):
            calls.append(("run", task["method"]))
            return (
                {
                    "problem": task["problem"],
                    "seed": task["seed"],
                    "tfes": task["tfes"],
                    "method": task["method"],
                    "status": "ok",
                },
                [{"problem": task["problem"], "method": task["method"]}],
            )

        class FakeExecutor:
            def __init__(self, max_workers):
                calls.append(("executor", max_workers))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, items):
                for item in items:
                    yield func(item)

        run_rows, relation_rows = module.run_tasks(tasks, workers=2, run_case=fake_run, executor_cls=FakeExecutor)

        self.assertIn(("executor", 2), calls)
        self.assertEqual(2, len(run_rows))
        self.assertEqual(2, len(relation_rows))

    def test_summary_rows_include_distribution_and_baseline_gaps(self):
        module = load_v0_7_module()
        run_rows = [
            {
                "problem": "S6",
                "tfes": 5000,
                "seed": 1,
                "method": "no-coordination",
                "best_error": 100.0,
                "final_error": 100.0,
                "fusion_count": 0,
                "freeze_count": 0,
                "disable_count": 0,
                "fusion_validation_accept_rate": 0.0,
                "validation_accept_rate": 0.0,
                "validation_extra_fe_ratio": 0.0,
                "cc_pass_count": 2,
                "relation_history_size": 0,
            },
            {
                "problem": "S6",
                "tfes": 5000,
                "seed": 1,
                "method": "validated-selective-conflict",
                "best_error": 90.0,
                "final_error": 90.0,
                "fusion_count": 2,
                "freeze_count": 1,
                "disable_count": 0,
                "fusion_validation_accept_rate": 1.0,
                "validation_accept_rate": 1.0,
                "validation_extra_fe_ratio": 0.005,
                "cc_pass_count": 2,
                "relation_history_size": 1,
            },
            {
                "problem": "S6",
                "tfes": 5000,
                "seed": 1,
                "method": "arac-lite-v0.1-disable-fast",
                "best_error": 80.0,
                "final_error": 80.0,
                "fusion_count": 1,
                "freeze_count": 1,
                "disable_count": 3,
                "fusion_validation_accept_rate": 1.0,
                "validation_accept_rate": 0.5,
                "validation_extra_fe_ratio": 0.003,
                "cc_pass_count": 2,
                "relation_history_size": 3,
            },
            {
                "problem": "S6",
                "tfes": 5000,
                "seed": 1,
                "method": module.CANDIDATE_METHOD,
                "best_error": 70.0,
                "final_error": 70.0,
                "fusion_count": 2,
                "freeze_count": 1,
                "disable_count": 2,
                "fusion_validation_accept_rate": 1.0,
                "validation_accept_rate": 0.75,
                "validation_extra_fe_ratio": 0.004,
                "cc_pass_count": 2,
                "relation_history_size": 3,
            },
        ]

        rows = module.summarize_runs(run_rows)
        candidate = next(row for row in rows if row["method"] == module.CANDIDATE_METHOD)

        self.assertAlmostEqual(-0.125, candidate["gap_vs_disable_fast"])
        self.assertAlmostEqual(-0.2222222222222222, candidate["gap_vs_validated"])
        self.assertAlmostEqual(-0.3, candidate["gap_vs_no_coordination"])
        self.assertEqual(2, candidate["fusion_count"])
        self.assertEqual(1, candidate["freeze_count"])
        self.assertEqual(2, candidate["disable_count"])

    def test_robustness_rows_are_paired_by_seed_and_include_iqr_and_worst_gap(self):
        module = load_v0_7_module()
        run_rows = []
        for seed, candidate_error, baseline_error in [
            (1, 90.0, 100.0),
            (2, 110.0, 100.0),
            (3, 100.0, 100.0),
        ]:
            run_rows.append(
                {
                    "problem": "R6",
                    "tfes": 5000,
                    "seed": seed,
                    "method": module.CANDIDATE_METHOD,
                    "best_error": candidate_error,
                }
            )
            run_rows.append(
                {
                    "problem": "R6",
                    "tfes": 5000,
                    "seed": seed,
                    "method": "arac-lite-v0.1-disable-fast",
                    "best_error": baseline_error,
                }
            )

        rows = module.build_robustness_rows(run_rows)
        row = next(
            item
            for item in rows
            if item["problem"] == "R6"
            and item["tfes"] == 5000
            and item["baseline"] == "arac-lite-v0.1-disable-fast"
        )

        self.assertEqual(3, row["paired_runs"])
        self.assertEqual(1, row["wins"])
        self.assertEqual(1, row["losses"])
        self.assertEqual(1, row["ties"])
        self.assertAlmostEqual(0.0, row["gap_median"])
        self.assertAlmostEqual(0.1, row["worst_case_gap"])
        self.assertIn("gap_iqr", row)

    def test_targeted_probe_metrics_include_matched_bad_recovered_and_extra_fe(self):
        module = load_v0_7_module()
        relation_rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 5000,
                "method": module.CANDIDATE_METHOD,
                "phase": "middle",
                "var_id": 42,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_accepted": "True",
                "validation_delta": "5.0",
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 5000,
                "method": module.CANDIDATE_METHOD,
                "phase": "middle",
                "var_id": 8,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_accepted": "False",
                "validation_delta": "-2.0",
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 5000,
                "method": module.ABLATION_METHOD,
                "phase": "middle",
                "var_id": 77,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_targeted_probe_candidate": "True",
                "arac_targeted_probe_signature_matched": "True",
                "validation_accepted": "True",
                "validation_delta": "3.0",
            },
        ]
        summary_rows = [
            {"problem": "S6", "tfes": 5000, "method": module.CANDIDATE_METHOD, "validation_extra_fe_ratio": 0.004},
            {"problem": "R6", "tfes": 5000, "method": module.CANDIDATE_METHOD, "validation_extra_fe_ratio": 0.002},
            {"problem": "S6", "tfes": 5000, "method": module.ABLATION_METHOD, "validation_extra_fe_ratio": 0.005},
        ]
        offline_candidates = [{"problem": "S6", "seed": 1, "tfes": 5000, "phase": "middle", "var_id": 42}]

        rows = module.attach_extra_fe_ratio(
            module.build_probe_metric_rows(relation_rows, offline_candidates),
            summary_rows,
        )
        by_key = {(row["problem"], row["method"]): row for row in rows}

        self.assertEqual(1, by_key[("S6", module.CANDIDATE_METHOD)]["targeted_probe_count"])
        self.assertEqual(1, by_key[("S6", module.CANDIDATE_METHOD)]["matched_probe_count"])
        self.assertEqual(1, by_key[("S6", module.CANDIDATE_METHOD)]["recovered_fusion_count"])
        self.assertEqual(1, by_key[("R6", module.CANDIDATE_METHOD)]["bad_probe_count"])
        self.assertAlmostEqual(0.004, by_key[("S6", module.CANDIDATE_METHOD)]["extra_fe_ratio"])
        self.assertEqual(1, by_key[("S6", module.ABLATION_METHOD)]["targeted_probe_count"])


class AracLiteV07ReportTests(unittest.TestCase):
    def test_report_contains_freeze_claim_and_ablation_section(self):
        module = load_v0_7_module()
        args = SimpleNamespace(problems=["S6"], seeds=[1], tfes=[5000], cc_pass_group_fes=20)

        report = module.build_report_text(
            run_rows=[],
            summary_rows=[],
            robustness_rows=[],
            rank_rows=[],
            probe_metric_rows=[],
            args=args,
        )

        self.assertIn("冻结 V0.6", report)
        self.assertIn("no-delta-hard-block", report)
        self.assertIn("paired win/loss/tie", report)


if __name__ == "__main__":
    unittest.main()
