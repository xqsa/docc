import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
V08_SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_arac_lite_v0_8_mechanism_ablation.py"


def load_hcc_es_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    spec = importlib.util.spec_from_file_location("hcc_es_module_v08", SOURCE_ROOT / "HCC-ES.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_info_config():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from HCC.info_aware_nda import InfoAwareNDAConfig

    return InfoAwareNDAConfig


def load_v0_8_module():
    spec = importlib.util.spec_from_file_location("arac_lite_v0_8", V08_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive_history(mean_delta=1.0):
    return {
        "attempt_count": 3,
        "accept_count": 3,
        "delta_sum": float(mean_delta) * 3.0,
        "positive_delta_count": 3 if mean_delta > 0.0 else 0,
        "reject_streak": 3,
        "recent_attempts": [{"accepted": True, "delta": float(mean_delta)}] * 3,
    }


class AracLiteV08ConfigTests(unittest.TestCase):
    def test_random_same_budget_config_normalizes_budget_and_phase(self):
        InfoAwareNDAConfig = load_info_config()
        config = InfoAwareNDAConfig(
            arac_lite_random_probe_same_budget_enabled=True,
            arac_lite_random_probe_budget=2,
            arac_lite_random_probe_phase="bad",
            arac_lite_random_probe_seed=11,
        ).normalized()

        self.assertTrue(config.arac_lite_random_probe_same_budget_enabled)
        self.assertEqual(2, config.arac_lite_random_probe_budget)
        self.assertEqual("middle", config.arac_lite_random_probe_phase)
        self.assertEqual(11, config.arac_lite_random_probe_seed)


class AracLiteV08RandomProbeAlgorithmTests(unittest.TestCase):
    def test_random_same_budget_uses_exact_remaining_budget_and_records_state(self):
        hcc_es = load_hcc_es_module()
        best = np.zeros(5, dtype=float)
        proposal_rows = []
        for var_id in [1, 2, 3]:
            proposal_rows.extend(
                [
                    {
                        "var_id": var_id,
                        "proposal_value": 0.10 * var_id,
                        "delta": 1.0,
                        "fitness_before": 10.0,
                        "fitness_after": 9.0,
                        "group_id": 0,
                    },
                    {
                        "var_id": var_id,
                        "proposal_value": 0.50 * var_id,
                        "delta": 2.0,
                        "fitness_before": 10.0,
                        "fitness_after": 8.0,
                        "group_id": 1,
                    },
                ]
            )
        hypergraph = {"overlap_vars": [1, 2, 3], "var_to_groups": {1: [0, 1], 2: [0, 1], 3: [0, 1]}}
        random_state = {}

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(5, dtype=float) * 100.0,
            selective_max_proposal_std_ratio=0.001,
            arac_relation_history={1: positive_history(1.0), 2: positive_history(1.0), 3: positive_history(1.0)},
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=2,
            arac_lite_probe_every_n_pass=1,
            arac_lite_probe_max_per_pass=3,
            arac_lite_probe_min_phase="middle",
            arac_lite_targeted_probe_enabled=False,
            arac_lite_random_probe_same_budget_enabled=True,
            arac_lite_random_probe_budget=2,
            arac_lite_random_probe_phase="middle",
            arac_lite_random_probe_seed=123,
            arac_lite_random_probe_state=random_state,
        )

        probe_rows = [row for row in rows if row["arac_probe_candidate"]]
        self.assertEqual(2, len(probe_rows))
        self.assertEqual(2, random_state["used_by_phase"]["middle"])
        self.assertTrue(all(row["arac_probe_reason"] == "random_same_budget_probe_selected" for row in probe_rows))
        self.assertTrue(all(str(row["action_reason"]).startswith("random_same_budget_probe_fusion") for row in probe_rows))

        rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best,
            proposal_rows,
            hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.ones(5, dtype=float) * 100.0,
            selective_max_proposal_std_ratio=0.001,
            arac_relation_history={1: positive_history(1.0), 2: positive_history(1.0), 3: positive_history(1.0)},
            arac_lite_history_min_attempts=1,
            arac_lite_disable_reject_streak=1,
            arac_lite_probe_enabled=True,
            arac_lite_probe_phase="middle",
            arac_lite_probe_pass_id=3,
            arac_lite_probe_every_n_pass=1,
            arac_lite_probe_max_per_pass=3,
            arac_lite_probe_min_phase="middle",
            arac_lite_targeted_probe_enabled=False,
            arac_lite_random_probe_same_budget_enabled=True,
            arac_lite_random_probe_budget=2,
            arac_lite_random_probe_phase="middle",
            arac_lite_random_probe_seed=123,
            arac_lite_random_probe_state=random_state,
        )

        self.assertEqual(0, sum(1 for row in rows if row["arac_probe_candidate"]))
        self.assertEqual(2, random_state["used_by_phase"]["middle"])


class AracLiteV08GeneratorTests(unittest.TestCase):
    def test_method_matrix_reuses_v0_6_targeted_and_adds_random_same_budget(self):
        module = load_v0_8_module()

        methods = [name for name, _ in module.fixed_method_configs(cc_pass_group_fes=20, budget_by_case_phase={})]

        self.assertEqual(
            [
                "no-coordination",
                "validated-selective-conflict",
                "arac-lite-v0.1-disable-fast",
                "arac-lite-v0.6-targeted-probe",
                "arac-lite-v0.6-random-probe-same-budget",
            ],
            methods,
        )

    def test_targeted_probe_budget_counts_actual_probe_actions_by_case_phase(self):
        module = load_v0_8_module()
        rows = [
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 5000,
                "method": module.CANDIDATE_METHOD,
                "phase": "middle",
                "var_id": 1,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
            },
            {
                "problem": "S6",
                "seed": 1,
                "tfes": 5000,
                "method": module.CANDIDATE_METHOD,
                "phase": "middle",
                "var_id": 2,
                "action_candidate": "Disable",
                "arac_probe_candidate": "False",
                "arac_targeted_probe_signature_matched": "True",
            },
            {
                "problem": "R6",
                "seed": 2,
                "tfes": 10000,
                "method": module.CANDIDATE_METHOD,
                "arac_probe_phase": "middle",
                "var_id": 3,
                "action_candidate": "Fusion",
                "action_reason": "targeted_probe_fusion_large_update_damped",
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "False",
            },
        ]

        budget = module.targeted_probe_budget_by_case_phase(rows)

        self.assertEqual(1, budget[("S6", 1, 5000, "middle")])
        self.assertEqual(1, budget[("R6", 2, 10000, "middle")])
        self.assertNotIn(("S6", 1, 5000, ""), budget)

    def test_random_tasks_receive_per_case_phase_budget(self):
        module = load_v0_8_module()
        args = SimpleNamespace(problems=["S6"], seeds=[1], tfes=[5000], cc_pass_group_fes=20)
        budget = {("S6", 1, 5000, "middle"): 3}

        tasks = module.build_case_tasks(args, budget)
        random_task = next(task for task in tasks if task["method"] == module.RANDOM_METHOD)

        self.assertEqual(3, random_task["random_probe_budget"])
        self.assertEqual("middle", random_task["random_probe_phase"])

    def test_budget_filter_keeps_only_requested_cases(self):
        module = load_v0_8_module()
        args = SimpleNamespace(problems=["S6"], seeds=[1], tfes=[5000])
        budget = {
            ("S6", 1, 5000, "middle"): 3,
            ("S6", 2, 5000, "middle"): 4,
            ("R6", 1, 5000, "middle"): 5,
        }

        filtered = module.filter_budget_by_args(budget, args)

        self.assertEqual({("S6", 1, 5000, "middle"): 3}, filtered)

    def test_probe_metrics_are_generalized_for_targeted_and_random(self):
        module = load_v0_8_module()
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
                "arac_probe_candidate": "True",
                "arac_targeted_probe_candidate": "True",
                "validation_accepted": "True",
                "validation_delta": "5.0",
            },
            {
                "problem": "R6",
                "seed": 1,
                "tfes": 5000,
                "method": module.RANDOM_METHOD,
                "phase": "middle",
                "var_id": 8,
                "action_candidate": "Fusion",
                "action_reason": "random_same_budget_probe_fusion_candidate",
                "arac_probe_candidate": "True",
                "arac_probe_reason": "random_same_budget_probe_selected",
                "validation_accepted": "False",
                "validation_delta": "-2.0",
            },
        ]
        offline_candidates = [{"problem": "S6", "seed": 1, "tfes": 5000, "phase": "middle", "var_id": 42}]

        rows = module.build_probe_metric_rows(relation_rows, offline_candidates)
        by_key = {(row["problem"], row["method"]): row for row in rows}

        self.assertEqual(1, by_key[("S6", module.CANDIDATE_METHOD)]["probe_count"])
        self.assertEqual(1, by_key[("S6", module.CANDIDATE_METHOD)]["matched_probe_count"])
        self.assertEqual(1, by_key[("S6", module.CANDIDATE_METHOD)]["recovered_fusion_count"])
        self.assertEqual(1, by_key[("R6", module.RANDOM_METHOD)]["probe_count"])
        self.assertEqual(1, by_key[("R6", module.RANDOM_METHOD)]["bad_probe_count"])
        self.assertAlmostEqual(-2.0, by_key[("R6", module.RANDOM_METHOD)]["probe_delta_mean"])

    def test_delta_stress_variants_split_accept_only_from_accept_delta(self):
        module = load_v0_8_module()

        accept_only = module.accept_only_recovery_variant().overrides
        accept_delta = module.accept_delta_recovery_variant().overrides

        self.assertLess(accept_only["arac_lite_recovery_delta_threshold"], -1.0e200)
        self.assertEqual(0.0, accept_only["arac_lite_recovery_positive_delta_rate_threshold"])
        self.assertEqual(0.0, accept_delta["arac_lite_recovery_delta_threshold"])
        self.assertGreater(accept_delta["arac_lite_recovery_positive_delta_rate_threshold"], 0.0)

    def test_recovery_metrics_count_recovery_fusion_and_bad_r6_rows(self):
        module = load_v0_8_module()
        relation_rows = [
            {
                "problem": "R6",
                "tfes": 5000,
                "method": module.ACCEPT_ONLY_METHOD,
                "action_candidate": "Fusion",
                "action_reason": "recovery_fusion_supported",
                "arac_recovery_candidate": "True",
                "validation_delta": "-2.0",
                "validation_accepted": "False",
            },
            {
                "problem": "R6",
                "tfes": 5000,
                "method": module.ACCEPT_ONLY_METHOD,
                "action_candidate": "Fusion",
                "action_reason": "recovery_fusion_supported",
                "arac_recovery_candidate": "True",
                "validation_delta": "3.0",
                "validation_accepted": "True",
            },
            {
                "problem": "S6",
                "tfes": 5000,
                "method": module.ACCEPT_DELTA_METHOD,
                "action_candidate": "Disable",
                "arac_recovery_candidate": "False",
                "validation_delta": "0.0",
            },
        ]

        rows = module.build_recovery_metric_rows(relation_rows)
        by_key = {(row["problem"], row["method"]): row for row in rows}

        self.assertEqual(2, by_key[("R6", module.ACCEPT_ONLY_METHOD)]["recovery_fusion_count"])
        self.assertEqual(1, by_key[("R6", module.ACCEPT_ONLY_METHOD)]["bad_recovery_count"])
        self.assertAlmostEqual(0.5, by_key[("R6", module.ACCEPT_ONLY_METHOD)]["recovery_accept_rate"])


if __name__ == "__main__":
    unittest.main()
