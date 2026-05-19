import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_hcc_es_module():
    workspace_root = Path(__file__).resolve().parents[1]
    source_root = workspace_root / "HCC_SRC"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module_path = source_root / "HCC-ES.py"
    spec = importlib.util.spec_from_file_location("hcc_es_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_info_aware_module():
    workspace_root = Path(__file__).resolve().parents[1]
    source_root = workspace_root / "HCC_SRC"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import HCC.info_aware_nda as info_aware_nda_module

    return info_aware_nda_module


class SyntheticSphere:
    def __init__(self):
        self.fitness_record = []

    def __call__(self, x_batch):
        x_array = np.asarray(x_batch, dtype=float)
        values = np.sum(np.square(x_array), axis=-1)
        self.fitness_record.extend(np.asarray(values, dtype=float).reshape(-1).tolist())
        return values


def build_test_cc_prior(info_module, priorities):
    priorities = np.asarray(priorities, dtype=float)
    dimension = 8
    return info_module.CCPrior(
        group_priority=priorities,
        variable_contribution=np.ones(dimension, dtype=float),
        variable_stability=np.ones(dimension, dtype=float),
        variable_direction=np.zeros(dimension, dtype=int),
        conflict_prior=np.zeros(dimension, dtype=float),
        overlap_degree=np.ones(dimension, dtype=float),
        diagnostics={},
    )


class InfoAwareNdaTests(unittest.TestCase):
    def test_priority_mode_normalization_accepts_new_modes_and_degrades_unknown_to_off(self):
        info_module = load_info_aware_module()

        self.assertEqual("diagnostic_only", info_module.InfoAwareNDAConfig().normalized().priority_mode)
        self.assertEqual(
            "diagnostic_only",
            info_module.InfoAwareNDAConfig(priority_mode="diagnostic_only").normalized().priority_mode,
        )
        self.assertEqual(
            "sort_dangerous_ablation",
            info_module.InfoAwareNDAConfig(priority_mode="sort_dangerous_ablation").normalized().priority_mode,
        )
        self.assertEqual("off", info_module.InfoAwareNDAConfig(priority_mode="sort").normalized().priority_mode)
        self.assertEqual(
            "off",
            info_module.InfoAwareNDAConfig(priority_mode="budget_original_order").normalized().priority_mode,
        )
        self.assertEqual("off", info_module.InfoAwareNDAConfig(priority_mode="mystery-mode").normalized().priority_mode)

    def test_compute_variable_contribution_prefers_larger_weighted_steps(self):
        info_module = load_info_aware_module()
        success_steps = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        success_gains = np.array([1.0, 3.0])

        contribution = info_module.compute_variable_contribution(success_steps, success_gains)

        self.assertEqual((3,), contribution.shape)
        self.assertTrue(np.all(np.isfinite(contribution)))
        self.assertGreater(contribution[1], contribution[0])
        self.assertGreater(contribution[0], contribution[2])

    def test_compute_variable_stability_matches_direction_consistency(self):
        info_module = load_info_aware_module()
        stable_steps = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )
        stable_gains = np.array([1.0, 1.0])
        stability, direction = info_module.compute_variable_stability(stable_steps, stable_gains)

        self.assertAlmostEqual(1.0, stability[0])
        self.assertEqual(1, direction[0])

        oscillating_steps = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        )
        oscillating_gains = np.array([1.0, 1.0])
        oscillating_stability, oscillating_direction = info_module.compute_variable_stability(
            oscillating_steps,
            oscillating_gains,
        )

        self.assertAlmostEqual(0.0, oscillating_stability[0], places=6)
        self.assertEqual(0, oscillating_direction[0])

    def test_compute_variable_stability_without_success_steps_returns_safe_default(self):
        info_module = load_info_aware_module()
        stability, direction = info_module.compute_variable_stability(
            np.empty((0, 4)),
            np.empty((0,)),
        )

        np.testing.assert_allclose(stability, np.ones(4))
        np.testing.assert_array_equal(direction, np.zeros(4, dtype=int))

    def test_build_cc_prior_tracks_overlap_degree_and_conflict_prior(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        grouping_result = [
            [0, 1, 2],
            [2, 3],
            [1, 4],
        ]
        hypergraph = hcc_es.build_overlap_hypergraph(grouping_result)
        contribution = np.array([1.0, 2.0, 3.0, 1.0, 1.0])
        stability = np.array([1.0, 0.2, 0.5, 1.0, 1.0])
        direction = np.array([0, 1, -1, 0, 0])
        config = info_module.InfoAwareNDAConfig(enable=True)

        cc_prior = info_module.build_cc_prior(
            grouping_result,
            dimension=5,
            overlap_hypergraph=hypergraph,
            var_contribution=contribution,
            var_stability=stability,
            var_direction=direction,
            config=config,
        )

        self.assertEqual((3,), cc_prior.group_priority.shape)
        self.assertTrue(np.all(np.isfinite(cc_prior.group_priority)))
        self.assertEqual(2.0, cc_prior.overlap_degree[1])
        self.assertEqual(2.0, cc_prior.overlap_degree[2])
        self.assertAlmostEqual(0.8, cc_prior.conflict_prior[1])
        self.assertAlmostEqual(0.5, cc_prior.conflict_prior[2])
        self.assertAlmostEqual(0.0, cc_prior.conflict_prior[0])

    def test_early_switch_detects_plateau_but_not_continued_improvement(self):
        info_module = load_info_aware_module()
        config = info_module.InfoAwareNDAConfig(
            enable=True,
            min_nda_fe_ratio=0.1,
            max_nda_fe_ratio=0.5,
            window_size=2,
            patience=2,
            eps_improve=1e-4,
            eps_center_shift=1e-4,
        )

        state_1 = info_module.evaluate_early_switch(
            best_history=[10.0, 10.0, 10.0],
            center_shift_history=[0.0, 0.0, 0.0],
            diversity_history=[0.1, 0.1, 0.1],
            fe_used=20,
            total_max_fes=100,
            original_max_nda_fes=50,
            config=config,
            stagnant_counter=0,
        )
        state_2 = info_module.evaluate_early_switch(
            best_history=[10.0, 10.0, 10.0],
            center_shift_history=[0.0, 0.0, 0.0],
            diversity_history=[0.1, 0.1, 0.1],
            fe_used=22,
            total_max_fes=100,
            original_max_nda_fes=50,
            config=config,
            stagnant_counter=state_1["stagnant_counter"],
        )

        self.assertFalse(state_1["should_stop"])
        self.assertTrue(state_2["should_stop"])
        self.assertEqual("early_stagnation", state_2["reason"])

        improving_state = info_module.evaluate_early_switch(
            best_history=[10.0, 9.0, 8.0],
            center_shift_history=[0.0, 0.01, 0.02],
            diversity_history=[0.1, 0.1, 0.1],
            fe_used=20,
            total_max_fes=100,
            original_max_nda_fes=50,
            config=config,
            stagnant_counter=0,
        )

        self.assertFalse(improving_state["should_stop"])
        self.assertEqual(0, improving_state["stagnant_counter"])

    def test_diagnostic_only_keeps_uniform_budgets_and_original_order(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        grouping_result = [[0, 1], [1, 2], [2, 3], [3, 4]]
        adjacent_overlaps = [[1], [2], [3]]
        cc_prior = build_test_cc_prior(info_module, [0.6, 1.4, 1.0, 1.2])
        config = info_module.InfoAwareNDAConfig(enable=True, priority_mode="diagnostic_only")

        scheduled_groups, scheduled_adjacent_overlaps, budgets, execution_order, schedule_metadata = hcc_es._resolve_cc_schedule(
            grouping_result,
            adjacent_overlaps,
            remaining_fes=9,
            cc_prior=cc_prior,
            info_aware_config=config,
        )

        self.assertEqual(grouping_result, scheduled_groups)
        self.assertEqual(adjacent_overlaps, scheduled_adjacent_overlaps)
        self.assertEqual([0, 1, 2, 3], execution_order)
        self.assertEqual([3, 3, 3, 3], budgets)
        self.assertEqual("diagnostic_only", schedule_metadata["priority_mode_effective"])
        self.assertFalse(schedule_metadata["sort_dangerous_ablation_changed_order"])

    def test_sort_dangerous_ablation_reorders_groups_and_emits_overlap_warning(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        grouping_result = [[0, 1], [1, 2], [2, 3], [3, 4]]
        adjacent_overlaps = [[1], [2], [3]]
        cc_prior = build_test_cc_prior(info_module, [1.0, 4.0, 2.0, 3.0])
        config = info_module.InfoAwareNDAConfig(enable=True, priority_mode="sort_dangerous_ablation")

        scheduled_groups, _, _, execution_order, schedule_metadata = hcc_es._resolve_cc_schedule(
            grouping_result,
            adjacent_overlaps,
            remaining_fes=9,
            cc_prior=cc_prior,
            info_aware_config=config,
        )

        self.assertEqual([1, 3, 2, 0], execution_order)
        self.assertEqual(grouping_result[1], scheduled_groups[0])
        self.assertTrue(schedule_metadata["sort_dangerous_ablation_changed_order"])
        self.assertIn("sort_dangerous_ablation_changes_overlap_order", schedule_metadata["warnings"])

    def test_priority_audit_reports_positive_and_negative_rank_correlation(self):
        info_module = load_info_aware_module()

        positive_trace = [
            {"original_group_id": 0, "priority": 3.0, "actual_delta": 30.0},
            {"original_group_id": 1, "priority": 2.0, "actual_delta": 20.0},
            {"original_group_id": 2, "priority": 1.0, "actual_delta": 10.0},
        ]
        positive_audit = info_module.compute_priority_audit(positive_trace, topk=3)
        self.assertGreater(positive_audit["priority_delta_spearman"], 0.9)
        self.assertEqual(1.0, positive_audit["topk_priority_hit_rate"])

        negative_trace = [
            {"original_group_id": 0, "priority": 3.0, "actual_delta": 10.0},
            {"original_group_id": 1, "priority": 2.0, "actual_delta": 20.0},
            {"original_group_id": 2, "priority": 1.0, "actual_delta": 30.0},
        ]
        negative_audit = info_module.compute_priority_audit(negative_trace, topk=3)
        self.assertLess(negative_audit["priority_delta_spearman"], -0.9)

    def test_diagnostic_only_matches_early_switch_only_behavior_and_emits_trace(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        grouping_result = [
            list(range(0, 10)),
            list(range(8, 18)),
            list(range(16, 20)),
        ]
        adjacent_overlaps = hcc_es.compute_adjacent_overlaps_for_groups(grouping_result)
        info = {
            "dimension": 20,
            "lower": -5.0,
            "upper": 5.0,
        }
        best_individual = np.full(info["dimension"], 3.0)

        common_kwargs = {
            "enable": True,
            "min_nda_fe_ratio": 0.05,
            "max_nda_fe_ratio": 0.6,
            "window_size": 2,
            "patience": 1,
            "eps_improve": 0.05,
            "eps_center_shift": 0.05,
        }
        early_switch_config = info_module.InfoAwareNDAConfig(
            enable_trajectory_distill=False,
            enable_group_priority=False,
            priority_mode="off",
            **common_kwargs,
        )
        diagnostic_only_config = info_module.InfoAwareNDAConfig(
            enable_trajectory_distill=True,
            enable_group_priority=True,
            priority_mode="diagnostic_only",
            **common_kwargs,
        )

        early_fun = SyntheticSphere()
        early_curve, _, _, early_metadata = hcc_es.run_hcc_core(
            early_fun,
            output_path="",
            best_individual=best_individual,
            max_fes=120,
            grouping_result=grouping_result,
            info=info,
            adjacent_overlapping_elements=adjacent_overlaps,
            seed=7,
            info_aware_config=early_switch_config,
            return_metadata=True,
        )
        diagnostic_fun = SyntheticSphere()
        diagnostic_curve, _, _, diagnostic_metadata = hcc_es.run_hcc_core(
            diagnostic_fun,
            output_path="",
            best_individual=best_individual,
            max_fes=120,
            grouping_result=grouping_result,
            info=info,
            adjacent_overlapping_elements=adjacent_overlaps,
            seed=7,
            info_aware_config=diagnostic_only_config,
            return_metadata=True,
        )

        np.testing.assert_allclose(diagnostic_curve, early_curve)
        diagnostic_payload = diagnostic_metadata["info_aware_diagnostics"]
        self.assertEqual("diagnostic_only", diagnostic_payload["priority_mode_effective"])
        self.assertFalse(diagnostic_payload["sort_dangerous_ablation_changed_order"])
        self.assertEqual([0, 1, 2], diagnostic_payload["execution_order"])
        self.assertGreater(diagnostic_payload["group_trace_count"], 0)
        self.assertEqual(diagnostic_payload["group_trace_count"], len(diagnostic_metadata["group_trace_rows"]))
        self.assertEqual(
            early_metadata["info_aware_diagnostics"]["early_switch_reason"],
            diagnostic_payload["early_switch_reason"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics_path = Path(tmp_dir) / "info_aware_nda_diagnostics.json"
            info_module.save_info_aware_diagnostics(
                diagnostics_path,
                diagnostic_payload,
            )
            self.assertTrue(diagnostics_path.exists())


if __name__ == "__main__":
    unittest.main()
