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
        self.assertEqual("original", info_module.InfoAwareNDAConfig().normalized().overlap_blend_mode)
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
        self.assertEqual(
            "safe_conflict",
            info_module.InfoAwareNDAConfig(overlap_blend_mode="safe_conflict").normalized().overlap_blend_mode,
        )
        self.assertEqual(
            "equation8_correct",
            info_module.InfoAwareNDAConfig(overlap_blend_mode="equation8_correct").normalized().overlap_blend_mode,
        )
        self.assertEqual(
            "original",
            info_module.InfoAwareNDAConfig(overlap_blend_mode="mystery-blend").normalized().overlap_blend_mode,
        )
        self.assertEqual("rddsm", info_module.InfoAwareNDAConfig().normalized().group_order_mode)
        self.assertEqual(
            "aob_topology",
            info_module.InfoAwareNDAConfig(group_order_mode="aob_topology").normalized().group_order_mode,
        )
        self.assertEqual("rddsm", info_module.InfoAwareNDAConfig(group_order_mode="mystery-order").normalized().group_order_mode)
        self.assertEqual("match_execution", info_module.InfoAwareNDAConfig().normalized().coordination_order_mode)
        self.assertEqual(
            "rddsm",
            info_module.InfoAwareNDAConfig(coordination_order_mode="rddsm").normalized().coordination_order_mode,
        )
        self.assertEqual(
            "aob_topology",
            info_module.InfoAwareNDAConfig(coordination_order_mode="aob_topology").normalized().coordination_order_mode,
        )
        self.assertEqual(
            "match_execution",
            info_module.InfoAwareNDAConfig(coordination_order_mode="mystery-order").normalized().coordination_order_mode,
        )
        self.assertEqual("ephemeral", info_module.InfoAwareNDAConfig().normalized().optimizer_state_mode)
        self.assertEqual(
            "persistent_mean",
            info_module.InfoAwareNDAConfig(optimizer_state_mode="persistent_mean").normalized().optimizer_state_mode,
        )
        self.assertEqual(
            "ephemeral",
            info_module.InfoAwareNDAConfig(optimizer_state_mode="persistent_covariance").normalized().optimizer_state_mode,
        )
        self.assertEqual(
            "hypergraph_pass_end",
            info_module.InfoAwareNDAConfig(shared_variable_coordination_mode="hypergraph_pass_end").normalized().shared_variable_coordination_mode,
        )
        self.assertEqual(
            "selective_hypergraph_pass_end",
            info_module.InfoAwareNDAConfig(shared_variable_coordination_mode="selective_hypergraph_pass_end").normalized().shared_variable_coordination_mode,
        )
        self.assertEqual(
            "no_coordination",
            info_module.InfoAwareNDAConfig(shared_variable_coordination_mode="no_coordination").normalized().shared_variable_coordination_mode,
        )
        self.assertEqual(
            "adjacent",
            info_module.InfoAwareNDAConfig(shared_variable_coordination_mode="mystery-mode").normalized().shared_variable_coordination_mode,
        )
        normalized_owner_config = info_module.InfoAwareNDAConfig(
            shared_variable_owner_soft_eta=3.0,
            shared_variable_owner_min_delta_ratio=-1.0,
        ).normalized()
        self.assertAlmostEqual(1.0, normalized_owner_config.shared_variable_owner_soft_eta)
        self.assertAlmostEqual(0.0, normalized_owner_config.shared_variable_owner_min_delta_ratio)

    def test_arac_lite_config_normalizes_rule_mode_and_thresholds(self):
        info_module = load_info_aware_module()

        normalized = info_module.InfoAwareNDAConfig(
            shared_variable_coordination_mode="arac_lite_rule",
            arac_lite_history_min_attempts=0,
            arac_lite_disable_accept_rate_threshold=2.0,
            arac_lite_disable_mean_delta_threshold=-3.0,
            arac_lite_disable_reject_streak=0,
        ).normalized()

        self.assertEqual("arac_lite_rule", normalized.shared_variable_coordination_mode)
        self.assertEqual(1, normalized.arac_lite_history_min_attempts)
        self.assertAlmostEqual(1.0, normalized.arac_lite_disable_accept_rate_threshold)
        self.assertAlmostEqual(-3.0, normalized.arac_lite_disable_mean_delta_threshold)
        self.assertEqual(1, normalized.arac_lite_disable_reject_streak)

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

    def test_owner_soft_effect_audit_tracks_best_hit_overwrite_and_top_vars(self):
        hcc_es = load_hcc_es_module()

        proposal_rows = [
            {"cycle_id": 2, "var_id": 3, "proposal_value": 4.0},
            {"cycle_id": 2, "var_id": 7, "proposal_value": -2.0},
        ]
        fusion_rows = [
            {
                "cycle_id": 0,
                "var_id": 3,
                "ownership_mode": "owner_soft",
                "was_updated": True,
                "applied_value": 2.0,
                "update_magnitude": 0.2,
                "post_coordination_best_improved": True,
            },
            {
                "cycle_id": 0,
                "var_id": 7,
                "ownership_mode": "owner_soft",
                "was_updated": True,
                "applied_value": -2.0,
                "update_magnitude": 0.4,
                "post_coordination_best_improved": False,
            },
            {
                "cycle_id": 0,
                "var_id": 9,
                "ownership_mode": "multi_support_fusion",
                "was_updated": True,
                "applied_value": 1.0,
                "update_magnitude": 0.1,
                "post_coordination_best_improved": False,
            },
        ]

        annotated_rows, audit = hcc_es.build_owner_soft_effect_audit(
            proposal_rows,
            fusion_rows,
            top_k=5,
        )

        self.assertEqual(2, audit["owner_soft_update_count"])
        self.assertAlmostEqual(0.3, audit["owner_soft_update_magnitude_mean"])
        self.assertEqual(2, audit["owner_soft_unique_var_count"])
        self.assertEqual(1, audit["owner_soft_followed_by_best_improvement_count"])
        self.assertEqual(1, audit["owner_soft_overwritten_count"])
        self.assertEqual(3, annotated_rows[0]["var_id"])
        self.assertTrue(annotated_rows[0]["owner_soft_followed_by_best_improvement"])
        self.assertTrue(annotated_rows[0]["owner_soft_overwritten"])
        self.assertFalse(annotated_rows[1]["owner_soft_followed_by_best_improvement"])
        self.assertFalse(annotated_rows[1]["owner_soft_overwritten"])
        self.assertEqual(3, audit["owner_soft_top_vars"][0]["var_id"])
        self.assertEqual(1, audit["owner_soft_top_vars"][0]["best_improvement_count"])
        self.assertEqual(1, audit["owner_soft_top_vars"][0]["overwritten_count"])

    def test_owner_soft_effect_audit_returns_safe_defaults_without_updates(self):
        hcc_es = load_hcc_es_module()

        annotated_rows, audit = hcc_es.build_owner_soft_effect_audit([], [], top_k=3)

        self.assertEqual([], annotated_rows)
        self.assertEqual(0, audit["owner_soft_update_count"])
        self.assertAlmostEqual(0.0, audit["owner_soft_update_magnitude_mean"])
        self.assertEqual(0, audit["owner_soft_unique_var_count"])
        self.assertEqual(0, audit["owner_soft_followed_by_best_improvement_count"])
        self.assertEqual(0, audit["owner_soft_overwritten_count"])
        self.assertEqual([], audit["owner_soft_top_vars"])

    def test_validation_config_normalization_accepts_pass_end_and_caps_ratio(self):
        info_module = load_info_aware_module()

        normalized = info_module.InfoAwareNDAConfig(
            enable_validated_coordination=True,
            validation_mode="pass_end",
            validation_accept_eps=-1.0,
            validation_max_extra_fe_ratio=3.0,
        ).normalized()

        self.assertTrue(normalized.enable_validated_coordination)
        self.assertEqual("pass_end", normalized.validation_mode)
        self.assertAlmostEqual(0.0, normalized.validation_accept_eps)
        self.assertAlmostEqual(1.0, normalized.validation_max_extra_fe_ratio)
        self.assertEqual(
            "off",
            info_module.InfoAwareNDAConfig(validation_mode="mystery-mode").normalized().validation_mode,
        )

    def test_finalize_pass_end_coordination_accepts_rejects_and_respects_fe_cap(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()

        accept_fun = SyntheticSphere()
        accept_config = info_module.InfoAwareNDAConfig(
            enable_validated_coordination=True,
            validation_mode="pass_end",
            validation_accept_eps=0.0,
            validation_max_extra_fe_ratio=0.2,
        )
        accept_result = hcc_es.finalize_pass_end_coordination(
            fun=accept_fun,
            pre_coord_individual=np.array([2.0, 0.0]),
            pre_coord_fitness=4.0,
            coordinated_candidate=np.array([1.0, 0.0]),
            cycle_fusion_rows=[
                {
                    "positive_proposal_count": 1,
                    "ownership_mode": "owner_soft",
                    "was_updated": True,
                    "applied_update": True,
                    "update_magnitude": 1.0,
                    "conflict_prior": 0.25,
                }
            ],
            cycle_id=3,
            sum_fes=5,
            max_fes=20,
            validation_fes_used=0,
            info_aware_config=accept_config,
        )
        self.assertTrue(accept_result["validation_trace_row"]["validation_attempted"])
        self.assertTrue(accept_result["validation_trace_row"]["validation_accepted"])
        self.assertEqual(6, accept_result["sum_fes"])
        self.assertEqual(1, accept_result["validation_fes_used"])
        np.testing.assert_allclose(np.array([1.0, 0.0]), accept_result["best_individual"])
        self.assertAlmostEqual(1.0, accept_result["current_best_fitness"])
        self.assertTrue(accept_result["cycle_fusion_rows"][0]["validation_accepted"])

        reject_fun = SyntheticSphere()
        reject_config = info_module.InfoAwareNDAConfig(
            enable_validated_coordination=True,
            validation_mode="pass_end",
            validation_accept_eps=0.0,
            validation_max_extra_fe_ratio=0.2,
        )
        reject_result = hcc_es.finalize_pass_end_coordination(
            fun=reject_fun,
            pre_coord_individual=np.array([1.0, 0.0]),
            pre_coord_fitness=1.0,
            coordinated_candidate=np.array([2.0, 0.0]),
            cycle_fusion_rows=[
                {
                    "positive_proposal_count": 2,
                    "ownership_mode": "multi_support_fusion",
                    "was_updated": True,
                    "applied_update": True,
                    "update_magnitude": 1.0,
                    "conflict_prior": 0.5,
                }
            ],
            cycle_id=4,
            sum_fes=7,
            max_fes=20,
            validation_fes_used=0,
            info_aware_config=reject_config,
        )
        self.assertTrue(reject_result["validation_trace_row"]["validation_attempted"])
        self.assertFalse(reject_result["validation_trace_row"]["validation_accepted"])
        self.assertEqual("candidate_not_improved", reject_result["validation_trace_row"]["reject_reason"])
        self.assertEqual(8, reject_result["sum_fes"])
        self.assertEqual(1, reject_result["validation_fes_used"])
        np.testing.assert_allclose(np.array([1.0, 0.0]), reject_result["best_individual"])
        self.assertAlmostEqual(1.0, reject_result["current_best_fitness"])
        self.assertFalse(reject_result["cycle_fusion_rows"][0]["validation_accepted"])

        cap_fun = SyntheticSphere()
        cap_config = info_module.InfoAwareNDAConfig(
            enable_validated_coordination=True,
            validation_mode="pass_end",
            validation_accept_eps=0.0,
            validation_max_extra_fe_ratio=0.0,
        )
        cap_result = hcc_es.finalize_pass_end_coordination(
            fun=cap_fun,
            pre_coord_individual=np.array([1.5, 0.0]),
            pre_coord_fitness=2.25,
            coordinated_candidate=np.array([0.5, 0.0]),
            cycle_fusion_rows=[
                {
                    "positive_proposal_count": 1,
                    "ownership_mode": "owner_soft",
                    "was_updated": True,
                    "applied_update": True,
                    "update_magnitude": 1.0,
                    "conflict_prior": 0.1,
                }
            ],
            cycle_id=5,
            sum_fes=9,
            max_fes=20,
            validation_fes_used=0,
            info_aware_config=cap_config,
        )
        self.assertFalse(cap_result["validation_trace_row"]["validation_attempted"])
        self.assertFalse(cap_result["validation_trace_row"]["validation_accepted"])
        self.assertEqual("validation_fe_cap_reached", cap_result["validation_trace_row"]["reject_reason"])
        self.assertEqual(9, cap_result["sum_fes"])
        self.assertEqual(0, cap_result["validation_fes_used"])
        np.testing.assert_allclose(np.array([1.5, 0.0]), cap_result["best_individual"])
        self.assertAlmostEqual(2.25, cap_result["current_best_fitness"])

    def test_validated_coordination_summary_reports_acceptance_modes_and_fe_ratio(self):
        hcc_es = load_hcc_es_module()

        summary = hcc_es.summarize_validated_coordination_rows(
            [
                {
                    "validation_attempted": True,
                    "validation_accepted": True,
                    "fitness_delta": 0.3,
                    "validation_fe_used": 1,
                    "candidate_owner_soft_count": 2,
                    "candidate_multi_support_count": 1,
                },
                {
                    "validation_attempted": True,
                    "validation_accepted": False,
                    "fitness_delta": -0.1,
                    "validation_fe_used": 1,
                    "candidate_owner_soft_count": 1,
                    "candidate_multi_support_count": 3,
                },
                {
                    "validation_attempted": False,
                    "validation_accepted": False,
                    "fitness_delta": 0.0,
                    "validation_fe_used": 0,
                    "candidate_owner_soft_count": 4,
                    "candidate_multi_support_count": 5,
                },
            ],
            total_max_fes=100,
        )

        self.assertEqual(2, summary["validation_attempt_count"])
        self.assertEqual(1, summary["validation_accept_count"])
        self.assertEqual(1, summary["validation_reject_count"])
        self.assertAlmostEqual(0.5, summary["validation_accept_rate"])
        self.assertAlmostEqual(0.02, summary["validation_extra_fe_ratio"])
        self.assertEqual(2, summary["accepted_owner_soft_count"])
        self.assertEqual(1, summary["rejected_owner_soft_count"])
        self.assertEqual(1, summary["accepted_multi_support_count"])
        self.assertEqual(3, summary["rejected_multi_support_count"])
        self.assertAlmostEqual(0.3, summary["mean_accepted_fitness_delta"])
        self.assertAlmostEqual(-0.1, summary["mean_rejected_fitness_delta"])

    def test_coordination_selector_decides_off_without_probe_accepts_and_on_with_positive_signal(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()

        config = info_module.InfoAwareNDAConfig(
            enable_coordination_selector=True,
            selector_probe_passes=1,
            selector_min_attempts=1,
            selector_min_accepts=1,
            selector_accept_rate_threshold=0.3,
            selector_mean_delta_threshold=0.0,
        ).normalized()

        rejected_summary = hcc_es.summarize_coordination_selector_probe(
            [
                {"phase": "probe", "validation_attempted": True, "validation_accepted": False, "fitness_delta": -2.0},
                {"phase": "probe", "validation_attempted": True, "validation_accepted": False, "fitness_delta": -1.0},
            ],
            config,
        )
        rejected_decision = hcc_es.decide_coordination_selector_state(rejected_summary, config)

        self.assertEqual("off", rejected_decision["coordination_state"])
        self.assertEqual("probe_accept_count_below_min", rejected_decision["selector_reason"])
        self.assertEqual(2, rejected_summary["probe_attempt_count"])
        self.assertEqual(0, rejected_summary["probe_accept_count"])
        self.assertAlmostEqual(0.0, rejected_summary["probe_accept_rate"])

        accepted_summary = hcc_es.summarize_coordination_selector_probe(
            [
                {"phase": "probe", "validation_attempted": True, "validation_accepted": True, "fitness_delta": 3.0},
                {"phase": "probe", "validation_attempted": True, "validation_accepted": False, "fitness_delta": -1.0},
            ],
            config,
        )
        accepted_decision = hcc_es.decide_coordination_selector_state(accepted_summary, config)

        self.assertEqual("validated_on", accepted_decision["coordination_state"])
        self.assertEqual("probe_accept_rate_and_delta_passed", accepted_decision["selector_reason"])
        self.assertEqual(2, accepted_summary["probe_attempt_count"])
        self.assertEqual(1, accepted_summary["probe_accept_count"])
        self.assertAlmostEqual(0.5, accepted_summary["probe_accept_rate"])
        self.assertAlmostEqual(1.0, accepted_summary["probe_mean_validation_delta"])

    def test_cc_pass_budget_config_caps_uniform_group_budget_without_changing_default(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        grouping_result = [[0], [1], [2], [3]]
        adjacent_overlaps = hcc_es.compute_adjacent_overlaps_for_groups(grouping_result)

        default_config = info_module.InfoAwareNDAConfig(enable=True, enable_group_priority=False).normalized()
        _, _, default_budgets, _, default_metadata = hcc_es._resolve_cc_schedule(
            grouping_result,
            adjacent_overlaps,
            remaining_fes=40,
            cc_prior=None,
            info_aware_config=default_config,
        )

        capped_config = info_module.InfoAwareNDAConfig(
            enable=True,
            enable_group_priority=False,
            cc_pass_group_fes=3,
            cc_min_passes=3,
        ).normalized()
        _, _, capped_budgets, _, capped_metadata = hcc_es._resolve_cc_schedule(
            grouping_result,
            adjacent_overlaps,
            remaining_fes=40,
            cc_prior=None,
            info_aware_config=capped_config,
        )

        self.assertEqual([10, 10, 10, 10], default_budgets)
        self.assertEqual([3, 3, 3, 3], capped_budgets)
        self.assertFalse(default_metadata["cc_pass_budget_cap_enabled"])
        self.assertTrue(capped_metadata["cc_pass_budget_cap_enabled"])
        self.assertEqual(3, capped_metadata["cc_pass_group_fes"])

    def test_run_hcc_core_applies_aob_topology_group_order_before_adjacent_coordination(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        inputs = hcc_es.build_hcc_es_inputs("E4")
        benchmark = hcc_es.Benchmark(None)
        fun = benchmark.get_function(inputs["fun_name"], inputs["fun_id"])
        config = info_module.InfoAwareNDAConfig(
            enable=False,
            save_diagnostics=True,
            group_order_mode="aob_topology",
            overlap_blend_mode="equation8_correct",
        ).normalized()

        _, _, _, metadata = hcc_es.run_hcc_core(
            fun=fun,
            output_path=None,
            best_individual=inputs["best_individual"],
            max_fes=30,
            grouping_result=inputs["grouping_result"],
            info=inputs["info"],
            adjacent_overlapping_elements=inputs["adjacent_overlapping_elements"],
            seed=1,
            method=hcc_es.HCC_ES_METHOD,
            problem_code="E4",
            info_aware_config=config,
            return_metadata=True,
        )

        self.assertEqual("aob_topology", metadata["group_order_mode"])
        self.assertTrue(metadata["group_order_audit"]["group_order_changed"])
        self.assertAlmostEqual(1.0, metadata["group_order_audit"]["ordered_adjacent_overlap_ratio"])
        self.assertEqual("equation8_correct", metadata["blend_strategy"])
        self.assertEqual("equation8_correct", metadata["overlap_blend_summary"]["blend_strategy"])

    def test_run_hcc_core_decouples_rddsm_execution_from_aob_coordination_order(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        inputs = hcc_es.build_hcc_es_inputs("E4")
        benchmark = hcc_es.Benchmark(None)
        fun = benchmark.get_function(inputs["fun_name"], inputs["fun_id"])
        config = info_module.InfoAwareNDAConfig(
            enable=False,
            save_diagnostics=True,
            group_order_mode="rddsm",
            coordination_order_mode="aob_topology",
            overlap_blend_mode="equation8_correct",
        ).normalized()

        _, _, _, metadata = hcc_es.run_hcc_core(
            fun=fun,
            output_path=None,
            best_individual=inputs["best_individual"],
            max_fes=120,
            grouping_result=inputs["grouping_result"],
            info=inputs["info"],
            adjacent_overlapping_elements=inputs["adjacent_overlapping_elements"],
            seed=1,
            method=hcc_es.HCC_ES_METHOD,
            problem_code="E4",
            info_aware_config=config,
            return_metadata=True,
        )

        self.assertEqual("rddsm", metadata["group_order_mode"])
        self.assertEqual("aob_topology", metadata["coordination_order_mode"])
        self.assertFalse(metadata["execution_order_audit"]["group_order_changed"])
        self.assertTrue(metadata["coordination_order_audit"]["group_order_changed"])
        self.assertAlmostEqual(0.0, metadata["execution_order_audit"]["ordered_adjacent_overlap_ratio"])
        self.assertAlmostEqual(1.0, metadata["coordination_order_audit"]["ordered_adjacent_overlap_ratio"])
        self.assertGreater(metadata["overlap_blend_summary"]["active_overlap_count"], 0)
        self.assertIn("group_delta_summary", metadata)
        self.assertGreater(metadata["group_delta_summary"]["total_count"], 0)
        self.assertIn("early_delta_mean", metadata["group_delta_summary"])

    def test_coordination_selector_run_emits_trace_summary_and_can_turn_off_after_probe(self):
        hcc_es = load_hcc_es_module()
        info_module = load_info_aware_module()
        grouping_result = [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5, 6, 7],
        ]
        adjacent_overlaps = hcc_es.compute_adjacent_overlaps_for_groups(grouping_result)
        info = {
            "dimension": 8,
            "lower": -5.0,
            "upper": 5.0,
        }
        config = info_module.InfoAwareNDAConfig(
            enable=False,
            save_diagnostics=True,
            shared_variable_coordination_mode="selective_hypergraph_pass_end",
            shared_variable_conflict_damping=True,
            enable_validated_coordination=True,
            validation_mode="pass_end",
            validation_max_extra_fe_ratio=0.2,
            enable_coordination_selector=True,
            selector_probe_passes=1,
            selector_min_attempts=1,
            selector_min_accepts=1,
            selector_accept_rate_threshold=0.3,
            selector_mean_delta_threshold=0.0,
            cc_pass_group_fes=3,
            cc_min_passes=3,
        )

        _, _, _, metadata = hcc_es.run_hcc_core(
            SyntheticSphere(),
            output_path="",
            best_individual=np.full(info["dimension"], 2.0),
            max_fes=80,
            grouping_result=grouping_result,
            info=info,
            adjacent_overlapping_elements=adjacent_overlaps,
            seed=11,
            info_aware_config=config,
            return_metadata=True,
        )

        selector_rows = metadata["coordination_selector_rows"]
        selector_summary = metadata["coordination_selector_summary"]

        self.assertGreaterEqual(len(selector_rows), 1)
        self.assertEqual("probe", selector_rows[0]["phase"])
        self.assertIn(selector_summary["final_coordination_state"], {"off", "validated_on"})
        self.assertIn("selector_reason", selector_summary)
        self.assertEqual(
            selector_summary["probe_attempt_count"],
            sum(bool(row.get("validation_attempted")) for row in selector_rows if row.get("phase") == "probe"),
        )
        self.assertGreaterEqual(metadata["cc_pass_count"], 3)
        self.assertGreaterEqual(selector_summary["post_probe_pass_count"], 2)
        self.assertEqual(
            selector_summary["post_probe_pass_count"],
            sum(row.get("phase") == "selected" for row in selector_rows),
        )
        self.assertIn("coordination_selector_summary", metadata["info_aware_diagnostics"])

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
        self.assertGreater(diagnostic_payload["overlap_blend_count"], 0)
        self.assertIn("overlap_blend_summary", diagnostic_payload)
        self.assertIn("skip_ratio", diagnostic_payload["overlap_blend_summary"])
        self.assertIn("shared_variable_coordination_summary", diagnostic_payload)
        self.assertIn("coordination_visibility_audit", diagnostic_payload)
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
