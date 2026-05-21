import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

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


class HccEsNoneGuardTests(unittest.TestCase):
    def test_default_problem_codes_match_current_test_subset(self):
        hcc_es = load_hcc_es_module()

        self.assertEqual(hcc_es.DEFAULT_PROBLEM_CODES, ("E4", "E6", "S4", "S6", "A6", "R6"))

    def test_problem_code_maps_to_aob_function_and_id(self):
        hcc_es = load_hcc_es_module()

        self.assertEqual(("elliptic", 4, "E4"), hcc_es.parse_problem_code("e4"))
        self.assertEqual(("schwefel", 6, "S6"), hcc_es.parse_problem_code("S6"))
        self.assertEqual(("rastrigin", 6, "R6"), hcc_es.parse_problem_code("R6"))
        self.assertEqual(("ackley", 6, "A6"), hcc_es.parse_problem_code("A6"))

    def test_group_problem_codes_matches_current_subset_by_family(self):
        hcc_es = load_hcc_es_module()

        self.assertEqual(
            hcc_es.group_problem_codes(("E4", "E6", "S4", "S6", "A6", "R6")),
            {
                "elliptic": [4, 6],
                "schwefel": [4, 6],
                "ackley": [6],
                "rastrigin": [6],
            },
        )

    def test_only_original_hcc_method_is_accepted(self):
        hcc_es = load_hcc_es_module()

        self.assertEqual("hcc_es_original", hcc_es.canonicalize_method("hcc_es_original"))
        with self.assertRaises(ValueError):
            hcc_es.canonicalize_method("chcfr_hyper_dual_full")

    def test_summary_row_uses_method_seed_and_best_fitness(self):
        hcc_es = load_hcc_es_module()

        row = hcc_es.build_summary_row(
            problem_code="E4",
            method="hcc_es_original",
            seed=3,
            curve=[5.0, 3.0, 4.0],
            runtime=1.25,
            status="ok",
        )

        self.assertEqual("E4", row["problem"])
        self.assertEqual("hcc_es_original", row["method"])
        self.assertEqual(3, row["seed"])
        self.assertEqual(4.0, row["final_fitness"])
        self.assertEqual(3.0, row["best_fitness"])
        self.assertEqual(3, row["fe_used"])
        self.assertEqual("ok", row["status"])
        self.assertEqual(0, row["diagnostics_count"])
        self.assertEqual(0.0, row["rollback_ratio"])

    def test_summary_row_includes_requested_checkpoints(self):
        hcc_es = load_hcc_es_module()

        row = hcc_es.build_summary_row(
            problem_code="E4",
            method="hcc_es_original",
            seed=3,
            curve=[5.0, 3.0, 4.0, 2.0],
            runtime=1.25,
            status="ok",
            record_fes=[2, 4, 10],
        )

        self.assertEqual(3.0, row["best_at_2"])
        self.assertEqual(2.0, row["best_at_4"])
        self.assertTrue(row["best_at_10"] != row["best_at_10"])

    def test_overlap_weight_stays_finite_when_improvements_are_zero(self):
        hcc_es = load_hcc_es_module()

        self.assertEqual(0.5, hcc_es.safe_overlap_weight(0.0, 0.0))

    def test_build_overlap_hypergraph_tracks_incident_groups(self):
        hcc_es = load_hcc_es_module()

        hypergraph = hcc_es.build_overlap_hypergraph(
            [
                [0, 1, 2],
                [2, 3],
                [1, 4],
            ]
        )

        self.assertEqual([0, 2], hypergraph["var_to_groups"][1])
        self.assertEqual([0, 1], hypergraph["var_to_groups"][2])
        self.assertEqual([1, 2], hypergraph["overlap_vars"])
        self.assertEqual([1, 2], hypergraph["group_to_overlap_vars"][0])
        self.assertEqual([2], hypergraph["group_to_overlap_vars"][1])
        self.assertEqual([1], hypergraph["group_to_overlap_vars"][2])

    def test_build_overlap_features_counts_overlap_ratio(self):
        hcc_es = load_hcc_es_module()

        grouping_result = [
            [0, 1, 2],
            [2, 3],
            [1, 4],
        ]
        hypergraph = hcc_es.build_overlap_hypergraph(grouping_result)
        features = hcc_es.build_overlap_features(grouping_result, hypergraph)

        self.assertAlmostEqual(2.0 / 5.0, features["overlap_ratio"])
        self.assertAlmostEqual(0.5, features["nonadjacent_overlap_ratio"])
        self.assertEqual([2, 1, 1], features["group_overlap_var_count"])
        self.assertEqual([2.0, 1.0, 1.0], features["group_overlap_load"])

    def test_topology_order_reorders_groups_without_changing_membership(self):
        hcc_es = load_hcc_es_module()
        rddsm_order = [
            [20, 21],
            [0, 1],
            [10, 11, 20],
            [1, 10],
        ]
        natural_order = [
            [0, 1],
            [1, 10],
            [10, 11, 20],
            [20, 21],
        ]

        reordered, order, audit = hcc_es.reorder_groups_by_reference_topology(
            rddsm_order,
            natural_order,
        )

        self.assertEqual([1, 3, 2, 0], order)
        self.assertEqual(natural_order, reordered)
        self.assertEqual(
            sorted(sorted(group) for group in rddsm_order),
            sorted(sorted(group) for group in reordered),
        )
        self.assertAlmostEqual(1.0, audit["ordered_adjacent_overlap_ratio"])
        self.assertGreater(audit["ordered_adjacent_overlap_ratio"], audit["original_adjacent_overlap_ratio"])

    def test_aob_topology_order_restores_e4_adjacent_overlap_visibility(self):
        hcc_es = load_hcc_es_module()

        inputs = hcc_es.build_hcc_es_inputs("E4")
        natural_groups = hcc_es.build_aob_natural_groups("E4")
        reordered, order, audit = hcc_es.reorder_groups_by_reference_topology(
            inputs["grouping_result"],
            natural_groups,
        )

        self.assertEqual(len(inputs["grouping_result"]), len(reordered))
        self.assertEqual(set(range(len(inputs["grouping_result"]))), set(order))
        self.assertEqual(
            sorted(sorted(group) for group in inputs["grouping_result"]),
            sorted(sorted(group) for group in reordered),
        )
        self.assertAlmostEqual(0.0, audit["original_adjacent_overlap_ratio"])
        self.assertAlmostEqual(1.0, audit["reference_adjacent_overlap_ratio"])
        self.assertGreaterEqual(audit["ordered_adjacent_overlap_ratio"], 0.95)

    def test_group_order_plan_can_decouple_execution_and_coordination_orders(self):
        hcc_es = load_hcc_es_module()
        rddsm_order = [
            [20, 21],
            [0, 1],
            [10, 11, 20],
            [1, 10],
        ]
        natural_order = [
            [0, 1],
            [1, 10],
            [10, 11, 20],
            [20, 21],
        ]

        plan = hcc_es.build_group_order_plan(
            rddsm_order,
            problem_code=None,
            execution_order_mode="rddsm",
            coordination_order_mode="aob_topology",
            reference_groups=natural_order,
        )

        self.assertEqual(rddsm_order, plan["execution_groups"])
        self.assertEqual(natural_order, plan["coordination_groups"])
        self.assertEqual([0, 1, 2, 3], plan["execution_group_order"])
        self.assertEqual([1, 3, 2, 0], plan["coordination_group_order"])
        self.assertAlmostEqual(1.0 / 3.0, plan["execution_order_audit"]["ordered_adjacent_overlap_ratio"])
        self.assertAlmostEqual(1.0, plan["coordination_order_audit"]["ordered_adjacent_overlap_ratio"])

    def test_coordination_edges_wait_until_both_groups_have_run(self):
        hcc_es = load_hcc_es_module()
        coordination_groups = [
            [0, 1],
            [1, 10],
            [10, 11, 20],
            [20, 21],
        ]
        coordination_order = [1, 3, 2, 0]
        execution_ids = [0, 1, 2, 3]
        completed = set()

        completed.add(0)
        self.assertEqual(
            [],
            hcc_es.pop_ready_coordination_edges(
                execution_group_id=0,
                completed_group_ids=completed,
                coordination_groups=coordination_groups,
                coordination_group_order=coordination_order,
                processed_coordination_edges=set(),
            ),
        )

        completed.add(2)
        ready = hcc_es.pop_ready_coordination_edges(
            execution_group_id=2,
            completed_group_ids=completed,
            coordination_groups=coordination_groups,
            coordination_group_order=coordination_order,
            processed_coordination_edges=set(),
        )

        self.assertEqual(1, len(ready))
        self.assertEqual(2, ready[0]["left_group_id"])
        self.assertEqual(0, ready[0]["right_group_id"])
        self.assertEqual([20], ready[0]["overlap_indices"])
        self.assertEqual([execution_ids[0], execution_ids[2]], [0, 2])

    def test_decoupled_coordination_blend_uses_both_endpoint_proposals(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0, 0.0])
        previous_proposal = np.array([10.0, 0.0])
        current_proposal = np.array([30.0, 0.0])
        current_original = np.array([2.0, 0.0])

        diagnostics = hcc_es.apply_coordination_edge_blend(
            best_individual=best_individual,
            previous_proposal_individual=previous_proposal,
            current_proposal_individual=current_proposal,
            current_original_individual=current_original,
            overlapping_elements=[0],
            previous_delta=1.0,
            current_delta=3.0,
            overlap_blend_mode="equation8_correct",
            use_pairwise_endpoint_values=True,
        )

        self.assertTrue(diagnostics["applied_update"])
        self.assertAlmostEqual(0.75, diagnostics["weight"])
        self.assertAlmostEqual(25.0, best_individual[0])

    def test_sync_omega_cache_for_variables_updates_all_incident_group_means(self):
        hcc_es = load_hcc_es_module()
        group_index = hcc_es.build_group_variable_index_maps(
            [
                [0, 1, 2],
                [2, 3],
                [1, 4],
            ],
            source_group_ids=[10, 20, 30],
        )
        omega_cache = {
            10: np.array([10.0, 11.0, 12.0]),
            20: np.array([22.0, 23.0]),
            30: np.array([31.0, 34.0]),
        }
        best_individual = np.array([0.0, 100.0, 200.0, 3.0, 4.0])

        rows = hcc_es.sync_omega_cache_for_variables(
            omega_cache,
            best_individual,
            variable_ids=[1, 2],
            group_variable_index=group_index["group_variable_index"],
            variable_to_group_ids=group_index["variable_to_group_ids"],
            source_group_dims=group_index["source_group_dims"],
        )

        np.testing.assert_allclose(omega_cache[10], np.array([10.0, 100.0, 200.0]))
        np.testing.assert_allclose(omega_cache[20], np.array([200.0, 23.0]))
        np.testing.assert_allclose(omega_cache[30], np.array([100.0, 34.0]))
        self.assertEqual(4, len(rows))
        self.assertEqual({10, 20, 30}, {int(row["group_id"]) for row in rows})
        self.assertTrue(all(row["event"] == "coordination_sync" for row in rows))
        self.assertAlmostEqual(0.0, max(float(row["mean_coord_mismatch_after_sync"]) for row in rows))

    def test_run_hcc_core_default_ephemeral_emits_optimizer_state_audit(self):
        hcc_es = load_hcc_es_module()
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

        _, _, _, metadata = hcc_es.run_hcc_core(
            fun=lambda x_batch: np.sum(np.square(np.asarray(x_batch, dtype=float)), axis=-1),
            output_path="",
            best_individual=np.full(info["dimension"], 2.0),
            max_fes=40,
            grouping_result=grouping_result,
            info=info,
            adjacent_overlapping_elements=adjacent_overlaps,
            seed=13,
            return_metadata=True,
        )

        rows = metadata["optimizer_state_rows"]
        init_rows = [row for row in rows if row["event"] == "mean_init"]
        self.assertEqual("ephemeral", metadata["optimizer_state_mode"])
        self.assertGreater(len(init_rows), 0)
        self.assertTrue(all(row["mean_init_source"] == "best_individual" for row in init_rows))
        self.assertTrue(all(bool(row["optimizer_reinitialized"]) for row in init_rows))
        self.assertEqual(0, metadata["optimizer_state_summary"]["omega_sync_count"])

    def test_run_hcc_core_persistent_mean_syncs_coordination_into_cached_means(self):
        hcc_es = load_hcc_es_module()
        workspace_root = Path(__file__).resolve().parents[1]
        source_root = workspace_root / "HCC_SRC"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from HCC.info_aware_nda import InfoAwareNDAConfig

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
        config = InfoAwareNDAConfig(
            enable=False,
            save_diagnostics=True,
            optimizer_state_mode="persistent_mean",
            cc_pass_group_fes=3,
            cc_min_passes=2,
        ).normalized()

        _, _, _, metadata = hcc_es.run_hcc_core(
            fun=lambda x_batch: np.sum(np.square(np.asarray(x_batch, dtype=float)), axis=-1),
            output_path="",
            best_individual=np.full(info["dimension"], 2.0),
            max_fes=80,
            grouping_result=grouping_result,
            info=info,
            adjacent_overlapping_elements=adjacent_overlaps,
            seed=17,
            info_aware_config=config,
            return_metadata=True,
        )

        sync_rows = [row for row in metadata["optimizer_state_rows"] if row["event"] == "coordination_sync"]
        self.assertEqual("persistent_mean", metadata["optimizer_state_mode"])
        self.assertGreater(metadata["optimizer_state_summary"]["omega_sync_count"], 0)
        self.assertGreater(len(sync_rows), 0)
        self.assertAlmostEqual(0.0, metadata["optimizer_state_summary"]["mean_coord_mismatch_after_sync_max"])
        self.assertTrue(all(row["mean_init_source"] in {"omega_cache_init", "omega_cache"} for row in metadata["optimizer_state_rows"] if row["event"] == "mean_init"))
        self.assertTrue(all(bool(row["optimizer_reinitialized"]) for row in metadata["optimizer_state_rows"] if row["event"] == "mean_init"))

    def test_audit_grouping_dimension_coverage_reports_uncovered_variables(self):
        hcc_es = load_hcc_es_module()

        audit = hcc_es.audit_grouping_dimension_coverage(
            [
                [0, 1],
                [3],
            ],
            benchmark_dimension=5,
            design_matrix_dimension=4,
        )

        self.assertEqual(5, audit["benchmark_dimension"])
        self.assertEqual(4, audit["design_matrix_dimension"])
        self.assertEqual(3, audit["covered_variable_count"])
        self.assertEqual(2, audit["uncovered_variable_count"])
        self.assertEqual([2, 4], audit["uncovered_variables"])

    def test_ensure_grouping_covers_benchmark_dimension_appends_residual_group(self):
        hcc_es = load_hcc_es_module()

        grouping_result, audit = hcc_es.ensure_grouping_covers_benchmark_dimension(
            [
                [0, 1],
                [3],
            ],
            benchmark_dimension=5,
            design_matrix_dimension=4,
        )

        self.assertEqual([[0, 1], [3], [2, 4]], grouping_result)
        self.assertTrue(audit["coverage_patch_applied"])
        self.assertEqual(3, audit["covered_variable_count"])
        self.assertEqual(2, audit["uncovered_variable_count"])
        self.assertEqual(5, audit["effective_covered_variable_count"])
        self.assertEqual(0, audit["effective_uncovered_variable_count"])
        self.assertEqual(2, audit["residual_group_size"])

    def test_aob_info_uses_decision_dimension_and_keeps_expanded_dimension_for_audit(self):
        hcc_es = load_hcc_es_module()
        benchmark = hcc_es.Benchmark(None)

        e4_info = benchmark.get_info("elliptic", 4)
        e6_info = benchmark.get_info("elliptic", 6)

        self.assertEqual(1000, e4_info["dimension"])
        self.assertEqual(1000, e4_info["decision_dimension"])
        self.assertEqual(1095, e4_info["expanded_dimension"])
        self.assertEqual(1000, e6_info["dimension"])
        self.assertEqual(1000, e6_info["decision_dimension"])
        self.assertEqual(1190, e6_info["expanded_dimension"])

    def test_build_hcc_es_inputs_uses_true_decision_dimension_for_e4(self):
        hcc_es = load_hcc_es_module()

        inputs = hcc_es.build_hcc_es_inputs("E4")
        audit = inputs["coverage_audit"]
        covered_variables = {int(var_id) for group in inputs["grouping_result"] for var_id in group}

        self.assertEqual(1000, audit["benchmark_dimension"])
        self.assertEqual(1000, audit["design_matrix_dimension"])
        self.assertEqual(1000, audit["covered_variable_count"])
        self.assertEqual(0, audit["uncovered_variable_count"])
        self.assertFalse(audit["coverage_patch_applied"])
        self.assertEqual(1000, len(covered_variables))
        self.assertEqual(set(range(1000)), covered_variables)
        self.assertEqual(20, len(inputs["grouping_result"]))

    def test_shared_variable_visibility_audit_distinguishes_adjacent_and_hypergraph_modes(self):
        hcc_es = load_hcc_es_module()
        grouping_result = [
            [0, 1],
            [2, 3],
            [1, 4],
        ]
        adjacent_overlaps = hcc_es.compute_adjacent_overlaps_for_groups(grouping_result)
        hypergraph = hcc_es.build_overlap_hypergraph(grouping_result)

        adjacent_audit = hcc_es.build_shared_variable_visibility_audit(
            grouping_result,
            adjacent_overlaps,
            coordination_mode="adjacent",
            overlap_hypergraph=hypergraph,
        )
        hypergraph_audit = hcc_es.build_shared_variable_visibility_audit(
            grouping_result,
            adjacent_overlaps,
            coordination_mode="hypergraph_pass_end",
            overlap_hypergraph=hypergraph,
        )

        self.assertEqual(1, adjacent_audit["true_overlap_var_count"])
        self.assertEqual(0, adjacent_audit["coordinated_overlap_var_count"])
        self.assertEqual(1, hypergraph_audit["coordinated_overlap_var_count"])
        self.assertAlmostEqual(1.0, hypergraph_audit["coordinated_overlap_ratio"])

    def test_collect_group_overlap_variable_proposals_records_requested_fields(self):
        hcc_es = load_hcc_es_module()

        rows = hcc_es.collect_group_overlap_variable_proposals(
            best_individual=np.array([1.0, 2.0, 3.0]),
            overlap_variables=[2, 0],
            group_id=4,
            delta=3.5,
            fitness_before=10.0,
            fitness_after=6.5,
            cycle_id=2,
            scheduled_position=1,
        )

        self.assertEqual(
            [
                {
                    "cycle_id": 2,
                    "scheduled_position": 1,
                    "group_id": 4,
                    "var_id": 0,
                    "proposal_value": 1.0,
                    "delta": 3.5,
                    "fitness_before": 10.0,
                    "fitness_after": 6.5,
                },
                {
                    "cycle_id": 2,
                    "scheduled_position": 1,
                    "group_id": 4,
                    "var_id": 2,
                    "proposal_value": 3.0,
                    "delta": 3.5,
                    "fitness_before": 10.0,
                    "fitness_after": 6.5,
                },
            ],
            rows,
        )

    def test_hypergraph_pass_end_coordination_uses_positive_delta_weighting(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0, 10.0, 20.0])
        hypergraph = {
            "overlap_vars": [1],
            "var_to_groups": {1: [0, 2]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 1, "proposal_value": 4.0, "delta": 1.0},
                {"var_id": 1, "proposal_value": 8.0, "delta": 3.0},
                {"var_id": 1, "proposal_value": 100.0, "delta": -2.0},
            ],
            overlap_hypergraph=hypergraph,
        )

        self.assertEqual(1, len(fusion_rows))
        self.assertEqual(2, fusion_rows[0]["positive_proposal_count"])
        self.assertTrue(fusion_rows[0]["applied_update"])
        self.assertAlmostEqual(7.0, best_individual[1])
        self.assertAlmostEqual(7.0, fusion_rows[0]["applied_value"])

    def test_hypergraph_pass_end_coordination_keeps_old_value_without_positive_proposals(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0, 5.0])
        hypergraph = {
            "overlap_vars": [1],
            "var_to_groups": {1: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 1, "proposal_value": 2.0, "delta": 0.0},
                {"var_id": 1, "proposal_value": 1.0, "delta": -3.0},
            ],
            overlap_hypergraph=hypergraph,
        )

        self.assertFalse(fusion_rows[0]["applied_update"])
        self.assertAlmostEqual(5.0, best_individual[1])
        self.assertAlmostEqual(5.0, fusion_rows[0]["applied_value"])

    def test_hypergraph_pass_end_coordination_supports_conflict_damping(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([2.0, 0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 0, "proposal_value": 10.0, "delta": 2.0},
                {"var_id": 0, "proposal_value": 10.0, "delta": 2.0},
            ],
            overlap_hypergraph=hypergraph,
            conflict_prior=np.array([0.5, 0.0]),
            use_conflict_damping=True,
            conflict_gamma=0.5,
            min_damping=0.0,
        )

        self.assertAlmostEqual(0.75, fusion_rows[0]["damping"])
        self.assertAlmostEqual(8.0, best_individual[0])

    def test_selective_hypergraph_skips_multi_support_fusion_when_positive_support_is_too_weak(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0, 5.0])
        hypergraph = {
            "overlap_vars": [1],
            "var_to_groups": {1: [0, 1, 2]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 1, "proposal_value": 5.2, "delta": 2.0, "group_id": 0},
                {"var_id": 1, "proposal_value": 5.3, "delta": 1.5, "group_id": 1},
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="selective_hypergraph_pass_end",
            variable_ranges=np.array([10.0, 10.0]),
            selective_min_positive_proposals=3,
        )

        self.assertFalse(fusion_rows[0]["was_updated"])
        self.assertEqual("insufficient_positive_support", fusion_rows[0]["skip_reason"])
        self.assertAlmostEqual(5.0, best_individual[1])

    def test_selective_hypergraph_owner_soft_applies_small_step_for_single_strong_owner(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {
                    "var_id": 0,
                    "proposal_value": 0.8,
                    "delta": 2.0,
                    "group_id": 3,
                    "fitness_before": 1000.0,
                    "fitness_after": 998.0,
                },
                {
                    "var_id": 0,
                    "proposal_value": 0.82,
                    "delta": -0.5,
                    "group_id": 4,
                    "fitness_before": 1000.0,
                    "fitness_after": 1000.5,
                },
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="selective_hypergraph_pass_end",
            variable_ranges=np.array([100.0]),
            selective_owner_soft_eta=0.2,
            selective_owner_min_delta_ratio=0.001,
        )

        self.assertTrue(fusion_rows[0]["was_updated"])
        self.assertEqual("owner_soft", fusion_rows[0]["ownership_mode"])
        self.assertEqual("owner_soft_applied", fusion_rows[0]["skip_reason"])
        self.assertEqual(3, fusion_rows[0]["owner_group_id"])
        self.assertAlmostEqual(0.002, fusion_rows[0]["owner_delta_ratio"])
        self.assertAlmostEqual(0.2, fusion_rows[0]["owner_step_weight"])
        self.assertAlmostEqual(0.16, best_individual[0])
        self.assertFalse(fusion_rows[0]["harmful_update_proxy_flag"])

    def test_selective_hypergraph_owner_soft_freezes_when_owner_delta_is_small(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {
                    "var_id": 0,
                    "proposal_value": 10.0,
                    "delta": 0.2,
                    "group_id": 3,
                    "fitness_before": 1000.0,
                    "fitness_after": 999.8,
                },
                {
                    "var_id": 0,
                    "proposal_value": 10.1,
                    "delta": -0.5,
                    "group_id": 4,
                    "fitness_before": 1000.0,
                    "fitness_after": 1000.5,
                },
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="selective_hypergraph_pass_end",
            variable_ranges=np.array([100.0]),
            selective_owner_soft_eta=0.2,
            selective_owner_min_delta_ratio=0.001,
        )

        self.assertFalse(fusion_rows[0]["was_updated"])
        self.assertEqual("freeze", fusion_rows[0]["ownership_mode"])
        self.assertEqual("owner_delta_too_small", fusion_rows[0]["skip_reason"])
        self.assertAlmostEqual(0.0, best_individual[0])

    def test_selective_hypergraph_skips_update_when_proposal_std_is_too_large(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 0, "proposal_value": -10.0, "delta": 2.0, "group_id": 0},
                {"var_id": 0, "proposal_value": 10.0, "delta": 2.0, "group_id": 1},
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="selective_hypergraph_pass_end",
            variable_ranges=np.array([100.0]),
            selective_max_proposal_std_ratio=0.04,
        )

        self.assertFalse(fusion_rows[0]["was_updated"])
        self.assertEqual("proposal_std_too_large", fusion_rows[0]["skip_reason"])
        self.assertGreater(fusion_rows[0]["proposal_value_std_ratio"], 0.04)
        self.assertAlmostEqual(0.0, best_individual[0])

    def test_selective_hypergraph_damps_large_update_when_support_is_strong(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 0, "proposal_value": 10.0, "delta": 2.0, "group_id": 0},
                {"var_id": 0, "proposal_value": 10.0, "delta": 3.0, "group_id": 1},
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="selective_hypergraph_pass_end",
            variable_ranges=np.array([100.0]),
            selective_max_update_ratio=0.05,
            selective_large_update_damping=0.25,
        )

        self.assertTrue(fusion_rows[0]["was_updated"])
        self.assertEqual("multi_support_large_update_damped", fusion_rows[0]["skip_reason"])
        self.assertAlmostEqual(2.5, best_individual[0])
        self.assertAlmostEqual(0.025, fusion_rows[0]["update_magnitude_ratio"])

    def test_arac_lite_rule_maps_agreeing_positive_support_to_fusion(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 0, "proposal_value": 0.20, "delta": 2.0, "group_id": 0},
                {"var_id": 0, "proposal_value": 0.22, "delta": 3.0, "group_id": 1},
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.array([100.0]),
            selective_max_proposal_std_ratio=0.01,
        )

        self.assertEqual("Fusion", fusion_rows[0]["action_candidate"])
        self.assertEqual("multi_support_fusion", fusion_rows[0]["ownership_mode"])
        self.assertTrue(fusion_rows[0]["was_updated"])
        self.assertAlmostEqual(0.212, best_individual[0])
        self.assertEqual("fusion_supported", fusion_rows[0]["action_reason"])

    def test_arac_lite_rule_freezes_conflicting_positive_support(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 0, "proposal_value": -10.0, "delta": 2.0, "group_id": 0},
                {"var_id": 0, "proposal_value": 10.0, "delta": 3.0, "group_id": 1},
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.array([100.0]),
            selective_max_proposal_std_ratio=0.04,
        )

        self.assertEqual("Freeze", fusion_rows[0]["action_candidate"])
        self.assertEqual("freeze", fusion_rows[0]["ownership_mode"])
        self.assertFalse(fusion_rows[0]["was_updated"])
        self.assertEqual("proposal_conflict", fusion_rows[0]["action_reason"])
        self.assertAlmostEqual(0.0, best_individual[0])

    def test_arac_lite_rule_disables_relation_with_bad_validation_history(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0])
        hypergraph = {
            "overlap_vars": [0],
            "var_to_groups": {0: [0, 1]},
        }

        fusion_rows = hcc_es.apply_hypergraph_pass_end_coordination(
            best_individual,
            proposal_rows=[
                {"var_id": 0, "proposal_value": 0.20, "delta": 2.0, "group_id": 0},
                {"var_id": 0, "proposal_value": 0.22, "delta": 3.0, "group_id": 1},
            ],
            overlap_hypergraph=hypergraph,
            coordination_mode="arac_lite_rule",
            variable_ranges=np.array([100.0]),
            selective_max_proposal_std_ratio=0.01,
            arac_relation_history={
                0: {
                    "attempt_count": 2,
                    "accept_count": 0,
                    "delta_sum": -1.0,
                    "reject_streak": 2,
                }
            },
            arac_lite_history_min_attempts=2,
            arac_lite_disable_reject_streak=2,
        )

        self.assertEqual("Disable", fusion_rows[0]["action_candidate"])
        self.assertEqual("disable", fusion_rows[0]["ownership_mode"])
        self.assertFalse(fusion_rows[0]["was_updated"])
        self.assertEqual("history_rejected", fusion_rows[0]["action_reason"])
        self.assertAlmostEqual(0.0, best_individual[0])

    def test_arac_lite_history_updates_only_validated_fusion_actions(self):
        hcc_es = load_hcc_es_module()
        history = {}

        hcc_es.update_arac_relation_history(
            history,
            [
                {
                    "var_id": 0,
                    "action_candidate": "Fusion",
                    "was_updated": True,
                    "validation_attempted": True,
                    "validation_accepted": False,
                    "fitness_delta": -0.5,
                },
                {
                    "var_id": 1,
                    "action_candidate": "Freeze",
                    "was_updated": False,
                    "validation_attempted": True,
                    "validation_accepted": True,
                    "fitness_delta": 1.0,
                },
            ],
        )

        self.assertEqual({0}, set(history.keys()))
        self.assertEqual(1, history[0]["attempt_count"])
        self.assertEqual(0, history[0]["accept_count"])
        self.assertAlmostEqual(-0.5, history[0]["delta_sum"])
        self.assertEqual(1, history[0]["reject_streak"])

    def test_shared_variable_summary_reports_freeze_owner_and_harmful_update_metrics(self):
        hcc_es = load_hcc_es_module()

        summary = hcc_es.summarize_shared_variable_coordination_rows(
            [
                {
                    "positive_proposal_count": 2,
                    "proposal_count": 2,
                    "was_updated": True,
                    "applied_update": True,
                    "update_magnitude": 1.0,
                    "update_magnitude_ratio": 0.01,
                    "proposal_value_std": 0.2,
                    "proposal_value_std_ratio": 0.002,
                    "damping": 0.5,
                    "gate_passed": True,
                    "harmful_update_proxy_flag": True,
                    "skip_reason": "multi_support_large_update_damped",
                    "ownership_mode": "multi_support_fusion",
                },
                {
                    "positive_proposal_count": 2,
                    "proposal_count": 2,
                    "was_updated": False,
                    "applied_update": False,
                    "update_magnitude": 0.0,
                    "update_magnitude_ratio": 0.0,
                    "proposal_value_std": 5.0,
                    "proposal_value_std_ratio": 0.05,
                    "damping": 1.0,
                    "gate_passed": False,
                    "harmful_update_proxy_flag": False,
                    "skip_reason": "owner_delta_too_small",
                    "ownership_mode": "freeze",
                },
                {
                    "positive_proposal_count": 1,
                    "proposal_count": 2,
                    "was_updated": True,
                    "applied_update": True,
                    "update_magnitude": 0.2,
                    "update_magnitude_ratio": 0.002,
                    "proposal_value_std": 0.1,
                    "proposal_value_std_ratio": 0.001,
                    "damping": 0.2,
                    "gate_passed": True,
                    "harmful_update_proxy_flag": False,
                    "skip_reason": "owner_soft_applied",
                    "ownership_mode": "owner_soft",
                },
            ],
            visibility_audit={
                "coordination_mode": "selective_hypergraph_pass_end",
                "true_overlap_var_count": 3,
                "adjacent_visible_overlap_var_count": 0,
                "coordinated_overlap_var_count": 3,
            },
        )

        self.assertAlmostEqual(2.0 / 3.0, summary["update_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["skip_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["owner_soft_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["multi_support_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["freeze_ratio"])
        self.assertAlmostEqual(0.6, summary["mean_update_magnitude"])
        self.assertAlmostEqual((0.2 + 5.0 + 0.1) / 3.0, summary["proposal_std_mean"])
        self.assertAlmostEqual(5.0 / 3.0, summary["positive_proposal_count_mean"])
        self.assertAlmostEqual(1.0 / 3.0, summary["owner_delta_blocked_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["large_update_damped_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["harmful_update_proxy"])

    def test_blending_overlap_never_writes_nan_for_degenerate_improvements(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([1.0, 2.0, 3.0])
        original_best_individual = np.array([4.0, 5.0, 6.0])

        hcc_es.blend_overlapping_elements(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 2]),
            previous_delta=0.0,
            current_delta=0.0,
        )

        self.assertTrue(np.all(np.isfinite(best_individual)))
        np.testing.assert_allclose(best_individual, np.array([2.5, 2.0, 4.5]))

    def test_safe_conflict_aware_overlap_blend_damps_both_positive_updates(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([4.0, 2.0, 6.0])
        original_best_individual = np.array([0.0, 2.0, 0.0])
        conflict_prior = np.array([0.5, 0.0, 0.5])

        diagnostics = hcc_es.safe_conflict_aware_overlap_blend(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 2]),
            previous_delta=1.0,
            current_delta=3.0,
            conflict_prior=conflict_prior,
        )

        self.assertEqual("safe_conflict", diagnostics["blend_strategy"])
        self.assertEqual("both_positive", diagnostics["blend_mode"])
        self.assertFalse(diagnostics["was_skipped"])
        self.assertAlmostEqual(0.75, diagnostics["raw_weight"])
        self.assertAlmostEqual(0.75, diagnostics["weight"])
        self.assertAlmostEqual(0.5, diagnostics["conflict_mean"])
        self.assertAlmostEqual(0.75, diagnostics["damping"])
        np.testing.assert_allclose(best_individual, np.array([2.25, 2.0, 3.375]))

    def test_safe_conflict_previous_only_keeps_original_overlap_when_current_delta_is_negative(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([0.0, 10.0, 6.0])
        original_best_individual = np.array([100.0, 200.0, 3.0])

        diagnostics = hcc_es.safe_conflict_aware_overlap_blend(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 1]),
            previous_delta=10.0,
            current_delta=-5.0,
        )

        self.assertEqual("previous_only", diagnostics["blend_mode"])
        self.assertFalse(diagnostics["was_skipped"])
        self.assertAlmostEqual(0.0, diagnostics["raw_weight"])
        self.assertAlmostEqual(0.0, diagnostics["weight"])
        self.assertAlmostEqual(1.0, diagnostics["damping"])
        np.testing.assert_allclose(best_individual, np.array([100.0, 200.0, 6.0]))

    def test_safe_conflict_damping_reduces_applied_step_when_conflict_mean_is_positive(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([10.0, 0.0])
        original_best_individual = np.array([0.0, 0.0])
        conflict_prior = np.array([0.5, 0.5])

        diagnostics = hcc_es.safe_conflict_aware_overlap_blend(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 1]),
            previous_delta=1.0,
            current_delta=1.0,
            conflict_prior=conflict_prior,
            conflict_gamma=0.5,
            min_damping=0.0,
        )

        self.assertEqual("both_positive", diagnostics["blend_mode"])
        self.assertAlmostEqual(0.5, diagnostics["conflict_mean"])
        self.assertAlmostEqual(0.75, diagnostics["damping"])
        self.assertLess(diagnostics["damping"], 1.0)
        np.testing.assert_allclose(best_individual, np.array([3.75, 0.0]))

    def test_safe_conflict_aware_overlap_blend_skips_when_no_positive_delta(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([4.0, 2.0, 6.0])
        original_best_individual = np.array([1.0, 2.0, 3.0])

        diagnostics = hcc_es.safe_conflict_aware_overlap_blend(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 2]),
            previous_delta=0.0,
            current_delta=-1.0,
        )

        self.assertEqual("skip_no_positive_delta", diagnostics["blend_mode"])
        self.assertTrue(diagnostics["was_skipped"])
        self.assertFalse(diagnostics["applied_update"])
        np.testing.assert_allclose(best_individual, np.array([4.0, 2.0, 6.0]))

    def test_apply_overlap_blend_strategy_supports_no_blend_ablation(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([4.0, 2.0, 6.0])
        original_best_individual = np.array([1.0, 2.0, 3.0])

        diagnostics = hcc_es.apply_overlap_blend_strategy(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 2]),
            previous_delta=1.0,
            current_delta=3.0,
            overlap_blend_mode="no_blend",
        )

        self.assertEqual("no_blend", diagnostics["blend_strategy"])
        self.assertEqual("no_blend", diagnostics["blend_mode"])
        self.assertFalse(diagnostics["was_skipped"])
        self.assertFalse(diagnostics["applied_update"])
        np.testing.assert_allclose(best_individual, np.array([4.0, 2.0, 6.0]))

    def test_equation8_correct_blend_weights_current_by_current_delta(self):
        hcc_es = load_hcc_es_module()
        best_individual = np.array([10.0, 0.0, 30.0])
        original_best_individual = np.array([2.0, 0.0, 6.0])

        diagnostics = hcc_es.apply_overlap_blend_strategy(
            best_individual,
            original_best_individual,
            overlapping_elements=np.array([0, 2]),
            previous_delta=1.0,
            current_delta=3.0,
            overlap_blend_mode="equation8_correct",
        )

        self.assertEqual("equation8_correct", diagnostics["blend_strategy"])
        self.assertEqual("equation8_correct", diagnostics["blend_mode"])
        self.assertAlmostEqual(0.75, diagnostics["weight"])
        np.testing.assert_allclose(best_individual, np.array([8.0, 0.0, 24.0]))

    def test_summarize_overlap_blend_rows_reports_gate_ratios(self):
        hcc_es = load_hcc_es_module()
        rows = [
            {
                "overlap_count": 2,
                "blend_strategy": "safe_conflict",
                "blend_mode": "both_positive",
                "previous_delta": 1.0,
                "current_delta": 2.0,
                "conflict_mean": 0.2,
                "damping": 0.9,
                "was_skipped": False,
                "applied_update": True,
            },
            {
                "overlap_count": 2,
                "blend_strategy": "safe_conflict",
                "blend_mode": "current_only",
                "previous_delta": 0.0,
                "current_delta": 2.0,
                "conflict_mean": 0.4,
                "damping": 0.8,
                "was_skipped": False,
                "applied_update": True,
            },
            {
                "overlap_count": 2,
                "blend_strategy": "safe_conflict",
                "blend_mode": "skip_no_positive_delta",
                "previous_delta": 0.0,
                "current_delta": 0.0,
                "conflict_mean": 0.6,
                "damping": 1.0,
                "was_skipped": True,
                "applied_update": False,
            },
        ]

        summary = hcc_es.summarize_overlap_blend_rows(rows)

        self.assertEqual("safe_conflict", summary["blend_strategy"])
        self.assertEqual(3, summary["active_overlap_count"])
        self.assertAlmostEqual(1.0 / 3.0, summary["skip_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["both_positive_ratio"])
        self.assertAlmostEqual(1.0 / 3.0, summary["current_only_ratio"])
        self.assertAlmostEqual(0.4, summary["mean_conflict"])
        self.assertAlmostEqual(0.85, summary["mean_damping"])
        self.assertAlmostEqual(1.0 / 3.0, summary["negative_delta_blend_ratio"])

    def test_override_info_aware_config_allows_overlap_blend_without_nda_enable(self):
        hcc_es = load_hcc_es_module()

        config = hcc_es.override_info_aware_config(None, overlap_blend_mode="safe_delta")

        self.assertFalse(config.enable)
        self.assertEqual("safe_delta", config.overlap_blend_mode)

    def test_write_aggregate_summary_rolls_up_best_fitness(self):
        hcc_es = load_hcc_es_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detail_path = root / "run_details.csv"
            diagnostics_path = root / "diagnostics.csv"
            summary_path = root / "summary.csv"

            detail_rows = [
                {
                    "problem": "S4",
                    "method": "hcc_es_safe_conflict_blend",
                    "tfes": 100,
                    "blend_strategy": "safe_conflict",
                    "seed": 1,
                    "final_fitness": 10.0,
                    "best_fitness": 7.0,
                    "fe_used": 100,
                    "runtime": 1.0,
                    "diagnostics_count": 0,
                    "rollback_ratio": 0.0,
                    "conflict_mean": "",
                    "status": "ok",
                },
                {
                    "problem": "S4",
                    "method": "hcc_es_safe_conflict_blend",
                    "tfes": 100,
                    "blend_strategy": "safe_conflict",
                    "seed": 2,
                    "final_fitness": 9.0,
                    "best_fitness": 5.0,
                    "fe_used": 100,
                    "runtime": 1.1,
                    "diagnostics_count": 0,
                    "rollback_ratio": 0.0,
                    "conflict_mean": "",
                    "status": "ok",
                },
            ]

            for row in detail_rows:
                hcc_es.append_csv_row(detail_path, row, hcc_es.RUN_DETAIL_FIELDNAMES)
            hcc_es.ensure_csv_header(diagnostics_path, hcc_es.DIAGNOSTIC_FIELDNAMES)

            hcc_es.write_aggregate_summary(detail_path, diagnostics_path, summary_path)

            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual("S4", row["problem"])
            self.assertEqual("hcc_es_safe_conflict_blend", row["method"])
            self.assertEqual("100", row["tfes"])
            self.assertEqual("safe_conflict", row["blend_strategy"])
            self.assertEqual("2", row["runs"])
            self.assertAlmostEqual(6.0, float(row["best_mean"]))
            self.assertAlmostEqual(1.0, float(row["best_std"]))
            self.assertAlmostEqual(5.0, float(row["best_min"]))
            self.assertAlmostEqual(7.0, float(row["best_max"]))
            self.assertEqual("0", row["diagnostics_count"])
            self.assertAlmostEqual(0.0, float(row["rollback_ratio"]))


if __name__ == "__main__":
    unittest.main()
