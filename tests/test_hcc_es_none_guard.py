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
    def test_problem_code_maps_to_aob_function_and_id(self):
        hcc_es = load_hcc_es_module()

        self.assertEqual(("elliptic", 4, "E4"), hcc_es.parse_problem_code("e4"))
        self.assertEqual(("schwefel", 6, "S6"), hcc_es.parse_problem_code("S6"))
        self.assertEqual(("rastrigin", 6, "R6"), hcc_es.parse_problem_code("R6"))
        self.assertEqual(("ackley", 6, "A6"), hcc_es.parse_problem_code("A6"))

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
                    "method": "hcc_es_original",
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
                    "method": "hcc_es_original",
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
            self.assertEqual("hcc_es_original", row["method"])
            self.assertEqual("2", row["runs"])
            self.assertAlmostEqual(6.0, float(row["best_mean"]))
            self.assertAlmostEqual(1.0, float(row["best_std"]))
            self.assertAlmostEqual(5.0, float(row["best_min"]))
            self.assertAlmostEqual(7.0, float(row["best_max"]))
            self.assertEqual("0", row["diagnostics_count"])
            self.assertAlmostEqual(0.0, float(row["rollback_ratio"]))


if __name__ == "__main__":
    unittest.main()
