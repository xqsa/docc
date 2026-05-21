import json
import unittest
from pathlib import Path


class ExternalCppBaselineTests(unittest.TestCase):
    @staticmethod
    def _baseline_root() -> Path:
        return (
            Path(__file__).resolve().parents[1]
            / "external_baselines"
            / "Large-Scale-Overlapping-Optimization"
        )

    def test_manifest_covers_all_functions(self):
        baseline_root = self._baseline_root()
        manifest_path = baseline_root / "benchmark_manifest.json"
        self.assertTrue(manifest_path.exists())

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("external_cpp_baseline", manifest["kind"])
        self.assertEqual(
            "https://github.com/Flyki/Large-Scale-Overlapping-Optimization",
            manifest["source_repo"],
        )

        functions = manifest["functions"]
        self.assertEqual(list(range(1, 13)), [entry["id"] for entry in functions])

        expected_base = {
            **{function_id: "schwefel" for function_id in range(1, 5)},
            **{function_id: "elliptic" for function_id in range(5, 9)},
            **{function_id: "rastrigin" for function_id in range(9, 13)},
        }

        for entry in functions:
            function_id = entry["id"]
            self.assertEqual(expected_base[function_id], entry["base_function"])
            self.assertEqual(
                "conforming" if function_id % 2 == 1 else "conflicting",
                entry["mode"],
            )
            self.assertEqual(905, entry["decision_dimension"])
            self.assertEqual(1000, entry["expanded_subspace_dimension"])
            self.assertEqual(20, entry["subcomponent_count"])
            self.assertEqual(5, entry["adjacent_overlap_size"])
            if function_id % 2 == 1:
                self.assertEqual(905, entry["shift_vector_dimension"])
            else:
                self.assertEqual(1000, entry["shift_vector_dimension"])
            self.assertTrue(entry["rotation_sizes"])
            for relative_path in entry["files"]:
                self.assertTrue((Path(__file__).resolve().parents[1] / relative_path).exists())

    def test_snapshot_keeps_upstream_benchmark_assets(self):
        baseline_root = self._baseline_root()
        cdata_root = baseline_root / "cdatafiles"

        self.assertTrue((baseline_root / "Benchmarks.cpp").exists())
        self.assertTrue((baseline_root / "Benchmarks.h").exists())
        self.assertTrue((baseline_root / "README.md").exists())
        self.assertTrue((baseline_root / "1po.txt").exists())
        self.assertTrue((baseline_root / "1oo.txt").exists())
        self.assertEqual(72, len(list(cdata_root.glob("*"))))


if __name__ == "__main__":
    unittest.main()
