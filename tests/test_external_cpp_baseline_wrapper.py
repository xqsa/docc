import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_wrapper_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "external_cpp_baseline.py"
    )
    spec = importlib.util.spec_from_file_location("external_cpp_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExternalCppBaselineWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = load_wrapper_module()

    def test_build_compile_plan_uses_vendored_cmaes_sources(self):
        baseline = self.wrapper.baseline_root()
        cmaes = self.wrapper.cmaes_root()
        output_path = Path(tempfile.gettempdir()) / "lsgo-wrapper-test.exe"

        plan = self.wrapper.build_compile_plan(
            baseline=baseline,
            cmaes=cmaes,
            output_path=output_path,
        )

        self.assertIn(baseline / "CBOCC.cpp", plan.cxx_sources)
        self.assertNotIn(baseline / "Header.cpp", plan.cxx_sources)
        self.assertEqual(
            ("cmaes.c", "boundary_transformation.c"),
            tuple(path.name for path in plan.c_sources),
        )
        self.assertEqual(("-std=c++17", "-O2", "-include", "algorithm"), plan.cxx_flags)

    def test_function_one_defaults_to_bundled_partition_files(self):
        baseline = self.wrapper.baseline_root()

        group_file, overlap_file = self.wrapper.resolve_partition_files(
            baseline=baseline,
            function_id=1,
            group_file=None,
            overlap_file=None,
        )

        self.assertEqual("1po.txt", group_file.name)
        self.assertEqual("1oo.txt", overlap_file.name)

    def test_other_functions_require_explicit_partition_files(self):
        baseline = self.wrapper.baseline_root()

        with self.assertRaisesRegex(ValueError, "No bundled partition files exist for F2"):
            self.wrapper.resolve_partition_files(
                baseline=baseline,
                function_id=2,
                group_file=None,
                overlap_file=None,
            )

    def test_prepare_run_workspace_copies_inputs_and_weight_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            baseline = temp_root / "baseline"
            cdata_root = baseline / "cdatafiles"
            cdata_root.mkdir(parents=True)
            (cdata_root / "F1-w.txt").write_text("original-weight\n", encoding="utf-8")
            (cdata_root / "F1-xopt.txt").write_text("0,1,2\n", encoding="utf-8")
            group_file = baseline / "1po.txt"
            overlap_file = baseline / "1oo.txt"
            weight_override = baseline / "weight0.txt"
            group_file.write_text("grouping\n", encoding="utf-8")
            overlap_file.write_text("overlap\n", encoding="utf-8")
            weight_override.write_text("replacement-weight\n", encoding="utf-8")

            run_dir = temp_root / "run"
            self.wrapper.prepare_run_workspace(
                baseline=baseline,
                run_dir=run_dir,
                function_id=1,
                group_file=group_file,
                overlap_file=overlap_file,
                weight_override=weight_override,
            )

            self.assertTrue((run_dir / "cdatafiles" / "F1-xopt.txt").exists())
            self.assertEqual(
                "grouping\n",
                (run_dir / "1po.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                "overlap\n",
                (run_dir / "1oo.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                "replacement-weight\n",
                (run_dir / "cdatafiles" / "F1-w.txt").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
