import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from HCC_SRC.run_pypop7_baselines import (
    DEFAULT_PROBLEM_CODES,
    _PROBLEM_CACHE,
    build_summary_row,
    get_cached_aob_problem,
    parse_problem_code,
    run_parity_checks,
    run_local_optimizer,
    run_suite,
)


class RunPyPop7BaselinesTests(unittest.TestCase):
    def setUp(self):
        _PROBLEM_CACHE.clear()

    @staticmethod
    def _toy_vectorized_objective(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x ** 2).sum(axis=1)

    def test_parse_problem_code_maps_short_names(self):
        self.assertEqual(parse_problem_code("E4"), ("elliptic", 4, "E4"))
        self.assertEqual(parse_problem_code("S6"), ("schwefel", 6, "S6"))
        self.assertEqual(parse_problem_code("R6"), ("rastrigin", 6, "R6"))
        self.assertEqual(parse_problem_code("A6"), ("ackley", 6, "A6"))

    def test_default_problem_codes_match_current_test_subset(self):
        self.assertEqual(DEFAULT_PROBLEM_CODES, ("E4", "E6", "S4", "S6", "A6", "R6"))

    def test_build_summary_row_uses_required_columns(self):
        row = build_summary_row(
            result={
                "problem_name": "E4",
                "optimizer_name": "CMAES",
                "seed": 11,
                "best_y": 1.25,
                "n_function_evaluations": 123,
                "runtime": 0.75,
                "fitness_curve": [4.0, 2.0, 1.25, 2.5],
            },
            status="ok",
        )

        self.assertEqual(
            row,
            {
                "problem": "E4",
                "optimizer": "CMAES",
                "seed": 11,
                "final_fitness": 2.5,
                "best_fitness": 1.25,
                "fe_used": 123,
                "runtime": 0.75,
                "status": "ok",
            },
        )

    def test_build_summary_row_includes_requested_checkpoints(self):
        row = build_summary_row(
            result={
                "problem_name": "E4",
                "optimizer_name": "CMAES",
                "seed": 11,
                "best_y": 1.25,
                "n_function_evaluations": 4,
                "runtime": 0.75,
                "fitness_curve": [4.0, 2.0, 3.0, 1.25],
            },
            status="ok",
            record_fes=[2, 4, 10],
        )

        self.assertEqual(2.0, row["best_at_2"])
        self.assertEqual(1.25, row["best_at_4"])
        self.assertTrue(row["best_at_10"] != row["best_at_10"])

    def test_run_suite_writes_summary_csv(self):
        fake_result = {
            "best_x": None,
            "best_y": 1.5,
            "n_function_evaluations": 20,
            "runtime": 0.1,
            "fitness_curve": [3.0, 2.0, 1.5],
            "optimizer_name": "CMAES",
            "seed": 1,
            "problem_name": "E4",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("HCC_SRC.run_pypop7_baselines.run_pypop7_optimizer", return_value=fake_result):
                run_suite(
                    output_dir=tmpdir,
                    problem_codes=["E4"],
                    optimizer_names=["CMAES"],
                    seeds=[1],
                    tfes=20,
                    sigma=1.0,
                    run_parity=False,
                )

            summary_path = Path(tmpdir) / "summary.csv"
            self.assertTrue(summary_path.exists())
            contents = summary_path.read_text()
            self.assertIn("problem,optimizer,seed,final_fitness,best_fitness,fe_used,runtime,status", contents)
            self.assertIn("E4,CMAES,1,1.5,1.5,20,0.1,ok", contents)

    def test_run_suite_appends_each_result_before_next_task_starts(self):
        fake_problem = {
            "problem": "E4",
            "function_name": "elliptic",
            "function_id": 4,
            "objective": lambda x: x,
            "ndim": 4,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
            "best": 0.0,
        }
        calls = []

        def fake_optimizer(**kwargs):
            calls.append((kwargs["optimizer_name"], kwargs["seed"]))
            if len(calls) == 2:
                summary_path = Path(tmpdir) / "summary.csv"
                self.assertTrue(summary_path.exists())
                self.assertEqual(len(summary_path.read_text().strip().splitlines()), 2)
            return {
                "best_x": None,
                "best_y": float(len(calls)),
                "n_function_evaluations": kwargs["max_function_evaluations"],
                "runtime": 0.01,
                "fitness_curve": [float(len(calls))],
                "optimizer_name": kwargs["optimizer_name"],
                "seed": kwargs["seed"],
                "problem_name": kwargs["options"]["problem_name"],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("HCC_SRC.run_pypop7_baselines.get_cached_aob_problem", return_value=fake_problem):
                with patch("HCC_SRC.run_pypop7_baselines.run_pypop7_optimizer", side_effect=fake_optimizer):
                    run_suite(
                        output_dir=tmpdir,
                        problem_codes=["E4"],
                        optimizer_names=["CMAES"],
                        seeds=[1, 2],
                        tfes=20,
                        sigma=1.0,
                        run_parity=False,
                    )

            contents = (Path(tmpdir) / "summary.csv").read_text()
            self.assertIn("E4,CMAES,1,1.0,1.0,20,0.01,ok", contents)
            self.assertIn("E4,CMAES,2,2.0,2.0,20,0.01,ok", contents)

    def test_run_suite_resume_skips_existing_summary_rows(self):
        fake_problem = {
            "problem": "E4",
            "function_name": "elliptic",
            "function_id": 4,
            "objective": lambda x: x,
            "ndim": 4,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
            "best": 0.0,
        }
        fake_result = {
            "best_x": None,
            "best_y": 2.5,
            "n_function_evaluations": 20,
            "runtime": 0.2,
            "fitness_curve": [2.5],
            "optimizer_name": "CMAES",
            "seed": 2,
            "problem_name": "E4",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            summary_path.write_text(
                "problem,optimizer,seed,final_fitness,best_fitness,fe_used,runtime,status\n"
                "E4,CMAES,1,1.5,1.5,20,0.1,ok\n"
            )

            with patch("HCC_SRC.run_pypop7_baselines.get_cached_aob_problem", return_value=fake_problem):
                with patch("HCC_SRC.run_pypop7_baselines.run_pypop7_optimizer", return_value=fake_result) as mocked:
                    run_suite(
                        output_dir=tmpdir,
                        problem_codes=["E4"],
                        optimizer_names=["CMAES"],
                        seeds=[1, 2],
                        tfes=20,
                        sigma=1.0,
                        run_parity=False,
                        resume=True,
                    )

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(mocked.call_args.kwargs["seed"], 2)
            contents = summary_path.read_text()
            self.assertIn("E4,CMAES,1,1.5,1.5,20,0.1,ok", contents)
            self.assertIn("E4,CMAES,2,2.5,2.5,20,0.2,ok", contents)

    def test_get_cached_aob_problem_reuses_problem_instance(self):
        fake_problem = {"problem": "E4"}
        _PROBLEM_CACHE.clear()

        with patch("HCC_SRC.run_pypop7_baselines.build_aob_problem", return_value=fake_problem) as mocked:
            first = get_cached_aob_problem("E4")
            second = get_cached_aob_problem("e4")

        self.assertIs(first, second)
        mocked.assert_called_once_with("E4")

    def test_run_parity_checks_writes_status_when_local_side_fails(self):
        fake_problem = {
            "problem": "E4",
            "function_name": "elliptic",
            "function_id": 4,
            "objective": lambda x: x,
            "ndim": 4,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
            "best": 0.0,
        }
        fake_upstream = {
            "best_x": None,
            "best_y": 1.5,
            "n_function_evaluations": 20,
            "runtime": 0.1,
            "fitness_curve": [3.0, 2.0, 1.5],
            "optimizer_name": "MMES",
            "seed": 1,
            "problem_name": "E4",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("HCC_SRC.run_pypop7_baselines.build_aob_problem", return_value=fake_problem):
                with patch("HCC_SRC.run_pypop7_baselines.run_pypop7_optimizer", return_value=fake_upstream):
                    with patch(
                        "HCC_SRC.run_pypop7_baselines.run_local_optimizer",
                        side_effect=RuntimeError("local parity failure"),
                    ):
                        parity_path = run_parity_checks(
                            output_dir=Path(tmpdir),
                            problem_codes=["E4"],
                            seeds=[1],
                            tfes=20,
                            sigma=1.0,
                        )

            contents = parity_path.read_text()
            self.assertIn(
                "problem,optimizer,seed,upstream_final_fitness,local_final_fitness,upstream_best_fitness,local_best_fitness,upstream_fe_used,local_fe_used,upstream_runtime,local_runtime,curve_trend,status",
                contents,
            )
            self.assertIn("E4,MMES,1,1.5,nan,1.5,nan,20,0,0.1,nan,missing_curve,local_error:RuntimeError", contents)

    def test_run_local_mmes_respects_small_fe_budget(self):
        result = run_local_optimizer(
            optimizer_name="MMES",
            objective=self._toy_vectorized_objective,
            ndim=8,
            lower_bound=-5.0,
            upper_bound=5.0,
            max_function_evaluations=4,
            seed=7,
            x0=[0.0] * 8,
            sigma=1.0,
            problem_name="toy",
        )

        self.assertLessEqual(result["n_function_evaluations"], 4)
        self.assertEqual(len(result["fitness_curve"]), result["n_function_evaluations"])
        self.assertEqual(result["best_x"].shape, (8,))

    def test_run_local_cmaes_respects_fe_budget(self):
        result = run_local_optimizer(
            optimizer_name="CMAES",
            objective=self._toy_vectorized_objective,
            ndim=4,
            lower_bound=-5.0,
            upper_bound=5.0,
            max_function_evaluations=5,
            seed=7,
            x0=[0.0] * 4,
            sigma=1.0,
            problem_name="toy",
        )

        self.assertLessEqual(result["n_function_evaluations"], 5)
        self.assertEqual(len(result["fitness_curve"]), result["n_function_evaluations"])
        self.assertEqual(result["best_x"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
