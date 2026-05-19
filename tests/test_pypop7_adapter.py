import unittest

import numpy as np

from HCC_SRC.baselines.pypop7_adapter import SUPPORTED_OPTIMIZERS, run_pypop7_optimizer


class PyPop7AdapterTests(unittest.TestCase):
    def test_supported_optimizer_names_match_goal(self):
        self.assertEqual(
            SUPPORTED_OPTIMIZERS,
            ("SEPCMAES", "LMMAES", "LMCMA", "MMES", "CMAES"),
        )

    def test_unknown_optimizer_is_rejected(self):
        with self.assertRaises(ValueError):
            run_pypop7_optimizer(
                optimizer_name="NOPE",
                objective=lambda x: float(np.sum(np.square(x))),
                ndim=4,
                lower_bound=-5.0,
                upper_bound=5.0,
                max_function_evaluations=20,
                seed=7,
                x0=np.zeros(4),
                sigma=1.0,
                options={},
            )

    def test_adapter_normalizes_result_shape_for_all_supported_optimizers(self):
        def objective(x):
            return np.array([float(np.sum(np.square(x)))])

        for optimizer_name in SUPPORTED_OPTIMIZERS:
            with self.subTest(optimizer_name=optimizer_name):
                result = run_pypop7_optimizer(
                    optimizer_name=optimizer_name,
                    objective=objective,
                    ndim=4,
                    lower_bound=-5.0,
                    upper_bound=5.0,
                    max_function_evaluations=40,
                    seed=7,
                    x0=np.zeros(4),
                    sigma=1.0,
                    options={"problem_name": "toy-sphere"},
                )

                self.assertEqual(
                    set(result.keys()),
                    {
                        "best_x",
                        "best_y",
                        "n_function_evaluations",
                        "runtime",
                        "fitness_curve",
                        "optimizer_name",
                        "seed",
                        "problem_name",
                    },
                )
                self.assertEqual(result["optimizer_name"], optimizer_name)
                self.assertEqual(result["seed"], 7)
                self.assertEqual(result["problem_name"], "toy-sphere")
                self.assertEqual(result["best_x"].shape, (4,))
                self.assertGreater(len(result["fitness_curve"]), 0)
                self.assertLessEqual(result["n_function_evaluations"], 40)
                self.assertAlmostEqual(result["best_y"], min(result["fitness_curve"]))
                self.assertGreaterEqual(result["runtime"], 0.0)


if __name__ == "__main__":
    unittest.main()
