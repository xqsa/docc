"""Thin adapter layer over upstream PyPop7 ES optimizers."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

SUPPORTED_OPTIMIZERS = ("SEPCMAES", "LMMAES", "LMCMA", "MMES", "CMAES")

_OPTIMIZER_IMPORTS = {
    "SEPCMAES": ("pypop7.optimizers.es.sepcmaes", "SEPCMAES"),
    "LMMAES": ("pypop7.optimizers.es.lmmaes", "LMMAES"),
    "LMCMA": ("pypop7.optimizers.es.lmcma", "LMCMA"),
    "MMES": ("pypop7.optimizers.es.mmes", "MMES"),
    "CMAES": ("pypop7.optimizers.es.cmaes", "CMAES"),
}


def _coerce_vector(value: Any, ndim: int, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(ndim, float(array), dtype=float)
    if array.shape != (ndim,):
        raise ValueError(f"{field_name} must be a scalar or shape ({ndim},)")
    return array.astype(float, copy=True)


def _coerce_scalar_fitness(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError("objective must return a scalar fitness value")
    return float(array.reshape(-1)[0])


def _get_optimizer_class(optimizer_name: str):
    normalized_name = optimizer_name.upper()
    if normalized_name not in _OPTIMIZER_IMPORTS:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
    module_name, class_name = _OPTIMIZER_IMPORTS[normalized_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name), normalized_name


def run_pypop7_optimizer(
    optimizer_name: str,
    objective: Callable[[np.ndarray], Any],
    ndim: int,
    lower_bound: Any,
    upper_bound: Any,
    max_function_evaluations: int,
    seed: int,
    x0: Any,
    sigma: float,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    optimizer_class, normalized_name = _get_optimizer_class(optimizer_name)
    options = dict(options or {})
    problem_name = options.pop("problem_name", "unknown")

    lower_boundary = _coerce_vector(lower_bound, ndim, "lower_bound")
    upper_boundary = _coerce_vector(upper_bound, ndim, "upper_bound")
    mean = _coerce_vector(x0, ndim, "x0")

    raw_fitness_curve: list[float] = []

    def wrapped_objective(candidate: np.ndarray) -> float:
        fitness = _coerce_scalar_fitness(objective(candidate))
        raw_fitness_curve.append(fitness)
        return fitness

    problem = {
        "fitness_function": wrapped_objective,
        "ndim_problem": int(ndim),
        "lower_boundary": lower_boundary,
        "upper_boundary": upper_boundary,
    }

    optimizer_options = {
        "max_function_evaluations": int(max_function_evaluations),
        "seed_rng": int(seed),
        "mean": mean,
        "sigma": float(sigma),
        "verbose": False,
    }
    optimizer_options.update(options)

    results = optimizer_class(problem, optimizer_options).optimize()

    return {
        "best_x": np.asarray(results["best_so_far_x"], dtype=float).copy(),
        "best_y": float(results["best_so_far_y"]),
        "n_function_evaluations": int(results["n_function_evaluations"]),
        "runtime": float(results["runtime"]),
        "fitness_curve": list(raw_fitness_curve),
        "optimizer_name": normalized_name,
        "seed": int(seed),
        "problem_name": problem_name,
    }
