from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from HCC_SRC.AOB.AOB import Benchmark
    from HCC_SRC.HCC.NDAs.MMES.mmes import MMES as LocalMMES
    from HCC_SRC.HCC.OPT.CMAES.cmaes import CMAES as LocalCMAES
    from HCC_SRC.baselines.pypop7_adapter import SUPPORTED_OPTIMIZERS, run_pypop7_optimizer
except ImportError:
    from AOB.AOB import Benchmark
    from HCC.NDAs.MMES.mmes import MMES as LocalMMES
    from HCC.OPT.CMAES.cmaes import CMAES as LocalCMAES
    from baselines.pypop7_adapter import SUPPORTED_OPTIMIZERS, run_pypop7_optimizer

DEFAULT_PROBLEM_CODES = ("E4", "E6", "S4", "S6", "R6", "A6")
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_TFES = int(1e5)
DEFAULT_SIGMA = 1.0
SUMMARY_FIELDNAMES = ["problem", "optimizer", "seed", "final_fitness", "best_fitness", "fe_used", "runtime", "status"]

_PROBLEM_PREFIX_MAP = {
    "E": "elliptic",
    "S": "schwefel",
    "R": "rastrigin",
    "A": "ackley",
}

_LOCAL_OPTIMIZERS = {
    "MMES": LocalMMES,
    "CMAES": LocalCMAES,
}
_PROBLEM_CACHE: dict[str, dict[str, Any]] = {}


def parse_problem_code(problem_code: str) -> tuple[str, int, str]:
    normalized = problem_code.strip().upper()
    if len(normalized) < 2:
        raise ValueError(f"Invalid problem code: {problem_code}")
    prefix = normalized[0]
    if prefix not in _PROBLEM_PREFIX_MAP:
        raise ValueError(f"Unsupported problem prefix: {problem_code}")
    return _PROBLEM_PREFIX_MAP[prefix], int(normalized[1:]), normalized


def build_aob_problem(problem_code: str) -> dict[str, Any]:
    function_name, function_id, normalized = parse_problem_code(problem_code)
    benchmark = Benchmark(None)
    objective = benchmark.get_function(function_name, function_id)
    info = benchmark.get_info(function_name, function_id)
    return {
        "problem": normalized,
        "function_name": function_name,
        "function_id": function_id,
        "objective": objective,
        "ndim": int(info["dimension"]),
        "lower_bound": float(info["lower"]),
        "upper_bound": float(info["upper"]),
        "best": float(info["best"]),
    }


def get_cached_aob_problem(problem_code: str) -> dict[str, Any]:
    _, _, normalized = parse_problem_code(problem_code)
    if normalized not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[normalized] = build_aob_problem(normalized)
    return _PROBLEM_CACHE[normalized]


def _make_zero_mean(ndim: int) -> np.ndarray:
    return np.zeros(int(ndim), dtype=float)


def checkpoint_fieldnames(record_fes: list[int] | None) -> list[str]:
    return [f"best_at_{int(point)}" for point in (record_fes or [])]


def best_at_record_points(curve: list[float], record_fes: list[int] | None) -> dict[str, float]:
    if not record_fes:
        return {}
    best_curve = _best_so_far_curve(curve)
    values = {}
    for point in record_fes:
        field = f"best_at_{int(point)}"
        index = int(point) - 1
        values[field] = best_curve[index] if 0 <= index < len(best_curve) else math.nan
    return values


def build_summary_row(result: dict[str, Any], status: str, record_fes: list[int] | None = None) -> dict[str, Any]:
    curve = list(result.get("fitness_curve", []))
    final_fitness = curve[-1] if curve else result.get("best_y", math.nan)
    row = {
        "problem": result.get("problem_name", "unknown"),
        "optimizer": result.get("optimizer_name", "unknown"),
        "seed": result.get("seed"),
        "final_fitness": final_fitness,
        "best_fitness": result.get("best_y", math.nan),
        "fe_used": result.get("n_function_evaluations", 0),
        "runtime": result.get("runtime", math.nan),
        "status": status,
    }
    row.update(best_at_record_points(curve, record_fes))
    return row


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        try:
            needs_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def _read_summary_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, int]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                keys.add((row["problem"].upper(), row["optimizer"].upper(), int(row["seed"])))
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def _recording_objective(objective):
    raw_curve: list[float] = []

    def wrapped(candidate):
        value = objective(candidate)
        array = np.asarray(value, dtype=float).reshape(-1)
        raw_curve.extend(array.tolist())
        return array

    return wrapped, raw_curve


def run_local_optimizer(
    optimizer_name: str,
    objective,
    ndim: int,
    lower_bound: float,
    upper_bound: float,
    max_function_evaluations: int,
    seed: int,
    x0,
    sigma: float,
    problem_name: str,
) -> dict[str, Any]:
    if optimizer_name not in _LOCAL_OPTIMIZERS:
        raise ValueError(f"Unsupported local optimizer: {optimizer_name}")

    wrapped_objective, raw_curve = _recording_objective(objective)
    problem = {
        "fitness_function": wrapped_objective,
        "ndim_problem": int(ndim),
        "lower_boundary": np.full(int(ndim), lower_bound, dtype=float),
        "upper_boundary": np.full(int(ndim), upper_bound, dtype=float),
    }
    options = {
        "max_function_evaluations": int(max_function_evaluations),
        "seed_rng": int(seed),
        "mean": np.asarray(x0, dtype=float),
        "sigma": float(sigma),
        "verbose": False,
    }

    optimizer = _LOCAL_OPTIMIZERS[optimizer_name](problem, options)
    results = optimizer.optimize()
    return {
        "best_x": np.asarray(results["best_so_far_x"], dtype=float).copy(),
        "best_y": float(results["best_so_far_y"]),
        "n_function_evaluations": int(results["n_function_evaluations"]),
        "runtime": float(results["runtime"]),
        "fitness_curve": list(raw_curve),
        "optimizer_name": optimizer_name,
        "seed": int(seed),
        "problem_name": problem_name,
    }


def _best_so_far_curve(curve: list[float]) -> list[float]:
    best = math.inf
    out = []
    for value in curve:
        if value < best:
            best = value
        out.append(best)
    return out


def compare_curve_trend(curve_a: list[float], curve_b: list[float]) -> str:
    if not curve_a or not curve_b:
        return "missing_curve"
    best_a = _best_so_far_curve(curve_a)
    best_b = _best_so_far_curve(curve_b)
    improve_a = best_a[0] / max(best_a[-1], 1e-300)
    improve_b = best_b[0] / max(best_b[-1], 1e-300)
    if improve_a <= 0 or improve_b <= 0:
        return "mixed"
    delta = abs(math.log10(max(improve_a, 1e-300)) - math.log10(max(improve_b, 1e-300)))
    return "consistent" if delta <= 1.5 else "mixed"


def run_parity_checks(
    output_dir: Path,
    problem_codes: list[str],
    seeds: list[int],
    tfes: int,
    sigma: float,
) -> Path:
    parity_rows = []
    for optimizer_name in ("MMES", "CMAES"):
        for problem_code in problem_codes:
            for seed in seeds:
                problem = build_aob_problem(problem_code)
                x0 = _make_zero_mean(problem["ndim"])

                row = {
                    "problem": problem["problem"],
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "upstream_final_fitness": math.nan,
                    "local_final_fitness": math.nan,
                    "upstream_best_fitness": math.nan,
                    "local_best_fitness": math.nan,
                    "upstream_fe_used": 0,
                    "local_fe_used": 0,
                    "upstream_runtime": math.nan,
                    "local_runtime": math.nan,
                    "curve_trend": "missing_curve",
                    "status": "ok",
                }

                try:
                    upstream = run_pypop7_optimizer(
                        optimizer_name=optimizer_name,
                        objective=problem["objective"],
                        ndim=problem["ndim"],
                        lower_bound=problem["lower_bound"],
                        upper_bound=problem["upper_bound"],
                        max_function_evaluations=tfes,
                        seed=seed,
                        x0=x0,
                        sigma=sigma,
                        options={"problem_name": problem["problem"]},
                    )
                    row["upstream_final_fitness"] = upstream["fitness_curve"][-1] if upstream["fitness_curve"] else math.nan
                    row["upstream_best_fitness"] = upstream["best_y"]
                    row["upstream_fe_used"] = upstream["n_function_evaluations"]
                    row["upstream_runtime"] = upstream["runtime"]
                except Exception as exc:
                    row["status"] = f"upstream_error:{type(exc).__name__}"
                    parity_rows.append(row)
                    continue

                try:
                    local_problem = build_aob_problem(problem_code)
                    local = run_local_optimizer(
                        optimizer_name=optimizer_name,
                        objective=local_problem["objective"],
                        ndim=local_problem["ndim"],
                        lower_bound=local_problem["lower_bound"],
                        upper_bound=local_problem["upper_bound"],
                        max_function_evaluations=tfes,
                        seed=seed,
                        x0=x0,
                        sigma=sigma,
                        problem_name=local_problem["problem"],
                    )
                    row["local_final_fitness"] = local["fitness_curve"][-1] if local["fitness_curve"] else math.nan
                    row["local_best_fitness"] = local["best_y"]
                    row["local_fe_used"] = local["n_function_evaluations"]
                    row["local_runtime"] = local["runtime"]
                    row["curve_trend"] = compare_curve_trend(local["fitness_curve"], upstream["fitness_curve"])
                except Exception as exc:
                    row["status"] = f"local_error:{type(exc).__name__}"

                parity_rows.append(row)

    parity_path = output_dir / "parity_check.csv"
    _write_csv(
        parity_path,
        [
            "problem",
            "optimizer",
            "seed",
            "upstream_final_fitness",
            "local_final_fitness",
            "upstream_best_fitness",
            "local_best_fitness",
            "upstream_fe_used",
            "local_fe_used",
            "upstream_runtime",
            "local_runtime",
            "curve_trend",
            "status",
        ],
        parity_rows,
    )
    return parity_path


def _build_suite_tasks(
    problem_codes: list[str],
    optimizer_names: list[str],
    seeds: list[int],
    tfes: int,
    sigma: float,
    record_fes: list[int] | None = None,
) -> list[dict[str, Any]]:
    tasks = []
    for problem_code in problem_codes:
        _, _, normalized_problem = parse_problem_code(problem_code)
        for optimizer_name in optimizer_names:
            normalized_optimizer = optimizer_name.upper()
            for seed in seeds:
                tasks.append(
                    {
                        "problem_code": normalized_problem,
                        "optimizer_name": normalized_optimizer,
                        "seed": int(seed),
                        "tfes": int(tfes),
                        "sigma": float(sigma),
                        "record_fes": list(record_fes or []),
                    }
                )
    return tasks


def _task_key(task: dict[str, Any]) -> tuple[str, str, int]:
    return (str(task["problem_code"]).upper(), str(task["optimizer_name"]).upper(), int(task["seed"]))


def _run_suite_task(task: dict[str, Any]) -> dict[str, Any]:
    problem = get_cached_aob_problem(task["problem_code"])
    x0 = _make_zero_mean(problem["ndim"])
    optimizer_name = str(task["optimizer_name"]).upper()
    seed = int(task["seed"])
    try:
        result = run_pypop7_optimizer(
            optimizer_name=optimizer_name,
            objective=problem["objective"],
            ndim=problem["ndim"],
            lower_bound=problem["lower_bound"],
            upper_bound=problem["upper_bound"],
            max_function_evaluations=int(task["tfes"]),
            seed=seed,
            x0=x0,
            sigma=float(task["sigma"]),
            options={"problem_name": problem["problem"]},
        )
        status = "ok"
    except Exception as exc:
        result = {
            "problem_name": problem["problem"],
            "optimizer_name": optimizer_name,
            "seed": seed,
            "best_y": math.nan,
            "n_function_evaluations": 0,
            "runtime": math.nan,
            "fitness_curve": [],
        }
        status = f"error:{type(exc).__name__}"
    return build_summary_row(result, status, task.get("record_fes"))


def run_suite(
    output_dir: str | Path,
    problem_codes: list[str],
    optimizer_names: list[str],
    seeds: list[int],
    tfes: int,
    sigma: float,
    run_parity: bool = True,
    workers: int = 1,
    resume: bool = False,
    record_fes: list[int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if not resume and summary_path.exists():
        summary_path.unlink()

    completed_keys = _read_summary_keys(summary_path) if resume else set()
    tasks = [
        task
        for task in _build_suite_tasks(problem_codes, optimizer_names, seeds, tfes, sigma, record_fes)
        if _task_key(task) not in completed_keys
    ]

    workers = max(1, int(workers))
    if workers == 1:
        for task in tasks:
            _append_csv_row(summary_path, SUMMARY_FIELDNAMES + checkpoint_fieldnames(record_fes), _run_suite_task(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_suite_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                _append_csv_row(summary_path, SUMMARY_FIELDNAMES + checkpoint_fieldnames(record_fes), future.result())

    if not summary_path.exists():
        _write_csv(summary_path, SUMMARY_FIELDNAMES + checkpoint_fieldnames(record_fes), [])

    parity_path = None
    if run_parity:
        parity_path = run_parity_checks(output_dir, problem_codes, seeds, tfes, sigma)

    return {
        "summary_path": str(summary_path),
        "parity_path": str(parity_path) if parity_path is not None else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyPop7 baselines on HCC AOB benchmarks.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEM_CODES))
    parser.add_argument("--optimizers", nargs="+", default=list(SUPPORTED_OPTIMIZERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", type=int, default=DEFAULT_TFES)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--record-fes", nargs="*", type=int, default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_dir = Path("HCC_SRC") / "result" / "pypop7-baselines" / timestamp
    else:
        output_dir = Path(args.output_dir)

    result = run_suite(
        output_dir=output_dir,
        problem_codes=list(args.problems),
        optimizer_names=[name.upper() for name in args.optimizers],
        seeds=list(args.seeds),
        tfes=int(args.tfes),
        sigma=float(args.sigma),
        run_parity=not args.skip_parity,
        workers=int(args.workers),
        resume=bool(args.resume),
        record_fes=list(args.record_fes),
    )
    print(f"summary.csv -> {result['summary_path']}")
    if result["parity_path"]:
        print(f"parity_check.csv -> {result['parity_path']}")


if __name__ == "__main__":
    main()
