import argparse
import concurrent.futures
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np

from AOB.AOB import Benchmark
from AOB.utils import (
    combine,
    evaluation_record,
    plot_evaluation_curve,
    plot_evaluation_curve_best_so_far,
    remove_overlapping_groups,
)
from HCC.NDAs.MMES.mmes import MMES
from HCC.OPT.CMAES.cmaes import CMAES
from HCC.RDDSM import Decomposition
from experiment_protocols import protocol_choices, resolve_protocol


HCC_ES_METHOD = "hcc_es_original"
METHOD_CHOICES = (HCC_ES_METHOD,)
PROBLEM_PREFIX_MAP = {
    "E": "elliptic",
    "S": "schwefel",
    "R": "rastrigin",
    "A": "ackley",
}
RUN_DETAIL_FIELDNAMES = [
    "problem",
    "method",
    "seed",
    "final_fitness",
    "best_fitness",
    "fe_used",
    "runtime",
    "diagnostics_count",
    "rollback_ratio",
    "conflict_mean",
    "status",
]
SUMMARY_FIELDNAMES = [
    "problem",
    "method",
    "runs",
    "best_mean",
    "best_std",
    "best_min",
    "best_max",
    "diagnostics_count",
    "rollback_ratio",
    "conflict_mean",
]
DIAGNOSTIC_FIELDNAMES = [
    "problem",
    "run_id",
    "method",
]


def parse_problem_code(problem_code):
    normalized = str(problem_code).strip().upper()
    if len(normalized) < 2:
        raise ValueError(f"Invalid problem code: {problem_code}")
    prefix = normalized[0]
    if prefix not in PROBLEM_PREFIX_MAP:
        raise ValueError(f"Unsupported problem prefix: {problem_code}")
    try:
        function_id = int(normalized[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid problem id: {problem_code}") from exc
    return PROBLEM_PREFIX_MAP[prefix], function_id, normalized


def canonicalize_method(method):
    normalized = str(method or HCC_ES_METHOD).strip().lower()
    if normalized != HCC_ES_METHOD:
        raise ValueError(f"Unsupported method: {method}")
    return HCC_ES_METHOD


def stage_seed(seed, offset):
    if seed is None:
        return None
    return int(seed) * 1_000_003 + int(offset)


def safe_overlap_weight(previous_delta, current_delta):
    previous_delta = float(previous_delta)
    current_delta = float(current_delta)
    denominator = previous_delta + current_delta
    if not np.isfinite(previous_delta) or not np.isfinite(current_delta) or not np.isfinite(denominator):
        return 0.5
    if denominator == 0.0:
        return 0.5
    return float(np.clip(previous_delta / denominator, 0.0, 1.0))


def build_overlap_hypergraph(grouping_result):
    var_to_groups = {}
    for group_id, group in enumerate(grouping_result):
        for var_id in group:
            var_to_groups.setdefault(int(var_id), []).append(int(group_id))

    var_to_groups = {var_id: sorted(group_ids) for var_id, group_ids in var_to_groups.items()}
    overlap_vars = sorted(var_id for var_id, group_ids in var_to_groups.items() if len(group_ids) >= 2)
    overlap_var_set = set(overlap_vars)
    group_to_overlap_vars = {
        int(group_id): [int(var_id) for var_id in group if int(var_id) in overlap_var_set]
        for group_id, group in enumerate(grouping_result)
    }
    membership_histogram = {}
    for group_ids in var_to_groups.values():
        membership = len(group_ids)
        membership_histogram[membership] = membership_histogram.get(membership, 0) + 1
    return {
        "var_to_groups": var_to_groups,
        "group_to_overlap_vars": group_to_overlap_vars,
        "overlap_vars": overlap_vars,
        "membership_histogram": membership_histogram,
    }


def build_overlap_features(grouping_result, overlap_hypergraph):
    var_to_groups = overlap_hypergraph.get("var_to_groups", {})
    group_to_overlap_vars = overlap_hypergraph.get("group_to_overlap_vars", {})
    overlap_vars = [int(var_id) for var_id in overlap_hypergraph.get("overlap_vars", [])]
    all_vars = {int(var_id) for group in grouping_result for var_id in group}
    dimension = len(all_vars)

    nonadjacent_overlap_count = 0
    for var_id in overlap_vars:
        group_ids = sorted(int(group_id) for group_id in var_to_groups.get(int(var_id), []))
        if any(abs(group_ids[right] - group_ids[left]) > 1 for left in range(len(group_ids)) for right in range(left + 1, len(group_ids))):
            nonadjacent_overlap_count += 1

    group_overlap_var_count = []
    group_overlap_load = []
    for group_id in range(len(grouping_result)):
        group_vars = [int(var_id) for var_id in group_to_overlap_vars.get(int(group_id), [])]
        group_overlap_var_count.append(len(group_vars))
        overlap_load = 0.0
        for var_id in group_vars:
            overlap_load += max(0, len(var_to_groups.get(int(var_id), ())) - 1)
        group_overlap_load.append(float(overlap_load))

    overlap_ratio = float(len(overlap_vars) / dimension) if dimension else 0.0
    nonadjacent_overlap_ratio = float(nonadjacent_overlap_count / len(overlap_vars)) if overlap_vars else 0.0
    return {
        "overlap_ratio": overlap_ratio,
        "nonadjacent_overlap_ratio": nonadjacent_overlap_ratio,
        "group_overlap_var_count": group_overlap_var_count,
        "group_overlap_load": group_overlap_load,
    }


def compute_original_glofes(original_do, max_fes, has_overlap):
    if not has_overlap:
        return 0
    ratio = float(0.2 + 0.8 * float(original_do))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return int(ratio * int(max_fes))


def blend_overlapping_elements(best_individual, original_best_individual, overlapping_elements, previous_delta, current_delta):
    if len(overlapping_elements) == 0:
        return
    weight = safe_overlap_weight(previous_delta, current_delta)
    best_individual[overlapping_elements] = (
        weight * best_individual[overlapping_elements]
        + (1.0 - weight) * original_best_individual[overlapping_elements]
    )


def best_so_far_curve(curve):
    best = math.inf
    output = []
    for value in curve:
        numeric = float(value)
        if numeric < best:
            best = numeric
        output.append(best)
    return output


def checkpoint_fieldnames(record_fes):
    return [f"best_at_{int(point)}" for point in (record_fes or [])]


def best_at_record_points(curve, record_fes):
    if not record_fes:
        return {}
    best_curve = best_so_far_curve(curve)
    values = {}
    for point in record_fes:
        index = int(point) - 1
        field = f"best_at_{int(point)}"
        values[field] = best_curve[index] if 0 <= index < len(best_curve) else float("nan")
    return values


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_or_nan(values):
    numeric = [float(value) for value in values if np.isfinite(_to_float(value))]
    return float(np.mean(numeric)) if numeric else float("nan")


def _load_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rollback_ratio(diagnostics):
    if not diagnostics:
        return 0.0
    return float(
        np.mean(
            [
                str(row.get("validation_status", "")).strip().lower() == "rejected"
                or str(row.get("rollback", "")).strip().lower() == "true"
                for row in diagnostics
            ]
        )
    )


def conflict_mean(diagnostics):
    return _mean_or_nan([row.get("conflict") for row in diagnostics])


def build_run_detail_row(problem_code, method, seed, curve, runtime, status, diagnostics=None, record_fes=None):
    diagnostics = diagnostics or []
    if curve:
        final_fitness = float(curve[-1])
        best_fitness = float(np.min(curve))
        fe_used = int(len(curve))
    else:
        final_fitness = float("nan")
        best_fitness = float("nan")
        fe_used = 0
    row = {
        "problem": str(problem_code).upper(),
        "method": canonicalize_method(method),
        "seed": int(seed),
        "final_fitness": final_fitness,
        "best_fitness": best_fitness,
        "fe_used": fe_used,
        "runtime": float(runtime),
        "diagnostics_count": int(len(diagnostics)),
        "rollback_ratio": rollback_ratio(diagnostics),
        "conflict_mean": conflict_mean(diagnostics),
        "status": str(status),
    }
    row.update(best_at_record_points(curve, record_fes))
    return row


def build_summary_row(problem_code, method, seed, curve, runtime, status, record_fes=None):
    return build_run_detail_row(problem_code, method, seed, curve, runtime, status, diagnostics=[], record_fes=record_fes)


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def append_csv_rows(path, rows, fieldnames):
    for row in rows:
        append_csv_row(path, row, fieldnames)


def ensure_csv_header(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def read_summary_keys(path):
    path = Path(path)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (str(row["problem"]).upper(), str(row["method"]), int(row["seed"]))
            for row in rows
            if row.get("problem") and row.get("method") and row.get("seed")
        }


def write_aggregate_summary(detail_path, diagnostics_path, summary_path):
    detail_rows = _load_csv_rows(detail_path)
    diagnostics_rows = _load_csv_rows(diagnostics_path)
    diagnostics_by_key = {}
    for row in diagnostics_rows:
        key = (str(row.get("problem", "")).upper(), str(row.get("method", "")))
        if key[0] and key[1]:
            diagnostics_by_key.setdefault(key, []).append(row)

    grouped_rows = {}
    for row in detail_rows:
        key = (str(row.get("problem", "")).upper(), str(row.get("method", "")))
        if key[0] and key[1]:
            grouped_rows.setdefault(key, []).append(row)

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for key in sorted(grouped_rows):
            problem, method = key
            rows = grouped_rows[key]
            diagnostics = diagnostics_by_key.get(key, [])
            best_values = [_to_float(row.get("best_fitness")) for row in rows if np.isfinite(_to_float(row.get("best_fitness")))]
            writer.writerow(
                {
                    "problem": problem,
                    "method": method,
                    "runs": int(len(rows)),
                    "best_mean": float(np.mean(best_values)) if best_values else float("nan"),
                    "best_std": float(np.std(best_values)) if best_values else float("nan"),
                    "best_min": float(np.min(best_values)) if best_values else float("nan"),
                    "best_max": float(np.max(best_values)) if best_values else float("nan"),
                    "diagnostics_count": int(len(diagnostics)),
                    "rollback_ratio": rollback_ratio(diagnostics),
                    "conflict_mean": conflict_mean(diagnostics),
                }
            )


def optimization_task(
    fun_name,
    fun_id,
    output_path,
    best_individual,
    max_fes,
    grouping_result,
    info,
    adjacent_overlapping_elements,
    seed=None,
    method=HCC_ES_METHOD,
    problem_code=None,
):
    time_start = time.time()
    canonicalize_method(method)
    bench = Benchmark(output_path)
    fun = bench.get_function(fun_name, fun_id)
    best_individual = np.asarray(best_individual, dtype=float).copy()
    overlap_hypergraph = build_overlap_hypergraph(grouping_result)
    overlap_features = build_overlap_features(grouping_result, overlap_hypergraph)
    original_do = float(overlap_features["overlap_ratio"])
    has_overlap = bool(overlap_hypergraph["overlap_vars"])
    glofes = compute_original_glofes(original_do, max_fes, has_overlap)
    seed_offset = 0
    current_best_fitness = None
    sum_fes = 0

    if glofes > 0:
        problem_global = {
            "fitness_function": fun,
            "ndim_problem": info["dimension"],
            "lower_boundary": info["lower"] * np.ones((info["dimension"],)),
            "upper_boundary": info["upper"] * np.ones((info["dimension"],)),
        }
        options_global = {
            "max_function_evaluations": int(min(glofes, max_fes)),
            "mean": (best_individual,),
            "sigma": 0.5,
            "is_restart": False,
            "verbose": 1000,
            "seed_rng": stage_seed(seed, seed_offset),
        }
        seed_offset += 1
        optimizer_global = MMES(problem_global, options_global)
        results_global = optimizer_global.optimize()
        best_individual = np.asarray(results_global["best_so_far_x"], dtype=float).copy()
        current_best_fitness = float(results_global["best_so_far_y"])
        sum_fes = int(results_global["n_function_evaluations"])

    if current_best_fitness is None and sum_fes < int(max_fes):
        current_best_fitness = float(np.asarray(fun(best_individual)).reshape(-1)[0])
        sum_fes += 1

    while sum_fes < int(max_fes):
        group_count = len(grouping_result)
        if group_count <= 0:
            break
        remaining_fes = int(max_fes - sum_fes)
        if remaining_fes <= 0:
            break
        uniform_subfes = int(math.ceil(remaining_fes / float(group_count)))
        group_deltas = np.full(group_count, float("nan"), dtype=float)
        for group_id, dims in enumerate(grouping_result):
            sub_fes = min(uniform_subfes, int(max_fes - sum_fes))
            if sub_fes <= 0:
                continue
            dims = list(dims)
            original_best_individual = best_individual.copy()
            original_best_fitness = float(current_best_fitness)
            objective = lambda x_batch, dims=dims, current=best_individual: fun(combine(x_batch, current, dims))
            problem_cc = {
                "fitness_function": objective,
                "ndim_problem": len(dims),
                "lower_boundary": info["lower"] * np.ones((len(dims),)),
                "upper_boundary": info["upper"] * np.ones((len(dims),)),
            }
            options_cc = {
                "max_function_evaluations": int(sub_fes),
                "mean": (best_individual[dims],),
                "sigma": 0.5,
                "is_restart": False,
                "verbose": 1000,
                "early_stopping_evaluations": 1000,
                "seed_rng": stage_seed(seed, seed_offset),
            }
            seed_offset += 1
            optimizer_cc = CMAES(problem_cc, options_cc)
            results_cc = optimizer_cc.optimize()
            best_individual[dims] = np.asarray(results_cc["best_so_far_x"], dtype=float).copy()
            sum_fes += int(results_cc["n_function_evaluations"])
            subspace_candidate_fitness = float(results_cc["best_so_far_y"])
            delta = float(original_best_fitness - subspace_candidate_fitness)
            group_deltas[group_id] = delta
            if group_id > 0:
                overlap_indices = np.asarray(adjacent_overlapping_elements[group_id - 1], dtype=int)
                blend_overlapping_elements(
                    best_individual,
                    original_best_individual,
                    overlap_indices,
                    group_deltas[group_id - 1],
                    group_deltas[group_id],
                )
            current_best_fitness = subspace_candidate_fitness
            if sum_fes >= int(max_fes):
                break

    time_end = time.time()
    return fun.fitness_record, (time_end - time_start), []


def parallel_optimization(
    fun_name,
    fun_id,
    output_path,
    best_individual,
    max_fes,
    cycle_num,
    grouping_result,
    info,
    output_data,
    adjacent_overlapping_elements,
    method=HCC_ES_METHOD,
):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                optimization_task,
                fun_name,
                fun_id,
                output_path,
                best_individual,
                max_fes,
                grouping_result,
                info,
                adjacent_overlapping_elements,
                seed,
                method,
                None,
            )
            for seed in range(1, int(cycle_num) + 1)
        ]
        algorithm = f"{fun_name}_{fun_id}"
        average_time = 0.0
        for future in futures:
            curve, runtime, _ = future.result()
            output_data[algorithm].append(curve)
            average_time += float(runtime)
        return average_time / float(cycle_num)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the original HCC experiments.")
    parser.add_argument(
        "--protocol",
        choices=protocol_choices(),
        default="smoke",
        help="Experiment preset controlling MaxFEs, cycle count, and evaluation checkpoints.",
    )
    parser.add_argument("--problems", nargs="*", help="Problem codes such as E4 E6 S4 S6 R6 A6.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Seed values for independent HCC runs.")
    parser.add_argument("--tfes", type=int, help="Total function evaluations per problem/seed run.")
    parser.add_argument("--method", nargs="+", choices=METHOD_CHOICES, default=[HCC_ES_METHOD], help="Only hcc_es_original is supported.")
    parser.add_argument("--workers", type=int, default=1, help="Number of independent problem/seed workers.")
    parser.add_argument("--record-fes", nargs="*", type=int, default=[], help="Evaluation checkpoints written as best_at_<FE> columns.")
    parser.add_argument(
        "--summary-refresh-every",
        type=int,
        default=1,
        help="Rewrite summary.csv after this many completed runs.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in run_details.csv.")
    parser.add_argument("--output-dir", help="Output directory for problem-code batch runs.")
    return parser.parse_args()


def build_hcc_es_inputs(problem_code):
    fun_name, fun_id, normalized = parse_problem_code(problem_code)
    bench = Benchmark(None)
    file_path = f"HCC_SRC/AOB/AOBG/datafile/F{fun_id}-design.txt"
    design_matrix = np.loadtxt(file_path, delimiter=",")
    decomposition = Decomposition(design_matrix)
    grouping_result = decomposition.decomposition()
    _, _, adjacent_overlapping_elements = remove_overlapping_groups(grouping_result)
    info = bench.get_info(fun_name, fun_id)
    return {
        "problem": normalized,
        "fun_name": fun_name,
        "fun_id": fun_id,
        "grouping_result": grouping_result,
        "info": info,
        "best_individual": np.zeros(info["dimension"]),
        "adjacent_overlapping_elements": adjacent_overlapping_elements,
    }


def run_problem_seed_task(problem_code, seed, tfes, method, output_dir, record_fes=None):
    inputs = build_hcc_es_inputs(problem_code)
    run_output_path = str(Path(output_dir) / inputs["problem"] / f"seed-{seed}") + "/"
    curve, runtime, diagnostics = optimization_task(
        inputs["fun_name"],
        inputs["fun_id"],
        run_output_path,
        inputs["best_individual"],
        int(tfes),
        inputs["grouping_result"],
        inputs["info"],
        inputs["adjacent_overlapping_elements"],
        int(seed),
        method,
        inputs["problem"],
    )
    return {
        "detail": build_run_detail_row(inputs["problem"], method, seed, curve, runtime, "ok", diagnostics, record_fes),
        "diagnostics": diagnostics,
    }


def run_problem_code_batch(args):
    problems = [parse_problem_code(problem)[2] for problem in args.problems]
    seeds = args.seeds or [1, 2, 3]
    tfes = int(args.tfes or resolve_protocol(args.protocol)["max_fes"])
    methods = [canonicalize_method(method) for method in args.method]
    record_fes = list(args.record_fes)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    method_slug = methods[0]
    output_dir = Path(args.output_dir or f"HCC_SRC/result/hcc-es-baselines/{method_slug}-{'-'.join(problems)}-{len(seeds)}seeds-{tfes}-{timestamp}")
    summary_path = output_dir / "summary.csv"
    detail_path = output_dir / "run_details.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    summary_refresh_every = max(1, int(args.summary_refresh_every))
    existing_keys = read_summary_keys(detail_path) if args.resume else set()
    ensure_csv_header(diagnostics_path, DIAGNOSTIC_FIELDNAMES)
    tasks = [
        (problem, method, seed)
        for method in methods
        for problem in problems
        for seed in seeds
        if (problem, method, int(seed)) not in existing_keys
    ]

    print(
        f"Running methods={methods} with problems={problems}, seeds={seeds}, "
        f"tfes={tfes}, workers={args.workers}, output_dir={output_dir}"
    )
    if not tasks:
        write_aggregate_summary(detail_path, diagnostics_path, summary_path)
        print(f"summary.csv -> {summary_path}")
        return

    pending_summary_updates = 0

    def maybe_refresh_summary(force=False):
        nonlocal pending_summary_updates
        if pending_summary_updates == 0:
            if force and not summary_path.exists():
                write_aggregate_summary(detail_path, diagnostics_path, summary_path)
            return
        if not force and pending_summary_updates < summary_refresh_every:
            return
        write_aggregate_summary(detail_path, diagnostics_path, summary_path)
        pending_summary_updates = 0

    if args.workers <= 1:
        for problem, method, seed in tasks:
            try:
                result = run_problem_seed_task(problem, seed, tfes, method, output_dir, record_fes)
            except Exception as exc:
                result = {
                    "detail": build_run_detail_row(problem, method, seed, [], 0.0, f"error: {exc}", [], record_fes),
                    "diagnostics": [],
                }
            append_csv_row(detail_path, result["detail"], RUN_DETAIL_FIELDNAMES + checkpoint_fieldnames(record_fes))
            append_csv_rows(diagnostics_path, result["diagnostics"], DIAGNOSTIC_FIELDNAMES)
            pending_summary_updates += 1
            maybe_refresh_summary()
            print(f"finished {problem} {method} seed={seed}: {result['detail']['status']}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_problem_seed_task, problem, seed, tfes, method, output_dir, record_fes): (problem, method, seed)
                for problem, method, seed in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                problem, method, seed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "detail": build_run_detail_row(problem, method, seed, [], 0.0, f"error: {exc}", [], record_fes),
                        "diagnostics": [],
                    }
                append_csv_row(detail_path, result["detail"], RUN_DETAIL_FIELDNAMES + checkpoint_fieldnames(record_fes))
                append_csv_rows(diagnostics_path, result["diagnostics"], DIAGNOSTIC_FIELDNAMES)
                pending_summary_updates += 1
                maybe_refresh_summary()
                print(f"finished {problem} {method} seed={seed}: {result['detail']['status']}")
    maybe_refresh_summary(force=True)
    print(f"summary.csv -> {summary_path}")


def run_protocol_batch(args):
    protocol = resolve_protocol(args.protocol)
    methods = [canonicalize_method(method) for method in (args.method or [HCC_ES_METHOD])]
    if len(methods) != 1:
        raise ValueError("Protocol batch supports exactly one method.")
    method = methods[0]
    max_fes = int(protocol["max_fes"])
    cycle_num = int(protocol["cycle_num"])
    record_fes = list(protocol["record_fes"])
    fun_name_list = ["elliptic", "schwefel", "rastrigin", "ackley"]

    output_data_map = {
        "schwefel": {f"schwefel_{idx}": [] for idx in range(1, 7)},
        "elliptic": {f"elliptic_{idx}": [] for idx in range(1, 7)},
        "rastrigin": {f"rastrigin_{idx}": [] for idx in range(1, 7)},
        "ackley": {f"ackley_{idx}": [] for idx in range(1, 7)},
    }
    for prefix in output_data_map.values():
        for idx in range(1, 7):
            key = next(name for name in prefix if name.endswith(f"_{idx}"))
            prefix[f"{key}_time"] = []

    print(
        f"Running protocol '{protocol['name']}' with "
        f"MaxFEs={max_fes}, cycle_num={cycle_num}, record_FEs={record_fes}"
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for fun_name in fun_name_list:
        output_path = f"HCC_SRC/result/{timestamp}/{fun_name}/"
        bench = Benchmark(output_path)
        output_data = output_data_map[fun_name]
        for fun_id in range(1, 7):
            file_path = f"HCC_SRC/AOB/AOBG/datafile/F{fun_id}-design.txt"
            design_matrix = np.loadtxt(file_path, delimiter=",")
            decomposition = Decomposition(design_matrix)
            grouping_result = decomposition.decomposition()
            _, _, adjacent_overlapping_elements = remove_overlapping_groups(grouping_result)
            info = bench.get_info(fun_name, fun_id)
            fun = bench.get_function(fun_name, fun_id)
            best_individual = np.zeros(info["dimension"])
            _ = fun(best_individual)[0].copy()

            average_time = parallel_optimization(
                fun_name,
                fun_id,
                output_path,
                best_individual,
                max_fes,
                cycle_num,
                grouping_result,
                info,
                output_data,
                adjacent_overlapping_elements,
                method,
            )
            print(f"{fun_name}_{fun_id} average time: {average_time}")
            output_data[f"{fun_name}_{fun_id}_time"].append(average_time)

        evaluation_record(output_data, output_path, record_FEs_list=record_fes)
        plot_evaluation_curve(output_data, output_path, font_size=12, log_scale=True)
        plot_evaluation_curve_best_so_far(output_data, output_path, font_size=12, log_scale=True, show_variance=True)


def main():
    args = parse_args()
    if args.problems:
        run_problem_code_batch(args)
    else:
        run_protocol_batch(args)


if __name__ == "__main__":
    freeze_support()
    main()
