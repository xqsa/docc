import argparse
import csv
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_runs"
CASE_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_cases"
CASE_RUN_DETAILS_ROOT = CASE_ROOT / "run_details"
CASE_RELATION_AUDIT_ROOT = CASE_ROOT / "relation_action_audit"

RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_relation_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_summary.csv"
ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_robustness.csv"
ACTION_DISTRIBUTION_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_action_distribution.csv"
PROBE_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_probe_metrics.csv"
BUDGET_ALIGNMENT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_budget_alignment.csv"
STRESS_RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_delta_stress_run_details.csv"
STRESS_RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_delta_stress_relation_action_audit.csv"
STRESS_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_delta_stress_summary.csv"
STRESS_ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_delta_stress_robustness.csv"
STRESS_PROBE_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_delta_stress_probe_metrics.csv"
STRESS_RECOVERY_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_delta_stress_recovery_metrics.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_8_mechanism_ablation_report.md"

V07_RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_run_details.csv"
V07_RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_relation_action_audit.csv"

DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_STRESS_PROBLEMS = ["R6", "S6", "E6"]
DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_TFES = [5000, 10000, 20000]
DEFAULT_CC_PASS_GROUP_FES = 20

DISABLE_FAST_METHOD = "arac-lite-v0.1-disable-fast"
CANDIDATE_METHOD = "arac-lite-v0.6-targeted-probe"
RANDOM_METHOD = "arac-lite-v0.6-random-probe-same-budget"
ACCEPT_ONLY_METHOD = "arac-lite-v0.6-accept-only-recovery"
ACCEPT_DELTA_METHOD = "arac-lite-v0.6-accept-delta-recovery"
BASELINE_METHODS = ["no-coordination", "validated-selective-conflict", DISABLE_FAST_METHOD]
METHODS = [*BASELINE_METHODS, CANDIDATE_METHOD, RANDOM_METHOD]
STRESS_METHODS = [ACCEPT_ONLY_METHOD, ACCEPT_DELTA_METHOD]
TIE_EPS = 1e-12


spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_7",
    SCRIPT_ROOT / "generate_arac_lite_v0_7_generalization.py",
)
v07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v07)

v06 = v07.v06
v05 = v06.v05
v02 = v06.v02
v01 = v06.v01
v01.RUNS_ROOT = RUNS_ROOT


SUMMARY_FIELDNAMES = list(v07.SUMMARY_FIELDNAMES)
ROBUSTNESS_FIELDNAMES = list(v07.ROBUSTNESS_FIELDNAMES)
ACTION_DISTRIBUTION_FIELDNAMES = list(v07.ACTION_DISTRIBUTION_FIELDNAMES)
PROBE_METRIC_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "signature_candidate_count",
    "probe_count",
    "probe_accept_count",
    "probe_accept_rate",
    "probe_delta_mean",
    "matched_probe_count",
    "recovered_fusion_count",
    "bad_probe_count",
    "extra_fe_ratio",
]
BUDGET_ALIGNMENT_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "phase",
    "target_budget",
    "random_probe_count",
    "budget_gap",
]
RECOVERY_METRIC_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "recovery_candidate_count",
    "recovery_fusion_count",
    "recovery_accept_count",
    "recovery_accept_rate",
    "recovery_delta_mean",
    "bad_recovery_count",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.8 mechanism ablation artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--stress-problems", nargs="+", default=list(DEFAULT_STRESS_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-reuse-v0-7",
        action="store_true",
        help="Rerun all main-matrix methods instead of reusing V0.7 rows for existing methods.",
    )
    parser.add_argument("--skip-stress", action="store_true")
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def stream_csv(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def safe_case_component(value):
    return v07.safe_case_component(value)


def task_key(task):
    return (
        str(task["problem"]).upper(),
        int(task["tfes"]),
        int(task["seed"]),
        str(task["method"]),
    )


def run_row_key(row):
    return (
        str(row.get("problem", "")).upper(),
        v01.to_int(row.get("tfes")),
        v01.to_int(row.get("seed")),
        str(row.get("method", "")),
    )


def row_phase(row):
    return str(row.get("arac_probe_phase") or row.get("arac_recovery_phase") or row.get("phase") or "")


def is_targeted_probe(row):
    return str(row.get("action_candidate", "")) == "Fusion" and v01.to_bool(row.get("arac_probe_candidate")) and (
        v01.to_bool(row.get("arac_targeted_probe_candidate"))
        or str(row.get("action_reason", "")).startswith("targeted_probe_fusion")
    )


def is_random_probe(row):
    return str(row.get("action_candidate", "")) == "Fusion" and v01.to_bool(row.get("arac_probe_candidate")) and (
        str(row.get("arac_probe_reason", "")) == "random_same_budget_probe_selected"
        or str(row.get("action_reason", "")).startswith("random_same_budget_probe_fusion")
    )


def is_probe_action(row):
    method = str(row.get("method", ""))
    if method == CANDIDATE_METHOD:
        return is_targeted_probe(row)
    if method == RANDOM_METHOD:
        return is_random_probe(row)
    return str(row.get("action_candidate", "")) == "Fusion" and v01.to_bool(row.get("arac_probe_candidate"))


def targeted_probe_budget_by_case_phase(relation_rows):
    budget = {}
    for row in relation_rows:
        if str(row.get("method", "")) != CANDIDATE_METHOD:
            continue
        if not is_targeted_probe(row):
            continue
        key = (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("seed")),
            v01.to_int(row.get("tfes")),
            row_phase(row),
        )
        budget[key] = budget.get(key, 0) + 1
    return budget


def targeted_probe_budget_from_v0_7(path=V07_RELATION_AUDIT_PATH):
    return targeted_probe_budget_by_case_phase(stream_csv(path))


def filter_budget_by_args(budget_by_case_phase, args):
    problems = {str(v01.hcc_es.parse_problem_code(problem)[2]).upper() for problem in args.problems}
    seeds = {int(seed) for seed in args.seeds}
    tfes_values = {int(tfes) for tfes in args.tfes}
    return {
        key: int(value)
        for key, value in dict(budget_by_case_phase or {}).items()
        if key[0] in problems and int(key[1]) in seeds and int(key[2]) in tfes_values
    }


def targeted_probe_variant():
    return v06.targeted_probe_variant()


def random_probe_same_budget_variant(budget=0, phase="middle", seed=0):
    overrides = dict(targeted_probe_variant().overrides)
    overrides.update(
        {
            "arac_lite_targeted_probe_enabled": False,
            "arac_lite_random_probe_same_budget_enabled": True,
            "arac_lite_random_probe_budget": max(0, int(budget)),
            "arac_lite_random_probe_phase": str(phase or "middle").strip().lower(),
            "arac_lite_random_probe_seed": int(seed),
            "arac_lite_probe_every_n_pass": 1,
        }
    )
    return v01.ThresholdVariant(RANDOM_METHOD, overrides)


def accept_only_recovery_variant():
    overrides = dict(targeted_probe_variant().overrides)
    overrides.update(
        {
            "arac_lite_targeted_probe_enabled": False,
            "arac_lite_recovery_enabled": True,
            "arac_lite_recovery_min_attempts": 2,
            "arac_lite_recovery_accept_rate_threshold": 0.3,
            "arac_lite_recovery_delta_threshold": -1.0e300,
            "arac_lite_recovery_positive_delta_rate_threshold": 0.0,
            "arac_lite_probe_recovery_min_attempts": 1,
            "arac_lite_probe_recovery_delta_threshold": -1.0e300,
        }
    )
    return v01.ThresholdVariant(ACCEPT_ONLY_METHOD, overrides)


def accept_delta_recovery_variant():
    overrides = dict(accept_only_recovery_variant().overrides)
    overrides.update(
        {
            "arac_lite_recovery_delta_threshold": 0.0,
            "arac_lite_recovery_positive_delta_rate_threshold": 0.25,
            "arac_lite_probe_recovery_delta_threshold": 0.0,
        }
    )
    return v01.ThresholdVariant(ACCEPT_DELTA_METHOD, overrides)


def fixed_method_configs(cc_pass_group_fes, budget_by_case_phase=None):
    base_arac = v01.load_base_arac_config(cc_pass_group_fes)
    return [
        ("no-coordination", v01.load_named_config(CONFIG_ROOT / "no-coordination.json", cc_pass_group_fes)),
        (
            "validated-selective-conflict",
            v01.load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", cc_pass_group_fes),
        ),
        (DISABLE_FAST_METHOD, v01.apply_variant(base_arac, v02.disable_fast_variant())),
        (CANDIDATE_METHOD, v01.apply_variant(base_arac, targeted_probe_variant())),
        (RANDOM_METHOD, v01.apply_variant(base_arac, random_probe_same_budget_variant())),
    ]


def config_for_task(task):
    base_arac = v01.load_base_arac_config(task["cc_pass_group_fes"])
    method = str(task["method"])
    if method == "no-coordination":
        return v01.load_named_config(CONFIG_ROOT / "no-coordination.json", task["cc_pass_group_fes"])
    if method == "validated-selective-conflict":
        return v01.load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", task["cc_pass_group_fes"])
    if method == DISABLE_FAST_METHOD:
        return v01.apply_variant(base_arac, v02.disable_fast_variant())
    if method == CANDIDATE_METHOD:
        return v01.apply_variant(base_arac, targeted_probe_variant())
    if method == RANDOM_METHOD:
        variant = random_probe_same_budget_variant(
            budget=task.get("random_probe_budget", 0),
            phase=task.get("random_probe_phase", "middle"),
            seed=task.get("random_probe_seed", 0),
        )
        return v01.apply_variant(base_arac, variant)
    if method == ACCEPT_ONLY_METHOD:
        return v01.apply_variant(base_arac, accept_only_recovery_variant())
    if method == ACCEPT_DELTA_METHOD:
        return v01.apply_variant(base_arac, accept_delta_recovery_variant())
    raise ValueError(f"unknown V0.8 method: {method}")


def build_case_tasks(args, budget_by_case_phase=None, methods=None, stress=False):
    budget_by_case_phase = dict(budget_by_case_phase or {})
    problems = [v01.hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    selected_methods = list(methods or METHODS)
    tasks = []
    for method_name in selected_methods:
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    phase = "middle"
                    budget = int(budget_by_case_phase.get((str(problem_code).upper(), int(seed), int(tfes), phase), 0))
                    tasks.append(
                        {
                            "problem": str(problem_code).upper(),
                            "seed": int(seed),
                            "tfes": int(tfes),
                            "method": str(method_name),
                            "cc_pass_group_fes": int(args.cc_pass_group_fes),
                            "random_probe_budget": budget,
                            "random_probe_phase": phase,
                            "random_probe_seed": int(seed) + int(tfes) * 100 + sum(ord(ch) for ch in str(problem_code).upper()),
                            "stress": bool(stress),
                        }
                    )
    return tasks


def completed_case_keys(existing_rows):
    return {run_row_key(row) for row in existing_rows if str(row.get("status", "")).lower() == "ok"}


def filter_completed_tasks(tasks, completed_keys):
    return [task for task in tasks if task_key(task) not in completed_keys]


def case_file_stem(task):
    problem, tfes, seed, method_name = task_key(task)
    return "__".join(
        [
            safe_case_component(problem),
            f"tfes-{int(tfes)}",
            f"seed-{int(seed)}",
            safe_case_component(method_name),
        ]
    )


def case_roots_for_task(task):
    prefix = "stress_" if task.get("stress") else ""
    return (
        CASE_RUN_DETAILS_ROOT / f"{prefix}{case_file_stem(task)}.csv",
        CASE_RELATION_AUDIT_ROOT / f"{prefix}{case_file_stem(task)}.csv",
    )


def write_case_outputs(task, run_row, relation_rows):
    run_path, relation_path = case_roots_for_task(task)
    write_csv(run_path, v01.RUN_DETAIL_FIELDNAMES, [run_row])
    write_csv(relation_path, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)


def read_case_run_rows(tasks):
    rows = []
    for task in tasks:
        run_path, _ = case_roots_for_task(task)
        rows.extend(read_csv(run_path))
    return rows


def fallback_rows_by_case(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(run_row_key(row), []).append(row)
    return grouped


def read_case_outputs(tasks, fallback_run_rows=None, fallback_relation_rows=None):
    fallback_runs = fallback_rows_by_case(fallback_run_rows or [])
    fallback_relations = fallback_rows_by_case(fallback_relation_rows or [])
    run_rows = []
    relation_rows = []
    for task in tasks:
        key = task_key(task)
        run_path, relation_path = case_roots_for_task(task)
        case_runs = read_csv(run_path)
        if case_runs:
            run_rows.extend(case_runs)
        else:
            run_rows.extend(fallback_runs.get(key, []))
        case_relations = read_csv(relation_path)
        if case_relations:
            relation_rows.extend(case_relations)
        else:
            relation_rows.extend(fallback_relations.get(key, []))
    return run_rows, relation_rows


def run_case_task(task):
    config = config_for_task(task)
    run_row, relation_rows = v01.run_one_case(
        task["method"],
        config,
        task["problem"],
        task["seed"],
        task["tfes"],
    )
    write_case_outputs(task, run_row, relation_rows)
    return run_row, relation_rows


def print_case_progress(run_row):
    print(
        f"{run_row['problem']} {run_row['method']} seed={run_row['seed']} tfes={run_row['tfes']}: "
        f"{run_row.get('status')} fusion={run_row.get('fusion_count', 0)} "
        f"freeze={run_row.get('freeze_count', 0)} disable={run_row.get('disable_count', 0)}"
    )


def run_tasks(tasks, workers=1, run_case=run_case_task, executor_cls=ProcessPoolExecutor):
    tasks = list(tasks)
    run_rows = []
    relation_rows = []
    workers = max(1, int(workers or 1))
    if workers <= 1 or len(tasks) <= 1:
        results = map(run_case, tasks)
    else:
        with executor_cls(max_workers=workers) as executor:
            results = executor.map(run_case, tasks)
            for run_row, case_relation_rows in results:
                run_rows.append(run_row)
                relation_rows.extend(case_relation_rows)
                print_case_progress(run_row)
        return run_rows, relation_rows
    for run_row, case_relation_rows in results:
        run_rows.append(run_row)
        relation_rows.extend(case_relation_rows)
        print_case_progress(run_row)
    return run_rows, relation_rows


def summarize_runs(run_rows):
    return v07.summarize_runs(run_rows)


def build_robustness_rows(run_rows, candidates=None, baselines=None):
    return v07.build_robustness_rows(
        run_rows,
        candidates=list(candidates or [CANDIDATE_METHOD, RANDOM_METHOD]),
        baselines=list(baselines or [*BASELINE_METHODS, RANDOM_METHOD]),
    )


def build_action_distribution_rows(relation_rows):
    return v07.build_action_distribution_rows(relation_rows)


def stat(values, name):
    return v07.stat(values, name)


def build_probe_metric_rows(relation_rows, offline_candidates=None, methods=None):
    methods = set(methods or [CANDIDATE_METHOD, RANDOM_METHOD, ACCEPT_ONLY_METHOD, ACCEPT_DELTA_METHOD])
    offline_candidates = list(offline_candidates or [])
    offline_keys = {
        (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("seed")),
            v01.to_int(row.get("tfes")),
            str(row.get("phase", "")),
            v01.to_int(row.get("var_id")),
        )
        for row in offline_candidates
    }
    buckets = {}
    for row in relation_rows:
        method = str(row.get("method", ""))
        if method not in methods:
            continue
        problem = str(row.get("problem", "")).upper()
        tfes = v01.to_int(row.get("tfes"))
        bucket = buckets.setdefault(
            (problem, tfes, method),
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "signature_candidate_count": 0,
                "probe_count": 0,
                "probe_accept_count": 0,
                "probe_delta_values": [],
                "matched_probe_count": 0,
                "recovered_fusion_count": 0,
                "bad_probe_count": 0,
            },
        )
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")):
            bucket["signature_candidate_count"] += 1
        if not is_probe_action(row):
            continue
        key = (
            problem,
            v01.to_int(row.get("seed")),
            tfes,
            row_phase(row),
            v01.to_int(row.get("var_id")),
        )
        accepted = v01.to_bool(row.get("validation_accepted"))
        delta = v01.to_float(row.get("validation_delta"))
        bucket["probe_count"] += 1
        bucket["probe_accept_count"] += int(accepted)
        if np.isfinite(delta):
            bucket["probe_delta_values"].append(delta)
        if key in offline_keys:
            bucket["matched_probe_count"] += 1
        if accepted and np.isfinite(delta) and delta > 0.0:
            bucket["recovered_fusion_count"] += 1
        if problem == "R6" and np.isfinite(delta) and delta <= 0.0:
            bucket["bad_probe_count"] += 1

    rows = []
    for key in sorted(buckets):
        bucket = buckets[key]
        probe_count = int(bucket["probe_count"])
        rows.append(
            {
                "problem": bucket["problem"],
                "tfes": int(bucket["tfes"]),
                "method": bucket["method"],
                "signature_candidate_count": int(bucket["signature_candidate_count"]),
                "probe_count": probe_count,
                "probe_accept_count": int(bucket["probe_accept_count"]),
                "probe_accept_rate": float(bucket["probe_accept_count"] / probe_count) if probe_count else 0.0,
                "probe_delta_mean": stat(bucket["probe_delta_values"], "mean"),
                "matched_probe_count": int(bucket["matched_probe_count"]),
                "recovered_fusion_count": int(bucket["recovered_fusion_count"]),
                "bad_probe_count": int(bucket["bad_probe_count"]),
                "extra_fe_ratio": 0.0,
            }
        )
    return rows


def attach_extra_fe_ratio(metric_rows, summary_rows):
    ratio_by_key = {
        (str(row.get("problem", "")).upper(), v01.to_int(row.get("tfes")), str(row.get("method", ""))): v01.to_float(
            row.get("validation_extra_fe_ratio")
        )
        for row in summary_rows
    }
    for row in metric_rows:
        key = (str(row.get("problem", "")).upper(), v01.to_int(row.get("tfes")), str(row.get("method", "")))
        row["extra_fe_ratio"] = ratio_by_key.get(key, 0.0)
    return metric_rows


def build_budget_alignment_rows(budget_by_case_phase, relation_rows):
    actual = {}
    for row in relation_rows:
        if str(row.get("method", "")) != RANDOM_METHOD:
            continue
        if not is_random_probe(row):
            continue
        key = (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("seed")),
            v01.to_int(row.get("tfes")),
            row_phase(row),
        )
        actual[key] = actual.get(key, 0) + 1
    rows = []
    for key in sorted(set(budget_by_case_phase) | set(actual)):
        problem, seed, tfes, phase = key
        target_budget = int(budget_by_case_phase.get(key, 0) or 0)
        random_count = int(actual.get(key, 0) or 0)
        rows.append(
            {
                "problem": problem,
                "seed": int(seed),
                "tfes": int(tfes),
                "phase": phase,
                "target_budget": target_budget,
                "random_probe_count": random_count,
                "budget_gap": int(random_count - target_budget),
            }
        )
    return rows


def build_recovery_metric_rows(relation_rows, methods=None):
    methods = set(methods or STRESS_METHODS)
    buckets = {}
    for row in relation_rows:
        method = str(row.get("method", ""))
        if method not in methods:
            continue
        problem = str(row.get("problem", "")).upper()
        tfes = v01.to_int(row.get("tfes"))
        bucket = buckets.setdefault(
            (problem, tfes, method),
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "recovery_candidate_count": 0,
                "recovery_fusion_count": 0,
                "recovery_accept_count": 0,
                "recovery_deltas": [],
                "bad_recovery_count": 0,
            },
        )
        if v01.to_bool(row.get("arac_recovery_candidate")):
            bucket["recovery_candidate_count"] += 1
        if str(row.get("action_candidate", "")) != "Fusion":
            continue
        if not str(row.get("action_reason", "")).startswith("recovery_fusion"):
            continue
        delta = v01.to_float(row.get("validation_delta"))
        accepted = v01.to_bool(row.get("validation_accepted"))
        bucket["recovery_fusion_count"] += 1
        bucket["recovery_accept_count"] += int(accepted)
        if np.isfinite(delta):
            bucket["recovery_deltas"].append(delta)
            if problem == "R6" and delta <= 0.0:
                bucket["bad_recovery_count"] += 1

    rows = []
    for key in sorted(buckets):
        bucket = buckets[key]
        fusion_count = int(bucket["recovery_fusion_count"])
        rows.append(
            {
                "problem": bucket["problem"],
                "tfes": int(bucket["tfes"]),
                "method": bucket["method"],
                "recovery_candidate_count": int(bucket["recovery_candidate_count"]),
                "recovery_fusion_count": fusion_count,
                "recovery_accept_count": int(bucket["recovery_accept_count"]),
                "recovery_accept_rate": float(bucket["recovery_accept_count"] / fusion_count) if fusion_count else 0.0,
                "recovery_delta_mean": stat(bucket["recovery_deltas"], "mean"),
                "bad_recovery_count": int(bucket["bad_recovery_count"]),
            }
        )
    return rows


def filter_existing_v0_7_rows(args):
    requested_problems = {str(v01.hcc_es.parse_problem_code(problem)[2]).upper() for problem in args.problems}
    requested_seeds = {int(seed) for seed in args.seeds}
    requested_tfes = {int(tfes) for tfes in args.tfes}
    requested_methods = set(BASELINE_METHODS + [CANDIDATE_METHOD])
    runs = [
        row
        for row in read_csv(V07_RUN_DETAILS_PATH)
        if str(row.get("problem", "")).upper() in requested_problems
        and v01.to_int(row.get("seed")) in requested_seeds
        and v01.to_int(row.get("tfes")) in requested_tfes
        and str(row.get("method", "")) in requested_methods
    ]
    relations = [
        row
        for row in stream_csv(V07_RELATION_AUDIT_PATH) or []
        if str(row.get("problem", "")).upper() in requested_problems
        and v01.to_int(row.get("seed")) in requested_seeds
        and v01.to_int(row.get("tfes")) in requested_tfes
        and str(row.get("method", "")) in requested_methods
    ]
    return runs, relations


def run_main_sweep(args, budget_by_case_phase):
    methods_to_run = METHODS if args.no_reuse_v0_7 else [RANDOM_METHOD]
    tasks = build_case_tasks(args, budget_by_case_phase, methods=methods_to_run)
    fallback_run_rows = read_csv(RUN_DETAILS_PATH)
    fallback_relation_rows = read_csv(RELATION_AUDIT_PATH)
    cached_run_rows = read_case_run_rows(tasks)
    completed_keys = completed_case_keys([*fallback_run_rows, *cached_run_rows]) if args.resume else set()
    pending_tasks = filter_completed_tasks(tasks, completed_keys)
    print(
        f"V0.8 main requested cases={len(tasks)} pending={len(pending_tasks)} "
        f"workers={max(1, int(args.workers or 1))} resume={bool(args.resume)} reuse_v0_7={not args.no_reuse_v0_7}"
    )
    if pending_tasks:
        run_tasks(pending_tasks, workers=args.workers)
    run_rows, relation_rows = read_case_outputs(
        tasks,
        fallback_run_rows=fallback_run_rows if args.resume else None,
        fallback_relation_rows=fallback_relation_rows if args.resume else None,
    )
    if not args.no_reuse_v0_7:
        v07_runs, v07_relations = filter_existing_v0_7_rows(args)
        run_rows = [*v07_runs, *run_rows]
        relation_rows = [*v07_relations, *relation_rows]
    return run_rows, relation_rows


def run_stress_sweep(args):
    stress_args = replace_namespace(args, problems=list(args.stress_problems))
    tasks = build_case_tasks(stress_args, {}, methods=STRESS_METHODS, stress=True)
    fallback_run_rows = read_csv(STRESS_RUN_DETAILS_PATH)
    fallback_relation_rows = read_csv(STRESS_RELATION_AUDIT_PATH)
    cached_run_rows = read_case_run_rows(tasks)
    completed_keys = completed_case_keys([*fallback_run_rows, *cached_run_rows]) if args.resume else set()
    pending_tasks = filter_completed_tasks(tasks, completed_keys)
    print(
        f"V0.8 stress requested cases={len(tasks)} pending={len(pending_tasks)} "
        f"workers={max(1, int(args.workers or 1))} resume={bool(args.resume)}"
    )
    if pending_tasks:
        run_tasks(pending_tasks, workers=args.workers)
    return read_case_outputs(
        tasks,
        fallback_run_rows=fallback_run_rows if args.resume else None,
        fallback_relation_rows=fallback_relation_rows if args.resume else None,
    )


def replace_namespace(namespace, **updates):
    values = vars(namespace).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def report_probe_table(rows):
    lines = [
        "| problem | tfes | method | probes | matched | recovered | bad | delta | extra_fe |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["problem"], int(item["tfes"]), item["method"])):
        lines.append(
            "| {problem} | {tfes} | {method} | {probe} | {matched} | {recovered} | {bad} | {delta} | {extra} |".format(
                problem=row["problem"],
                tfes=int(row["tfes"]),
                method=row["method"],
                probe=int(row.get("probe_count", 0) or 0),
                matched=int(row.get("matched_probe_count", 0) or 0),
                recovered=int(row.get("recovered_fusion_count", 0) or 0),
                bad=int(row.get("bad_probe_count", 0) or 0),
                delta=v01.format_metric(row.get("probe_delta_mean")),
                extra=v01.format_percent(row.get("extra_fe_ratio")),
            )
        )
    return "\n".join(lines)


def report_recovery_table(rows):
    lines = [
        "| problem | tfes | method | candidates | recovery_fusion | accepted | bad | delta |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["problem"], int(item["tfes"]), item["method"])):
        lines.append(
            "| {problem} | {tfes} | {method} | {candidates} | {fusion} | {accepted} | {bad} | {delta} |".format(
                problem=row["problem"],
                tfes=int(row["tfes"]),
                method=row["method"],
                candidates=int(row.get("recovery_candidate_count", 0) or 0),
                fusion=int(row.get("recovery_fusion_count", 0) or 0),
                accepted=int(row.get("recovery_accept_count", 0) or 0),
                bad=int(row.get("bad_recovery_count", 0) or 0),
                delta=v01.format_metric(row.get("recovery_delta_mean")),
            )
        )
    return "\n".join(lines)


def report_targeted_vs_random_table(robustness_rows):
    focus = [
        row
        for row in robustness_rows
        if row.get("candidate") == CANDIDATE_METHOD
        and row.get("baseline") == RANDOM_METHOD
        and str(row.get("tfes")) == "all"
    ]
    lines = [
        "| problem | paired | W/L/T | median_gap | worst_gap | IQR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(focus, key=lambda item: item["problem"]):
        lines.append(
            "| {problem} | {paired} | {wins}/{losses}/{ties} | {median} | {worst} | {iqr} |".format(
                problem=row["problem"],
                paired=int(row.get("paired_runs", 0) or 0),
                wins=int(row.get("wins", 0) or 0),
                losses=int(row.get("losses", 0) or 0),
                ties=int(row.get("ties", 0) or 0),
                median=v01.format_percent(row.get("gap_median")),
                worst=v01.format_percent(row.get("worst_case_gap")),
                iqr=v01.format_percent(row.get("gap_iqr")),
            )
        )
    return "\n".join(lines)


def build_report_text(args, main_rows, main_summary, main_robustness, main_probe, budget_rows, stress_rows, stress_robustness, stress_probe, stress_recovery):
    ok_count = sum(1 for row in main_rows if str(row.get("status")) == "ok")
    max_budget_gap = max((abs(int(row.get("budget_gap", 0) or 0)) for row in budget_rows), default=0)
    return "\n".join(
        [
            "# ARAC-lite V0.8 Mechanism Ablation",
            "",
            "- 日期：2026-05-21",
            "- 执行者：Codex",
            "- 结论边界：冻结 V0.6 targeted-probe，不继续调 probe 阈值或动作规则。",
            f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
            f"- Stress problems: {', '.join(str(value).upper() for value in args.stress_problems)}",
            f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
            f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
            f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
            f"- workers: {max(1, int(args.workers or 1))}",
            f"- reuse_v0_7: {not args.no_reuse_v0_7}",
            "",
            "## Main Matrix",
            "",
            f"- Runs: {ok_count}/{len(main_rows)} ok",
            "- Methods: no-coordination, validated-selective-conflict, arac-lite-v0.1-disable-fast, arac-lite-v0.6-targeted-probe, arac-lite-v0.6-random-probe-same-budget。",
            f"- Max absolute random budget gap: {max_budget_gap}",
            "",
            "## Targeted vs Same-Budget Random",
            "",
            report_targeted_vs_random_table(main_robustness),
            "",
            "## Probe Metrics",
            "",
            report_probe_table(main_probe),
            "",
            "## Delta Gate Stress",
            "",
            f"- Stress runs: {sum(1 for row in stress_rows if str(row.get('status')) == 'ok')}/{len(stress_rows)} ok",
            "",
            report_probe_table(stress_probe),
            "",
            "## Delta Gate Recovery Metrics",
            "",
            report_recovery_table(stress_recovery),
            "",
            "## Artifacts",
            "",
            f"- main run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- main relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
            f"- main summary: `{SUMMARY_PATH.as_posix()}`",
            f"- main robustness: `{ROBUSTNESS_PATH.as_posix()}`",
            f"- main probe metrics: `{PROBE_METRICS_PATH.as_posix()}`",
            f"- budget alignment: `{BUDGET_ALIGNMENT_PATH.as_posix()}`",
            f"- stress run details: `{STRESS_RUN_DETAILS_PATH.as_posix()}`",
            f"- stress robustness: `{STRESS_ROBUSTNESS_PATH.as_posix()}`",
            f"- stress recovery metrics: `{STRESS_RECOVERY_METRICS_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
            "",
        ]
    )


def main():
    args = parse_args()
    budget_by_case_phase = filter_budget_by_args(targeted_probe_budget_from_v0_7(), args)
    main_run_rows, main_relation_rows = run_main_sweep(args, budget_by_case_phase)
    main_summary_rows = summarize_runs(main_run_rows)
    main_robustness_rows = build_robustness_rows(main_run_rows)
    main_action_rows = build_action_distribution_rows(main_relation_rows)
    offline_candidates = v06.build_offline_candidates_for_args(args)
    main_probe_rows = attach_extra_fe_ratio(build_probe_metric_rows(main_relation_rows, offline_candidates), main_summary_rows)
    budget_rows = build_budget_alignment_rows(budget_by_case_phase, main_relation_rows)

    if args.skip_stress:
        stress_run_rows = []
        stress_relation_rows = []
        stress_summary_rows = []
        stress_robustness_rows = []
        stress_probe_rows = []
        stress_recovery_rows = []
    else:
        stress_run_rows, stress_relation_rows = run_stress_sweep(args)
        stress_summary_rows = summarize_runs(stress_run_rows)
        stress_robustness_rows = v07.build_robustness_rows(
            stress_run_rows,
            candidates=[ACCEPT_ONLY_METHOD],
            baselines=[ACCEPT_DELTA_METHOD],
        )
        stress_probe_rows = attach_extra_fe_ratio(
            build_probe_metric_rows(stress_relation_rows, offline_candidates, methods=STRESS_METHODS),
            stress_summary_rows,
        )
        stress_recovery_rows = build_recovery_metric_rows(stress_relation_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, main_run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, main_relation_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, main_summary_rows)
    write_csv(ROBUSTNESS_PATH, ROBUSTNESS_FIELDNAMES, main_robustness_rows)
    write_csv(ACTION_DISTRIBUTION_PATH, ACTION_DISTRIBUTION_FIELDNAMES, main_action_rows)
    write_csv(PROBE_METRICS_PATH, PROBE_METRIC_FIELDNAMES, main_probe_rows)
    write_csv(BUDGET_ALIGNMENT_PATH, BUDGET_ALIGNMENT_FIELDNAMES, budget_rows)
    write_csv(STRESS_RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, stress_run_rows)
    write_csv(STRESS_RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, stress_relation_rows)
    write_csv(STRESS_SUMMARY_PATH, SUMMARY_FIELDNAMES, stress_summary_rows)
    write_csv(STRESS_ROBUSTNESS_PATH, ROBUSTNESS_FIELDNAMES, stress_robustness_rows)
    write_csv(STRESS_PROBE_METRICS_PATH, PROBE_METRIC_FIELDNAMES, stress_probe_rows)
    write_csv(STRESS_RECOVERY_METRICS_PATH, RECOVERY_METRIC_FIELDNAMES, stress_recovery_rows)
    REPORT_PATH.write_text(
        build_report_text(
            args,
            main_run_rows,
            main_summary_rows,
            main_robustness_rows,
            main_probe_rows,
            budget_rows,
            stress_run_rows,
            stress_robustness_rows,
            stress_probe_rows,
            stress_recovery_rows,
        ),
        encoding="utf-8",
    )

    print(f"completed main {sum(1 for row in main_run_rows if str(row.get('status')) == 'ok')}/{len(main_run_rows)} runs")
    print(f"main relation-action audit rows -> {len(main_relation_rows)}")
    print(f"main probe metric rows -> {len(main_probe_rows)}")
    print(f"stress runs -> {len(stress_run_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    freeze_support()
    main()
