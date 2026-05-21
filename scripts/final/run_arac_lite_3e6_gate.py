import argparse
import contextlib
import csv
import importlib.util
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "final"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_ARTIFACT_PREFIX = "arac_lite_final_3e6_gate"
RUNS_ROOT = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_runs"
CASE_ROOT = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_cases"
CASE_RUN_DETAILS_ROOT = CASE_ROOT / "run_details"
CASE_RELATION_AUDIT_ROOT = CASE_ROOT / "relation_action_audit"

RUN_DETAILS_PATH = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_run_details.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_summary.csv"
ACTION_DISTRIBUTION_PATH = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_action_distribution.csv"
PROBE_METRICS_PATH = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_probe_metrics.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_relation_action_audit.csv"
REPORT_PATH = ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_report.md"

FINAL_METHOD = "arac-lite-final"
DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_TFES = 3_000_000
DEFAULT_CC_PASS_GROUP_FES = 20
PROBLEM_RUNTIME_PRIORITY = {
    "E6": 600,
    "R6": 550,
    "S6": 500,
    "A6": 450,
    "E4": 400,
    "S4": 300,
}


spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_1",
    SCRIPT_ROOT / "generate_arac_lite_v0_1_artifacts.py",
)
v01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v01)
v01.RUNS_ROOT = RUNS_ROOT


def sanitize_artifact_prefix(value):
    text = str(value or DEFAULT_ARTIFACT_PREFIX).strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return safe or DEFAULT_ARTIFACT_PREFIX


def configure_artifact_paths(artifact_prefix=DEFAULT_ARTIFACT_PREFIX):
    global RUNS_ROOT
    global CASE_ROOT
    global CASE_RUN_DETAILS_ROOT
    global CASE_RELATION_AUDIT_ROOT
    global RUN_DETAILS_PATH
    global SUMMARY_PATH
    global ACTION_DISTRIBUTION_PATH
    global PROBE_METRICS_PATH
    global RELATION_AUDIT_PATH
    global REPORT_PATH

    prefix = sanitize_artifact_prefix(artifact_prefix)
    RUNS_ROOT = ARTIFACTS_ROOT / f"{prefix}_runs"
    CASE_ROOT = ARTIFACTS_ROOT / f"{prefix}_cases"
    CASE_RUN_DETAILS_ROOT = CASE_ROOT / "run_details"
    CASE_RELATION_AUDIT_ROOT = CASE_ROOT / "relation_action_audit"
    RUN_DETAILS_PATH = ARTIFACTS_ROOT / f"{prefix}_run_details.csv"
    SUMMARY_PATH = ARTIFACTS_ROOT / f"{prefix}_summary.csv"
    ACTION_DISTRIBUTION_PATH = ARTIFACTS_ROOT / f"{prefix}_action_distribution.csv"
    PROBE_METRICS_PATH = ARTIFACTS_ROOT / f"{prefix}_probe_metrics.csv"
    RELATION_AUDIT_PATH = ARTIFACTS_ROOT / f"{prefix}_relation_action_audit.csv"
    REPORT_PATH = ARTIFACTS_ROOT / f"{prefix}_report.md"
    v01.RUNS_ROOT = RUNS_ROOT
    return prefix


GATE_RUN_DETAIL_FIELDNAMES = [
    *v01.RUN_DETAIL_FIELDNAMES,
    "total_FEs",
    "extra_FE_ratio",
    "targeted_probe_count",
    "matched_probe_count",
    "recovered_fusion_count",
    "bad_probe_count",
]

SUMMARY_FIELDNAMES = [
    "problem",
    "tfes",
    "runs",
    "ok_runs",
    "expected_runs",
    "ARAC_best_mean",
    "ARAC_best_std",
    "ARAC_best_median",
    "ARAC_final_mean",
    "ARAC_final_std",
    "runtime_mean",
    "runtime_total",
    "total_FEs_mean",
    "extra_FE_ratio_mean",
    "Fusion_count",
    "Freeze_count",
    "Disable_count",
    "targeted_probe_count",
    "matched_probe_count",
    "recovered_fusion_count",
    "bad_probe_count",
    "paper_HCC_mean",
    "paper_HCC_std",
    "gap_vs_paper_HCC",
    "improved_or_not",
]

ACTION_DISTRIBUTION_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "action",
    "action_count",
    "action_share",
    "validation_attempt_count",
    "validation_accept_count",
    "validation_accept_rate",
    "action_delta_mean",
]

PROBE_METRIC_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "signature_candidate_count",
    "targeted_probe_count",
    "targeted_probe_accept_count",
    "targeted_probe_accept_rate",
    "targeted_probe_delta_mean",
    "probe_delta_mean",
    "matched_probe_count",
    "recovered_fusion_count",
    "bad_probe_count",
    "extra_fe_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen ARAC-lite-final 3e6 FEs gate experiment."
    )
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", type=int, default=DEFAULT_TFES)
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fast-gate",
        action="store_true",
        help="Disable heavy diagnostics traces for the 3e6 performance gate without changing ARAC-lite rules.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=DEFAULT_ARTIFACT_PREFIX,
        help="Artifact filename prefix. Use a separate prefix for fast-gate runs to avoid overwriting full diagnostics.",
    )
    parser.add_argument(
        "--reuse-default-cache",
        action="store_true",
        help="When using a non-default artifact prefix, also reuse successful rows from the default gate aggregate cache.",
    )
    parser.add_argument(
        "--schedule",
        choices=("input-order", "longest-first", "balanced"),
        default="input-order",
        help="Pending task order. balanced interleaves problem families; longest-first starts slower families first.",
    )
    parser.add_argument(
        "--paper-reference-csv",
        type=Path,
        help="Optional CSV with problem,paper_HCC_mean,paper_HCC_std columns.",
    )
    parser.add_argument(
        "--write-relation-audit",
        action="store_true",
        help="Also materialize one combined relation-action audit CSV. Disabled by default for 3e6 runs.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only rebuild aggregate CSV/report from case cache; do not run pending cases.",
    )
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def default_run_details_path():
    return ARTIFACTS_ROOT / f"{DEFAULT_ARTIFACT_PREFIX}_run_details.csv"


def finite_values(values):
    result = []
    for value in values:
        number = v01.to_float(value)
        if np.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def stat(values, name):
    values = finite_values(values)
    if values.size == 0:
        return float("nan")
    if name == "mean":
        return float(np.mean(values))
    if name == "std":
        return float(np.std(values))
    if name == "median":
        return float(np.median(values))
    if name == "sum":
        return float(np.sum(values))
    return float("nan")


def safe_int(value, default=0):
    return v01.to_int(value, default=default)


def gap(value, baseline):
    value = v01.to_float(value)
    baseline = v01.to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-300:
        return ""
    return float((value - baseline) / abs(baseline))


def format_metric(value):
    number = v01.to_float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.6e}"


def format_percent(value):
    number = v01.to_float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number * 100.0:+.3f}%"


def load_final_config(cc_pass_group_fes=DEFAULT_CC_PASS_GROUP_FES, fast_gate=False):
    config = v01.hcc_es.load_info_aware_nda_config(CONFIG_ROOT / "arac-lite-final.json", enable=False)
    overrides = {
        "cc_pass_group_fes": max(1, int(cc_pass_group_fes)),
        "cc_min_passes": 3,
    }
    if fast_gate:
        overrides.update(
            {
                "save_diagnostics": False,
                "save_shared_variable_trace": False,
                "save_overlap_blend_trace": False,
                "validation_trace_enabled": True,
                "selector_trace_enabled": False,
                "enable_group_delta_trace": False,
            }
        )
    return replace(config, **overrides).normalized()


def build_case_tasks(args):
    problems = [v01.hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    return [
        {
            "problem": str(problem).upper(),
            "seed": int(seed),
            "tfes": int(args.tfes),
            "method": FINAL_METHOD,
            "cc_pass_group_fes": int(args.cc_pass_group_fes),
            "fast_gate": bool(getattr(args, "fast_gate", False)),
        }
        for problem in problems
        for seed in args.seeds
    ]


def problem_runtime_priority(problem):
    return int(PROBLEM_RUNTIME_PRIORITY.get(str(problem).upper(), 0))


def schedule_tasks(tasks, schedule="input-order"):
    tasks = list(tasks)
    schedule = str(schedule or "input-order")
    if schedule == "input-order":
        return tasks
    by_problem = {}
    for task in sorted(
        tasks,
        key=lambda item: (
            -problem_runtime_priority(item.get("problem")),
            str(item.get("problem", "")).upper(),
            safe_int(item.get("seed")),
        ),
    ):
        by_problem.setdefault(str(task.get("problem", "")).upper(), []).append(task)
    problem_order = sorted(by_problem, key=lambda problem: (-problem_runtime_priority(problem), problem))
    if schedule == "balanced":
        scheduled = []
        while any(by_problem.values()):
            for problem in problem_order:
                if by_problem[problem]:
                    scheduled.append(by_problem[problem].pop(0))
        return scheduled
    if schedule == "longest-first":
        scheduled = []
        for problem in problem_order:
            scheduled.extend(by_problem[problem])
        return scheduled
    return tasks


def safe_case_component(value):
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def task_key(task):
    return (
        str(task["problem"]).upper(),
        safe_int(task["tfes"]),
        safe_int(task["seed"]),
        str(task["method"]),
    )


def run_row_key(row):
    return (
        str(row.get("problem", "")).upper(),
        safe_int(row.get("tfes")),
        safe_int(row.get("seed")),
        str(row.get("method", "")),
    )


def case_file_stem(task):
    return "_".join(
        [
            safe_case_component(task["problem"]),
            f"tfes-{safe_case_component(task['tfes'])}",
            f"seed-{safe_case_component(task['seed'])}",
            safe_case_component(task["method"]),
        ]
    )


def case_run_path(task):
    return CASE_RUN_DETAILS_ROOT / f"{case_file_stem(task)}.csv"


def case_relation_path(task):
    return CASE_RELATION_AUDIT_ROOT / f"{case_file_stem(task)}.csv"


def is_targeted_probe(row):
    return (
        str(row.get("action_candidate", "")) == "Fusion"
        and v01.to_bool(row.get("arac_probe_candidate"))
        and (
            v01.to_bool(row.get("arac_targeted_probe_candidate"))
            or str(row.get("action_reason", "")).startswith("targeted_probe_fusion")
        )
    )


def probe_stats_for_relation_rows(relation_rows):
    stats = {
        "targeted_probe_count": 0,
        "matched_probe_count": 0,
        "recovered_fusion_count": 0,
        "bad_probe_count": 0,
    }
    for row in relation_rows:
        if not is_targeted_probe(row):
            continue
        accepted = v01.to_bool(row.get("validation_accepted"))
        delta = v01.to_float(row.get("validation_delta"))
        problem = str(row.get("problem", "")).upper()
        stats["targeted_probe_count"] += 1
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")):
            stats["matched_probe_count"] += 1
        if accepted and np.isfinite(delta) and delta > 0.0:
            stats["recovered_fusion_count"] += 1
        if problem == "R6" and np.isfinite(delta) and delta <= 0.0:
            stats["bad_probe_count"] += 1
    return stats


def enrich_run_row(run_row, relation_rows):
    row = dict(run_row)
    row["total_FEs"] = safe_int(row.get("tfes"))
    row["extra_FE_ratio"] = v01.to_float(row.get("validation_extra_fe_ratio"))
    row.update(probe_stats_for_relation_rows(relation_rows))
    return row


def write_case_outputs(task, run_row, relation_rows):
    write_csv(case_run_path(task), GATE_RUN_DETAIL_FIELDNAMES, [run_row])
    if relation_rows or not bool(task.get("fast_gate", False)):
        write_csv(case_relation_path(task), v01.RELATION_AUDIT_FIELDNAMES, relation_rows)


def fallback_rows_by_case(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(run_row_key(row), []).append(row)
    return grouped


def read_case_run_rows(tasks):
    rows = []
    for task in tasks:
        rows.extend(read_csv(case_run_path(task)))
    return rows


def read_case_run_outputs(tasks, fallback_run_rows=None):
    fallback_runs = fallback_rows_by_case(fallback_run_rows or [])
    run_rows = []
    for task in tasks:
        key = task_key(task)
        case_runs = read_csv(case_run_path(task))
        if case_runs:
            run_rows.extend(case_runs)
        else:
            run_rows.extend(fallback_runs.get(key, []))
    return run_rows


def iter_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def iter_case_relation_rows(tasks):
    for task in tasks:
        yield from iter_csv_rows(case_relation_path(task))


def write_combined_relation_audit(tasks, path=RELATION_AUDIT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=v01.RELATION_AUDIT_FIELDNAMES)
        writer.writeheader()
        for row in iter_case_relation_rows(tasks):
            writer.writerow({field: row.get(field, "") for field in v01.RELATION_AUDIT_FIELDNAMES})
            count += 1
    return count


def completed_case_keys(rows):
    return {run_row_key(row) for row in rows if str(row.get("status", "")) == "ok"}


def filter_completed_tasks(tasks, completed_keys):
    return [task for task in tasks if task_key(task) not in completed_keys]


def run_case_task(task):
    config = load_final_config(task["cc_pass_group_fes"], fast_gate=bool(task.get("fast_gate", False)))
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            run_row, relation_rows = v01.run_one_case(
                task["method"],
                config,
                task["problem"],
                task["seed"],
                task["tfes"],
            )
    run_row = enrich_run_row(run_row, relation_rows)
    write_case_outputs(task, run_row, relation_rows)
    return run_row


def print_case_progress(run_row):
    print(
        f"{run_row.get('problem')} {run_row.get('method')} seed={run_row.get('seed')} "
        f"tfes={run_row.get('tfes')}: {run_row.get('status')} "
        f"best={format_metric(run_row.get('best_error'))} "
        f"runtime={format_metric(run_row.get('runtime'))} "
        f"extra={format_percent(run_row.get('extra_FE_ratio'))}"
    )


def run_tasks(tasks, workers=1, run_case=run_case_task, executor_cls=ProcessPoolExecutor):
    tasks = list(tasks)
    run_rows = []
    workers = max(1, int(workers or 1))
    if workers <= 1 or len(tasks) <= 1:
        results = map(run_case, tasks)
    else:
        with executor_cls(max_workers=workers) as executor:
            results = executor.map(run_case, tasks)
            for run_row in results:
                run_rows.append(run_row)
                print_case_progress(run_row)
        return run_rows

    for run_row in results:
        run_rows.append(run_row)
        print_case_progress(run_row)
    return run_rows


def load_paper_references(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    refs = {}
    for row in read_csv(path):
        problem = str(row.get("problem", "")).upper()
        if not problem:
            continue
        mean = row.get("paper_HCC_mean", row.get("HCC_mean", row.get("mean")))
        std = row.get("paper_HCC_std", row.get("HCC_std", row.get("std")))
        mean_value = v01.to_float(mean)
        if not np.isfinite(mean_value):
            continue
        refs[problem] = {
            "paper_HCC_mean": float(mean_value),
            "paper_HCC_std": v01.to_float(std),
        }
    return refs


def classify_paper_gap(best_mean, paper_mean):
    value = gap(best_mean, paper_mean)
    if value == "":
        return "reference_missing"
    if value < 0.0:
        return "improved"
    if value > 0.0:
        return "worse"
    return "tie"


def build_summary_rows(run_rows, paper_refs=None):
    paper_refs = dict(paper_refs or {})
    grouped = {}
    for row in run_rows:
        grouped.setdefault((str(row.get("problem", "")).upper(), safe_int(row.get("tfes"))), []).append(row)

    summary_rows = []
    for key in sorted(grouped):
        problem, tfes = key
        rows = grouped[key]
        ok_rows = [row for row in rows if str(row.get("status")) == "ok"]
        best_mean = stat([row.get("best_error") for row in ok_rows], "mean")
        paper_ref = paper_refs.get(problem, {})
        paper_mean = paper_ref.get("paper_HCC_mean", "")
        paper_std = paper_ref.get("paper_HCC_std", "")
        summary_rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "runs": int(len(rows)),
                "ok_runs": int(len(ok_rows)),
                "ARAC_best_mean": best_mean,
                "ARAC_best_std": stat([row.get("best_error") for row in ok_rows], "std"),
                "ARAC_best_median": stat([row.get("best_error") for row in ok_rows], "median"),
                "ARAC_final_mean": stat([row.get("final_error") for row in ok_rows], "mean"),
                "ARAC_final_std": stat([row.get("final_error") for row in ok_rows], "std"),
                "runtime_mean": stat([row.get("runtime") for row in ok_rows], "mean"),
                "runtime_total": stat([row.get("runtime") for row in ok_rows], "sum"),
                "total_FEs_mean": stat([row.get("fe_used") for row in ok_rows], "mean"),
                "extra_FE_ratio_mean": stat([row.get("validation_extra_fe_ratio") for row in ok_rows], "mean"),
                "Fusion_count": int(sum(safe_int(row.get("fusion_count")) for row in ok_rows)),
                "Freeze_count": int(sum(safe_int(row.get("freeze_count")) for row in ok_rows)),
                "Disable_count": int(sum(safe_int(row.get("disable_count")) for row in ok_rows)),
                "targeted_probe_count": int(sum(safe_int(row.get("targeted_probe_count")) for row in ok_rows)),
                "matched_probe_count": int(sum(safe_int(row.get("matched_probe_count")) for row in ok_rows)),
                "recovered_fusion_count": int(sum(safe_int(row.get("recovered_fusion_count")) for row in ok_rows)),
                "bad_probe_count": int(sum(safe_int(row.get("bad_probe_count")) for row in ok_rows)),
                "paper_HCC_mean": paper_mean if np.isfinite(v01.to_float(paper_mean)) else "",
                "paper_HCC_std": paper_std if np.isfinite(v01.to_float(paper_std)) else "",
                "gap_vs_paper_HCC": gap(best_mean, paper_mean),
                "improved_or_not": classify_paper_gap(best_mean, paper_mean),
            }
        )
    return summary_rows


def build_action_distribution_rows(relation_rows):
    grouped = {}
    totals = {}
    for row in relation_rows:
        key = (
            str(row.get("problem", "")).upper(),
            safe_int(row.get("tfes")),
            str(row.get("method", "")),
        )
        action = str(row.get("action_candidate", "") or "Unknown")
        bucket = grouped.setdefault(
            (*key, action),
            {
                "action_count": 0,
                "validation_attempt_count": 0,
                "validation_accept_count": 0,
                "delta_sum": 0.0,
                "delta_count": 0,
            },
        )
        bucket["action_count"] += 1
        if v01.to_bool(row.get("validation_attempted")):
            bucket["validation_attempt_count"] += 1
            if v01.to_bool(row.get("validation_accepted")):
                bucket["validation_accept_count"] += 1
        delta = v01.to_float(row.get("validation_delta"))
        if np.isfinite(delta):
            bucket["delta_sum"] += float(delta)
            bucket["delta_count"] += 1
        totals[key] = totals.get(key, 0) + 1

    rows = []
    for key in sorted(grouped):
        problem, tfes, method, action = key
        bucket = grouped[key]
        total = totals.get((problem, tfes, method), 0)
        attempts = int(bucket["validation_attempt_count"])
        accepts = int(bucket["validation_accept_count"])
        delta_count = int(bucket["delta_count"])
        rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "action": action,
                "action_count": int(bucket["action_count"]),
                "action_share": float(bucket["action_count"] / total) if total else 0.0,
                "validation_attempt_count": attempts,
                "validation_accept_count": accepts,
                "validation_accept_rate": float(accepts / attempts) if attempts else 0.0,
                "action_delta_mean": float(bucket["delta_sum"] / delta_count) if delta_count else float("nan"),
            }
        )
    return rows


def build_probe_metric_rows(relation_rows):
    buckets = {}
    for row in relation_rows:
        if str(row.get("method", "")) != FINAL_METHOD:
            continue
        problem = str(row.get("problem", "")).upper()
        tfes = safe_int(row.get("tfes"))
        bucket = buckets.setdefault(
            (problem, tfes, FINAL_METHOD),
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": FINAL_METHOD,
                "signature_candidate_count": 0,
                "targeted_probe_count": 0,
                "targeted_probe_accept_count": 0,
                "probe_delta_sum": 0.0,
                "probe_delta_count": 0,
                "matched_probe_count": 0,
                "recovered_fusion_count": 0,
                "bad_probe_count": 0,
                "extra_fe_ratio": 0.0,
            },
        )
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")):
            bucket["signature_candidate_count"] += 1
        if not is_targeted_probe(row):
            continue
        accepted = v01.to_bool(row.get("validation_accepted"))
        delta = v01.to_float(row.get("validation_delta"))
        bucket["targeted_probe_count"] += 1
        bucket["targeted_probe_accept_count"] += int(accepted)
        if np.isfinite(delta):
            bucket["probe_delta_sum"] += float(delta)
            bucket["probe_delta_count"] += 1
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")):
            bucket["matched_probe_count"] += 1
        if accepted and np.isfinite(delta) and delta > 0.0:
            bucket["recovered_fusion_count"] += 1
        if problem == "R6" and np.isfinite(delta) and delta <= 0.0:
            bucket["bad_probe_count"] += 1

    rows = []
    for key in sorted(buckets):
        bucket = buckets[key]
        probe_count = int(bucket["targeted_probe_count"])
        delta_count = int(bucket["probe_delta_count"])
        delta_mean = float(bucket["probe_delta_sum"] / delta_count) if delta_count else float("nan")
        rows.append(
            {
                "problem": bucket["problem"],
                "tfes": int(bucket["tfes"]),
                "method": bucket["method"],
                "signature_candidate_count": int(bucket["signature_candidate_count"]),
                "targeted_probe_count": probe_count,
                "targeted_probe_accept_count": int(bucket["targeted_probe_accept_count"]),
                "targeted_probe_accept_rate": float(bucket["targeted_probe_accept_count"] / probe_count)
                if probe_count
                else 0.0,
                "targeted_probe_delta_mean": delta_mean,
                "probe_delta_mean": delta_mean,
                "matched_probe_count": int(bucket["matched_probe_count"]),
                "recovered_fusion_count": int(bucket["recovered_fusion_count"]),
                "bad_probe_count": int(bucket["bad_probe_count"]),
                "extra_fe_ratio": float(bucket["extra_fe_ratio"]),
            }
        )
    return rows


def attach_extra_fe_ratio(metric_rows, summary_rows):
    ratios = {
        (str(row.get("problem", "")).upper(), safe_int(row.get("tfes"))): v01.to_float(
            row.get("extra_FE_ratio_mean", row.get("extra_fe_ratio", 0.0))
        )
        for row in summary_rows
    }
    for row in metric_rows:
        key = (str(row.get("problem", "")).upper(), safe_int(row.get("tfes")))
        row["extra_fe_ratio"] = ratios.get(key, 0.0)
    return metric_rows


def run_sweep(args):
    tasks = build_case_tasks(args)
    fallback_run_rows = read_csv(RUN_DETAILS_PATH)
    if bool(getattr(args, "reuse_default_cache", False)) and RUN_DETAILS_PATH != default_run_details_path():
        fallback_run_rows.extend(read_csv(default_run_details_path()))
    cached_run_rows = read_case_run_rows(tasks)
    completed_keys = completed_case_keys([*fallback_run_rows, *cached_run_rows]) if args.resume else set()
    pending_tasks = schedule_tasks(filter_completed_tasks(tasks, completed_keys), getattr(args, "schedule", "input-order"))
    print(
        f"ARAC-lite-final gate requested cases={len(tasks)} pending={len(pending_tasks)} "
        f"workers={max(1, int(args.workers or 1))} resume={bool(args.resume)} "
        f"fast_gate={bool(getattr(args, 'fast_gate', False))} schedule={getattr(args, 'schedule', 'input-order')}"
    )
    if pending_tasks and not args.summarize_only:
        run_tasks(pending_tasks, workers=args.workers)
    elif pending_tasks and args.summarize_only:
        print(f"summarize-only: leaving {len(pending_tasks)} pending cases untouched")

    return read_case_run_outputs(
        tasks,
        fallback_run_rows=fallback_run_rows if args.resume else None,
    )


def gate_decision(summary_rows, expected_problem_count):
    if any(int(row.get("ok_runs", 0) or 0) < int(row.get("expected_runs", row.get("ok_runs", 0)) or 0) for row in summary_rows):
        return "incomplete: at least one problem is missing successful seeds."
    if any(int(row.get("ok_runs", 0) or 0) == 0 for row in summary_rows):
        return "incomplete: at least one problem has no successful run."
    if len(summary_rows) < expected_problem_count:
        return "incomplete: not all requested problems are represented."
    if any(str(row.get("improved_or_not")) == "reference_missing" for row in summary_rows):
        return "reference_missing: run results are available, but paper HCC references are not verified."
    improved = sum(1 for row in summary_rows if str(row.get("improved_or_not")) == "improved")
    worse = sum(1 for row in summary_rows if str(row.get("improved_or_not")) == "worse")
    if improved >= 4 and worse <= 2:
        return "strong_pass_candidate: at least four problems improve vs paper reference."
    if improved >= 3:
        return "borderline_pass_candidate: three problems improve vs paper reference."
    return "fail_gate_candidate: fewer than three problems improve vs paper reference."


def write_report(run_rows, summary_rows, action_rows, probe_rows, args, paper_refs):
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    expected = len(build_case_tasks(args))
    cached_count = len(run_rows)
    pending_count = max(0, expected - ok_count)
    decision = gate_decision(summary_rows, expected_problem_count=len(set(args.problems)))
    lines = [
        "# ARAC-lite-final 3e6 Gate Experiment",
        "",
        "- Date: 2026-05-21",
        "- Executor: Codex",
        f"- Method: `{FINAL_METHOD}`",
        f"- Problems: `{', '.join(str(item).upper() for item in args.problems)}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- TFEs: `{int(args.tfes)}`",
        f"- cc_pass_group_fes: `{int(args.cc_pass_group_fes)}`",
        f"- Workers requested: `{int(args.workers)}`",
        f"- Fast gate: `{bool(getattr(args, 'fast_gate', False))}`",
        f"- Schedule: `{getattr(args, 'schedule', 'input-order')}`",
        f"- Artifact prefix: `{sanitize_artifact_prefix(getattr(args, 'artifact_prefix', DEFAULT_ARTIFACT_PREFIX))}`",
        f"- Reuse default cache: `{bool(getattr(args, 'reuse_default_cache', False))}`",
        f"- Cached run rows: `{cached_count}`",
        f"- Successful runs: `{ok_count}/{expected}`",
        f"- Pending or failed runs: `{pending_count}`",
        f"- Gate decision: `{decision}`",
        "",
        "## Scope",
        "",
        "This is a decision-gate experiment only. It freezes the current ARAC-lite-final configuration and runs the final method alone at 3e6 FEs. It is not a paper-complete paired comparison; if this gate is promising, HCC-ES original and no-coordination still need to be rerun under the same code, seeds, machine, and budget.",
        "",
    ]
    if bool(getattr(args, "fast_gate", False)):
        lines.extend(
            [
                "## Fast Gate Mode",
                "",
                "Fast gate mode disables heavyweight diagnostic traces while keeping ARAC-lite-final relation-to-action and recovery rules unchanged. Full per-relation action/probe counts may be unavailable for cases that were only run in fast gate mode; use a smaller full-diagnostic rerun for mechanism reporting if this performance gate passes.",
                "",
            ]
        )
    if not paper_refs:
        lines.extend(
            [
                "## Paper Reference Status",
                "",
                "No verified local paper HCC-ES reference table was provided or found by this script. `paper_HCC_mean`, `paper_HCC_std`, and `gap_vs_paper_HCC` are therefore left empty instead of being fabricated.",
                "",
            ]
        )
    else:
        reference_path = args.paper_reference_csv.as_posix() if args.paper_reference_csv else "provided in memory"
        lines.extend(
            [
                "## Paper Reference Status",
                "",
                f"Paper HCC-ES references were loaded from `{reference_path}`. The table is used only for this gate check; paper-quality claims still require same-code baseline reruns.",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            "| problem | runs | ARAC best mean | ARAC best std | ARAC best median | paper HCC mean | gap vs paper | extra FE | Fusion | Freeze | Disable | recovered | bad probe | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {problem} | {ok_runs}/{expected} | {best} | {std} | {median} | {paper} | {gap} | {extra} | {fusion} | {freeze} | {disable} | {recovered} | {bad} | {status} |".format(
                problem=row.get("problem"),
                ok_runs=row.get("ok_runs"),
                expected=row.get("expected_runs", row.get("runs")),
                best=format_metric(row.get("ARAC_best_mean")),
                std=format_metric(row.get("ARAC_best_std")),
                median=format_metric(row.get("ARAC_best_median")),
                paper=format_metric(row.get("paper_HCC_mean")),
                gap=format_percent(row.get("gap_vs_paper_HCC")),
                extra=format_percent(row.get("extra_FE_ratio_mean")),
                fusion=row.get("Fusion_count"),
                freeze=row.get("Freeze_count"),
                disable=row.get("Disable_count"),
                recovered=row.get("recovered_fusion_count"),
                bad=row.get("bad_probe_count"),
                status=row.get("improved_or_not"),
            )
        )
    lines.extend(
        [
            "",
            "## Probe Metrics",
            "",
            "| problem | targeted probes | matched | recovered | bad probes | probe delta mean | extra FE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in probe_rows:
        lines.append(
            "| {problem} | {probe} | {matched} | {recovered} | {bad} | {delta} | {extra} |".format(
                problem=row.get("problem"),
                probe=row.get("targeted_probe_count"),
                matched=row.get("matched_probe_count"),
                recovered=row.get("recovered_fusion_count"),
                bad=row.get("bad_probe_count"),
                delta=format_metric(row.get("probe_delta_mean")),
                extra=format_percent(row.get("extra_fe_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Run details: `{RUN_DETAILS_PATH.relative_to(REPO_ROOT).as_posix()}`",
            f"- Summary: `{SUMMARY_PATH.relative_to(REPO_ROOT).as_posix()}`",
            f"- Action distribution: `{ACTION_DISTRIBUTION_PATH.relative_to(REPO_ROOT).as_posix()}`",
            f"- Probe metrics: `{PROBE_METRICS_PATH.relative_to(REPO_ROOT).as_posix()}`",
            f"- Relation-action case cache: `{CASE_RELATION_AUDIT_ROOT.relative_to(REPO_ROOT).as_posix()}`",
            f"- Combined relation-action audit: `{'enabled: ' + RELATION_AUDIT_PATH.relative_to(REPO_ROOT).as_posix() if args.write_relation_audit else 'disabled by default for 3e6 runs'}`",
            f"- Case cache: `{CASE_ROOT.relative_to(REPO_ROOT).as_posix()}`",
            "",
            "## Next Step Rule",
            "",
            "If this gate shows clear 3e6 benefit after verified paper references are added, the next experiment should rerun HCC-ES original / repo default and no-coordination under the same protocol before any paper claim.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    configure_artifact_paths(args.artifact_prefix)
    paper_refs = load_paper_references(args.paper_reference_csv)
    tasks = build_case_tasks(args)
    run_rows = run_sweep(args)
    summary_rows = build_summary_rows(run_rows, paper_refs)
    expected_runs_by_problem = {}
    for task in tasks:
        key = (str(task["problem"]).upper(), safe_int(task["tfes"]))
        expected_runs_by_problem[key] = expected_runs_by_problem.get(key, 0) + 1
    for row in summary_rows:
        key = (str(row.get("problem", "")).upper(), safe_int(row.get("tfes")))
        row["expected_runs"] = expected_runs_by_problem.get(key, row.get("runs", ""))
    action_rows = build_action_distribution_rows(iter_case_relation_rows(tasks))
    probe_rows = attach_extra_fe_ratio(build_probe_metric_rows(iter_case_relation_rows(tasks)), summary_rows)

    write_csv(RUN_DETAILS_PATH, GATE_RUN_DETAIL_FIELDNAMES, run_rows)
    relation_count = write_combined_relation_audit(tasks) if args.write_relation_audit else ""
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(ACTION_DISTRIBUTION_PATH, ACTION_DISTRIBUTION_FIELDNAMES, action_rows)
    write_csv(PROBE_METRICS_PATH, PROBE_METRIC_FIELDNAMES, probe_rows)
    write_report(run_rows, summary_rows, action_rows, probe_rows, args, paper_refs)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    if args.write_relation_audit:
        print(f"combined relation-action audit rows -> {relation_count}")
    else:
        print(f"combined relation-action audit -> skipped; case audit dir={CASE_RELATION_AUDIT_ROOT}")
    print(f"summary -> {SUMMARY_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    freeze_support()
    main()
