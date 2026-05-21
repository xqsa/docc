import argparse
import csv
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_runs"
CASE_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_cases"
CASE_RUN_DETAILS_ROOT = CASE_ROOT / "run_details"
CASE_RELATION_AUDIT_ROOT = CASE_ROOT / "relation_action_audit"

RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_relation_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_summary.csv"
ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_robustness.csv"
RANK_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_rank_summary.csv"
ACTION_DISTRIBUTION_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_action_distribution.csv"
PROBE_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_probe_metrics.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_7_generalization_report.md"

DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_TFES = [5000, 10000, 20000]
DEFAULT_CC_PASS_GROUP_FES = 20

DISABLE_FAST_METHOD = "arac-lite-v0.1-disable-fast"
CANDIDATE_METHOD = "arac-lite-v0.6-targeted-probe"
ABLATION_METHOD = "arac-lite-v0.6-no-delta-hard-block"
BASELINE_METHODS = ["no-coordination", "validated-selective-conflict", DISABLE_FAST_METHOD]
METHODS = [*BASELINE_METHODS, CANDIDATE_METHOD, ABLATION_METHOD]
TIE_EPS = 1e-12


spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_6",
    SCRIPT_ROOT / "generate_arac_lite_v0_6_targeted_probe_artifacts.py",
)
v06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v06)

v05 = v06.v05
v04 = v06.v04
v03 = v06.v03
v02 = v06.v02
v01 = v06.v01
v01.RUNS_ROOT = RUNS_ROOT


SUMMARY_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "runs",
    "best_mean",
    "best_std",
    "best_median",
    "best_q1",
    "best_q3",
    "best_iqr",
    "final_mean",
    "final_std",
    "gap_vs_no_coordination",
    "gap_vs_validated",
    "gap_vs_disable_fast",
    "fusion_count",
    "freeze_count",
    "disable_count",
    "fusion_validation_accept_rate",
    "validation_accept_rate",
    "validation_extra_fe_ratio",
    "cc_pass_count_mean",
    "relation_history_size_mean",
]

ROBUSTNESS_FIELDNAMES = [
    "problem",
    "tfes",
    "candidate",
    "baseline",
    "paired_runs",
    "wins",
    "losses",
    "ties",
    "non_worse_count",
    "non_worse_rate",
    "gap_mean",
    "gap_std",
    "gap_median",
    "gap_q1",
    "gap_q3",
    "gap_iqr",
    "worst_case_gap",
    "best_case_gap",
]

RANK_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "rank",
    "best_mean",
    "rank_mean",
    "rank_best",
    "rank_worst",
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
    "targeted_candidate_count",
    "targeted_probe_count",
    "targeted_probe_accept_count",
    "targeted_probe_accept_rate",
    "targeted_probe_delta_mean",
    "matched_probe_count",
    "recovered_fusion_count",
    "bad_probe_count",
    "extra_fe_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.7 fixed-candidate generalization artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel case workers.")
    parser.add_argument("--resume", action="store_true", help="Skip requested cases that already finished with status ok.")
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


def finite_values(values):
    return np.asarray([v01.to_float(value) for value in values if np.isfinite(v01.to_float(value))], dtype=float)


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
    if name == "q1":
        return float(np.percentile(values, 25))
    if name == "q3":
        return float(np.percentile(values, 75))
    if name == "iqr":
        return float(np.percentile(values, 75) - np.percentile(values, 25))
    if name == "min":
        return float(np.min(values))
    if name == "max":
        return float(np.max(values))
    return float("nan")


def gap(value, baseline):
    value = v01.to_float(value)
    baseline = v01.to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-300:
        return float("nan")
    return float((value - baseline) / abs(baseline))


def targeted_probe_variant():
    return v06.targeted_probe_variant()


def no_delta_hard_block_variant():
    overrides = dict(targeted_probe_variant().overrides)
    overrides["arac_lite_targeted_probe_min_relation_delta"] = -1.0e300
    overrides["arac_lite_targeted_probe_min_positive_delta_rate"] = 0.0
    return v01.ThresholdVariant(ABLATION_METHOD, overrides)


def fixed_method_configs(cc_pass_group_fes):
    base_arac = v01.load_base_arac_config(cc_pass_group_fes)
    return [
        ("no-coordination", v01.load_named_config(CONFIG_ROOT / "no-coordination.json", cc_pass_group_fes)),
        (
            "validated-selective-conflict",
            v01.load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", cc_pass_group_fes),
        ),
        (DISABLE_FAST_METHOD, v01.apply_variant(base_arac, v02.disable_fast_variant())),
        (CANDIDATE_METHOD, v01.apply_variant(base_arac, targeted_probe_variant())),
        (ABLATION_METHOD, v01.apply_variant(base_arac, no_delta_hard_block_variant())),
    ]


def config_for_method(method_name, cc_pass_group_fes):
    for candidate_name, config in fixed_method_configs(cc_pass_group_fes):
        if candidate_name == method_name:
            return config
    raise ValueError(f"unknown V0.7 method: {method_name}")


def safe_case_component(value):
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


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


def build_case_tasks(args):
    problems = [v01.hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    methods = [method_name for method_name, _ in fixed_method_configs(args.cc_pass_group_fes)]
    tasks = []
    for method_name in methods:
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    tasks.append(
                        {
                            "problem": str(problem_code).upper(),
                            "seed": int(seed),
                            "tfes": int(tfes),
                            "method": str(method_name),
                            "cc_pass_group_fes": int(args.cc_pass_group_fes),
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


def case_run_path(task):
    return CASE_RUN_DETAILS_ROOT / f"{case_file_stem(task)}.csv"


def case_relation_path(task):
    return CASE_RELATION_AUDIT_ROOT / f"{case_file_stem(task)}.csv"


def write_case_outputs(task, run_row, relation_rows):
    write_csv(case_run_path(task), v01.RUN_DETAIL_FIELDNAMES, [run_row])
    write_csv(case_relation_path(task), v01.RELATION_AUDIT_FIELDNAMES, relation_rows)


def fallback_rows_by_case(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(run_row_key(row), []).append(row)
    return grouped


def read_case_run_rows(tasks):
    rows = []
    for task in tasks:
        case_rows = read_csv(case_run_path(task))
        if case_rows:
            rows.extend(case_rows)
    return rows


def read_case_outputs(tasks, fallback_run_rows=None, fallback_relation_rows=None):
    fallback_runs = fallback_rows_by_case(fallback_run_rows or [])
    fallback_relations = fallback_rows_by_case(fallback_relation_rows or [])
    run_rows = []
    relation_rows = []
    for task in tasks:
        key = task_key(task)
        case_runs = read_csv(case_run_path(task))
        if case_runs:
            run_rows.extend(case_runs)
        else:
            run_rows.extend(fallback_runs.get(key, []))

        case_relations = read_csv(case_relation_path(task))
        if case_relations:
            relation_rows.extend(case_relations)
        else:
            relation_rows.extend(fallback_relations.get(key, []))
    return run_rows, relation_rows


def run_case_task(task):
    config = config_for_method(task["method"], task["cc_pass_group_fes"])
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
    grouped = {}
    for row in run_rows:
        grouped.setdefault((row["problem"], int(row["tfes"]), row["method"]), []).append(row)

    baselines = {}
    for (problem, tfes, method), rows in grouped.items():
        if method in BASELINE_METHODS:
            baselines[(problem, tfes, method)] = stat([row.get("best_error") for row in rows], "mean")

    summary_rows = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_values = [row.get("best_error") for row in rows]
        final_values = [row.get("final_error") for row in rows]
        best_mean = stat(best_values, "mean")
        summary_rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "runs": int(len(rows)),
                "best_mean": best_mean,
                "best_std": stat(best_values, "std"),
                "best_median": stat(best_values, "median"),
                "best_q1": stat(best_values, "q1"),
                "best_q3": stat(best_values, "q3"),
                "best_iqr": stat(best_values, "iqr"),
                "final_mean": stat(final_values, "mean"),
                "final_std": stat(final_values, "std"),
                "gap_vs_no_coordination": gap(best_mean, baselines.get((problem, tfes, "no-coordination"))),
                "gap_vs_validated": gap(best_mean, baselines.get((problem, tfes, "validated-selective-conflict"))),
                "gap_vs_disable_fast": gap(best_mean, baselines.get((problem, tfes, DISABLE_FAST_METHOD))),
                "fusion_count": int(sum(int(row.get("fusion_count", 0) or 0) for row in rows)),
                "freeze_count": int(sum(int(row.get("freeze_count", 0) or 0) for row in rows)),
                "disable_count": int(sum(int(row.get("disable_count", 0) or 0) for row in rows)),
                "fusion_validation_accept_rate": stat([row.get("fusion_validation_accept_rate") for row in rows], "mean"),
                "validation_accept_rate": stat([row.get("validation_accept_rate") for row in rows], "mean"),
                "validation_extra_fe_ratio": stat([row.get("validation_extra_fe_ratio") for row in rows], "mean"),
                "cc_pass_count_mean": stat([row.get("cc_pass_count") for row in rows], "mean"),
                "relation_history_size_mean": stat([row.get("relation_history_size") for row in rows], "mean"),
            }
        )
    return summary_rows


def build_robustness_rows(run_rows, candidates=None, baselines=None):
    candidates = list(candidates or [CANDIDATE_METHOD, ABLATION_METHOD])
    baselines = list(baselines or BASELINE_METHODS)
    by_case = {
        (row["problem"], int(row["tfes"]), int(row["seed"]), row["method"]): row
        for row in run_rows
    }
    problems = sorted({row["problem"] for row in run_rows})
    tfes_values = sorted({int(row["tfes"]) for row in run_rows})
    rows = []
    for problem in problems:
        for tfes_key in [*tfes_values, "all"]:
            selected_tfes = tfes_values if tfes_key == "all" else [tfes_key]
            for candidate_method in candidates:
                for baseline in baselines:
                    gaps = []
                    for tfes in selected_tfes:
                        seeds = sorted(
                            {
                                int(row["seed"])
                                for row in run_rows
                                if row["problem"] == problem and int(row["tfes"]) == int(tfes)
                            }
                        )
                        for seed in seeds:
                            candidate = by_case.get((problem, int(tfes), int(seed), candidate_method))
                            base = by_case.get((problem, int(tfes), int(seed), baseline))
                            if not candidate or not base:
                                continue
                            value = gap(candidate.get("best_error"), base.get("best_error"))
                            if np.isfinite(value):
                                gaps.append(value)
                    wins = sum(1 for value in gaps if value < -TIE_EPS)
                    ties = sum(1 for value in gaps if abs(value) <= TIE_EPS)
                    losses = sum(1 for value in gaps if value > TIE_EPS)
                    rows.append(
                        {
                            "problem": problem,
                            "tfes": tfes_key,
                            "candidate": candidate_method,
                            "baseline": baseline,
                            "paired_runs": int(len(gaps)),
                            "wins": int(wins),
                            "losses": int(losses),
                            "ties": int(ties),
                            "non_worse_count": int(wins + ties),
                            "non_worse_rate": float((wins + ties) / len(gaps)) if gaps else float("nan"),
                            "gap_mean": stat(gaps, "mean"),
                            "gap_std": stat(gaps, "std"),
                            "gap_median": stat(gaps, "median"),
                            "gap_q1": stat(gaps, "q1"),
                            "gap_q3": stat(gaps, "q3"),
                            "gap_iqr": stat(gaps, "iqr"),
                            "worst_case_gap": stat(gaps, "max"),
                            "best_case_gap": stat(gaps, "min"),
                        }
                    )
    return rows


def build_rank_rows(summary_rows):
    rows = []
    ranks_by_problem_method = {}
    grouped = {}
    for row in summary_rows:
        grouped.setdefault((row["problem"], int(row["tfes"])), []).append(row)
    for (problem, tfes), items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda row: v01.to_float(row.get("best_mean")))
        for index, row in enumerate(sorted_items, start=1):
            ranks_by_problem_method.setdefault((problem, row["method"]), []).append(index)
            rows.append(
                {
                    "problem": problem,
                    "tfes": int(tfes),
                    "method": row["method"],
                    "rank": int(index),
                    "best_mean": row.get("best_mean"),
                    "rank_mean": "",
                    "rank_best": "",
                    "rank_worst": "",
                }
            )
    for (problem, method), ranks in sorted(ranks_by_problem_method.items()):
        rows.append(
            {
                "problem": problem,
                "tfes": "all",
                "method": method,
                "rank": "",
                "best_mean": "",
                "rank_mean": float(np.mean(ranks)),
                "rank_best": int(min(ranks)),
                "rank_worst": int(max(ranks)),
            }
        )
    return rows


def build_action_distribution_rows(relation_rows):
    grouped = {}
    totals = {}
    for row in relation_rows:
        key = (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("tfes")),
            str(row.get("method", "")),
        )
        action = str(row.get("action_candidate", "") or "Unknown")
        grouped.setdefault((*key, action), []).append(row)
        totals[key] = totals.get(key, 0) + 1

    rows = []
    for key in sorted(grouped):
        problem, tfes, method, action = key
        items = grouped[key]
        attempts = [row for row in items if v01.to_bool(row.get("validation_attempted"))]
        accepts = [row for row in attempts if v01.to_bool(row.get("validation_accepted"))]
        total = totals.get((problem, tfes, method), 0)
        rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "action": action,
                "action_count": int(len(items)),
                "action_share": float(len(items) / total) if total else 0.0,
                "validation_attempt_count": int(len(attempts)),
                "validation_accept_count": int(len(accepts)),
                "validation_accept_rate": float(len(accepts) / len(attempts)) if attempts else 0.0,
                "action_delta_mean": stat([row.get("validation_delta") for row in items], "mean"),
            }
        )
    return rows


def is_targeted_probe(row):
    return str(row.get("action_candidate", "")) == "Fusion" and (
        v01.to_bool(row.get("arac_targeted_probe_candidate"))
        or str(row.get("action_reason", "")).startswith("targeted_probe_fusion")
    )


def row_phase(row):
    return str(row.get("arac_probe_phase") or row.get("arac_recovery_phase") or row.get("phase") or "")


def build_probe_metric_rows(relation_rows, offline_candidates=None):
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
        if method not in {CANDIDATE_METHOD, ABLATION_METHOD}:
            continue
        problem = str(row.get("problem", "")).upper()
        tfes = v01.to_int(row.get("tfes"))
        bucket = buckets.setdefault(
            (problem, tfes, method),
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "targeted_candidate_count": 0,
                "targeted_probe_count": 0,
                "targeted_probe_accept_count": 0,
                "targeted_probe_delta_values": [],
                "matched_probe_count": 0,
                "recovered_fusion_count": 0,
                "bad_probe_count": 0,
            },
        )
        key = (
            problem,
            v01.to_int(row.get("seed")),
            tfes,
            row_phase(row),
            v01.to_int(row.get("var_id")),
        )
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")):
            bucket["targeted_candidate_count"] += 1
        if is_targeted_probe(row):
            accepted = v01.to_bool(row.get("validation_accepted"))
            delta = v01.to_float(row.get("validation_delta"))
            bucket["targeted_probe_count"] += 1
            bucket["targeted_probe_accept_count"] += int(accepted)
            if np.isfinite(delta):
                bucket["targeted_probe_delta_values"].append(delta)
            if key in offline_keys:
                bucket["matched_probe_count"] += 1
            if accepted and np.isfinite(delta) and delta > 0.0:
                bucket["recovered_fusion_count"] += 1
            if problem == "R6" and np.isfinite(delta) and delta <= 0.0:
                bucket["bad_probe_count"] += 1

    rows = []
    for key in sorted(buckets):
        bucket = buckets[key]
        probe_count = int(bucket["targeted_probe_count"])
        deltas = bucket["targeted_probe_delta_values"]
        rows.append(
            {
                "problem": bucket["problem"],
                "tfes": int(bucket["tfes"]),
                "method": bucket["method"],
                "targeted_candidate_count": int(bucket["targeted_candidate_count"]),
                "targeted_probe_count": int(probe_count),
                "targeted_probe_accept_count": int(bucket["targeted_probe_accept_count"]),
                "targeted_probe_accept_rate": float(bucket["targeted_probe_accept_count"] / probe_count)
                if probe_count
                else 0.0,
                "targeted_probe_delta_mean": stat(deltas, "mean"),
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


def report_robustness_table(robustness_rows):
    lines = [
        "| problem | tfes | candidate | baseline | paired | W/L/T | non_worse | median_gap | worst_gap | IQR |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    focus = [
        row
        for row in robustness_rows
        if row.get("candidate") == CANDIDATE_METHOD and str(row.get("tfes")) == "all"
    ]
    for row in sorted(focus, key=lambda item: (item["problem"], item["baseline"])):
        lines.append(
            "| {problem} | {tfes} | {candidate} | {baseline} | {paired} | {wins}/{losses}/{ties} | {non_worse:.3f} | {median} | {worst} | {iqr} |".format(
                problem=row["problem"],
                tfes=row["tfes"],
                candidate=row["candidate"],
                baseline=row["baseline"],
                paired=int(row.get("paired_runs", 0) or 0),
                wins=int(row.get("wins", 0) or 0),
                losses=int(row.get("losses", 0) or 0),
                ties=int(row.get("ties", 0) or 0),
                non_worse=v01.to_float(row.get("non_worse_rate")),
                median=v01.format_percent(row.get("gap_median")),
                worst=v01.format_percent(row.get("worst_case_gap")),
                iqr=v01.format_percent(row.get("gap_iqr")),
            )
        )
    return "\n".join(lines)


def report_probe_table(probe_metric_rows):
    lines = [
        "| problem | tfes | method | targeted | matched | recovered | bad | delta | extra_fe |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(probe_metric_rows, key=lambda item: (item["problem"], int(item["tfes"]), item["method"])):
        lines.append(
            "| {problem} | {tfes} | {method} | {targeted} | {matched} | {recovered} | {bad} | {delta} | {extra} |".format(
                problem=row["problem"],
                tfes=int(row["tfes"]),
                method=row["method"],
                targeted=int(row.get("targeted_probe_count", 0) or 0),
                matched=int(row.get("matched_probe_count", 0) or 0),
                recovered=int(row.get("recovered_fusion_count", 0) or 0),
                bad=int(row.get("bad_probe_count", 0) or 0),
                delta=v01.format_metric(row.get("targeted_probe_delta_mean")),
                extra=v01.format_percent(row.get("extra_fe_ratio")),
            )
        )
    return "\n".join(lines)


def acceptance_notes(summary_rows, robustness_rows, probe_metric_rows):
    candidate_summary = [row for row in summary_rows if row.get("method") == CANDIDATE_METHOD]
    max_extra_fe = stat([row.get("validation_extra_fe_ratio") for row in candidate_summary], "max")
    fusion_count = sum(int(row.get("fusion_count", 0) or 0) for row in candidate_summary)
    r6_bad = sum(
        int(row.get("bad_probe_count", 0) or 0)
        for row in probe_metric_rows
        if row.get("method") == CANDIDATE_METHOD and row.get("problem") == "R6"
    )
    disable_rows = [
        row
        for row in robustness_rows
        if row.get("candidate") == CANDIDATE_METHOD
        and row.get("baseline") == DISABLE_FAST_METHOD
        and str(row.get("tfes")) == "all"
    ]
    non_worse = sum(1 for row in disable_rows if v01.to_float(row.get("non_worse_rate")) >= 0.5)
    return "\n".join(
        [
            f"- vs disable-fast majority non-worse problems: {non_worse}/{len(disable_rows)}。",
            f"- max extra FE ratio: {v01.format_percent(max_extra_fe)}。",
            f"- V0.6 Fusion count: {fusion_count}。",
            f"- R6 bad_probe_count: {r6_bad}。",
        ]
    )


def build_report_text(run_rows, summary_rows, robustness_rows, rank_rows, probe_metric_rows, args):
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    return "\n".join(
        [
            "# ARAC-lite V0.7 Fixed Candidate Generalization",
            "",
            "- 日期：2026-05-21",
            "- 执行者：Codex",
            "- 结论边界：冻结 V0.6 targeted-probe，不继续调 probe 阈值或动作规则。",
            f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
            f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
            f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
            f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
            f"- workers: {max(1, int(getattr(args, 'workers', 1) or 1))}",
            f"- resume: {bool(getattr(args, 'resume', False))}",
            f"- case cache: `{CASE_ROOT.as_posix()}`",
            "",
            "## Matrix",
            "",
            f"- Runs: {ok_count}/{len(run_rows)} ok",
            "- Methods: no-coordination, validated-selective-conflict, arac-lite-v0.1-disable-fast, arac-lite-v0.6-targeted-probe, arac-lite-v0.6-no-delta-hard-block。",
            "- Ablation: no-delta-hard-block 放宽 targeted relation delta 类 hard gate，用于判断 delta rule 是否必要。",
            "",
            "## Acceptance Snapshot",
            "",
            acceptance_notes(summary_rows, robustness_rows, probe_metric_rows),
            "",
            "## Paired Robustness",
            "",
            "下表是 paired win/loss/tie 汇总，gap 使用同 problem/tfes/seed 下 candidate 相对 baseline 的 best_error。",
            "",
            report_robustness_table(robustness_rows),
            "",
            "## Probe Metrics",
            "",
            report_probe_table(probe_metric_rows),
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
            f"- summary: `{SUMMARY_PATH.as_posix()}`",
            f"- paired robustness: `{ROBUSTNESS_PATH.as_posix()}`",
            f"- rank summary: `{RANK_SUMMARY_PATH.as_posix()}`",
            f"- action distribution: `{ACTION_DISTRIBUTION_PATH.as_posix()}`",
            f"- probe metrics: `{PROBE_METRICS_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
            "",
        ]
    )


def write_report(run_rows, summary_rows, robustness_rows, rank_rows, probe_metric_rows, args):
    REPORT_PATH.write_text(
        build_report_text(run_rows, summary_rows, robustness_rows, rank_rows, probe_metric_rows, args),
        encoding="utf-8",
    )


def run_sweep(args):
    tasks = build_case_tasks(args)
    fallback_run_rows = read_csv(RUN_DETAILS_PATH)
    fallback_relation_rows = read_csv(RELATION_AUDIT_PATH)
    cached_run_rows = read_case_run_rows(tasks)
    completed_keys = completed_case_keys([*fallback_run_rows, *cached_run_rows]) if args.resume else set()
    pending_tasks = filter_completed_tasks(tasks, completed_keys)

    print(
        f"V0.7 requested cases={len(tasks)} pending={len(pending_tasks)} "
        f"workers={max(1, int(args.workers or 1))} resume={bool(args.resume)}"
    )
    if pending_tasks:
        run_tasks(pending_tasks, workers=args.workers)

    return read_case_outputs(
        tasks,
        fallback_run_rows=fallback_run_rows if args.resume else None,
        fallback_relation_rows=fallback_relation_rows if args.resume else None,
    )


def main():
    args = parse_args()
    run_rows, relation_rows = run_sweep(args)
    summary_rows = summarize_runs(run_rows)
    robustness_rows = build_robustness_rows(run_rows)
    rank_rows = build_rank_rows(summary_rows)
    action_rows = build_action_distribution_rows(relation_rows)
    offline_candidates = v06.build_offline_candidates_for_args(args)
    probe_rows = attach_extra_fe_ratio(build_probe_metric_rows(relation_rows, offline_candidates), summary_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(ROBUSTNESS_PATH, ROBUSTNESS_FIELDNAMES, robustness_rows)
    write_csv(RANK_SUMMARY_PATH, RANK_FIELDNAMES, rank_rows)
    write_csv(ACTION_DISTRIBUTION_PATH, ACTION_DISTRIBUTION_FIELDNAMES, action_rows)
    write_csv(PROBE_METRICS_PATH, PROBE_METRIC_FIELDNAMES, probe_rows)
    write_report(run_rows, summary_rows, robustness_rows, rank_rows, probe_rows, args)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"probe metric rows -> {len(probe_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    freeze_support()
    main()
