import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_3_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_relation_action_audit.csv"
ACTION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_summary.csv"
ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_robustness.csv"
RANK_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_rank_summary.csv"
ACTION_STABILITY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_action_stability.csv"
PASS_ACTION_DYNAMICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_pass_action_dynamics.csv"
PHASE_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_phase_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_report.md"

DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_TFES = [5000, 10000, 20000]
DEFAULT_CC_PASS_GROUP_FES = 20
CANDIDATE_METHOD = "arac-lite-v0.1-disable-fast"
BASELINE_METHODS = ["no-coordination", "validated-selective-conflict", "arac-lite-v0"]
METHODS = [*BASELINE_METHODS, CANDIDATE_METHOD]
PHASES = ["early", "middle", "late"]
TIE_EPS = 1e-12

spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_2",
    SCRIPT_ROOT / "generate_arac_lite_v0_2_artifacts.py",
)
v02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v02)

v01 = v02.v01
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
    "gap_vs_arac_lite_v0",
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

ACTION_STABILITY_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "action",
    "runs",
    "total_action_count",
    "action_count_min",
    "action_count_max",
    "action_share_mean",
    "action_share_std",
    "action_share_q1",
    "action_share_q3",
    "action_share_iqr",
    "fusion_accept_rate_mean",
    "fusion_accept_rate_std",
    "action_delta_mean",
    "action_delta_std",
]

PASS_ACTION_DYNAMICS_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "pass_id",
    "phase",
    "action",
    "action_count",
    "validation_attempt_count",
    "validation_accept_count",
    "fusion_accept_rate",
    "disable_false_negative_count",
    "disable_false_negative_rate",
    "action_delta_mean",
    "best_improvement_after_action_count",
    "best_improvement_after_action_rate",
]

PHASE_SUMMARY_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "phase",
    "action",
    "action_count",
    "validation_attempt_count",
    "validation_accept_count",
    "fusion_accept_rate",
    "disable_false_negative_count",
    "disable_false_negative_rate",
    "action_delta_mean",
    "best_improvement_after_action_count",
    "best_improvement_after_action_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.3 robustness and pass dynamics artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


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


def fixed_method_configs(cc_pass_group_fes):
    base_arac = v01.load_base_arac_config(cc_pass_group_fes)
    return [
        ("no-coordination", v01.load_named_config(CONFIG_ROOT / "no-coordination.json", cc_pass_group_fes)),
        (
            "validated-selective-conflict",
            v01.load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", cc_pass_group_fes),
        ),
        ("arac-lite-v0", v01.apply_variant(base_arac, v01.ThresholdVariant("arac-lite-v0", {}))),
        ("arac-lite-v0.1-disable-fast", v01.apply_variant(base_arac, v02.disable_fast_variant())),
    ]


def summarize_runs(run_rows):
    grouped = {}
    for row in run_rows:
        grouped.setdefault((row["problem"], int(row["tfes"]), row["method"]), []).append(row)

    best_baselines = {}
    for (problem, tfes, method), rows in grouped.items():
        if method in BASELINE_METHODS:
            best_baselines[(problem, tfes, method)] = stat([row.get("best_error") for row in rows], "mean")

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
                "gap_vs_no_coordination": gap(best_mean, best_baselines.get((problem, tfes, "no-coordination"))),
                "gap_vs_validated": gap(best_mean, best_baselines.get((problem, tfes, "validated-selective-conflict"))),
                "gap_vs_arac_lite_v0": gap(best_mean, best_baselines.get((problem, tfes, "arac-lite-v0"))),
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


def build_robustness_rows(run_rows):
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
            for baseline in BASELINE_METHODS:
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
                        candidate = by_case.get((problem, int(tfes), int(seed), CANDIDATE_METHOD))
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
                        "candidate": CANDIDATE_METHOD,
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


def phase_for_pass(pass_id, max_pass_id):
    pass_id = max(0, int(pass_id))
    max_pass_id = max(0, int(max_pass_id))
    if max_pass_id <= 0:
        return "early"
    ratio = pass_id / max_pass_id
    if ratio <= 1.0 / 3.0:
        return "early"
    if ratio <= 2.0 / 3.0:
        return "middle"
    return "late"


def relation_false_negative_lookup(relation_rows):
    future_lookup = v01.build_future_accepted_fusion_lookup(relation_rows)
    false_negative = set()
    for row in relation_rows:
        if row.get("action_candidate") != "Disable":
            continue
        key = (
            str(row.get("problem", "")).upper(),
            int(v01.to_float(row.get("seed"))),
            int(v01.to_float(row.get("tfes"))),
            str(row.get("method", "")),
            int(v01.to_float(row.get("var_id"))),
            int(v01.to_float(row.get("pass_id"))),
        )
        if key in future_lookup:
            false_negative.add(key)
    return false_negative


def aggregate_action_rows(rows, key_fields, false_negative_lookup=None):
    grouped = {}
    for row in rows:
        action = str(row.get("action_candidate", "") or "Unknown")
        key = tuple(row.get(field) for field in key_fields) + (action,)
        grouped.setdefault(key, []).append(row)
    result = []
    for key in sorted(grouped):
        action_rows = grouped[key]
        action = key[-1]
        attempts = [row for row in action_rows if v01.to_bool(row.get("validation_attempted"))]
        accepts = [row for row in attempts if v01.to_bool(row.get("validation_accepted"))]
        deltas = [v01.to_float(row.get("validation_delta")) for row in action_rows]
        best_improvements = [row for row in action_rows if v01.to_bool(row.get("best_improvement_after_action"))]
        false_negative_count = 0
        if false_negative_lookup and action == "Disable":
            for row in action_rows:
                relation_key = (
                    str(row.get("problem", "")).upper(),
                    int(v01.to_float(row.get("seed"))),
                    int(v01.to_float(row.get("tfes"))),
                    str(row.get("method", "")),
                    int(v01.to_float(row.get("var_id"))),
                    int(v01.to_float(row.get("pass_id"))),
                )
                if relation_key in false_negative_lookup:
                    false_negative_count += 1
        base = {field: key[index] for index, field in enumerate(key_fields)}
        base.update(
            {
                "action": action,
                "action_count": int(len(action_rows)),
                "validation_attempt_count": int(len(attempts)),
                "validation_accept_count": int(len(accepts)),
                "fusion_accept_rate": float(len(accepts) / len(attempts)) if action == "Fusion" and attempts else 0.0,
                "disable_false_negative_count": int(false_negative_count),
                "disable_false_negative_rate": float(false_negative_count / len(action_rows)) if action_rows else 0.0,
                "action_delta_mean": stat(deltas, "mean"),
                "best_improvement_after_action_count": int(len(best_improvements)),
                "best_improvement_after_action_rate": float(len(best_improvements) / len(action_rows)) if action_rows else 0.0,
            }
        )
        result.append(base)
    return result


def build_pass_action_dynamics(relation_rows):
    max_pass = {}
    for row in relation_rows:
        key = (
            str(row.get("problem", "")).upper(),
            int(v01.to_float(row.get("seed"))),
            int(v01.to_float(row.get("tfes"))),
            str(row.get("method", "")),
        )
        max_pass[key] = max(max_pass.get(key, 0), int(v01.to_float(row.get("pass_id"))))

    rows = []
    for row in relation_rows:
        normalized = dict(row)
        run_key = (
            str(row.get("problem", "")).upper(),
            int(v01.to_float(row.get("seed"))),
            int(v01.to_float(row.get("tfes"))),
            str(row.get("method", "")),
        )
        normalized["phase"] = phase_for_pass(row.get("pass_id"), max_pass.get(run_key, 0))
        rows.append(normalized)

    false_negative_lookup = relation_false_negative_lookup(rows)
    return aggregate_action_rows(
        rows,
        ["problem", "seed", "tfes", "method", "pass_id", "phase"],
        false_negative_lookup=false_negative_lookup,
    )


def build_phase_summary(relation_rows):
    max_pass = {}
    for row in relation_rows:
        key = (
            str(row.get("problem", "")).upper(),
            int(v01.to_float(row.get("seed"))),
            int(v01.to_float(row.get("tfes"))),
            str(row.get("method", "")),
        )
        max_pass[key] = max(max_pass.get(key, 0), int(v01.to_float(row.get("pass_id"))))
    enriched = []
    for row in relation_rows:
        item = dict(row)
        key = (
            str(row.get("problem", "")).upper(),
            int(v01.to_float(row.get("seed"))),
            int(v01.to_float(row.get("tfes"))),
            str(row.get("method", "")),
        )
        item["phase"] = phase_for_pass(row.get("pass_id"), max_pass.get(key, 0))
        enriched.append(item)
    return aggregate_action_rows(
        enriched,
        ["problem", "tfes", "method", "phase"],
        false_negative_lookup=relation_false_negative_lookup(enriched),
    )


def build_action_stability_rows(relation_rows):
    per_run = aggregate_action_rows(
        relation_rows,
        ["problem", "tfes", "method", "seed"],
        false_negative_lookup=relation_false_negative_lookup(relation_rows),
    )
    totals = {}
    for row in per_run:
        totals[(row["problem"], int(row["tfes"]), row["method"], int(row["seed"]))] = totals.get(
            (row["problem"], int(row["tfes"]), row["method"], int(row["seed"])),
            0,
        ) + int(row.get("action_count", 0) or 0)

    grouped = {}
    for row in per_run:
        key = (row["problem"], int(row["tfes"]), row["method"], row["action"])
        run_total = totals[(row["problem"], int(row["tfes"]), row["method"], int(row["seed"]))]
        item = dict(row)
        item["action_share"] = float(int(row.get("action_count", 0) or 0) / run_total) if run_total else 0.0
        grouped.setdefault(key, []).append(item)

    rows = []
    for key in sorted(grouped):
        problem, tfes, method, action = key
        items = grouped[key]
        counts = [int(row.get("action_count", 0) or 0) for row in items]
        shares = [row.get("action_share") for row in items]
        rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "action": action,
                "runs": int(len(items)),
                "total_action_count": int(sum(counts)),
                "action_count_min": int(min(counts)) if counts else 0,
                "action_count_max": int(max(counts)) if counts else 0,
                "action_share_mean": stat(shares, "mean"),
                "action_share_std": stat(shares, "std"),
                "action_share_q1": stat(shares, "q1"),
                "action_share_q3": stat(shares, "q3"),
                "action_share_iqr": stat(shares, "iqr"),
                "fusion_accept_rate_mean": stat([row.get("fusion_accept_rate") for row in items], "mean"),
                "fusion_accept_rate_std": stat([row.get("fusion_accept_rate") for row in items], "std"),
                "action_delta_mean": stat([row.get("action_delta_mean") for row in items], "mean"),
                "action_delta_std": stat([row.get("action_delta_mean") for row in items], "std"),
            }
        )
    return rows


def run_sweep(args):
    problems = [v01.hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    run_rows = []
    relation_rows = []
    for method_name, config in fixed_method_configs(args.cc_pass_group_fes):
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    run_row, case_relation_rows = v01.run_one_case(method_name, config, problem_code, seed, tfes)
                    run_rows.append(run_row)
                    relation_rows.extend(case_relation_rows)
                    print(
                        f"{problem_code} {method_name} seed={seed} tfes={tfes}: "
                        f"{run_row['status']} fusion={run_row['fusion_count']} "
                        f"freeze={run_row['freeze_count']} disable={run_row['disable_count']}"
                    )
    return run_rows, relation_rows


def count_problem_non_worse(robustness_rows, baseline, min_rate=0.5):
    rows = [
        row
        for row in robustness_rows
        if str(row.get("baseline")) == baseline and str(row.get("tfes")) == "all"
    ]
    return sum(1 for row in rows if v01.to_float(row.get("non_worse_rate")) >= min_rate), len(rows)


def phase_signal_lines(phase_rows):
    lines = []
    candidate_rows = [row for row in phase_rows if row["method"] == CANDIDATE_METHOD and row["action"] == "Fusion"]
    by_problem_phase = {}
    for row in candidate_rows:
        key = (row["problem"], row["phase"])
        bucket = by_problem_phase.setdefault(
            key,
            {
                "action_count": 0,
                "validation_attempt_count": 0,
                "validation_accept_count": 0,
                "delta_weighted_sum": 0.0,
            },
        )
        count = int(row.get("action_count", 0) or 0)
        bucket["action_count"] += count
        bucket["validation_attempt_count"] += int(row.get("validation_attempt_count", 0) or 0)
        bucket["validation_accept_count"] += int(row.get("validation_accept_count", 0) or 0)
        delta = v01.to_float(row.get("action_delta_mean"))
        if np.isfinite(delta):
            bucket["delta_weighted_sum"] += delta * count

    def accept_rate(problem, phase):
        bucket = by_problem_phase.get((problem, phase), {})
        attempts = int(bucket.get("validation_attempt_count", 0) or 0)
        accepts = int(bucket.get("validation_accept_count", 0) or 0)
        return float(accepts / attempts) if attempts else float("nan")

    def action_count(problem, phase):
        return int(by_problem_phase.get((problem, phase), {}).get("action_count", 0) or 0)

    def delta_mean(problem, phase):
        bucket = by_problem_phase.get((problem, phase), {})
        count = int(bucket.get("action_count", 0) or 0)
        return float(bucket.get("delta_weighted_sum", 0.0) / count) if count else float("nan")

    for problem in ["R6", "E6", "S6", "A6"]:
        early = accept_rate(problem, "early")
        middle = accept_rate(problem, "middle")
        late = accept_rate(problem, "late")
        delta = late - early if np.isfinite(early) and np.isfinite(late) else float("nan")
        lines.append(
            f"- {problem}: early Fusion accept={v01.format_metric(early)} (n={action_count(problem, 'early')}), "
            f"middle={v01.format_metric(middle)} (n={action_count(problem, 'middle')}), "
            f"late={v01.format_metric(late)} (n={action_count(problem, 'late')}), "
            f"late-early={v01.format_metric(delta)}, "
            f"middle delta={v01.format_metric(delta_mean(problem, 'middle'))}."
        )
    return lines


def report_robustness_table(robustness_rows):
    lines = [
        "| problem | baseline | paired | W/L/T | non_worse | median_gap | worst_gap | IQR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [
        item
        for item in robustness_rows
        if str(item.get("tfes")) == "all"
        and str(item.get("baseline")) in {"validated-selective-conflict", "arac-lite-v0", "no-coordination"}
    ]:
        lines.append(
            "| {problem} | {baseline} | {paired} | {wins}/{losses}/{ties} | {rate:.3f} | {median} | {worst} | {iqr} |".format(
                problem=row["problem"],
                baseline=row["baseline"],
                paired=int(row.get("paired_runs", 0) or 0),
                wins=int(row.get("wins", 0) or 0),
                losses=int(row.get("losses", 0) or 0),
                ties=int(row.get("ties", 0) or 0),
                rate=v01.to_float(row.get("non_worse_rate")),
                median=v01.format_percent(row.get("gap_median")),
                worst=v01.format_percent(row.get("worst_case_gap")),
                iqr=v01.format_percent(row.get("gap_iqr")),
            )
        )
    return "\n".join(lines)


def report_phase_table(phase_rows):
    lines = [
        "| problem | phase | action | count | fusion_accept | disable_fn | delta_mean | improvement_rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [
        row
        for row in phase_rows
        if row["method"] == CANDIDATE_METHOD and int(v01.to_float(row["tfes"])) == 10000
    ]
    phase_order = {phase: index for index, phase in enumerate(PHASES)}
    for row in sorted(rows, key=lambda item: (item["problem"], phase_order.get(item["phase"], 99), item["action"])):
        lines.append(
            "| {problem} | {phase} | {action} | {count} | {accept:.3f} | {false_neg:.3f} | {delta} | {improve:.3f} |".format(
                problem=row["problem"],
                phase=row["phase"],
                action=row["action"],
                count=int(row.get("action_count", 0) or 0),
                accept=v01.to_float(row.get("fusion_accept_rate")),
                false_neg=v01.to_float(row.get("disable_false_negative_rate")),
                delta=v01.format_metric(row.get("action_delta_mean")),
                improve=v01.to_float(row.get("best_improvement_after_action_rate")),
            )
        )
    return "\n".join(lines)


def write_report(run_rows, summary_rows, robustness_rows, action_rows, phase_rows, args):
    v0_non_worse, v0_total = count_problem_non_worse(robustness_rows, "arac-lite-v0")
    validated_non_worse, validated_total = count_problem_non_worse(robustness_rows, "validated-selective-conflict")
    no_non_worse, no_total = count_problem_non_worse(robustness_rows, "no-coordination")
    candidate_summary = [row for row in summary_rows if row["method"] == CANDIDATE_METHOD]
    total_fusion = sum(int(row.get("fusion_count", 0) or 0) for row in candidate_summary)
    total_disable = sum(int(row.get("disable_count", 0) or 0) for row in candidate_summary)
    disable_fn_rows = [
        row
        for row in action_rows
        if row["method"] == CANDIDATE_METHOD and row["action_candidate"] == "Disable"
    ]
    disable_fn = stat([row.get("disable_false_negative_rate") for row in disable_fn_rows], "mean")
    report = [
        "# ARAC-lite V0.3 稳健性验证 + 阶段动作动态审计",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Frozen candidate: `arac-lite-v0.1-disable-fast`。未调阈值，未加入 UCB。",
        "",
        "## Quick Read",
        "",
        f"- Total runs: {len(run_rows)}",
        f"- OK runs: {sum(1 for row in run_rows if str(row.get('status')) == 'ok')}/{len(run_rows)}",
        f"- Candidate vs V0: {v0_non_worse}/{v0_total} problems have >=50% paired non-worse rate.",
        f"- Candidate vs validated: {validated_non_worse}/{validated_total} problems have >=50% paired non-worse rate.",
        f"- Candidate vs no-coordination: {no_non_worse}/{no_total} problems have >=50% paired non-worse rate.",
        f"- Candidate Fusion count: {total_fusion}; Disable count: {total_disable}; mean Disable false-negative rate: {v01.format_metric(disable_fn)}.",
        "",
        "## Robustness",
        "",
        report_robustness_table(robustness_rows),
        "",
        "## Phase Dynamics Signals",
        "",
        *phase_signal_lines(phase_rows),
        "",
        "## Candidate Phase Summary At TFEs=10000",
        "",
        report_phase_table(phase_rows),
        "",
        "## UCB Decision",
        "",
        "- V0.3 的用途是判断是否有足够阶段性证据进入非平稳 UCB；最终判断基于 late-vs-early Fusion accept、action_delta 和阶段样本量，而不是单次均值。",
        "- 当前证据不支持立刻进入非平稳 UCB：E6/S6 的 middle/late Fusion accept 有上升信号，但 middle/late Fusion 样本远少于 early，且 S6 late delta 仍为负。",
        "- 更合适的下一步是保留固定规则主线，同时把 S6/E6 的中后期 Fusion recovery 作为单独审计项，而不是直接引入 bandit。",
        "",
        "## Artifacts",
        "",
        f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
        f"- summary: `{SUMMARY_PATH.as_posix()}`",
        f"- robustness: `{ROBUSTNESS_PATH.as_posix()}`",
        f"- rank summary: `{RANK_SUMMARY_PATH.as_posix()}`",
        f"- action stability: `{ACTION_STABILITY_PATH.as_posix()}`",
        f"- pass dynamics: `{PASS_ACTION_DYNAMICS_PATH.as_posix()}`",
        f"- phase summary: `{PHASE_SUMMARY_PATH.as_posix()}`",
        f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
        f"- action audit: `{ACTION_AUDIT_PATH.as_posix()}`",
        f"- report: `{REPORT_PATH.as_posix()}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_rows, relation_rows = run_sweep(args)
    summary_rows = summarize_runs(run_rows)
    robustness_rows = build_robustness_rows(run_rows)
    rank_rows = build_rank_rows(summary_rows)
    action_rows = v01.build_action_attribution_rows(relation_rows)
    action_stability_rows = build_action_stability_rows(relation_rows)
    pass_action_rows = build_pass_action_dynamics(relation_rows)
    phase_rows = build_phase_summary(relation_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)
    write_csv(ACTION_AUDIT_PATH, v01.ATTRIBUTION_FIELDNAMES, action_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(ROBUSTNESS_PATH, ROBUSTNESS_FIELDNAMES, robustness_rows)
    write_csv(RANK_SUMMARY_PATH, RANK_FIELDNAMES, rank_rows)
    write_csv(ACTION_STABILITY_PATH, ACTION_STABILITY_FIELDNAMES, action_stability_rows)
    write_csv(PASS_ACTION_DYNAMICS_PATH, PASS_ACTION_DYNAMICS_FIELDNAMES, pass_action_rows)
    write_csv(PHASE_SUMMARY_PATH, PHASE_SUMMARY_FIELDNAMES, phase_rows)
    write_report(run_rows, summary_rows, robustness_rows, action_rows, phase_rows, args)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"robustness rows -> {len(robustness_rows)}")
    print(f"phase rows -> {len(phase_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
