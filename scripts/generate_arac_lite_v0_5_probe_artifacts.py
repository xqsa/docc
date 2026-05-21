import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_relation_action_audit.csv"
ACTION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_summary.csv"
ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_robustness.csv"
PHASE_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_phase_summary.csv"
PROBE_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_metrics.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_report.md"

DEFAULT_PROBLEMS = ["E6", "S6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20
CANDIDATE_METHOD = "arac-lite-v0.5-low-frequency-probe"
DISABLE_FAST_METHOD = "arac-lite-v0.1-disable-fast"
METHODS = ["no-coordination", "validated-selective-conflict", DISABLE_FAST_METHOD, CANDIDATE_METHOD]
TIE_EPS = 1e-12


spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_4",
    SCRIPT_ROOT / "generate_arac_lite_v0_4_recovery_artifacts.py",
)
v04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v04)

v03 = v04.v03
v02 = v04.v02
v01 = v04.v01
v01.RUNS_ROOT = RUNS_ROOT


PROBE_METRIC_FIELDNAMES = [
    "problem",
    "method",
    "probe_count",
    "probe_accept_count",
    "probe_accept_rate",
    "probe_delta_mean",
    "recovery_candidate_count",
    "recovered_fusion_count",
    "bad_recovery_count",
    "extra_fe_ratio",
    "S6_middle_recovery_count",
    "R6_bad_probe_count",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.5 low-frequency probe artifacts.")
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


def low_frequency_probe_variant():
    overrides = dict(v02.disable_fast_variant().overrides)
    overrides.update(
        {
            "arac_lite_recovery_enabled": True,
            "arac_lite_recovery_min_attempts": 20,
            "arac_lite_recovery_accept_rate_threshold": 0.25,
            "arac_lite_recovery_delta_threshold": 0.0,
            "arac_lite_recovery_positive_delta_rate_threshold": 0.25,
            "arac_lite_recovery_recent_window": 20,
            "arac_lite_recovery_min_phase": "middle",
            "arac_lite_probe_enabled": True,
            "arac_lite_probe_every_n_pass": 2,
            "arac_lite_probe_max_per_pass": 5,
            "arac_lite_probe_min_phase": "middle",
            "arac_lite_probe_recovery_min_attempts": 5,
            "arac_lite_probe_recovery_accept_rate_threshold": 0.3,
            "arac_lite_probe_recovery_delta_threshold": 0.0,
            "arac_lite_probe_recovery_recent_window": 20,
        }
    )
    return v01.ThresholdVariant(CANDIDATE_METHOD, overrides)


def fixed_method_configs(cc_pass_group_fes):
    base_arac = v01.load_base_arac_config(cc_pass_group_fes)
    return [
        ("no-coordination", v01.load_named_config(CONFIG_ROOT / "no-coordination.json", cc_pass_group_fes)),
        (
            "validated-selective-conflict",
            v01.load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", cc_pass_group_fes),
        ),
        (DISABLE_FAST_METHOD, v01.apply_variant(base_arac, v02.disable_fast_variant())),
        (CANDIDATE_METHOD, v01.apply_variant(base_arac, low_frequency_probe_variant())),
    ]


def stat(values, name):
    return v04.stat(values, name)


def gap(value, baseline):
    return v04.gap(value, baseline)


def summarize_runs(run_rows):
    grouped = {}
    for row in run_rows:
        grouped.setdefault((row["problem"], int(row["tfes"]), row["method"]), []).append(row)

    baselines = {}
    for (problem, tfes, method), rows in grouped.items():
        if method in {"no-coordination", "validated-selective-conflict", DISABLE_FAST_METHOD}:
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


def build_probe_metric_rows(relation_rows):
    by_problem = {}
    for row in relation_rows:
        if str(row.get("method", "")) != CANDIDATE_METHOD:
            continue
        problem = str(row.get("problem", "")).upper()
        bucket = by_problem.setdefault(
            problem,
            {
                "problem": problem,
                "method": CANDIDATE_METHOD,
                "probe_count": 0,
                "probe_accept_count": 0,
                "probe_delta_values": [],
                "recovery_candidate_count": 0,
                "recovered_fusion_count": 0,
                "bad_recovery_count": 0,
                "validation_attempt_count": 0,
                "S6_middle_recovery_count": 0,
                "R6_bad_probe_count": 0,
            },
        )
        is_probe = (
            str(row.get("action_candidate", "")) == "Fusion"
            and (
                v01.to_bool(row.get("arac_probe_candidate"))
                or str(row.get("action_reason", "")).startswith("probe_fusion")
            )
        )
        is_recovery = v01.to_bool(row.get("arac_recovery_candidate")) or str(row.get("action_reason", "")).startswith("recovery_")
        attempted = v01.to_bool(row.get("validation_attempted"))
        accepted = v01.to_bool(row.get("validation_accepted"))
        delta = v01.to_float(row.get("validation_delta"))
        phase = str(row.get("arac_recovery_phase", "") or row.get("arac_probe_phase", ""))
        if attempted:
            bucket["validation_attempt_count"] += 1
        if is_probe:
            bucket["probe_count"] += 1
            if accepted:
                bucket["probe_accept_count"] += 1
            if np.isfinite(delta):
                bucket["probe_delta_values"].append(float(delta))
            if problem == "R6" and np.isfinite(delta) and delta <= 0.0:
                bucket["R6_bad_probe_count"] += 1
        if is_recovery:
            bucket["recovery_candidate_count"] += 1
            if str(row.get("action_candidate", "")) == "Fusion":
                bucket["recovered_fusion_count"] += 1
                if problem == "S6" and phase == "middle":
                    bucket["S6_middle_recovery_count"] += 1
            if np.isfinite(delta) and delta <= 0.0:
                bucket["bad_recovery_count"] += 1

    metric_rows = []
    for problem, bucket in sorted(by_problem.items()):
        probe_count = int(bucket["probe_count"])
        probe_accept_count = int(bucket["probe_accept_count"])
        delta_values = np.asarray(bucket["probe_delta_values"], dtype=float)
        metric_rows.append(
            {
                "problem": problem,
                "method": CANDIDATE_METHOD,
                "probe_count": int(probe_count),
                "probe_accept_count": int(probe_accept_count),
                "probe_accept_rate": float(probe_accept_count / probe_count) if probe_count else 0.0,
                "probe_delta_mean": float(np.mean(delta_values)) if delta_values.size else 0.0,
                "recovery_candidate_count": int(bucket["recovery_candidate_count"]),
                "recovered_fusion_count": int(bucket["recovered_fusion_count"]),
                "bad_recovery_count": int(bucket["bad_recovery_count"]),
                "extra_fe_ratio": 0.0,
                "S6_middle_recovery_count": int(bucket["S6_middle_recovery_count"]),
                "R6_bad_probe_count": int(bucket["R6_bad_probe_count"]),
            }
        )
    return metric_rows


def attach_extra_fe_ratio(probe_rows, summary_rows):
    ratio_by_problem = {
        row["problem"]: v01.to_float(row.get("validation_extra_fe_ratio"))
        for row in summary_rows
        if row.get("method") == CANDIDATE_METHOD
    }
    for row in probe_rows:
        row["extra_fe_ratio"] = ratio_by_problem.get(row["problem"], 0.0)
    return probe_rows


def build_robustness_rows(run_rows):
    old_candidate = v04.CANDIDATE_METHOD
    v04.CANDIDATE_METHOD = CANDIDATE_METHOD
    try:
        return v04.build_robustness_rows(run_rows)
    finally:
        v04.CANDIDATE_METHOD = old_candidate


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


def report_probe_table(probe_rows):
    lines = [
        "| problem | probe_count | probe_accept | probe_accept_rate | probe_delta | recovered_fusion | bad_recovery | extra_fe | S6_middle_recovery | R6_bad_probe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in probe_rows:
        lines.append(
            "| {problem} | {probe_count} | {probe_accept_count} | {accept:.3f} | {delta} | {recovered} | {bad_recovery} | {extra_fe:.3%} | {s6_middle} | {r6_bad} |".format(
                problem=row["problem"],
                probe_count=int(row.get("probe_count", 0) or 0),
                probe_accept_count=int(row.get("probe_accept_count", 0) or 0),
                accept=v01.to_float(row.get("probe_accept_rate")),
                delta=v01.format_metric(row.get("probe_delta_mean")),
                recovered=int(row.get("recovered_fusion_count", 0) or 0),
                bad_recovery=int(row.get("bad_recovery_count", 0) or 0),
                extra_fe=v01.to_float(row.get("extra_fe_ratio")),
                s6_middle=int(row.get("S6_middle_recovery_count", 0) or 0),
                r6_bad=int(row.get("R6_bad_probe_count", 0) or 0),
            )
        )
    return "\n".join(lines)


def report_summary_table(summary_rows):
    old_candidate = v04.CANDIDATE_METHOD
    v04.CANDIDATE_METHOD = CANDIDATE_METHOD
    try:
        return v04.report_summary_table(summary_rows)
    finally:
        v04.CANDIDATE_METHOD = old_candidate


def build_acceptance_notes(summary_rows, probe_rows, max_extra_fe_ratio=0.01):
    summary_by_problem = {
        str(row.get("problem", "")).upper(): row
        for row in summary_rows
        if row.get("method") == CANDIDATE_METHOD
    }
    probe_by_problem = {str(row.get("problem", "")).upper(): row for row in probe_rows}
    s6_row = summary_by_problem.get("S6", {})
    r6_row = summary_by_problem.get("R6", {})
    max_extra_fe = max([v01.to_float(row.get("extra_fe_ratio")) for row in probe_rows] or [0.0])
    cost_passed = bool(max_extra_fe <= float(max_extra_fe_ratio))
    s6_passed = bool(s6_row and v01.to_float(s6_row.get("gap_vs_disable_fast")) <= 0.0)
    r6_passed = bool(r6_row and v01.to_float(r6_row.get("gap_vs_disable_fast")) <= 0.0)
    bad_recovery_count = int(sum(int(row.get("bad_recovery_count", 0) or 0) for row in probe_rows))
    r6_bad_probe_count = int(probe_by_problem.get("R6", {}).get("R6_bad_probe_count", 0) or 0)
    benefit_passed = bool(cost_passed and s6_passed and r6_passed and bad_recovery_count == 0)

    notes = [
        f"- 成本验收：{'通过' if cost_passed else '不通过'}；probe 额外 FE 成本{'达标' if cost_passed else '超标'}，max extra FE = {v01.format_percent(max_extra_fe)}。",
        f"- 收益验收：{'通过' if benefit_passed else '不通过'}。",
    ]
    if not s6_passed:
        notes.append(
            f"- S6 未改善：gap vs disable-fast = {v01.format_percent(s6_row.get('gap_vs_disable_fast')) if s6_row else 'n/a'}。"
        )
    if not r6_passed:
        notes.append(
            f"- R6 被拉坏：gap vs disable-fast = {v01.format_percent(r6_row.get('gap_vs_disable_fast')) if r6_row else 'n/a'}，R6_bad_probe_count = {r6_bad_probe_count}。"
        )
    if bad_recovery_count > 0:
        notes.append(f"- bad_recovery_count = {bad_recovery_count}，说明恢复门控仍会放出负向恢复。")
    return "\n".join(notes)


def write_report(run_rows, summary_rows, probe_rows, args):
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    s6_candidate = next(
        (
            row
            for row in summary_rows
            if row["problem"] == "S6" and row["method"] == CANDIDATE_METHOD and int(row["tfes"]) == 10000
        ),
        None,
    )
    r6_candidate = next(
        (
            row
            for row in summary_rows
            if row["problem"] == "R6" and row["method"] == CANDIDATE_METHOD and int(row["tfes"]) == 10000
        ),
        None,
    )
    report = [
        "# ARAC-lite V0.5 Low-Frequency Probe",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Scope: default disable-fast; middle/late low-frequency probe only; no UCB.",
        "",
        "## Quick Read",
        "",
        f"- Matrix runs: {ok_count}/{len(run_rows)} ok",
        f"- S6 V0.5 gap vs no-coordination: {v01.format_percent(s6_candidate.get('gap_vs_no_coordination')) if s6_candidate else 'n/a'}",
        f"- S6 V0.5 gap vs disable-fast: {v01.format_percent(s6_candidate.get('gap_vs_disable_fast')) if s6_candidate else 'n/a'}",
        f"- R6 V0.5 gap vs disable-fast: {v01.format_percent(r6_candidate.get('gap_vs_disable_fast')) if r6_candidate else 'n/a'}",
        "",
        "## Acceptance",
        "",
        build_acceptance_notes(summary_rows, probe_rows),
        "",
        "## Matrix Summary",
        "",
        report_summary_table(summary_rows),
        "",
        "## Probe Metrics",
        "",
        report_probe_table(probe_rows),
        "",
        "## Interpretation",
        "",
        "- V0.5 probes are low-frequency review candidates for relations already blocked by disable-fast.",
        "- Probe candidates still pass through full-fitness validation; rejected candidates are rolled back by the existing validated coordination path.",
        "- Passing requires S6 to improve without R6 degradation and without excessive validation extra FE.",
        "",
        "## Artifacts",
        "",
        f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
        f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
        f"- action audit: `{ACTION_AUDIT_PATH.as_posix()}`",
        f"- summary: `{SUMMARY_PATH.as_posix()}`",
        f"- robustness: `{ROBUSTNESS_PATH.as_posix()}`",
        f"- phase summary: `{PHASE_SUMMARY_PATH.as_posix()}`",
        f"- probe metrics: `{PROBE_METRICS_PATH.as_posix()}`",
        f"- report: `{REPORT_PATH.as_posix()}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_rows, relation_rows = run_sweep(args)
    summary_rows = summarize_runs(run_rows)
    robustness_rows = build_robustness_rows(run_rows)
    action_rows = v01.build_action_attribution_rows(relation_rows)
    phase_rows = v03.build_phase_summary(relation_rows)
    probe_rows = attach_extra_fe_ratio(build_probe_metric_rows(relation_rows), summary_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)
    write_csv(ACTION_AUDIT_PATH, v01.ATTRIBUTION_FIELDNAMES, action_rows)
    write_csv(SUMMARY_PATH, v04.SUMMARY_FIELDNAMES, summary_rows)
    write_csv(ROBUSTNESS_PATH, v04.ROBUSTNESS_FIELDNAMES, robustness_rows)
    write_csv(PHASE_SUMMARY_PATH, v03.PHASE_SUMMARY_FIELDNAMES, phase_rows)
    write_csv(PROBE_METRICS_PATH, PROBE_METRIC_FIELDNAMES, probe_rows)
    write_report(run_rows, summary_rows, probe_rows, args)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"probe metric rows -> {len(probe_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
