import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_runs"

RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_relation_action_audit.csv"
ACTION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_summary.csv"
ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_robustness.csv"
PHASE_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_phase_summary.csv"
TARGETED_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_metrics.csv"
MATCH_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_offline_match.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_6_targeted_probe_report.md"

DEFAULT_PROBLEMS = ["E6", "S6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20
CANDIDATE_METHOD = "arac-lite-v0.6-targeted-probe"
V05_METHOD = "arac-lite-v0.5-low-frequency-probe"
DISABLE_FAST_METHOD = "arac-lite-v0.1-disable-fast"


spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_5",
    SCRIPT_ROOT / "generate_arac_lite_v0_5_probe_artifacts.py",
)
v05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v05)

spec = importlib.util.spec_from_file_location(
    "v0_5_failure_audit",
    SCRIPT_ROOT / "generate_arac_lite_v0_5_failure_audit.py",
)
failure_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failure_audit)

v04 = v05.v04
v03 = v05.v03
v02 = v05.v02
v01 = v05.v01
v01.RUNS_ROOT = RUNS_ROOT


TARGETED_METRIC_FIELDNAMES = [
    "problem",
    "method",
    "targeted_candidate_count",
    "targeted_probe_count",
    "targeted_probe_accept_count",
    "targeted_probe_accept_rate",
    "targeted_probe_delta_mean",
    "matched_offline_candidate_count",
    "matched_probe_count",
    "matched_positive_delta_count",
    "extra_fe_ratio",
    "S6_middle_matched_probe_count",
    "R6_bad_probe_count",
]

MATCH_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "phase",
    "offline_candidate_count",
    "targeted_candidate_count",
    "targeted_probe_count",
    "matched_offline_candidate_count",
    "matched_probe_count",
    "matched_positive_delta_count",
    "targeted_probe_delta_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.6 targeted probe artifacts.")
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


def targeted_probe_variant():
    overrides = dict(v05.low_frequency_probe_variant().overrides)
    overrides.update(
        {
            "arac_lite_probe_max_per_pass": 1,
            "arac_lite_targeted_probe_enabled": True,
            "arac_lite_targeted_probe_phase": "middle",
            "arac_lite_targeted_probe_min_support": 2,
            "arac_lite_targeted_probe_min_relation_attempts": 2,
            "arac_lite_targeted_probe_min_accept_rate": 1.0,
            "arac_lite_targeted_probe_min_positive_delta_rate": 1.0,
            "arac_lite_targeted_probe_min_relation_delta": 0.0,
            "arac_lite_targeted_probe_min_proposal_std_ratio": 0.00125,
            "arac_lite_targeted_probe_max_proposal_std_ratio": 0.0045,
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
        (V05_METHOD, v01.apply_variant(base_arac, v05.low_frequency_probe_variant())),
        (CANDIDATE_METHOD, v01.apply_variant(base_arac, targeted_probe_variant())),
    ]


def stat(values, name):
    return v04.stat(values, name)


def gap(value, baseline):
    return v04.gap(value, baseline)


def summarize_runs(run_rows):
    return v05.summarize_runs(run_rows)


def build_robustness_rows(run_rows):
    old_candidate = v04.CANDIDATE_METHOD
    v04.CANDIDATE_METHOD = CANDIDATE_METHOD
    try:
        return v04.build_robustness_rows(run_rows)
    finally:
        v04.CANDIDATE_METHOD = old_candidate


def is_targeted_probe(row):
    return str(row.get("action_candidate", "")) == "Fusion" and (
        v01.to_bool(row.get("arac_targeted_probe_candidate"))
        or str(row.get("action_reason", "")).startswith("targeted_probe_fusion")
    )


def row_phase(row):
    return str(row.get("arac_probe_phase") or row.get("arac_recovery_phase") or row.get("phase") or "")


def filter_offline_candidates_for_args(offline_candidates, args):
    problems = {str(problem).upper() for problem in args.problems}
    seeds = {int(seed) for seed in args.seeds}
    tfes_values = {int(tfes) for tfes in args.tfes}
    return [
        row
        for row in offline_candidates
        if str(row.get("problem", "")).upper() in problems
        and v01.to_int(row.get("seed")) in seeds
        and v01.to_int(row.get("tfes")) in tfes_values
    ]


def build_offline_candidates_for_args(args):
    problems = {str(problem).upper() for problem in args.problems}
    tfes_values = {int(tfes) for tfes in args.tfes}
    v03_rows = failure_audit.read_csv(failure_audit.V03_RELATION_AUDIT_PATH)
    offline_candidates = failure_audit.build_offline_candidate_rows(v03_rows, problems, tfes_values, {})
    return filter_offline_candidates_for_args(offline_candidates, args)


def build_match_rows(relation_rows, offline_candidates):
    relation_rows = [
        dict(row)
        for row in relation_rows
        if str(row.get("method", "")) == CANDIDATE_METHOD
    ]
    offline_by_bucket = {}
    targeted_by_bucket = {}
    for row in offline_candidates:
        key = (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("seed")),
            v01.to_int(row.get("tfes")),
            str(row.get("phase", "")),
        )
        offline_by_bucket.setdefault(key, []).append(row)
    for row in relation_rows:
        key = (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("seed")),
            v01.to_int(row.get("tfes")),
            row_phase(row),
        )
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")) or is_targeted_probe(row):
            targeted_by_bucket.setdefault(key, []).append(row)

    rows = []
    for key in sorted(set(offline_by_bucket) | set(targeted_by_bucket)):
        offline_rows = offline_by_bucket.get(key, [])
        targeted_rows = targeted_by_bucket.get(key, [])
        offline_vars = {v01.to_int(row.get("var_id")) for row in offline_rows}
        targeted_vars = {v01.to_int(row.get("var_id")) for row in targeted_rows}
        targeted_probe_rows = [row for row in targeted_rows if is_targeted_probe(row)]
        matched_probe_rows = [row for row in targeted_probe_rows if v01.to_int(row.get("var_id")) in offline_vars]
        matched_offline_rows = [row for row in offline_rows if v01.to_int(row.get("var_id")) in targeted_vars]
        positive_matched = [
            row
            for row in matched_probe_rows
            if np.isfinite(v01.to_float(row.get("validation_delta"))) and v01.to_float(row.get("validation_delta")) > 0.0
        ]
        problem, seed, tfes, phase = key
        rows.append(
            {
                "problem": problem,
                "seed": int(seed),
                "tfes": int(tfes),
                "phase": phase,
                "offline_candidate_count": int(len(offline_rows)),
                "targeted_candidate_count": int(len(targeted_rows)),
                "targeted_probe_count": int(len(targeted_probe_rows)),
                "matched_offline_candidate_count": int(len(matched_offline_rows)),
                "matched_probe_count": int(len(matched_probe_rows)),
                "matched_positive_delta_count": int(len(positive_matched)),
                "targeted_probe_delta_mean": stat([row.get("validation_delta") for row in matched_probe_rows], "mean"),
            }
        )
    return rows


def build_targeted_probe_metric_rows(relation_rows, offline_candidates):
    annotated_rows = [dict(row) for row in relation_rows]
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
    matched_offline_by_problem = {}
    for row in annotated_rows:
        if str(row.get("method", "")) != CANDIDATE_METHOD:
            continue
        problem = str(row.get("problem", "")).upper()
        bucket = buckets.setdefault(
            problem,
            {
                "problem": problem,
                "method": CANDIDATE_METHOD,
                "targeted_candidate_count": 0,
                "targeted_probe_count": 0,
                "targeted_probe_accept_count": 0,
                "targeted_probe_delta_values": [],
                "matched_probe_count": 0,
                "matched_positive_delta_count": 0,
                "S6_middle_matched_probe_count": 0,
                "R6_bad_probe_count": 0,
            },
        )
        key = (
            problem,
            v01.to_int(row.get("seed")),
            v01.to_int(row.get("tfes")),
            row_phase(row),
            v01.to_int(row.get("var_id")),
        )
        if v01.to_bool(row.get("arac_targeted_probe_signature_matched")):
            bucket["targeted_candidate_count"] += 1
            if key in offline_keys:
                matched_offline_by_problem.setdefault(problem, set()).add(key)
        if is_targeted_probe(row):
            bucket["targeted_probe_count"] += 1
            accepted = v01.to_bool(row.get("validation_accepted"))
            delta = v01.to_float(row.get("validation_delta"))
            bucket["targeted_probe_accept_count"] += int(accepted)
            if np.isfinite(delta):
                bucket["targeted_probe_delta_values"].append(float(delta))
            if key in offline_keys:
                bucket["matched_probe_count"] += 1
                if problem == "S6" and row_phase(row) == "middle":
                    bucket["S6_middle_matched_probe_count"] += 1
                if np.isfinite(delta) and delta > 0.0:
                    bucket["matched_positive_delta_count"] += 1
            if problem == "R6" and np.isfinite(delta) and delta <= 0.0:
                bucket["R6_bad_probe_count"] += 1

    metric_rows = []
    for problem, bucket in sorted(buckets.items()):
        probe_count = int(bucket["targeted_probe_count"])
        deltas = np.asarray(bucket["targeted_probe_delta_values"], dtype=float)
        metric_rows.append(
            {
                "problem": problem,
                "method": CANDIDATE_METHOD,
                "targeted_candidate_count": int(bucket["targeted_candidate_count"]),
                "targeted_probe_count": int(probe_count),
                "targeted_probe_accept_count": int(bucket["targeted_probe_accept_count"]),
                "targeted_probe_accept_rate": float(bucket["targeted_probe_accept_count"] / probe_count)
                if probe_count
                else 0.0,
                "targeted_probe_delta_mean": float(np.mean(deltas)) if deltas.size else 0.0,
                "matched_offline_candidate_count": int(len(matched_offline_by_problem.get(problem, set()))),
                "matched_probe_count": int(bucket["matched_probe_count"]),
                "matched_positive_delta_count": int(bucket["matched_positive_delta_count"]),
                "extra_fe_ratio": 0.0,
                "S6_middle_matched_probe_count": int(bucket["S6_middle_matched_probe_count"]),
                "R6_bad_probe_count": int(bucket["R6_bad_probe_count"]),
            }
        )
    return metric_rows


def attach_extra_fe_ratio(metric_rows, summary_rows):
    ratio_by_problem = {
        str(row.get("problem", "")).upper(): v01.to_float(row.get("validation_extra_fe_ratio"))
        for row in summary_rows
        if row.get("method") == CANDIDATE_METHOD
    }
    for row in metric_rows:
        row["extra_fe_ratio"] = ratio_by_problem.get(str(row.get("problem", "")).upper(), 0.0)
    return metric_rows


def report_summary_table(summary_rows):
    old_candidate = v04.CANDIDATE_METHOD
    v04.CANDIDATE_METHOD = CANDIDATE_METHOD
    try:
        return v04.report_summary_table(summary_rows)
    finally:
        v04.CANDIDATE_METHOD = old_candidate


def report_targeted_table(metric_rows):
    lines = [
        "| problem | targeted_candidate | targeted_probe | accept | delta | matched_offline | matched_probe | matched_pos | extra_fe | R6_bad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| {problem} | {candidate} | {probe} | {accept:.3f} | {delta} | {matched_offline} | {matched_probe} | {matched_pos} | {extra_fe:.3%} | {r6_bad} |".format(
                problem=row["problem"],
                candidate=int(row.get("targeted_candidate_count", 0) or 0),
                probe=int(row.get("targeted_probe_count", 0) or 0),
                accept=v01.to_float(row.get("targeted_probe_accept_rate")),
                delta=v01.format_metric(row.get("targeted_probe_delta_mean")),
                matched_offline=int(row.get("matched_offline_candidate_count", 0) or 0),
                matched_probe=int(row.get("matched_probe_count", 0) or 0),
                matched_pos=int(row.get("matched_positive_delta_count", 0) or 0),
                extra_fe=v01.to_float(row.get("extra_fe_ratio")),
                r6_bad=int(row.get("R6_bad_probe_count", 0) or 0),
            )
        )
    return "\n".join(lines)


def build_acceptance_notes(summary_rows, metric_rows, v05_probe_rows, max_extra_fe_ratio=0.01):
    summary_by_key = {
        (str(row.get("problem", "")).upper(), str(row.get("method", ""))): row
        for row in summary_rows
    }
    metrics_by_problem = {str(row.get("problem", "")).upper(): row for row in metric_rows}
    v05_by_problem = {str(row.get("problem", "")).upper(): row for row in v05_probe_rows}
    s6_metric = metrics_by_problem.get("S6", {})
    r6_metric = metrics_by_problem.get("R6", {})
    s6_v06 = summary_by_key.get(("S6", CANDIDATE_METHOD), {})
    r6_v06 = summary_by_key.get(("R6", CANDIDATE_METHOD), {})
    e6_v06 = summary_by_key.get(("E6", CANDIDATE_METHOD), {})
    v05_s6_delta = v01.to_float(v05_by_problem.get("S6", {}).get("probe_delta_mean"))
    v06_s6_delta = v01.to_float(s6_metric.get("targeted_probe_delta_mean"))
    v05_r6_bad = int(v05_by_problem.get("R6", {}).get("R6_bad_probe_count", 0) or 0)
    v06_r6_bad = int(r6_metric.get("R6_bad_probe_count", 0) or 0)
    max_extra_fe = max([v01.to_float(row.get("extra_fe_ratio")) for row in metric_rows] or [0.0])

    mechanism_passed = bool(
        int(s6_metric.get("S6_middle_matched_probe_count", 0) or 0) > 0
        and np.isfinite(v06_s6_delta)
        and np.isfinite(v05_s6_delta)
        and v06_s6_delta > v05_s6_delta
        and v06_r6_bad <= v05_r6_bad
        and max_extra_fe <= max_extra_fe_ratio
    )
    performance_passed = bool(
        s6_v06
        and r6_v06
        and e6_v06
        and v01.to_float(s6_v06.get("gap_vs_disable_fast")) < 0.0
        and v01.to_float(r6_v06.get("gap_vs_disable_fast")) <= 0.01
        and v01.to_float(e6_v06.get("gap_vs_disable_fast")) <= 0.01
    )

    return "\n".join(
        [
            f"- 机制验收：{'通过' if mechanism_passed else '不通过'}。",
            f"- S6 middle matched_probe_count = {int(s6_metric.get('S6_middle_matched_probe_count', 0) or 0)}。",
            f"- S6 targeted_probe_delta_mean = {v01.format_metric(v06_s6_delta)}；V0.5 random probe delta = {v01.format_metric(v05_s6_delta)}。",
            f"- R6 bad_probe_count：V0.6 = {v06_r6_bad}，V0.5 = {v05_r6_bad}。",
            f"- max extra_fe_ratio = {v01.format_percent(max_extra_fe)}。",
            f"- 性能验收：{'通过' if performance_passed else '不通过'}。",
            f"- S6 gap vs disable-fast = {v01.format_percent(s6_v06.get('gap_vs_disable_fast')) if s6_v06 else 'n/a'}。",
            f"- R6 gap vs disable-fast = {v01.format_percent(r6_v06.get('gap_vs_disable_fast')) if r6_v06 else 'n/a'}。",
            f"- E6 gap vs disable-fast = {v01.format_percent(e6_v06.get('gap_vs_disable_fast')) if e6_v06 else 'n/a'}。",
        ]
    )


def write_report(run_rows, summary_rows, metric_rows, v05_probe_rows, args):
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    report = [
        "# ARAC-lite V0.6 Targeted Recovery Probe",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Scope: targeted middle recovery probe; max targeted probe per pass = 1; no UCB.",
        "",
        "## Quick Read",
        "",
        f"- Matrix runs: {ok_count}/{len(run_rows)} ok",
        "",
        "## Acceptance",
        "",
        build_acceptance_notes(summary_rows, metric_rows, v05_probe_rows),
        "",
        "## Matrix Summary",
        "",
        report_summary_table(summary_rows),
        "",
        "## Targeted Probe Metrics",
        "",
        report_targeted_table(metric_rows),
        "",
        "## Interpretation",
        "",
        "- V0.6 does not increase random probe frequency; it redirects the probe budget toward middle-phase relations that match the recovery signature.",
        "- `matched_probe_count > 0` is the first mechanism gate because V0.5 failed with S6 middle matched_probe_count = 0.",
        "- R6 high accept with negative delta remains a hard conflict signal; targeted probe requires positive rolling relation delta.",
        "",
        "## Artifacts",
        "",
        f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
        f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
        f"- action audit: `{ACTION_AUDIT_PATH.as_posix()}`",
        f"- summary: `{SUMMARY_PATH.as_posix()}`",
        f"- robustness: `{ROBUSTNESS_PATH.as_posix()}`",
        f"- phase summary: `{PHASE_SUMMARY_PATH.as_posix()}`",
        f"- targeted metrics: `{TARGETED_METRICS_PATH.as_posix()}`",
        f"- offline match: `{MATCH_PATH.as_posix()}`",
        f"- report: `{REPORT_PATH.as_posix()}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


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


def main():
    args = parse_args()
    run_rows, relation_rows = run_sweep(args)
    summary_rows = summarize_runs(run_rows)
    robustness_rows = build_robustness_rows(run_rows)
    action_rows = v01.build_action_attribution_rows(relation_rows)
    phase_rows = v03.build_phase_summary(relation_rows)
    offline_candidates = build_offline_candidates_for_args(args)
    targeted_rows = attach_extra_fe_ratio(build_targeted_probe_metric_rows(relation_rows, offline_candidates), summary_rows)
    match_rows = build_match_rows(relation_rows, offline_candidates)
    v05_probe_rows = v05.attach_extra_fe_ratio(v05.build_probe_metric_rows(relation_rows), summary_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)
    write_csv(ACTION_AUDIT_PATH, v01.ATTRIBUTION_FIELDNAMES, action_rows)
    write_csv(SUMMARY_PATH, v04.SUMMARY_FIELDNAMES, summary_rows)
    write_csv(ROBUSTNESS_PATH, v04.ROBUSTNESS_FIELDNAMES, robustness_rows)
    write_csv(PHASE_SUMMARY_PATH, v03.PHASE_SUMMARY_FIELDNAMES, phase_rows)
    write_csv(TARGETED_METRICS_PATH, TARGETED_METRIC_FIELDNAMES, targeted_rows)
    write_csv(MATCH_PATH, MATCH_FIELDNAMES, match_rows)
    write_report(run_rows, summary_rows, targeted_rows, v05_probe_rows, args)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"targeted metric rows -> {len(targeted_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
