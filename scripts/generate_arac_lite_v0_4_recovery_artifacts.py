import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_runs"
V03_RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_relation_action_audit.csv"
OFFLINE_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_offline_audit.csv"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_relation_action_audit.csv"
ACTION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_summary.csv"
ROBUSTNESS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_robustness.csv"
PHASE_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_phase_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_report.md"

DEFAULT_PROBLEMS = ["E6", "S6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20
CANDIDATE_METHOD = "arac-lite-v0.4-recovery"
DISABLE_FAST_METHOD = "arac-lite-v0.1-disable-fast"
METHODS = ["no-coordination", "validated-selective-conflict", DISABLE_FAST_METHOD, CANDIDATE_METHOD]
PHASES = ["early", "middle", "late"]
TIE_EPS = 1e-12


spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_3",
    SCRIPT_ROOT / "generate_arac_lite_v0_3_artifacts.py",
)
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)

v02 = v03.v02
v01 = v03.v01
v01.RUNS_ROOT = RUNS_ROOT


OFFLINE_AUDIT_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "phase",
    "fusion_count",
    "fusion_validation_attempt_count",
    "fusion_validation_accept_count",
    "fusion_accept_rate",
    "fusion_delta_mean",
    "fusion_positive_delta_rate",
    "disable_count",
    "disable_false_negative_count",
    "disable_false_negative_rate",
    "freeze_count",
    "freeze_false_negative_count",
    "freeze_false_negative_rate",
    "recovery_candidate_count",
    "recovery_candidate_rate",
]

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
    "gap_median",
    "worst_case_gap",
    "best_case_gap",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.4 Fusion Recovery artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--v0-3-audit-path", type=Path, default=V03_RELATION_AUDIT_PATH)
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def iter_csv(path):
    if not Path(path).exists():
        return
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


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
    return float("nan")


def gap(value, baseline):
    value = v01.to_float(value)
    baseline = v01.to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-300:
        return float("nan")
    return float((value - baseline) / abs(baseline))


def relation_run_key(row):
    return (
        str(row.get("problem", "")).upper(),
        v01.to_int(row.get("seed")),
        v01.to_int(row.get("tfes")),
        str(row.get("method", "")),
    )


def relation_var_key(row):
    return (*relation_run_key(row), v01.to_int(row.get("var_id")))


def relation_phase(row, max_pass_by_run):
    return v03.phase_for_pass(v01.to_int(row.get("pass_id")), max_pass_by_run.get(relation_run_key(row), 0))


def build_relation_scan_state(rows):
    max_pass_by_run = {}
    future_accepted_fusion_max_pass = {}
    for row in rows:
        run_key = relation_run_key(row)
        pass_id = v01.to_int(row.get("pass_id"))
        max_pass_by_run[run_key] = max(max_pass_by_run.get(run_key, 0), pass_id)
        if (
            str(row.get("action_candidate", "")) == "Fusion"
            and v01.to_bool(row.get("validation_attempted"))
            and v01.to_bool(row.get("validation_accepted"))
        ):
            var_key = relation_var_key(row)
            future_accepted_fusion_max_pass[var_key] = max(
                future_accepted_fusion_max_pass.get(var_key, -1),
                pass_id,
            )
    return max_pass_by_run, future_accepted_fusion_max_pass


def init_offline_bucket(problem, tfes, method, phase):
    return {
        "problem": str(problem).upper(),
        "tfes": int(tfes),
        "method": str(method),
        "phase": str(phase),
        "fusion_count": 0,
        "fusion_validation_attempt_count": 0,
        "fusion_validation_accept_count": 0,
        "fusion_delta_values": [],
        "disable_count": 0,
        "disable_false_negative_count": 0,
        "freeze_count": 0,
        "freeze_false_negative_count": 0,
        "recovery_candidate_count": 0,
    }


def accumulate_offline_bucket(bucket, row, future_accepted_fusion_max_pass):
    action = str(row.get("action_candidate", ""))
    pass_id = v01.to_int(row.get("pass_id"))
    future_accepted = future_accepted_fusion_max_pass.get(relation_var_key(row), -1) > pass_id
    if action == "Fusion":
        bucket["fusion_count"] += 1
        if v01.to_bool(row.get("validation_attempted")):
            bucket["fusion_validation_attempt_count"] += 1
            if v01.to_bool(row.get("validation_accepted")):
                bucket["fusion_validation_accept_count"] += 1
        delta = v01.to_float(row.get("validation_delta"))
        if np.isfinite(delta):
            bucket["fusion_delta_values"].append(float(delta))
    elif action == "Disable":
        bucket["disable_count"] += 1
        if future_accepted:
            bucket["disable_false_negative_count"] += 1
            if str(bucket["phase"]) in {"middle", "late"}:
                bucket["recovery_candidate_count"] += 1
    elif action == "Freeze":
        bucket["freeze_count"] += 1
        if future_accepted:
            bucket["freeze_false_negative_count"] += 1
            if str(bucket["phase"]) in {"middle", "late"}:
                bucket["recovery_candidate_count"] += 1


def finalize_offline_bucket(bucket):
    fusion_attempts = int(bucket.pop("fusion_validation_attempt_count"))
    fusion_accepts = int(bucket.pop("fusion_validation_accept_count"))
    fusion_deltas = np.asarray(bucket.pop("fusion_delta_values"), dtype=float)
    disable_count = int(bucket["disable_count"])
    freeze_count = int(bucket["freeze_count"])
    disabled_or_frozen = disable_count + freeze_count
    bucket.update(
        {
            "fusion_validation_attempt_count": int(fusion_attempts),
            "fusion_validation_accept_count": int(fusion_accepts),
            "fusion_accept_rate": float(fusion_accepts / fusion_attempts) if fusion_attempts else 0.0,
            "fusion_delta_mean": float(np.mean(fusion_deltas)) if fusion_deltas.size else float("nan"),
            "fusion_positive_delta_rate": float(np.mean(fusion_deltas > 0.0)) if fusion_deltas.size else 0.0,
            "disable_false_negative_rate": float(bucket["disable_false_negative_count"] / disable_count)
            if disable_count
            else 0.0,
            "freeze_false_negative_rate": float(bucket["freeze_false_negative_count"] / freeze_count)
            if freeze_count
            else 0.0,
            "recovery_candidate_rate": float(bucket["recovery_candidate_count"] / disabled_or_frozen)
            if disabled_or_frozen
            else 0.0,
        }
    )
    return bucket


def build_offline_recovery_audit_rows(rows, method_filter=DISABLE_FAST_METHOD):
    rows = [dict(row) for row in rows if str(row.get("method", "")) == str(method_filter)]
    max_pass_by_run, future_accepted_fusion_max_pass = build_relation_scan_state(rows)
    grouped = {}
    for row in rows:
        phase = relation_phase(row, max_pass_by_run)
        key = (
            str(row.get("problem", "")).upper(),
            v01.to_int(row.get("tfes")),
            str(row.get("method", "")),
            phase,
        )
        bucket = grouped.setdefault(
            key,
            init_offline_bucket(key[0], key[1], key[2], key[3]),
        )
        accumulate_offline_bucket(bucket, row, future_accepted_fusion_max_pass)
    return [finalize_offline_bucket(grouped[key]) for key in sorted(grouped)]


def build_offline_recovery_audit_rows_from_csv(path, method_filter=DISABLE_FAST_METHOD):
    filtered_rows = [row for row in iter_csv(path) if str(row.get("method", "")) == str(method_filter)]
    return build_offline_recovery_audit_rows(filtered_rows, method_filter=method_filter)


def recovery_variant():
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
        (CANDIDATE_METHOD, v01.apply_variant(base_arac, recovery_variant())),
    ]


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


def build_robustness_rows(run_rows):
    by_case = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), int(row["seed"]))
        by_case.setdefault(key, {})[row["method"]] = row
    rows = []
    for baseline in ["no-coordination", "validated-selective-conflict", DISABLE_FAST_METHOD]:
        for problem in sorted({key[0] for key in by_case}):
            values = []
            for key, methods in by_case.items():
                if key[0] != problem or CANDIDATE_METHOD not in methods or baseline not in methods:
                    continue
                values.append(
                    {
                        "tfes": key[1],
                        "candidate": v01.to_float(methods[CANDIDATE_METHOD].get("best_error")),
                        "baseline": v01.to_float(methods[baseline].get("best_error")),
                    }
                )
            if not values:
                continue
            for tfes_value in sorted({item["tfes"] for item in values}):
                selected = [item for item in values if item["tfes"] == tfes_value]
                rows.append(build_robustness_row(problem, tfes_value, baseline, selected))
            rows.append(build_robustness_row(problem, "all", baseline, values))
    return rows


def build_robustness_row(problem, tfes, baseline, values):
    gaps = []
    wins = losses = ties = 0
    for item in values:
        candidate = float(item["candidate"])
        baseline_value = float(item["baseline"])
        if not np.isfinite(candidate) or not np.isfinite(baseline_value):
            continue
        current_gap = gap(candidate, baseline_value)
        gaps.append(current_gap)
        if candidate < baseline_value - TIE_EPS:
            wins += 1
        elif candidate > baseline_value + TIE_EPS:
            losses += 1
        else:
            ties += 1
    paired = len(gaps)
    non_worse = wins + ties
    return {
        "problem": str(problem),
        "tfes": tfes,
        "candidate": CANDIDATE_METHOD,
        "baseline": baseline,
        "paired_runs": int(paired),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "non_worse_count": int(non_worse),
        "non_worse_rate": float(non_worse / paired) if paired else 0.0,
        "gap_mean": stat(gaps, "mean"),
        "gap_median": stat(gaps, "median"),
        "worst_case_gap": float(np.max(finite_values(gaps))) if finite_values(gaps).size else float("nan"),
        "best_case_gap": float(np.min(finite_values(gaps))) if finite_values(gaps).size else float("nan"),
    }


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


def report_summary_table(summary_rows):
    lines = [
        "| problem | method | best_mean | gap_vs_no | gap_vs_validated | gap_vs_disable_fast | Fusion | Freeze | Disable | fusion_accept |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {problem} | {method} | {best} | {gap_no} | {gap_val} | {gap_disable} | {fusion} | {freeze} | {disable} | {accept:.3f} |".format(
                problem=row["problem"],
                method=row["method"],
                best=v01.format_metric(row.get("best_mean")),
                gap_no=v01.format_percent(row.get("gap_vs_no_coordination")),
                gap_val=v01.format_percent(row.get("gap_vs_validated")),
                gap_disable=v01.format_percent(row.get("gap_vs_disable_fast")),
                fusion=int(row.get("fusion_count", 0) or 0),
                freeze=int(row.get("freeze_count", 0) or 0),
                disable=int(row.get("disable_count", 0) or 0),
                accept=v01.to_float(row.get("fusion_validation_accept_rate")),
            )
        )
    return "\n".join(lines)


def report_offline_table(offline_rows):
    lines = [
        "| problem | phase | fusion_count | fusion_accept | fusion_delta | fusion_pos_delta | disable_fn | freeze_fn | recovery_candidates |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    selected = [
        row
        for row in offline_rows
        if row["problem"] in {"E6", "S6", "A6", "R6"} and int(row["tfes"]) == 10000
    ]
    phase_order = {phase: index for index, phase in enumerate(PHASES)}
    for row in sorted(selected, key=lambda item: (item["problem"], phase_order.get(item["phase"], 99))):
        lines.append(
            "| {problem} | {phase} | {fusion} | {accept:.3f} | {delta} | {pos:.3f} | {disable_fn:.3f} | {freeze_fn:.3f} | {candidates} |".format(
                problem=row["problem"],
                phase=row["phase"],
                fusion=int(row.get("fusion_count", 0) or 0),
                accept=v01.to_float(row.get("fusion_accept_rate")),
                delta=v01.format_metric(row.get("fusion_delta_mean")),
                pos=v01.to_float(row.get("fusion_positive_delta_rate")),
                disable_fn=v01.to_float(row.get("disable_false_negative_rate")),
                freeze_fn=v01.to_float(row.get("freeze_false_negative_rate")),
                candidates=int(row.get("recovery_candidate_count", 0) or 0),
            )
        )
    return "\n".join(lines)


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_online_recovery_diagnostics(relation_rows):
    v04_rows = [row for row in relation_rows if str(row.get("method", "")) == CANDIDATE_METHOD]
    reason_counts = {}
    problem_counts = {}
    candidate_count = 0
    for row in v04_rows:
        problem = str(row.get("problem", ""))
        action = str(row.get("action_candidate", ""))
        reason = str(row.get("arac_recovery_reason", ""))
        is_candidate = truthy(row.get("arac_recovery_candidate", ""))
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if is_candidate:
            candidate_count += 1
        if problem:
            problem_counts.setdefault(
                problem,
                {"Fusion": 0, "Freeze": 0, "Disable": 0, "recovery_candidate": 0},
            )
            if action in {"Fusion", "Freeze", "Disable"}:
                problem_counts[problem][action] += 1
            if is_candidate:
                problem_counts[problem]["recovery_candidate"] += 1
    return {
        "v04_rows": len(v04_rows),
        "candidate_count": candidate_count,
        "reason_counts": reason_counts,
        "problem_counts": problem_counts,
    }


def report_online_recovery_diagnostics(relation_rows):
    diagnostics = build_online_recovery_diagnostics(relation_rows)
    reason_text = ", ".join(
        f"{reason}={count}" for reason, count in sorted(diagnostics["reason_counts"].items())
    )
    if not reason_text:
        reason_text = "n/a"
    lines = [
        "## Online Recovery Diagnostics",
        "",
        f"- V0.4 relation rows: {diagnostics['v04_rows']}",
        f"- V0.4 recovery candidate rows: {diagnostics['candidate_count']}",
        f"- V0.4 recovery blocking reasons: {reason_text}",
    ]
    if diagnostics["problem_counts"]:
        lines.extend(
            [
                "",
                "| problem | Fusion | Freeze | Disable | recovery_candidate |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for problem, counts in sorted(diagnostics["problem_counts"].items()):
            lines.append(
                "| {problem} | {fusion} | {freeze} | {disable} | {candidate} |".format(
                    problem=problem,
                    fusion=counts["Fusion"],
                    freeze=counts["Freeze"],
                    disable=counts["Disable"],
                    candidate=counts["recovery_candidate"],
                )
            )
    return "\n".join(lines)


def write_report(offline_rows, run_rows, summary_rows, robustness_rows, args, relation_rows=None):
    relation_rows = relation_rows or []
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
        "# ARAC-lite V0.4 Fusion Recovery",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Scope: freeze `arac-lite-v0.1-disable-fast`; no UCB; no global threshold sweep.",
        "",
        "## Quick Read",
        "",
        f"- Offline recovery audit rows: {len(offline_rows)}",
        f"- Matrix runs: {ok_count}/{len(run_rows)} ok",
        f"- S6 V0.4 gap vs disable-fast: {v01.format_percent(s6_candidate.get('gap_vs_disable_fast')) if s6_candidate else 'n/a'}",
        f"- R6 V0.4 gap vs disable-fast: {v01.format_percent(r6_candidate.get('gap_vs_disable_fast')) if r6_candidate else 'n/a'}",
        "",
        "## Offline Recovery Audit",
        "",
        report_offline_table(offline_rows),
        "",
        "## Matrix Summary",
        "",
        report_summary_table(summary_rows),
        "",
        report_online_recovery_diagnostics(relation_rows),
        "",
        "## Interpretation",
        "",
        "- V0.4 is intentionally conservative: recovery is only allowed in middle/late and requires enough rolling Fusion attempts, accept rate, positive mean delta, and positive-delta rate.",
        "- If online recovery candidate rows stay at zero, the offline recovery window exists but the current gate cannot reach it from disable-fast history; this points to a probing problem rather than an immediate UCB/bandit requirement.",
        "- If R6 worsens against disable-fast, recovery is too permissive for noisy/conflict-heavy relations and should not advance.",
        "- If S6/E6 improve without R6 degradation, V0.4 supports ARAC as a recoverable relation-to-action controller rather than only a bad-coordination shield.",
        "",
        "## Artifacts",
        "",
        f"- offline audit: `{OFFLINE_AUDIT_PATH.as_posix()}`",
        f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
        f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
        f"- action audit: `{ACTION_AUDIT_PATH.as_posix()}`",
        f"- summary: `{SUMMARY_PATH.as_posix()}`",
        f"- robustness: `{ROBUSTNESS_PATH.as_posix()}`",
        f"- phase summary: `{PHASE_SUMMARY_PATH.as_posix()}`",
        f"- report: `{REPORT_PATH.as_posix()}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    offline_rows = build_offline_recovery_audit_rows_from_csv(args.v0_3_audit_path)
    write_csv(OFFLINE_AUDIT_PATH, OFFLINE_AUDIT_FIELDNAMES, offline_rows)
    if args.offline_only:
        write_report(offline_rows, [], [], [], args)
        print(f"offline recovery audit rows -> {len(offline_rows)}")
        print(f"offline audit -> {OFFLINE_AUDIT_PATH}")
        print(f"report -> {REPORT_PATH}")
        return

    run_rows, relation_rows = run_sweep(args)
    summary_rows = summarize_runs(run_rows)
    robustness_rows = build_robustness_rows(run_rows)
    action_rows = v01.build_action_attribution_rows(relation_rows)
    phase_rows = v03.build_phase_summary(relation_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)
    write_csv(ACTION_AUDIT_PATH, v01.ATTRIBUTION_FIELDNAMES, action_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(ROBUSTNESS_PATH, ROBUSTNESS_FIELDNAMES, robustness_rows)
    write_csv(PHASE_SUMMARY_PATH, v03.PHASE_SUMMARY_FIELDNAMES, phase_rows)
    write_report(offline_rows, run_rows, summary_rows, robustness_rows, args, relation_rows)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"offline recovery audit rows -> {len(offline_rows)}")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
