import argparse
import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_relation_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_report.md"

DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20
METHOD_CONFIGS = [
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("eq8-correct", CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct.json"),
    ("validated-selective-conflict", CONFIG_ROOT / "validated-selective-conflict.json"),
    ("arac-lite-rule", CONFIG_ROOT / "arac-lite-rule.json"),
]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)


RUN_DETAIL_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "coordination_mode",
    "cc_pass_group_fes",
    "cc_pass_count",
    "best_error",
    "final_error",
    "fe_used",
    "runtime",
    "fusion_count",
    "freeze_count",
    "disable_count",
    "fusion_validation_attempt_count",
    "fusion_validation_accept_count",
    "fusion_validation_accept_rate",
    "validation_attempt_count",
    "validation_accept_count",
    "validation_reject_count",
    "validation_accept_rate",
    "validation_extra_fe_ratio",
    "relation_history_size",
    "status",
]

RELATION_AUDIT_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "pass_id",
    "var_id",
    "proposal_support",
    "positive_proposal_count",
    "negative_proposal_count",
    "proposal_std",
    "proposal_std_ratio",
    "raw_update_ratio",
    "update_magnitude_ratio",
    "validation_attempted",
    "validation_accepted",
    "validation_accept_rate",
    "validation_delta",
    "action_candidate",
    "action_reason",
    "best_improvement_after_action",
    "relation_attempt_count",
    "relation_accept_rate",
    "relation_mean_validation_delta",
    "relation_reject_streak",
]

SUMMARY_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "runs",
    "best_mean",
    "best_std",
    "final_mean",
    "final_std",
    "gap_vs_no_coordination",
    "gap_vs_eq8_correct",
    "gap_vs_validated_selective",
    "fusion_count",
    "freeze_count",
    "disable_count",
    "fusion_validation_accept_rate",
    "validation_accept_rate",
    "validation_extra_fe_ratio",
    "cc_pass_count_mean",
    "relation_history_size_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0 relation-to-action artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    return parser.parse_args()


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_or_nan(values):
    numeric = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
    return float(np.mean(numeric)) if numeric.size else float("nan")


def std_or_nan(values):
    numeric = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
    return float(np.std(numeric)) if numeric.size else float("nan")


def format_metric(value):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6e}"


def format_percent(value):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100.0:+.3f}%"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def load_method_config(config_path, cc_pass_group_fes):
    config = hcc_es.load_info_aware_nda_config(config_path, enable=False)
    return replace(
        config,
        cc_pass_group_fes=max(1, int(cc_pass_group_fes)),
        cc_min_passes=3,
    ).normalized()


def summarize_action_rows(rows):
    rows = list(rows or [])
    fusion_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Fusion"]
    freeze_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Freeze"]
    disable_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Disable"]
    validated_fusion_rows = [row for row in fusion_rows if bool(row.get("validation_attempted"))]
    accepted_fusion_rows = [row for row in validated_fusion_rows if bool(row.get("validation_accepted"))]
    return {
        "fusion_count": int(len(fusion_rows)),
        "freeze_count": int(len(freeze_rows)),
        "disable_count": int(len(disable_rows)),
        "fusion_validation_attempt_count": int(len(validated_fusion_rows)),
        "fusion_validation_accept_count": int(len(accepted_fusion_rows)),
        "fusion_validation_accept_rate": float(len(accepted_fusion_rows) / len(validated_fusion_rows))
        if validated_fusion_rows
        else 0.0,
    }


def normalize_relation_rows(fusion_rows, problem, seed, tfes, method):
    relation_rows = []
    for row in list(fusion_rows or []):
        attempted = bool(row.get("validation_attempted"))
        accepted = bool(row.get("validation_accepted"))
        relation_rows.append(
            {
                "problem": str(problem).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method),
                "pass_id": int(row.get("cycle_id", 0) or 0),
                "var_id": int(row.get("var_id", -1)),
                "proposal_support": int(row.get("proposal_count", 0) or 0),
                "positive_proposal_count": int(row.get("positive_proposal_count", 0) or 0),
                "negative_proposal_count": int(row.get("negative_proposal_count", 0) or 0),
                "proposal_std": to_float(row.get("proposal_value_std")),
                "proposal_std_ratio": to_float(row.get("proposal_value_std_ratio")),
                "raw_update_ratio": to_float(row.get("raw_update_ratio")),
                "update_magnitude_ratio": to_float(row.get("update_magnitude_ratio")),
                "validation_attempted": bool(attempted),
                "validation_accepted": bool(accepted),
                "validation_accept_rate": 1.0 if attempted and accepted else 0.0,
                "validation_delta": to_float(row.get("fitness_delta")),
                "action_candidate": str(row.get("action_candidate", "")),
                "action_reason": str(row.get("action_reason", "")),
                "best_improvement_after_action": bool(row.get("post_coordination_best_improved")),
                "relation_attempt_count": int(row.get("relation_attempt_count", 0) or 0),
                "relation_accept_rate": to_float(row.get("relation_accept_rate")),
                "relation_mean_validation_delta": to_float(row.get("relation_mean_validation_delta")),
                "relation_reject_streak": int(row.get("relation_reject_streak", 0) or 0),
            }
        )
    return relation_rows


def run_one_case(method_name, config, problem_code, seed, tfes):
    output_dir = RUNS_ROOT / f"tfes-{int(tfes)}" / method_name
    try:
        result = hcc_es.run_problem_seed_task(
            problem_code,
            int(seed),
            int(tfes),
            hcc_es.HCC_ES_METHOD,
            output_dir,
            record_fes=[],
            info_aware_config=config,
            method_label=method_name,
        )
        detail = dict(result.get("detail", {}))
        metadata = dict(result.get("metadata", {}))
        fusion_rows = list(metadata.get("shared_variable_fusion_rows", []))
        action_summary = summarize_action_rows(fusion_rows)
        validation_summary = hcc_es.summarize_validated_coordination_rows(
            metadata.get("validated_coordination_rows", []),
            total_max_fes=tfes,
        )
        run_row = {
            "problem": str(problem_code).upper(),
            "seed": int(seed),
            "tfes": int(tfes),
            "method": str(method_name),
            "coordination_mode": hcc_es.resolve_shared_variable_coordination_mode(config),
            "cc_pass_group_fes": int(getattr(config, "cc_pass_group_fes", 0) or 0),
            "cc_pass_count": int(metadata.get("cc_pass_count", 0) or 0),
            "best_error": to_float(detail.get("best_fitness")),
            "final_error": to_float(detail.get("final_fitness")),
            "fe_used": int(to_float(detail.get("fe_used"))),
            "runtime": to_float(detail.get("runtime")),
            **action_summary,
            "validation_attempt_count": int(validation_summary.get("validation_attempt_count", 0)),
            "validation_accept_count": int(validation_summary.get("validation_accept_count", 0)),
            "validation_reject_count": int(validation_summary.get("validation_reject_count", 0)),
            "validation_accept_rate": to_float(validation_summary.get("validation_accept_rate")),
            "validation_extra_fe_ratio": to_float(validation_summary.get("validation_extra_fe_ratio")),
            "relation_history_size": int(len(metadata.get("arac_relation_history", {}) or {})),
            "status": str(detail.get("status", "ok")),
        }
        relation_rows = normalize_relation_rows(fusion_rows, problem_code, seed, tfes, method_name)
        return run_row, relation_rows
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "coordination_mode": hcc_es.resolve_shared_variable_coordination_mode(config),
                "cc_pass_group_fes": int(getattr(config, "cc_pass_group_fes", 0) or 0),
                "cc_pass_count": 0,
                "best_error": float("nan"),
                "final_error": float("nan"),
                "fe_used": 0,
                "runtime": float("nan"),
                "fusion_count": 0,
                "freeze_count": 0,
                "disable_count": 0,
                "fusion_validation_attempt_count": 0,
                "fusion_validation_accept_count": 0,
                "fusion_validation_accept_rate": 0.0,
                "validation_attempt_count": 0,
                "validation_accept_count": 0,
                "validation_reject_count": 0,
                "validation_accept_rate": 0.0,
                "validation_extra_fe_ratio": 0.0,
                "relation_history_size": 0,
                "status": f"error:{type(exc).__name__}:{exc}",
            },
            [],
        )


def gap(value, baseline):
    value = to_float(value)
    baseline = to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-300:
        return float("nan")
    return float((value - baseline) / abs(baseline))


def summarize_runs(run_rows):
    grouped = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        grouped.setdefault(key, []).append(row)
    best_baselines = {}
    for (problem, tfes, method), rows in grouped.items():
        if method in {"no-coordination", "eq8-correct", "validated-selective-conflict"}:
            best_baselines[(problem, tfes, method)] = mean_or_nan(row.get("best_error") for row in rows)
    summary_rows = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_mean = mean_or_nan(row.get("best_error") for row in rows)
        summary_rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "runs": int(len(rows)),
                "best_mean": best_mean,
                "best_std": std_or_nan(row.get("best_error") for row in rows),
                "final_mean": mean_or_nan(row.get("final_error") for row in rows),
                "final_std": std_or_nan(row.get("final_error") for row in rows),
                "gap_vs_no_coordination": gap(best_mean, best_baselines.get((problem, tfes, "no-coordination"))),
                "gap_vs_eq8_correct": gap(best_mean, best_baselines.get((problem, tfes, "eq8-correct"))),
                "gap_vs_validated_selective": gap(best_mean, best_baselines.get((problem, tfes, "validated-selective-conflict"))),
                "fusion_count": int(sum(int(row.get("fusion_count", 0) or 0) for row in rows)),
                "freeze_count": int(sum(int(row.get("freeze_count", 0) or 0) for row in rows)),
                "disable_count": int(sum(int(row.get("disable_count", 0) or 0) for row in rows)),
                "fusion_validation_accept_rate": mean_or_nan(row.get("fusion_validation_accept_rate") for row in rows),
                "validation_accept_rate": mean_or_nan(row.get("validation_accept_rate") for row in rows),
                "validation_extra_fe_ratio": mean_or_nan(row.get("validation_extra_fe_ratio") for row in rows),
                "cc_pass_count_mean": mean_or_nan(row.get("cc_pass_count") for row in rows),
                "relation_history_size_mean": mean_or_nan(row.get("relation_history_size") for row in rows),
            }
        )
    return summary_rows


def report_table(summary_rows, tfes):
    lines = [
        f"## TFEs = {int(tfes)}",
        "",
        "| problem | method | best_mean | final_mean | gap_vs_no | gap_vs_eq8 | gap_vs_validated | Fusion | Freeze | Disable | fusion_accept | cc_pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [item for item in summary_rows if int(item["tfes"]) == int(tfes)]:
        lines.append(
            "| {problem} | {method} | {best} | {final} | {gap_no} | {gap_eq8} | {gap_val} | {fusion} | {freeze} | {disable} | {accept:.3f} | {cc:.1f} |".format(
                problem=row["problem"],
                method=row["method"],
                best=format_metric(row.get("best_mean")),
                final=format_metric(row.get("final_mean")),
                gap_no=format_percent(row.get("gap_vs_no_coordination")),
                gap_eq8=format_percent(row.get("gap_vs_eq8_correct")),
                gap_val=format_percent(row.get("gap_vs_validated_selective")),
                fusion=int(row.get("fusion_count", 0) or 0),
                freeze=int(row.get("freeze_count", 0) or 0),
                disable=int(row.get("disable_count", 0) or 0),
                accept=to_float(row.get("fusion_validation_accept_rate")),
                cc=to_float(row.get("cc_pass_count_mean")),
            )
        )
    return "\n".join(lines)


def write_report(summary_rows, run_rows, relation_rows, args):
    arac_rows = [row for row in summary_rows if row["method"] == "arac-lite-rule"]
    r6_rows = [row for row in arac_rows if row["problem"] == "R6"]
    r6_action_line = "- ARAC-lite R6 actions: n/a"
    if r6_rows:
        r6_row = r6_rows[0]
        r6_action_line = (
            "- ARAC-lite R6 actions: "
            f"Fusion={int(r6_row.get('fusion_count', 0) or 0)}, "
            f"Freeze={int(r6_row.get('freeze_count', 0) or 0)}, "
            f"Disable={int(r6_row.get('disable_count', 0) or 0)}, "
            f"gap_vs_no={format_percent(r6_row.get('gap_vs_no_coordination'))}"
        )
    report = [
        "# ARAC-lite V0",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Scope: relation-to-action audit + rule-based Fusion/Freeze/Disable. No UCB, no selector tuning, no owner-soft primary action.",
        "",
        "## Quick Read",
        "",
        f"- Total runs: {len(run_rows)}",
        f"- Relation-action audit rows: {len(relation_rows)}",
        r6_action_line,
        "",
        "## Performance And Actions",
    ]
    for tfes in args.tfes:
        report.extend(["", report_table(summary_rows, tfes)])
    report.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
            f"- summary: `{SUMMARY_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    problems = [hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    run_rows = []
    relation_rows = []
    for method_name, config_path in METHOD_CONFIGS:
        config = load_method_config(config_path, args.cc_pass_group_fes)
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    run_row, case_relation_rows = run_one_case(method_name, config, problem_code, seed, tfes)
                    run_rows.append(run_row)
                    relation_rows.extend(case_relation_rows)
                    print(
                        f"{problem_code} {method_name} seed={seed} tfes={tfes}: "
                        f"{run_row['status']} fusion={run_row['fusion_count']} "
                        f"freeze={run_row['freeze_count']} disable={run_row['disable_count']}"
                    )
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, RELATION_AUDIT_FIELDNAMES, relation_rows)
    summary_rows = summarize_runs(run_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_report(summary_rows, run_rows, relation_rows, args)
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
