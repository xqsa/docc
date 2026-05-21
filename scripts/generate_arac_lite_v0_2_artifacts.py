import argparse
import csv
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_2_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_2_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_2_relation_action_audit.csv"
ACTION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_2_action_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_2_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_2_report.md"

DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20
HELD_OUT_PROBLEMS = {"E4", "S4", "S6"}
TUNING_PROBLEMS = {"A6", "E6", "R6"}

spec = importlib.util.spec_from_file_location(
    "arac_lite_v0_1",
    SCRIPT_ROOT / "generate_arac_lite_v0_1_artifacts.py",
)
v01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v01)

v01.RUNS_ROOT = RUNS_ROOT

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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.2 fixed-rule generalization artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    return parser.parse_args()


def disable_fast_variant():
    return v01.ThresholdVariant(
        "arac-lite-v0.1-disable-fast",
        {
            "arac_lite_history_min_attempts": 1,
            "arac_lite_disable_reject_streak": 1,
            "arac_lite_disable_accept_rate_threshold": 0.0,
            "arac_lite_disable_mean_delta_threshold": 0.0,
        },
    )


def fixed_method_configs(cc_pass_group_fes):
    base_arac = v01.load_base_arac_config(cc_pass_group_fes)
    return [
        ("no-coordination", v01.load_named_config(CONFIG_ROOT / "no-coordination.json", cc_pass_group_fes)),
        ("eq8-correct", v01.load_named_config(CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct.json", cc_pass_group_fes)),
        (
            "validated-selective-conflict",
            v01.load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", cc_pass_group_fes),
        ),
        ("arac-lite-v0", v01.apply_variant(base_arac, v01.ThresholdVariant("arac-lite-v0", {}))),
        ("arac-lite-v0.1-disable-fast", v01.apply_variant(base_arac, disable_fast_variant())),
    ]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def gap(value, baseline):
    value = v01.to_float(value)
    baseline = v01.to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-300:
        return float("nan")
    return float((value - baseline) / abs(baseline))


def summarize_runs(run_rows):
    grouped = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        grouped.setdefault(key, []).append(row)

    best_baselines = {}
    baseline_methods = {
        "no-coordination",
        "eq8-correct",
        "validated-selective-conflict",
        "arac-lite-v0",
    }
    for (problem, tfes, method), rows in grouped.items():
        if method in baseline_methods:
            best_baselines[(problem, tfes, method)] = v01.mean_or_nan(row.get("best_error") for row in rows)

    summary_rows = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_mean = v01.mean_or_nan(row.get("best_error") for row in rows)
        summary_rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "runs": int(len(rows)),
                "best_mean": best_mean,
                "best_std": v01.std_or_nan(row.get("best_error") for row in rows),
                "final_mean": v01.mean_or_nan(row.get("final_error") for row in rows),
                "final_std": v01.std_or_nan(row.get("final_error") for row in rows),
                "gap_vs_no_coordination": gap(best_mean, best_baselines.get((problem, tfes, "no-coordination"))),
                "gap_vs_eq8_correct": gap(best_mean, best_baselines.get((problem, tfes, "eq8-correct"))),
                "gap_vs_validated": gap(best_mean, best_baselines.get((problem, tfes, "validated-selective-conflict"))),
                "gap_vs_arac_lite_v0": gap(best_mean, best_baselines.get((problem, tfes, "arac-lite-v0"))),
                "fusion_count": int(sum(int(row.get("fusion_count", 0) or 0) for row in rows)),
                "freeze_count": int(sum(int(row.get("freeze_count", 0) or 0) for row in rows)),
                "disable_count": int(sum(int(row.get("disable_count", 0) or 0) for row in rows)),
                "fusion_validation_accept_rate": v01.mean_or_nan(row.get("fusion_validation_accept_rate") for row in rows),
                "validation_accept_rate": v01.mean_or_nan(row.get("validation_accept_rate") for row in rows),
                "validation_extra_fe_ratio": v01.mean_or_nan(row.get("validation_extra_fe_ratio") for row in rows),
                "cc_pass_count_mean": v01.mean_or_nan(row.get("cc_pass_count") for row in rows),
                "relation_history_size_mean": v01.mean_or_nan(row.get("relation_history_size") for row in rows),
            }
        )
    return summary_rows


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
    return run_rows, relation_rows, summarize_runs(run_rows)


def method_summary(summary_rows, method, problems=None):
    selected = [row for row in summary_rows if row["method"] == method]
    if problems is not None:
        selected = [row for row in selected if row["problem"] in problems]
    return selected


def count_non_worse(rows, gap_field):
    finite = [v01.to_float(row.get(gap_field)) for row in rows if np.isfinite(v01.to_float(row.get(gap_field)))]
    return sum(1 for value in finite if value <= 0.0), len(finite)


def total_action_count(action_rows, method, action, problems=None):
    selected = [
        row
        for row in action_rows
        if row["method"] == method and row["action_candidate"] == action
    ]
    if problems is not None:
        selected = [row for row in selected if row["problem"] in problems]
    return sum(int(row.get("action_count", 0) or 0) for row in selected)


def avg_action_metric(action_rows, method, action, metric, problems=None):
    selected = [
        row
        for row in action_rows
        if row["method"] == method and row["action_candidate"] == action
    ]
    if problems is not None:
        selected = [row for row in selected if row["problem"] in problems]
    return v01.mean_or_nan(row.get(metric) for row in selected)


def generalization_judgment(summary_rows, action_rows):
    candidate = method_summary(summary_rows, "arac-lite-v0.1-disable-fast")
    held_out = [row for row in candidate if row["problem"] in HELD_OUT_PROBLEMS]
    tuning = [row for row in candidate if row["problem"] in TUNING_PROBLEMS]
    held_v0_wins, held_v0_total = count_non_worse(held_out, "gap_vs_arac_lite_v0")
    held_validated_wins, held_validated_total = count_non_worse(held_out, "gap_vs_validated")
    all_v0_wins, all_v0_total = count_non_worse(candidate, "gap_vs_arac_lite_v0")
    all_validated_wins, all_validated_total = count_non_worse(candidate, "gap_vs_validated")
    held_fusion_count = total_action_count(action_rows, "arac-lite-v0.1-disable-fast", "Fusion", HELD_OUT_PROBLEMS)
    held_disable_false_negative = avg_action_metric(
        action_rows,
        "arac-lite-v0.1-disable-fast",
        "Disable",
        "disable_false_negative_rate",
        HELD_OUT_PROBLEMS,
    )
    held_max_validated_gap = max(
        [v01.to_float(row.get("gap_vs_validated")) for row in held_out if np.isfinite(v01.to_float(row.get("gap_vs_validated")))]
        or [float("nan")]
    )
    stable = (
        held_v0_wins >= 2
        and held_fusion_count > 0
        and (not np.isfinite(held_max_validated_gap) or held_max_validated_gap < 0.05)
    )
    return {
        "stable": bool(stable),
        "held_v0_wins": held_v0_wins,
        "held_v0_total": held_v0_total,
        "held_validated_wins": held_validated_wins,
        "held_validated_total": held_validated_total,
        "all_v0_wins": all_v0_wins,
        "all_v0_total": all_v0_total,
        "all_validated_wins": all_validated_wins,
        "all_validated_total": all_validated_total,
        "held_fusion_count": held_fusion_count,
        "held_disable_false_negative_rate": held_disable_false_negative,
        "held_max_validated_gap": held_max_validated_gap,
        "tuning_avg_gap_vs_no": v01.mean_or_nan(row.get("gap_vs_no_coordination") for row in tuning),
        "held_avg_gap_vs_no": v01.mean_or_nan(row.get("gap_vs_no_coordination") for row in held_out),
    }


def report_summary_table(summary_rows):
    lines = [
        "| problem | method | best_mean | gap_vs_no | gap_vs_eq8 | gap_vs_validated | gap_vs_v0 | Fusion | Freeze | Disable | fusion_accept |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = {problem: index for index, problem in enumerate(DEFAULT_PROBLEMS)}
    rows = sorted(summary_rows, key=lambda row: (order.get(row["problem"], 99), row["method"]))
    for row in rows:
        lines.append(
            "| {problem} | {method} | {best} | {gap_no} | {gap_eq8} | {gap_val} | {gap_v0} | {fusion} | {freeze} | {disable} | {accept:.3f} |".format(
                problem=row["problem"],
                method=row["method"],
                best=v01.format_metric(row.get("best_mean")),
                gap_no=v01.format_percent(row.get("gap_vs_no_coordination")),
                gap_eq8=v01.format_percent(row.get("gap_vs_eq8_correct")),
                gap_val=v01.format_percent(row.get("gap_vs_validated")),
                gap_v0=v01.format_percent(row.get("gap_vs_arac_lite_v0")),
                fusion=int(row.get("fusion_count", 0) or 0),
                freeze=int(row.get("freeze_count", 0) or 0),
                disable=int(row.get("disable_count", 0) or 0),
                accept=v01.to_float(row.get("fusion_validation_accept_rate")),
            )
        )
    return "\n".join(lines)


def report_action_table(action_rows):
    lines = [
        "| problem | method | action | count | fusion_accept | disable_fn | delta_mean | positive_delta | regret |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    focus_methods = {"arac-lite-v0", "arac-lite-v0.1-disable-fast", "validated-selective-conflict"}
    rows = [row for row in action_rows if row["method"] in focus_methods]
    for row in sorted(rows, key=lambda item: (item["problem"], item["method"], item["action_candidate"])):
        lines.append(
            "| {problem} | {method} | {action} | {count} | {accept:.3f} | {false_neg:.3f} | {delta} | {positive:.3f} | {regret} |".format(
                problem=row["problem"],
                method=row["method"],
                action=row["action_candidate"],
                count=int(row.get("action_count", 0) or 0),
                accept=v01.to_float(row.get("fusion_accept_rate")),
                false_neg=v01.to_float(row.get("disable_false_negative_rate")),
                delta=v01.format_metric(row.get("action_delta_mean")),
                positive=v01.to_float(row.get("action_delta_positive_rate")),
                regret=v01.format_metric(row.get("action_regret_vs_oracle")),
            )
        )
    return "\n".join(lines)


def write_report(summary_rows, run_rows, relation_rows, action_rows, args):
    judgment = generalization_judgment(summary_rows, action_rows)
    stable_line = "稳定" if judgment["stable"] else "不稳定或证据不足"
    report = [
        "# ARAC-lite V0.2 固定规则泛化验证",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Frozen candidate: `arac-lite-v0.1-disable-fast`，未继续调 Disable / Fusion / Freeze / owner_soft / selector / omega / group order。",
        "",
        "## Quick Read",
        "",
        f"- Total runs: {len(run_rows)}",
        f"- OK runs: {sum(1 for row in run_rows if str(row.get('status')) == 'ok')}/{len(run_rows)}",
        f"- Relation-action audit rows: {len(relation_rows)}",
        f"- Action audit rows: {len(action_rows)}",
        f"- Held-out judgment: `{stable_line}`。",
        f"- Held-out E4/S4/S6 vs V0: {judgment['held_v0_wins']}/{judgment['held_v0_total']} non-worse; vs validated: {judgment['held_validated_wins']}/{judgment['held_validated_total']} non-worse。",
        f"- All six vs V0: {judgment['all_v0_wins']}/{judgment['all_v0_total']} non-worse; vs validated: {judgment['all_validated_wins']}/{judgment['all_validated_total']} non-worse。",
        f"- Held-out Fusion count: {judgment['held_fusion_count']}; held-out Disable false-negative rate: {v01.format_metric(judgment['held_disable_false_negative_rate'])}。",
        f"- Tuning-set avg gap_vs_no: {v01.format_percent(judgment['tuning_avg_gap_vs_no'])}; held-out avg gap_vs_no: {v01.format_percent(judgment['held_avg_gap_vs_no'])}。",
        "",
        "## Performance And Action Distribution",
        "",
        report_summary_table(summary_rows),
        "",
        "## Action Audit",
        "",
        report_action_table(action_rows),
        "",
        "## Interpretation",
        "",
        "- 本轮只回答 fixed `disable-fast` 是否泛化，不再进行阈值搜索。",
        "- `Fusion count` 和 `fusion_accept_rate` 用于判断方法是否仍是 relation-to-action 映射，而不是退化成全关协调。",
        "- `Disable false-negative rate` 用于观察 E6/S6 等协同问题是否被过度防御误杀。",
        "- `gap_vs_no_coordination`、`gap_vs_validated`、`gap_vs_arac_lite_v0` 是主要 held-out 判据。",
        "",
        "## Artifacts",
        "",
        f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
        f"- summary: `{SUMMARY_PATH.as_posix()}`",
        f"- relation-action audit: `{RELATION_AUDIT_PATH.as_posix()}`",
        f"- action audit: `{ACTION_AUDIT_PATH.as_posix()}`",
        f"- report: `{REPORT_PATH.as_posix()}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    run_rows, relation_rows, summary_rows = run_sweep(args)
    action_rows = v01.build_action_attribution_rows(relation_rows)

    write_csv(RUN_DETAILS_PATH, v01.RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(RELATION_AUDIT_PATH, v01.RELATION_AUDIT_FIELDNAMES, relation_rows)
    write_csv(ACTION_AUDIT_PATH, v01.ATTRIBUTION_FIELDNAMES, action_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_report(summary_rows, run_rows, relation_rows, action_rows, args)

    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"relation-action audit rows -> {len(relation_rows)}")
    print(f"action audit rows -> {len(action_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
