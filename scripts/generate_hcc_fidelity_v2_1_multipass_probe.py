import argparse
import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "hcc_fidelity_v2_1_multipass_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "hcc_fidelity_v2_1_multipass_run_details.csv"
STATE_AUDIT_PATH = ARTIFACTS_ROOT / "hcc_fidelity_v2_1_multipass_state_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "hcc_fidelity_v2_1_multipass_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "hcc_fidelity_v2_1_multipass_report.md"

DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20
METHOD_CONFIGS = [
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("rddsm-exec-aob-coord-eq8-correct", CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct.json"),
    (
        "rddsm-exec-aob-coord-eq8-correct-persistent-mean",
        CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct-persistent-mean.json",
    ),
]

spec = importlib.util.spec_from_file_location(
    "v2_omega_artifacts",
    SCRIPT_ROOT / "generate_hcc_fidelity_patch_v2_omega_artifacts.py",
)
v2 = importlib.util.module_from_spec(spec)
sys.modules["v2_omega_artifacts"] = v2
spec.loader.exec_module(v2)
v2.RUNS_ROOT = RUNS_ROOT
hcc_es = v2.hcc_es


RUN_DETAIL_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "execution_order_mode",
    "coordination_order_mode",
    "coordination_mode",
    "blend_strategy",
    "optimizer_state_mode",
    "cc_pass_group_fes",
    "best_fitness",
    "final_fitness",
    "fe_used",
    "runtime",
    "overlap_blend_count",
    "active_overlap_count",
    "negative_delta_blend_ratio",
    "group_delta_mean",
    "group_delta_std",
    "positive_delta_rate",
    "early_delta_mean",
    "early_positive_delta_rate",
    "coordination_visibility",
    "execution_visibility",
    "cc_pass_count",
    "mean_init_count",
    "persistent_mean_init_rows",
    "post_sync_init_count",
    "omega_sync_count",
    "mean_init_shift_after_sync",
    "mean_init_shift_after_sync_max",
    "post_sync_init_mismatch_mean",
    "post_sync_init_mismatch_max",
    "mean_coord_mismatch_max",
    "mean_coord_mismatch_mean",
    "mean_coord_mismatch_after_sync_max",
    "mean_coord_mismatch_after_sync_mean",
    "optimizer_reinitialized",
    "status",
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
    "gap_vs_no_coord_best",
    "gap_vs_ephemeral_aob_eq8_best",
    "cc_pass_count_mean",
    "post_sync_init_count_mean",
    "persistent_mean_init_rows_mean",
    "omega_sync_count_mean",
    "mean_init_shift_after_sync_mean",
    "mean_init_shift_after_sync_max_mean",
    "post_sync_init_mismatch_mean",
    "post_sync_init_mismatch_max_mean",
    "mean_coord_mismatch_after_sync_max_mean",
    "coordination_visibility_mean",
    "execution_visibility_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HCC-Fidelity V2.1 multi-pass probe artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    return parser.parse_args()


def to_float(value):
    return v2.to_float(value)


def mean_or_nan(values):
    return v2.mean_or_nan(values)


def std_or_nan(values):
    return v2.std_or_nan(values)


def format_metric(value):
    return v2.format_metric(value)


def format_percent(value):
    return v2.format_percent(value)


def row_int(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def compute_post_sync_audit(state_rows):
    init_rows = [row for row in state_rows if str(row.get("event", "")) == "mean_init"]
    sync_rows = [row for row in state_rows if str(row.get("event", "")) == "coordination_sync"]
    persistent_init_rows = [
        row for row in init_rows
        if str(row.get("mean_init_source", "")) == "omega_cache"
    ]
    post_sync_init_rows = []
    post_sync_init_mismatches = []
    for init_row in persistent_init_rows:
        init_pass = row_int(init_row, "pass_id")
        source_group_id = row_int(init_row, "source_group_id")
        if init_pass is None or source_group_id is None:
            continue
        has_previous_sync = any(
            row_int(sync_row, "source_group_id") == source_group_id
            and row_int(sync_row, "pass_id") is not None
            and row_int(sync_row, "pass_id") < init_pass
            for sync_row in sync_rows
        )
        if has_previous_sync:
            post_sync_init_rows.append(init_row)
            mismatch = to_float(init_row.get("mean_coord_mismatch"))
            if np.isfinite(mismatch):
                post_sync_init_mismatches.append(mismatch)
    consumed_sync_shifts = []
    for sync_row in sync_rows:
        sync_pass = row_int(sync_row, "pass_id")
        source_group_id = row_int(sync_row, "source_group_id")
        if sync_pass is None or source_group_id is None:
            continue
        has_later_init = any(
            row_int(init_row, "source_group_id") == source_group_id
            and row_int(init_row, "pass_id") is not None
            and row_int(init_row, "pass_id") > sync_pass
            for init_row in persistent_init_rows
        )
        if has_later_init:
            shift = to_float(sync_row.get("mean_coord_mismatch"))
            if np.isfinite(shift):
                consumed_sync_shifts.append(shift)
    return {
        "persistent_mean_init_rows": int(len(persistent_init_rows)),
        "post_sync_init_count": int(len(post_sync_init_rows)),
        "mean_init_shift_after_sync": float(np.mean(consumed_sync_shifts)) if consumed_sync_shifts else 0.0,
        "mean_init_shift_after_sync_max": float(np.max(consumed_sync_shifts)) if consumed_sync_shifts else 0.0,
        "post_sync_init_mismatch_mean": float(np.mean(post_sync_init_mismatches)) if post_sync_init_mismatches else 0.0,
        "post_sync_init_mismatch_max": float(np.max(post_sync_init_mismatches)) if post_sync_init_mismatches else 0.0,
    }


def load_method_config(config_path, cc_pass_group_fes):
    config = hcc_es.load_info_aware_nda_config(config_path, enable=False)
    return replace(config, cc_pass_group_fes=max(1, int(cc_pass_group_fes))).normalized()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def summarize_rows(run_rows):
    grouped = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        grouped.setdefault(key, []).append(row)

    no_coord_best = {}
    ephemeral_best = {}
    for (problem, tfes, method), rows in grouped.items():
        best_mean = mean_or_nan(row.get("best_fitness") for row in rows)
        if method == "no-coordination":
            no_coord_best[(problem, tfes)] = best_mean
        if method == "rddsm-exec-aob-coord-eq8-correct":
            ephemeral_best[(problem, tfes)] = best_mean

    summary = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_mean = mean_or_nan(row.get("best_fitness") for row in rows)
        final_mean = mean_or_nan(row.get("final_fitness") for row in rows)
        no_coord_value = no_coord_best.get((problem, tfes), float("nan"))
        ephemeral_value = ephemeral_best.get((problem, tfes), float("nan"))
        summary.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "runs": int(len(rows)),
                "best_mean": best_mean,
                "best_std": std_or_nan(row.get("best_fitness") for row in rows),
                "final_mean": final_mean,
                "final_std": std_or_nan(row.get("final_fitness") for row in rows),
                "gap_vs_no_coord_best": float((best_mean - no_coord_value) / max(abs(no_coord_value), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(no_coord_value)
                else float("nan"),
                "gap_vs_ephemeral_aob_eq8_best": float((best_mean - ephemeral_value) / max(abs(ephemeral_value), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(ephemeral_value)
                else float("nan"),
                "cc_pass_count_mean": mean_or_nan(row.get("cc_pass_count") for row in rows),
                "post_sync_init_count_mean": mean_or_nan(row.get("post_sync_init_count") for row in rows),
                "persistent_mean_init_rows_mean": mean_or_nan(row.get("persistent_mean_init_rows") for row in rows),
                "omega_sync_count_mean": mean_or_nan(row.get("omega_sync_count") for row in rows),
                "mean_init_shift_after_sync_mean": mean_or_nan(row.get("mean_init_shift_after_sync") for row in rows),
                "mean_init_shift_after_sync_max_mean": mean_or_nan(row.get("mean_init_shift_after_sync_max") for row in rows),
                "post_sync_init_mismatch_mean": mean_or_nan(row.get("post_sync_init_mismatch_mean") for row in rows),
                "post_sync_init_mismatch_max_mean": mean_or_nan(row.get("post_sync_init_mismatch_max") for row in rows),
                "mean_coord_mismatch_after_sync_max_mean": mean_or_nan(row.get("mean_coord_mismatch_after_sync_max") for row in rows),
                "coordination_visibility_mean": mean_or_nan(row.get("coordination_visibility") for row in rows),
                "execution_visibility_mean": mean_or_nan(row.get("execution_visibility") for row in rows),
            }
        )
    return summary


def report_table(summary_rows, tfes):
    rows = [row for row in summary_rows if int(row["tfes"]) == int(tfes)]
    lines = [
        f"## TFEs = {int(tfes)}",
        "",
        "| problem | method | best_mean | final_mean | gap_vs_no_coord | gap_vs_ephemeral_eq8 | cc_pass | post_sync_init | omega_sync | persistent_init | mean_shift_after_sync | mismatch_after_sync |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {problem} | {method} | {best} | {final} | {gap_no} | {gap_ephem} | {cc_pass:.1f} | {post_sync:.1f} | {omega_sync:.1f} | {persistent_init:.1f} | {shift} | {mismatch} |".format(
                problem=row["problem"],
                method=row["method"],
                best=format_metric(row.get("best_mean")),
                final=format_metric(row.get("final_mean")),
                gap_no=format_percent(row.get("gap_vs_no_coord_best")),
                gap_ephem=format_percent(row.get("gap_vs_ephemeral_aob_eq8_best")),
                cc_pass=to_float(row.get("cc_pass_count_mean")),
                post_sync=to_float(row.get("post_sync_init_count_mean")),
                omega_sync=to_float(row.get("omega_sync_count_mean")),
                persistent_init=to_float(row.get("persistent_mean_init_rows_mean")),
                shift=format_metric(row.get("mean_init_shift_after_sync_mean")),
                mismatch=format_metric(row.get("mean_coord_mismatch_after_sync_max_mean")),
            )
        )
    return "\n".join(lines)


def write_report(summary_rows, run_rows, state_rows, args):
    persistent_rows = [
        row for row in summary_rows
        if row["method"] == "rddsm-exec-aob-coord-eq8-correct-persistent-mean"
    ]
    persistent_better = sum(
        np.isfinite(to_float(row.get("gap_vs_ephemeral_aob_eq8_best")))
        and to_float(row.get("gap_vs_ephemeral_aob_eq8_best")) < 0.0
        for row in persistent_rows
    )
    any_post_sync = any(to_float(row.get("post_sync_init_count_mean")) > 0.0 for row in persistent_rows)
    report = [
        "# HCC-Fidelity V2.1 Controlled Multi-pass Probe",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Scope: no selector, no ARAC-lite, no order tuning, no owner_soft tuning.",
        "",
        "## Quick Read",
        "",
        f"- Total runs: {len(run_rows)}",
        f"- Optimizer-state audit rows: {len(state_rows)}",
        f"- Persistent post-sync init observed: {bool(any_post_sync)}",
        f"- Persistent-mean wins vs ephemeral Eq8 on best_mean: {persistent_better}/{len(persistent_rows)} problem/TFEs cells",
        "",
        "## Performance And State Audit",
    ]
    for tfes in args.tfes:
        report.extend(["", report_table(summary_rows, tfes)])
    report.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `post_sync_init_count > 0` means synchronized `omega_cache` was consumed by a later CMA-ES mean initialization.",
            "- `mean_init_shift_after_sync` summarizes the pre-sync cache/global mismatch of sync rows that were later consumed.",
            "- If persistent_mean improves best/final over ephemeral Eq8, state-consistent coordination is supported.",
            "- If persistent_mean remains tied, optimizer mean sync is not the main bottleneck.",
            "- If R6 worsens vs no-coordination, R6 still supports Disable/Freeze relation actions.",
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- optimizer-state audit: `{STATE_AUDIT_PATH.as_posix()}`",
            f"- summary: `{SUMMARY_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    problems = [hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    run_rows = []
    state_rows = []
    for method_name, config_path in METHOD_CONFIGS:
        config = load_method_config(config_path, args.cc_pass_group_fes)
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    run_row, case_state_rows = v2.run_one_case(
                        method_name,
                        config,
                        problem_code,
                        int(seed),
                        int(tfes),
                    )
                    run_row["cc_pass_group_fes"] = int(args.cc_pass_group_fes)
                    run_row.update(compute_post_sync_audit(case_state_rows))
                    run_rows.append(run_row)
                    state_rows.extend(case_state_rows)
                    print(
                        f"{problem_code} {method_name} seed={seed} tfes={tfes}: "
                        f"{run_row['status']} cc_pass={run_row['cc_pass_count']} "
                        f"post_sync_init={run_row['post_sync_init_count']}"
                    )
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(STATE_AUDIT_PATH, v2.STATE_AUDIT_FIELDNAMES, state_rows)
    summary_rows = summarize_rows(run_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_report(summary_rows, run_rows, state_rows, args)
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"state audit rows -> {len(state_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
