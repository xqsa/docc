import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "hcc_fidelity_patch_v2_omega_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v2_omega_run_details.csv"
STATE_AUDIT_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v2_omega_state_audit.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v2_omega_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v2_omega_report.md"

DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
METHOD_CONFIGS = [
    ("old-rddsm-order-original", CONFIG_ROOT / "adjacent-original.json"),
    ("old-rddsm-order-eq8-correct", CONFIG_ROOT / "rddsm-eq8-correct.json"),
    ("rddsm-exec-aob-coord-original", CONFIG_ROOT / "rddsm-exec-aob-coord-original.json"),
    ("rddsm-exec-aob-coord-eq8-correct", CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct.json"),
    (
        "rddsm-exec-aob-coord-eq8-correct-persistent-mean",
        CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct-persistent-mean.json",
    ),
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
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
    "execution_order_mode",
    "coordination_order_mode",
    "coordination_mode",
    "blend_strategy",
    "optimizer_state_mode",
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
    "omega_sync_count",
    "mean_coord_mismatch_max",
    "mean_coord_mismatch_mean",
    "mean_coord_mismatch_after_sync_max",
    "mean_coord_mismatch_after_sync_mean",
    "optimizer_reinitialized",
    "status",
]
STATE_AUDIT_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "cycle_id",
    "pass_id",
    "scheduled_position",
    "group_id",
    "source_group_id",
    "event",
    "var_id",
    "mean_init_source",
    "mean_before_coord_overlap",
    "mean_after_coord_overlap",
    "best_before_coord_overlap",
    "best_after_coord_overlap",
    "mean_coord_mismatch",
    "mean_coord_mismatch_after_sync",
    "optimizer_reinitialized",
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
    "gap_vs_old_best",
    "gap_vs_no_coord_best",
    "gap_vs_ephemeral_aob_eq8_best",
    "execution_visibility_mean",
    "coordination_visibility_mean",
    "active_overlap_count_mean",
    "cc_pass_count_mean",
    "omega_sync_count_mean",
    "mean_coord_mismatch_max_mean",
    "mean_coord_mismatch_after_sync_max_mean",
    "positive_delta_rate_mean",
    "early_delta_mean",
    "early_positive_delta_rate_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HCC-Fidelity V2 omega-state artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
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


def load_config(config_path):
    return hcc_es.load_info_aware_nda_config(config_path, enable=False).normalized()


def normalize_state_rows(rows, problem_code, seed, tfes, method_name):
    normalized_rows = []
    for row in rows or []:
        normalized_rows.append(
            {
                "problem": str(row.get("problem") or problem_code).upper(),
                "seed": int(row.get("seed") or seed),
                "tfes": int(row.get("tfes") or tfes),
                "method": str(method_name),
                "cycle_id": row.get("cycle_id"),
                "pass_id": row.get("pass_id"),
                "scheduled_position": row.get("scheduled_position"),
                "group_id": row.get("group_id"),
                "source_group_id": row.get("source_group_id"),
                "event": row.get("event"),
                "var_id": row.get("var_id"),
                "mean_init_source": row.get("mean_init_source"),
                "mean_before_coord_overlap": row.get("mean_before_coord_overlap"),
                "mean_after_coord_overlap": row.get("mean_after_coord_overlap"),
                "best_before_coord_overlap": row.get("best_before_coord_overlap"),
                "best_after_coord_overlap": row.get("best_after_coord_overlap"),
                "mean_coord_mismatch": row.get("mean_coord_mismatch"),
                "mean_coord_mismatch_after_sync": row.get("mean_coord_mismatch_after_sync"),
                "optimizer_reinitialized": row.get("optimizer_reinitialized"),
            }
        )
    return normalized_rows


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
        overlap_summary = dict(metadata.get("overlap_blend_summary", {}))
        group_delta_summary = dict(metadata.get("group_delta_summary", {}))
        execution_audit = dict(metadata.get("execution_order_audit", {}))
        coordination_audit = dict(metadata.get("coordination_order_audit", {}))
        optimizer_summary = dict(metadata.get("optimizer_state_summary", {}))
        state_rows = normalize_state_rows(
            metadata.get("optimizer_state_rows", []),
            problem_code,
            seed,
            tfes,
            method_name,
        )
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": method_name,
                "execution_order_mode": str(metadata.get("group_order_mode", config.group_order_mode)),
                "coordination_order_mode": str(metadata.get("coordination_order_mode", config.coordination_order_mode)),
                "coordination_mode": str(config.shared_variable_coordination_mode),
                "blend_strategy": str(metadata.get("blend_strategy", config.overlap_blend_mode)),
                "optimizer_state_mode": str(metadata.get("optimizer_state_mode", config.optimizer_state_mode)),
                "best_fitness": to_float(detail.get("best_fitness")),
                "final_fitness": to_float(detail.get("final_fitness")),
                "fe_used": int(to_float(detail.get("fe_used"))),
                "runtime": to_float(detail.get("runtime")),
                "overlap_blend_count": int(overlap_summary.get("total_count", 0) or 0),
                "active_overlap_count": int(overlap_summary.get("active_overlap_count", 0) or 0),
                "negative_delta_blend_ratio": to_float(overlap_summary.get("negative_delta_blend_ratio")),
                "group_delta_mean": to_float(group_delta_summary.get("delta_mean")),
                "group_delta_std": to_float(group_delta_summary.get("delta_std")),
                "positive_delta_rate": to_float(group_delta_summary.get("positive_delta_rate")),
                "early_delta_mean": to_float(group_delta_summary.get("early_delta_mean")),
                "early_positive_delta_rate": to_float(group_delta_summary.get("early_positive_delta_rate")),
                "coordination_visibility": to_float(coordination_audit.get("ordered_adjacent_overlap_ratio")),
                "execution_visibility": to_float(execution_audit.get("ordered_adjacent_overlap_ratio")),
                "cc_pass_count": int(metadata.get("cc_pass_count", 0) or 0),
                "mean_init_count": int(optimizer_summary.get("mean_init_count", 0) or 0),
                "omega_sync_count": int(optimizer_summary.get("omega_sync_count", 0) or 0),
                "mean_coord_mismatch_max": to_float(optimizer_summary.get("mean_coord_mismatch_max")),
                "mean_coord_mismatch_mean": to_float(optimizer_summary.get("mean_coord_mismatch_mean")),
                "mean_coord_mismatch_after_sync_max": to_float(optimizer_summary.get("mean_coord_mismatch_after_sync_max")),
                "mean_coord_mismatch_after_sync_mean": to_float(optimizer_summary.get("mean_coord_mismatch_after_sync_mean")),
                "optimizer_reinitialized": bool(optimizer_summary.get("optimizer_reinitialized", True)),
                "status": str(detail.get("status", "ok")),
            },
            state_rows,
        )
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": method_name,
                "execution_order_mode": str(config.group_order_mode),
                "coordination_order_mode": str(config.coordination_order_mode),
                "coordination_mode": str(config.shared_variable_coordination_mode),
                "blend_strategy": str(config.overlap_blend_mode),
                "optimizer_state_mode": str(config.optimizer_state_mode),
                "best_fitness": float("nan"),
                "final_fitness": float("nan"),
                "fe_used": 0,
                "runtime": float("nan"),
                "overlap_blend_count": 0,
                "active_overlap_count": 0,
                "negative_delta_blend_ratio": float("nan"),
                "group_delta_mean": float("nan"),
                "group_delta_std": float("nan"),
                "positive_delta_rate": float("nan"),
                "early_delta_mean": float("nan"),
                "early_positive_delta_rate": float("nan"),
                "coordination_visibility": float("nan"),
                "execution_visibility": float("nan"),
                "cc_pass_count": 0,
                "mean_init_count": 0,
                "omega_sync_count": 0,
                "mean_coord_mismatch_max": float("nan"),
                "mean_coord_mismatch_mean": float("nan"),
                "mean_coord_mismatch_after_sync_max": float("nan"),
                "mean_coord_mismatch_after_sync_mean": float("nan"),
                "optimizer_reinitialized": True,
                "status": f"error: {exc}",
            },
            [],
        )


def summarize_rows(run_rows):
    grouped = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        grouped.setdefault(key, []).append(row)

    old_best = {}
    no_coord_best = {}
    ephemeral_aob_eq8_best = {}
    for (problem, tfes, method), rows in grouped.items():
        best_mean = mean_or_nan(row.get("best_fitness") for row in rows)
        if method == "old-rddsm-order-original":
            old_best[(problem, tfes)] = best_mean
        if method == "no-coordination":
            no_coord_best[(problem, tfes)] = best_mean
        if method == "rddsm-exec-aob-coord-eq8-correct":
            ephemeral_aob_eq8_best[(problem, tfes)] = best_mean

    summary = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_mean = mean_or_nan(row.get("best_fitness") for row in rows)
        final_mean = mean_or_nan(row.get("final_fitness") for row in rows)
        old_value = old_best.get((problem, tfes), float("nan"))
        no_coord_value = no_coord_best.get((problem, tfes), float("nan"))
        ephemeral_eq8_value = ephemeral_aob_eq8_best.get((problem, tfes), float("nan"))
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
                "gap_vs_old_best": float((best_mean - old_value) / max(abs(old_value), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(old_value)
                else float("nan"),
                "gap_vs_no_coord_best": float((best_mean - no_coord_value) / max(abs(no_coord_value), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(no_coord_value)
                else float("nan"),
                "gap_vs_ephemeral_aob_eq8_best": float((best_mean - ephemeral_eq8_value) / max(abs(ephemeral_eq8_value), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(ephemeral_eq8_value)
                else float("nan"),
                "execution_visibility_mean": mean_or_nan(row.get("execution_visibility") for row in rows),
                "coordination_visibility_mean": mean_or_nan(row.get("coordination_visibility") for row in rows),
                "active_overlap_count_mean": mean_or_nan(row.get("active_overlap_count") for row in rows),
                "cc_pass_count_mean": mean_or_nan(row.get("cc_pass_count") for row in rows),
                "omega_sync_count_mean": mean_or_nan(row.get("omega_sync_count") for row in rows),
                "mean_coord_mismatch_max_mean": mean_or_nan(row.get("mean_coord_mismatch_max") for row in rows),
                "mean_coord_mismatch_after_sync_max_mean": mean_or_nan(row.get("mean_coord_mismatch_after_sync_max") for row in rows),
                "positive_delta_rate_mean": mean_or_nan(row.get("positive_delta_rate") for row in rows),
                "early_delta_mean": mean_or_nan(row.get("early_delta_mean") for row in rows),
                "early_positive_delta_rate_mean": mean_or_nan(row.get("early_positive_delta_rate") for row in rows),
            }
        )
    return summary


def report_table(summary_rows, tfes):
    rows = [row for row in summary_rows if int(row["tfes"]) == int(tfes)]
    lines = [
        f"## TFEs = {int(tfes)}",
        "",
        "| problem | method | best_mean | final_mean | gap_vs_old | gap_vs_no_coord | gap_vs_ephemeral_eq8 | cc_pass | coord_vis | omega_sync | mismatch_after_sync |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {problem} | {method} | {best} | {final} | {gap_old} | {gap_no} | {gap_ephem} | {cc_pass:.1f} | {coord_vis:.3f} | {omega_sync:.1f} | {mismatch} |".format(
                problem=row["problem"],
                method=row["method"],
                best=format_metric(row.get("best_mean")),
                final=format_metric(row.get("final_mean")),
                gap_old=format_percent(row.get("gap_vs_old_best")),
                gap_no=format_percent(row.get("gap_vs_no_coord_best")),
                gap_ephem=format_percent(row.get("gap_vs_ephemeral_aob_eq8_best")),
                cc_pass=to_float(row.get("cc_pass_count_mean")),
                coord_vis=to_float(row.get("coordination_visibility_mean")),
                omega_sync=to_float(row.get("omega_sync_count_mean")),
                mismatch=format_metric(row.get("mean_coord_mismatch_after_sync_max_mean")),
            )
        )
    return "\n".join(lines)


def write_report(summary_rows, run_rows, state_rows, args):
    persistent_rows = [
        row for row in summary_rows
        if row["method"] == "rddsm-exec-aob-coord-eq8-correct-persistent-mean"
    ]
    persistent_better_count = sum(
        np.isfinite(to_float(row.get("gap_vs_ephemeral_aob_eq8_best")))
        and to_float(row.get("gap_vs_ephemeral_aob_eq8_best")) < 0.0
        for row in persistent_rows
    )
    report = [
        "# HCC-Fidelity V2: omega_i / optimizer-state synchronization",
        "",
        "- Executor: Codex",
        "- Date: 2026-05-20",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        "",
        "## Audit Focus",
        "",
        "This run tests whether adjacent overlap coordination must also update the per-group optimizer mean cache. `persistent_mean` keeps CMA-ES reinitialized per group, but replaces the next mean source with `omega_cache[group_id]` and synchronizes coordinated overlap variables into every cached group that contains them.",
        "",
        "## Quick Read",
        "",
        f"- Total runs: {len(run_rows)}",
        f"- Optimizer-state audit rows: {len(state_rows)}",
        f"- Persistent-mean wins vs ephemeral AOB-Eq8 on best_mean: {persistent_better_count}/{len(persistent_rows)} problem/TFEs cells",
        "",
        "## Performance",
    ]
    for tfes in args.tfes:
        report.extend(["", report_table(summary_rows, tfes)])
    report.extend(
        [
            "",
        "## Interpretation Guide",
        "",
        "- When `cc_pass` is 1.0, a persistent mean cache can be synchronized but cannot influence a later optimizer initialization inside the same run. In that case identical ephemeral/persistent scores mean the protocol gave state sync no downstream opportunity.",
        "- If persistent_mean improves the Eq8 restored-coordination variant, the failed Eq8 result was likely partly caused by optimizer-state inconsistency.",
            "- If persistent_mean still loses, coordination action selection is more suspect than state absorption, and ARAC-lite/action selection becomes the next cleaner direction.",
            "- If R6 still favors no-coordination, that supports a Disable/Freeze mapping for noisy or conflict-heavy relations.",
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
        config = load_config(config_path)
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    run_row, case_state_rows = run_one_case(method_name, config, problem_code, int(seed), int(tfes))
                    run_rows.append(run_row)
                    state_rows.extend(case_state_rows)
                    print(f"{problem_code} {method_name} seed={seed} tfes={tfes}: {run_row['status']}")
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(STATE_AUDIT_PATH, STATE_AUDIT_FIELDNAMES, state_rows)
    summary_rows = summarize_rows(run_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_report(summary_rows, run_rows, state_rows, args)
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"state audit rows -> {len(state_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
