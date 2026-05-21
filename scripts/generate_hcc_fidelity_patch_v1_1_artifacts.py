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
RUNS_ROOT = ARTIFACTS_ROOT / "hcc_fidelity_patch_v1_1_runs"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v1_1_run_details.csv"
VISIBILITY_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v1_1_visibility.csv"
SUMMARY_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v1_1_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "hcc_fidelity_patch_v1_1_report.md"

DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [1000, 5000, 10000]
METHOD_CONFIGS = [
    ("old-rddsm-order-original", CONFIG_ROOT / "adjacent-original.json"),
    ("old-rddsm-order-eq8-correct", CONFIG_ROOT / "rddsm-eq8-correct.json"),
    ("aob-full-order-original", CONFIG_ROOT / "topology-original.json"),
    ("aob-full-order-eq8-correct", CONFIG_ROOT / "topology-eq8-correct.json"),
    ("rddsm-exec-aob-coord-original", CONFIG_ROOT / "rddsm-exec-aob-coord-original.json"),
    ("rddsm-exec-aob-coord-eq8-correct", CONFIG_ROOT / "rddsm-exec-aob-coord-eq8-correct.json"),
    ("aob-exec-rddsm-coord-original", CONFIG_ROOT / "aob-exec-rddsm-coord-original.json"),
    ("aob-exec-rddsm-coord-eq8-correct", CONFIG_ROOT / "aob-exec-rddsm-coord-eq8-correct.json"),
]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)


VISIBILITY_FIELDNAMES = [
    "problem",
    "method",
    "execution_order_mode",
    "coordination_order_mode",
    "blend_strategy",
    "true_overlap_var_count",
    "execution_adjacent_overlap_count",
    "execution_adjacent_overlap_ratio",
    "coordination_adjacent_overlap_count",
    "coordination_adjacent_overlap_ratio",
    "reference_adjacent_overlap_count",
    "reference_adjacent_overlap_ratio",
    "coordination_visibility_gain_vs_execution",
    "execution_order_changed",
    "coordination_order_changed",
]
RUN_DETAIL_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "execution_order_mode",
    "coordination_order_mode",
    "blend_strategy",
    "coordination_mode",
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
    "gap_vs_old_best",
    "gap_vs_aob_full_best",
    "execution_visibility_mean",
    "coordination_visibility_mean",
    "overlap_blend_count_mean",
    "active_overlap_count_mean",
    "negative_delta_blend_ratio_mean",
    "group_delta_mean",
    "group_delta_std_mean",
    "positive_delta_rate_mean",
    "early_delta_mean",
    "early_positive_delta_rate_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HCC-fidelity patch V1.1 order-role decoupling artifacts.")
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


def build_visibility_rows(method_configs, problems):
    rows = []
    for method_name, config_path in method_configs:
        config = load_config(config_path)
        for problem_code in problems:
            inputs = hcc_es.build_hcc_es_inputs(problem_code)
            plan = hcc_es.build_group_order_plan(
                inputs["grouping_result"],
                problem_code=problem_code,
                execution_order_mode=config.group_order_mode,
                coordination_order_mode=config.coordination_order_mode,
            )
            execution_audit = plan["execution_order_audit"]
            coordination_audit = plan["coordination_order_audit"]
            rows.append(
                {
                    "problem": str(problem_code).upper(),
                    "method": method_name,
                    "execution_order_mode": config.group_order_mode,
                    "coordination_order_mode": config.coordination_order_mode,
                    "blend_strategy": config.overlap_blend_mode,
                    "true_overlap_var_count": coordination_audit["true_overlap_var_count"],
                    "execution_adjacent_overlap_count": execution_audit["ordered_adjacent_overlap_count"],
                    "execution_adjacent_overlap_ratio": execution_audit["ordered_adjacent_overlap_ratio"],
                    "coordination_adjacent_overlap_count": coordination_audit["ordered_adjacent_overlap_count"],
                    "coordination_adjacent_overlap_ratio": coordination_audit["ordered_adjacent_overlap_ratio"],
                    "reference_adjacent_overlap_count": coordination_audit["reference_adjacent_overlap_count"],
                    "reference_adjacent_overlap_ratio": coordination_audit["reference_adjacent_overlap_ratio"],
                    "coordination_visibility_gain_vs_execution": float(
                        coordination_audit["ordered_adjacent_overlap_ratio"]
                        - execution_audit["ordered_adjacent_overlap_ratio"]
                    ),
                    "execution_order_changed": execution_audit["group_order_changed"],
                    "coordination_order_changed": coordination_audit["group_order_changed"],
                }
            )
    return rows


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
        return {
            "problem": str(problem_code).upper(),
            "seed": int(seed),
            "tfes": int(tfes),
            "method": method_name,
            "execution_order_mode": str(metadata.get("group_order_mode", config.group_order_mode)),
            "coordination_order_mode": str(metadata.get("coordination_order_mode", config.coordination_order_mode)),
            "blend_strategy": str(metadata.get("blend_strategy", config.overlap_blend_mode)),
            "coordination_mode": str(config.shared_variable_coordination_mode),
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
            "status": str(detail.get("status", "ok")),
        }
    except Exception as exc:
        return {
            "problem": str(problem_code).upper(),
            "seed": int(seed),
            "tfes": int(tfes),
            "method": method_name,
            "execution_order_mode": str(config.group_order_mode),
            "coordination_order_mode": str(config.coordination_order_mode),
            "blend_strategy": str(config.overlap_blend_mode),
            "coordination_mode": str(config.shared_variable_coordination_mode),
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
            "status": f"error: {exc}",
        }


def summarize_rows(run_rows):
    grouped = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        grouped.setdefault(key, []).append(row)

    old_best_by_problem_tfes = {}
    aob_best_by_problem_tfes = {}
    for (problem, tfes, method), rows in grouped.items():
        best_mean = mean_or_nan(row.get("best_fitness") for row in rows)
        if method == "old-rddsm-order-original":
            old_best_by_problem_tfes[(problem, tfes)] = best_mean
        if method == "aob-full-order-original":
            aob_best_by_problem_tfes[(problem, tfes)] = best_mean

    summary = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_mean = mean_or_nan(row.get("best_fitness") for row in rows)
        final_mean = mean_or_nan(row.get("final_fitness") for row in rows)
        old_best = old_best_by_problem_tfes.get((problem, tfes), float("nan"))
        aob_best = aob_best_by_problem_tfes.get((problem, tfes), float("nan"))
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
                "gap_vs_old_best": float((best_mean - old_best) / max(abs(old_best), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(old_best)
                else float("nan"),
                "gap_vs_aob_full_best": float((best_mean - aob_best) / max(abs(aob_best), 1e-300))
                if np.isfinite(best_mean) and np.isfinite(aob_best)
                else float("nan"),
                "execution_visibility_mean": mean_or_nan(row.get("execution_visibility") for row in rows),
                "coordination_visibility_mean": mean_or_nan(row.get("coordination_visibility") for row in rows),
                "overlap_blend_count_mean": mean_or_nan(row.get("overlap_blend_count") for row in rows),
                "active_overlap_count_mean": mean_or_nan(row.get("active_overlap_count") for row in rows),
                "negative_delta_blend_ratio_mean": mean_or_nan(row.get("negative_delta_blend_ratio") for row in rows),
                "group_delta_mean": mean_or_nan(row.get("group_delta_mean") for row in rows),
                "group_delta_std_mean": mean_or_nan(row.get("group_delta_std") for row in rows),
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
        "| problem | method | exec_vis | coord_vis | best_mean | final_mean | gap_vs_old | gap_vs_aob_full | active_coord | early_delta | pos_delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {problem} | {method} | {exec_vis:.3f} | {coord_vis:.3f} | {best} | {final} | {gap_old} | {gap_aob} | {active:.1f} | {early_delta} | {pos_delta:.3f} |".format(
                problem=row["problem"],
                method=row["method"],
                exec_vis=to_float(row.get("execution_visibility_mean")),
                coord_vis=to_float(row.get("coordination_visibility_mean")),
                best=format_metric(row.get("best_mean")),
                final=format_metric(row.get("final_mean")),
                gap_old=format_percent(row.get("gap_vs_old_best")),
                gap_aob=format_percent(row.get("gap_vs_aob_full_best")),
                active=to_float(row.get("active_overlap_count_mean")),
                early_delta=format_metric(row.get("early_delta_mean")),
                pos_delta=to_float(row.get("positive_delta_rate_mean")),
            )
        )
    return "\n".join(lines)


def write_report(summary_rows, visibility_rows, args):
    report = [
        "# HCC-Fidelity Patch V1.1: Group Order Role Decoupling",
        "",
        "- Executor: Codex",
        "- Date: 2026-05-20",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        "",
        "## Visibility Check",
        "",
        "| problem | method | execution visibility | coordination visibility | gain |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in visibility_rows:
        report.append(
            "| {problem} | {method} | {exec_vis:.3f} | {coord_vis:.3f} | {gain:+.3f} |".format(
                problem=row["problem"],
                method=row["method"],
                exec_vis=to_float(row.get("execution_adjacent_overlap_ratio")),
                coord_vis=to_float(row.get("coordination_adjacent_overlap_ratio")),
                gain=to_float(row.get("coordination_visibility_gain_vs_execution")),
            )
        )
    report.extend(["", "## Performance"])
    for tfes in args.tfes:
        report.extend(["", report_table(summary_rows, tfes)])
    report.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- visibility: `{VISIBILITY_PATH.as_posix()}`",
            f"- summary: `{SUMMARY_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    problems = [hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    method_configs = [(name, path) for name, path in METHOD_CONFIGS]
    visibility_rows = build_visibility_rows(method_configs, problems)
    write_csv(VISIBILITY_PATH, VISIBILITY_FIELDNAMES, visibility_rows)

    run_rows = []
    for method_name, config_path in method_configs:
        config = load_config(config_path)
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    row = run_one_case(method_name, config, problem_code, int(seed), int(tfes))
                    run_rows.append(row)
                    print(f"{problem_code} {method_name} seed={seed} tfes={tfes}: {row['status']}")
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    summary_rows = summarize_rows(run_rows)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDNAMES, summary_rows)
    write_report(summary_rows, visibility_rows, args)
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"completed {ok_count}/{len(run_rows)} runs")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
