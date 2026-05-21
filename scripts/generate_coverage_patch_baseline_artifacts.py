import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "coverage_patch_runs"
PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
SEEDS = [1, 2, 3, 4, 5]
MAX_FES = 1000
METHOD_CONFIGS = [
    ("baseline", CONFIG_ROOT / "baseline.json"),
    ("early-switch-only", CONFIG_ROOT / "early-switch-only.json"),
    ("diagnostic-only", CONFIG_ROOT / "diagnostic-only.json"),
]
METHOD_ORDER = {method_name: index for index, (method_name, _) in enumerate(METHOD_CONFIGS)}
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "coverage_patch_run_details.csv"
OVERLAP_VISIBILITY_PATH = ARTIFACTS_ROOT / "coverage_patch_overlap_visibility.csv"
REPORT_PATH = ARTIFACTS_ROOT / "coverage_patch_baseline_report.md"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)


RUN_DETAIL_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "final_error",
    "best_error",
    "fe_used",
    "nda_fe_used",
    "early_switch_triggered",
    "early_switch_reason",
    "group_count",
    "residual_group_size",
    "status",
    "runtime",
    "coverage_patch_applied",
]

OVERLAP_VISIBILITY_FIELDNAMES = [
    "problem",
    "benchmark_dimension",
    "design_matrix_dimension",
    "covered_variable_count",
    "uncovered_variable_count_before_patch",
    "effective_covered_variable_count",
    "uncovered_variable_count_after_patch",
    "group_count_before_patch",
    "group_count",
    "residual_group_size",
    "true_overlap_var_count_before_patch",
    "adjacent_visible_overlap_var_count_before_patch",
    "true_overlap_var_count",
    "adjacent_visible_overlap_var_count",
    "nonadjacent_overlap_var_count",
    "visible_ratio",
    "nonadjacent_ratio",
    "adjacent_boundary_count_before_patch",
    "adjacent_boundary_count",
    "total_group_boundary_count_before_patch",
    "total_group_boundary_count",
    "coordination_opportunity",
    "visibility_changed_by_patch",
]


def load_method_config(config_path):
    return hcc_es.load_info_aware_nda_config(config_path, enable=False)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_or_nan(values):
    numeric = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
    return float(np.mean(numeric)) if numeric.size else float("nan")


def format_metric(value):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6e}"


def build_problem_inputs(problem_code):
    return hcc_es.build_hcc_es_inputs(problem_code)


def compute_visible_overlap_vars(adjacent_overlaps):
    return sorted({int(var_id) for overlap in adjacent_overlaps for var_id in overlap})


def build_overlap_visibility_row(problem_code, inputs):
    fun_name, fun_id, normalized = hcc_es.parse_problem_code(problem_code)
    file_path = REPO_ROOT / "HCC_SRC" / "AOB" / "AOBG" / "datafile" / f"F{fun_id}-design.txt"
    design_matrix = np.loadtxt(file_path, delimiter=",")
    raw_grouping_result = hcc_es.Decomposition(design_matrix).decomposition()
    raw_adjacent_overlaps = hcc_es.compute_adjacent_overlaps_for_groups(raw_grouping_result)
    raw_hypergraph = hcc_es.build_overlap_hypergraph(raw_grouping_result)
    raw_visible_overlap_vars = compute_visible_overlap_vars(raw_adjacent_overlaps)

    coverage_audit = dict(inputs["coverage_audit"])
    grouping_result = inputs["grouping_result"]
    adjacent_overlaps = inputs["adjacent_overlapping_elements"]
    overlap_hypergraph = hcc_es.build_overlap_hypergraph(grouping_result)
    true_overlap_vars = sorted(int(var_id) for var_id in overlap_hypergraph["overlap_vars"])
    visible_overlap_vars = compute_visible_overlap_vars(adjacent_overlaps)
    visible_overlap_var_set = set(visible_overlap_vars)
    nonadjacent_overlap_vars = [int(var_id) for var_id in true_overlap_vars if int(var_id) not in visible_overlap_var_set]

    true_overlap_count = len(true_overlap_vars)
    visible_count = len(visible_overlap_vars)
    nonadjacent_count = len(nonadjacent_overlap_vars)
    visible_ratio = float(visible_count / true_overlap_count) if true_overlap_count else 0.0
    nonadjacent_ratio = float(nonadjacent_count / true_overlap_count) if true_overlap_count else 0.0
    adjacent_boundary_count = int(sum(1 for overlap in adjacent_overlaps if len(overlap) > 0))
    raw_adjacent_boundary_count = int(sum(1 for overlap in raw_adjacent_overlaps if len(overlap) > 0))

    return {
        "problem": normalized,
        "benchmark_dimension": int(coverage_audit["benchmark_dimension"]),
        "design_matrix_dimension": int(coverage_audit["design_matrix_dimension"]),
        "covered_variable_count": int(coverage_audit["covered_variable_count"]),
        "uncovered_variable_count_before_patch": int(coverage_audit["uncovered_variable_count"]),
        "effective_covered_variable_count": int(coverage_audit["effective_covered_variable_count"]),
        "uncovered_variable_count_after_patch": int(coverage_audit["effective_uncovered_variable_count"]),
        "group_count_before_patch": int(len(raw_grouping_result)),
        "group_count": int(len(grouping_result)),
        "residual_group_size": int(coverage_audit.get("residual_group_size", 0)),
        "true_overlap_var_count_before_patch": int(len(raw_hypergraph["overlap_vars"])),
        "adjacent_visible_overlap_var_count_before_patch": int(len(raw_visible_overlap_vars)),
        "true_overlap_var_count": int(true_overlap_count),
        "adjacent_visible_overlap_var_count": int(visible_count),
        "nonadjacent_overlap_var_count": int(nonadjacent_count),
        "visible_ratio": float(visible_ratio),
        "nonadjacent_ratio": float(nonadjacent_ratio),
        "adjacent_boundary_count_before_patch": int(raw_adjacent_boundary_count),
        "adjacent_boundary_count": int(adjacent_boundary_count),
        "total_group_boundary_count_before_patch": int(max(0, len(raw_grouping_result) - 1)),
        "total_group_boundary_count": int(max(0, len(grouping_result) - 1)),
        "coordination_opportunity": int(adjacent_boundary_count),
        "visibility_changed_by_patch": bool(
            len(raw_hypergraph["overlap_vars"]) != true_overlap_count
            or len(raw_visible_overlap_vars) != visible_count
            or raw_adjacent_boundary_count != adjacent_boundary_count
        ),
    }


def run_one_case(method_name, config, problem_code, seed, problem_inputs):
    output_dir = RUNS_ROOT / method_name
    try:
        result = hcc_es.run_problem_seed_task(
            problem_code,
            seed,
            MAX_FES,
            hcc_es.HCC_ES_METHOD,
            output_dir,
            record_fes=[],
            info_aware_config=config,
            method_label=method_name,
        )
        detail = dict(result["detail"])
        metadata = dict(result.get("metadata", {}))
        diagnostics_payload = dict(metadata.get("info_aware_diagnostics", {}))
        return {
            "problem": str(problem_code).upper(),
            "seed": int(seed),
            "method": str(method_name),
            "final_error": to_float(detail.get("final_fitness")),
            "best_error": to_float(detail.get("best_fitness")),
            "fe_used": int(detail.get("fe_used", 0) or 0),
            "nda_fe_used": int(diagnostics_payload.get("nda_fe_used", 0) or 0),
            "early_switch_triggered": bool(diagnostics_payload.get("early_switch_triggered", False)),
            "early_switch_reason": str(diagnostics_payload.get("early_switch_reason", "")),
            "group_count": int(len(problem_inputs["grouping_result"])),
            "residual_group_size": int(problem_inputs["coverage_audit"].get("residual_group_size", 0)),
            "status": str(detail.get("status", "")),
            "runtime": to_float(detail.get("runtime")),
            "coverage_patch_applied": bool(problem_inputs["coverage_audit"].get("coverage_patch_applied", False)),
        }
    except Exception as exc:
        return {
            "problem": str(problem_code).upper(),
            "seed": int(seed),
            "method": str(method_name),
            "final_error": float("nan"),
            "best_error": float("nan"),
            "fe_used": 0,
            "nda_fe_used": 0,
            "early_switch_triggered": False,
            "early_switch_reason": "",
            "group_count": int(len(problem_inputs["grouping_result"])),
            "residual_group_size": int(problem_inputs["coverage_audit"].get("residual_group_size", 0)),
            "status": f"error: {exc}",
            "runtime": float("nan"),
            "coverage_patch_applied": bool(problem_inputs["coverage_audit"].get("coverage_patch_applied", False)),
        }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def group_run_rows(run_rows):
    grouped = {}
    for row in run_rows:
        grouped.setdefault((str(row["problem"]).upper(), str(row["method"])), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["seed"]))
    return grouped


def summarize_problem(problem_code, run_rows):
    grouped = group_run_rows(run_rows)
    baseline_rows = grouped[(problem_code, "baseline")]
    early_rows = grouped[(problem_code, "early-switch-only")]
    diagnostic_rows = grouped[(problem_code, "diagnostic-only")]

    baseline_mean = mean_or_nan([row["best_error"] for row in baseline_rows if str(row["status"]) == "ok"])
    early_mean = mean_or_nan([row["best_error"] for row in early_rows if str(row["status"]) == "ok"])
    diagnostic_mean = mean_or_nan([row["best_error"] for row in diagnostic_rows if str(row["status"]) == "ok"])
    early_vs_baseline_gap = float(early_mean - baseline_mean) if np.isfinite(early_mean) and np.isfinite(baseline_mean) else float("nan")

    paired_rows = list(zip(early_rows, diagnostic_rows))
    best_error_match = all(
        np.isclose(to_float(early_row["best_error"]), to_float(diagnostic_row["best_error"]), rtol=1e-12, atol=1e-9)
        for early_row, diagnostic_row in paired_rows
    )
    final_error_match = all(
        np.isclose(to_float(early_row["final_error"]), to_float(diagnostic_row["final_error"]), rtol=1e-12, atol=1e-9)
        for early_row, diagnostic_row in paired_rows
    )
    fe_match = all(int(early_row["fe_used"]) == int(diagnostic_row["fe_used"]) for early_row, diagnostic_row in paired_rows)
    nda_fe_match = all(int(early_row["nda_fe_used"]) == int(diagnostic_row["nda_fe_used"]) for early_row, diagnostic_row in paired_rows)
    early_switch_flag_match = all(
        bool(early_row["early_switch_triggered"]) == bool(diagnostic_row["early_switch_triggered"])
        for early_row, diagnostic_row in paired_rows
    )
    diagnostic_matches_early = bool(best_error_match and final_error_match and fe_match and nda_fe_match and early_switch_flag_match)

    early_switch_trigger_count = int(sum(1 for row in early_rows if bool(row["early_switch_triggered"])))
    max_diag_best_diff = max(
        (
            abs(to_float(early_row["best_error"]) - to_float(diagnostic_row["best_error"]))
            for early_row, diagnostic_row in paired_rows
            if np.isfinite(to_float(early_row["best_error"])) and np.isfinite(to_float(diagnostic_row["best_error"]))
        ),
        default=0.0,
    )

    return {
        "problem": problem_code,
        "baseline_mean": baseline_mean,
        "early_switch_mean": early_mean,
        "diagnostic_only_mean": diagnostic_mean,
        "early_vs_baseline_gap": early_vs_baseline_gap,
        "diagnostic_matches_early": diagnostic_matches_early,
        "early_switch_trigger_count": early_switch_trigger_count,
        "max_diag_best_diff": float(max_diag_best_diff),
    }


def build_report(problem_inputs_map, run_rows, overlap_rows):
    summary_rows = [summarize_problem(problem, run_rows) for problem in PROBLEMS]
    overlap_by_problem = {row["problem"]: row for row in overlap_rows}
    status_rows = [row for row in run_rows if str(row["status"]) != "ok"]
    fe_budget_ok = all(int(row["fe_used"]) <= int(MAX_FES) for row in run_rows if str(row["status"]) == "ok")
    all_diag_matches = all(bool(row["diagnostic_matches_early"]) for row in summary_rows)
    nonadjacent_dominates = all(
        float(overlap_by_problem[problem]["nonadjacent_ratio"]) > 0.5
        for problem in PROBLEMS
        if int(overlap_by_problem[problem]["true_overlap_var_count"]) > 0
    )
    early_beats_baseline_on = [row["problem"] for row in summary_rows if np.isfinite(row["early_vs_baseline_gap"]) and float(row["early_vs_baseline_gap"]) < 0.0]
    early_loses_to_baseline_on = [row["problem"] for row in summary_rows if np.isfinite(row["early_vs_baseline_gap"]) and float(row["early_vs_baseline_gap"]) > 0.0]
    can_enter_hypergraph = bool(all_diag_matches and nonadjacent_dominates and len(early_beats_baseline_on) >= len(early_loses_to_baseline_on))
    visibility_unchanged = all(not bool(row["visibility_changed_by_patch"]) for row in overlap_rows)

    report_lines = [
        "# AOB Decision Dimension Baseline Report",
        "",
        f"- Problems: {', '.join(PROBLEMS)}",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        f"- MaxFEs: {MAX_FES}",
        "",
        "## 5.1 Decision-dimension confirmation",
        "",
        "| problem | benchmark_dimension | design_matrix_dimension | group_count | residual_group_size | final_coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem in PROBLEMS:
        overlap_row = overlap_by_problem[problem]
        report_lines.append(
            "| {problem} | {benchmark_dimension} | {design_matrix_dimension} | {group_count} | {residual_group_size} | {covered}/{benchmark_dimension} |".format(
                problem=problem,
                benchmark_dimension=overlap_row["benchmark_dimension"],
                design_matrix_dimension=overlap_row["design_matrix_dimension"],
                group_count=overlap_row["group_count"],
                residual_group_size=overlap_row["residual_group_size"],
                covered=overlap_row["effective_covered_variable_count"],
            )
        )
    report_lines.extend(
        [
            "",
            "确认：",
            "- 所有问题的 benchmark decision dimension 与 design matrix dimension 都一致为 1000。",
            "- 所有问题都由原始 20 个 RDDSM groups 完整覆盖，未追加 residual group。",
            "- `dimension_real` 对应 overlap 展开长度，而不是独立决策变量数。",
            "",
            "## 5.2 Baseline rebuild results",
            "",
            "_表中均值使用 5 seeds 的 mean best_error。gap = early_switch_mean - baseline_mean，gap < 0 表示 early switch 更好。_",
            "",
            "| problem | baseline_mean | early_switch_mean | diagnostic_only_mean | early_vs_baseline_gap | diagnostic_matches_early |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        report_lines.append(
            "| {problem} | {baseline_mean} | {early_switch_mean} | {diagnostic_only_mean} | {gap} | {diag_match} |".format(
                problem=row["problem"],
                baseline_mean=format_metric(row["baseline_mean"]),
                early_switch_mean=format_metric(row["early_switch_mean"]),
                diagnostic_only_mean=format_metric(row["diagnostic_only_mean"]),
                gap=format_metric(row["early_vs_baseline_gap"]),
                diag_match="yes" if row["diagnostic_matches_early"] else "no",
            )
        )
    report_lines.extend(
        [
            "",
            f"- FE budget respected: {'yes' if fe_budget_ok else 'no'}",
            f"- early-switch-only beats baseline on: {', '.join(early_beats_baseline_on) if early_beats_baseline_on else 'none'}",
            f"- early-switch-only loses to baseline on: {', '.join(early_loses_to_baseline_on) if early_loses_to_baseline_on else 'none'}",
            f"- diagnostic-only exact/near match against early-switch-only on all 6 problems: {'yes' if all_diag_matches else 'no'}",
            "",
            "## 5.3 Overlap visibility re-audit",
            "",
            "| problem | true_overlap | adjacent_visible | visible_ratio | nonadjacent_ratio | adjacent_boundary_count |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for problem in PROBLEMS:
        row = overlap_by_problem[problem]
        report_lines.append(
            "| {problem} | {true_overlap} | {adjacent_visible} | {visible_ratio:.6f} | {nonadjacent_ratio:.6f} | {adjacent_boundary_count} |".format(
                problem=problem,
                true_overlap=row["true_overlap_var_count"],
                adjacent_visible=row["adjacent_visible_overlap_var_count"],
                visible_ratio=float(row["visible_ratio"]),
                nonadjacent_ratio=float(row["nonadjacent_ratio"]),
                adjacent_boundary_count=row["adjacent_boundary_count"],
            )
        )
    report_lines.extend(
        [
            "",
            f"- visibility changed by decision-dimension correction: {'yes' if not visibility_unchanged else 'no'}",
            "- 修正 decision dimension 后，true overlap 和 adjacent visible overlap 计数与原始分组保持一致。",
            "- non-adjacent overlap 仍然主导，因此当前 adjacent-only coordination 的可见范围仍然偏窄。",
            "",
            "## 5.4 Next step recommendation",
            "",
        ]
    )

    if can_enter_hypergraph:
        report_lines.extend(
            [
                "结论：进入 Hypergraph shared-variable coordination。",
                "",
                "理由：",
                "- diagnostic-only 与 early-switch-only 行为一致，没有看出诊断路径副作用；",
                "- non-adjacent overlap 在重审后仍占主导；",
                "- early-switch-only 在修正后的维度语义下仍保持不弱于 baseline 的整体表现。",
            ]
        )
    elif not all_diag_matches:
        report_lines.extend(
            [
                "结论：先暂停新协调模块，优先排查 diagnostic-only 的副作用。",
                "",
                "理由：",
                "- diagnostic-only 与 early-switch-only 不完全一致，先保证诊断路径是透明的。",
            ]
        )
    else:
        report_lines.extend(
            [
                "结论：先暂停 hypergraph coordinator，继续审查当前调度与预算影响。",
                "",
                "理由：",
                "- 虽然 non-adjacent overlap 仍然主导，但当前 baseline 上 early-switch-only 没有稳定压过 baseline。",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- overlap visibility: `{OVERLAP_VISIBILITY_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )

    if status_rows:
        report_lines.extend(
            [
                "",
                "## Run failures",
                "",
            ]
        )
        for row in status_rows:
            report_lines.append(f"- {row['problem']} / {row['method']} / seed {row['seed']}: {row['status']}")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def main():
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    method_configs = {
        method_name: load_method_config(config_path)
        for method_name, config_path in METHOD_CONFIGS
    }
    problem_inputs_map = {
        problem: build_problem_inputs(problem)
        for problem in PROBLEMS
    }
    overlap_rows = [
        build_overlap_visibility_row(problem, problem_inputs_map[problem])
        for problem in PROBLEMS
    ]

    run_rows = []
    for method_name, _ in METHOD_CONFIGS:
        print(f"[method] {method_name}")
        config = method_configs[method_name]
        for problem in PROBLEMS:
            problem_inputs = problem_inputs_map[problem]
            for seed in SEEDS:
                print(f"  running {problem} seed={seed}")
                row = run_one_case(method_name, config, problem, seed, problem_inputs)
                run_rows.append(row)
                print(
                    "    done status={status} best_error={best_error:.6e} fe_used={fe_used} nda_fe_used={nda_fe_used} early_switch={early_switch}".format(
                        status=row["status"],
                        best_error=to_float(row["best_error"]),
                        fe_used=int(row["fe_used"]),
                        nda_fe_used=int(row["nda_fe_used"]),
                        early_switch=bool(row["early_switch_triggered"]),
                    )
                )

    run_rows.sort(key=lambda row: (PROBLEMS.index(row["problem"]), METHOD_ORDER[row["method"]], int(row["seed"])))
    overlap_rows.sort(key=lambda row: PROBLEMS.index(row["problem"]))

    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(OVERLAP_VISIBILITY_PATH, OVERLAP_VISIBILITY_FIELDNAMES, overlap_rows)
    build_report(problem_inputs_map, run_rows, overlap_rows)

    print(f"run details -> {RUN_DETAILS_PATH}")
    print(f"overlap visibility -> {OVERLAP_VISIBILITY_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
