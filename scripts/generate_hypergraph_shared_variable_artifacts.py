import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "hypergraph_shared_variable_runs"
PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
SEEDS = [1, 2, 3, 4, 5]
MAX_FES = 1000
METHOD_CONFIGS = [
    ("baseline", CONFIG_ROOT / "baseline.json"),
    ("adjacent-original", CONFIG_ROOT / "adjacent-original.json"),
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("hypergraph-pass-end", CONFIG_ROOT / "hypergraph-pass-end.json"),
    ("hypergraph-pass-end-conflict", CONFIG_ROOT / "hypergraph-pass-end-conflict.json"),
    ("selective-hypergraph", CONFIG_ROOT / "selective-hypergraph.json"),
    ("selective-hypergraph-conflict", CONFIG_ROOT / "selective-hypergraph-conflict.json"),
]
METHOD_ORDER = {method_name: index for index, (method_name, _) in enumerate(METHOD_CONFIGS)}
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "hypergraph_shared_variable_run_details.csv"
VISIBILITY_DETAILS_PATH = ARTIFACTS_ROOT / "hypergraph_shared_variable_visibility.csv"
ARBITRATION_DETAILS_PATH = ARTIFACTS_ROOT / "hypergraph_shared_variable_arbitration.csv"
REPORT_PATH = ARTIFACTS_ROOT / "hypergraph_shared_variable_report.md"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)


RUN_DETAIL_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "coordination_mode",
    "conflict_damping",
    "selective_gate_enabled",
    "final_error",
    "best_error",
    "fe_used",
    "nda_fe_used",
    "early_switch_triggered",
    "early_switch_reason",
    "status",
    "runtime",
]

VISIBILITY_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "coordination_mode",
    "true_overlap_var_count",
    "adjacent_visible_overlap_var_count",
    "coordinated_overlap_var_count",
    "coordinated_overlap_ratio",
    "proposal_covered_overlap_var_count",
    "proposal_coverage_ratio",
    "positive_overlap_var_count",
    "positive_overlap_ratio",
    "applied_overlap_var_count",
    "applied_overlap_ratio",
    "candidate_overlap_var_count",
    "update_ratio",
    "skip_ratio",
    "freeze_ratio",
    "owner_soft_ratio",
    "multi_support_ratio",
    "mean_update_magnitude",
    "mean_update_magnitude_ratio",
    "proposal_std_mean",
    "proposal_std_ratio_mean",
    "positive_proposal_count_mean",
    "harmful_update_proxy",
    "support_blocked_ratio",
    "std_blocked_ratio",
    "owner_delta_blocked_ratio",
    "large_update_damped_ratio",
    "owner_soft_damped_ratio",
    "multi_support_damped_ratio",
    "mean_positive_proposal_count",
    "mean_damping",
]

ARBITRATION_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "coordination_mode",
    "cycle_id",
    "var_id",
    "membership_count",
    "proposal_count",
    "positive_proposal_count",
    "negative_proposal_count",
    "proposal_value_min",
    "proposal_value_max",
    "proposal_value_std",
    "proposal_value_std_ratio",
    "old_value",
    "fused_value",
    "applied_value",
    "raw_update_magnitude",
    "raw_update_ratio",
    "update_magnitude",
    "update_magnitude_ratio",
    "ownership_mode",
    "owner_group_id",
    "owner_value",
    "owner_delta",
    "owner_delta_ratio",
    "owner_delta_share",
    "owner_step_weight",
    "weighted_delta_sum",
    "sum_positive_delta",
    "max_delta",
    "top_group_id",
    "conflict_prior",
    "damping",
    "skip_reason",
    "gate_passed",
    "risky_candidate",
    "harmful_update_proxy_flag",
    "applied_update",
    "was_updated",
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


def format_ratio(value):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.3f}"


def build_problem_inputs(problem_code):
    return hcc_es.build_hcc_es_inputs(problem_code)


def build_default_visibility(problem_inputs, config):
    return hcc_es.build_shared_variable_visibility_audit(
        problem_inputs["grouping_result"],
        problem_inputs["adjacent_overlapping_elements"],
        coordination_mode=hcc_es.resolve_shared_variable_coordination_mode(config),
    )


def build_default_coordination_summary(problem_inputs, config):
    visibility = build_default_visibility(problem_inputs, config)
    return hcc_es.summarize_shared_variable_coordination_rows([], visibility_audit=visibility)


def run_one_case(method_name, config, problem_code, seed, problem_inputs):
    output_dir = RUNS_ROOT / method_name
    default_visibility = build_default_visibility(problem_inputs, config)
    default_summary = build_default_coordination_summary(problem_inputs, config)
    default_arbitration_rows = []
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
        visibility = dict(default_visibility)
        visibility.update(metadata.get("coordination_visibility_audit", {}))
        visibility.update(diagnostics_payload.get("coordination_visibility_audit", {}))
        coordination_summary = dict(default_summary)
        coordination_summary.update(metadata.get("shared_variable_coordination_summary", {}))
        coordination_summary.update(diagnostics_payload.get("shared_variable_coordination_summary", {}))
        coordination_summary["coordination_mode"] = visibility["coordination_mode"]
        arbitration_rows = [
            {
                **row,
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "coordination_mode": str(coordination_summary["coordination_mode"]),
            }
            for row in metadata.get("shared_variable_fusion_rows", [])
        ]
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "coordination_mode": str(visibility["coordination_mode"]),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "selective_gate_enabled": hcc_es.is_selective_hypergraph_mode(visibility["coordination_mode"]),
                "final_error": to_float(detail.get("final_fitness")),
                "best_error": to_float(detail.get("best_fitness")),
                "fe_used": int(detail.get("fe_used", 0) or 0),
                "nda_fe_used": int(diagnostics_payload.get("nda_fe_used", 0) or 0),
                "early_switch_triggered": bool(diagnostics_payload.get("early_switch_triggered", False)),
                "early_switch_reason": str(diagnostics_payload.get("early_switch_reason", "")),
                "status": str(detail.get("status", "")),
                "runtime": to_float(detail.get("runtime")),
            },
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "coordination_mode": str(coordination_summary["coordination_mode"]),
                "true_overlap_var_count": int(coordination_summary["true_overlap_var_count"]),
                "adjacent_visible_overlap_var_count": int(coordination_summary["adjacent_visible_overlap_var_count"]),
                "coordinated_overlap_var_count": int(coordination_summary["coordinated_overlap_var_count"]),
                "coordinated_overlap_ratio": float(coordination_summary["coordinated_overlap_ratio"]),
                "proposal_covered_overlap_var_count": int(coordination_summary["proposal_covered_overlap_var_count"]),
                "proposal_coverage_ratio": float(coordination_summary["proposal_coverage_ratio"]),
                "positive_overlap_var_count": int(coordination_summary["positive_overlap_var_count"]),
                "positive_overlap_ratio": float(coordination_summary["positive_overlap_ratio"]),
                "applied_overlap_var_count": int(coordination_summary["applied_overlap_var_count"]),
                "applied_overlap_ratio": float(coordination_summary["applied_overlap_ratio"]),
                "candidate_overlap_var_count": int(coordination_summary["candidate_overlap_var_count"]),
                "update_ratio": float(coordination_summary["update_ratio"]),
                "skip_ratio": float(coordination_summary["skip_ratio"]),
                "freeze_ratio": float(coordination_summary["freeze_ratio"]),
                "owner_soft_ratio": float(coordination_summary["owner_soft_ratio"]),
                "multi_support_ratio": float(coordination_summary["multi_support_ratio"]),
                "mean_update_magnitude": to_float(coordination_summary["mean_update_magnitude"]),
                "mean_update_magnitude_ratio": to_float(coordination_summary["mean_update_magnitude_ratio"]),
                "proposal_std_mean": to_float(coordination_summary["proposal_std_mean"]),
                "proposal_std_ratio_mean": to_float(coordination_summary["proposal_std_ratio_mean"]),
                "positive_proposal_count_mean": float(coordination_summary["positive_proposal_count_mean"]),
                "harmful_update_proxy": float(coordination_summary["harmful_update_proxy"]),
                "support_blocked_ratio": float(coordination_summary["support_blocked_ratio"]),
                "std_blocked_ratio": float(coordination_summary["std_blocked_ratio"]),
                "owner_delta_blocked_ratio": float(coordination_summary["owner_delta_blocked_ratio"]),
                "large_update_damped_ratio": float(coordination_summary["large_update_damped_ratio"]),
                "owner_soft_damped_ratio": float(coordination_summary["owner_soft_damped_ratio"]),
                "multi_support_damped_ratio": float(coordination_summary["multi_support_damped_ratio"]),
                "mean_positive_proposal_count": float(coordination_summary["mean_positive_proposal_count"]),
                "mean_damping": to_float(coordination_summary["mean_damping"]),
            },
            arbitration_rows,
        )
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "coordination_mode": str(default_visibility["coordination_mode"]),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "selective_gate_enabled": hcc_es.is_selective_hypergraph_mode(default_visibility["coordination_mode"]),
                "final_error": float("nan"),
                "best_error": float("nan"),
                "fe_used": 0,
                "nda_fe_used": 0,
                "early_switch_triggered": False,
                "early_switch_reason": "",
                "status": f"error: {exc}",
                "runtime": float("nan"),
            },
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "coordination_mode": str(default_summary["coordination_mode"]),
                "true_overlap_var_count": int(default_summary["true_overlap_var_count"]),
                "adjacent_visible_overlap_var_count": int(default_summary["adjacent_visible_overlap_var_count"]),
                "coordinated_overlap_var_count": int(default_summary["coordinated_overlap_var_count"]),
                "coordinated_overlap_ratio": float(default_summary["coordinated_overlap_ratio"]),
                "proposal_covered_overlap_var_count": int(default_summary["proposal_covered_overlap_var_count"]),
                "proposal_coverage_ratio": float(default_summary["proposal_coverage_ratio"]),
                "positive_overlap_var_count": int(default_summary["positive_overlap_var_count"]),
                "positive_overlap_ratio": float(default_summary["positive_overlap_ratio"]),
                "applied_overlap_var_count": int(default_summary["applied_overlap_var_count"]),
                "applied_overlap_ratio": float(default_summary["applied_overlap_ratio"]),
                "candidate_overlap_var_count": int(default_summary["candidate_overlap_var_count"]),
                "update_ratio": float(default_summary["update_ratio"]),
                "skip_ratio": float(default_summary["skip_ratio"]),
                "freeze_ratio": float(default_summary["freeze_ratio"]),
                "owner_soft_ratio": float(default_summary["owner_soft_ratio"]),
                "multi_support_ratio": float(default_summary["multi_support_ratio"]),
                "mean_update_magnitude": to_float(default_summary["mean_update_magnitude"]),
                "mean_update_magnitude_ratio": to_float(default_summary["mean_update_magnitude_ratio"]),
                "proposal_std_mean": to_float(default_summary["proposal_std_mean"]),
                "proposal_std_ratio_mean": to_float(default_summary["proposal_std_ratio_mean"]),
                "positive_proposal_count_mean": float(default_summary["positive_proposal_count_mean"]),
                "harmful_update_proxy": float(default_summary["harmful_update_proxy"]),
                "support_blocked_ratio": float(default_summary["support_blocked_ratio"]),
                "std_blocked_ratio": float(default_summary["std_blocked_ratio"]),
                "owner_delta_blocked_ratio": float(default_summary["owner_delta_blocked_ratio"]),
                "large_update_damped_ratio": float(default_summary["large_update_damped_ratio"]),
                "owner_soft_damped_ratio": float(default_summary["owner_soft_damped_ratio"]),
                "multi_support_damped_ratio": float(default_summary["multi_support_damped_ratio"]),
                "mean_positive_proposal_count": float(default_summary["mean_positive_proposal_count"]),
                "mean_damping": to_float(default_summary["mean_damping"]),
            },
            default_arbitration_rows,
        )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def group_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((str(row["problem"]).upper(), str(row["method"])), []).append(row)
    for grouped_rows in grouped.values():
        grouped_rows.sort(key=lambda row: int(row["seed"]))
    return grouped


def summarize_method_problem(run_rows, visibility_rows, problem_code, method_name):
    method_runs = [
        row
        for row in run_rows
        if str(row["problem"]).upper() == problem_code and str(row["method"]) == method_name and str(row["status"]) == "ok"
    ]
    method_visibility_rows = [
        row
        for row in visibility_rows
        if str(row["problem"]).upper() == problem_code and str(row["method"]) == method_name
    ]
    summary = {
        "best_mean": mean_or_nan([row["best_error"] for row in method_runs]),
        "coordinated_overlap_var_count": mean_or_nan([row["coordinated_overlap_var_count"] for row in method_visibility_rows]),
        "coordinated_overlap_ratio": mean_or_nan([row["coordinated_overlap_ratio"] for row in method_visibility_rows]),
        "proposal_coverage_ratio": mean_or_nan([row["proposal_coverage_ratio"] for row in method_visibility_rows]),
        "applied_overlap_ratio": mean_or_nan([row["applied_overlap_ratio"] for row in method_visibility_rows]),
        "update_ratio": mean_or_nan([row["update_ratio"] for row in method_visibility_rows]),
        "skip_ratio": mean_or_nan([row["skip_ratio"] for row in method_visibility_rows]),
        "freeze_ratio": mean_or_nan([row["freeze_ratio"] for row in method_visibility_rows]),
        "owner_soft_ratio": mean_or_nan([row["owner_soft_ratio"] for row in method_visibility_rows]),
        "multi_support_ratio": mean_or_nan([row["multi_support_ratio"] for row in method_visibility_rows]),
        "mean_update_magnitude": mean_or_nan([row["mean_update_magnitude"] for row in method_visibility_rows]),
        "proposal_std_mean": mean_or_nan([row["proposal_std_mean"] for row in method_visibility_rows]),
        "positive_proposal_count_mean": mean_or_nan([row["positive_proposal_count_mean"] for row in method_visibility_rows]),
        "harmful_update_proxy": mean_or_nan([row["harmful_update_proxy"] for row in method_visibility_rows]),
        "support_blocked_ratio": mean_or_nan([row["support_blocked_ratio"] for row in method_visibility_rows]),
        "std_blocked_ratio": mean_or_nan([row["std_blocked_ratio"] for row in method_visibility_rows]),
        "owner_delta_blocked_ratio": mean_or_nan([row["owner_delta_blocked_ratio"] for row in method_visibility_rows]),
        "large_update_damped_ratio": mean_or_nan([row["large_update_damped_ratio"] for row in method_visibility_rows]),
        "owner_soft_damped_ratio": mean_or_nan([row["owner_soft_damped_ratio"] for row in method_visibility_rows]),
        "multi_support_damped_ratio": mean_or_nan([row["multi_support_damped_ratio"] for row in method_visibility_rows]),
        "mean_damping": mean_or_nan([row["mean_damping"] for row in method_visibility_rows]),
    }
    if method_visibility_rows:
        summary["true_overlap_var_count"] = int(method_visibility_rows[0]["true_overlap_var_count"])
        summary["adjacent_visible_overlap_var_count"] = int(method_visibility_rows[0]["adjacent_visible_overlap_var_count"])
        summary["coordination_mode"] = str(method_visibility_rows[0]["coordination_mode"])
    else:
        summary["true_overlap_var_count"] = 0
        summary["adjacent_visible_overlap_var_count"] = 0
        summary["coordination_mode"] = ""
    return summary


def build_report(run_rows, visibility_rows):
    summary_by_key = {
        (problem_code, method_name): summarize_method_problem(run_rows, visibility_rows, problem_code, method_name)
        for problem_code in PROBLEMS
        for method_name, _ in METHOD_CONFIGS
    }

    adjacent_total_visible = sum(
        int(round(to_float(summary_by_key[(problem_code, "adjacent-original")]["coordinated_overlap_var_count"])))
        for problem_code in PROBLEMS
    )
    hypergraph_total_visible = sum(
        int(round(to_float(summary_by_key[(problem_code, "hypergraph-pass-end")]["coordinated_overlap_var_count"])))
        for problem_code in PROBLEMS
    )
    selective_total_visible = sum(
        int(round(to_float(summary_by_key[(problem_code, "selective-hypergraph")]["coordinated_overlap_var_count"])))
        for problem_code in PROBLEMS
    )
    total_true_overlap = sum(int(summary_by_key[(problem_code, "adjacent-original")]["true_overlap_var_count"]) for problem_code in PROBLEMS)

    report_lines = [
        "# Dynamic Soft Ownership Report",
        "",
        f"- Problems: {', '.join(PROBLEMS)}",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        f"- MaxFEs: {MAX_FES}",
        "",
        "## 1. Visibility-first audit",
        "",
        "| problem | true_overlap | adjacent_original_visible | hypergraph_visible | selective_visible | selective_conflict_visible |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem_code in PROBLEMS:
        adjacent_summary = summary_by_key[(problem_code, "adjacent-original")]
        hyper_summary = summary_by_key[(problem_code, "hypergraph-pass-end")]
        selective_summary = summary_by_key[(problem_code, "selective-hypergraph")]
        selective_conflict_summary = summary_by_key[(problem_code, "selective-hypergraph-conflict")]
        report_lines.append(
            "| {problem} | {true_overlap} | {adjacent_visible} | {hyper_visible} | {selective_visible} | {selective_conflict_visible} |".format(
                problem=problem_code,
                true_overlap=int(adjacent_summary["true_overlap_var_count"]),
                adjacent_visible=int(round(to_float(adjacent_summary["coordinated_overlap_var_count"]))),
                hyper_visible=int(round(to_float(hyper_summary["coordinated_overlap_var_count"]))),
                selective_visible=int(round(to_float(selective_summary["coordinated_overlap_var_count"]))),
                selective_conflict_visible=int(round(to_float(selective_conflict_summary["coordinated_overlap_var_count"]))),
            )
        )

    report_lines.extend(
        [
            "",
            f"- overall adjacent-original effective intervention ratio: {format_ratio(adjacent_total_visible / total_true_overlap if total_true_overlap else float('nan'))}",
            f"- overall hypergraph-pass-end effective intervention ratio: {format_ratio(hypergraph_total_visible / total_true_overlap if total_true_overlap else float('nan'))}",
            f"- overall selective-hypergraph effective intervention ratio: {format_ratio(selective_total_visible / total_true_overlap if total_true_overlap else float('nan'))}",
            "- adjacent-original keeps the old adjacent visibility ceiling;",
            "- both hypergraph and selective-hypergraph see the full RDDSM overlap set; dynamic soft ownership changes arbitration, not visibility.",
            "",
            "## 2. Ownership arbitration metrics",
            "",
            "| problem | method | owner_soft_ratio | multi_support_ratio | freeze_ratio | mean_update_magnitude | harmful_update_proxy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for problem_code in PROBLEMS:
        for method_name, _ in METHOD_CONFIGS:
            summary = summary_by_key[(problem_code, method_name)]
            report_lines.append(
                "| {problem} | {method} | {owner_soft_ratio} | {multi_support_ratio} | {freeze_ratio} | {update_mag} | {harmful_proxy} |".format(
                    problem=problem_code,
                    method=method_name,
                    owner_soft_ratio=format_ratio(summary["owner_soft_ratio"]),
                    multi_support_ratio=format_ratio(summary["multi_support_ratio"]),
                    freeze_ratio=format_ratio(summary["freeze_ratio"]),
                    update_mag=format_metric(summary["mean_update_magnitude"]),
                    harmful_proxy=format_ratio(summary["harmful_update_proxy"]),
                )
            )

    report_lines.extend(
        [
            "",
            "## 3. Gate diagnostics",
            "",
            "| problem | method | update_ratio | skip_ratio | proposal_std_mean | positive_count_mean | support_blocked | std_blocked | owner_delta_blocked |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for problem_code in PROBLEMS:
        for method_name, _ in METHOD_CONFIGS:
            summary = summary_by_key[(problem_code, method_name)]
            report_lines.append(
                "| {problem} | {method} | {update_ratio} | {skip_ratio} | {proposal_std} | {positive_mean} | {support_blocked} | {std_blocked} | {owner_delta_blocked} |".format(
                    problem=problem_code,
                    method=method_name,
                    update_ratio=format_ratio(summary["update_ratio"]),
                    skip_ratio=format_ratio(summary["skip_ratio"]),
                    proposal_std=format_metric(summary["proposal_std_mean"]),
                    positive_mean=format_ratio(summary["positive_proposal_count_mean"]),
                    support_blocked=format_ratio(summary["support_blocked_ratio"]),
                    std_blocked=format_ratio(summary["std_blocked_ratio"]),
                    owner_delta_blocked=format_ratio(summary["owner_delta_blocked_ratio"]),
                )
            )

    report_lines.extend(
        [
            "",
            "## 4. Mean best_error at 1000 FEs",
            "",
            "| problem | method | best_mean | coordinated_ratio | applied_ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for problem_code in PROBLEMS:
        for method_name, _ in METHOD_CONFIGS:
            summary = summary_by_key[(problem_code, method_name)]
            report_lines.append(
                "| {problem} | {method} | {best_mean} | {coord_ratio} | {applied_ratio} |".format(
                    problem=problem_code,
                    method=method_name,
                    best_mean=format_metric(summary["best_mean"]),
                    coord_ratio=format_ratio(summary["coordinated_overlap_ratio"]),
                    applied_ratio=format_ratio(summary["applied_overlap_ratio"]),
                )
            )

    report_lines.extend(
        [
            "",
            "## 5. Readout",
            "",
            "先验收 visibility：",
            "- `E4/S4` 仍然是 `0 -> 95`，`E6/S6/A6/R6` 仍然是 `20 -> 190`；",
            "- dynamic soft ownership 没有缩小 visibility，它只是把弱证据变量回退成 freeze，或让单一 owner 小步接管。",
            "",
            "再看 ownership：",
            "- `hypergraph-pass-end` 的 `owner_soft_ratio` 应当接近 `0.000`，因为它看到就更；",
            "- selective 版本如果出现非零 `owner_soft_ratio`，说明单 owner 小步接管开始替代原来的硬 freeze；",
            "- `harmful_update_proxy` 继续看 full fusion 的潜在伤害有没有被压低；",
            "- 如果 selective 在 `best_error` 上更接近或超过 `no-coordination`，说明伤害主要来自错误融合，而不是缺少共享变量视野。",
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- visibility details: `{VISIBILITY_DETAILS_PATH.as_posix()}`",
            f"- arbitration details: `{ARBITRATION_DETAILS_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    return "\n".join(report_lines) + "\n"


def main():
    run_rows = []
    visibility_rows = []
    arbitration_rows = []
    problem_inputs_map = {problem_code: build_problem_inputs(problem_code) for problem_code in PROBLEMS}
    method_configs = [(method_name, load_method_config(config_path)) for method_name, config_path in METHOD_CONFIGS]

    for method_name, config in method_configs:
        for problem_code in PROBLEMS:
            for seed in SEEDS:
                print(f"[method] {method_name} problem={problem_code} seed={seed}")
                run_row, visibility_row, case_arbitration_rows = run_one_case(
                    method_name,
                    config,
                    problem_code,
                    seed,
                    problem_inputs_map[problem_code],
                )
                run_rows.append(run_row)
                visibility_rows.append(visibility_row)
                arbitration_rows.extend(case_arbitration_rows)
                print(
                    "    done status={status} best_error={best_error:.6e} coord_ratio={coord_ratio:.3f} update_ratio={update_ratio:.3f} skip_ratio={skip_ratio:.3f}".format(
                        status=run_row["status"],
                        best_error=to_float(run_row["best_error"]),
                        coord_ratio=to_float(visibility_row["coordinated_overlap_ratio"]),
                        update_ratio=to_float(visibility_row["update_ratio"]),
                        skip_ratio=to_float(visibility_row["skip_ratio"]),
                    )
                )

    run_rows.sort(key=lambda row: (row["problem"], int(row["seed"]), METHOD_ORDER[row["method"]]))
    visibility_rows.sort(key=lambda row: (row["problem"], int(row["seed"]), METHOD_ORDER[row["method"]]))
    arbitration_rows.sort(key=lambda row: (row["problem"], int(row["seed"]), METHOD_ORDER[row["method"]], int(row["cycle_id"]), int(row["var_id"])))
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(VISIBILITY_DETAILS_PATH, VISIBILITY_FIELDNAMES, visibility_rows)
    write_csv(ARBITRATION_DETAILS_PATH, ARBITRATION_FIELDNAMES, arbitration_rows)
    REPORT_PATH.write_text(build_report(run_rows, visibility_rows), encoding="utf-8")
    print(f"run details -> {RUN_DETAILS_PATH}")
    print(f"visibility details -> {VISIBILITY_DETAILS_PATH}")
    print(f"arbitration details -> {ARBITRATION_DETAILS_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
