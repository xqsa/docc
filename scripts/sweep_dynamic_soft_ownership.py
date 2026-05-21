import csv
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "dynamic_soft_ownership_sweep_runs"
PROBLEMS = ["A6", "R6", "E6"]
SEEDS = [1, 2, 3, 4, 5]
MAX_FES = 1000
STRICT_R6_RELATIVE_TOLERANCE = 0.002
HARMFUL_UPDATE_PROXY_TOLERANCE = 0.01
QUANTILE_LEVEL = 0.70
THRESHOLD_MODES = [
    ("strict", 1.0),
    ("medium", 0.5),
    ("loose", 0.25),
    ("quantile", None),
]
OWNER_ETAS = [0.1, 0.2, 0.3, 0.5]
BASELINE_CONFIGS = [
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("selective-current", CONFIG_ROOT / "selective-hypergraph.json"),
    ("selective-current-conflict", CONFIG_ROOT / "selective-hypergraph-conflict.json"),
]
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "dynamic_soft_ownership_sweep_run_details.csv"
VISIBILITY_DETAILS_PATH = ARTIFACTS_ROOT / "dynamic_soft_ownership_sweep_visibility.csv"
ARBITRATION_DETAILS_PATH = ARTIFACTS_ROOT / "dynamic_soft_ownership_sweep_arbitration.csv"
THRESHOLD_DETAILS_PATH = ARTIFACTS_ROOT / "dynamic_soft_ownership_sweep_thresholds.csv"
REPORT_PATH = ARTIFACTS_ROOT / "dynamic_soft_ownership_sweep_report.md"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)


RUN_DETAIL_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "owner_eta",
    "is_baseline",
    "coordination_mode",
    "conflict_damping",
    "final_error",
    "best_error",
    "status",
    "runtime",
]

VISIBILITY_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "owner_eta",
    "is_baseline",
    "coordination_mode",
    "true_overlap_var_count",
    "coordinated_overlap_var_count",
    "coordinated_overlap_ratio",
    "candidate_overlap_var_count",
    "update_ratio",
    "skip_ratio",
    "freeze_ratio",
    "owner_soft_ratio",
    "multi_support_ratio",
    "mean_update_magnitude",
    "mean_update_magnitude_ratio",
    "proposal_std_mean",
    "positive_proposal_count_mean",
    "harmful_update_proxy",
    "std_blocked_ratio",
    "owner_delta_blocked_ratio",
    "mean_damping",
    "owner_soft_selected_count",
    "owner_soft_updated_count",
    "owner_soft_updated_ratio",
    "owner_soft_update_hit_rate",
]

ARBITRATION_FIELDNAMES = [
    "problem",
    "seed",
    "method",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "owner_eta",
    "is_baseline",
    "coordination_mode",
    "cycle_id",
    "var_id",
    "proposal_count",
    "positive_proposal_count",
    "negative_proposal_count",
    "proposal_value_std_ratio",
    "raw_update_ratio",
    "update_magnitude_ratio",
    "ownership_mode",
    "owner_group_id",
    "owner_delta",
    "owner_delta_ratio",
    "owner_delta_share",
    "owner_step_weight",
    "skip_reason",
    "harmful_update_proxy_flag",
    "applied_update",
    "was_updated",
]

THRESHOLD_FIELDNAMES = [
    "problem",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "sample_count",
    "strict_threshold",
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


def format_percent_delta(value):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100.0:+.3f}%"


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


def build_case_arbitration_summary(arbitration_rows):
    candidate_rows = [
        row for row in arbitration_rows if int(row.get("positive_proposal_count", 0) or 0) > 0
    ]
    owner_soft_rows = [
        row for row in candidate_rows if str(row.get("ownership_mode", "")) == "owner_soft"
    ]
    owner_soft_updated_rows = [
        row for row in owner_soft_rows if bool(row.get("was_updated"))
    ]
    return {
        "owner_soft_selected_count": int(len(owner_soft_rows)),
        "owner_soft_updated_count": int(len(owner_soft_updated_rows)),
        "owner_soft_updated_ratio": float(len(owner_soft_updated_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "owner_soft_update_hit_rate": float(len(owner_soft_updated_rows) / len(owner_soft_rows)) if owner_soft_rows else 0.0,
    }


def run_one_case(
    method_name,
    config,
    problem_code,
    seed,
    problem_inputs,
    threshold_mode,
    threshold_value,
    threshold_source,
    owner_eta,
    is_baseline,
):
    output_dir = RUNS_ROOT / method_name
    default_visibility = build_default_visibility(problem_inputs, config)
    default_summary = build_default_coordination_summary(problem_inputs, config)
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
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(owner_eta) if owner_eta is not None else float("nan"),
                "is_baseline": bool(is_baseline),
                "coordination_mode": str(coordination_summary["coordination_mode"]),
            }
            for row in metadata.get("shared_variable_fusion_rows", [])
        ]
        arbitration_case_summary = build_case_arbitration_summary(arbitration_rows)
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(owner_eta) if owner_eta is not None else float("nan"),
                "is_baseline": bool(is_baseline),
                "coordination_mode": str(visibility["coordination_mode"]),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "final_error": to_float(detail.get("final_fitness")),
                "best_error": to_float(detail.get("best_fitness")),
                "status": str(detail.get("status", "")),
                "runtime": to_float(detail.get("runtime")),
            },
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(owner_eta) if owner_eta is not None else float("nan"),
                "is_baseline": bool(is_baseline),
                "coordination_mode": str(coordination_summary["coordination_mode"]),
                "true_overlap_var_count": int(coordination_summary["true_overlap_var_count"]),
                "coordinated_overlap_var_count": int(coordination_summary["coordinated_overlap_var_count"]),
                "coordinated_overlap_ratio": float(coordination_summary["coordinated_overlap_ratio"]),
                "candidate_overlap_var_count": int(coordination_summary["candidate_overlap_var_count"]),
                "update_ratio": float(coordination_summary["update_ratio"]),
                "skip_ratio": float(coordination_summary["skip_ratio"]),
                "freeze_ratio": float(coordination_summary["freeze_ratio"]),
                "owner_soft_ratio": float(coordination_summary["owner_soft_ratio"]),
                "multi_support_ratio": float(coordination_summary["multi_support_ratio"]),
                "mean_update_magnitude": to_float(coordination_summary["mean_update_magnitude"]),
                "mean_update_magnitude_ratio": to_float(coordination_summary["mean_update_magnitude_ratio"]),
                "proposal_std_mean": to_float(coordination_summary["proposal_std_mean"]),
                "positive_proposal_count_mean": float(coordination_summary["positive_proposal_count_mean"]),
                "harmful_update_proxy": float(coordination_summary["harmful_update_proxy"]),
                "std_blocked_ratio": float(coordination_summary["std_blocked_ratio"]),
                "owner_delta_blocked_ratio": float(coordination_summary["owner_delta_blocked_ratio"]),
                "mean_damping": to_float(coordination_summary["mean_damping"]),
                **arbitration_case_summary,
            },
            arbitration_rows,
        )
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(owner_eta) if owner_eta is not None else float("nan"),
                "is_baseline": bool(is_baseline),
                "coordination_mode": str(default_visibility["coordination_mode"]),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "final_error": float("nan"),
                "best_error": float("nan"),
                "status": f"error: {exc}",
                "runtime": float("nan"),
            },
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(owner_eta) if owner_eta is not None else float("nan"),
                "is_baseline": bool(is_baseline),
                "coordination_mode": str(default_summary["coordination_mode"]),
                "true_overlap_var_count": int(default_summary["true_overlap_var_count"]),
                "coordinated_overlap_var_count": int(default_summary["coordinated_overlap_var_count"]),
                "coordinated_overlap_ratio": float(default_summary["coordinated_overlap_ratio"]),
                "candidate_overlap_var_count": int(default_summary["candidate_overlap_var_count"]),
                "update_ratio": float(default_summary["update_ratio"]),
                "skip_ratio": float(default_summary["skip_ratio"]),
                "freeze_ratio": float(default_summary["freeze_ratio"]),
                "owner_soft_ratio": float(default_summary["owner_soft_ratio"]),
                "multi_support_ratio": float(default_summary["multi_support_ratio"]),
                "mean_update_magnitude": to_float(default_summary["mean_update_magnitude"]),
                "mean_update_magnitude_ratio": to_float(default_summary["mean_update_magnitude_ratio"]),
                "proposal_std_mean": to_float(default_summary["proposal_std_mean"]),
                "positive_proposal_count_mean": float(default_summary["positive_proposal_count_mean"]),
                "harmful_update_proxy": float(default_summary["harmful_update_proxy"]),
                "std_blocked_ratio": float(default_summary["std_blocked_ratio"]),
                "owner_delta_blocked_ratio": float(default_summary["owner_delta_blocked_ratio"]),
                "mean_damping": to_float(default_summary["mean_damping"]),
                "owner_soft_selected_count": 0,
                "owner_soft_updated_count": 0,
                "owner_soft_updated_ratio": 0.0,
                "owner_soft_update_hit_rate": 0.0,
            },
            [],
        )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


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
        "final_mean": mean_or_nan([row["final_error"] for row in method_runs]),
        "runtime_mean": mean_or_nan([row["runtime"] for row in method_runs]),
        "owner_soft_ratio": mean_or_nan([row["owner_soft_ratio"] for row in method_visibility_rows]),
        "owner_soft_updated_ratio": mean_or_nan([row["owner_soft_updated_ratio"] for row in method_visibility_rows]),
        "owner_soft_update_hit_rate": mean_or_nan([row["owner_soft_update_hit_rate"] for row in method_visibility_rows]),
        "owner_soft_selected_count_mean": mean_or_nan([row["owner_soft_selected_count"] for row in method_visibility_rows]),
        "owner_soft_updated_count_mean": mean_or_nan([row["owner_soft_updated_count"] for row in method_visibility_rows]),
        "multi_support_ratio": mean_or_nan([row["multi_support_ratio"] for row in method_visibility_rows]),
        "freeze_ratio": mean_or_nan([row["freeze_ratio"] for row in method_visibility_rows]),
        "update_ratio": mean_or_nan([row["update_ratio"] for row in method_visibility_rows]),
        "skip_ratio": mean_or_nan([row["skip_ratio"] for row in method_visibility_rows]),
        "mean_update_magnitude": mean_or_nan([row["mean_update_magnitude"] for row in method_visibility_rows]),
        "harmful_update_proxy": mean_or_nan([row["harmful_update_proxy"] for row in method_visibility_rows]),
        "std_blocked_ratio": mean_or_nan([row["std_blocked_ratio"] for row in method_visibility_rows]),
        "owner_delta_blocked_ratio": mean_or_nan([row["owner_delta_blocked_ratio"] for row in method_visibility_rows]),
        "threshold_value": mean_or_nan([row["threshold_value"] for row in method_visibility_rows]),
        "owner_eta": mean_or_nan([row["owner_eta"] for row in method_visibility_rows]),
        "threshold_mode": str(method_visibility_rows[0]["threshold_mode"]) if method_visibility_rows else "",
        "threshold_source": str(method_visibility_rows[0]["threshold_source"]) if method_visibility_rows else "",
        "is_baseline": bool(method_visibility_rows[0]["is_baseline"]) if method_visibility_rows else False,
    }
    return summary


def build_quantile_thresholds_from_baseline(arbitration_rows, strict_threshold):
    threshold_rows = []
    thresholds_by_problem = {}
    for problem_code in PROBLEMS:
        values = [
            float(row["owner_delta_ratio"])
            for row in arbitration_rows
            if str(row["problem"]).upper() == problem_code
            and str(row["method"]) == "selective-current"
            and int(row.get("positive_proposal_count", 0) or 0) == 1
            and np.isfinite(float(row.get("owner_delta_ratio", float("nan"))))
            and float(row.get("owner_delta_ratio", 0.0)) > 0.0
        ]
        if values:
            threshold_value = float(np.quantile(np.asarray(values, dtype=float), QUANTILE_LEVEL))
            threshold_source = f"selective-current_q{int(QUANTILE_LEVEL * 100)}"
        else:
            threshold_value = float(strict_threshold)
            threshold_source = "strict_fallback_no_single_owner"
        thresholds_by_problem[problem_code] = {
            "strict": float(strict_threshold),
            "medium": float(strict_threshold * 0.5),
            "loose": float(strict_threshold * 0.25),
            "quantile": float(threshold_value),
            "quantile_source": str(threshold_source),
            "sample_count": int(len(values)),
        }
        threshold_rows.extend(
            [
                {
                    "problem": problem_code,
                    "threshold_mode": "strict",
                    "threshold_value": float(strict_threshold),
                    "threshold_source": "current_strict",
                    "sample_count": int(len(values)),
                    "strict_threshold": float(strict_threshold),
                },
                {
                    "problem": problem_code,
                    "threshold_mode": "medium",
                    "threshold_value": float(strict_threshold * 0.5),
                    "threshold_source": "current_strict_x0.5",
                    "sample_count": int(len(values)),
                    "strict_threshold": float(strict_threshold),
                },
                {
                    "problem": problem_code,
                    "threshold_mode": "loose",
                    "threshold_value": float(strict_threshold * 0.25),
                    "threshold_source": "current_strict_x0.25",
                    "sample_count": int(len(values)),
                    "strict_threshold": float(strict_threshold),
                },
                {
                    "problem": problem_code,
                    "threshold_mode": "quantile",
                    "threshold_value": float(threshold_value),
                    "threshold_source": str(threshold_source),
                    "sample_count": int(len(values)),
                    "strict_threshold": float(strict_threshold),
                },
            ]
        )
    return thresholds_by_problem, threshold_rows


def build_combination_screen(summary_by_key):
    selective_best_map = {}
    no_coord_map = {}
    for problem_code in PROBLEMS:
        selective_best_map[problem_code] = min(
            to_float(summary_by_key[(problem_code, "selective-current")]["best_mean"]),
            to_float(summary_by_key[(problem_code, "selective-current-conflict")]["best_mean"]),
        )
        no_coord_map[problem_code] = to_float(summary_by_key[(problem_code, "no-coordination")]["best_mean"])

    sweep_methods = sorted(
        {
            method_name
            for problem_code, method_name in summary_by_key
            if method_name.startswith("sweep-")
        }
    )
    screen_rows = []
    for method_name in sweep_methods:
        a6 = summary_by_key[("A6", method_name)]
        r6 = summary_by_key[("R6", method_name)]
        e6 = summary_by_key[("E6", method_name)]
        a6_vs_no_coord = float((to_float(a6["best_mean"]) - no_coord_map["A6"]) / no_coord_map["A6"])
        a6_vs_selective = float((to_float(a6["best_mean"]) - selective_best_map["A6"]) / selective_best_map["A6"])
        a6_final_vs_no_coord = float((to_float(a6["final_mean"]) - summary_by_key[("A6", "no-coordination")]["final_mean"]) / summary_by_key[("A6", "no-coordination")]["final_mean"])
        a6_final_vs_selective = float((to_float(a6["final_mean"]) - summary_by_key[("A6", "selective-current")]["final_mean"]) / summary_by_key[("A6", "selective-current")]["final_mean"])
        r6_vs_no_coord = float((to_float(r6["best_mean"]) - no_coord_map["R6"]) / no_coord_map["R6"])
        e6_vs_no_coord = float((to_float(e6["best_mean"]) - no_coord_map["E6"]) / no_coord_map["E6"])
        max_harmful_proxy = float(
            max(
                to_float(a6["harmful_update_proxy"]),
                to_float(r6["harmful_update_proxy"]),
                to_float(e6["harmful_update_proxy"]),
            )
        )
        a6_owner_soft_ratio = to_float(a6["owner_soft_ratio"])
        screen_rows.append(
            {
                "method": method_name,
                "threshold_mode": str(a6["threshold_mode"]),
                "owner_eta": to_float(a6["owner_eta"]),
                "a6_best_mean": to_float(a6["best_mean"]),
                "a6_final_mean": to_float(a6["final_mean"]),
                "a6_vs_no_coord": a6_vs_no_coord,
                "a6_vs_selective": a6_vs_selective,
                "a6_final_vs_no_coord": a6_final_vs_no_coord,
                "a6_final_vs_selective": a6_final_vs_selective,
                "a6_owner_soft_ratio": a6_owner_soft_ratio,
                "a6_owner_soft_updated_ratio": to_float(a6["owner_soft_updated_ratio"]),
                "a6_owner_soft_update_hit_rate": to_float(a6["owner_soft_update_hit_rate"]),
                "a6_mean_update_magnitude": to_float(a6["mean_update_magnitude"]),
                "r6_vs_no_coord": r6_vs_no_coord,
                "e6_vs_no_coord": e6_vs_no_coord,
                "max_harmful_update_proxy": max_harmful_proxy,
                "a6_pass": bool(to_float(a6["best_mean"]) <= no_coord_map["A6"]),
                "r6_guard": bool(r6_vs_no_coord <= STRICT_R6_RELATIVE_TOLERANCE),
                "e6_guard": bool(to_float(e6["best_mean"]) <= no_coord_map["E6"]),
                "harm_guard": bool(max_harmful_proxy <= HARMFUL_UPDATE_PROXY_TOLERANCE),
                "a6_owner_soft_above_baseline": bool(
                    a6_owner_soft_ratio > to_float(summary_by_key[("A6", "selective-current")]["owner_soft_ratio"])
                ),
                "a6_owner_soft_in_band": bool(0.08 <= a6_owner_soft_ratio <= 0.15),
            }
        )
    screen_rows.sort(
        key=lambda row: (
            not (row["a6_pass"] and row["r6_guard"] and row["e6_guard"] and row["harm_guard"]),
            row["a6_best_mean"],
            -row["a6_owner_soft_ratio"],
        )
    )
    return screen_rows


def build_report(run_rows, visibility_rows, threshold_rows):
    all_methods = sorted({str(row["method"]) for row in run_rows})
    summary_by_key = {
        (problem_code, method_name): summarize_method_problem(run_rows, visibility_rows, problem_code, method_name)
        for problem_code in PROBLEMS
        for method_name in all_methods
    }
    screen_rows = build_combination_screen(summary_by_key)
    threshold_lookup = {
        (str(row["problem"]).upper(), str(row["threshold_mode"])): row
        for row in threshold_rows
    }

    report_lines = [
        "# Dynamic Soft Ownership Focused Sweep",
        "",
        f"- Problems: {', '.join(PROBLEMS)}",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        f"- MaxFEs: {MAX_FES}",
        "- Sweep axes: owner_delta_ratio_threshold in `strict / medium / loose / quantile70`, owner_eta in `0.1 / 0.2 / 0.3 / 0.5`",
        "- Sweep base config: `selective-hypergraph` (conflict damping held fixed off); current selective and selective-conflict are retained as baselines only.",
        "",
        "## 1. Threshold schedule",
        "",
        "| problem | strict | medium | loose | quantile70 | quantile_source | single-owner samples |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for problem_code in PROBLEMS:
        quantile_row = threshold_lookup[(problem_code, "quantile")]
        report_lines.append(
            "| {problem} | {strict_value} | {medium_value} | {loose_value} | {quantile_value} | {quantile_source} | {sample_count} |".format(
                problem=problem_code,
                strict_value=format_metric(threshold_lookup[(problem_code, "strict")]["threshold_value"]),
                medium_value=format_metric(threshold_lookup[(problem_code, "medium")]["threshold_value"]),
                loose_value=format_metric(threshold_lookup[(problem_code, "loose")]["threshold_value"]),
                quantile_value=format_metric(quantile_row["threshold_value"]),
                quantile_source=str(quantile_row["threshold_source"]),
                sample_count=int(quantile_row["sample_count"]),
            )
        )

    report_lines.extend(
        [
            "",
            "## 2. Baseline snapshot",
            "",
            "| problem | method | best_mean | final_mean | owner_soft_ratio | owner_soft_updated_ratio | mean_update_magnitude | harmful_update_proxy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for problem_code in PROBLEMS:
        for method_name in ["no-coordination", "selective-current", "selective-current-conflict"]:
            summary = summary_by_key[(problem_code, method_name)]
            report_lines.append(
                "| {problem} | {method} | {best_mean} | {final_mean} | {owner_soft_ratio} | {owner_soft_updated_ratio} | {update_mag} | {harmful} |".format(
                    problem=problem_code,
                    method=method_name,
                    best_mean=format_metric(summary["best_mean"]),
                    final_mean=format_metric(summary["final_mean"]),
                    owner_soft_ratio=format_ratio(summary["owner_soft_ratio"]),
                    owner_soft_updated_ratio=format_ratio(summary["owner_soft_updated_ratio"]),
                    update_mag=format_metric(summary["mean_update_magnitude"]),
                    harmful=format_ratio(summary["harmful_update_proxy"]),
                )
            )

    report_lines.extend(
        [
            "",
            "## 3. Guardrail screen",
            "",
            "| config | A6 best vs no-coord | A6 final vs selective | A6 owner_soft | A6 owner_soft_updated | R6 vs no-coord | E6 vs no-coord | max harmful | hard guards | owner band |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in screen_rows:
        hard_guards = "yes" if (row["a6_pass"] and row["r6_guard"] and row["e6_guard"] and row["harm_guard"]) else "no"
        owner_band = "yes" if row["a6_owner_soft_in_band"] else "no"
        report_lines.append(
            "| {method} | {a6_vs_no_coord} | {a6_final_vs_selective} | {a6_owner_soft} | {a6_owner_soft_updated} | {r6_vs_no_coord} | {e6_vs_no_coord} | {max_harmful} | {hard_guards} | {owner_band} |".format(
                method=row["method"],
                a6_vs_no_coord=format_percent_delta(row["a6_vs_no_coord"]),
                a6_final_vs_selective=format_percent_delta(row["a6_final_vs_selective"]),
                a6_owner_soft=format_ratio(row["a6_owner_soft_ratio"]),
                a6_owner_soft_updated=format_ratio(row["a6_owner_soft_updated_ratio"]),
                r6_vs_no_coord=format_percent_delta(row["r6_vs_no_coord"]),
                e6_vs_no_coord=format_percent_delta(row["e6_vs_no_coord"]),
                max_harmful=format_ratio(row["max_harmful_update_proxy"]),
                hard_guards=hard_guards,
                owner_band=owner_band,
            )
        )

    for problem_code in PROBLEMS:
        report_lines.extend(
            [
                "",
                f"## 4. Problem Detail: {problem_code}",
                "",
                "| config | threshold_mode | owner_eta | threshold_value | best_mean | final_mean | vs no-coord | vs current selective | owner_soft | owner_soft_updated | owner_hit_rate | freeze | multi_support | mean_update_magnitude | harmful |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        no_coord_best = to_float(summary_by_key[(problem_code, "no-coordination")]["best_mean"])
        selective_best = min(
            to_float(summary_by_key[(problem_code, "selective-current")]["best_mean"]),
            to_float(summary_by_key[(problem_code, "selective-current-conflict")]["best_mean"]),
        )
        problem_rows = [row for row in screen_rows]
        for row in problem_rows:
            summary = summary_by_key[(problem_code, row["method"])]
            report_lines.append(
                "| {method} | {threshold_mode} | {owner_eta} | {threshold_value} | {best_mean} | {final_mean} | {vs_no_coord} | {vs_selective} | {owner_soft} | {owner_soft_updated} | {owner_hit_rate} | {freeze_ratio} | {multi_support_ratio} | {update_mag} | {harmful} |".format(
                    method=row["method"],
                    threshold_mode=str(summary["threshold_mode"]),
                    owner_eta=f"{to_float(summary['owner_eta']):.3f}",
                    threshold_value=format_metric(summary["threshold_value"]),
                    best_mean=format_metric(summary["best_mean"]),
                    final_mean=format_metric(summary["final_mean"]),
                    vs_no_coord=format_percent_delta((to_float(summary["best_mean"]) - no_coord_best) / no_coord_best),
                    vs_selective=format_percent_delta((to_float(summary["best_mean"]) - selective_best) / selective_best),
                    owner_soft=format_ratio(summary["owner_soft_ratio"]),
                    owner_soft_updated=format_ratio(summary["owner_soft_updated_ratio"]),
                    owner_hit_rate=format_ratio(summary["owner_soft_update_hit_rate"]),
                    freeze_ratio=format_ratio(summary["freeze_ratio"]),
                    multi_support_ratio=format_ratio(summary["multi_support_ratio"]),
                    update_mag=format_metric(summary["mean_update_magnitude"]),
                    harmful=format_ratio(summary["harmful_update_proxy"]),
                )
            )

    report_lines.extend(
        [
            "",
            "## 5. Readout",
            "",
            "- `A6` is the only focus problem with nonzero single-owner evidence in the baseline sweep source; `R6` and `E6` quantile thresholds fall back to strict because their single-owner sample count is zero.",
            "- Hard-guard screening uses: `A6 <= no-coordination`, `R6 within 0.2% of no-coordination`, `E6 <= no-coordination`, and `max harmful_update_proxy <= 0.01`.",
            "- The practical tradeoff to watch is whether loosening the threshold lifts `A6 owner_soft_ratio` and `owner_soft_updated_ratio` without giving back `R6` or the existing `E6` gain.",
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- visibility details: `{VISIBILITY_DETAILS_PATH.as_posix()}`",
            f"- arbitration details: `{ARBITRATION_DETAILS_PATH.as_posix()}`",
            f"- threshold details: `{THRESHOLD_DETAILS_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    return "\n".join(report_lines) + "\n"


def main():
    run_rows = []
    visibility_rows = []
    arbitration_rows = []
    problem_inputs_map = {problem_code: build_problem_inputs(problem_code) for problem_code in PROBLEMS}

    for method_name, config_path in BASELINE_CONFIGS:
        config = load_method_config(config_path)
        for problem_code in PROBLEMS:
            for seed in SEEDS:
                print(f"[baseline] {method_name} problem={problem_code} seed={seed}")
                run_row, visibility_row, case_arbitration_rows = run_one_case(
                    method_name,
                    config,
                    problem_code,
                    seed,
                    problem_inputs_map[problem_code],
                    threshold_mode="baseline",
                    threshold_value=float(getattr(config, "shared_variable_owner_min_delta_ratio", float("nan"))),
                    threshold_source="baseline_config",
                    owner_eta=float(getattr(config, "shared_variable_owner_soft_eta", float("nan"))),
                    is_baseline=True,
                )
                run_rows.append(run_row)
                visibility_rows.append(visibility_row)
                arbitration_rows.extend(case_arbitration_rows)
                print(
                    "    done status={status} best_error={best_error:.6e} owner_soft={owner_soft:.3f} owner_soft_updated={owner_soft_updated:.3f}".format(
                        status=run_row["status"],
                        best_error=to_float(run_row["best_error"]),
                        owner_soft=to_float(visibility_row["owner_soft_ratio"]),
                        owner_soft_updated=to_float(visibility_row["owner_soft_updated_ratio"]),
                    )
                )

    base_sweep_config = load_method_config(CONFIG_ROOT / "selective-hypergraph.json")
    strict_threshold = float(base_sweep_config.shared_variable_owner_min_delta_ratio)
    thresholds_by_problem, threshold_rows = build_quantile_thresholds_from_baseline(arbitration_rows, strict_threshold)

    for threshold_mode, factor in THRESHOLD_MODES:
        for owner_eta in OWNER_ETAS:
            method_name = f"sweep-{threshold_mode}-eta{owner_eta:.1f}"
            for problem_code in PROBLEMS:
                if threshold_mode == "quantile":
                    threshold_value = float(thresholds_by_problem[problem_code]["quantile"])
                    threshold_source = str(thresholds_by_problem[problem_code]["quantile_source"])
                else:
                    threshold_value = float(strict_threshold * factor)
                    threshold_source = f"strict_x{factor:.2f}"
                config = replace(
                    base_sweep_config,
                    shared_variable_owner_soft_eta=float(owner_eta),
                    shared_variable_owner_min_delta_ratio=float(threshold_value),
                ).normalized()
                for seed in SEEDS:
                    print(
                        f"[sweep] {method_name} problem={problem_code} seed={seed} threshold={threshold_value:.6e}"
                    )
                    run_row, visibility_row, case_arbitration_rows = run_one_case(
                        method_name,
                        config,
                        problem_code,
                        seed,
                        problem_inputs_map[problem_code],
                        threshold_mode=threshold_mode,
                        threshold_value=threshold_value,
                        threshold_source=threshold_source,
                        owner_eta=owner_eta,
                        is_baseline=False,
                    )
                    run_rows.append(run_row)
                    visibility_rows.append(visibility_row)
                    arbitration_rows.extend(case_arbitration_rows)
                    print(
                        "    done status={status} best_error={best_error:.6e} owner_soft={owner_soft:.3f} owner_soft_updated={owner_soft_updated:.3f}".format(
                            status=run_row["status"],
                            best_error=to_float(run_row["best_error"]),
                            owner_soft=to_float(visibility_row["owner_soft_ratio"]),
                            owner_soft_updated=to_float(visibility_row["owner_soft_updated_ratio"]),
                        )
                    )

    run_rows.sort(key=lambda row: (row["problem"], str(row["method"]), int(row["seed"])))
    visibility_rows.sort(key=lambda row: (row["problem"], str(row["method"]), int(row["seed"])))
    arbitration_rows.sort(
        key=lambda row: (
            row["problem"],
            str(row["method"]),
            int(row["seed"]),
            int(row["cycle_id"]),
            int(row["var_id"]),
        )
    )
    threshold_rows.sort(key=lambda row: (row["problem"], row["threshold_mode"]))
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(VISIBILITY_DETAILS_PATH, VISIBILITY_FIELDNAMES, visibility_rows)
    write_csv(ARBITRATION_DETAILS_PATH, ARBITRATION_FIELDNAMES, arbitration_rows)
    write_csv(THRESHOLD_DETAILS_PATH, THRESHOLD_FIELDNAMES, threshold_rows)
    REPORT_PATH.write_text(build_report(run_rows, visibility_rows, threshold_rows), encoding="utf-8")
    print(f"run details -> {RUN_DETAILS_PATH}")
    print(f"visibility details -> {VISIBILITY_DETAILS_PATH}")
    print(f"arbitration details -> {ARBITRATION_DETAILS_PATH}")
    print(f"threshold details -> {THRESHOLD_DETAILS_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
