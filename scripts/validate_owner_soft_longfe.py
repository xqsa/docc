import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "owner_soft_longfe_runs"
DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [5000, 10000]
QUANTILE_LEVEL = 0.70
BASELINE_CONFIGS = [
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("selective-hypergraph", CONFIG_ROOT / "selective-hypergraph.json"),
    ("selective-hypergraph-conflict", CONFIG_ROOT / "selective-hypergraph-conflict.json"),
]
OWNER_SOFT_METHOD = "quantile-owner-soft-eta0.2"
OWNER_SOFT_ETA = 0.2

RUN_DETAILS_PATH = ARTIFACTS_ROOT / "owner_soft_longfe_run_details.csv"
VISIBILITY_DETAILS_PATH = ARTIFACTS_ROOT / "owner_soft_longfe_visibility.csv"
ARBITRATION_DETAILS_PATH = ARTIFACTS_ROOT / "owner_soft_longfe_arbitration.csv"
THRESHOLD_DETAILS_PATH = ARTIFACTS_ROOT / "owner_soft_longfe_thresholds.csv"
REPORT_PATH = ARTIFACTS_ROOT / "owner_soft_longfe_report.md"

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
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "owner_eta",
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
    "tfes",
    "method",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "owner_eta",
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
    "owner_soft_update_count",
    "owner_soft_update_magnitude_mean",
    "owner_soft_unique_var_count",
    "owner_soft_followed_by_best_improvement_count",
    "owner_soft_overwritten_count",
    "owner_soft_top_vars_json",
]

ARBITRATION_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "owner_eta",
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
    "post_coordination_best_improved",
    "owner_soft_followed_by_best_improvement",
    "owner_soft_overwritten",
]

THRESHOLD_FIELDNAMES = [
    "problem",
    "tfes",
    "threshold_mode",
    "threshold_value",
    "threshold_source",
    "sample_count",
    "strict_threshold",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate owner_soft long-FE behavior.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    return parser.parse_args()


def load_method_config(config_path):
    return hcc_es.load_info_aware_nda_config(config_path, enable=False)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_or_nan(values):
    numeric = [to_float(value) for value in values]
    finite = np.asarray([value for value in numeric if np.isfinite(value)], dtype=float)
    return float(np.mean(finite)) if finite.size else float("nan")


def std_or_nan(values):
    numeric = [to_float(value) for value in values]
    finite = np.asarray([value for value in numeric if np.isfinite(value)], dtype=float)
    return float(np.std(finite)) if finite.size else float("nan")


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


def run_one_case(method_name, config, problem_code, seed, tfes, problem_inputs, threshold_mode, threshold_value, threshold_source):
    output_dir = RUNS_ROOT / f"tfes-{int(tfes)}" / method_name
    default_visibility = build_default_visibility(problem_inputs, config)
    default_summary = build_default_coordination_summary(problem_inputs, config)
    try:
        result = hcc_es.run_problem_seed_task(
            problem_code,
            seed,
            tfes,
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
        owner_top_vars = coordination_summary.get("owner_soft_top_vars", [])
        arbitration_rows = [
            {
                **row,
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(getattr(config, "shared_variable_owner_soft_eta", float("nan"))),
                "coordination_mode": str(coordination_summary["coordination_mode"]),
            }
            for row in metadata.get("shared_variable_fusion_rows", [])
        ]
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(getattr(config, "shared_variable_owner_soft_eta", float("nan"))),
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
                "tfes": int(tfes),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(getattr(config, "shared_variable_owner_soft_eta", float("nan"))),
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
                "owner_soft_update_count": int(coordination_summary.get("owner_soft_update_count", 0)),
                "owner_soft_update_magnitude_mean": to_float(coordination_summary.get("owner_soft_update_magnitude_mean")),
                "owner_soft_unique_var_count": int(coordination_summary.get("owner_soft_unique_var_count", 0)),
                "owner_soft_followed_by_best_improvement_count": int(
                    coordination_summary.get("owner_soft_followed_by_best_improvement_count", 0)
                ),
                "owner_soft_overwritten_count": int(coordination_summary.get("owner_soft_overwritten_count", 0)),
                "owner_soft_top_vars_json": json.dumps(owner_top_vars, ensure_ascii=False),
            },
            arbitration_rows,
        )
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(getattr(config, "shared_variable_owner_soft_eta", float("nan"))),
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
                "tfes": int(tfes),
                "method": str(method_name),
                "threshold_mode": str(threshold_mode),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "owner_eta": float(getattr(config, "shared_variable_owner_soft_eta", float("nan"))),
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
                "owner_soft_update_count": 0,
                "owner_soft_update_magnitude_mean": 0.0,
                "owner_soft_unique_var_count": 0,
                "owner_soft_followed_by_best_improvement_count": 0,
                "owner_soft_overwritten_count": 0,
                "owner_soft_top_vars_json": "[]",
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


def build_quantile_thresholds_from_baseline(arbitration_rows, strict_threshold, tfes, problems):
    threshold_rows = []
    thresholds_by_problem = {}
    for problem_code in problems:
        values = [
            float(row["owner_delta_ratio"])
            for row in arbitration_rows
            if str(row["problem"]).upper() == str(problem_code).upper()
            and int(row["tfes"]) == int(tfes)
            and str(row["method"]) == "selective-hypergraph"
            and int(row.get("positive_proposal_count", 0) or 0) == 1
            and np.isfinite(float(row.get("owner_delta_ratio", float("nan"))))
        ]
        if values:
            threshold_value = float(np.quantile(np.asarray(values, dtype=float), QUANTILE_LEVEL))
            threshold_source = f"selective-hypergraph_q{int(QUANTILE_LEVEL * 100)}_tfes{int(tfes)}"
        else:
            threshold_value = float(strict_threshold)
            threshold_source = "strict_fallback_no_single_owner"
        thresholds_by_problem[str(problem_code).upper()] = {
            "quantile": float(threshold_value),
            "quantile_source": str(threshold_source),
            "sample_count": int(len(values)),
        }
        threshold_rows.append(
            {
                "problem": str(problem_code).upper(),
                "tfes": int(tfes),
                "threshold_mode": "quantile",
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "sample_count": int(len(values)),
                "strict_threshold": float(strict_threshold),
            }
        )
    return thresholds_by_problem, threshold_rows


def summarize_method_problem_tfes(run_rows, visibility_rows, problem_code, tfes, method_name):
    method_runs = [
        row
        for row in run_rows
        if str(row["problem"]).upper() == str(problem_code).upper()
        and int(row["tfes"]) == int(tfes)
        and str(row["method"]) == method_name
        and str(row["status"]) == "ok"
    ]
    method_visibility_rows = [
        row
        for row in visibility_rows
        if str(row["problem"]).upper() == str(problem_code).upper()
        and int(row["tfes"]) == int(tfes)
        and str(row["method"]) == method_name
    ]
    top_var_counter = {}
    for row in method_visibility_rows:
        try:
            top_vars = json.loads(str(row.get("owner_soft_top_vars_json", "[]")))
        except json.JSONDecodeError:
            top_vars = []
        for top_var in top_vars[:3]:
            var_id = int(top_var.get("var_id", -1))
            if var_id < 0:
                continue
            top_var_counter[var_id] = top_var_counter.get(var_id, 0) + 1
    top_var_signature = ", ".join(
        f"{var_id}x{count}" for var_id, count in sorted(top_var_counter.items(), key=lambda item: (-item[1], item[0]))[:3]
    )
    return {
        "best_mean": mean_or_nan([row["best_error"] for row in method_runs]),
        "best_std": std_or_nan([row["best_error"] for row in method_runs]),
        "final_mean": mean_or_nan([row["final_error"] for row in method_runs]),
        "final_std": std_or_nan([row["final_error"] for row in method_runs]),
        "owner_soft_update_count_mean": mean_or_nan([row["owner_soft_update_count"] for row in method_visibility_rows]),
        "owner_soft_update_magnitude_mean": mean_or_nan([row["owner_soft_update_magnitude_mean"] for row in method_visibility_rows]),
        "owner_soft_unique_var_count_mean": mean_or_nan([row["owner_soft_unique_var_count"] for row in method_visibility_rows]),
        "owner_soft_followed_by_best_improvement_count_mean": mean_or_nan(
            [row["owner_soft_followed_by_best_improvement_count"] for row in method_visibility_rows]
        ),
        "owner_soft_overwritten_count_mean": mean_or_nan(
            [row["owner_soft_overwritten_count"] for row in method_visibility_rows]
        ),
        "top_owner_vars": top_var_signature,
    }


def build_report(run_rows, visibility_rows, threshold_rows, problems, seeds, tfes_values):
    summary_by_key = {}
    for tfes in tfes_values:
        for problem_code in problems:
            for method_name in [name for name, _ in BASELINE_CONFIGS] + [OWNER_SOFT_METHOD]:
                summary_by_key[(str(problem_code).upper(), int(tfes), method_name)] = summarize_method_problem_tfes(
                    run_rows,
                    visibility_rows,
                    problem_code,
                    tfes,
                    method_name,
                )

    lines = [
        "# Owner Soft Long-FE Validation",
        "",
        f"- Problems: {', '.join(str(problem).upper() for problem in problems)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- MaxFEs: {', '.join(str(tfes) for tfes in tfes_values)}",
        "- Methods: no-coordination / selective-hypergraph / selective-hypergraph-conflict / quantile-owner-soft-eta0.2",
        "",
        "## 1. Quantile threshold schedule",
        "",
        "| tfes | problem | threshold | source | single-owner samples |",
        "| ---: | --- | ---: | --- | ---: |",
    ]
    for row in threshold_rows:
        lines.append(
            "| {tfes} | {problem} | {threshold} | {source} | {samples} |".format(
                tfes=int(row["tfes"]),
                problem=str(row["problem"]).upper(),
                threshold=format_metric(row["threshold_value"]),
                source=str(row["threshold_source"]),
                samples=int(row["sample_count"]),
            )
        )

    for tfes in tfes_values:
        lines.extend(
            [
                "",
                f"## 2. TFEs = {int(tfes)}",
                "",
                "| problem | method | best_mean | best_std | final_mean | final_std | owner_soft_updates | owner_soft_hit | owner_soft_overwritten | top_vars |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for problem_code in problems:
            for method_name in [name for name, _ in BASELINE_CONFIGS] + [OWNER_SOFT_METHOD]:
                summary = summary_by_key[(str(problem_code).upper(), int(tfes), method_name)]
                lines.append(
                    "| {problem} | {method} | {best_mean} | {best_std} | {final_mean} | {final_std} | {updates} | {hits} | {overwritten} | {top_vars} |".format(
                        problem=str(problem_code).upper(),
                        method=method_name,
                        best_mean=format_metric(summary["best_mean"]),
                        best_std=format_metric(summary["best_std"]),
                        final_mean=format_metric(summary["final_mean"]),
                        final_std=format_metric(summary["final_std"]),
                        updates=format_ratio(summary["owner_soft_update_count_mean"]),
                        hits=format_ratio(summary["owner_soft_followed_by_best_improvement_count_mean"]),
                        overwritten=format_ratio(summary["owner_soft_overwritten_count_mean"]),
                        top_vars=summary["top_owner_vars"] or "-",
                    )
                )

    lines.extend(["", "## 3. Readout", ""])
    for tfes in tfes_values:
        readout_parts = []
        if "A6" in problems:
            a6_owner = summary_by_key[("A6", int(tfes), OWNER_SOFT_METHOD)]
            a6_selective = summary_by_key[("A6", int(tfes), "selective-hypergraph")]
            a6_no_coord = summary_by_key[("A6", int(tfes), "no-coordination")]
            if np.isfinite(to_float(a6_selective["best_mean"])) and abs(to_float(a6_selective["best_mean"])) > 0:
                readout_parts.append(
                    "`A6 owner-soft vs selective(best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(a6_owner["best_mean"]) - to_float(a6_selective["best_mean"])) / to_float(a6_selective["best_mean"])
                        )
                    )
                )
            if np.isfinite(to_float(a6_no_coord["best_mean"])) and abs(to_float(a6_no_coord["best_mean"])) > 0:
                readout_parts.append(
                    "`A6 owner-soft vs no-coord(best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(a6_owner["best_mean"]) - to_float(a6_no_coord["best_mean"])) / to_float(a6_no_coord["best_mean"])
                        )
                    )
                )
            readout_parts.append("`A6 owner_soft_updates(mean)`={value}".format(value=format_ratio(a6_owner["owner_soft_update_count_mean"])))
            readout_parts.append(
                "`A6 owner_soft_hit(mean)`={value}".format(
                    value=format_ratio(a6_owner["owner_soft_followed_by_best_improvement_count_mean"])
                )
            )
            readout_parts.append(
                "`A6 owner_soft_overwritten(mean)`={value}".format(
                    value=format_ratio(a6_owner["owner_soft_overwritten_count_mean"])
                )
            )
        if "E6" in problems:
            e6_owner = summary_by_key[("E6", int(tfes), OWNER_SOFT_METHOD)]
            e6_selective = summary_by_key[("E6", int(tfes), "selective-hypergraph")]
            if np.isfinite(to_float(e6_selective["best_mean"])) and abs(to_float(e6_selective["best_mean"])) > 0:
                readout_parts.append(
                    "`E6 owner-soft vs selective(best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(e6_owner["best_mean"]) - to_float(e6_selective["best_mean"])) / to_float(e6_selective["best_mean"])
                        )
                    )
                )
        if "R6" in problems:
            r6_owner = summary_by_key[("R6", int(tfes), OWNER_SOFT_METHOD)]
            r6_no_coord = summary_by_key[("R6", int(tfes), "no-coordination")]
            if np.isfinite(to_float(r6_no_coord["best_mean"])) and abs(to_float(r6_no_coord["best_mean"])) > 0:
                readout_parts.append(
                    "`R6 owner-soft vs no-coord(best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(r6_owner["best_mean"]) - to_float(r6_no_coord["best_mean"])) / to_float(r6_no_coord["best_mean"])
                        )
                    )
                )
        if readout_parts:
            lines.append("- TFEs={tfes}: {parts}.".format(tfes=int(tfes), parts=", ".join(readout_parts)))

    lines.extend(
        [
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
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    problems = [str(problem).upper() for problem in args.problems]
    seeds = [int(seed) for seed in args.seeds]
    tfes_values = [int(tfes) for tfes in args.tfes]
    problem_inputs_map = {problem: build_problem_inputs(problem) for problem in problems}

    run_rows = []
    visibility_rows = []
    arbitration_rows = []
    threshold_rows = []
    base_owner_config = load_method_config(CONFIG_ROOT / "selective-hypergraph.json")
    strict_threshold = float(base_owner_config.shared_variable_owner_min_delta_ratio)

    for tfes in tfes_values:
        thresholds_by_problem = {}
        for method_name, config_path in BASELINE_CONFIGS:
            config = load_method_config(config_path)
            for problem_code in problems:
                for seed in seeds:
                    run_row, visibility_row, case_arbitration_rows = run_one_case(
                        method_name,
                        config,
                        problem_code,
                        seed,
                        tfes,
                        problem_inputs_map[problem_code],
                        threshold_mode="baseline",
                        threshold_value=float(getattr(config, "shared_variable_owner_min_delta_ratio", float("nan"))),
                        threshold_source="config",
                    )
                    run_rows.append(run_row)
                    visibility_rows.append(visibility_row)
                    arbitration_rows.extend(case_arbitration_rows)
                    print(
                        "baseline tfes={tfes} {problem} {method} seed={seed} status={status} best={best}".format(
                            tfes=int(tfes),
                            problem=problem_code,
                            method=method_name,
                            seed=int(seed),
                            status=run_row["status"],
                            best=format_metric(run_row["best_error"]),
                        )
                    )
        thresholds_by_problem, tfes_threshold_rows = build_quantile_thresholds_from_baseline(
            arbitration_rows,
            strict_threshold,
            tfes,
            problems,
        )
        threshold_rows.extend(tfes_threshold_rows)

        for problem_code in problems:
            threshold_value = float(thresholds_by_problem[problem_code]["quantile"])
            threshold_source = str(thresholds_by_problem[problem_code]["quantile_source"])
            owner_config = replace(
                base_owner_config,
                shared_variable_owner_soft_eta=float(OWNER_SOFT_ETA),
                shared_variable_owner_min_delta_ratio=float(threshold_value),
            )
            for seed in seeds:
                run_row, visibility_row, case_arbitration_rows = run_one_case(
                    OWNER_SOFT_METHOD,
                    owner_config,
                    problem_code,
                    seed,
                    tfes,
                    problem_inputs_map[problem_code],
                    threshold_mode="quantile",
                    threshold_value=threshold_value,
                    threshold_source=threshold_source,
                )
                run_rows.append(run_row)
                visibility_rows.append(visibility_row)
                arbitration_rows.extend(case_arbitration_rows)
                print(
                    "owner tfes={tfes} {problem} seed={seed} status={status} best={best} updates={updates}".format(
                        tfes=int(tfes),
                        problem=problem_code,
                        seed=int(seed),
                        status=run_row["status"],
                        best=format_metric(run_row["best_error"]),
                        updates=format_ratio(visibility_row["owner_soft_update_count"]),
                    )
                )

    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(VISIBILITY_DETAILS_PATH, VISIBILITY_FIELDNAMES, visibility_rows)
    write_csv(ARBITRATION_DETAILS_PATH, ARBITRATION_FIELDNAMES, arbitration_rows)
    write_csv(THRESHOLD_DETAILS_PATH, THRESHOLD_FIELDNAMES, threshold_rows)
    build_report(run_rows, visibility_rows, threshold_rows, problems, seeds, tfes_values)
    print(f"run details -> {RUN_DETAILS_PATH}")
    print(f"visibility details -> {VISIBILITY_DETAILS_PATH}")
    print(f"arbitration details -> {ARBITRATION_DETAILS_PATH}")
    print(f"threshold details -> {THRESHOLD_DETAILS_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
