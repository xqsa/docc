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
RUNS_ROOT = ARTIFACTS_ROOT / "validated_hypergraph_coordination_runs"
DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [1000, 5000]
QUANTILE_LEVEL = 0.70
OWNER_SOFT_METHOD = "quantile-owner-soft-eta0.2"
OWNER_SOFT_ETA = 0.2
BASELINE_CONFIGS = [
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("selective-hypergraph", CONFIG_ROOT / "selective-hypergraph.json"),
    ("selective-hypergraph-conflict", CONFIG_ROOT / "selective-hypergraph-conflict.json"),
]
VALIDATED_CONFIGS = [
    ("validated-selective", CONFIG_ROOT / "validated-selective.json"),
    ("validated-selective-conflict", CONFIG_ROOT / "validated-selective-conflict.json"),
]
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "validated_hypergraph_coordination_run_details.csv"
TRACE_PATH = ARTIFACTS_ROOT / "validated_hypergraph_coordination_trace.csv"
REPORT_PATH = ARTIFACTS_ROOT / "validated_hypergraph_coordination_report.md"

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
    "coordination_mode",
    "conflict_damping",
    "validation_enabled",
    "validation_attempt_count",
    "validation_accept_count",
    "validation_reject_count",
    "validation_accept_rate",
    "validation_extra_fe_used",
    "validation_extra_fe_ratio",
    "accepted_owner_soft_count",
    "rejected_owner_soft_count",
    "accepted_multi_support_count",
    "rejected_multi_support_count",
    "mean_accepted_fitness_delta",
    "mean_rejected_fitness_delta",
    "final_error",
    "best_error",
    "status",
    "runtime",
]
TRACE_FIELDNAMES = list(hcc_es.VALIDATED_COORDINATION_TRACE_FIELDNAMES)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate validated coordination artifacts.")
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
    finite = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
    return float(np.mean(finite)) if finite.size else float("nan")


def std_or_nan(values):
    finite = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
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
            "threshold_value": float(threshold_value),
            "threshold_source": str(threshold_source),
            "sample_count": int(len(values)),
        }
        threshold_rows.append(
            {
                "problem": str(problem_code).upper(),
                "tfes": int(tfes),
                "threshold_value": float(threshold_value),
                "threshold_source": str(threshold_source),
                "sample_count": int(len(values)),
            }
        )
    return thresholds_by_problem, threshold_rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run_one_case(method_name, config, problem_code, seed, tfes, problem_inputs, threshold_mode, threshold_value, threshold_source):
    output_dir = RUNS_ROOT / f"tfes-{int(tfes)}" / method_name
    default_visibility = build_default_visibility(problem_inputs, config)
    default_validation_summary = hcc_es.summarize_validated_coordination_rows([], total_max_fes=tfes)
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
        validated_summary = dict(default_validation_summary)
        validated_summary.update(metadata.get("validated_coordination_summary", {}))
        validated_summary.update(diagnostics_payload.get("validated_coordination_summary", {}))
        trace_rows = [
            {
                **row,
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
            }
            for row in metadata.get("validated_coordination_rows", [])
        ]
        arbitration_rows = [
            {
                **row,
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
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
                "coordination_mode": str(visibility["coordination_mode"]),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "validation_enabled": bool(getattr(config, "enable_validated_coordination", False)),
                "validation_attempt_count": int(validated_summary.get("validation_attempt_count", 0)),
                "validation_accept_count": int(validated_summary.get("validation_accept_count", 0)),
                "validation_reject_count": int(validated_summary.get("validation_reject_count", 0)),
                "validation_accept_rate": float(validated_summary.get("validation_accept_rate", 0.0)),
                "validation_extra_fe_used": int(validated_summary.get("validation_extra_fe_used", 0)),
                "validation_extra_fe_ratio": float(validated_summary.get("validation_extra_fe_ratio", 0.0)),
                "accepted_owner_soft_count": int(validated_summary.get("accepted_owner_soft_count", 0)),
                "rejected_owner_soft_count": int(validated_summary.get("rejected_owner_soft_count", 0)),
                "accepted_multi_support_count": int(validated_summary.get("accepted_multi_support_count", 0)),
                "rejected_multi_support_count": int(validated_summary.get("rejected_multi_support_count", 0)),
                "mean_accepted_fitness_delta": to_float(validated_summary.get("mean_accepted_fitness_delta")),
                "mean_rejected_fitness_delta": to_float(validated_summary.get("mean_rejected_fitness_delta")),
                "final_error": to_float(detail.get("final_fitness")),
                "best_error": to_float(detail.get("best_fitness")),
                "status": str(detail.get("status", "")),
                "runtime": to_float(detail.get("runtime")),
            },
            trace_rows,
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
                "coordination_mode": str(default_visibility["coordination_mode"]),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "validation_enabled": bool(getattr(config, "enable_validated_coordination", False)),
                "validation_attempt_count": 0,
                "validation_accept_count": 0,
                "validation_reject_count": 0,
                "validation_accept_rate": 0.0,
                "validation_extra_fe_used": 0,
                "validation_extra_fe_ratio": 0.0,
                "accepted_owner_soft_count": 0,
                "rejected_owner_soft_count": 0,
                "accepted_multi_support_count": 0,
                "rejected_multi_support_count": 0,
                "mean_accepted_fitness_delta": float("nan"),
                "mean_rejected_fitness_delta": float("nan"),
                "final_error": float("nan"),
                "best_error": float("nan"),
                "status": f"error: {exc}",
                "runtime": float("nan"),
            },
            [],
            [],
        )


def summarize_performance(run_rows, problem_code, tfes, method_name):
    rows = [
        row
        for row in run_rows
        if str(row["problem"]).upper() == str(problem_code).upper()
        and int(row["tfes"]) == int(tfes)
        and str(row["method"]) == method_name
        and str(row["status"]) == "ok"
    ]
    return {
        "run_count": int(len(rows)),
        "best_mean": mean_or_nan([row["best_error"] for row in rows]),
        "best_std": std_or_nan([row["best_error"] for row in rows]),
        "final_mean": mean_or_nan([row["final_error"] for row in rows]),
        "final_std": std_or_nan([row["final_error"] for row in rows]),
    }


def summarize_validation(run_rows, trace_rows, problem_code, tfes, method_name):
    matched_runs = [
        row
        for row in run_rows
        if str(row["problem"]).upper() == str(problem_code).upper()
        and int(row["tfes"]) == int(tfes)
        and str(row["method"]) == method_name
        and str(row["status"]) == "ok"
    ]
    matched_trace_rows = [
        row
        for row in trace_rows
        if str(row["problem"]).upper() == str(problem_code).upper()
        and int(row["tfes"]) == int(tfes)
        and str(row["method"]) == method_name
    ]
    attempted_rows = [row for row in matched_trace_rows if bool(row.get("validation_attempted"))]
    accepted_rows = [row for row in attempted_rows if bool(row.get("validation_accepted"))]
    rejected_rows = [row for row in attempted_rows if not bool(row.get("validation_accepted"))]
    run_count = max(1, len(matched_runs))
    return {
        "validation_attempt_count": int(len(attempted_rows)),
        "validation_accept_count": int(len(accepted_rows)),
        "validation_reject_count": int(len(rejected_rows)),
        "validation_accept_rate": float(len(accepted_rows) / len(attempted_rows)) if attempted_rows else 0.0,
        "validation_extra_fe_ratio": float(
            sum(int(row.get("validation_fe_used", 0) or 0) for row in matched_trace_rows) / float(run_count * max(1, int(tfes)))
        ),
        "accepted_owner_soft_count": int(sum(int(row.get("candidate_owner_soft_count", 0) or 0) for row in accepted_rows)),
        "rejected_owner_soft_count": int(sum(int(row.get("candidate_owner_soft_count", 0) or 0) for row in rejected_rows)),
        "accepted_multi_support_count": int(sum(int(row.get("candidate_multi_support_count", 0) or 0) for row in accepted_rows)),
        "rejected_multi_support_count": int(sum(int(row.get("candidate_multi_support_count", 0) or 0) for row in rejected_rows)),
        "freeze_count": int(sum(int(row.get("candidate_freeze_count", 0) or 0) for row in matched_trace_rows)),
        "mean_accepted_delta": mean_or_nan([row.get("fitness_delta") for row in accepted_rows]),
        "mean_rejected_delta": mean_or_nan([row.get("fitness_delta") for row in rejected_rows]),
    }


def build_report(run_rows, trace_rows, threshold_rows, problems, seeds, tfes_values):
    method_order = [name for name, _ in BASELINE_CONFIGS] + [OWNER_SOFT_METHOD] + [name for name, _ in VALIDATED_CONFIGS]
    performance_by_key = {}
    validation_by_key = {}
    for tfes in tfes_values:
        for problem_code in problems:
            for method_name in method_order:
                performance_by_key[(str(problem_code).upper(), int(tfes), method_name)] = summarize_performance(
                    run_rows,
                    problem_code,
                    tfes,
                    method_name,
                )
                validation_by_key[(str(problem_code).upper(), int(tfes), method_name)] = summarize_validation(
                    run_rows,
                    trace_rows,
                    problem_code,
                    tfes,
                    method_name,
                )

    lines = [
        "# Validated Hypergraph Coordination",
        "",
        f"- Problems: {', '.join(str(problem).upper() for problem in problems)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- MaxFEs: {', '.join(str(tfes) for tfes in tfes_values)}",
        "- Methods: no-coordination / selective-hypergraph / selective-hypergraph-conflict / quantile-owner-soft-eta0.2 / validated-selective / validated-selective-conflict",
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
                f"## 2. Performance @ TFEs = {int(tfes)}",
                "",
                "| problem | method | best_mean | best_std | final_mean | final_std | gap_vs_no_coordination | gap_vs_selective |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for problem_code in problems:
            no_coord = performance_by_key[(str(problem_code).upper(), int(tfes), "no-coordination")]
            selective = performance_by_key[(str(problem_code).upper(), int(tfes), "selective-hypergraph")]
            no_coord_best = to_float(no_coord["best_mean"])
            selective_best = to_float(selective["best_mean"])
            for method_name in method_order:
                summary = performance_by_key[(str(problem_code).upper(), int(tfes), method_name)]
                gap_vs_no_coord = (
                    (to_float(summary["best_mean"]) - no_coord_best) / no_coord_best
                    if np.isfinite(no_coord_best) and abs(no_coord_best) > 0.0
                    else float("nan")
                )
                gap_vs_selective = (
                    (to_float(summary["best_mean"]) - selective_best) / selective_best
                    if np.isfinite(selective_best) and abs(selective_best) > 0.0
                    else float("nan")
                )
                lines.append(
                    "| {problem} | {method} | {best_mean} | {best_std} | {final_mean} | {final_std} | {gap_vs_no_coord} | {gap_vs_selective} |".format(
                        problem=str(problem_code).upper(),
                        method=method_name,
                        best_mean=format_metric(summary["best_mean"]),
                        best_std=format_metric(summary["best_std"]),
                        final_mean=format_metric(summary["final_mean"]),
                        final_std=format_metric(summary["final_std"]),
                        gap_vs_no_coord=format_percent_delta(gap_vs_no_coord),
                        gap_vs_selective=format_percent_delta(gap_vs_selective),
                    )
                )

        lines.extend(
            [
                "",
                f"## 3. Validation @ TFEs = {int(tfes)}",
                "",
                "| problem | method | validation_attempt_count | accept_rate | extra_fe_ratio | mean_accepted_delta | mean_rejected_delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for problem_code in problems:
            for method_name in method_order:
                summary = validation_by_key[(str(problem_code).upper(), int(tfes), method_name)]
                lines.append(
                    "| {problem} | {method} | {attempts} | {accept_rate} | {extra_fe_ratio} | {accepted_delta} | {rejected_delta} |".format(
                        problem=str(problem_code).upper(),
                        method=method_name,
                        attempts=int(summary["validation_attempt_count"]),
                        accept_rate=format_ratio(summary["validation_accept_rate"]),
                        extra_fe_ratio=format_ratio(summary["validation_extra_fe_ratio"]),
                        accepted_delta=format_metric(summary["mean_accepted_delta"]),
                        rejected_delta=format_metric(summary["mean_rejected_delta"]),
                    )
                )

        lines.extend(
            [
                "",
                f"## 4. Accept / Reject Modes @ TFEs = {int(tfes)}",
                "",
                "| problem | method | owner_soft accepted | owner_soft rejected | multi_support accepted | multi_support rejected | freeze count |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for problem_code in problems:
            for method_name in method_order:
                summary = validation_by_key[(str(problem_code).upper(), int(tfes), method_name)]
                lines.append(
                    "| {problem} | {method} | {owner_soft_accepted} | {owner_soft_rejected} | {multi_support_accepted} | {multi_support_rejected} | {freeze_count} |".format(
                        problem=str(problem_code).upper(),
                        method=method_name,
                        owner_soft_accepted=int(summary["accepted_owner_soft_count"]),
                        owner_soft_rejected=int(summary["rejected_owner_soft_count"]),
                        multi_support_accepted=int(summary["accepted_multi_support_count"]),
                        multi_support_rejected=int(summary["rejected_multi_support_count"]),
                        freeze_count=int(summary["freeze_count"]),
                    )
                )

    lines.extend(["", "## 5. Readout", ""])
    for tfes in tfes_values:
        readout_parts = []
        if "A6" in problems:
            a6_validated = performance_by_key[("A6", int(tfes), "validated-selective")]
            a6_selective = performance_by_key[("A6", int(tfes), "selective-hypergraph")]
            validated_summary = validation_by_key[("A6", int(tfes), "validated-selective")]
            if np.isfinite(to_float(a6_selective["best_mean"])) and abs(to_float(a6_selective["best_mean"])) > 0.0:
                readout_parts.append(
                    "`validated-selective vs selective (A6 best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(a6_validated["best_mean"]) - to_float(a6_selective["best_mean"])) / to_float(a6_selective["best_mean"])
                        )
                    )
                )
            readout_parts.append("`validated accept_rate (A6)`={value}".format(value=format_ratio(validated_summary["validation_accept_rate"])))
            readout_parts.append("`validated extra_fe_ratio (A6)`={value}".format(value=format_ratio(validated_summary["validation_extra_fe_ratio"])))
        if "R6" in problems:
            r6_validated = performance_by_key[("R6", int(tfes), "validated-selective")]
            r6_no_coord = performance_by_key[("R6", int(tfes), "no-coordination")]
            if np.isfinite(to_float(r6_no_coord["best_mean"])) and abs(to_float(r6_no_coord["best_mean"])) > 0.0:
                readout_parts.append(
                    "`validated-selective vs no-coordination (R6 best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(r6_validated["best_mean"]) - to_float(r6_no_coord["best_mean"])) / to_float(r6_no_coord["best_mean"])
                        )
                    )
                )
        if "E6" in problems:
            e6_validated = performance_by_key[("E6", int(tfes), "validated-selective")]
            e6_selective = performance_by_key[("E6", int(tfes), "selective-hypergraph")]
            if np.isfinite(to_float(e6_selective["best_mean"])) and abs(to_float(e6_selective["best_mean"])) > 0.0:
                readout_parts.append(
                    "`validated-selective vs selective (E6 best)`={value}".format(
                        value=format_percent_delta(
                            (to_float(e6_validated["best_mean"]) - to_float(e6_selective["best_mean"])) / to_float(e6_selective["best_mean"])
                        )
                    )
                )
        owner_soft_accepts = sum(
            validation_by_key[(str(problem).upper(), int(tfes), "validated-selective")]["accepted_owner_soft_count"]
            + validation_by_key[(str(problem).upper(), int(tfes), "validated-selective-conflict")]["accepted_owner_soft_count"]
            for problem in problems
        )
        multi_support_accepts = sum(
            validation_by_key[(str(problem).upper(), int(tfes), "validated-selective")]["accepted_multi_support_count"]
            + validation_by_key[(str(problem).upper(), int(tfes), "validated-selective-conflict")]["accepted_multi_support_count"]
            for problem in problems
        )
        readout_parts.append("`accepted owner_soft count (all problems)`={value}".format(value=int(owner_soft_accepts)))
        readout_parts.append("`accepted multi_support count (all problems)`={value}".format(value=int(multi_support_accepts)))
        lines.append("- TFEs={tfes}: {parts}.".format(tfes=int(tfes), parts=", ".join(readout_parts)))

    validated_owner_soft_accepts = sum(
        int(row.get("candidate_owner_soft_count", 0) or 0)
        for row in trace_rows
        if bool(row.get("validation_attempted")) and bool(row.get("validation_accepted"))
    )
    validated_multi_support_accepts = sum(
        int(row.get("candidate_multi_support_count", 0) or 0)
        for row in trace_rows
        if bool(row.get("validation_attempted")) and bool(row.get("validation_accepted"))
    )
    lines.extend(
        [
            "",
            "## 6. Conclusion",
            "",
            f"1. A6 improves with validation only if `validated-selective` beats `selective-hypergraph` in the performance table above.",
            f"2. R6 is protected only if validated methods close the gap to or beat `no-coordination` while keeping rejects visible in the validation table.",
            f"3. E6 keeps selective gains only if validated methods remain near `selective-hypergraph` instead of drifting back toward `no-coordination`.",
            f"4. Owner-soft has real accepted signal only if accepted owner_soft count is nonzero; current aggregate accepted owner_soft count = {int(validated_owner_soft_accepts)}.",
            f"5. Multi-support is the stronger source only if accepted multi_support count materially exceeds accepted owner_soft count; current aggregate accepted multi_support count = {int(validated_multi_support_accepts)}.",
            "6. Validation FE is acceptable only if the validation table keeps extra_fe_ratio at or below 0.03.",
            "",
            "## Artifacts",
            "",
            f"- run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- trace: `{TRACE_PATH.as_posix()}`",
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
    trace_rows = []
    threshold_rows = []
    baseline_arbitration_rows = []
    base_owner_config = load_method_config(CONFIG_ROOT / "selective-hypergraph.json")
    strict_threshold = float(base_owner_config.shared_variable_owner_min_delta_ratio)

    for tfes in tfes_values:
        for method_name, config_path in BASELINE_CONFIGS:
            config = load_method_config(config_path)
            for problem_code in problems:
                for seed in seeds:
                    run_row, case_trace_rows, case_arbitration_rows = run_one_case(
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
                    trace_rows.extend(case_trace_rows)
                    baseline_arbitration_rows.extend(case_arbitration_rows)
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
            baseline_arbitration_rows,
            strict_threshold,
            tfes,
            problems,
        )
        threshold_rows.extend(tfes_threshold_rows)

        for problem_code in problems:
            owner_config = replace(
                base_owner_config,
                shared_variable_owner_soft_eta=float(OWNER_SOFT_ETA),
                shared_variable_owner_min_delta_ratio=float(thresholds_by_problem[problem_code]["threshold_value"]),
            )
            for seed in seeds:
                run_row, case_trace_rows, _ = run_one_case(
                    OWNER_SOFT_METHOD,
                    owner_config,
                    problem_code,
                    seed,
                    tfes,
                    problem_inputs_map[problem_code],
                    threshold_mode="quantile",
                    threshold_value=float(thresholds_by_problem[problem_code]["threshold_value"]),
                    threshold_source=str(thresholds_by_problem[problem_code]["threshold_source"]),
                )
                run_rows.append(run_row)
                trace_rows.extend(case_trace_rows)
                print(
                    "owner tfes={tfes} {problem} seed={seed} status={status} best={best}".format(
                        tfes=int(tfes),
                        problem=problem_code,
                        seed=int(seed),
                        status=run_row["status"],
                        best=format_metric(run_row["best_error"]),
                    )
                )

        for method_name, config_path in VALIDATED_CONFIGS:
            config = load_method_config(config_path)
            for problem_code in problems:
                for seed in seeds:
                    run_row, case_trace_rows, _ = run_one_case(
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
                    trace_rows.extend(case_trace_rows)
                    print(
                        "validated tfes={tfes} {problem} {method} seed={seed} status={status} best={best} accept_rate={accept_rate}".format(
                            tfes=int(tfes),
                            problem=problem_code,
                            method=method_name,
                            seed=int(seed),
                            status=run_row["status"],
                            best=format_metric(run_row["best_error"]),
                            accept_rate=format_ratio(run_row["validation_accept_rate"]),
                        )
                    )

    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(TRACE_PATH, TRACE_FIELDNAMES, trace_rows)
    build_report(run_rows, trace_rows, threshold_rows, problems, seeds, tfes_values)
    print(f"run details -> {RUN_DETAILS_PATH}")
    print(f"trace -> {TRACE_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
