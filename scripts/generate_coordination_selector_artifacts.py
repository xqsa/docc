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
RUNS_ROOT = ARTIFACTS_ROOT / "coordination_selector_runs"
DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [5000]
DEFAULT_CC_PASS_GROUP_FES = 20
METHOD_CONFIGS = [
    ("no-coordination", CONFIG_ROOT / "no-coordination.json"),
    ("validated-selective-conflict", CONFIG_ROOT / "validated-selective-conflict.json"),
    ("coordination-selector", CONFIG_ROOT / "coordination-selector.json"),
]
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "coordination_selector_run_details.csv"
TRACE_PATH = ARTIFACTS_ROOT / "coordination_selector_trace.csv"
REPORT_PATH = ARTIFACTS_ROOT / "coordination_selector_report.md"

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
    "conflict_damping",
    "validation_enabled",
    "selector_enabled",
    "cc_pass_group_fes",
    "cc_pass_count",
    "post_probe_pass_count",
    "probe_attempt_count",
    "probe_accept_count",
    "probe_reject_count",
    "probe_accept_rate",
    "probe_mean_validation_delta",
    "final_coordination_state",
    "selector_reason",
    "validation_attempt_count",
    "validation_accept_count",
    "validation_reject_count",
    "validation_accept_rate",
    "validation_extra_fe_ratio",
    "final_error",
    "best_error",
    "fe_used",
    "runtime",
    "status",
]
TRACE_FIELDNAMES = list(hcc_es.COORDINATION_SELECTOR_TRACE_FIELDNAMES)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate coordination selector artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    return parser.parse_args()


def load_method_config(config_path, cc_pass_group_fes):
    config = hcc_es.load_info_aware_nda_config(config_path, enable=False)
    return replace(
        config,
        cc_pass_group_fes=max(1, int(cc_pass_group_fes)),
        cc_min_passes=3,
    ).normalized()


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


def format_ratio(value):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.3f}"


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


def run_one_case(method_name, config, problem_code, seed, tfes):
    output_dir = RUNS_ROOT / f"tfes-{int(tfes)}" / method_name
    default_validation = hcc_es.summarize_validated_coordination_rows([], total_max_fes=tfes)
    default_selector = hcc_es.summarize_coordination_selector_rows([], config)
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
        validation_summary = dict(default_validation)
        validation_summary.update(metadata.get("validated_coordination_summary", {}))
        validation_summary.update(diagnostics_payload.get("validated_coordination_summary", {}))
        selector_summary = dict(default_selector)
        selector_summary.update(metadata.get("coordination_selector_summary", {}))
        selector_summary.update(diagnostics_payload.get("coordination_selector_summary", {}))
        selector_rows = [
            {
                **row,
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
            }
            for row in metadata.get("coordination_selector_rows", [])
        ]
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "coordination_mode": hcc_es.resolve_shared_variable_coordination_mode(config),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "validation_enabled": bool(getattr(config, "enable_validated_coordination", False)),
                "selector_enabled": bool(getattr(config, "enable_coordination_selector", False)),
                "cc_pass_group_fes": int(getattr(config, "cc_pass_group_fes", 0) or 0),
                "cc_pass_count": int(selector_summary.get("cc_pass_count", metadata.get("cc_pass_count", 0))),
                "post_probe_pass_count": int(selector_summary.get("post_probe_pass_count", 0)),
                "probe_attempt_count": int(selector_summary.get("probe_attempt_count", 0)),
                "probe_accept_count": int(selector_summary.get("probe_accept_count", 0)),
                "probe_reject_count": int(selector_summary.get("probe_reject_count", 0)),
                "probe_accept_rate": float(selector_summary.get("probe_accept_rate", 0.0)),
                "probe_mean_validation_delta": to_float(selector_summary.get("probe_mean_validation_delta", 0.0)),
                "final_coordination_state": str(selector_summary.get("final_coordination_state", "")),
                "selector_reason": str(selector_summary.get("selector_reason", "")),
                "validation_attempt_count": int(validation_summary.get("validation_attempt_count", 0)),
                "validation_accept_count": int(validation_summary.get("validation_accept_count", 0)),
                "validation_reject_count": int(validation_summary.get("validation_reject_count", 0)),
                "validation_accept_rate": float(validation_summary.get("validation_accept_rate", 0.0)),
                "validation_extra_fe_ratio": float(validation_summary.get("validation_extra_fe_ratio", 0.0)),
                "final_error": to_float(detail.get("final_fitness")),
                "best_error": to_float(detail.get("best_fitness")),
                "fe_used": int(detail.get("fe_used", 0) or 0),
                "runtime": to_float(detail.get("runtime")),
                "status": str(detail.get("status", "ok")),
            },
            selector_rows,
        )
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "coordination_mode": hcc_es.resolve_shared_variable_coordination_mode(config),
                "conflict_damping": bool(getattr(config, "shared_variable_conflict_damping", False)),
                "validation_enabled": bool(getattr(config, "enable_validated_coordination", False)),
                "selector_enabled": bool(getattr(config, "enable_coordination_selector", False)),
                "cc_pass_group_fes": int(getattr(config, "cc_pass_group_fes", 0) or 0),
                "cc_pass_count": 0,
                "post_probe_pass_count": 0,
                "probe_attempt_count": 0,
                "probe_accept_count": 0,
                "probe_reject_count": 0,
                "probe_accept_rate": 0.0,
                "probe_mean_validation_delta": float("nan"),
                "final_coordination_state": "",
                "selector_reason": "",
                "validation_attempt_count": 0,
                "validation_accept_count": 0,
                "validation_reject_count": 0,
                "validation_accept_rate": 0.0,
                "validation_extra_fe_ratio": 0.0,
                "final_error": float("nan"),
                "best_error": float("nan"),
                "fe_used": 0,
                "runtime": float("nan"),
                "status": f"error:{type(exc).__name__}:{exc}",
            },
            [],
        )


def aggregate_performance(rows):
    groups = {}
    for row in rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        groups.setdefault(key, []).append(row)
    performance = {}
    for key, group_rows in groups.items():
        best_values = [row["best_error"] for row in group_rows]
        final_values = [row["final_error"] for row in group_rows]
        performance[key] = {
            "runs": len(group_rows),
            "best_mean": mean_or_nan(best_values),
            "best_std": std_or_nan(best_values),
            "final_mean": mean_or_nan(final_values),
            "final_std": std_or_nan(final_values),
        }
    return performance


def gap(value, baseline):
    value = to_float(value)
    baseline = to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-12:
        return float("nan")
    return float((value - baseline) / abs(baseline))


def build_report(rows, selector_trace_rows=None):
    selector_trace_rows = list(selector_trace_rows or [])
    performance = aggregate_performance(rows)
    lines = [
        "# Coordination Selector",
        "",
        "- Executor: Codex",
        "- Date: 2026-05-20",
        f"- Problems: {', '.join(sorted({row['problem'] for row in rows}))}",
        f"- TFEs: {', '.join(str(tfes) for tfes in sorted({int(row['tfes']) for row in rows}))}",
        f"- Seeds: {', '.join(str(seed) for seed in sorted({int(row['seed']) for row in rows}))}",
        f"- CC pass group FEs: {DEFAULT_CC_PASS_GROUP_FES}",
        "- Rule: attempt >= 1, accept >= 1, accept_rate >= 0.3, mean_validation_delta > 0 -> validated_on; otherwise off.",
        "",
    ]
    for tfes in sorted({int(row["tfes"]) for row in rows}):
        lines.extend([f"## Performance @ TFEs = {tfes}", ""])
        lines.append("| problem | method | best_mean | best_std | final_mean | final_std | gap_vs_no_coordination | gap_vs_always_validated |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for problem in sorted({row["problem"] for row in rows if int(row["tfes"]) == tfes}):
            no_coord = performance.get((problem, tfes, "no-coordination"), {})
            always_validated = performance.get((problem, tfes, "validated-selective-conflict"), {})
            for method_name, _ in METHOD_CONFIGS:
                stats = performance.get((problem, tfes, method_name), {})
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            problem,
                            method_name,
                            format_metric(stats.get("best_mean")),
                            format_metric(stats.get("best_std")),
                            format_metric(stats.get("final_mean")),
                            format_metric(stats.get("final_std")),
                            format_percent(gap(stats.get("best_mean"), no_coord.get("best_mean"))),
                            format_percent(gap(stats.get("best_mean"), always_validated.get("best_mean"))),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    selector_rows = [row for row in rows if row["method"] == "coordination-selector"]
    lines.extend(["## Selector Decisions", ""])
    lines.append("| problem | tfes | seed | cc_passes | post_probe | attempts | accepts | accept_rate | mean_delta | final_state | reason |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in sorted(selector_rows, key=lambda item: (item["problem"], int(item["tfes"]), int(item["seed"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["problem"],
                    str(row["tfes"]),
                    str(row["seed"]),
                    str(row["cc_pass_count"]),
                    str(row["post_probe_pass_count"]),
                    str(row["probe_attempt_count"]),
                    str(row["probe_accept_count"]),
                    format_ratio(row["probe_accept_rate"]),
                    format_metric(row["probe_mean_validation_delta"]),
                    row["final_coordination_state"],
                    row["selector_reason"],
                ]
            )
            + " |"
        )
    lines.append("")

    lines.extend(["## Trace Coverage", ""])
    trace_groups = {}
    for row in selector_trace_rows:
        key = (row.get("problem"), int(row.get("tfes", 0)), int(row.get("seed", 0)))
        trace_groups.setdefault(key, 0)
        trace_groups[key] += 1
    if trace_groups:
        min_trace_count = min(trace_groups.values())
        max_trace_count = max(trace_groups.values())
        lines.append(
            f"- Selector trace rows per problem/tfes/seed run: min={min_trace_count}, max={max_trace_count}."
        )
        if max_trace_count <= 1:
            lines.append(
                "- Interpretation: under this 1000/5000 FE protocol, selector observes the probe pass and makes a decision, but there is no later recorded selector pass where the off/on state can materially diverge from always-on validated coordination."
            )
    else:
        lines.append("- No selector trace rows were recorded.")
    lines.append("")

    lines.extend(["## Key Judgement", ""])
    for problem in sorted({row["problem"] for row in selector_rows}):
        problem_selector = [row for row in selector_rows if row["problem"] == problem]
        off_count = sum(row["final_coordination_state"] == "off" for row in problem_selector)
        on_count = sum(row["final_coordination_state"] == "validated_on" for row in problem_selector)
        lines.append(f"- {problem}: selector off={off_count}, validated_on={on_count}.")
    max_extra_fe_ratio = max([to_float(row["validation_extra_fe_ratio"]) for row in rows] or [0.0])
    selector_cc_pass_counts = [int(row["cc_pass_count"]) for row in selector_rows]
    selector_post_probe_counts = [int(row["post_probe_pass_count"]) for row in selector_rows]
    min_cc_pass_count = min(selector_cc_pass_counts) if selector_cc_pass_counts else 0
    min_post_probe_pass_count = min(selector_post_probe_counts) if selector_post_probe_counts else 0
    lines.append(f"- Max validation extra FE ratio: {format_ratio(max_extra_fe_ratio)}.")
    lines.append(f"- Min selector cc_pass_count: {min_cc_pass_count}.")
    lines.append(f"- Min selector post_probe_pass_count: {min_post_probe_pass_count}.")
    r6_selector = [row for row in selector_rows if row["problem"] == "R6"]
    e6_selector = [row for row in selector_rows if row["problem"] == "E6"]
    a6_selector = [row for row in selector_rows if row["problem"] == "A6"]
    r6_all_off = bool(r6_selector) and all(row["final_coordination_state"] == "off" for row in r6_selector)
    e6_any_on = any(row["final_coordination_state"] == "validated_on" for row in e6_selector)
    a6_any_on = any(row["final_coordination_state"] == "validated_on" for row in a6_selector)
    lines.append(
        f"- R6 automatically closes coordination: {'yes' if r6_all_off else ('not evaluated' if not r6_selector else 'no')}."
    )
    lines.append(
        f"- E6 keeps coordination on accepted probes: {'yes' if e6_any_on else ('not evaluated' if not e6_selector else 'no')}."
    )
    lines.append(
        f"- A6 keeps coordination on accepted probes: {'yes' if a6_any_on else ('not evaluated' if not a6_selector else 'no')}."
    )
    if min_post_probe_pass_count >= 2:
        lines.append(
            "- Protocol check: multi-pass CC scheduling is active, so selector decisions can affect post-probe optimization."
        )
        lines.append(
            "- Recommendation: judge selector retention from R6 protection and A6/E6 coordination retention under this multi-pass protocol."
        )
    else:
        lines.append(
            "- Protocol check: post-probe pass count is still too low, so selector performance remains inconclusive."
        )
        lines.append(
            "- Recommendation: adjust CC pass scheduling before interpreting selector performance."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    run_rows = []
    selector_trace_rows = []
    configs = [(method_name, load_method_config(config_path, args.cc_pass_group_fes)) for method_name, config_path in METHOD_CONFIGS]
    for tfes in args.tfes:
        for problem_code in args.problems:
            normalized_problem = hcc_es.parse_problem_code(problem_code)[2]
            for seed in args.seeds:
                for method_name, config in configs:
                    print(f"Running {method_name} problem={normalized_problem} seed={seed} tfes={tfes}")
                    row, trace_rows = run_one_case(method_name, config, normalized_problem, seed, int(tfes))
                    run_rows.append(row)
                    selector_trace_rows.extend(trace_rows)
    write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
    write_csv(TRACE_PATH, TRACE_FIELDNAMES, selector_trace_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_report(run_rows, selector_trace_rows)
    report_text = report_text.replace(
        f"- CC pass group FEs: {DEFAULT_CC_PASS_GROUP_FES}",
        f"- CC pass group FEs: {int(args.cc_pass_group_fes)}",
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"run_details -> {RUN_DETAILS_PATH}")
    print(f"selector_trace -> {TRACE_PATH}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
