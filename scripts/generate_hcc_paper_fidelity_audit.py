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
RUNS_ROOT = ARTIFACTS_ROOT / "hcc_paper_fidelity_runs"

DEFAULT_GROUP_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_PROTOCOL_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = 10000

GROUP_ORDER_PATH = ARTIFACTS_ROOT / "hcc_paper_fidelity_group_order_audit.csv"
PROTOCOL_DETAIL_PATH = ARTIFACTS_ROOT / "hcc_paper_fidelity_protocol_audit.csv"
PROTOCOL_SUMMARY_PATH = ARTIFACTS_ROOT / "hcc_paper_fidelity_protocol_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "hcc_paper_fidelity_audit.md"

GROUP_ORDER_FIELDNAMES = [
    "problem",
    "true_overlap",
    "adjacent_overlap_in_rddsm_order",
    "adjacent_overlap_in_rddsm_order_ratio",
    "adjacent_overlap_in_aob_natural_order",
    "adjacent_overlap_in_aob_natural_order_ratio",
    "visibility_gap",
    "rddsm_group_count",
    "aob_natural_group_count",
]

PROTOCOL_DETAIL_FIELDNAMES = [
    "problem",
    "protocol",
    "seed",
    "tfes",
    "cc_pass_count",
    "post_probe_pass_count",
    "subspace_budget_mean",
    "selector_decision",
    "selector_reason",
    "probe_attempt_count",
    "probe_accept_count",
    "probe_accept_rate",
    "best_fitness",
    "final_fitness",
    "validation_extra_fe_ratio",
    "fe_used",
    "status",
]

PROTOCOL_SUMMARY_FIELDNAMES = [
    "problem",
    "protocol",
    "tfes",
    "runs",
    "cc_pass_count_mean",
    "post_probe_pass_count_mean",
    "subspace_budget_mean",
    "off_count",
    "validated_on_count",
    "best_mean",
    "best_std",
    "final_mean",
    "final_std",
    "validation_extra_fe_ratio_mean",
]


if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HCC paper-fidelity audit artifacts.")
    parser.add_argument("--group-problems", nargs="+", default=list(DEFAULT_GROUP_PROBLEMS))
    parser.add_argument("--protocol-problems", nargs="+", default=list(DEFAULT_PROTOCOL_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", type=int, default=DEFAULT_TFES)
    parser.add_argument("--skip-protocol-runs", action="store_true")
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


def fmt_float(value, digits=6):
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}g}"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def variable_membership(groups):
    membership = {}
    for group_id, group in enumerate(groups):
        for var_id in group:
            membership.setdefault(int(var_id), []).append(int(group_id))
    return membership


def true_overlap_vars(groups):
    return {var_id for var_id, group_ids in variable_membership(groups).items() if len(group_ids) > 1}


def adjacent_visible_overlap_vars(groups):
    visible = set()
    for left, right in zip(groups, groups[1:]):
        visible.update(set(int(value) for value in left) & set(int(value) for value in right))
    return visible


def build_aob_natural_groups(problem_code):
    fun_name, fun_id, _ = hcc_es.parse_problem_code(problem_code)
    bench = hcc_es.Benchmark(None)
    fun = bench.get_function(fun_name, fun_id)
    groups = []
    cursor = 0
    overlap = int(fun.overlap)
    for group_id, size in enumerate([int(value) for value in fun.s]):
        start = int(cursor - group_id * overlap)
        end = int(cursor + size - group_id * overlap)
        groups.append([int(value) for value in fun.Pvector[start:end]])
        cursor += size
    return groups


def build_group_order_audit(problem_codes):
    rows = []
    for problem_code in problem_codes:
        inputs = hcc_es.build_hcc_es_inputs(problem_code)
        rddsm_groups = inputs["grouping_result"]
        natural_groups = build_aob_natural_groups(problem_code)
        natural_true = true_overlap_vars(natural_groups)
        rddsm_true = true_overlap_vars(rddsm_groups)
        all_true = natural_true | rddsm_true
        true_count = len(all_true)
        rddsm_adjacent_count = len(adjacent_visible_overlap_vars(rddsm_groups) & all_true)
        natural_adjacent_count = len(adjacent_visible_overlap_vars(natural_groups) & all_true)
        rddsm_ratio = float(rddsm_adjacent_count / true_count) if true_count else 0.0
        natural_ratio = float(natural_adjacent_count / true_count) if true_count else 0.0
        rows.append(
            {
                "problem": str(problem_code).upper(),
                "true_overlap": int(true_count),
                "adjacent_overlap_in_rddsm_order": int(rddsm_adjacent_count),
                "adjacent_overlap_in_rddsm_order_ratio": rddsm_ratio,
                "adjacent_overlap_in_aob_natural_order": int(natural_adjacent_count),
                "adjacent_overlap_in_aob_natural_order_ratio": natural_ratio,
                "visibility_gap": float(natural_ratio - rddsm_ratio),
                "rddsm_group_count": int(len(rddsm_groups)),
                "aob_natural_group_count": int(len(natural_groups)),
            }
        )
    return rows


def load_selector_config(protocol):
    config = hcc_es.load_info_aware_nda_config(CONFIG_ROOT / "coordination-selector.json", enable=False)
    if protocol == "paper-like":
        return replace(
            config,
            cc_pass_group_fes=None,
            cc_min_passes=1,
            enable_group_delta_trace=True,
        ).normalized()
    if protocol == "selector-friendly":
        return replace(
            config,
            cc_pass_group_fes=20,
            cc_min_passes=3,
            enable_group_delta_trace=True,
        ).normalized()
    raise ValueError(f"Unknown protocol: {protocol}")


def run_protocol_case(problem_code, protocol, seed, tfes):
    config = load_selector_config(protocol)
    output_dir = RUNS_ROOT / protocol / f"tfes-{int(tfes)}"
    try:
        result = hcc_es.run_problem_seed_task(
            problem_code,
            int(seed),
            int(tfes),
            hcc_es.HCC_ES_METHOD,
            output_dir,
            record_fes=[],
            info_aware_config=config,
            method_label=f"coordination-selector-{protocol}",
        )
        detail = dict(result.get("detail", {}))
        metadata = dict(result.get("metadata", {}))
        selector_summary = dict(metadata.get("coordination_selector_summary", {}))
        validation_summary = dict(metadata.get("validated_coordination_summary", {}))
        group_trace_rows = list(metadata.get("group_trace_rows", []))
        budget_mean = mean_or_nan([row.get("budget") for row in group_trace_rows])
        return {
            "problem": str(problem_code).upper(),
            "protocol": str(protocol),
            "seed": int(seed),
            "tfes": int(tfes),
            "cc_pass_count": int(selector_summary.get("cc_pass_count", metadata.get("cc_pass_count", 0)) or 0),
            "post_probe_pass_count": int(selector_summary.get("post_probe_pass_count", 0) or 0),
            "subspace_budget_mean": float(budget_mean),
            "selector_decision": str(selector_summary.get("final_coordination_state", "")),
            "selector_reason": str(selector_summary.get("selector_reason", "")),
            "probe_attempt_count": int(selector_summary.get("probe_attempt_count", 0) or 0),
            "probe_accept_count": int(selector_summary.get("probe_accept_count", 0) or 0),
            "probe_accept_rate": float(selector_summary.get("probe_accept_rate", 0.0) or 0.0),
            "best_fitness": to_float(detail.get("best_fitness")),
            "final_fitness": to_float(detail.get("final_fitness")),
            "validation_extra_fe_ratio": float(validation_summary.get("validation_extra_fe_ratio", 0.0) or 0.0),
            "fe_used": int(detail.get("fe_used", 0) or 0),
            "status": str(detail.get("status", "ok")),
        }
    except Exception as exc:
        return {
            "problem": str(problem_code).upper(),
            "protocol": str(protocol),
            "seed": int(seed),
            "tfes": int(tfes),
            "cc_pass_count": 0,
            "post_probe_pass_count": 0,
            "subspace_budget_mean": float("nan"),
            "selector_decision": "",
            "selector_reason": "",
            "probe_attempt_count": 0,
            "probe_accept_count": 0,
            "probe_accept_rate": 0.0,
            "best_fitness": float("nan"),
            "final_fitness": float("nan"),
            "validation_extra_fe_ratio": 0.0,
            "fe_used": 0,
            "status": f"error:{type(exc).__name__}:{exc}",
        }


def build_protocol_audit(problem_codes, seeds, tfes):
    rows = []
    for problem_code in problem_codes:
        for protocol in ("paper-like", "selector-friendly"):
            for seed in seeds:
                rows.append(run_protocol_case(problem_code, protocol, int(seed), int(tfes)))
    return rows


def summarize_protocol_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["problem"], row["protocol"], int(row["tfes"]))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (problem, protocol, tfes), group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "problem": problem,
                "protocol": protocol,
                "tfes": int(tfes),
                "runs": int(len(group_rows)),
                "cc_pass_count_mean": mean_or_nan([row["cc_pass_count"] for row in group_rows]),
                "post_probe_pass_count_mean": mean_or_nan([row["post_probe_pass_count"] for row in group_rows]),
                "subspace_budget_mean": mean_or_nan([row["subspace_budget_mean"] for row in group_rows]),
                "off_count": int(sum(str(row["selector_decision"]) == "off" for row in group_rows)),
                "validated_on_count": int(sum(str(row["selector_decision"]) == "validated_on" for row in group_rows)),
                "best_mean": mean_or_nan([row["best_fitness"] for row in group_rows]),
                "best_std": std_or_nan([row["best_fitness"] for row in group_rows]),
                "final_mean": mean_or_nan([row["final_fitness"] for row in group_rows]),
                "final_std": std_or_nan([row["final_fitness"] for row in group_rows]),
                "validation_extra_fe_ratio_mean": mean_or_nan([row["validation_extra_fe_ratio"] for row in group_rows]),
            }
        )
    return summaries


def report_group_table(rows):
    lines = [
        "| problem | true_overlap | rddsm_adjacent | natural_adjacent | visibility_gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {problem} | {true_overlap} | {rddsm} ({rddsm_ratio:.3f}) | {natural} ({natural_ratio:.3f}) | {gap:.3f} |".format(
                problem=row["problem"],
                true_overlap=int(row["true_overlap"]),
                rddsm=int(row["adjacent_overlap_in_rddsm_order"]),
                rddsm_ratio=float(row["adjacent_overlap_in_rddsm_order_ratio"]),
                natural=int(row["adjacent_overlap_in_aob_natural_order"]),
                natural_ratio=float(row["adjacent_overlap_in_aob_natural_order_ratio"]),
                gap=float(row["visibility_gap"]),
            )
        )
    return "\n".join(lines)


def report_protocol_table(rows):
    lines = [
        "| problem | protocol | passes | post_probe | subFEs_mean | off/on | best_mean | final_mean | extra_fe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {problem} | {protocol} | {passes} | {post_probe} | {subfes} | {off}/{on} | {best} | {final} | {extra} |".format(
                problem=row["problem"],
                protocol=row["protocol"],
                passes=fmt_float(row["cc_pass_count_mean"], 4),
                post_probe=fmt_float(row["post_probe_pass_count_mean"], 4),
                subfes=fmt_float(row["subspace_budget_mean"], 4),
                off=int(row["off_count"]),
                on=int(row["validated_on_count"]),
                best=fmt_float(row["best_mean"], 6),
                final=fmt_float(row["final_mean"], 6),
                extra=fmt_float(row["validation_extra_fe_ratio_mean"], 4),
            )
        )
    return "\n".join(lines)


def write_report(group_rows, protocol_summary_rows, protocol_rows_available):
    max_gap = max((float(row["visibility_gap"]) for row in group_rows), default=0.0)
    rddsm_low = [row for row in group_rows if float(row["visibility_gap"]) > 0.5]
    protocol_note = (
        "Protocol audit was executed with selector-only A/B runs."
        if protocol_rows_available
        else "Protocol audit runs were skipped by command-line flag."
    )
    lines = [
        "# HCC Paper-Fidelity Audit",
        "",
        "Date: 2026-05-20",
        "Executor: Codex",
        "",
        "## Executive Summary",
        "",
        f"- RDDSM order audit max visibility gap: {max_gap:.3f}.",
        f"- Large RDDSM-order visibility gaps found: {len(rddsm_low)} problem(s).",
        "- Equation 8 original adjacent blend is directionally reversed in `overlap_blend_mode=original`: `previous_delta` becomes the weight on the current group's value.",
        "- Safe delta/conflict blend uses the current delta as the current-value weight and skips non-positive-delta cases, so it is closer to the paper contribution direction than original mode.",
        "- The current implementation does not maintain persistent per-subspace `omega_i`; CMA-ES is recreated for each group/pass and receives `mean=best_individual[dims]`.",
        f"- {protocol_note}",
        "",
        "## Audit 1: RDDSM Order Visibility",
        "",
        report_group_table(group_rows),
        "",
        "Interpretation: if natural adjacent visibility is near 1.0 but RDDSM visibility is low, adjacent Equation 8 coordination is order-fragile. That points to topology-preserving group ordering before adding more coordination policy.",
        "",
        "## Audit 2: Equation 8 Direction",
        "",
        "- Paper intent: for adjacent subspaces `i-1` and `i`, the shared variable value should be weighted by each subspace contribution. Larger `Delta_i` should pull more toward current group `i`; larger `Delta_{i-1}` should pull more toward previous group `i-1`.",
        "- Code evidence: `safe_overlap_weight(previous_delta, current_delta)` returns `previous_delta / (previous_delta + current_delta)`, and `blend_overlapping_elements` applies `weight * best_individual + (1 - weight) * original_best_individual`.",
        "- In the current loop, `original_best_individual` is the state before the current group update, while `best_individual` contains the current group proposal. Therefore original mode gives current value the previous group's weight.",
        "- Negative delta handling: original mode does not skip negative deltas and only clips the resulting ratio. Safe delta/conflict and pass-end hypergraph coordination use positive-delta gating.",
        "",
        "Conclusion: paper-fidelity risk is real for `overlap_blend_mode=original`. The later safe modes and hypergraph mechanisms were built around this, but any baseline claiming original Equation 8 fidelity should be treated carefully.",
        "",
        "## Audit 3: Optimizer Mean Synchronization",
        "",
        "- Paper Algorithm 2 says both `gbest_i` and `omega_i` are adjusted by Equation 8.",
        "- Code evidence: each CC subspace creates a fresh `CMAES(problem_cc, options_cc)` and sets `options_cc['mean'] = (best_individual[dims],)`.",
        "- Coordination updates `best_individual` after the subspace optimizer returns; it does not update the returned `results_cc['mean']`, and there is no persistent `omega_i` cache across passes.",
        "- Consequence: coordination can affect future subspace optimizers only if another pass revisits those variables. In paper-like one-pass CC, this absorption path is mostly absent.",
        "",
        "Conclusion: current code is not fully paper-faithful on `omega_i` synchronization. This is a strong explanation for coordination improving final vectors while failing to move best-so-far consistently.",
        "",
        "## Audit 4: Protocol Difference",
        "",
        report_protocol_table(protocol_summary_rows) if protocol_rows_available else "Protocol runs skipped.",
        "",
        "Interpretation: paper-like scheduling usually has little or no post-probe room, while selector-friendly scheduling creates many smaller CC passes. Selector validity must therefore be reported separately under both protocols.",
        "",
        "## Files",
        "",
        f"- Group order CSV: `{GROUP_ORDER_PATH.as_posix()}`",
        f"- Protocol detail CSV: `{PROTOCOL_DETAIL_PATH.as_posix()}`",
        f"- Protocol summary CSV: `{PROTOCOL_SUMMARY_PATH.as_posix()}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    group_rows = build_group_order_audit(args.group_problems)
    write_csv(GROUP_ORDER_PATH, GROUP_ORDER_FIELDNAMES, group_rows)

    protocol_rows = []
    protocol_summary_rows = []
    if not args.skip_protocol_runs:
        protocol_rows = build_protocol_audit(args.protocol_problems, args.seeds, int(args.tfes))
        protocol_summary_rows = summarize_protocol_rows(protocol_rows)
        write_csv(PROTOCOL_DETAIL_PATH, PROTOCOL_DETAIL_FIELDNAMES, protocol_rows)
        write_csv(PROTOCOL_SUMMARY_PATH, PROTOCOL_SUMMARY_FIELDNAMES, protocol_summary_rows)
    else:
        write_csv(PROTOCOL_DETAIL_PATH, PROTOCOL_DETAIL_FIELDNAMES, [])
        write_csv(PROTOCOL_SUMMARY_PATH, PROTOCOL_SUMMARY_FIELDNAMES, [])

    write_report(group_rows, protocol_summary_rows, bool(protocol_rows))


if __name__ == "__main__":
    main()
