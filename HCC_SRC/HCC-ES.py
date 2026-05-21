import argparse
import concurrent.futures
import csv
import hashlib
import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np

from AOB.AOB import Benchmark
from AOB.utils import (
    combine,
    evaluation_record,
    plot_evaluation_curve,
    plot_evaluation_curve_best_so_far,
    remove_overlapping_groups,
)
from HCC.NDAs.MMES.mmes import MMES
from HCC.OPT.CMAES.cmaes import CMAES
from HCC.RDDSM import Decomposition
from HCC.info_aware_nda import (
    CCPrior,
    InfoAwareNDAConfig,
    NDAInfo,
    allocate_priority_budgets,
    build_cc_prior,
    build_info_aware_diagnostics_payload,
    build_priority_order,
    compute_priority_audit,
    load_info_aware_nda_config,
    merge_warnings,
    run_adaptive_info_aware_nda,
    save_info_aware_diagnostics,
)
from experiment_protocols import protocol_choices, resolve_protocol


HCC_ES_METHOD = "hcc_es_original"
METHOD_CHOICES = (HCC_ES_METHOD,)
DEFAULT_PROBLEM_CODES = ("E4", "E6", "S4", "S6", "A6", "R6")
PROBLEM_PREFIX_MAP = {
    "E": "elliptic",
    "S": "schwefel",
    "R": "rastrigin",
    "A": "ackley",
}
RUN_DETAIL_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "blend_strategy",
    "seed",
    "final_fitness",
    "best_fitness",
    "fe_used",
    "runtime",
    "diagnostics_count",
    "rollback_ratio",
    "conflict_mean",
    "status",
]
SUMMARY_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "blend_strategy",
    "runs",
    "best_mean",
    "best_std",
    "best_min",
    "best_max",
    "diagnostics_count",
    "rollback_ratio",
    "conflict_mean",
]
DIAGNOSTIC_FIELDNAMES = [
    "problem",
    "run_id",
    "method",
]
GROUP_TRACE_FIELDNAMES = [
    "cycle_id",
    "scheduled_position",
    "original_group_id",
    "group_size",
    "priority",
    "budget",
    "actual_fe",
    "fitness_before",
    "fitness_after",
    "actual_delta",
    "overlap_var_count",
    "overlap_ratio",
    "conflict_prior_mean",
    "was_sorted",
]
OVERLAP_BLEND_TRACE_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "seed",
    "cycle_id",
    "group_id",
    "previous_group_id",
    "scheduled_position",
    "overlap_count",
    "blend_strategy",
    "blend_mode",
    "previous_delta",
    "current_delta",
    "raw_weight",
    "weight",
    "conflict_mean",
    "damping",
    "was_skipped",
    "applied_update",
]
SHARED_VARIABLE_PROPOSAL_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "seed",
    "cycle_id",
    "scheduled_position",
    "group_id",
    "var_id",
    "proposal_value",
    "delta",
    "fitness_before",
    "fitness_after",
]
SHARED_VARIABLE_FUSION_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "seed",
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
    "post_coordination_fitness_before",
    "post_coordination_fitness_after",
    "post_coordination_best_improved",
    "fitness_delta",
    "owner_soft_followed_by_best_improvement",
    "owner_soft_overwritten",
    "validation_attempted",
    "validation_accepted",
    "validation_status",
    "validation_reject_reason",
    "action_candidate",
    "action_reason",
    "arac_recovery_candidate",
    "arac_recovery_phase",
    "arac_recovery_reason",
    "arac_recovery_attempt_count",
    "arac_recovery_accept_rate",
    "arac_recovery_delta_mean",
    "arac_recovery_positive_delta_rate",
    "arac_probe_candidate",
    "arac_probe_reason",
    "arac_probe_phase",
    "arac_probe_attempt_count",
    "arac_probe_accept_rate",
    "arac_probe_delta_mean",
    "relation_attempt_count",
    "relation_accept_count",
    "relation_accept_rate",
    "relation_mean_validation_delta",
    "relation_reject_streak",
]
VALIDATED_COORDINATION_TRACE_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "seed",
    "cycle_id",
    "validation_attempted",
    "validation_accepted",
    "fitness_before_validation",
    "candidate_fitness",
    "fitness_delta",
    "validation_fe_used",
    "extra_fe_ratio",
    "candidate_update_count",
    "candidate_owner_soft_count",
    "candidate_multi_support_count",
    "candidate_freeze_count",
    "candidate_mean_update_magnitude",
    "candidate_mean_conflict_prior",
    "reject_reason",
]
COORDINATION_SELECTOR_TRACE_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "pass_id",
    "phase",
    "coordination_state",
    "validation_attempted",
    "validation_accepted",
    "fitness_before_validation",
    "candidate_fitness",
    "fitness_delta",
    "probe_attempt_count",
    "probe_accept_count",
    "probe_reject_count",
    "probe_accept_rate",
    "probe_mean_validation_delta",
    "selector_decision",
    "selector_reason",
    "extra_fe_ratio",
]
OPTIMIZER_STATE_TRACE_FIELDNAMES = [
    "problem",
    "method",
    "tfes",
    "seed",
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


DEFAULT_INFO_AWARE_DIAGNOSTICS_ARTIFACT = Path("artifacts") / "info_aware_nda_diagnostics.json"


def parse_problem_code(problem_code):
    normalized = str(problem_code).strip().upper()
    if len(normalized) < 2:
        raise ValueError(f"Invalid problem code: {problem_code}")
    prefix = normalized[0]
    if prefix not in PROBLEM_PREFIX_MAP:
        raise ValueError(f"Unsupported problem prefix: {problem_code}")
    try:
        function_id = int(normalized[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid problem id: {problem_code}") from exc
    return PROBLEM_PREFIX_MAP[prefix], function_id, normalized


def canonicalize_method(method):
    normalized = str(method or HCC_ES_METHOD).strip().lower()
    if normalized != HCC_ES_METHOD:
        raise ValueError(f"Unsupported method: {method}")
    return HCC_ES_METHOD


def resolve_blend_strategy_name(info_aware_config=None):
    if info_aware_config is None:
        return "original"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return str(getattr(normalized_config, "overlap_blend_mode", "original") or "original").strip().lower()


def resolve_group_order_mode(info_aware_config=None):
    if info_aware_config is None:
        return "rddsm"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return str(getattr(normalized_config, "group_order_mode", "rddsm") or "rddsm").strip().lower()


def resolve_coordination_order_mode(info_aware_config=None):
    if info_aware_config is None:
        return "match_execution"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return str(getattr(normalized_config, "coordination_order_mode", "match_execution") or "match_execution").strip().lower()


def resolve_optimizer_state_mode(info_aware_config=None):
    if info_aware_config is None:
        return "ephemeral"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    mode = str(getattr(normalized_config, "optimizer_state_mode", "ephemeral") or "ephemeral").strip().lower()
    return mode if mode in {"ephemeral", "persistent_mean"} else "ephemeral"


def resolve_shared_variable_coordination_mode(info_aware_config=None):
    if info_aware_config is None:
        return "adjacent"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return str(getattr(normalized_config, "shared_variable_coordination_mode", "adjacent") or "adjacent").strip().lower()


def is_hypergraph_pass_end_mode(coordination_mode):
    mode = str(coordination_mode or "").strip().lower()
    return mode in {"hypergraph_pass_end", "selective_hypergraph_pass_end", "arac_lite_rule"}


def is_selective_hypergraph_mode(coordination_mode):
    return str(coordination_mode or "").strip().lower() == "selective_hypergraph_pass_end"


def is_arac_lite_mode(coordination_mode):
    return str(coordination_mode or "").strip().lower() == "arac_lite_rule"


def resolve_validation_mode(info_aware_config=None):
    if info_aware_config is None:
        return "off"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return str(getattr(normalized_config, "validation_mode", "off") or "off").strip().lower()


def is_pass_level_validated_coordination_enabled(info_aware_config=None):
    if info_aware_config is None:
        return False
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return bool(getattr(normalized_config, "enable_validated_coordination", False)) and resolve_validation_mode(normalized_config) == "pass_end"


def resolve_validation_max_extra_fes(max_fes, info_aware_config=None):
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    ratio = 0.0 if normalized_config is None else float(getattr(normalized_config, "validation_max_extra_fe_ratio", 0.0) or 0.0)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return max(0, int(math.floor(float(max_fes) * ratio + 1e-12)))


def resolve_record_method(method, info_aware_config=None):
    canonical_method = canonicalize_method(method)
    if canonical_method != HCC_ES_METHOD:
        return canonical_method
    if info_aware_config is None:
        return "hcc_es_baseline"
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    coordination_mode = resolve_shared_variable_coordination_mode(normalized_config)
    validation_enabled = is_pass_level_validated_coordination_enabled(normalized_config)
    if bool(getattr(normalized_config, "enable_coordination_selector", False)):
        return "hcc_es_coordination_selector"
    if coordination_mode == "selective_hypergraph_pass_end":
        if validation_enabled:
            if bool(getattr(normalized_config, "shared_variable_conflict_damping", False)):
                return "hcc_es_validated_selective_hypergraph_pass_end_conflict"
            return "hcc_es_validated_selective_hypergraph_pass_end"
        if bool(getattr(normalized_config, "shared_variable_conflict_damping", False)):
            return "hcc_es_selective_hypergraph_pass_end_conflict"
        return "hcc_es_selective_hypergraph_pass_end"
    if coordination_mode == "hypergraph_pass_end":
        if validation_enabled:
            if bool(getattr(normalized_config, "shared_variable_conflict_damping", False)):
                return "hcc_es_validated_hypergraph_pass_end_conflict"
            return "hcc_es_validated_hypergraph_pass_end"
        if bool(getattr(normalized_config, "shared_variable_conflict_damping", False)):
            return "hcc_es_hypergraph_pass_end_conflict"
        return "hcc_es_hypergraph_pass_end"
    if coordination_mode == "no_coordination":
        return "hcc_es_no_coordination"
    if not getattr(normalized_config, "enable", False):
        return "hcc_es_baseline"
    blend_strategy = resolve_blend_strategy_name(normalized_config)
    if blend_strategy == "safe_conflict":
        return "hcc_es_safe_conflict_blend"
    if blend_strategy == "safe_delta":
        return "hcc_es_safe_delta_blend"
    if blend_strategy == "equation8_correct":
        return "hcc_es_equation8_correct_blend"
    if blend_strategy == "no_blend":
        return "hcc_es_no_blend"
    return "hcc_es_early_switch"


def group_problem_codes(problem_codes):
    grouped = {}
    for problem_code in problem_codes:
        fun_name, fun_id, _ = parse_problem_code(problem_code)
        grouped.setdefault(fun_name, []).append(int(fun_id))
    return {
        fun_name: list(dict.fromkeys(fun_ids))
        for fun_name, fun_ids in grouped.items()
    }


def stage_seed(seed, offset):
    if seed is None:
        return None
    return int(seed) * 1_000_003 + int(offset)


def safe_overlap_weight(previous_delta, current_delta):
    previous_delta = float(previous_delta)
    current_delta = float(current_delta)
    denominator = previous_delta + current_delta
    if not np.isfinite(previous_delta) or not np.isfinite(current_delta) or not np.isfinite(denominator):
        return 0.5
    if denominator == 0.0:
        return 0.5
    return float(np.clip(previous_delta / denominator, 0.0, 1.0))


def build_overlap_hypergraph(grouping_result):
    var_to_groups = {}
    for group_id, group in enumerate(grouping_result):
        for var_id in group:
            var_to_groups.setdefault(int(var_id), []).append(int(group_id))

    var_to_groups = {var_id: sorted(group_ids) for var_id, group_ids in var_to_groups.items()}
    overlap_vars = sorted(var_id for var_id, group_ids in var_to_groups.items() if len(group_ids) >= 2)
    overlap_var_set = set(overlap_vars)
    group_to_overlap_vars = {
        int(group_id): [int(var_id) for var_id in group if int(var_id) in overlap_var_set]
        for group_id, group in enumerate(grouping_result)
    }
    membership_histogram = {}
    for group_ids in var_to_groups.values():
        membership = len(group_ids)
        membership_histogram[membership] = membership_histogram.get(membership, 0) + 1
    return {
        "var_to_groups": var_to_groups,
        "group_to_overlap_vars": group_to_overlap_vars,
        "overlap_vars": overlap_vars,
        "membership_histogram": membership_histogram,
    }


def build_overlap_features(grouping_result, overlap_hypergraph):
    var_to_groups = overlap_hypergraph.get("var_to_groups", {})
    group_to_overlap_vars = overlap_hypergraph.get("group_to_overlap_vars", {})
    overlap_vars = [int(var_id) for var_id in overlap_hypergraph.get("overlap_vars", [])]
    all_vars = {int(var_id) for group in grouping_result for var_id in group}
    dimension = len(all_vars)

    nonadjacent_overlap_count = 0
    for var_id in overlap_vars:
        group_ids = sorted(int(group_id) for group_id in var_to_groups.get(int(var_id), []))
        if any(abs(group_ids[right] - group_ids[left]) > 1 for left in range(len(group_ids)) for right in range(left + 1, len(group_ids))):
            nonadjacent_overlap_count += 1

    group_overlap_var_count = []
    group_overlap_load = []
    for group_id in range(len(grouping_result)):
        group_vars = [int(var_id) for var_id in group_to_overlap_vars.get(int(group_id), [])]
        group_overlap_var_count.append(len(group_vars))
        overlap_load = 0.0
        for var_id in group_vars:
            overlap_load += max(0, len(var_to_groups.get(int(var_id), ())) - 1)
        group_overlap_load.append(float(overlap_load))

    overlap_ratio = float(len(overlap_vars) / dimension) if dimension else 0.0
    nonadjacent_overlap_ratio = float(nonadjacent_overlap_count / len(overlap_vars)) if overlap_vars else 0.0
    return {
        "overlap_ratio": overlap_ratio,
        "nonadjacent_overlap_ratio": nonadjacent_overlap_ratio,
        "group_overlap_var_count": group_overlap_var_count,
        "group_overlap_load": group_overlap_load,
    }


def audit_grouping_dimension_coverage(grouping_result, benchmark_dimension, design_matrix_dimension=None):
    benchmark_dimension = int(benchmark_dimension)
    covered_variables = sorted(
        {
            int(var_id)
            for group in (grouping_result or [])
            for var_id in group
            if 0 <= int(var_id) < benchmark_dimension
        }
    )
    covered_variable_set = set(covered_variables)
    uncovered_variables = [
        int(var_id)
        for var_id in range(benchmark_dimension)
        if int(var_id) not in covered_variable_set
    ]
    return {
        "benchmark_dimension": benchmark_dimension,
        "design_matrix_dimension": int(design_matrix_dimension) if design_matrix_dimension is not None else "",
        "covered_variable_count": int(len(covered_variables)),
        "uncovered_variable_count": int(len(uncovered_variables)),
        "uncovered_variables": uncovered_variables,
    }


def ensure_grouping_covers_benchmark_dimension(grouping_result, benchmark_dimension, design_matrix_dimension=None):
    normalized_groups = [
        list(dict.fromkeys(int(var_id) for var_id in group))
        for group in (grouping_result or [])
    ]
    coverage_audit = audit_grouping_dimension_coverage(
        normalized_groups,
        benchmark_dimension,
        design_matrix_dimension,
    )
    uncovered_variables = list(coverage_audit.get("uncovered_variables", []))
    coverage_audit["coverage_patch_applied"] = bool(uncovered_variables)
    if not uncovered_variables:
        coverage_audit["effective_covered_variable_count"] = int(coverage_audit["covered_variable_count"])
        coverage_audit["effective_uncovered_variable_count"] = int(coverage_audit["uncovered_variable_count"])
        return normalized_groups, coverage_audit

    augmented_groups = normalized_groups + [uncovered_variables]
    effective_audit = audit_grouping_dimension_coverage(
        augmented_groups,
        benchmark_dimension,
        design_matrix_dimension,
    )
    coverage_audit["effective_covered_variable_count"] = int(effective_audit["covered_variable_count"])
    coverage_audit["effective_uncovered_variable_count"] = int(effective_audit["uncovered_variable_count"])
    coverage_audit["residual_group_size"] = int(len(uncovered_variables))
    return augmented_groups, coverage_audit


def compute_original_glofes(original_do, max_fes, has_overlap):
    if not has_overlap:
        return 0
    ratio = float(0.2 + 0.8 * float(original_do))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return int(ratio * int(max_fes))


def compute_adjacent_overlaps_for_groups(grouping_result):
    if len(grouping_result) <= 1:
        return []
    _, _, adjacent_overlapping_elements = remove_overlapping_groups(grouping_result)
    return adjacent_overlapping_elements


def _variable_membership(groups):
    membership = {}
    for group_id, group in enumerate(groups or []):
        for var_id in group:
            membership.setdefault(int(var_id), []).append(int(group_id))
    return membership


def _true_overlap_vars(groups):
    return {
        int(var_id)
        for var_id, group_ids in _variable_membership(groups).items()
        if len(group_ids) > 1
    }


def _adjacent_visible_overlap_vars(groups):
    visible = set()
    for left_group, right_group in zip(groups or [], (groups or [])[1:]):
        visible.update(set(int(value) for value in left_group) & set(int(value) for value in right_group))
    return visible


def build_group_order_visibility_audit(original_groups, ordered_groups, reference_groups=None):
    reference_groups = reference_groups or []
    true_overlap = _true_overlap_vars(reference_groups) | _true_overlap_vars(original_groups) | _true_overlap_vars(ordered_groups)
    true_count = len(true_overlap)

    def ratio(groups):
        if true_count == 0:
            return 0.0
        return float(len(_adjacent_visible_overlap_vars(groups) & true_overlap) / true_count)

    original_visible = _adjacent_visible_overlap_vars(original_groups) & true_overlap
    ordered_visible = _adjacent_visible_overlap_vars(ordered_groups) & true_overlap
    reference_visible = _adjacent_visible_overlap_vars(reference_groups) & true_overlap
    return {
        "true_overlap_var_count": int(true_count),
        "original_adjacent_overlap_count": int(len(original_visible)),
        "original_adjacent_overlap_ratio": ratio(original_groups),
        "ordered_adjacent_overlap_count": int(len(ordered_visible)),
        "ordered_adjacent_overlap_ratio": ratio(ordered_groups),
        "reference_adjacent_overlap_count": int(len(reference_visible)),
        "reference_adjacent_overlap_ratio": ratio(reference_groups),
        "visibility_gain": float(ratio(ordered_groups) - ratio(original_groups)),
    }


def build_aob_natural_groups(problem_code):
    fun_name, fun_id, _ = parse_problem_code(problem_code)
    bench = Benchmark(None)
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


def reorder_groups_by_reference_topology(grouping_result, reference_groups):
    remaining = [
        (group_id, list(dict.fromkeys(int(var_id) for var_id in group)))
        for group_id, group in enumerate(grouping_result or [])
    ]
    ordered_pairs = []
    for reference_group in reference_groups or []:
        if not remaining:
            break
        reference_set = set(int(var_id) for var_id in reference_group)
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(set(remaining[index][1]) & reference_set),
                -abs(len(remaining[index][1]) - len(reference_set)),
                -remaining[index][0],
            ),
        )
        ordered_pairs.append(remaining.pop(best_index))
    ordered_pairs.extend(remaining)
    order = [int(group_id) for group_id, _ in ordered_pairs]
    ordered_groups = [list(group) for _, group in ordered_pairs]
    audit = build_group_order_visibility_audit(grouping_result, ordered_groups, reference_groups)
    audit["group_order_changed"] = order != list(range(len(grouping_result or [])))
    return ordered_groups, order, audit


def apply_group_order_mode(grouping_result, problem_code=None, group_order_mode="rddsm"):
    mode = str(group_order_mode or "rddsm").strip().lower()
    if mode != "aob_topology" or not problem_code:
        ordered_groups = [
            list(dict.fromkeys(int(var_id) for var_id in group))
            for group in (grouping_result or [])
        ]
        order = list(range(len(ordered_groups)))
        audit = build_group_order_visibility_audit(ordered_groups, ordered_groups, ordered_groups)
        audit["group_order_mode"] = "rddsm"
        audit["group_order_changed"] = False
        return ordered_groups, order, audit
    reference_groups = build_aob_natural_groups(problem_code)
    ordered_groups, order, audit = reorder_groups_by_reference_topology(grouping_result, reference_groups)
    audit["group_order_mode"] = "aob_topology"
    return ordered_groups, order, audit


def apply_group_order_mode_with_reference(grouping_result, problem_code=None, group_order_mode="rddsm", reference_groups=None):
    mode = str(group_order_mode or "rddsm").strip().lower()
    if mode == "aob_topology" and reference_groups is not None:
        ordered_groups, order, audit = reorder_groups_by_reference_topology(grouping_result, reference_groups)
        audit["group_order_mode"] = "aob_topology"
        return ordered_groups, order, audit
    return apply_group_order_mode(grouping_result, problem_code=problem_code, group_order_mode=mode)


def build_group_order_plan(
    grouping_result,
    problem_code=None,
    execution_order_mode="rddsm",
    coordination_order_mode="match_execution",
    reference_groups=None,
):
    original_groups = [
        list(dict.fromkeys(int(var_id) for var_id in group))
        for group in (grouping_result or [])
    ]
    if reference_groups is None and problem_code:
        reference_groups = build_aob_natural_groups(problem_code)
    reference_groups = reference_groups or original_groups

    execution_groups, execution_order, execution_audit = apply_group_order_mode_with_reference(
        original_groups,
        problem_code=problem_code,
        group_order_mode=execution_order_mode,
        reference_groups=reference_groups,
    )
    execution_audit["group_order_mode"] = str(execution_order_mode or "rddsm").strip().lower()

    resolved_coordination_mode = str(coordination_order_mode or "match_execution").strip().lower()
    if resolved_coordination_mode == "match_execution":
        coordination_groups = [list(group) for group in execution_groups]
        coordination_order = [int(group_id) for group_id in execution_order]
        coordination_audit = build_group_order_visibility_audit(
            original_groups,
            coordination_groups,
            reference_groups,
        )
        coordination_audit["group_order_changed"] = coordination_order != list(range(len(original_groups)))
        coordination_audit["group_order_mode"] = "match_execution"
    else:
        coordination_groups, coordination_order, coordination_audit = apply_group_order_mode_with_reference(
            original_groups,
            problem_code=problem_code,
            group_order_mode=resolved_coordination_mode,
            reference_groups=reference_groups,
        )
        coordination_audit["group_order_mode"] = resolved_coordination_mode

    return {
        "execution_groups": execution_groups,
        "execution_group_order": [int(group_id) for group_id in execution_order],
        "execution_order_audit": execution_audit,
        "coordination_groups": coordination_groups,
        "coordination_group_order": [int(group_id) for group_id in coordination_order],
        "coordination_order_audit": coordination_audit,
    }


def build_coordination_edge_lookup(coordination_groups, coordination_group_order):
    edge_lookup = {}
    groups_by_original_id = {
        int(group_id): list(coordination_groups[position])
        for position, group_id in enumerate(coordination_group_order or [])
        if position < len(coordination_groups)
    }
    for left_position, right_position in zip(range(len(coordination_group_order or []) - 1), range(1, len(coordination_group_order or []))):
        left_group_id = int(coordination_group_order[left_position])
        right_group_id = int(coordination_group_order[right_position])
        left_group = groups_by_original_id.get(left_group_id, [])
        right_group = groups_by_original_id.get(right_group_id, [])
        overlap = sorted(set(int(value) for value in left_group) & set(int(value) for value in right_group))
        edge = {
            "edge_key": tuple(sorted((left_group_id, right_group_id))),
            "left_group_id": left_group_id,
            "right_group_id": right_group_id,
            "left_position": int(left_position),
            "right_position": int(right_position),
            "overlap_indices": [int(value) for value in overlap],
        }
        edge_lookup.setdefault(left_group_id, []).append(edge)
        edge_lookup.setdefault(right_group_id, []).append(edge)
    return edge_lookup


def pop_ready_coordination_edges(
    execution_group_id,
    completed_group_ids,
    coordination_groups,
    coordination_group_order,
    processed_coordination_edges,
):
    edge_lookup = build_coordination_edge_lookup(coordination_groups, coordination_group_order)
    completed_group_ids = {int(group_id) for group_id in completed_group_ids}
    processed_coordination_edges = processed_coordination_edges if processed_coordination_edges is not None else set()
    ready_edges = []
    for edge in edge_lookup.get(int(execution_group_id), []):
        left_group_id = int(edge["left_group_id"])
        right_group_id = int(edge["right_group_id"])
        edge_key = tuple(edge["edge_key"])
        if edge_key in processed_coordination_edges:
            continue
        if left_group_id in completed_group_ids and right_group_id in completed_group_ids:
            processed_coordination_edges.add(edge_key)
            ready_edges.append(edge)
    return ready_edges


def blend_overlapping_elements(best_individual, original_best_individual, overlapping_elements, previous_delta, current_delta):
    if len(overlapping_elements) == 0:
        return
    weight = safe_overlap_weight(previous_delta, current_delta)
    best_individual[overlapping_elements] = (
        weight * best_individual[overlapping_elements]
        + (1.0 - weight) * original_best_individual[overlapping_elements]
    )


def _finite_conflict_mean(overlap_indices, conflict_prior):
    overlap_indices = np.asarray(overlap_indices, dtype=int).reshape(-1)
    if overlap_indices.size == 0 or conflict_prior is None:
        return 0.0
    conflict_prior = np.asarray(conflict_prior, dtype=float).reshape(-1)
    valid_indices = overlap_indices[(overlap_indices >= 0) & (overlap_indices < conflict_prior.size)]
    if valid_indices.size == 0:
        return 0.0
    values = conflict_prior[valid_indices]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _build_overlap_blend_diagnostics(
    blend_strategy,
    blend_mode,
    overlap_count,
    previous_delta,
    current_delta,
    raw_weight=float("nan"),
    weight=float("nan"),
    conflict_mean=0.0,
    damping=1.0,
    was_skipped=False,
    applied_update=False,
):
    return {
        "blend_strategy": str(blend_strategy),
        "blend_mode": str(blend_mode),
        "overlap_count": int(overlap_count),
        "previous_delta": float(previous_delta),
        "current_delta": float(current_delta),
        "raw_weight": float(raw_weight),
        "weight": float(weight),
        "conflict_mean": float(conflict_mean),
        "damping": float(damping),
        "was_skipped": bool(was_skipped),
        "applied_update": bool(applied_update),
    }


def safe_conflict_aware_overlap_blend(
    best_individual,
    original_best_individual,
    overlapping_elements,
    previous_delta,
    current_delta,
    conflict_prior=None,
    eps_delta=1e-12,
    min_weight=0.2,
    max_weight=0.8,
    conflict_gamma=0.5,
    min_damping=0.3,
):
    overlap_indices = np.asarray(overlapping_elements, dtype=int).reshape(-1)
    blend_strategy = "safe_conflict" if conflict_prior is not None else "safe_delta"
    prev_delta = float(previous_delta)
    curr_delta = float(current_delta)
    conflict_mean = _finite_conflict_mean(overlap_indices, conflict_prior)

    if overlap_indices.size == 0:
        return _build_overlap_blend_diagnostics(
            blend_strategy=blend_strategy,
            blend_mode="no_overlap",
            overlap_count=0,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=conflict_mean,
            was_skipped=True,
            applied_update=False,
        )

    if not np.isfinite(prev_delta) or not np.isfinite(curr_delta):
        return _build_overlap_blend_diagnostics(
            blend_strategy=blend_strategy,
            blend_mode="skip_nonfinite",
            overlap_count=overlap_indices.size,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=conflict_mean,
            was_skipped=True,
            applied_update=False,
        )

    if prev_delta > eps_delta and curr_delta > eps_delta:
        raw_weight = curr_delta / (prev_delta + curr_delta + eps_delta)
        blend_mode = "both_positive"
    elif curr_delta > eps_delta and prev_delta <= eps_delta:
        raw_weight = 1.0
        blend_mode = "current_only"
    elif prev_delta > eps_delta and curr_delta <= eps_delta:
        raw_weight = 0.0
        blend_mode = "previous_only"
    else:
        return _build_overlap_blend_diagnostics(
            blend_strategy=blend_strategy,
            blend_mode="skip_no_positive_delta",
            overlap_count=overlap_indices.size,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=conflict_mean,
            was_skipped=True,
            applied_update=False,
        )

    if blend_mode == "both_positive":
        weight = float(np.clip(raw_weight, min_weight, max_weight))
    else:
        weight = float(np.clip(raw_weight, 0.0, 1.0))
    damping = 1.0
    if conflict_prior is not None:
        damping = float(np.clip(1.0 - float(conflict_gamma) * conflict_mean, min_damping, 1.0))

    original_values = np.asarray(original_best_individual[overlap_indices], dtype=float)
    current_values = np.asarray(best_individual[overlap_indices], dtype=float)
    raw_blend = weight * current_values + (1.0 - weight) * original_values
    best_individual[overlap_indices] = original_values + damping * (raw_blend - original_values)
    return _build_overlap_blend_diagnostics(
        blend_strategy=blend_strategy,
        blend_mode=blend_mode,
        overlap_count=overlap_indices.size,
        previous_delta=prev_delta,
        current_delta=curr_delta,
        raw_weight=raw_weight,
        weight=weight,
        conflict_mean=conflict_mean,
        damping=damping,
        was_skipped=False,
        applied_update=True,
    )


def apply_overlap_blend_strategy(
    best_individual,
    original_best_individual,
    overlapping_elements,
    previous_delta,
    current_delta,
    overlap_blend_mode="original",
    conflict_prior=None,
    blend_config=None,
):
    overlap_indices = np.asarray(overlapping_elements, dtype=int).reshape(-1)
    prev_delta = float(previous_delta)
    curr_delta = float(current_delta)
    mode = str(overlap_blend_mode or "original").strip().lower()
    conflict_mean = _finite_conflict_mean(overlap_indices, conflict_prior)
    eps_delta = getattr(blend_config, "overlap_blend_eps_delta", 1e-12)
    min_weight = getattr(blend_config, "overlap_blend_min_weight", 0.2)
    max_weight = getattr(blend_config, "overlap_blend_max_weight", 0.8)
    conflict_gamma = getattr(blend_config, "overlap_blend_conflict_gamma", 0.5)
    min_damping = getattr(blend_config, "overlap_blend_min_damping", 0.3)

    if mode == "no_blend":
        return _build_overlap_blend_diagnostics(
            blend_strategy="no_blend",
            blend_mode="no_blend" if overlap_indices.size else "no_overlap",
            overlap_count=overlap_indices.size,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=conflict_mean,
            damping=0.0,
            was_skipped=False,
            applied_update=False,
        )

    if mode == "equation8_correct":
        if overlap_indices.size == 0:
            return _build_overlap_blend_diagnostics(
                blend_strategy="equation8_correct",
                blend_mode="no_overlap",
                overlap_count=0,
                previous_delta=prev_delta,
                current_delta=curr_delta,
                conflict_mean=conflict_mean,
                was_skipped=True,
                applied_update=False,
            )
        weight = safe_overlap_weight(curr_delta, prev_delta)
        best_individual[overlap_indices] = (
            weight * best_individual[overlap_indices]
            + (1.0 - weight) * original_best_individual[overlap_indices]
        )
        return _build_overlap_blend_diagnostics(
            blend_strategy="equation8_correct",
            blend_mode="equation8_correct",
            overlap_count=overlap_indices.size,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            raw_weight=weight,
            weight=weight,
            conflict_mean=conflict_mean,
            damping=1.0,
            was_skipped=False,
            applied_update=True,
        )

    if mode == "safe_delta":
        return safe_conflict_aware_overlap_blend(
            best_individual,
            original_best_individual,
            overlap_indices,
            prev_delta,
            curr_delta,
            conflict_prior=None,
            eps_delta=eps_delta,
            min_weight=min_weight,
            max_weight=max_weight,
            conflict_gamma=conflict_gamma,
            min_damping=min_damping,
        )

    if mode == "safe_conflict":
        return safe_conflict_aware_overlap_blend(
            best_individual,
            original_best_individual,
            overlap_indices,
            prev_delta,
            curr_delta,
            conflict_prior=conflict_prior,
            eps_delta=eps_delta,
            min_weight=min_weight,
            max_weight=max_weight,
            conflict_gamma=conflict_gamma,
            min_damping=min_damping,
        )

    if overlap_indices.size == 0:
        return _build_overlap_blend_diagnostics(
            blend_strategy="original",
            blend_mode="no_overlap",
            overlap_count=0,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=conflict_mean,
            was_skipped=True,
            applied_update=False,
        )

    weight = safe_overlap_weight(prev_delta, curr_delta)
    blend_overlapping_elements(
        best_individual,
        original_best_individual,
        overlap_indices,
        prev_delta,
        curr_delta,
    )
    return _build_overlap_blend_diagnostics(
        blend_strategy="original",
        blend_mode="original",
        overlap_count=overlap_indices.size,
        previous_delta=prev_delta,
        current_delta=curr_delta,
        raw_weight=weight,
        weight=weight,
        conflict_mean=conflict_mean,
        damping=1.0,
        was_skipped=False,
        applied_update=True,
    )


def apply_coordination_edge_blend(
    best_individual,
    previous_proposal_individual,
    current_proposal_individual,
    current_original_individual,
    overlapping_elements,
    previous_delta,
    current_delta,
    overlap_blend_mode="original",
    conflict_prior=None,
    blend_config=None,
    use_pairwise_endpoint_values=False,
):
    overlap_indices = np.asarray(overlapping_elements, dtype=int).reshape(-1)
    if not use_pairwise_endpoint_values:
        edge_current = np.asarray(current_proposal_individual, dtype=float).copy()
        diagnostics = apply_overlap_blend_strategy(
            edge_current,
            current_original_individual,
            overlap_indices,
            previous_delta,
            current_delta,
            overlap_blend_mode=overlap_blend_mode,
            conflict_prior=conflict_prior,
            blend_config=blend_config,
        )
        if overlap_indices.size > 0:
            best_individual[overlap_indices] = edge_current[overlap_indices]
        return diagnostics

    prev_delta = float(previous_delta)
    curr_delta = float(current_delta)
    mode = str(overlap_blend_mode or "original").strip().lower()
    if overlap_indices.size == 0:
        return _build_overlap_blend_diagnostics(
            blend_strategy=mode,
            blend_mode="no_overlap",
            overlap_count=0,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=_finite_conflict_mean(overlap_indices, conflict_prior),
            was_skipped=True,
            applied_update=False,
        )
    if mode == "no_blend":
        return _build_overlap_blend_diagnostics(
            blend_strategy="no_blend",
            blend_mode="no_blend",
            overlap_count=overlap_indices.size,
            previous_delta=prev_delta,
            current_delta=curr_delta,
            conflict_mean=_finite_conflict_mean(overlap_indices, conflict_prior),
            damping=0.0,
            was_skipped=False,
            applied_update=False,
        )
    if mode == "equation8_correct":
        weight = safe_overlap_weight(curr_delta, prev_delta)
    else:
        mode = "original"
        weight = safe_overlap_weight(prev_delta, curr_delta)
    previous_values = np.asarray(previous_proposal_individual[overlap_indices], dtype=float)
    current_values = np.asarray(current_proposal_individual[overlap_indices], dtype=float)
    best_individual[overlap_indices] = weight * current_values + (1.0 - weight) * previous_values
    return _build_overlap_blend_diagnostics(
        blend_strategy=mode,
        blend_mode=f"pairwise_{mode}",
        overlap_count=overlap_indices.size,
        previous_delta=prev_delta,
        current_delta=curr_delta,
        raw_weight=weight,
        weight=weight,
        conflict_mean=_finite_conflict_mean(overlap_indices, conflict_prior),
        damping=1.0,
        was_skipped=False,
        applied_update=True,
    )


def best_so_far_curve(curve):
    best = math.inf
    output = []
    for value in curve:
        numeric = float(value)
        if numeric < best:
            best = numeric
        output.append(best)
    return output


def checkpoint_fieldnames(record_fes):
    return [f"best_at_{int(point)}" for point in (record_fes or [])]


def best_at_record_points(curve, record_fes):
    if not record_fes:
        return {}
    best_curve = best_so_far_curve(curve)
    values = {}
    for point in record_fes:
        index = int(point) - 1
        field = f"best_at_{int(point)}"
        values[field] = best_curve[index] if 0 <= index < len(best_curve) else float("nan")
    return values


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_or_nan(values):
    numeric = [float(value) for value in values if np.isfinite(_to_float(value))]
    return float(np.mean(numeric)) if numeric else float("nan")


def _load_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rollback_ratio(diagnostics):
    if not diagnostics:
        return 0.0
    return float(
        np.mean(
            [
                str(row.get("validation_status", "")).strip().lower() == "rejected"
                or str(row.get("rollback", "")).strip().lower() == "true"
                for row in diagnostics
            ]
        )
    )


def conflict_mean(diagnostics):
    return _mean_or_nan([row.get("conflict") for row in diagnostics])


def build_run_detail_row(
    problem_code,
    method,
    seed,
    curve,
    runtime,
    status,
    diagnostics=None,
    record_fes=None,
    tfes=None,
    blend_strategy=None,
    method_label=None,
):
    diagnostics = diagnostics or []
    if curve:
        final_fitness = float(curve[-1])
        best_fitness = float(np.min(curve))
        fe_used = int(len(curve))
    else:
        final_fitness = float("nan")
        best_fitness = float("nan")
        fe_used = 0
    row = {
        "problem": str(problem_code).upper(),
        "method": str(method_label or canonicalize_method(method)),
        "tfes": int(tfes) if tfes is not None else "",
        "blend_strategy": str(blend_strategy or "original"),
        "seed": int(seed),
        "final_fitness": final_fitness,
        "best_fitness": best_fitness,
        "fe_used": fe_used,
        "runtime": float(runtime),
        "diagnostics_count": int(len(diagnostics)),
        "rollback_ratio": rollback_ratio(diagnostics),
        "conflict_mean": conflict_mean(diagnostics),
        "status": str(status),
    }
    row.update(best_at_record_points(curve, record_fes))
    return row


def build_summary_row(
    problem_code,
    method,
    seed,
    curve,
    runtime,
    status,
    record_fes=None,
    tfes=None,
    blend_strategy=None,
    method_label=None,
):
    return build_run_detail_row(
        problem_code,
        method,
        seed,
        curve,
        runtime,
        status,
        diagnostics=[],
        record_fes=record_fes,
        tfes=tfes,
        blend_strategy=blend_strategy,
        method_label=method_label,
    )


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def append_csv_rows(path, rows, fieldnames):
    for row in rows:
        append_csv_row(path, row, fieldnames)


def ensure_csv_header(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def save_group_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_TRACE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in GROUP_TRACE_FIELDNAMES})


def save_overlap_blend_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERLAP_BLEND_TRACE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in OVERLAP_BLEND_TRACE_FIELDNAMES})


def save_shared_variable_proposal_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHARED_VARIABLE_PROPOSAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SHARED_VARIABLE_PROPOSAL_FIELDNAMES})


def save_shared_variable_fusion_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHARED_VARIABLE_FUSION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SHARED_VARIABLE_FUSION_FIELDNAMES})


def save_validated_coordination_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VALIDATED_COORDINATION_TRACE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in VALIDATED_COORDINATION_TRACE_FIELDNAMES})


def save_coordination_selector_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COORDINATION_SELECTOR_TRACE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in COORDINATION_SELECTOR_TRACE_FIELDNAMES})


def save_optimizer_state_trace_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZER_STATE_TRACE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in OPTIMIZER_STATE_TRACE_FIELDNAMES})


def build_group_variable_index_maps(grouping_result, source_group_ids=None):
    source_group_ids = list(source_group_ids) if source_group_ids is not None else list(range(len(grouping_result or [])))
    source_group_dims = {}
    group_variable_index = {}
    variable_to_group_ids = {}
    for position, dims in enumerate(grouping_result or []):
        source_group_id = int(source_group_ids[position]) if position < len(source_group_ids) else int(position)
        normalized_dims = [int(var_id) for var_id in dims]
        source_group_dims[source_group_id] = normalized_dims
        local_index = {}
        for local_idx, var_id in enumerate(normalized_dims):
            local_index[int(var_id)] = int(local_idx)
            variable_to_group_ids.setdefault(int(var_id), []).append(source_group_id)
        group_variable_index[source_group_id] = local_index
    for var_id, group_ids in list(variable_to_group_ids.items()):
        variable_to_group_ids[int(var_id)] = sorted(set(int(group_id) for group_id in group_ids))
    return {
        "source_group_dims": source_group_dims,
        "group_variable_index": group_variable_index,
        "variable_to_group_ids": variable_to_group_ids,
    }


def sync_omega_cache_for_variables(
    omega_cache,
    best_individual,
    variable_ids,
    group_variable_index,
    variable_to_group_ids,
    source_group_dims=None,
    best_before_individual=None,
    cycle_id=None,
    pass_id=None,
    scheduled_position=None,
):
    rows = []
    if omega_cache is None:
        return rows
    best_individual = np.asarray(best_individual, dtype=float)
    best_before_individual = (
        np.asarray(best_before_individual, dtype=float)
        if best_before_individual is not None
        else best_individual
    )
    seen_pairs = set()
    variable_iterable = [] if variable_ids is None else np.asarray(variable_ids, dtype=int).reshape(-1).tolist()
    for var_id in sorted(set(int(value) for value in variable_iterable)):
        if var_id < 0 or var_id >= best_individual.size:
            continue
        for group_id in variable_to_group_ids.get(int(var_id), []):
            group_id = int(group_id)
            if group_id not in omega_cache:
                continue
            local_idx = group_variable_index.get(group_id, {}).get(int(var_id))
            if local_idx is None:
                continue
            pair_key = (group_id, int(var_id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            mean_vector = np.asarray(omega_cache[group_id], dtype=float).copy()
            if int(local_idx) < 0 or int(local_idx) >= mean_vector.size:
                continue
            mean_before = float(mean_vector[int(local_idx)])
            best_before = (
                float(best_before_individual[int(var_id)])
                if 0 <= int(var_id) < best_before_individual.size
                else float("nan")
            )
            best_after = float(best_individual[int(var_id)])
            mismatch_before = float(abs(mean_before - best_after))
            mean_vector[int(local_idx)] = best_after
            omega_cache[group_id] = mean_vector
            rows.append(
                {
                    "cycle_id": "" if cycle_id is None else int(cycle_id),
                    "pass_id": "" if pass_id is None else int(pass_id),
                    "scheduled_position": "" if scheduled_position is None else int(scheduled_position),
                    "group_id": group_id,
                    "source_group_id": group_id,
                    "event": "coordination_sync",
                    "var_id": int(var_id),
                    "mean_init_source": "omega_cache",
                    "mean_before_coord_overlap": mean_before,
                    "mean_after_coord_overlap": best_after,
                    "best_before_coord_overlap": best_before,
                    "best_after_coord_overlap": best_after,
                    "mean_coord_mismatch": mismatch_before,
                    "mean_coord_mismatch_after_sync": 0.0,
                    "optimizer_reinitialized": True,
                }
            )
    return rows


def summarize_optimizer_state_rows(rows):
    rows = list(rows or [])
    init_rows = [row for row in rows if str(row.get("event", "")) == "mean_init"]
    sync_rows = [row for row in rows if str(row.get("event", "")) == "coordination_sync"]
    mismatch_values = [
        _to_float(row.get("mean_coord_mismatch"))
        for row in sync_rows
        if np.isfinite(_to_float(row.get("mean_coord_mismatch")))
    ]
    after_sync_values = [
        _to_float(row.get("mean_coord_mismatch_after_sync"))
        for row in sync_rows
        if np.isfinite(_to_float(row.get("mean_coord_mismatch_after_sync")))
    ]
    sources = sorted(set(str(row.get("mean_init_source", "")) for row in init_rows if str(row.get("mean_init_source", ""))))
    return {
        "optimizer_state_row_count": int(len(rows)),
        "mean_init_count": int(len(init_rows)),
        "omega_sync_count": int(len(sync_rows)),
        "mean_init_sources": sources,
        "optimizer_reinitialized": bool(all(bool(row.get("optimizer_reinitialized")) for row in init_rows)) if init_rows else True,
        "mean_coord_mismatch_max": float(np.max(mismatch_values)) if mismatch_values else 0.0,
        "mean_coord_mismatch_mean": float(np.mean(mismatch_values)) if mismatch_values else 0.0,
        "mean_coord_mismatch_after_sync_max": float(np.max(after_sync_values)) if after_sync_values else 0.0,
        "mean_coord_mismatch_after_sync_mean": float(np.mean(after_sync_values)) if after_sync_values else 0.0,
    }


def build_shared_variable_visibility_audit(grouping_result, adjacent_overlapping_elements, coordination_mode="adjacent", overlap_hypergraph=None):
    coordination_mode = str(coordination_mode or "adjacent").strip().lower()
    overlap_hypergraph = overlap_hypergraph or build_overlap_hypergraph(grouping_result)
    true_overlap_vars = sorted(int(var_id) for var_id in overlap_hypergraph.get("overlap_vars", []))
    adjacent_visible_overlap_vars = sorted(
        {
            int(var_id)
            for overlap in (adjacent_overlapping_elements or [])
            for var_id in overlap
        }
    )
    if is_hypergraph_pass_end_mode(coordination_mode):
        coordinated_overlap_vars = true_overlap_vars
    elif coordination_mode == "no_coordination":
        coordinated_overlap_vars = []
    else:
        coordinated_overlap_vars = adjacent_visible_overlap_vars
    true_overlap_count = int(len(true_overlap_vars))
    coordinated_overlap_count = int(len(coordinated_overlap_vars))
    adjacent_visible_count = int(len(adjacent_visible_overlap_vars))
    return {
        "coordination_mode": coordination_mode,
        "true_overlap_var_count": true_overlap_count,
        "adjacent_visible_overlap_var_count": adjacent_visible_count,
        "coordinated_overlap_var_count": coordinated_overlap_count,
        "coordinated_overlap_ratio": float(coordinated_overlap_count / true_overlap_count) if true_overlap_count else 0.0,
        "true_overlap_vars": true_overlap_vars,
        "adjacent_visible_overlap_vars": adjacent_visible_overlap_vars,
        "coordinated_overlap_vars": coordinated_overlap_vars,
    }


def build_variable_ranges(lower_boundary, upper_boundary, dimension):
    dimension = max(0, int(dimension))
    if dimension <= 0:
        return np.empty((0,), dtype=float)
    lower = np.asarray(lower_boundary if lower_boundary is not None else np.zeros(dimension), dtype=float).reshape(-1)
    upper = np.asarray(upper_boundary if upper_boundary is not None else np.ones(dimension), dtype=float).reshape(-1)
    if lower.size == 1:
        lower = np.full(dimension, float(lower[0]), dtype=float)
    elif lower.size != dimension:
        lower = np.resize(lower, dimension).astype(float)
    if upper.size == 1:
        upper = np.full(dimension, float(upper[0]), dtype=float)
    elif upper.size != dimension:
        upper = np.resize(upper, dimension).astype(float)
    ranges = np.abs(upper - lower)
    return np.where(ranges <= 1e-12, 1.0, ranges)


def collect_group_overlap_variable_proposals(
    best_individual,
    overlap_variables,
    group_id,
    delta,
    fitness_before,
    fitness_after,
    cycle_id,
    scheduled_position,
):
    proposal_rows = []
    for var_id in sorted({int(var_id) for var_id in (overlap_variables or [])}):
        proposal_rows.append(
            {
                "cycle_id": int(cycle_id),
                "scheduled_position": int(scheduled_position),
                "group_id": int(group_id),
                "var_id": int(var_id),
                "proposal_value": float(best_individual[int(var_id)]),
                "delta": float(delta),
                "fitness_before": float(fitness_before),
                "fitness_after": float(fitness_after),
            }
        )
    return proposal_rows


def summarize_arac_relation_history_entry(history_entry):
    history_entry = dict(history_entry or {})
    attempt_count = int(history_entry.get("attempt_count", 0) or 0)
    accept_count = int(history_entry.get("accept_count", 0) or 0)
    delta_sum = float(history_entry.get("delta_sum", 0.0) or 0.0)
    reject_streak = int(history_entry.get("reject_streak", 0) or 0)
    positive_delta_count = int(history_entry.get("positive_delta_count", 0) or 0)
    probe_attempt_count = int(history_entry.get("probe_attempt_count", 0) or 0)
    probe_accept_count = int(history_entry.get("probe_accept_count", 0) or 0)
    probe_delta_sum = float(history_entry.get("probe_delta_sum", 0.0) or 0.0)
    probe_positive_delta_count = int(history_entry.get("probe_positive_delta_count", 0) or 0)
    return {
        "relation_attempt_count": int(attempt_count),
        "relation_accept_count": int(accept_count),
        "relation_accept_rate": float(accept_count / attempt_count) if attempt_count else 0.0,
        "relation_mean_validation_delta": float(delta_sum / attempt_count) if attempt_count else 0.0,
        "relation_reject_streak": int(reject_streak),
        "relation_positive_delta_count": int(positive_delta_count),
        "relation_positive_delta_rate": float(positive_delta_count / attempt_count) if attempt_count else 0.0,
        "probe_attempt_count": int(probe_attempt_count),
        "probe_accept_count": int(probe_accept_count),
        "probe_accept_rate": float(probe_accept_count / probe_attempt_count) if probe_attempt_count else 0.0,
        "probe_mean_validation_delta": float(probe_delta_sum / probe_attempt_count) if probe_attempt_count else 0.0,
        "probe_positive_delta_count": int(probe_positive_delta_count),
        "probe_positive_delta_rate": float(probe_positive_delta_count / probe_attempt_count) if probe_attempt_count else 0.0,
    }


def should_disable_arac_relation(
    history_entry,
    history_min_attempts=2,
    disable_accept_rate_threshold=0.0,
    disable_mean_delta_threshold=0.0,
    disable_reject_streak=2,
):
    summary = summarize_arac_relation_history_entry(history_entry)
    attempt_count = int(summary["relation_attempt_count"])
    if attempt_count < int(history_min_attempts):
        return False, ""
    if int(summary["relation_reject_streak"]) >= int(disable_reject_streak):
        return True, "history_rejected"
    if float(summary["relation_accept_rate"]) <= float(disable_accept_rate_threshold):
        return True, "history_low_accept_rate"
    if float(summary["relation_mean_validation_delta"]) <= float(disable_mean_delta_threshold):
        return True, "history_nonpositive_delta"
    return False, ""


def _arac_recent_attempts(history_entry, recent_window):
    history_entry = dict(history_entry or {})
    recent_window = max(1, int(recent_window))
    recent_attempts = list(history_entry.get("recent_attempts", []) or [])
    normalized = []
    for item in recent_attempts[-recent_window:]:
        item = dict(item or {})
        delta = _to_float(item.get("delta"))
        if not np.isfinite(delta):
            delta = 0.0
        normalized.append(
            {
                "accepted": bool(item.get("accepted")),
                "delta": float(delta),
            }
        )
    return normalized


def _arac_recent_probe_attempts(history_entry, recent_window):
    history_entry = dict(history_entry or {})
    recent_window = max(1, int(recent_window))
    recent_attempts = list(history_entry.get("probe_recent_attempts", []) or [])
    normalized = []
    for item in recent_attempts[-recent_window:]:
        item = dict(item or {})
        delta = _to_float(item.get("delta"))
        if not np.isfinite(delta):
            delta = 0.0
        normalized.append(
            {
                "accepted": bool(item.get("accepted")),
                "delta": float(delta),
            }
        )
    return normalized


def _phase_rank(value):
    return {"early": 0, "middle": 1, "late": 2}.get(str(value or "early").strip().lower(), 0)


def should_probe_arac_relation(
    history_blocks,
    phase="early",
    pass_id=0,
    probe_enabled=False,
    probe_every_n_pass=2,
    probe_min_phase="middle",
    remaining_probe_budget=0,
):
    if not bool(probe_enabled):
        return {"should_probe": False, "probe_reason": "probe_disabled"}
    if not bool(history_blocks):
        return {"should_probe": False, "probe_reason": "not_history_disabled"}
    if _phase_rank(phase) < _phase_rank(probe_min_phase):
        return {"should_probe": False, "probe_reason": "phase_not_probeable"}
    if int(remaining_probe_budget) <= 0:
        return {"should_probe": False, "probe_reason": "probe_budget_exhausted"}
    every_n = max(1, int(probe_every_n_pass))
    if int(pass_id) % every_n != 0:
        return {"should_probe": False, "probe_reason": "probe_frequency_skipped"}
    return {"should_probe": True, "probe_reason": "probe_selected"}


def default_targeted_probe_summary(reason="targeted_probe_not_evaluated"):
    return {
        "should_probe": False,
        "targeted_probe_candidate": False,
        "targeted_probe_signature_matched": False,
        "targeted_probe_reason": str(reason),
        "targeted_probe_score": 0.0,
    }


def evaluate_arac_targeted_probe_signature(
    *,
    history_blocks=False,
    proposal_conflict=False,
    phase="early",
    targeted_probe_phase="middle",
    proposal_support=0,
    proposal_std_ratio=0.0,
    relation_summary=None,
    targeted_probe_min_support=2,
    targeted_probe_min_relation_attempts=1,
    targeted_probe_min_accept_rate=0.3,
    targeted_probe_min_positive_delta_rate=0.3,
    targeted_probe_min_relation_delta=0.0,
    targeted_probe_min_proposal_std_ratio=0.00125,
    targeted_probe_max_proposal_std_ratio=0.0045,
):
    relation_summary = dict(relation_summary or {})
    phase = str(phase or "early").strip().lower()
    targeted_probe_phase = str(targeted_probe_phase or "middle").strip().lower()
    if phase != targeted_probe_phase:
        return default_targeted_probe_summary("targeted_phase_mismatch")
    if not bool(history_blocks) and not bool(proposal_conflict):
        return default_targeted_probe_summary("targeted_not_disable_or_freeze")
    if int(proposal_support) < int(targeted_probe_min_support):
        return default_targeted_probe_summary("targeted_support_below_threshold")
    proposal_std_ratio = float(proposal_std_ratio) if np.isfinite(float(proposal_std_ratio)) else 0.0
    if proposal_std_ratio < float(targeted_probe_min_proposal_std_ratio):
        return default_targeted_probe_summary("targeted_proposal_std_below_signature")
    if proposal_std_ratio > float(targeted_probe_max_proposal_std_ratio):
        return default_targeted_probe_summary("targeted_proposal_std_above_signature")

    relation_attempts = int(relation_summary.get("relation_attempt_count", 0) or 0)
    relation_accept_rate = float(relation_summary.get("relation_accept_rate", 0.0) or 0.0)
    relation_delta_mean = float(relation_summary.get("relation_mean_validation_delta", 0.0) or 0.0)
    relation_positive_delta_rate = float(relation_summary.get("relation_positive_delta_rate", 0.0) or 0.0)
    if relation_attempts < int(targeted_probe_min_relation_attempts):
        return default_targeted_probe_summary("targeted_attempts_below_threshold")
    if relation_delta_mean <= float(targeted_probe_min_relation_delta):
        return default_targeted_probe_summary("targeted_rolling_delta_nonpositive")
    if relation_accept_rate < float(targeted_probe_min_accept_rate):
        return default_targeted_probe_summary("targeted_accept_rate_below_signature")
    if relation_positive_delta_rate < float(targeted_probe_min_positive_delta_rate):
        return default_targeted_probe_summary("targeted_positive_delta_rate_below_signature")

    score = float(
        proposal_std_ratio
        + 0.01 * relation_accept_rate
        + 0.01 * relation_positive_delta_rate
        + 1e-12 * max(0.0, relation_delta_mean)
    )
    return {
        "should_probe": False,
        "targeted_probe_candidate": False,
        "targeted_probe_signature_matched": True,
        "targeted_probe_reason": "targeted_signature_matched",
        "targeted_probe_score": float(score),
    }


def build_arac_targeted_probe_summaries(
    best_individual,
    proposals_by_var,
    overlap_vars,
    variable_ranges,
    arac_relation_history=None,
    phase="early",
    pass_id=0,
    probe_enabled=False,
    targeted_probe_enabled=False,
    probe_every_n_pass=2,
    probe_max_per_pass=1,
    probe_min_phase="middle",
    targeted_probe_phase="middle",
    targeted_probe_min_support=2,
    targeted_probe_min_relation_attempts=1,
    targeted_probe_min_accept_rate=0.3,
    targeted_probe_min_positive_delta_rate=0.3,
    targeted_probe_min_relation_delta=0.0,
    targeted_probe_min_proposal_std_ratio=0.00125,
    targeted_probe_max_proposal_std_ratio=0.0045,
    selective_min_positive_proposals=2,
    selective_max_proposal_std_ratio=0.00125,
    history_min_attempts=2,
    disable_accept_rate_threshold=0.0,
    disable_mean_delta_threshold=0.0,
    disable_reject_streak=2,
):
    summaries = {}
    if not bool(targeted_probe_enabled):
        return summaries
    overlap_vars = [int(value) for value in (overlap_vars or [])]
    if not bool(probe_enabled):
        return {var_id: default_targeted_probe_summary("targeted_probe_disabled") for var_id in overlap_vars}
    if _phase_rank(phase) < _phase_rank(probe_min_phase):
        return {var_id: default_targeted_probe_summary("targeted_phase_not_probeable") for var_id in overlap_vars}
    every_n = max(1, int(probe_every_n_pass))
    frequency_allows_probe = int(pass_id) % every_n == 0
    candidates = []
    variable_ranges = np.asarray(variable_ranges, dtype=float).reshape(-1)
    best_individual = np.asarray(best_individual, dtype=float).reshape(-1)
    for var_id in overlap_vars:
        variable_rows = list(proposals_by_var.get(int(var_id), []) or [])
        finite_rows = [
            row
            for row in variable_rows
            if np.isfinite(_to_float(row.get("proposal_value"))) and np.isfinite(_to_float(row.get("delta")))
        ]
        positive_rows = [row for row in finite_rows if float(row.get("delta", 0.0)) > 0.0]
        if len(positive_rows) < int(selective_min_positive_proposals):
            summaries[var_id] = default_targeted_probe_summary("targeted_support_below_threshold")
            continue
        old_value = float(best_individual[int(var_id)]) if 0 <= int(var_id) < best_individual.size else 0.0
        variable_range = float(variable_ranges[int(var_id)]) if 0 <= int(var_id) < variable_ranges.size else 1.0
        if not np.isfinite(variable_range) or variable_range <= 0.0:
            variable_range = 1.0
        values = np.asarray([float(row["proposal_value"]) for row in positive_rows], dtype=float)
        deltas = np.asarray([float(row["delta"]) for row in positive_rows], dtype=float)
        proposal_value_std = float(np.std(values)) if values.size else 0.0
        proposal_std_ratio = float(proposal_value_std / variable_range) if variable_range > 0.0 else 0.0
        weight_sum = float(np.sum(deltas))
        if np.isfinite(weight_sum) and weight_sum > 1e-12:
            fused_value = float(np.sum((deltas / weight_sum) * values))
        else:
            fused_value = float(old_value)
        raw_update_ratio = float(abs(fused_value - old_value) / variable_range) if variable_range > 0.0 else 0.0
        relation_history_entry = dict((arac_relation_history or {}).get(int(var_id), {}))
        relation_summary = summarize_arac_relation_history_entry(relation_history_entry)
        history_blocks, _ = should_disable_arac_relation(
            relation_history_entry,
            history_min_attempts=history_min_attempts,
            disable_accept_rate_threshold=disable_accept_rate_threshold,
            disable_mean_delta_threshold=disable_mean_delta_threshold,
            disable_reject_streak=disable_reject_streak,
        )
        proposal_conflict = bool(
            np.isfinite(proposal_std_ratio)
            and proposal_std_ratio > float(selective_max_proposal_std_ratio)
        )
        summary = evaluate_arac_targeted_probe_signature(
            history_blocks=history_blocks,
            proposal_conflict=proposal_conflict,
            phase=phase,
            targeted_probe_phase=targeted_probe_phase,
            proposal_support=len(positive_rows),
            proposal_std_ratio=proposal_std_ratio,
            relation_summary=relation_summary,
            targeted_probe_min_support=targeted_probe_min_support,
            targeted_probe_min_relation_attempts=targeted_probe_min_relation_attempts,
            targeted_probe_min_accept_rate=targeted_probe_min_accept_rate,
            targeted_probe_min_positive_delta_rate=targeted_probe_min_positive_delta_rate,
            targeted_probe_min_relation_delta=targeted_probe_min_relation_delta,
            targeted_probe_min_proposal_std_ratio=targeted_probe_min_proposal_std_ratio,
            targeted_probe_max_proposal_std_ratio=targeted_probe_max_proposal_std_ratio,
        )
        summary["targeted_probe_score"] = float(summary["targeted_probe_score"]) + 1e-3 * float(raw_update_ratio)
        summaries[var_id] = summary
        if bool(summary["targeted_probe_signature_matched"]):
            candidates.append((float(summary["targeted_probe_score"]), int(var_id)))

    if not frequency_allows_probe:
        for _, var_id in candidates:
            summaries[var_id]["targeted_probe_reason"] = "targeted_probe_frequency_skipped"
        return summaries

    selected = {
        var_id
        for _, var_id in sorted(candidates, key=lambda item: (-item[0], item[1]))[: max(0, int(probe_max_per_pass))]
    }
    for _, var_id in candidates:
        if var_id in selected:
            summaries[var_id]["should_probe"] = True
            summaries[var_id]["targeted_probe_candidate"] = True
            summaries[var_id]["targeted_probe_reason"] = "targeted_probe_selected"
        else:
            summaries[var_id]["targeted_probe_reason"] = "targeted_probe_budget_exhausted"
    return summaries


def stable_random_probe_score(var_id, pass_id=0, seed=0, phase="middle"):
    payload = f"{int(seed)}|{int(pass_id)}|{str(phase)}|{int(var_id)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16)


def random_same_budget_remaining_probe_budget(state, phase, total_budget):
    phase = str(phase or "middle").strip().lower()
    total_budget = max(0, int(total_budget))
    if state is None:
        return total_budget
    used_by_phase = state.setdefault("used_by_phase", {})
    used = max(0, int(used_by_phase.get(phase, 0) or 0))
    return max(0, total_budget - used)


def mark_random_same_budget_probe_used(state, phase, count=1):
    if state is None:
        return
    phase = str(phase or "middle").strip().lower()
    used_by_phase = state.setdefault("used_by_phase", {})
    used_by_phase[phase] = max(0, int(used_by_phase.get(phase, 0) or 0)) + max(0, int(count))


def arac_lite_phase_for_cycle(cycle_id, expected_pass_count):
    cycle_id = max(0, int(cycle_id))
    expected_pass_count = max(1, int(expected_pass_count))
    max_pass_id = max(0, expected_pass_count - 1)
    if max_pass_id <= 0:
        return "early"
    ratio = float(cycle_id / max_pass_id)
    if ratio <= 1.0 / 3.0:
        return "early"
    if ratio <= 2.0 / 3.0:
        return "middle"
    return "late"


def estimate_arac_lite_cc_pass_count(max_fes, current_fes, group_count, info_aware_config=None, has_overlap=False):
    group_count = max(1, int(group_count))
    remaining_fes = max(0, int(max_fes) - int(current_fes))
    if remaining_fes <= 0:
        return 1
    min_passes = max(1, int(getattr(info_aware_config, "cc_min_passes", 1) or 1))
    cc_pass_group_fes = getattr(info_aware_config, "cc_pass_group_fes", None)
    if cc_pass_group_fes is not None and int(cc_pass_group_fes) > 0:
        per_pass_fes = group_count * int(cc_pass_group_fes)
    else:
        per_pass_fes = remaining_fes
    if bool(has_overlap) and is_pass_level_validated_coordination_enabled(info_aware_config):
        per_pass_fes += 1
    estimated = int(math.ceil(remaining_fes / max(1, per_pass_fes)))
    cc_max_passes = getattr(info_aware_config, "cc_max_passes", None)
    if cc_max_passes is not None:
        estimated = min(int(cc_max_passes), estimated)
    return max(min_passes, estimated)


def should_recover_arac_relation(
    history_entry,
    phase="early",
    recovery_enabled=False,
    recovery_min_attempts=20,
    recovery_accept_rate_threshold=0.25,
    recovery_delta_threshold=0.0,
    recovery_positive_delta_rate_threshold=0.25,
    recovery_recent_window=20,
    recovery_min_phase="middle",
    probe_recovery_enabled=False,
    probe_recovery_min_attempts=5,
    probe_recovery_accept_rate_threshold=0.3,
    probe_recovery_delta_threshold=0.0,
    probe_recovery_recent_window=20,
):
    phase = str(phase or "early").strip().lower()
    recovery_min_phase = str(recovery_min_phase or "middle").strip().lower()
    phase_rank = {"early": 0, "middle": 1, "late": 2}
    if not bool(recovery_enabled):
        return {
            "should_recover": False,
            "recovery_reason": "recovery_disabled",
            "recovery_attempt_count": 0,
            "recovery_accept_rate": 0.0,
            "recovery_delta_mean": 0.0,
            "recovery_positive_delta_rate": 0.0,
        }
    if phase_rank.get(phase, 0) < phase_rank.get(recovery_min_phase, 1):
        return {
            "should_recover": False,
            "recovery_reason": "phase_not_recoverable",
            "recovery_attempt_count": 0,
            "recovery_accept_rate": 0.0,
            "recovery_delta_mean": 0.0,
            "recovery_positive_delta_rate": 0.0,
        }

    if bool(probe_recovery_enabled):
        probe_recent_attempts = _arac_recent_probe_attempts(history_entry, probe_recovery_recent_window)
        if probe_recent_attempts:
            probe_attempt_count = int(len(probe_recent_attempts))
            probe_accept_count = int(sum(bool(item["accepted"]) for item in probe_recent_attempts))
            probe_deltas = np.asarray([float(item["delta"]) for item in probe_recent_attempts], dtype=float)
            probe_delta_mean = float(np.mean(probe_deltas)) if probe_deltas.size else 0.0
        else:
            summary = summarize_arac_relation_history_entry(history_entry)
            probe_attempt_count = int(summary.get("probe_attempt_count", 0))
            probe_accept_count = int(summary.get("probe_accept_count", 0))
            probe_delta_mean = float(summary.get("probe_mean_validation_delta", 0.0))
        probe_accept_rate = float(probe_accept_count / probe_attempt_count) if probe_attempt_count else 0.0
        if probe_attempt_count >= int(probe_recovery_min_attempts):
            if probe_accept_rate < float(probe_recovery_accept_rate_threshold):
                return {
                    "should_recover": False,
                    "recovery_reason": "probe_accept_rate_below_threshold",
                    "recovery_attempt_count": int(probe_attempt_count),
                    "recovery_accept_rate": float(probe_accept_rate),
                    "recovery_delta_mean": float(probe_delta_mean),
                    "recovery_positive_delta_rate": 0.0,
                }
            if probe_delta_mean <= float(probe_recovery_delta_threshold):
                return {
                    "should_recover": False,
                    "recovery_reason": "probe_delta_nonpositive",
                    "recovery_attempt_count": int(probe_attempt_count),
                    "recovery_accept_rate": float(probe_accept_rate),
                    "recovery_delta_mean": float(probe_delta_mean),
                    "recovery_positive_delta_rate": 0.0,
                }
            return {
                "should_recover": True,
                "recovery_reason": "probe_recovery_signal_passed",
                "recovery_attempt_count": int(probe_attempt_count),
                "recovery_accept_rate": float(probe_accept_rate),
                "recovery_delta_mean": float(probe_delta_mean),
                "recovery_positive_delta_rate": 0.0,
            }

    recent_attempts = _arac_recent_attempts(history_entry, recovery_recent_window)
    if recent_attempts:
        attempt_count = int(len(recent_attempts))
        accept_count = int(sum(bool(item["accepted"]) for item in recent_attempts))
        deltas = np.asarray([float(item["delta"]) for item in recent_attempts], dtype=float)
        delta_mean = float(np.mean(deltas)) if deltas.size else 0.0
        positive_delta_rate = float(np.mean(deltas > 0.0)) if deltas.size else 0.0
    else:
        summary = summarize_arac_relation_history_entry(history_entry)
        attempt_count = int(summary["relation_attempt_count"])
        accept_count = int(summary["relation_accept_count"])
        delta_mean = float(summary["relation_mean_validation_delta"])
        positive_delta_rate = float(summary.get("relation_positive_delta_rate", 0.0))
    accept_rate = float(accept_count / attempt_count) if attempt_count else 0.0

    if attempt_count < int(recovery_min_attempts):
        reason = "rolling_attempts_below_min"
        should_recover = False
    elif accept_rate < float(recovery_accept_rate_threshold):
        reason = "rolling_accept_rate_below_threshold"
        should_recover = False
    elif delta_mean <= float(recovery_delta_threshold):
        reason = "rolling_delta_nonpositive"
        should_recover = False
    elif positive_delta_rate < float(recovery_positive_delta_rate_threshold):
        reason = "rolling_positive_delta_rate_below_threshold"
        should_recover = False
    else:
        reason = "recovery_signal_passed"
        should_recover = True
    return {
        "should_recover": bool(should_recover),
        "recovery_reason": str(reason),
        "recovery_attempt_count": int(attempt_count),
        "recovery_accept_rate": float(accept_rate),
        "recovery_delta_mean": float(delta_mean),
        "recovery_positive_delta_rate": float(positive_delta_rate),
    }


def update_arac_relation_history(history, fusion_rows):
    if history is None:
        return {}
    for row in list(fusion_rows or []):
        if str(row.get("action_candidate", "")) != "Fusion":
            continue
        if not bool(row.get("was_updated")):
            continue
        if not bool(row.get("validation_attempted")):
            continue
        try:
            var_id = int(row.get("var_id"))
        except (TypeError, ValueError):
            continue
        entry = dict(history.get(var_id, {}))
        attempt_count = int(entry.get("attempt_count", 0) or 0) + 1
        accept_count = int(entry.get("accept_count", 0) or 0)
        accepted = bool(row.get("validation_accepted"))
        if accepted:
            accept_count += 1
        is_probe_candidate = bool(row.get("arac_probe_candidate"))
        row_delta = _to_float(row.get("fitness_delta"))
        if not np.isfinite(row_delta):
            row_delta = 0.0
        delta_sum = float(entry.get("delta_sum", 0.0) or 0.0) + float(row_delta)
        positive_delta_count = int(entry.get("positive_delta_count", 0) or 0) + int(row_delta > 0.0)
        reject_streak = 0 if accepted else int(entry.get("reject_streak", 0) or 0) + 1
        recent_attempts = _arac_recent_attempts(entry, 99)
        recent_attempts.append({"accepted": bool(accepted), "delta": float(row_delta)})
        probe_attempt_count = int(entry.get("probe_attempt_count", 0) or 0)
        probe_accept_count = int(entry.get("probe_accept_count", 0) or 0)
        probe_delta_sum = float(entry.get("probe_delta_sum", 0.0) or 0.0)
        probe_positive_delta_count = int(entry.get("probe_positive_delta_count", 0) or 0)
        probe_recent_attempts = _arac_recent_probe_attempts(entry, 99)
        if is_probe_candidate:
            probe_attempt_count += 1
            probe_accept_count += int(accepted)
            probe_delta_sum += float(row_delta)
            probe_positive_delta_count += int(row_delta > 0.0)
            probe_recent_attempts.append({"accepted": bool(accepted), "delta": float(row_delta)})
        history[var_id] = {
            "attempt_count": int(attempt_count),
            "accept_count": int(accept_count),
            "delta_sum": float(delta_sum),
            "positive_delta_count": int(positive_delta_count),
            "reject_streak": int(reject_streak),
            "recent_attempts": recent_attempts[-100:],
            "probe_attempt_count": int(probe_attempt_count),
            "probe_accept_count": int(probe_accept_count),
            "probe_delta_sum": float(probe_delta_sum),
            "probe_positive_delta_count": int(probe_positive_delta_count),
            "probe_recent_attempts": probe_recent_attempts[-100:],
        }
    return history


def apply_hypergraph_pass_end_coordination(
    best_individual,
    proposal_rows,
    overlap_hypergraph,
    conflict_prior=None,
    eps_delta=1e-12,
    use_conflict_damping=False,
    conflict_gamma=0.5,
    min_damping=0.3,
    coordination_mode="hypergraph_pass_end",
    variable_ranges=None,
    selective_min_positive_proposals=2,
    selective_max_proposal_std_ratio=0.00125,
    selective_max_update_ratio=0.0025,
    selective_large_update_damping=0.25,
    selective_owner_soft_eta=0.2,
    selective_owner_min_delta_ratio=0.001,
    arac_relation_history=None,
    arac_lite_history_min_attempts=2,
    arac_lite_disable_accept_rate_threshold=0.0,
    arac_lite_disable_mean_delta_threshold=0.0,
    arac_lite_disable_reject_streak=2,
    arac_lite_recovery_enabled=False,
    arac_lite_recovery_phase="early",
    arac_lite_recovery_min_attempts=20,
    arac_lite_recovery_accept_rate_threshold=0.25,
    arac_lite_recovery_delta_threshold=0.0,
    arac_lite_recovery_positive_delta_rate_threshold=0.25,
    arac_lite_recovery_recent_window=20,
    arac_lite_recovery_min_phase="middle",
    arac_lite_probe_enabled=False,
    arac_lite_probe_phase="early",
    arac_lite_probe_pass_id=0,
    arac_lite_probe_every_n_pass=2,
    arac_lite_probe_max_per_pass=5,
    arac_lite_probe_min_phase="middle",
    arac_lite_probe_recovery_min_attempts=5,
    arac_lite_probe_recovery_accept_rate_threshold=0.3,
    arac_lite_probe_recovery_delta_threshold=0.0,
    arac_lite_probe_recovery_recent_window=20,
    arac_lite_targeted_probe_enabled=False,
    arac_lite_targeted_probe_phase="middle",
    arac_lite_targeted_probe_min_support=2,
    arac_lite_targeted_probe_min_relation_attempts=1,
    arac_lite_targeted_probe_min_accept_rate=0.3,
    arac_lite_targeted_probe_min_positive_delta_rate=0.3,
    arac_lite_targeted_probe_min_relation_delta=0.0,
    arac_lite_targeted_probe_min_proposal_std_ratio=0.00125,
    arac_lite_targeted_probe_max_proposal_std_ratio=0.0045,
    arac_lite_random_probe_same_budget_enabled=False,
    arac_lite_random_probe_budget=0,
    arac_lite_random_probe_phase="middle",
    arac_lite_random_probe_seed=0,
    arac_lite_random_probe_state=None,
):
    proposal_rows = list(proposal_rows or [])
    overlap_hypergraph = overlap_hypergraph or {"overlap_vars": [], "var_to_groups": {}}
    coordination_mode = str(coordination_mode or "hypergraph_pass_end").strip().lower()
    selective_gate_enabled = is_selective_hypergraph_mode(coordination_mode)
    arac_lite_rule_enabled = is_arac_lite_mode(coordination_mode)
    variable_ranges = np.asarray(variable_ranges if variable_ranges is not None else np.ones(len(best_individual)), dtype=float).reshape(-1)
    if variable_ranges.size < len(best_individual):
        variable_ranges = np.resize(variable_ranges, len(best_individual)).astype(float)
    variable_ranges = np.where(variable_ranges <= 1e-12, 1.0, np.abs(variable_ranges))
    proposals_by_var = {}
    for row in proposal_rows:
        proposals_by_var.setdefault(int(row["var_id"]), []).append(dict(row))

    fusion_rows = []
    targeted_probe_summaries = build_arac_targeted_probe_summaries(
        best_individual,
        proposals_by_var,
        overlap_hypergraph.get("overlap_vars", []),
        variable_ranges,
        arac_relation_history=arac_relation_history,
        phase=arac_lite_probe_phase,
        pass_id=arac_lite_probe_pass_id,
        probe_enabled=arac_lite_probe_enabled,
        targeted_probe_enabled=arac_lite_targeted_probe_enabled,
        probe_every_n_pass=arac_lite_probe_every_n_pass,
        probe_max_per_pass=arac_lite_probe_max_per_pass,
        probe_min_phase=arac_lite_probe_min_phase,
        targeted_probe_phase=arac_lite_targeted_probe_phase,
        targeted_probe_min_support=arac_lite_targeted_probe_min_support,
        targeted_probe_min_relation_attempts=arac_lite_targeted_probe_min_relation_attempts,
        targeted_probe_min_accept_rate=arac_lite_targeted_probe_min_accept_rate,
        targeted_probe_min_positive_delta_rate=arac_lite_targeted_probe_min_positive_delta_rate,
        targeted_probe_min_relation_delta=arac_lite_targeted_probe_min_relation_delta,
        targeted_probe_min_proposal_std_ratio=arac_lite_targeted_probe_min_proposal_std_ratio,
        targeted_probe_max_proposal_std_ratio=arac_lite_targeted_probe_max_proposal_std_ratio,
        selective_min_positive_proposals=selective_min_positive_proposals,
        selective_max_proposal_std_ratio=selective_max_proposal_std_ratio,
        history_min_attempts=arac_lite_history_min_attempts,
        disable_accept_rate_threshold=arac_lite_disable_accept_rate_threshold,
        disable_mean_delta_threshold=arac_lite_disable_mean_delta_threshold,
        disable_reject_streak=arac_lite_disable_reject_streak,
    )
    probe_budget_remaining = max(0, int(arac_lite_probe_max_per_pass))
    random_probe_selected_vars = set()
    random_probe_enabled = bool(arac_lite_random_probe_same_budget_enabled) and bool(arac_lite_probe_enabled)
    random_probe_phase = str(arac_lite_random_probe_phase or "middle").strip().lower()
    random_probe_remaining_total = random_same_budget_remaining_probe_budget(
        arac_lite_random_probe_state,
        random_probe_phase,
        arac_lite_random_probe_budget,
    )
    if random_probe_enabled and str(arac_lite_probe_phase or "").strip().lower() == random_probe_phase:
        if _phase_rank(arac_lite_probe_phase) >= _phase_rank(arac_lite_probe_min_phase):
            if int(arac_lite_probe_pass_id) % max(1, int(arac_lite_probe_every_n_pass)) == 0:
                candidates = []
                for candidate_var_id in [int(value) for value in overlap_hypergraph.get("overlap_vars", [])]:
                    variable_rows = list(proposals_by_var.get(int(candidate_var_id), []) or [])
                    finite_rows = [
                        row
                        for row in variable_rows
                        if np.isfinite(float(row.get("proposal_value", float("nan"))))
                    ]
                    positive_rows = [
                        row
                        for row in finite_rows
                        if np.isfinite(float(row.get("delta", float("nan"))))
                        and float(row.get("delta", 0.0)) > float(eps_delta)
                    ]
                    if len(positive_rows) < int(selective_min_positive_proposals):
                        continue
                    relation_entry = dict((arac_relation_history or {}).get(int(candidate_var_id), {}))
                    history_blocks, _ = should_disable_arac_relation(
                        relation_entry,
                        history_min_attempts=arac_lite_history_min_attempts,
                        disable_accept_rate_threshold=arac_lite_disable_accept_rate_threshold,
                        disable_mean_delta_threshold=arac_lite_disable_mean_delta_threshold,
                        disable_reject_streak=arac_lite_disable_reject_streak,
                    )
                    values = np.asarray([float(row["proposal_value"]) for row in positive_rows], dtype=float)
                    proposal_std = float(np.std(values)) if values.size else 0.0
                    var_range = (
                        float(variable_ranges[int(candidate_var_id)])
                        if 0 <= int(candidate_var_id) < variable_ranges.size
                        else 1.0
                    )
                    if not np.isfinite(var_range) or var_range <= 0.0:
                        var_range = 1.0
                    proposal_std_ratio = float(proposal_std / var_range)
                    proposal_conflict = bool(proposal_std_ratio > float(selective_max_proposal_std_ratio))
                    if not history_blocks and not proposal_conflict:
                        continue
                    score = stable_random_probe_score(
                        candidate_var_id,
                        pass_id=arac_lite_probe_pass_id,
                        seed=arac_lite_random_probe_seed,
                        phase=arac_lite_probe_phase,
                    )
                    candidates.append((score, int(candidate_var_id)))
                limit = min(
                    max(0, int(arac_lite_probe_max_per_pass)),
                    max(0, int(random_probe_remaining_total)),
                )
                random_probe_selected_vars = {
                    var_id for _, var_id in sorted(candidates, key=lambda item: (item[0], item[1]))[:limit]
                }
    for var_id in [int(value) for value in overlap_hypergraph.get("overlap_vars", [])]:
        variable_rows = proposals_by_var.get(int(var_id), [])
        old_value = float(best_individual[int(var_id)])
        variable_range = float(variable_ranges[int(var_id)]) if 0 <= int(var_id) < variable_ranges.size else 1.0
        finite_rows = [
            row
            for row in variable_rows
            if np.isfinite(float(row.get("proposal_value", float("nan"))))
        ]
        positive_rows = [
            row
            for row in finite_rows
            if np.isfinite(float(row.get("delta", float("nan"))))
            and float(row.get("delta", 0.0)) > float(eps_delta)
        ]
        negative_rows = [
            row
            for row in finite_rows
            if np.isfinite(float(row.get("delta", float("nan"))))
            and float(row.get("delta", 0.0)) <= float(eps_delta)
        ]
        proposal_values = np.asarray([float(row["proposal_value"]) for row in finite_rows], dtype=float) if finite_rows else np.empty((0,), dtype=float)
        proposal_value_min = float(np.min(proposal_values)) if proposal_values.size else float("nan")
        proposal_value_max = float(np.max(proposal_values)) if proposal_values.size else float("nan")
        proposal_value_std = float(np.std(proposal_values)) if proposal_values.size else float("nan")
        proposal_value_std_ratio = float(proposal_value_std / variable_range) if np.isfinite(proposal_value_std) else float("nan")
        delta_rows = [
            row
            for row in finite_rows
            if np.isfinite(float(row.get("delta", float("nan"))))
        ]
        max_delta = float(max((float(row["delta"]) for row in delta_rows), default=float("nan")))
        top_group_id = ""
        if delta_rows:
            top_row = max(delta_rows, key=lambda row: float(row["delta"]))
            top_group_id = int(top_row.get("group_id", -1))
        conflict_value = 0.0
        if conflict_prior is not None and 0 <= int(var_id) < len(conflict_prior):
            conflict_value = float(conflict_prior[int(var_id)])
            if not np.isfinite(conflict_value):
                conflict_value = 0.0
        relation_history_entry = dict((arac_relation_history or {}).get(int(var_id), {}))
        relation_summary = summarize_arac_relation_history_entry(relation_history_entry)
        damping = 1.0
        fused_value = float(old_value)
        applied_value = float(old_value)
        raw_update_magnitude = 0.0
        raw_update_ratio = 0.0
        update_magnitude = 0.0
        update_magnitude_ratio = 0.0
        ownership_mode = "freeze"
        owner_group_id = ""
        owner_value = float(old_value)
        owner_delta = 0.0
        owner_delta_ratio = float("nan")
        owner_delta_share = float("nan")
        owner_step_weight = 0.0
        weighted_delta_sum = 0.0
        sum_positive_delta = 0.0
        skip_reason = "no_positive_proposal"
        action_candidate = "Disable"
        action_reason = "no_positive_proposal"
        gate_passed = False
        risky_candidate = False
        applied_update = False
        recovery_candidate = False
        probe_candidate = False
        probe_reason = "probe_disabled" if not bool(arac_lite_probe_enabled) else "not_evaluated"
        targeted_probe_summary = dict(
            targeted_probe_summaries.get(int(var_id), default_targeted_probe_summary())
        )
        targeted_probe_selected = bool(targeted_probe_summary.get("should_probe"))
        random_probe_selected = bool(int(var_id) in random_probe_selected_vars)
        recovery_summary = should_recover_arac_relation(
            relation_history_entry,
            phase=arac_lite_recovery_phase,
            recovery_enabled=arac_lite_recovery_enabled,
            recovery_min_attempts=arac_lite_recovery_min_attempts,
            recovery_accept_rate_threshold=arac_lite_recovery_accept_rate_threshold,
            recovery_delta_threshold=arac_lite_recovery_delta_threshold,
            recovery_positive_delta_rate_threshold=arac_lite_recovery_positive_delta_rate_threshold,
            recovery_recent_window=arac_lite_recovery_recent_window,
            recovery_min_phase=arac_lite_recovery_min_phase,
            probe_recovery_enabled=arac_lite_probe_enabled,
            probe_recovery_min_attempts=arac_lite_probe_recovery_min_attempts,
            probe_recovery_accept_rate_threshold=arac_lite_probe_recovery_accept_rate_threshold,
            probe_recovery_delta_threshold=arac_lite_probe_recovery_delta_threshold,
            probe_recovery_recent_window=arac_lite_probe_recovery_recent_window,
        )
        if positive_rows:
            deltas = np.asarray([float(row["delta"]) for row in positive_rows], dtype=float)
            values = np.asarray([float(row["proposal_value"]) for row in positive_rows], dtype=float)
            weight_sum = float(np.sum(deltas))
            weighted_delta_sum = float(weight_sum)
            sum_positive_delta = float(weight_sum)
            owner_row = max(positive_rows, key=lambda row: float(row.get("delta", float("-inf"))))
            owner_group_id = int(owner_row.get("group_id", -1))
            owner_value = float(owner_row.get("proposal_value", old_value))
            owner_delta = float(owner_row.get("delta", 0.0))
            owner_fitness_scale_candidates = [
                abs(float(owner_row.get("fitness_before", float("nan")))),
                abs(float(owner_row.get("fitness_after", float("nan")))),
                1.0,
            ]
            finite_owner_scales = [value for value in owner_fitness_scale_candidates if np.isfinite(value)]
            owner_fitness_scale = max(finite_owner_scales) if finite_owner_scales else 1.0
            owner_delta_ratio = float(owner_delta / owner_fitness_scale) if owner_fitness_scale > float(eps_delta) else float("nan")
            owner_delta_share = float(owner_delta / weight_sum) if weight_sum > float(eps_delta) else float("nan")
            if np.isfinite(weight_sum) and weight_sum > float(eps_delta):
                weights = deltas / weight_sum
                fused_value = float(np.sum(weights * values))
                raw_update_magnitude = float(abs(fused_value - old_value))
                raw_update_ratio = float(raw_update_magnitude / variable_range)
                risky_candidate = bool(
                    len(positive_rows) < int(selective_min_positive_proposals)
                    or (
                        np.isfinite(proposal_value_std_ratio)
                        and proposal_value_std_ratio > float(selective_max_proposal_std_ratio)
                    )
                    or raw_update_ratio > float(selective_max_update_ratio)
                )
                if arac_lite_rule_enabled:
                    history_blocks, history_reason = should_disable_arac_relation(
                        relation_history_entry,
                        history_min_attempts=arac_lite_history_min_attempts,
                        disable_accept_rate_threshold=arac_lite_disable_accept_rate_threshold,
                        disable_mean_delta_threshold=arac_lite_disable_mean_delta_threshold,
                        disable_reject_streak=arac_lite_disable_reject_streak,
                    )
                    if history_blocks:
                        recovery_candidate = bool(recovery_summary["should_recover"])
                        if bool(arac_lite_targeted_probe_enabled):
                            probe_selected = bool(targeted_probe_selected)
                            probe_reason = str(targeted_probe_summary["targeted_probe_reason"])
                        elif random_probe_enabled:
                            probe_selected = bool(random_probe_selected)
                            probe_reason = (
                                "random_same_budget_probe_selected"
                                if probe_selected
                                else "random_same_budget_probe_not_selected"
                            )
                        else:
                            probe_summary = should_probe_arac_relation(
                                history_blocks,
                                phase=arac_lite_probe_phase,
                                pass_id=arac_lite_probe_pass_id,
                                probe_enabled=arac_lite_probe_enabled,
                                probe_every_n_pass=arac_lite_probe_every_n_pass,
                                probe_min_phase=arac_lite_probe_min_phase,
                                remaining_probe_budget=probe_budget_remaining,
                            )
                            probe_selected = bool(probe_summary["should_probe"])
                            probe_reason = str(probe_summary["probe_reason"])
                        if recovery_candidate and len(positive_rows) >= int(selective_min_positive_proposals):
                            if np.isfinite(proposal_value_std_ratio) and proposal_value_std_ratio > float(selective_max_proposal_std_ratio):
                                ownership_mode = "freeze"
                                action_candidate = "Freeze"
                                action_reason = "proposal_conflict"
                                skip_reason = "proposal_conflict"
                            else:
                                ownership_mode = "multi_support_fusion"
                                action_candidate = "Fusion"
                                action_reason = "recovery_fusion_supported"
                                gate_passed = True
                                skip_reason = "recovery_fusion_supported"
                                if raw_update_ratio > float(selective_max_update_ratio):
                                    damping = min(float(damping), float(selective_large_update_damping))
                                    skip_reason = "recovery_fusion_large_update_damped"
                                    action_reason = "recovery_fusion_large_update_damped"
                        elif probe_selected and len(positive_rows) >= int(selective_min_positive_proposals):
                            if np.isfinite(proposal_value_std_ratio) and proposal_value_std_ratio > float(selective_max_proposal_std_ratio):
                                if bool(arac_lite_targeted_probe_enabled) or random_probe_selected:
                                    probe_budget_remaining -= 1
                                    if random_probe_selected:
                                        mark_random_same_budget_probe_used(
                                            arac_lite_random_probe_state,
                                            random_probe_phase,
                                        )
                                    probe_candidate = True
                                    ownership_mode = "multi_support_fusion"
                                    action_candidate = "Fusion"
                                    action_reason = (
                                        "random_same_budget_probe_fusion_candidate"
                                        if random_probe_selected
                                        else "targeted_probe_fusion_candidate"
                                    )
                                    probe_reason = (
                                        "random_same_budget_probe_selected"
                                        if random_probe_selected
                                        else str(targeted_probe_summary["targeted_probe_reason"])
                                    )
                                    gate_passed = True
                                    skip_reason = str(action_reason)
                                else:
                                    probe_candidate = False
                                    ownership_mode = "freeze"
                                    action_candidate = "Freeze"
                                    action_reason = "probe_proposal_conflict"
                                    probe_reason = "probe_proposal_conflict"
                                    skip_reason = "probe_proposal_conflict"
                            else:
                                probe_budget_remaining -= 1
                                if random_probe_selected:
                                    mark_random_same_budget_probe_used(
                                        arac_lite_random_probe_state,
                                        random_probe_phase,
                                    )
                                probe_candidate = True
                                ownership_mode = "multi_support_fusion"
                                action_candidate = "Fusion"
                                action_reason = (
                                    "targeted_probe_fusion_candidate"
                                    if bool(arac_lite_targeted_probe_enabled)
                                    else "random_same_budget_probe_fusion_candidate"
                                    if random_probe_selected
                                    else "probe_fusion_candidate"
                                )
                                gate_passed = True
                                skip_reason = str(action_reason)
                                if raw_update_ratio > float(selective_max_update_ratio):
                                    damping = min(float(damping), float(selective_large_update_damping))
                                    skip_reason = (
                                        "targeted_probe_fusion_large_update_damped"
                                        if bool(arac_lite_targeted_probe_enabled)
                                        else "random_same_budget_probe_fusion_large_update_damped"
                                        if random_probe_selected
                                        else "probe_fusion_large_update_damped"
                                    )
                                    action_reason = str(skip_reason)
                        else:
                            ownership_mode = "disable"
                            action_candidate = "Disable"
                            action_reason = history_reason
                            skip_reason = history_reason
                    elif len(positive_rows) >= int(selective_min_positive_proposals):
                        if targeted_probe_selected or random_probe_selected:
                            probe_budget_remaining -= 1
                            if random_probe_selected:
                                mark_random_same_budget_probe_used(
                                    arac_lite_random_probe_state,
                                    random_probe_phase,
                                )
                            probe_candidate = True
                            ownership_mode = "multi_support_fusion"
                            action_candidate = "Fusion"
                            action_reason = (
                                "random_same_budget_probe_fusion_candidate"
                                if random_probe_selected
                                else "targeted_probe_fusion_candidate"
                            )
                            probe_reason = (
                                "random_same_budget_probe_selected"
                                if random_probe_selected
                                else str(targeted_probe_summary["targeted_probe_reason"])
                            )
                            gate_passed = True
                            skip_reason = str(action_reason)
                            if raw_update_ratio > float(selective_max_update_ratio):
                                damping = min(float(damping), float(selective_large_update_damping))
                                skip_reason = (
                                    "random_same_budget_probe_fusion_large_update_damped"
                                    if random_probe_selected
                                    else "targeted_probe_fusion_large_update_damped"
                                )
                                action_reason = "targeted_probe_fusion_large_update_damped"
                                if random_probe_selected:
                                    action_reason = "random_same_budget_probe_fusion_large_update_damped"
                        elif np.isfinite(proposal_value_std_ratio) and proposal_value_std_ratio > float(selective_max_proposal_std_ratio):
                            ownership_mode = "freeze"
                            action_candidate = "Freeze"
                            action_reason = "proposal_conflict"
                            skip_reason = "proposal_conflict"
                        else:
                            ownership_mode = "multi_support_fusion"
                            action_candidate = "Fusion"
                            action_reason = "fusion_supported"
                            gate_passed = True
                            skip_reason = "fusion_supported"
                            if raw_update_ratio > float(selective_max_update_ratio):
                                damping = min(float(damping), float(selective_large_update_damping))
                                skip_reason = "fusion_large_update_damped"
                                action_reason = "fusion_large_update_damped"
                    else:
                        ownership_mode = "freeze"
                        action_candidate = "Freeze"
                        action_reason = "insufficient_positive_support"
                        skip_reason = "insufficient_positive_support"
                elif selective_gate_enabled:
                    if len(positive_rows) >= int(selective_min_positive_proposals):
                        if np.isfinite(proposal_value_std_ratio) and proposal_value_std_ratio > float(selective_max_proposal_std_ratio):
                            skip_reason = "proposal_std_too_large"
                        else:
                            ownership_mode = "multi_support_fusion"
                            gate_passed = True
                            skip_reason = "multi_support_applied"
                            if raw_update_ratio > float(selective_max_update_ratio):
                                damping = min(float(damping), float(selective_large_update_damping))
                                skip_reason = "multi_support_large_update_damped"
                    elif len(positive_rows) == 1:
                        fused_value = float(owner_value)
                        raw_update_magnitude = float(abs(fused_value - old_value))
                        raw_update_ratio = float(raw_update_magnitude / variable_range)
                        if np.isfinite(proposal_value_std_ratio) and proposal_value_std_ratio > float(selective_max_proposal_std_ratio):
                            skip_reason = "owner_proposal_std_too_large"
                        elif not np.isfinite(owner_delta_ratio) or owner_delta_ratio < float(selective_owner_min_delta_ratio):
                            skip_reason = "owner_delta_too_small"
                        else:
                            ownership_mode = "owner_soft"
                            gate_passed = True
                            damping = float(selective_owner_soft_eta)
                            owner_step_weight = float(damping)
                            skip_reason = "owner_soft_applied"
                            if raw_update_ratio > float(eps_delta) and damping * raw_update_ratio > float(selective_max_update_ratio):
                                damping = min(float(damping), float(selective_max_update_ratio / raw_update_ratio))
                                owner_step_weight = float(damping)
                                skip_reason = "owner_soft_large_update_damped"
                    else:
                        skip_reason = "insufficient_positive_support"
                else:
                    ownership_mode = "multi_support_fusion"
                    gate_passed = True
                    skip_reason = "multi_support_applied"
                if gate_passed and use_conflict_damping:
                    conflict_damping = float(np.clip(1.0 - float(conflict_gamma) * conflict_value, min_damping, 1.0))
                    if ownership_mode == "owner_soft":
                        damping = float(damping * conflict_damping)
                        owner_step_weight = float(damping)
                    else:
                        damping = min(float(damping), conflict_damping)
                if gate_passed:
                    applied_value = float(old_value + damping * (fused_value - old_value))
                    update_magnitude = float(abs(applied_value - old_value))
                    update_magnitude_ratio = float(update_magnitude / variable_range)
                    best_individual[int(var_id)] = applied_value
                    applied_update = not np.isclose(applied_value, old_value, rtol=1e-12, atol=1e-12)
            else:
                skip_reason = "weak_positive_delta"
                action_candidate = "Disable"
                action_reason = "weak_positive_delta"
        if not arac_lite_rule_enabled:
            if ownership_mode in {"multi_support_fusion", "owner_soft"} and gate_passed:
                action_candidate = "Fusion"
            elif ownership_mode == "freeze" and positive_rows:
                action_candidate = "Freeze"
            else:
                action_candidate = "Disable"
            action_reason = str(skip_reason)
        elif not positive_rows:
            ownership_mode = "disable"
            action_candidate = "Disable"
            action_reason = "no_positive_proposal"
        harmful_update_proxy_flag = bool(
            applied_update and risky_candidate and ownership_mode == "multi_support_fusion"
        )
        fusion_rows.append(
            {
                "var_id": int(var_id),
                "membership_count": int(len(overlap_hypergraph.get("var_to_groups", {}).get(int(var_id), []))),
                "group_count": int(len(overlap_hypergraph.get("var_to_groups", {}).get(int(var_id), []))),
                "proposal_count": int(len(finite_rows)),
                "positive_proposal_count": int(len(positive_rows)),
                "negative_proposal_count": int(len(negative_rows)),
                "proposal_value_min": float(proposal_value_min),
                "proposal_value_max": float(proposal_value_max),
                "proposal_value_std": float(proposal_value_std),
                "proposal_value_std_ratio": float(proposal_value_std_ratio),
                "old_value": float(old_value),
                "fused_value": float(fused_value),
                "applied_value": float(applied_value),
                "raw_update_magnitude": float(raw_update_magnitude),
                "raw_update_ratio": float(raw_update_ratio),
                "update_magnitude": float(update_magnitude),
                "update_magnitude_ratio": float(update_magnitude_ratio),
                "ownership_mode": str(ownership_mode),
                "owner_group_id": owner_group_id,
                "owner_value": float(owner_value),
                "owner_delta": float(owner_delta),
                "owner_delta_ratio": float(owner_delta_ratio),
                "owner_delta_share": float(owner_delta_share),
                "owner_step_weight": float(owner_step_weight),
                "weighted_delta_sum": float(weighted_delta_sum),
                "sum_positive_delta": float(sum_positive_delta),
                "max_delta": float(max_delta),
                "top_group_id": top_group_id,
                "conflict_prior": float(conflict_value),
                "damping": float(damping),
                "skip_reason": str(skip_reason),
                "gate_passed": bool(gate_passed),
                "risky_candidate": bool(risky_candidate),
                "harmful_update_proxy_flag": bool(harmful_update_proxy_flag),
                "applied_update": bool(applied_update),
                "was_updated": bool(applied_update),
                "action_candidate": str(action_candidate),
                "action_reason": str(action_reason),
                "arac_recovery_candidate": bool(recovery_candidate),
                "arac_recovery_phase": str(arac_lite_recovery_phase),
                "arac_recovery_reason": str(recovery_summary["recovery_reason"]),
                "arac_recovery_attempt_count": int(recovery_summary["recovery_attempt_count"]),
                "arac_recovery_accept_rate": float(recovery_summary["recovery_accept_rate"]),
                "arac_recovery_delta_mean": float(recovery_summary["recovery_delta_mean"]),
                "arac_recovery_positive_delta_rate": float(recovery_summary["recovery_positive_delta_rate"]),
                "arac_probe_candidate": bool(probe_candidate),
                "arac_probe_reason": str(probe_reason),
                "arac_probe_phase": str(arac_lite_probe_phase),
                "arac_probe_attempt_count": int(relation_summary.get("probe_attempt_count", 0)),
                "arac_probe_accept_rate": float(relation_summary.get("probe_accept_rate", 0.0)),
                "arac_probe_delta_mean": float(relation_summary.get("probe_mean_validation_delta", 0.0)),
                "arac_targeted_probe_candidate": bool(targeted_probe_summary.get("targeted_probe_candidate", False)),
                "arac_targeted_probe_signature_matched": bool(
                    targeted_probe_summary.get("targeted_probe_signature_matched", False)
                ),
                "arac_targeted_probe_reason": str(
                    targeted_probe_summary.get("targeted_probe_reason", "targeted_probe_not_evaluated")
                ),
                "arac_targeted_probe_score": float(targeted_probe_summary.get("targeted_probe_score", 0.0)),
                **relation_summary,
            }
        )
    return fusion_rows


def build_owner_soft_effect_audit(proposal_rows, fusion_rows, top_k=5, atol=1e-12, rtol=1e-12):
    proposal_rows = [dict(row) for row in (proposal_rows or [])]
    annotated_rows = [dict(row) for row in (fusion_rows or [])]
    proposals_by_var = {}
    for row in proposal_rows:
        try:
            var_id = int(row.get("var_id", -1))
        except (TypeError, ValueError):
            continue
        proposals_by_var.setdefault(var_id, []).append(dict(row))

    owner_soft_rows = []
    per_var = {}
    for row in annotated_rows:
        is_owner_soft_update = str(row.get("ownership_mode", "")) == "owner_soft" and bool(row.get("was_updated"))
        row["owner_soft_followed_by_best_improvement"] = bool(
            is_owner_soft_update and bool(row.get("post_coordination_best_improved"))
        )
        row["owner_soft_overwritten"] = False
        if not is_owner_soft_update:
            continue

        var_id = int(row.get("var_id", -1))
        cycle_id = int(row.get("cycle_id", -1))
        applied_value = float(row.get("applied_value", row.get("fused_value", row.get("old_value", float("nan")))))
        later_proposals = [
            proposal
            for proposal in proposals_by_var.get(var_id, [])
            if int(proposal.get("cycle_id", -1)) > cycle_id
        ]
        overwritten = any(
            np.isfinite(float(proposal.get("proposal_value", float("nan"))))
            and not np.isclose(float(proposal.get("proposal_value")), applied_value, rtol=rtol, atol=atol)
            for proposal in later_proposals
        )
        row["owner_soft_overwritten"] = bool(overwritten)
        owner_soft_rows.append(row)

        aggregate = per_var.setdefault(
            var_id,
            {
                "var_id": int(var_id),
                "update_count": 0,
                "best_improvement_count": 0,
                "overwritten_count": 0,
                "total_update_magnitude": 0.0,
            },
        )
        aggregate["update_count"] += 1
        aggregate["best_improvement_count"] += int(bool(row["owner_soft_followed_by_best_improvement"]))
        aggregate["overwritten_count"] += int(bool(overwritten))
        aggregate["total_update_magnitude"] += float(row.get("update_magnitude", 0.0) or 0.0)

    top_var_rows = []
    for aggregate in per_var.values():
        update_count = int(aggregate["update_count"])
        total_update_magnitude = float(aggregate["total_update_magnitude"])
        top_var_rows.append(
            {
                "var_id": int(aggregate["var_id"]),
                "update_count": update_count,
                "best_improvement_count": int(aggregate["best_improvement_count"]),
                "overwritten_count": int(aggregate["overwritten_count"]),
                "mean_update_magnitude": float(total_update_magnitude / update_count) if update_count else 0.0,
                "total_update_magnitude": float(total_update_magnitude),
            }
        )
    top_var_rows.sort(
        key=lambda row: (
            -int(row["best_improvement_count"]),
            -int(row["update_count"]),
            -float(row["total_update_magnitude"]),
            int(row["overwritten_count"]),
            int(row["var_id"]),
        )
    )

    update_magnitudes = [
        float(row.get("update_magnitude", 0.0) or 0.0)
        for row in owner_soft_rows
        if np.isfinite(float(row.get("update_magnitude", float("nan"))))
    ]
    audit = {
        "owner_soft_update_count": int(len(owner_soft_rows)),
        "owner_soft_update_magnitude_mean": float(np.mean(update_magnitudes)) if update_magnitudes else 0.0,
        "owner_soft_unique_var_count": int(len(per_var)),
        "owner_soft_followed_by_best_improvement_count": int(
            sum(bool(row["owner_soft_followed_by_best_improvement"]) for row in owner_soft_rows)
        ),
        "owner_soft_overwritten_count": int(sum(bool(row["owner_soft_overwritten"]) for row in owner_soft_rows)),
        "owner_soft_top_vars": top_var_rows[: max(0, int(top_k))],
    }
    return annotated_rows, audit


def summarize_shared_variable_coordination_rows(rows, visibility_audit=None):
    rows = list(rows or [])
    visibility_audit = visibility_audit or {}
    true_overlap_count = int(visibility_audit.get("true_overlap_var_count", len(rows)))
    visible_overlap_count = int(visibility_audit.get("coordinated_overlap_var_count", len(rows)))
    proposal_covered_count = int(sum(1 for row in rows if int(row.get("proposal_count", 0) or 0) > 0))
    positive_count = int(sum(1 for row in rows if int(row.get("positive_proposal_count", 0) or 0) > 0))
    applied_count = int(sum(1 for row in rows if bool(row.get("applied_update"))))
    candidate_rows = [row for row in rows if int(row.get("positive_proposal_count", 0) or 0) > 0]
    updated_candidate_rows = [row for row in candidate_rows if bool(row.get("was_updated"))]
    skipped_candidate_rows = [row for row in candidate_rows if not bool(row.get("was_updated"))]
    owner_soft_rows = [row for row in candidate_rows if str(row.get("ownership_mode", "")) == "owner_soft"]
    multi_support_rows = [row for row in candidate_rows if str(row.get("ownership_mode", "")) == "multi_support_fusion"]
    freeze_rows = [row for row in candidate_rows if str(row.get("ownership_mode", "")) == "freeze"]
    fusion_action_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Fusion"]
    freeze_action_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Freeze"]
    disable_action_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Disable"]
    validated_fusion_action_rows = [row for row in fusion_action_rows if bool(row.get("validation_attempted"))]
    accepted_fusion_action_rows = [row for row in validated_fusion_action_rows if bool(row.get("validation_accepted"))]
    std_blocked_rows = [
        row
        for row in candidate_rows
        if str(row.get("skip_reason", "")) in {"proposal_std_too_large", "owner_proposal_std_too_large"}
    ]
    support_blocked_rows = [row for row in candidate_rows if str(row.get("skip_reason", "")) == "insufficient_positive_support"]
    owner_delta_blocked_rows = [row for row in candidate_rows if str(row.get("skip_reason", "")) == "owner_delta_too_small"]
    large_update_damped_rows = [
        row
        for row in candidate_rows
        if str(row.get("skip_reason", "")) in {"multi_support_large_update_damped", "owner_soft_large_update_damped"}
    ]
    owner_soft_damped_rows = [
        row for row in candidate_rows if str(row.get("skip_reason", "")) == "owner_soft_large_update_damped"
    ]
    multi_support_damped_rows = [
        row for row in candidate_rows if str(row.get("skip_reason", "")) == "multi_support_large_update_damped"
    ]
    owner_soft_updated_rows = [row for row in owner_soft_rows if bool(row.get("was_updated"))]
    return {
        "coordination_mode": str(visibility_audit.get("coordination_mode", "")),
        "true_overlap_var_count": true_overlap_count,
        "adjacent_visible_overlap_var_count": int(visibility_audit.get("adjacent_visible_overlap_var_count", 0)),
        "coordinated_overlap_var_count": visible_overlap_count,
        "coordinated_overlap_ratio": float(visible_overlap_count / true_overlap_count) if true_overlap_count else 0.0,
        "proposal_covered_overlap_var_count": proposal_covered_count,
        "proposal_coverage_ratio": float(proposal_covered_count / true_overlap_count) if true_overlap_count else 0.0,
        "positive_overlap_var_count": positive_count,
        "positive_overlap_ratio": float(positive_count / true_overlap_count) if true_overlap_count else 0.0,
        "applied_overlap_var_count": applied_count,
        "applied_overlap_ratio": float(applied_count / true_overlap_count) if true_overlap_count else 0.0,
        "candidate_overlap_var_count": int(len(candidate_rows)),
        "update_ratio": float(len(updated_candidate_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "skip_ratio": float(len(skipped_candidate_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "freeze_ratio": float(len(freeze_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "owner_soft_ratio": float(len(owner_soft_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "multi_support_ratio": float(len(multi_support_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "fusion_count": int(len(fusion_action_rows)),
        "freeze_count": int(len(freeze_action_rows)),
        "disable_count": int(len(disable_action_rows)),
        "fusion_validation_attempt_count": int(len(validated_fusion_action_rows)),
        "fusion_validation_accept_count": int(len(accepted_fusion_action_rows)),
        "fusion_validation_accept_rate": float(len(accepted_fusion_action_rows) / len(validated_fusion_action_rows))
        if validated_fusion_action_rows
        else 0.0,
        "mean_update_magnitude": _mean_or_nan([row.get("update_magnitude") for row in updated_candidate_rows]),
        "mean_update_magnitude_ratio": _mean_or_nan([row.get("update_magnitude_ratio") for row in updated_candidate_rows]),
        "proposal_std_mean": _mean_or_nan([row.get("proposal_value_std") for row in candidate_rows]),
        "proposal_std_ratio_mean": _mean_or_nan([row.get("proposal_value_std_ratio") for row in candidate_rows]),
        "positive_proposal_count_mean": float(
            np.mean([int(row.get("positive_proposal_count", 0) or 0) for row in candidate_rows])
        ) if candidate_rows else 0.0,
        "mean_positive_proposal_count": float(
            np.mean([int(row.get("positive_proposal_count", 0) or 0) for row in candidate_rows])
        ) if candidate_rows else 0.0,
        "harmful_update_proxy": float(
            np.mean([bool(row.get("harmful_update_proxy_flag")) for row in candidate_rows])
        ) if candidate_rows else 0.0,
        "support_blocked_ratio": float(len(support_blocked_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "std_blocked_ratio": float(len(std_blocked_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "owner_delta_blocked_ratio": float(len(owner_delta_blocked_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "large_update_damped_ratio": float(len(large_update_damped_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "owner_soft_damped_ratio": float(len(owner_soft_damped_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "multi_support_damped_ratio": float(len(multi_support_damped_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "mean_damping": _mean_or_nan([row.get("damping") for row in candidate_rows if bool(row.get("gate_passed"))]),
        "owner_soft_update_count": int(len(owner_soft_updated_rows)),
        "owner_soft_update_magnitude_mean": _mean_or_nan([row.get("update_magnitude") for row in owner_soft_updated_rows]),
        "owner_soft_unique_var_count": int(len({int(row.get("var_id", -1)) for row in owner_soft_updated_rows})),
        "owner_soft_followed_by_best_improvement_count": int(
            sum(bool(row.get("owner_soft_followed_by_best_improvement")) for row in owner_soft_updated_rows)
        ),
        "owner_soft_overwritten_count": int(
            sum(bool(row.get("owner_soft_overwritten")) for row in owner_soft_updated_rows)
        ),
        "owner_soft_top_vars": [],
    }


def build_validated_coordination_trace_row(
    cycle_id,
    cycle_fusion_rows,
    fitness_before_validation,
    candidate_fitness,
    validation_attempted,
    validation_accepted,
    validation_fe_used,
    validation_fes_used_total,
    total_max_fes,
    reject_reason="",
):
    candidate_rows = [row for row in (cycle_fusion_rows or []) if int(row.get("positive_proposal_count", 0) or 0) > 0]
    updated_rows = [row for row in candidate_rows if bool(row.get("was_updated"))]
    owner_soft_rows = [row for row in updated_rows if str(row.get("ownership_mode", "")) == "owner_soft"]
    multi_support_rows = [row for row in updated_rows if str(row.get("ownership_mode", "")) == "multi_support_fusion"]
    freeze_rows = [row for row in candidate_rows if str(row.get("ownership_mode", "")) == "freeze"]
    candidate_fitness_value = float(candidate_fitness) if np.isfinite(_to_float(candidate_fitness)) else float(fitness_before_validation)
    fitness_before_value = float(fitness_before_validation)
    return {
        "cycle_id": int(cycle_id),
        "validation_attempted": bool(validation_attempted),
        "validation_accepted": bool(validation_accepted),
        "fitness_before_validation": float(fitness_before_value),
        "candidate_fitness": float(candidate_fitness_value),
        "fitness_delta": float(fitness_before_value - candidate_fitness_value),
        "validation_fe_used": int(validation_fe_used),
        "extra_fe_ratio": float(validation_fes_used_total / max(1, int(total_max_fes))),
        "candidate_update_count": int(len(updated_rows)),
        "candidate_owner_soft_count": int(len(owner_soft_rows)),
        "candidate_multi_support_count": int(len(multi_support_rows)),
        "candidate_freeze_count": int(len(freeze_rows)),
        "candidate_mean_update_magnitude": _mean_or_nan([row.get("update_magnitude") for row in updated_rows]),
        "candidate_mean_conflict_prior": _mean_or_nan([row.get("conflict_prior") for row in candidate_rows]),
        "reject_reason": str(reject_reason or ""),
    }


def summarize_validated_coordination_rows(rows, total_max_fes):
    rows = list(rows or [])
    attempted_rows = [row for row in rows if bool(row.get("validation_attempted"))]
    accepted_rows = [row for row in attempted_rows if bool(row.get("validation_accepted"))]
    rejected_rows = [row for row in attempted_rows if not bool(row.get("validation_accepted"))]
    skipped_rows = [row for row in rows if not bool(row.get("validation_attempted"))]
    validation_extra_fe_used = int(sum(int(row.get("validation_fe_used", 0) or 0) for row in rows))
    return {
        "validation_attempt_count": int(len(attempted_rows)),
        "validation_accept_count": int(len(accepted_rows)),
        "validation_reject_count": int(len(rejected_rows)),
        "validation_skip_count": int(len(skipped_rows)),
        "validation_accept_rate": float(len(accepted_rows) / len(attempted_rows)) if attempted_rows else 0.0,
        "validation_extra_fe_used": int(validation_extra_fe_used),
        "validation_extra_fe_ratio": float(validation_extra_fe_used / max(1, int(total_max_fes))),
        "accepted_owner_soft_count": int(sum(int(row.get("candidate_owner_soft_count", 0) or 0) for row in accepted_rows)),
        "rejected_owner_soft_count": int(sum(int(row.get("candidate_owner_soft_count", 0) or 0) for row in rejected_rows)),
        "accepted_multi_support_count": int(sum(int(row.get("candidate_multi_support_count", 0) or 0) for row in accepted_rows)),
        "rejected_multi_support_count": int(sum(int(row.get("candidate_multi_support_count", 0) or 0) for row in rejected_rows)),
        "accepted_freeze_count": int(sum(int(row.get("candidate_freeze_count", 0) or 0) for row in accepted_rows)),
        "rejected_freeze_count": int(sum(int(row.get("candidate_freeze_count", 0) or 0) for row in rejected_rows)),
        "mean_accepted_fitness_delta": _mean_or_nan([row.get("fitness_delta") for row in accepted_rows]),
        "mean_rejected_fitness_delta": _mean_or_nan([row.get("fitness_delta") for row in rejected_rows]),
    }


def is_coordination_selector_enabled(info_aware_config=None):
    if info_aware_config is None:
        return False
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    return bool(getattr(normalized_config, "enable_coordination_selector", False))


def summarize_coordination_selector_probe(rows, info_aware_config=None):
    rows = list(rows or [])
    probe_rows = [row for row in rows if str(row.get("phase", "probe")) == "probe"]
    attempted_rows = [row for row in probe_rows if bool(row.get("validation_attempted"))]
    accepted_rows = [row for row in attempted_rows if bool(row.get("validation_accepted"))]
    rejected_rows = [row for row in attempted_rows if not bool(row.get("validation_accepted"))]
    mean_delta = _mean_or_nan([row.get("fitness_delta") for row in attempted_rows])
    if not np.isfinite(_to_float(mean_delta)):
        mean_delta = 0.0
    return {
        "probe_passes": int(getattr(info_aware_config, "selector_probe_passes", 1) or 1),
        "probe_attempt_count": int(len(attempted_rows)),
        "probe_accept_count": int(len(accepted_rows)),
        "probe_reject_count": int(len(rejected_rows)),
        "probe_accept_rate": float(len(accepted_rows) / len(attempted_rows)) if attempted_rows else 0.0,
        "probe_mean_validation_delta": float(mean_delta),
    }


def decide_coordination_selector_state(probe_summary, info_aware_config=None):
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    min_attempts = int(getattr(normalized_config, "selector_min_attempts", 1) or 1)
    min_accepts = int(getattr(normalized_config, "selector_min_accepts", 1) or 1)
    accept_rate_threshold = float(getattr(normalized_config, "selector_accept_rate_threshold", 0.3) or 0.0)
    mean_delta_threshold = float(getattr(normalized_config, "selector_mean_delta_threshold", 0.0) or 0.0)
    accept_state = str(getattr(normalized_config, "selector_after_probe_state_if_accept", "validated_on") or "validated_on")
    reject_state = str(getattr(normalized_config, "selector_after_probe_state_if_reject", "off") or "off")
    probe_attempt_count = int(probe_summary.get("probe_attempt_count", 0) or 0)
    probe_accept_count = int(probe_summary.get("probe_accept_count", 0) or 0)
    probe_accept_rate = float(probe_summary.get("probe_accept_rate", 0.0) or 0.0)
    probe_mean_validation_delta = float(probe_summary.get("probe_mean_validation_delta", 0.0) or 0.0)

    if probe_attempt_count < min_attempts:
        return {
            "coordination_state": reject_state,
            "selector_reason": "probe_attempt_count_below_min",
        }
    if probe_accept_count < min_accepts:
        return {
            "coordination_state": reject_state,
            "selector_reason": "probe_accept_count_below_min",
        }
    if probe_accept_rate >= accept_rate_threshold and probe_mean_validation_delta > mean_delta_threshold:
        return {
            "coordination_state": accept_state,
            "selector_reason": "probe_accept_rate_and_delta_passed",
        }
    return {
        "coordination_state": reject_state,
        "selector_reason": "probe_signal_below_threshold",
    }


def build_coordination_selector_trace_row(
    pass_id,
    phase,
    coordination_state,
    validation_trace_row=None,
    probe_summary=None,
    selector_decision="",
    selector_reason="",
):
    validation_trace_row = dict(validation_trace_row or {})
    probe_summary = dict(probe_summary or {})
    return {
        "pass_id": int(pass_id),
        "phase": str(phase),
        "coordination_state": str(coordination_state),
        "validation_attempted": bool(validation_trace_row.get("validation_attempted", False)),
        "validation_accepted": bool(validation_trace_row.get("validation_accepted", False)),
        "fitness_before_validation": float(validation_trace_row.get("fitness_before_validation", 0.0) or 0.0),
        "candidate_fitness": float(validation_trace_row.get("candidate_fitness", 0.0) or 0.0),
        "fitness_delta": float(validation_trace_row.get("fitness_delta", 0.0) or 0.0),
        "probe_attempt_count": int(probe_summary.get("probe_attempt_count", 0) or 0),
        "probe_accept_count": int(probe_summary.get("probe_accept_count", 0) or 0),
        "probe_reject_count": int(probe_summary.get("probe_reject_count", 0) or 0),
        "probe_accept_rate": float(probe_summary.get("probe_accept_rate", 0.0) or 0.0),
        "probe_mean_validation_delta": float(probe_summary.get("probe_mean_validation_delta", 0.0) or 0.0),
        "selector_decision": str(selector_decision or ""),
        "selector_reason": str(selector_reason or validation_trace_row.get("reject_reason", "") or ""),
        "extra_fe_ratio": float(validation_trace_row.get("extra_fe_ratio", 0.0) or 0.0),
    }


def summarize_coordination_selector_rows(rows, info_aware_config=None):
    rows = list(rows or [])
    probe_summary = summarize_coordination_selector_probe(rows, info_aware_config)
    decision_rows = [row for row in rows if str(row.get("selector_decision", ""))]
    post_probe_rows = [row for row in rows if str(row.get("phase", "")) == "selected"]
    final_decision = str(decision_rows[-1].get("selector_decision")) if decision_rows else "probe"
    final_reason = str(decision_rows[-1].get("selector_reason", "")) if decision_rows else "probe_not_completed"
    if final_decision == "probe" and rows:
        final_decision = str(rows[-1].get("coordination_state", "probe"))
    return {
        **probe_summary,
        "final_coordination_state": final_decision,
        "selector_reason": final_reason,
        "selector_trace_count": int(len(rows)),
        "post_probe_pass_count": int(len(post_probe_rows)),
        "post_probe_off_count": int(sum(str(row.get("coordination_state", "")) == "off" for row in post_probe_rows)),
        "post_probe_validated_on_count": int(
            sum(str(row.get("coordination_state", "")) == "validated_on" for row in post_probe_rows)
        ),
    }


def annotate_coordination_outcome_rows(
    cycle_fusion_rows,
    fitness_before_validation,
    candidate_fitness,
    best_improved,
    validation_attempted,
    validation_accepted,
    reject_reason="",
):
    validation_status = "accepted" if validation_accepted else ("rejected" if validation_attempted else "skipped")
    annotated_rows = []
    candidate_fitness_value = float(candidate_fitness) if np.isfinite(_to_float(candidate_fitness)) else float(fitness_before_validation)
    for row in list(cycle_fusion_rows or []):
        annotated_row = dict(row)
        annotated_row["post_coordination_fitness_before"] = float(fitness_before_validation)
        annotated_row["post_coordination_fitness_after"] = float(candidate_fitness_value)
        annotated_row["post_coordination_best_improved"] = bool(bool(annotated_row.get("was_updated")) and best_improved)
        annotated_row["fitness_delta"] = float(fitness_before_validation - candidate_fitness_value)
        annotated_row["validation_attempted"] = bool(validation_attempted)
        annotated_row["validation_accepted"] = bool(validation_accepted)
        annotated_row["validation_status"] = str(validation_status)
        annotated_row["validation_reject_reason"] = str(reject_reason or "")
        annotated_rows.append(annotated_row)
    return annotated_rows


def finalize_pass_end_coordination(
    fun,
    pre_coord_individual,
    pre_coord_fitness,
    coordinated_candidate,
    cycle_fusion_rows,
    cycle_id,
    sum_fes,
    max_fes,
    validation_fes_used,
    info_aware_config=None,
):
    normalized_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    validation_enabled = is_pass_level_validated_coordination_enabled(normalized_config)
    validation_accept_eps = 0.0 if normalized_config is None else float(getattr(normalized_config, "validation_accept_eps", 0.0) or 0.0)
    validation_max_extra_fes = resolve_validation_max_extra_fes(max_fes, normalized_config)
    candidate_updated = any(bool(row.get("applied_update")) for row in (cycle_fusion_rows or []))
    best_individual = np.asarray(pre_coord_individual, dtype=float).copy()
    current_best_fitness = float(pre_coord_fitness)
    candidate_fitness = float(pre_coord_fitness)
    validation_attempted = False
    validation_accepted = False
    validation_fe_used = 0
    reject_reason = ""

    if not candidate_updated:
        reject_reason = "no_candidate_update"
    elif not validation_enabled:
        best_individual = np.asarray(coordinated_candidate, dtype=float).copy()
        if int(sum_fes) < int(max_fes):
            candidate_fitness = float(np.asarray(fun(best_individual)).reshape(-1)[0])
            sum_fes += 1
            current_best_fitness = float(candidate_fitness)
        else:
            candidate_fitness = float(pre_coord_fitness)
            current_best_fitness = float(pre_coord_fitness)
    else:
        if int(sum_fes) >= int(max_fes):
            reject_reason = "max_fes_exhausted"
        elif int(validation_fes_used) >= int(validation_max_extra_fes):
            reject_reason = "validation_fe_cap_reached"
        else:
            validation_attempted = True
            candidate_fitness = float(np.asarray(fun(np.asarray(coordinated_candidate, dtype=float))).reshape(-1)[0])
            sum_fes += 1
            validation_fes_used += 1
            validation_fe_used = 1
            if candidate_fitness < float(pre_coord_fitness) - float(validation_accept_eps):
                validation_accepted = True
                best_individual = np.asarray(coordinated_candidate, dtype=float).copy()
                current_best_fitness = float(candidate_fitness)
            else:
                reject_reason = "candidate_not_improved"

    best_improved = bool(validation_accepted) if validation_enabled else bool(
        candidate_updated and candidate_fitness < float(pre_coord_fitness) - 1e-12
    )
    annotated_rows = annotate_coordination_outcome_rows(
        cycle_fusion_rows,
        fitness_before_validation=pre_coord_fitness,
        candidate_fitness=candidate_fitness,
        best_improved=best_improved,
        validation_attempted=validation_attempted,
        validation_accepted=validation_accepted,
        reject_reason=reject_reason,
    )
    validation_trace_row = None
    if validation_enabled:
        validation_trace_row = build_validated_coordination_trace_row(
            cycle_id=cycle_id,
            cycle_fusion_rows=annotated_rows,
            fitness_before_validation=pre_coord_fitness,
            candidate_fitness=candidate_fitness,
            validation_attempted=validation_attempted,
            validation_accepted=validation_accepted,
            validation_fe_used=validation_fe_used,
            validation_fes_used_total=validation_fes_used,
            total_max_fes=max_fes,
            reject_reason=reject_reason,
        )
    return {
        "best_individual": np.asarray(best_individual, dtype=float).copy(),
        "current_best_fitness": float(current_best_fitness),
        "candidate_fitness": float(candidate_fitness),
        "sum_fes": int(sum_fes),
        "validation_fes_used": int(validation_fes_used),
        "validation_trace_row": validation_trace_row,
        "cycle_fusion_rows": annotated_rows,
    }


def summarize_overlap_blend_rows(rows, eps_delta=1e-12):
    rows = list(rows or [])
    active_rows = [row for row in rows if int(row.get("overlap_count", 0) or 0) > 0]
    applied_rows = [row for row in active_rows if bool(row.get("applied_update"))]
    total_active = len(active_rows)
    if total_active == 0:
        return {
            "blend_strategy": "",
            "total_count": int(len(rows)),
            "active_overlap_count": 0,
            "skip_ratio": 0.0,
            "both_positive_ratio": 0.0,
            "current_only_ratio": 0.0,
            "previous_only_ratio": 0.0,
            "mean_conflict": 0.0,
            "mean_damping": 0.0,
            "negative_delta_blend_ratio": 0.0,
        }

    def ratio_for_mode(mode_name):
        return float(np.mean([str(row.get("blend_mode", "")) == mode_name for row in active_rows]))

    negative_delta_blend_ratio = float(
        np.mean(
            [
                bool(row.get("applied_update"))
                and (
                    float(row.get("previous_delta", float("nan"))) <= float(eps_delta)
                    or float(row.get("current_delta", float("nan"))) <= float(eps_delta)
                )
                for row in active_rows
            ]
        )
    )
    blend_strategy = str(active_rows[0].get("blend_strategy", ""))
    return {
        "blend_strategy": blend_strategy,
        "total_count": int(len(rows)),
        "active_overlap_count": int(total_active),
        "skip_ratio": float(np.mean([bool(row.get("was_skipped")) for row in active_rows])),
        "both_positive_ratio": ratio_for_mode("both_positive"),
        "current_only_ratio": ratio_for_mode("current_only"),
        "previous_only_ratio": ratio_for_mode("previous_only"),
        "mean_conflict": _mean_or_nan([row.get("conflict_mean") for row in active_rows]),
        "mean_damping": _mean_or_nan([row.get("damping") for row in applied_rows]) if applied_rows else 0.0,
        "negative_delta_blend_ratio": negative_delta_blend_ratio,
    }


def summarize_group_delta_rows(rows, early_fraction=0.25):
    rows = list(rows or [])
    delta_values = np.asarray(
        [
            _to_float(row.get("actual_delta"))
            for row in rows
            if np.isfinite(_to_float(row.get("actual_delta")))
        ],
        dtype=float,
    )
    if delta_values.size == 0:
        return {
            "total_count": 0,
            "positive_delta_rate": 0.0,
            "delta_mean": float("nan"),
            "delta_median": float("nan"),
            "delta_std": float("nan"),
            "early_count": 0,
            "early_delta_mean": float("nan"),
            "early_positive_delta_rate": 0.0,
        }
    early_count = max(1, int(math.ceil(delta_values.size * float(np.clip(early_fraction, 0.0, 1.0)))))
    early_values = delta_values[:early_count]
    return {
        "total_count": int(delta_values.size),
        "positive_delta_rate": float(np.mean(delta_values > 0.0)),
        "delta_mean": float(np.mean(delta_values)),
        "delta_median": float(np.median(delta_values)),
        "delta_std": float(np.std(delta_values)),
        "early_count": int(early_values.size),
        "early_delta_mean": float(np.mean(early_values)),
        "early_positive_delta_rate": float(np.mean(early_values > 0.0)),
    }


def override_info_aware_config(info_aware_config=None, overlap_blend_mode=None):
    if info_aware_config is None and overlap_blend_mode is None:
        return None
    if info_aware_config is None:
        config = InfoAwareNDAConfig()
    elif isinstance(info_aware_config, InfoAwareNDAConfig):
        config = InfoAwareNDAConfig(**info_aware_config.to_dict())
    else:
        return info_aware_config
    if overlap_blend_mode is not None:
        config.overlap_blend_mode = overlap_blend_mode
    return config.normalized()


def _default_nda_info(dimension, warning):
    info = NDAInfo(
        fe_used=0,
        warnings=[str(warning)],
    )
    info.var_contribution = np.ones(int(dimension), dtype=float)
    info.var_stability = np.ones(int(dimension), dtype=float)
    info.var_direction = np.zeros(int(dimension), dtype=int)
    return info


def _resolve_cc_schedule(grouping_result, adjacent_overlapping_elements, remaining_fes, cc_prior, info_aware_config):
    group_count = len(grouping_result)
    if group_count <= 0:
        return [], [], [], [], {
            "priority_mode_effective": "off",
            "sort_dangerous_ablation_changed_order": False,
            "warnings": [],
            "original_group_ids": [],
            "cc_pass_budget_cap_enabled": False,
            "cc_pass_group_fes": "",
            "cc_min_passes": 1,
            "cc_max_passes": "",
            "raw_uniform_subfes": 0,
            "effective_uniform_subfes": 0,
        }

    raw_uniform_subfes = int(math.ceil(int(remaining_fes) / float(group_count)))
    cc_pass_group_fes = getattr(info_aware_config, "cc_pass_group_fes", None)
    cc_pass_budget_cap_enabled = cc_pass_group_fes is not None and int(cc_pass_group_fes) > 0
    uniform_subfes = min(raw_uniform_subfes, int(cc_pass_group_fes)) if cc_pass_budget_cap_enabled else raw_uniform_subfes
    default_budgets = [int(uniform_subfes)] * group_count
    execution_order = list(range(group_count))
    scheduled_groups = grouping_result
    scheduled_adjacent_overlaps = adjacent_overlapping_elements
    schedule_metadata = {
        "priority_mode_effective": "off",
        "sort_dangerous_ablation_changed_order": False,
        "warnings": [],
        "original_group_ids": execution_order.copy(),
        "cc_pass_budget_cap_enabled": bool(cc_pass_budget_cap_enabled),
        "cc_pass_group_fes": int(cc_pass_group_fes) if cc_pass_budget_cap_enabled else "",
        "cc_min_passes": int(getattr(info_aware_config, "cc_min_passes", 1) or 1),
        "cc_max_passes": (
            int(getattr(info_aware_config, "cc_max_passes"))
            if getattr(info_aware_config, "cc_max_passes", None) is not None
            else ""
        ),
        "raw_uniform_subfes": int(raw_uniform_subfes),
        "effective_uniform_subfes": int(uniform_subfes),
    }

    if cc_prior is None or info_aware_config is None or not info_aware_config.enable_group_priority:
        return scheduled_groups, scheduled_adjacent_overlaps, default_budgets, execution_order, schedule_metadata

    schedule_metadata["priority_mode_effective"] = str(info_aware_config.priority_mode or "off")

    if info_aware_config.priority_mode == "off":
        return scheduled_groups, scheduled_adjacent_overlaps, default_budgets, execution_order, schedule_metadata

    if info_aware_config.priority_mode == "diagnostic_only":
        return scheduled_groups, scheduled_adjacent_overlaps, default_budgets, execution_order, schedule_metadata

    if info_aware_config.priority_mode == "sort_dangerous_ablation":
        execution_order = build_priority_order(cc_prior.group_priority)
        if execution_order:
            scheduled_groups = [grouping_result[group_id] for group_id in execution_order]
            scheduled_adjacent_overlaps = compute_adjacent_overlaps_for_groups(scheduled_groups)
        schedule_metadata["original_group_ids"] = execution_order.copy()
        schedule_metadata["sort_dangerous_ablation_changed_order"] = execution_order != list(range(group_count))
        schedule_metadata["warnings"].append("sort_dangerous_ablation_changes_overlap_order")
        return scheduled_groups, scheduled_adjacent_overlaps, default_budgets, execution_order, schedule_metadata

    if info_aware_config.priority_mode == "budget":
        budgets = allocate_priority_budgets(
            remaining_fes,
            cc_prior.group_priority,
            min_budget=info_aware_config.budget_min_fe,
        )
        if cc_pass_budget_cap_enabled:
            budgets = [min(int(budget), int(cc_pass_group_fes)) for budget in budgets]
        return scheduled_groups, scheduled_adjacent_overlaps, budgets, execution_order, schedule_metadata

    return scheduled_groups, scheduled_adjacent_overlaps, default_budgets, execution_order, schedule_metadata


def run_hcc_core(
    fun,
    output_path,
    best_individual,
    max_fes,
    grouping_result,
    info,
    adjacent_overlapping_elements,
    seed=None,
    method=HCC_ES_METHOD,
    problem_code=None,
    info_aware_config=None,
    return_metadata=False,
):
    time_start = time.time()
    canonicalize_method(method)
    best_individual = np.asarray(best_individual, dtype=float).copy()
    normalized_info_aware_config = info_aware_config.normalized() if isinstance(info_aware_config, InfoAwareNDAConfig) else info_aware_config
    group_order_mode = resolve_group_order_mode(normalized_info_aware_config)
    coordination_order_mode = resolve_coordination_order_mode(normalized_info_aware_config)
    optimizer_state_mode = resolve_optimizer_state_mode(normalized_info_aware_config)
    group_order_plan = build_group_order_plan(
        grouping_result,
        problem_code=problem_code,
        execution_order_mode=group_order_mode,
        coordination_order_mode=coordination_order_mode,
    )
    grouping_result = group_order_plan["execution_groups"]
    group_order = group_order_plan["execution_group_order"]
    group_order_audit = group_order_plan["execution_order_audit"]
    coordination_groups = group_order_plan["coordination_groups"]
    coordination_group_order = group_order_plan["coordination_group_order"]
    coordination_order_audit = group_order_plan["coordination_order_audit"]
    group_variable_index_maps = build_group_variable_index_maps(grouping_result, source_group_ids=group_order)
    adjacent_overlapping_elements = compute_adjacent_overlaps_for_groups(grouping_result)
    coordination_adjacent_overlapping_elements = compute_adjacent_overlaps_for_groups(coordination_groups)
    overlap_hypergraph = build_overlap_hypergraph(grouping_result)
    overlap_features = build_overlap_features(grouping_result, overlap_hypergraph)
    original_do = float(overlap_features["overlap_ratio"])
    has_overlap = bool(overlap_hypergraph["overlap_vars"])
    original_glofes = compute_original_glofes(original_do, max_fes, has_overlap)
    blend_strategy_name = resolve_blend_strategy_name(normalized_info_aware_config)
    coordination_mode = resolve_shared_variable_coordination_mode(normalized_info_aware_config)
    validated_coordination_enabled = is_pass_level_validated_coordination_enabled(normalized_info_aware_config)
    validation_max_extra_fes = resolve_validation_max_extra_fes(max_fes, normalized_info_aware_config)
    coordination_selector_enabled = bool(
        is_coordination_selector_enabled(normalized_info_aware_config)
        and is_hypergraph_pass_end_mode(coordination_mode)
    )
    record_method = resolve_record_method(method, normalized_info_aware_config)
    lower_boundary = np.asarray(info["lower"] * np.ones((info["dimension"],)), dtype=float)
    upper_boundary = np.asarray(info["upper"] * np.ones((info["dimension"],)), dtype=float)
    variable_ranges = build_variable_ranges(lower_boundary, upper_boundary, info["dimension"])
    coordination_visibility_audit = build_shared_variable_visibility_audit(
        coordination_groups if coordination_mode == "adjacent" else grouping_result,
        coordination_adjacent_overlapping_elements if coordination_mode == "adjacent" else adjacent_overlapping_elements,
        coordination_mode=coordination_mode,
        overlap_hypergraph=overlap_hypergraph,
    )

    seed_offset = 0
    current_best_fitness = None
    sum_fes = 0
    nda_info = None
    cc_prior = None
    execution_order = list(range(len(grouping_result)))
    extra_warnings = []
    group_trace_rows = []
    group_delta_rows = []
    overlap_blend_rows = []
    shared_variable_proposal_rows = []
    shared_variable_fusion_rows = []
    validated_coordination_rows = []
    coordination_selector_rows = []
    optimizer_state_rows = []
    omega_cache = {}
    arac_relation_history = {}
    arac_random_probe_state = {}
    coordination_selector_state = "probe" if coordination_selector_enabled else "disabled"
    coordination_selector_reason = ""
    coordination_selector_decision_made = False
    priority_audit = {}
    priority_mode_effective = "off"
    sort_dangerous_ablation_changed_order = False
    schedule_warnings = []
    validation_fes_used = 0
    overlap_blend_mode = str(getattr(normalized_info_aware_config, "overlap_blend_mode", "original") or "original")
    trace_enabled = bool(
        normalized_info_aware_config is not None
        and normalized_info_aware_config.enable
        and normalized_info_aware_config.enable_group_delta_trace
    )
    overlap_blend_diagnostics_enabled = bool(
        normalized_info_aware_config is not None
        and (normalized_info_aware_config.save_diagnostics or normalized_info_aware_config.save_overlap_blend_trace)
    )
    shared_variable_trace_enabled = bool(
        normalized_info_aware_config is not None
        and (normalized_info_aware_config.save_diagnostics or normalized_info_aware_config.save_shared_variable_trace)
    )
    validated_coordination_trace_enabled = bool(
        validated_coordination_enabled
        and normalized_info_aware_config is not None
        and (normalized_info_aware_config.save_diagnostics or normalized_info_aware_config.validation_trace_enabled)
    )
    coordination_selector_trace_enabled = bool(
        coordination_selector_enabled
        and normalized_info_aware_config is not None
        and (normalized_info_aware_config.save_diagnostics or normalized_info_aware_config.selector_trace_enabled)
    )
    group_overlap_counts = [int(value) for value in overlap_features["group_overlap_var_count"]]
    group_overlap_ratios = [
        float(group_overlap_counts[group_id] / max(1, len(grouping_result[group_id])))
        for group_id in range(len(grouping_result))
    ]
    group_conflict_prior_means = [0.0] * len(grouping_result)

    if original_glofes > 0:
        problem_global = {
            "fitness_function": fun,
            "ndim_problem": info["dimension"],
            "lower_boundary": lower_boundary,
            "upper_boundary": upper_boundary,
        }
        options_global = {
            "max_function_evaluations": int(min(original_glofes, max_fes)),
            "mean": (best_individual,),
            "sigma": 0.5,
            "is_restart": False,
            "verbose": 1000,
            "seed_rng": stage_seed(seed, seed_offset),
        }
        seed_offset += 1
        if normalized_info_aware_config is not None and normalized_info_aware_config.enable:
            results_global = run_adaptive_info_aware_nda(
                problem_global,
                options_global,
                normalized_info_aware_config,
                int(max_fes),
                int(original_glofes),
            )
            nda_info = results_global.get("nda_info")
        else:
            optimizer_global = MMES(problem_global, options_global)
            results_global = optimizer_global.optimize()
        best_individual = np.asarray(results_global["best_so_far_x"], dtype=float).copy()
        current_best_fitness = float(results_global["best_so_far_y"])
        sum_fes = int(results_global["n_function_evaluations"])
    elif normalized_info_aware_config is not None and normalized_info_aware_config.enable:
        nda_info = _default_nda_info(info["dimension"], "nda_stage_skipped_due_to_zero_glofes")

    if current_best_fitness is None and sum_fes < int(max_fes):
        current_best_fitness = float(np.asarray(fun(best_individual)).reshape(-1)[0])
        sum_fes += 1

    if normalized_info_aware_config is not None:
        if nda_info is None:
            nda_info = _default_nda_info(
                info["dimension"],
                "info_aware_nda_disabled" if not normalized_info_aware_config.enable else "nda_info_unavailable",
            )
        if normalized_info_aware_config.enable:
            contribution = nda_info.var_contribution
            stability = nda_info.var_stability
            direction = nda_info.var_direction
            if contribution is None:
                contribution = np.ones(info["dimension"], dtype=float)
                extra_warnings.append("var_contribution_missing_fallback_to_ones")
            if stability is None:
                stability = np.ones(info["dimension"], dtype=float)
                extra_warnings.append("var_stability_missing_fallback_to_ones")
            if direction is None:
                direction = np.zeros(info["dimension"], dtype=int)
                extra_warnings.append("var_direction_missing_fallback_to_zeros")
            cc_prior = build_cc_prior(
                grouping_result,
                info["dimension"],
                overlap_hypergraph,
                contribution,
                stability,
                direction,
                normalized_info_aware_config,
            )
            for group_id in range(len(grouping_result)):
                overlap_vars = np.asarray(overlap_hypergraph["group_to_overlap_vars"].get(int(group_id), []), dtype=int)
                if overlap_vars.size > 0:
                    group_conflict_prior_means[group_id] = float(np.mean(cc_prior.conflict_prior[overlap_vars]))

    estimated_arac_lite_pass_count = estimate_arac_lite_cc_pass_count(
        max_fes,
        sum_fes,
        len(grouping_result),
        normalized_info_aware_config,
        has_overlap=has_overlap,
    )
    cycle_id = 0
    while sum_fes < int(max_fes):
        group_count = len(grouping_result)
        if group_count <= 0:
            break
        remaining_fes = int(max_fes - sum_fes)
        if remaining_fes <= 0:
            break
        selector_phase = "disabled"
        selector_cycle_state = "disabled"
        effective_coordination_mode = coordination_mode
        effective_validated_coordination_enabled = validated_coordination_enabled
        if coordination_selector_enabled:
            probe_passes = int(getattr(normalized_info_aware_config, "selector_probe_passes", 1) or 1)
            if not coordination_selector_decision_made and int(cycle_id) < probe_passes:
                selector_phase = "probe"
                selector_cycle_state = "probe"
                effective_coordination_mode = "selective_hypergraph_pass_end"
                effective_validated_coordination_enabled = True
            else:
                if not coordination_selector_decision_made:
                    probe_summary = summarize_coordination_selector_probe(
                        coordination_selector_rows,
                        normalized_info_aware_config,
                    )
                    selector_decision = decide_coordination_selector_state(
                        probe_summary,
                        normalized_info_aware_config,
                    )
                    coordination_selector_state = str(selector_decision["coordination_state"])
                    coordination_selector_reason = str(selector_decision["selector_reason"])
                    coordination_selector_decision_made = True
                selector_phase = "selected"
                selector_cycle_state = coordination_selector_state
                if coordination_selector_state == "off":
                    effective_coordination_mode = "no_coordination"
                    effective_validated_coordination_enabled = False
                else:
                    effective_coordination_mode = "selective_hypergraph_pass_end"
                    effective_validated_coordination_enabled = True
        cycle_validation_budget_available = bool(
            effective_validated_coordination_enabled and int(validation_fes_used) < int(validation_max_extra_fes)
        )
        reserve_pass_end_evaluation = bool(
            is_hypergraph_pass_end_mode(effective_coordination_mode)
            and has_overlap
            and remaining_fes > 1
            and (cycle_validation_budget_available or not effective_validated_coordination_enabled)
        )
        scheduled_remaining_fes = int(remaining_fes - 1) if reserve_pass_end_evaluation else int(remaining_fes)
        if scheduled_remaining_fes <= 0:
            break
        arac_lite_recovery_phase = arac_lite_phase_for_cycle(cycle_id, estimated_arac_lite_pass_count)

        scheduled_groups, scheduled_adjacent_overlaps, subspace_budgets, execution_order, schedule_metadata = _resolve_cc_schedule(
            grouping_result,
            adjacent_overlapping_elements,
            scheduled_remaining_fes,
            cc_prior,
            normalized_info_aware_config,
        )
        priority_mode_effective = schedule_metadata["priority_mode_effective"]
        sort_dangerous_ablation_changed_order = bool(schedule_metadata["sort_dangerous_ablation_changed_order"])
        schedule_warnings = merge_warnings(schedule_warnings, schedule_metadata.get("warnings", []))
        scheduled_original_group_ids = list(schedule_metadata.get("original_group_ids", execution_order))
        scheduled_source_group_ids = [
            int(group_order[int(group_id)]) if int(group_id) < len(group_order) else int(group_id)
            for group_id in scheduled_original_group_ids
        ]
        if coordination_order_mode == "match_execution":
            active_coordination_groups = [list(group) for group in scheduled_groups]
            active_coordination_group_order = [int(group_id) for group_id in scheduled_source_group_ids]
        else:
            active_coordination_groups = coordination_groups
            active_coordination_group_order = coordination_group_order
        group_deltas = np.full(len(scheduled_groups), float("nan"), dtype=float)
        group_deltas_by_source = {}
        group_snapshots_before = {}
        group_proposals_after = {}
        completed_group_ids = set()
        processed_coordination_edges = set()
        cycle_shared_variable_proposals = []

        for group_id, dims in enumerate(scheduled_groups):
            reserved_eval_fe = 1 if reserve_pass_end_evaluation else 0
            max_group_fes = int(max_fes - sum_fes - reserved_eval_fe)
            sub_fes = min(int(subspace_budgets[group_id]), max_group_fes)
            if sub_fes <= 0:
                continue
            dims = list(dims)
            original_group_id = int(scheduled_original_group_ids[group_id]) if group_id < len(scheduled_original_group_ids) else int(group_id)
            source_group_id = int(scheduled_source_group_ids[group_id]) if group_id < len(scheduled_source_group_ids) else int(original_group_id)
            original_best_individual = best_individual.copy()
            group_snapshots_before[int(source_group_id)] = original_best_individual
            original_best_fitness = float(current_best_fitness)
            assigned_budget = int(subspace_budgets[group_id])
            objective = lambda x_batch, dims=dims, current=best_individual: fun(combine(x_batch, current, dims))
            problem_cc = {
                "fitness_function": objective,
                "ndim_problem": len(dims),
                "lower_boundary": lower_boundary[dims],
                "upper_boundary": upper_boundary[dims],
            }
            if optimizer_state_mode == "persistent_mean":
                if int(source_group_id) not in omega_cache:
                    omega_cache[int(source_group_id)] = np.asarray(best_individual[dims], dtype=float).copy()
                    mean_init_source = "omega_cache_init"
                else:
                    mean_init_source = "omega_cache"
                cc_mean = np.asarray(omega_cache[int(source_group_id)], dtype=float).copy()
            else:
                mean_init_source = "best_individual"
                cc_mean = np.asarray(best_individual[dims], dtype=float).copy()
            mean_mismatch = float(
                np.max(np.abs(cc_mean - np.asarray(best_individual[dims], dtype=float)))
            ) if cc_mean.size else 0.0
            optimizer_state_rows.append(
                {
                    "problem": str(problem_code).upper() if problem_code else "",
                    "method": str(record_method),
                    "tfes": int(max_fes),
                    "seed": int(seed) if seed is not None else "",
                    "cycle_id": int(cycle_id),
                    "pass_id": int(cycle_id),
                    "scheduled_position": int(group_id),
                    "group_id": int(original_group_id),
                    "source_group_id": int(source_group_id),
                    "event": "mean_init",
                    "var_id": "",
                    "mean_init_source": mean_init_source,
                    "mean_before_coord_overlap": float("nan"),
                    "mean_after_coord_overlap": float("nan"),
                    "best_before_coord_overlap": float("nan"),
                    "best_after_coord_overlap": float("nan"),
                    "mean_coord_mismatch": mean_mismatch,
                    "mean_coord_mismatch_after_sync": mean_mismatch,
                    "optimizer_reinitialized": True,
                }
            )
            options_cc = {
                "max_function_evaluations": int(sub_fes),
                "mean": (cc_mean,),
                "sigma": 0.5,
                "is_restart": False,
                "verbose": 1000,
                "early_stopping_evaluations": 1000,
                "seed_rng": stage_seed(seed, seed_offset),
            }
            seed_offset += 1
            optimizer_cc = CMAES(problem_cc, options_cc)
            results_cc = optimizer_cc.optimize()
            best_individual[dims] = np.asarray(results_cc["best_so_far_x"], dtype=float).copy()
            if optimizer_state_mode == "persistent_mean":
                omega_cache[int(source_group_id)] = np.asarray(results_cc["best_so_far_x"], dtype=float).copy()
            group_proposals_after[int(source_group_id)] = best_individual.copy()
            actual_fe = int(results_cc["n_function_evaluations"])
            sum_fes += actual_fe
            subspace_candidate_fitness = float(results_cc["best_so_far_y"])
            delta = float(original_best_fitness - subspace_candidate_fitness)
            group_deltas[group_id] = delta
            group_deltas_by_source[int(source_group_id)] = delta
            group_delta_rows.append(
                {
                    "cycle_id": int(cycle_id),
                    "scheduled_position": int(group_id),
                    "original_group_id": int(original_group_id),
                    "source_group_id": int(source_group_id),
                    "actual_delta": float(delta),
                    "fitness_before": float(original_best_fitness),
                    "fitness_after": float(subspace_candidate_fitness),
                    "actual_fe": int(actual_fe),
                    "group_size": int(len(dims)),
                }
            )
            if trace_enabled:
                priority_value = float("nan")
                if cc_prior is not None and original_group_id < len(cc_prior.group_priority):
                    priority_value = float(cc_prior.group_priority[original_group_id])
                group_trace_rows.append(
                    {
                        "cycle_id": int(cycle_id),
                        "scheduled_position": int(group_id),
                        "original_group_id": int(original_group_id),
                        "group_size": int(len(dims)),
                        "priority": priority_value,
                        "budget": int(assigned_budget),
                        "actual_fe": int(actual_fe),
                        "fitness_before": float(original_best_fitness),
                        "fitness_after": float(subspace_candidate_fitness),
                        "actual_delta": float(delta),
                        "overlap_var_count": int(group_overlap_counts[original_group_id]),
                        "overlap_ratio": float(group_overlap_ratios[original_group_id]),
                        "conflict_prior_mean": float(group_conflict_prior_means[original_group_id]),
                        "was_sorted": bool(priority_mode_effective == "sort_dangerous_ablation"),
                    }
                )
            if is_hypergraph_pass_end_mode(effective_coordination_mode):
                cycle_shared_variable_proposals.extend(
                    collect_group_overlap_variable_proposals(
                        best_individual,
                        overlap_hypergraph["group_to_overlap_vars"].get(int(original_group_id), []),
                        original_group_id,
                        delta,
                        original_best_fitness,
                        subspace_candidate_fitness,
                        cycle_id,
                        group_id,
                    )
                )
            if effective_coordination_mode == "adjacent":
                completed_group_ids.add(int(source_group_id))
                ready_coordination_edges = pop_ready_coordination_edges(
                    execution_group_id=source_group_id,
                    completed_group_ids=completed_group_ids,
                    coordination_groups=active_coordination_groups,
                    coordination_group_order=active_coordination_group_order,
                    processed_coordination_edges=processed_coordination_edges,
                )
            else:
                ready_coordination_edges = []
            for coordination_edge in ready_coordination_edges:
                left_original_group_id = int(coordination_edge["left_group_id"])
                right_original_group_id = int(coordination_edge["right_group_id"])
                previous_original_group_id = left_original_group_id
                current_coordination_group_id = right_original_group_id
                overlap_indices = np.asarray(coordination_edge["overlap_indices"], dtype=int)
                previous_delta = group_deltas_by_source.get(previous_original_group_id, float("nan"))
                current_delta_for_edge = group_deltas_by_source.get(current_coordination_group_id, float("nan"))
                edge_original_best_individual = group_snapshots_before.get(
                    current_coordination_group_id,
                    original_best_individual,
                )
                previous_proposal_individual = group_proposals_after.get(
                    previous_original_group_id,
                    best_individual,
                )
                current_proposal_individual = group_proposals_after.get(
                    current_coordination_group_id,
                    best_individual,
                )
                pre_coordination_best_individual = np.asarray(best_individual, dtype=float).copy()
                blend_diagnostics = apply_coordination_edge_blend(
                    best_individual,
                    previous_proposal_individual,
                    current_proposal_individual,
                    edge_original_best_individual,
                    overlap_indices,
                    previous_delta,
                    current_delta_for_edge,
                    overlap_blend_mode=overlap_blend_mode,
                    conflict_prior=cc_prior.conflict_prior if cc_prior is not None else None,
                    blend_config=normalized_info_aware_config,
                    use_pairwise_endpoint_values=coordination_order_mode != "match_execution",
                )
                if optimizer_state_mode == "persistent_mean" and bool(blend_diagnostics.get("applied_update")):
                    sync_rows = sync_omega_cache_for_variables(
                        omega_cache,
                        best_individual,
                        overlap_indices,
                        group_variable_index_maps["group_variable_index"],
                        group_variable_index_maps["variable_to_group_ids"],
                        source_group_dims=group_variable_index_maps["source_group_dims"],
                        best_before_individual=pre_coordination_best_individual,
                        cycle_id=cycle_id,
                        pass_id=cycle_id,
                        scheduled_position=group_id,
                    )
                    for row in sync_rows:
                        row.update(
                            {
                                "problem": str(problem_code).upper() if problem_code else "",
                                "method": str(record_method),
                                "tfes": int(max_fes),
                                "seed": int(seed) if seed is not None else "",
                            }
                        )
                    optimizer_state_rows.extend(sync_rows)
                if overlap_blend_diagnostics_enabled:
                    overlap_blend_rows.append(
                        {
                            "problem": str(problem_code).upper() if problem_code else "",
                            "method": str(record_method),
                            "tfes": int(max_fes),
                            "seed": int(seed) if seed is not None else "",
                            "cycle_id": int(cycle_id),
                            "group_id": int(current_coordination_group_id),
                            "previous_group_id": int(previous_original_group_id),
                            "scheduled_position": int(group_id),
                            **blend_diagnostics,
                        }
                    )
            current_best_fitness = subspace_candidate_fitness
            if sum_fes >= int(max_fes):
                break
        finalized_coordination = None
        if is_hypergraph_pass_end_mode(effective_coordination_mode) and cycle_shared_variable_proposals:
            pre_coordination_individual = np.asarray(best_individual, dtype=float).copy()
            pre_coordination_fitness = float(current_best_fitness)
            coordinated_candidate = np.asarray(pre_coordination_individual, dtype=float).copy()
            conflict_prior = None
            use_conflict_damping = False
            if normalized_info_aware_config is not None and bool(normalized_info_aware_config.shared_variable_conflict_damping):
                conflict_prior = cc_prior.conflict_prior if cc_prior is not None else None
                use_conflict_damping = True
                if conflict_prior is None:
                    schedule_warnings = merge_warnings(schedule_warnings, ["shared_variable_conflict_damping_missing_conflict_prior"])
            cycle_fusion_rows = apply_hypergraph_pass_end_coordination(
                coordinated_candidate,
                cycle_shared_variable_proposals,
                overlap_hypergraph,
                conflict_prior=conflict_prior,
                eps_delta=getattr(normalized_info_aware_config, "shared_variable_eps_delta", 1e-12),
                use_conflict_damping=use_conflict_damping,
                conflict_gamma=getattr(normalized_info_aware_config, "shared_variable_conflict_gamma", 0.5),
                min_damping=getattr(normalized_info_aware_config, "shared_variable_min_damping", 0.3),
                coordination_mode=effective_coordination_mode,
                variable_ranges=variable_ranges,
                selective_min_positive_proposals=getattr(normalized_info_aware_config, "shared_variable_selective_min_positive_proposals", 2),
                selective_max_proposal_std_ratio=getattr(normalized_info_aware_config, "shared_variable_selective_max_proposal_std_ratio", 0.00125),
                selective_max_update_ratio=getattr(normalized_info_aware_config, "shared_variable_selective_max_update_ratio", 0.0025),
                selective_large_update_damping=getattr(normalized_info_aware_config, "shared_variable_selective_large_update_damping", 0.25),
                selective_owner_soft_eta=getattr(normalized_info_aware_config, "shared_variable_owner_soft_eta", 0.2),
                selective_owner_min_delta_ratio=getattr(normalized_info_aware_config, "shared_variable_owner_min_delta_ratio", 0.001),
                arac_relation_history=arac_relation_history,
                arac_lite_history_min_attempts=getattr(normalized_info_aware_config, "arac_lite_history_min_attempts", 2),
                arac_lite_disable_accept_rate_threshold=getattr(
                    normalized_info_aware_config,
                    "arac_lite_disable_accept_rate_threshold",
                    0.0,
                ),
                arac_lite_disable_mean_delta_threshold=getattr(
                    normalized_info_aware_config,
                    "arac_lite_disable_mean_delta_threshold",
                    0.0,
                ),
                arac_lite_disable_reject_streak=getattr(normalized_info_aware_config, "arac_lite_disable_reject_streak", 2),
                arac_lite_recovery_enabled=getattr(normalized_info_aware_config, "arac_lite_recovery_enabled", False),
                arac_lite_recovery_phase=arac_lite_recovery_phase,
                arac_lite_recovery_min_attempts=getattr(normalized_info_aware_config, "arac_lite_recovery_min_attempts", 20),
                arac_lite_recovery_accept_rate_threshold=getattr(
                    normalized_info_aware_config,
                    "arac_lite_recovery_accept_rate_threshold",
                    0.25,
                ),
                arac_lite_recovery_delta_threshold=getattr(normalized_info_aware_config, "arac_lite_recovery_delta_threshold", 0.0),
                arac_lite_recovery_positive_delta_rate_threshold=getattr(
                    normalized_info_aware_config,
                    "arac_lite_recovery_positive_delta_rate_threshold",
                    0.25,
                ),
                arac_lite_recovery_recent_window=getattr(normalized_info_aware_config, "arac_lite_recovery_recent_window", 20),
                arac_lite_recovery_min_phase=getattr(normalized_info_aware_config, "arac_lite_recovery_min_phase", "middle"),
                arac_lite_probe_enabled=getattr(normalized_info_aware_config, "arac_lite_probe_enabled", False),
                arac_lite_probe_phase=arac_lite_recovery_phase,
                arac_lite_probe_pass_id=cycle_id,
                arac_lite_probe_every_n_pass=getattr(normalized_info_aware_config, "arac_lite_probe_every_n_pass", 2),
                arac_lite_probe_max_per_pass=getattr(normalized_info_aware_config, "arac_lite_probe_max_per_pass", 5),
                arac_lite_probe_min_phase=getattr(normalized_info_aware_config, "arac_lite_probe_min_phase", "middle"),
                arac_lite_probe_recovery_min_attempts=getattr(normalized_info_aware_config, "arac_lite_probe_recovery_min_attempts", 5),
                arac_lite_probe_recovery_accept_rate_threshold=getattr(
                    normalized_info_aware_config,
                    "arac_lite_probe_recovery_accept_rate_threshold",
                    0.3,
                ),
                arac_lite_probe_recovery_delta_threshold=getattr(
                    normalized_info_aware_config,
                    "arac_lite_probe_recovery_delta_threshold",
                    0.0,
                ),
                arac_lite_probe_recovery_recent_window=getattr(normalized_info_aware_config, "arac_lite_probe_recovery_recent_window", 20),
                arac_lite_targeted_probe_enabled=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_enabled",
                    False,
                ),
                arac_lite_targeted_probe_phase=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_phase",
                    "middle",
                ),
                arac_lite_targeted_probe_min_support=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_min_support",
                    2,
                ),
                arac_lite_targeted_probe_min_relation_attempts=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_min_relation_attempts",
                    1,
                ),
                arac_lite_targeted_probe_min_accept_rate=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_min_accept_rate",
                    0.3,
                ),
                arac_lite_targeted_probe_min_positive_delta_rate=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_min_positive_delta_rate",
                    0.3,
                ),
                arac_lite_targeted_probe_min_relation_delta=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_min_relation_delta",
                    0.0,
                ),
                arac_lite_targeted_probe_min_proposal_std_ratio=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_min_proposal_std_ratio",
                    0.00125,
                ),
                arac_lite_targeted_probe_max_proposal_std_ratio=getattr(
                    normalized_info_aware_config,
                    "arac_lite_targeted_probe_max_proposal_std_ratio",
                    0.0045,
                ),
                arac_lite_random_probe_same_budget_enabled=getattr(
                    normalized_info_aware_config,
                    "arac_lite_random_probe_same_budget_enabled",
                    False,
                ),
                arac_lite_random_probe_budget=getattr(
                    normalized_info_aware_config,
                    "arac_lite_random_probe_budget",
                    0,
                ),
                arac_lite_random_probe_phase=getattr(
                    normalized_info_aware_config,
                    "arac_lite_random_probe_phase",
                    "middle",
                ),
                arac_lite_random_probe_seed=getattr(
                    normalized_info_aware_config,
                    "arac_lite_random_probe_seed",
                    0,
                ),
                arac_lite_random_probe_state=arac_random_probe_state,
            )
            finalized_coordination = finalize_pass_end_coordination(
                fun=fun,
                pre_coord_individual=pre_coordination_individual,
                pre_coord_fitness=pre_coordination_fitness,
                coordinated_candidate=coordinated_candidate,
                cycle_fusion_rows=cycle_fusion_rows,
                cycle_id=cycle_id,
                sum_fes=sum_fes,
                max_fes=max_fes,
                validation_fes_used=validation_fes_used,
                info_aware_config=normalized_info_aware_config,
            )
            best_individual = np.asarray(finalized_coordination["best_individual"], dtype=float).copy()
            current_best_fitness = float(finalized_coordination["current_best_fitness"])
            sum_fes = int(finalized_coordination["sum_fes"])
            validation_fes_used = int(finalized_coordination["validation_fes_used"])
            cycle_fusion_rows = list(finalized_coordination["cycle_fusion_rows"])
            if is_arac_lite_mode(effective_coordination_mode):
                update_arac_relation_history(arac_relation_history, cycle_fusion_rows)
            if shared_variable_trace_enabled:
                shared_variable_proposal_rows.extend(
                    {
                        "problem": str(problem_code).upper() if problem_code else "",
                        "method": str(record_method),
                        "tfes": int(max_fes),
                        "seed": int(seed) if seed is not None else "",
                        **row,
                    }
                    for row in cycle_shared_variable_proposals
                )
                shared_variable_fusion_rows.extend(
                    {
                        "problem": str(problem_code).upper() if problem_code else "",
                        "method": str(record_method),
                        "tfes": int(max_fes),
                        "seed": int(seed) if seed is not None else "",
                        "cycle_id": int(cycle_id),
                        **row,
                    }
                    for row in cycle_fusion_rows
                )
            if validated_coordination_trace_enabled and finalized_coordination["validation_trace_row"] is not None:
                validated_coordination_rows.append(
                    {
                        "problem": str(problem_code).upper() if problem_code else "",
                        "method": str(record_method),
                        "tfes": int(max_fes),
                        "seed": int(seed) if seed is not None else "",
                        **finalized_coordination["validation_trace_row"],
                    }
                )
        if coordination_selector_trace_enabled:
            validation_trace_row = (
                dict(finalized_coordination["validation_trace_row"])
                if finalized_coordination is not None and finalized_coordination.get("validation_trace_row") is not None
                else {
                    "validation_attempted": False,
                    "validation_accepted": False,
                    "fitness_before_validation": float(current_best_fitness),
                    "candidate_fitness": float(current_best_fitness),
                    "fitness_delta": 0.0,
                    "validation_fe_used": 0,
                    "extra_fe_ratio": float(validation_fes_used / max(1, int(max_fes))),
                    "reject_reason": "coordination_off" if effective_coordination_mode == "no_coordination" else "no_candidate_update",
                }
            )
            projected_selector_row = build_coordination_selector_trace_row(
                pass_id=cycle_id,
                phase=selector_phase,
                coordination_state=selector_cycle_state,
                validation_trace_row=validation_trace_row,
            )
            projected_selector_rows = coordination_selector_rows + [projected_selector_row]
            current_probe_summary = summarize_coordination_selector_probe(
                projected_selector_rows,
                normalized_info_aware_config,
            )
            selector_decision_value = ""
            selector_reason_value = validation_trace_row.get("reject_reason", "")
            if (
                selector_phase == "probe"
                and not coordination_selector_decision_made
                and int(cycle_id + 1) >= int(getattr(normalized_info_aware_config, "selector_probe_passes", 1) or 1)
            ):
                selector_decision = decide_coordination_selector_state(
                    current_probe_summary,
                    normalized_info_aware_config,
                )
                coordination_selector_state = str(selector_decision["coordination_state"])
                coordination_selector_reason = str(selector_decision["selector_reason"])
                coordination_selector_decision_made = True
                selector_decision_value = coordination_selector_state
                selector_reason_value = coordination_selector_reason
            coordination_selector_rows.append(
                {
                    "problem": str(problem_code).upper() if problem_code else "",
                    "seed": int(seed) if seed is not None else "",
                    "tfes": int(max_fes),
                    "method": str(record_method),
                    **build_coordination_selector_trace_row(
                        pass_id=cycle_id,
                        phase=selector_phase,
                        coordination_state=selector_cycle_state,
                        validation_trace_row=validation_trace_row,
                        probe_summary=current_probe_summary,
                        selector_decision=selector_decision_value,
                        selector_reason=selector_reason_value,
                    ),
                }
            )
        cycle_id += 1

    time_end = time.time()
    curve = list(getattr(fun, "fitness_record", []))
    metadata = {}
    if trace_enabled:
        priority_audit = compute_priority_audit(
            group_trace_rows,
            topk=normalized_info_aware_config.priority_audit_topk if normalized_info_aware_config is not None else 3,
        )
        metadata["group_trace_rows"] = group_trace_rows
    overlap_blend_summary = summarize_overlap_blend_rows(
        overlap_blend_rows,
        eps_delta=getattr(normalized_info_aware_config, "overlap_blend_eps_delta", 1e-12),
    )
    group_delta_summary = summarize_group_delta_rows(group_delta_rows)
    validated_coordination_summary = summarize_validated_coordination_rows(validated_coordination_rows, total_max_fes=max_fes)
    validated_coordination_summary["validation_enabled"] = bool(validated_coordination_enabled)
    validated_coordination_summary["validation_max_extra_fes"] = int(validation_max_extra_fes)
    coordination_selector_summary = summarize_coordination_selector_rows(
        coordination_selector_rows,
        normalized_info_aware_config,
    )
    coordination_selector_summary["selector_enabled"] = bool(coordination_selector_enabled)
    coordination_selector_summary["selector_decision_made"] = bool(coordination_selector_decision_made)
    coordination_selector_summary["cc_pass_count"] = int(cycle_id)
    if coordination_selector_enabled and not coordination_selector_decision_made:
        coordination_selector_summary["final_coordination_state"] = str(coordination_selector_state)
        coordination_selector_summary["selector_reason"] = str(coordination_selector_reason or "probe_not_completed")
    optimizer_state_summary = summarize_optimizer_state_rows(optimizer_state_rows)
    shared_variable_fusion_rows, owner_soft_effect_audit = build_owner_soft_effect_audit(
        shared_variable_proposal_rows,
        shared_variable_fusion_rows,
    )
    shared_variable_coordination_summary = summarize_shared_variable_coordination_rows(
        shared_variable_fusion_rows,
        visibility_audit=coordination_visibility_audit,
    )
    shared_variable_coordination_summary.update(owner_soft_effect_audit)
    if overlap_blend_diagnostics_enabled:
        metadata["overlap_blend_rows"] = overlap_blend_rows
        metadata["overlap_blend_summary"] = overlap_blend_summary
        metadata["record_method"] = record_method
        metadata["blend_strategy"] = blend_strategy_name
    if shared_variable_trace_enabled:
        metadata["shared_variable_proposal_rows"] = shared_variable_proposal_rows
        metadata["shared_variable_fusion_rows"] = shared_variable_fusion_rows
        metadata["shared_variable_coordination_summary"] = shared_variable_coordination_summary
        metadata["coordination_visibility_audit"] = coordination_visibility_audit
        metadata["owner_soft_effect_audit"] = owner_soft_effect_audit
    metadata["cc_pass_count"] = int(cycle_id)
    metadata["validated_coordination_summary"] = validated_coordination_summary
    metadata["group_delta_summary"] = group_delta_summary
    if coordination_selector_trace_enabled:
        metadata["coordination_selector_rows"] = coordination_selector_rows
        metadata["coordination_selector_summary"] = coordination_selector_summary
    metadata["validation_fes_used"] = int(validation_fes_used)
    metadata["final_selected_fitness"] = float(current_best_fitness)
    metadata["optimizer_state_mode"] = str(optimizer_state_mode)
    metadata["optimizer_state_rows"] = optimizer_state_rows
    metadata["optimizer_state_summary"] = optimizer_state_summary
    metadata["group_order_mode"] = str(group_order_mode)
    metadata["coordination_order_mode"] = str(coordination_order_mode)
    metadata["group_order"] = [int(group_id) for group_id in group_order]
    metadata["group_order_audit"] = group_order_audit
    metadata["execution_group_order"] = [int(group_id) for group_id in group_order]
    metadata["coordination_group_order"] = [int(group_id) for group_id in coordination_group_order]
    metadata["execution_order_audit"] = group_order_audit
    metadata["coordination_order_audit"] = coordination_order_audit
    metadata["arac_relation_history"] = {
        int(var_id): dict(history_entry)
        for var_id, history_entry in sorted(arac_relation_history.items())
    }
    if validated_coordination_trace_enabled:
        metadata["validated_coordination_rows"] = validated_coordination_rows
    if normalized_info_aware_config is not None and normalized_info_aware_config.save_diagnostics:
        metadata["info_aware_diagnostics"] = build_info_aware_diagnostics_payload(
            normalized_info_aware_config,
            nda_info,
            cc_prior,
            int(max_fes),
            execution_order=execution_order,
            warnings=merge_warnings(extra_warnings, schedule_warnings),
            priority_mode_effective=priority_mode_effective,
            sort_dangerous_ablation_changed_order=sort_dangerous_ablation_changed_order,
            group_trace_rows=group_trace_rows,
            priority_audit=priority_audit,
            overlap_blend_rows=overlap_blend_rows,
            overlap_blend_summary=overlap_blend_summary,
            shared_variable_proposal_rows=shared_variable_proposal_rows,
            shared_variable_fusion_rows=shared_variable_fusion_rows,
            shared_variable_coordination_summary=shared_variable_coordination_summary,
            coordination_visibility_audit=coordination_visibility_audit,
            validated_coordination_rows=validated_coordination_rows,
            validated_coordination_summary=validated_coordination_summary,
            coordination_selector_rows=coordination_selector_rows,
            coordination_selector_summary=coordination_selector_summary,
        )
    if return_metadata:
        return curve, (time_end - time_start), [], metadata
    return curve, (time_end - time_start), []


def read_summary_keys(path):
    path = Path(path)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (str(row["problem"]).upper(), str(row["method"]), int(row["seed"]))
            for row in rows
            if row.get("problem") and row.get("method") and row.get("seed")
        }


def write_aggregate_summary(detail_path, diagnostics_path, summary_path):
    detail_rows = _load_csv_rows(detail_path)
    diagnostics_rows = _load_csv_rows(diagnostics_path)
    diagnostics_by_key = {}
    for row in diagnostics_rows:
        key = (str(row.get("problem", "")).upper(), str(row.get("method", "")))
        if key[0] and key[1]:
            diagnostics_by_key.setdefault(key, []).append(row)

    grouped_rows = {}
    for row in detail_rows:
        key = (str(row.get("problem", "")).upper(), str(row.get("method", "")))
        if key[0] and key[1]:
            grouped_rows.setdefault(key, []).append(row)

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for key in sorted(grouped_rows):
            problem, method = key
            rows = grouped_rows[key]
            diagnostics = diagnostics_by_key.get(key, [])
            best_values = [_to_float(row.get("best_fitness")) for row in rows if np.isfinite(_to_float(row.get("best_fitness")))]
            writer.writerow(
                {
                    "problem": problem,
                    "method": method,
                    "tfes": rows[0].get("tfes", "") if rows else "",
                    "blend_strategy": rows[0].get("blend_strategy", "") if rows else "",
                    "runs": int(len(rows)),
                    "best_mean": float(np.mean(best_values)) if best_values else float("nan"),
                    "best_std": float(np.std(best_values)) if best_values else float("nan"),
                    "best_min": float(np.min(best_values)) if best_values else float("nan"),
                    "best_max": float(np.max(best_values)) if best_values else float("nan"),
                    "diagnostics_count": int(len(diagnostics)),
                    "rollback_ratio": rollback_ratio(diagnostics),
                    "conflict_mean": conflict_mean(diagnostics),
                }
            )


def optimization_task(
    fun_name,
    fun_id,
    output_path,
    best_individual,
    max_fes,
    grouping_result,
    info,
    adjacent_overlapping_elements,
    seed=None,
    method=HCC_ES_METHOD,
    problem_code=None,
    info_aware_config=None,
    return_metadata=False,
):
    bench = Benchmark(output_path)
    fun = bench.get_function(fun_name, fun_id)
    return run_hcc_core(
        fun,
        output_path,
        best_individual,
        max_fes,
        grouping_result,
        info,
        adjacent_overlapping_elements,
        seed=seed,
        method=method,
        problem_code=problem_code,
        info_aware_config=info_aware_config,
        return_metadata=return_metadata,
    )


def parallel_optimization(
    fun_name,
    fun_id,
    output_path,
    best_individual,
    max_fes,
    cycle_num,
    grouping_result,
    info,
    output_data,
    adjacent_overlapping_elements,
    method=HCC_ES_METHOD,
    info_aware_config=None,
):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                optimization_task,
                fun_name,
                fun_id,
                output_path,
                best_individual,
                max_fes,
                grouping_result,
                info,
                adjacent_overlapping_elements,
                seed,
                method,
                None,
                info_aware_config,
            )
            for seed in range(1, int(cycle_num) + 1)
        ]
        algorithm = f"{fun_name}_{fun_id}"
        average_time = 0.0
        for future in futures:
            curve, runtime, _ = future.result()
            output_data[algorithm].append(curve)
            average_time += float(runtime)
        return average_time / float(cycle_num)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the original HCC experiments.")
    parser.add_argument(
        "--protocol",
        choices=protocol_choices(),
        default="smoke",
        help="Experiment preset controlling MaxFEs, cycle count, and evaluation checkpoints.",
    )
    parser.add_argument("--problems", nargs="*", help="Problem codes such as E4 E6 S4 S6 R6 A6.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Seed values for independent HCC runs.")
    parser.add_argument("--tfes", type=int, help="Total function evaluations per problem/seed run.")
    parser.add_argument("--method", nargs="+", choices=METHOD_CHOICES, default=[HCC_ES_METHOD], help="Only hcc_es_original is supported.")
    parser.add_argument("--workers", type=int, default=1, help="Number of independent problem/seed workers.")
    parser.add_argument("--record-fes", nargs="*", type=int, default=[], help="Evaluation checkpoints written as best_at_<FE> columns.")
    parser.add_argument("--enable-info-aware-nda", action="store_true", help="Enable the optional adaptive information-aware NDA warm-start.")
    parser.add_argument("--info-aware-nda-config", help="Optional JSON config for the adaptive information-aware NDA warm-start.")
    parser.add_argument(
        "--overlap-blend-mode",
        choices=("original", "equation8_correct", "safe_delta", "safe_conflict", "no_blend"),
        help="Overlap blend strategy used after each fixed RDDSM subspace update.",
    )
    parser.add_argument(
        "--summary-refresh-every",
        type=int,
        default=1,
        help="Rewrite summary.csv after this many completed runs.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in run_details.csv.")
    parser.add_argument("--output-dir", help="Output directory for problem-code batch runs.")
    return parser.parse_args()


def build_hcc_es_inputs(problem_code):
    fun_name, fun_id, normalized = parse_problem_code(problem_code)
    bench = Benchmark(None)
    file_path = f"HCC_SRC/AOB/AOBG/datafile/F{fun_id}-design.txt"
    design_matrix = np.loadtxt(file_path, delimiter=",")
    design_matrix_dimension = int(np.asarray(design_matrix).shape[0]) if np.asarray(design_matrix).ndim >= 1 else 0
    decomposition = Decomposition(design_matrix)
    raw_grouping_result = decomposition.decomposition()
    info = bench.get_info(fun_name, fun_id)
    benchmark_dimension = int(info.get("decision_dimension", info["dimension"]))
    grouping_result, coverage_audit = ensure_grouping_covers_benchmark_dimension(
        raw_grouping_result,
        benchmark_dimension,
        design_matrix_dimension,
    )
    adjacent_overlapping_elements = compute_adjacent_overlaps_for_groups(grouping_result)
    return {
        "problem": normalized,
        "fun_name": fun_name,
        "fun_id": fun_id,
        "grouping_result": grouping_result,
        "info": info,
        "best_individual": np.zeros(info["dimension"]),
        "adjacent_overlapping_elements": adjacent_overlapping_elements,
        "coverage_audit": coverage_audit,
    }


def run_problem_seed_task(problem_code, seed, tfes, method, output_dir, record_fes=None, info_aware_config=None, method_label=None):
    inputs = build_hcc_es_inputs(problem_code)
    run_output_path = str(Path(output_dir) / inputs["problem"] / f"seed-{seed}") + "/"
    record_method = str(method_label or resolve_record_method(method, info_aware_config))
    blend_strategy = resolve_blend_strategy_name(info_aware_config)
    curve, runtime, diagnostics, metadata = optimization_task(
        inputs["fun_name"],
        inputs["fun_id"],
        run_output_path,
        inputs["best_individual"],
        int(tfes),
        inputs["grouping_result"],
        inputs["info"],
        inputs["adjacent_overlapping_elements"],
        int(seed),
        method,
        inputs["problem"],
        info_aware_config,
        True,
    )
    if metadata.get("overlap_blend_rows"):
        overlap_trace_csv_path = Path(run_output_path) / "overlap_blend_trace.csv"
        save_overlap_blend_trace_csv(overlap_trace_csv_path, metadata["overlap_blend_rows"])
        if metadata.get("info_aware_diagnostics") is not None:
            metadata["info_aware_diagnostics"]["overlap_blend_trace_csv"] = overlap_trace_csv_path.as_posix()
    if metadata.get("shared_variable_proposal_rows"):
        proposal_trace_csv_path = Path(run_output_path) / "shared_variable_proposals.csv"
        save_shared_variable_proposal_trace_csv(proposal_trace_csv_path, metadata["shared_variable_proposal_rows"])
        if metadata.get("info_aware_diagnostics") is not None:
            metadata["info_aware_diagnostics"]["shared_variable_proposals_csv"] = proposal_trace_csv_path.as_posix()
    if metadata.get("shared_variable_fusion_rows"):
        fusion_trace_csv_path = Path(run_output_path) / "shared_variable_fusion.csv"
        save_shared_variable_fusion_trace_csv(fusion_trace_csv_path, metadata["shared_variable_fusion_rows"])
        if metadata.get("info_aware_diagnostics") is not None:
            metadata["info_aware_diagnostics"]["shared_variable_fusion_csv"] = fusion_trace_csv_path.as_posix()
    if metadata.get("validated_coordination_rows"):
        validated_trace_csv_path = Path(run_output_path) / "validated_coordination_trace.csv"
        save_validated_coordination_trace_csv(validated_trace_csv_path, metadata["validated_coordination_rows"])
        if metadata.get("info_aware_diagnostics") is not None:
            metadata["info_aware_diagnostics"]["validated_coordination_trace_csv"] = validated_trace_csv_path.as_posix()
    if metadata.get("coordination_selector_rows"):
        selector_trace_csv_path = Path(run_output_path) / "coordination_selector_trace.csv"
        save_coordination_selector_trace_csv(selector_trace_csv_path, metadata["coordination_selector_rows"])
        if metadata.get("info_aware_diagnostics") is not None:
            metadata["info_aware_diagnostics"]["coordination_selector_trace_csv"] = selector_trace_csv_path.as_posix()
    if metadata.get("optimizer_state_rows"):
        optimizer_state_trace_csv_path = Path(run_output_path) / "optimizer_state_trace.csv"
        save_optimizer_state_trace_csv(optimizer_state_trace_csv_path, metadata["optimizer_state_rows"])
        if metadata.get("info_aware_diagnostics") is not None:
            metadata["info_aware_diagnostics"]["optimizer_state_trace_csv"] = optimizer_state_trace_csv_path.as_posix()
    if metadata.get("info_aware_diagnostics") is not None:
        if metadata.get("group_trace_rows"):
            trace_csv_path = Path(run_output_path) / "group_priority_trace.csv"
            save_group_trace_csv(trace_csv_path, metadata["group_trace_rows"])
            metadata["info_aware_diagnostics"]["group_trace_csv"] = trace_csv_path.as_posix()
        diagnostics_path = Path(run_output_path) / (
            info_aware_config.diagnostics_filename if isinstance(info_aware_config, InfoAwareNDAConfig) else DEFAULT_INFO_AWARE_DIAGNOSTICS_ARTIFACT.name
        )
        save_info_aware_diagnostics(diagnostics_path, metadata["info_aware_diagnostics"])
    detail_row = build_run_detail_row(
        inputs["problem"],
        method,
        seed,
        curve,
        runtime,
        "ok",
        diagnostics,
        record_fes,
        tfes=tfes,
        blend_strategy=blend_strategy,
        method_label=record_method,
    )
    if metadata.get("final_selected_fitness") is not None and np.isfinite(_to_float(metadata.get("final_selected_fitness"))):
        detail_row["final_fitness"] = float(metadata["final_selected_fitness"])
    return {
        "detail": detail_row,
        "diagnostics": diagnostics,
        "metadata": metadata,
    }


def run_problem_code_batch(args):
    problems = [parse_problem_code(problem)[2] for problem in args.problems]
    seeds = args.seeds or [1, 2, 3]
    tfes = int(args.tfes or resolve_protocol(args.protocol)["max_fes"])
    methods = [canonicalize_method(method) for method in args.method]
    record_fes = list(args.record_fes)
    info_aware_config = override_info_aware_config(
        load_info_aware_nda_config(args.info_aware_nda_config, args.enable_info_aware_nda),
        overlap_blend_mode=args.overlap_blend_mode,
    )
    blend_strategy = resolve_blend_strategy_name(info_aware_config)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    method_slug = resolve_record_method(methods[0], info_aware_config)
    output_dir = Path(args.output_dir or f"HCC_SRC/result/hcc-es-baselines/{method_slug}-{'-'.join(problems)}-{len(seeds)}seeds-{tfes}-{timestamp}")
    summary_path = output_dir / "summary.csv"
    detail_path = output_dir / "run_details.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    summary_refresh_every = max(1, int(args.summary_refresh_every))
    existing_keys = read_summary_keys(detail_path) if args.resume else set()
    ensure_csv_header(diagnostics_path, DIAGNOSTIC_FIELDNAMES)
    tasks = [
        (problem, method, resolve_record_method(method, info_aware_config), seed)
        for method in methods
        for problem in problems
        for seed in seeds
        if (problem, resolve_record_method(method, info_aware_config), int(seed)) not in existing_keys
    ]

    print(
        f"Running methods={methods} with problems={problems}, seeds={seeds}, "
        f"tfes={tfes}, workers={args.workers}, output_dir={output_dir}"
    )
    if not tasks:
        write_aggregate_summary(detail_path, diagnostics_path, summary_path)
        print(f"summary.csv -> {summary_path}")
        return

    pending_summary_updates = 0

    def maybe_refresh_summary(force=False):
        nonlocal pending_summary_updates
        if pending_summary_updates == 0:
            if force and not summary_path.exists():
                write_aggregate_summary(detail_path, diagnostics_path, summary_path)
            return
        if not force and pending_summary_updates < summary_refresh_every:
            return
        write_aggregate_summary(detail_path, diagnostics_path, summary_path)
        pending_summary_updates = 0

    if args.workers <= 1:
        for problem, method, method_label, seed in tasks:
            try:
                result = run_problem_seed_task(problem, seed, tfes, method, output_dir, record_fes, info_aware_config, method_label)
            except Exception as exc:
                result = {
                    "detail": build_run_detail_row(
                        problem,
                        method,
                        seed,
                        [],
                        0.0,
                        f"error: {exc}",
                        [],
                        record_fes,
                        tfes=tfes,
                        blend_strategy=blend_strategy,
                        method_label=method_label,
                    ),
                    "diagnostics": [],
                    "metadata": {},
                }
            append_csv_row(detail_path, result["detail"], RUN_DETAIL_FIELDNAMES + checkpoint_fieldnames(record_fes))
            append_csv_rows(diagnostics_path, result["diagnostics"], DIAGNOSTIC_FIELDNAMES)
            pending_summary_updates += 1
            maybe_refresh_summary()
            print(f"finished {problem} {method_label} seed={seed}: {result['detail']['status']}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_problem_seed_task, problem, seed, tfes, method, output_dir, record_fes, info_aware_config, method_label): (problem, method, method_label, seed)
                for problem, method, method_label, seed in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                problem, method, method_label, seed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "detail": build_run_detail_row(
                            problem,
                            method,
                            seed,
                            [],
                            0.0,
                            f"error: {exc}",
                            [],
                            record_fes,
                            tfes=tfes,
                            blend_strategy=blend_strategy,
                            method_label=method_label,
                        ),
                        "diagnostics": [],
                        "metadata": {},
                    }
                append_csv_row(detail_path, result["detail"], RUN_DETAIL_FIELDNAMES + checkpoint_fieldnames(record_fes))
                append_csv_rows(diagnostics_path, result["diagnostics"], DIAGNOSTIC_FIELDNAMES)
                pending_summary_updates += 1
                maybe_refresh_summary()
                print(f"finished {problem} {method_label} seed={seed}: {result['detail']['status']}")
    maybe_refresh_summary(force=True)
    print(f"summary.csv -> {summary_path}")


def run_protocol_batch(args):
    protocol = resolve_protocol(args.protocol)
    methods = [canonicalize_method(method) for method in (args.method or [HCC_ES_METHOD])]
    info_aware_config = override_info_aware_config(
        load_info_aware_nda_config(args.info_aware_nda_config, args.enable_info_aware_nda),
        overlap_blend_mode=args.overlap_blend_mode,
    )
    if len(methods) != 1:
        raise ValueError("Protocol batch supports exactly one method.")
    method = methods[0]
    max_fes = int(protocol["max_fes"])
    cycle_num = int(protocol["cycle_num"])
    record_fes = list(protocol["record_fes"])
    selected_problem_codes = list(DEFAULT_PROBLEM_CODES)
    selected_problem_groups = group_problem_codes(selected_problem_codes)

    output_data_map = {
        fun_name: {f"{fun_name}_{idx}": [] for idx in fun_ids}
        for fun_name, fun_ids in selected_problem_groups.items()
    }
    for prefix in output_data_map.values():
        for idx in [int(key.rsplit("_", 1)[1]) for key in prefix]:
            key = next(name for name in prefix if name.endswith(f"_{idx}"))
            prefix[f"{key}_time"] = []

    print(
        f"Running protocol '{protocol['name']}' with "
        f"MaxFEs={max_fes}, cycle_num={cycle_num}, record_FEs={record_fes}, "
        f"problems={selected_problem_codes}"
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for fun_name, fun_ids in selected_problem_groups.items():
        output_path = f"HCC_SRC/result/{timestamp}/{fun_name}/"
        bench = Benchmark(output_path)
        output_data = output_data_map[fun_name]
        for fun_id in fun_ids:
            file_path = f"HCC_SRC/AOB/AOBG/datafile/F{fun_id}-design.txt"
            design_matrix = np.loadtxt(file_path, delimiter=",")
            decomposition = Decomposition(design_matrix)
            grouping_result = decomposition.decomposition()
            adjacent_overlapping_elements = compute_adjacent_overlaps_for_groups(grouping_result)
            info = bench.get_info(fun_name, fun_id)
            fun = bench.get_function(fun_name, fun_id)
            best_individual = np.zeros(info["dimension"])
            _ = fun(best_individual)[0].copy()

            average_time = parallel_optimization(
                fun_name,
                fun_id,
                output_path,
                best_individual,
                max_fes,
                cycle_num,
                grouping_result,
                info,
                output_data,
                adjacent_overlapping_elements,
                method,
                info_aware_config,
            )
            print(f"{fun_name}_{fun_id} average time: {average_time}")
            output_data[f"{fun_name}_{fun_id}_time"].append(average_time)

        evaluation_record(output_data, output_path, record_FEs_list=record_fes)
        plot_evaluation_curve(output_data, output_path, font_size=12, log_scale=True)
        plot_evaluation_curve_best_so_far(output_data, output_path, font_size=12, log_scale=True, show_variance=True)


def main():
    args = parse_args()
    if args.problems:
        run_problem_code_batch(args)
    else:
        run_protocol_batch(args)


if __name__ == "__main__":
    freeze_support()
    main()
