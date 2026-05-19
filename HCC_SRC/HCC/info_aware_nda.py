import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from HCC.NDAs.MMES.es import ES
from HCC.NDAs.MMES.mmes import MMES


EPSILON = 1e-12


@dataclass
class InfoAwareNDAConfig:
    enable: bool = False

    # early switch
    enable_early_switch: bool = True
    min_nda_fe_ratio: float = 0.05
    max_nda_fe_ratio: Optional[float] = None
    window_size: int = 10
    patience: int = 3
    eps_improve: float = 1e-4
    eps_center_shift: float = 1e-3
    eps_diversity: Optional[float] = None

    # trajectory distillation
    enable_trajectory_distill: bool = True
    max_history_size: int = 200
    normalize_by_bounds: bool = True
    min_success_gain: float = 0.0

    # CC prior
    enable_group_priority: bool = True
    priority_alpha_conflict: float = 0.5
    priority_beta_overlap: float = 0.2
    priority_mode: str = "diagnostic_only"
    budget_min_fe: int = 1
    enable_group_delta_trace: bool = True
    priority_audit_topk: int = 3

    # diagnostics
    save_diagnostics: bool = True
    diagnostics_filename: str = "info_aware_nda_diagnostics.json"

    def normalized(self) -> "InfoAwareNDAConfig":
        config = InfoAwareNDAConfig(**asdict(self))
        config.min_nda_fe_ratio = float(np.clip(config.min_nda_fe_ratio, 0.0, 1.0))
        if config.max_nda_fe_ratio is not None:
            config.max_nda_fe_ratio = float(np.clip(config.max_nda_fe_ratio, 0.0, 1.0))
        config.window_size = max(1, int(config.window_size))
        config.patience = max(1, int(config.patience))
        config.max_history_size = max(1, int(config.max_history_size))
        config.priority_mode = str(config.priority_mode or "off").strip().lower()
        if config.priority_mode not in {"off", "diagnostic_only", "sort_dangerous_ablation", "budget"}:
            config.priority_mode = "off"
        config.budget_min_fe = max(1, int(config.budget_min_fe))
        config.priority_audit_topk = max(1, int(config.priority_audit_topk))
        if config.max_nda_fe_ratio is not None and config.max_nda_fe_ratio < config.min_nda_fe_ratio:
            config.max_nda_fe_ratio = config.min_nda_fe_ratio
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.normalized())


@dataclass
class NDAInfo:
    fe_used: int = 0
    early_switch_triggered: bool = False
    early_switch_reason: str = ""

    best_history: list = field(default_factory=list)
    mean_x_history: list = field(default_factory=list)
    diversity_history: list = field(default_factory=list)
    center_shift_history: list = field(default_factory=list)
    improvement_rate_history: list = field(default_factory=list)

    success_steps: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    success_gains: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))

    var_contribution: Optional[np.ndarray] = None
    var_stability: Optional[np.ndarray] = None
    var_direction: Optional[np.ndarray] = None

    best_before_nda: float = float("nan")
    best_after_nda: float = float("nan")
    warnings: list = field(default_factory=list)


@dataclass
class CCPrior:
    group_priority: np.ndarray
    variable_contribution: np.ndarray
    variable_stability: np.ndarray
    variable_direction: np.ndarray
    conflict_prior: np.ndarray
    overlap_degree: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _ensure_1d_array(values, ndim, fill_value):
    if values is None:
        return np.full(int(ndim), float(fill_value), dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == int(ndim):
        return array
    if array.size == 1:
        return np.full(int(ndim), float(array[0]), dtype=float)
    raise ValueError(f"Expected array size {ndim}, got {array.size}.")


def _safe_ranges(lower_boundary, upper_boundary, ndim):
    lower = _ensure_1d_array(lower_boundary, ndim, 0.0)
    upper = _ensure_1d_array(upper_boundary, ndim, 1.0)
    ranges = upper - lower
    return np.where(np.abs(ranges) <= EPSILON, 1.0, ranges)


def _trim_list_inplace(values, max_size):
    if len(values) > max_size:
        del values[: len(values) - max_size]


def _finite_mean(values, default):
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _safe_normalize_mean(values, default_value=1.0):
    array = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.full_like(array, float(default_value), dtype=float)
    mean_value = np.mean(array[finite_mask])
    if abs(float(mean_value)) <= EPSILON:
        return np.full_like(array, float(default_value), dtype=float)
    normalized = array / float(mean_value)
    normalized[~finite_mask] = float(default_value)
    return normalized


def resolve_nda_budget(total_max_fes, original_max_nda_fes, config: InfoAwareNDAConfig):
    total_max_fes = max(0, int(total_max_fes))
    original_max_nda_fes = max(0, int(original_max_nda_fes))
    normalized_config = config.normalized()
    min_nda_fes = int(math.floor(normalized_config.min_nda_fe_ratio * total_max_fes))
    if normalized_config.max_nda_fe_ratio is None:
        max_nda_fes = original_max_nda_fes
    else:
        max_nda_fes = int(math.floor(normalized_config.max_nda_fe_ratio * total_max_fes))
        if original_max_nda_fes > 0:
            max_nda_fes = min(max_nda_fes, original_max_nda_fes)
    max_nda_fes = max(min_nda_fes, min(total_max_fes, max_nda_fes))
    return int(min_nda_fes), int(max_nda_fes)


def compute_population_diversity(population, lower_boundary, upper_boundary, normalize_by_bounds=True):
    if population is None:
        return float("nan")
    population = np.asarray(population, dtype=float)
    if population.ndim != 2 or population.size == 0:
        return float("nan")
    feature_std = np.std(population, axis=0)
    if normalize_by_bounds:
        ranges = np.abs(_safe_ranges(lower_boundary, upper_boundary, population.shape[1]))
        feature_std = feature_std / ranges
    return float(np.mean(feature_std))


def compute_center_shift(previous_mean, current_mean, lower_boundary, upper_boundary, normalize_by_bounds=True):
    previous_mean = np.asarray(previous_mean, dtype=float).reshape(-1)
    current_mean = np.asarray(current_mean, dtype=float).reshape(-1)
    if previous_mean.size == 0 or current_mean.size == 0:
        return 0.0
    delta = current_mean - previous_mean
    if normalize_by_bounds:
        delta = delta / _safe_ranges(lower_boundary, upper_boundary, current_mean.size)
    return float(np.linalg.norm(delta) / max(1.0, math.sqrt(current_mean.size)))


def compute_window_improvement(best_history, window_size, eps=EPSILON):
    if len(best_history) <= 1:
        return 1.0
    window_size = max(1, int(window_size))
    end_value = float(best_history[-1])
    start_index = max(0, len(best_history) - 1 - window_size)
    start_value = float(best_history[start_index])
    denominator = abs(start_value) + float(eps)
    return float((start_value - end_value) / denominator)


def evaluate_early_switch(
    best_history,
    center_shift_history,
    diversity_history,
    fe_used,
    total_max_fes,
    original_max_nda_fes,
    config: InfoAwareNDAConfig,
    stagnant_counter=0,
):
    normalized_config = config.normalized()
    min_nda_fes, max_nda_fes = resolve_nda_budget(total_max_fes, original_max_nda_fes, normalized_config)
    improvement_rate = compute_window_improvement(best_history, normalized_config.window_size)
    center_shift = float(center_shift_history[-1]) if center_shift_history else 0.0
    diversity = float(diversity_history[-1]) if diversity_history else float("nan")

    if int(fe_used) < min_nda_fes:
        return {
            "should_stop": False,
            "reason": "",
            "stagnant_counter": int(stagnant_counter),
            "improvement_rate": float(improvement_rate),
            "center_shift": float(center_shift),
            "diversity": float(diversity),
            "min_nda_fes": int(min_nda_fes),
            "max_nda_fes": int(max_nda_fes),
        }

    if int(fe_used) >= max_nda_fes:
        return {
            "should_stop": True,
            "reason": "max_nda_fes_reached",
            "stagnant_counter": int(stagnant_counter),
            "improvement_rate": float(improvement_rate),
            "center_shift": float(center_shift),
            "diversity": float(diversity),
            "min_nda_fes": int(min_nda_fes),
            "max_nda_fes": int(max_nda_fes),
        }

    is_stagnant = float(improvement_rate) < float(normalized_config.eps_improve) and float(center_shift) < float(normalized_config.eps_center_shift)
    if normalized_config.eps_diversity is not None and np.isfinite(diversity):
        is_stagnant = is_stagnant and float(diversity) < float(normalized_config.eps_diversity)

    stagnant_counter = int(stagnant_counter) + 1 if is_stagnant else 0
    should_stop = stagnant_counter >= int(normalized_config.patience)
    reason = "early_stagnation" if should_stop else ""
    return {
        "should_stop": bool(should_stop),
        "reason": reason,
        "stagnant_counter": int(stagnant_counter),
        "improvement_rate": float(improvement_rate),
        "center_shift": float(center_shift),
        "diversity": float(diversity),
        "min_nda_fes": int(min_nda_fes),
        "max_nda_fes": int(max_nda_fes),
    }


def compute_variable_contribution(
    success_steps,
    success_gains,
    lower_boundary=None,
    upper_boundary=None,
    normalize_by_bounds=True,
    default_value=1.0,
):
    success_steps = np.asarray(success_steps, dtype=float)
    success_gains = np.asarray(success_gains, dtype=float).reshape(-1)
    if success_steps.ndim != 2 or success_steps.shape[0] == 0 or success_gains.size == 0:
        return np.full(success_steps.shape[1] if success_steps.ndim == 2 else 0, float(default_value), dtype=float)

    ndim = success_steps.shape[1]
    if normalize_by_bounds:
        ranges = np.abs(_safe_ranges(lower_boundary, upper_boundary, ndim))
    else:
        ranges = np.ones(ndim, dtype=float)

    weighted_steps = success_gains[:, None] * np.abs(success_steps) / ranges[None, :]
    contribution = np.sum(weighted_steps, axis=0)
    contribution = np.where(np.isfinite(contribution), contribution, 0.0)
    if contribution.size == 0 or np.all(contribution <= EPSILON):
        return np.full(ndim, float(default_value), dtype=float)
    return _safe_normalize_mean(contribution, default_value=default_value)


def compute_variable_stability(
    success_steps,
    success_gains,
    lower_boundary=None,
    upper_boundary=None,
    normalize_by_bounds=True,
    default_stability=1.0,
):
    success_steps = np.asarray(success_steps, dtype=float)
    success_gains = np.asarray(success_gains, dtype=float).reshape(-1)
    if success_steps.ndim != 2 or success_steps.shape[0] == 0 or success_gains.size == 0:
        ndim = success_steps.shape[1] if success_steps.ndim == 2 else 0
        return (
            np.full(ndim, float(default_stability), dtype=float),
            np.zeros(ndim, dtype=int),
        )

    ndim = success_steps.shape[1]
    if normalize_by_bounds:
        ranges = np.abs(_safe_ranges(lower_boundary, upper_boundary, ndim))
    else:
        ranges = np.ones(ndim, dtype=float)

    normalized_steps = success_steps / ranges[None, :]
    signed_sum = np.sum(success_gains[:, None] * normalized_steps, axis=0)
    abs_sum = np.sum(success_gains[:, None] * np.abs(normalized_steps), axis=0)
    stability = np.abs(signed_sum) / (abs_sum + EPSILON)
    stability = np.clip(np.where(np.isfinite(stability), stability, float(default_stability)), 0.0, 1.0)
    direction = np.sign(signed_sum)
    direction[np.abs(signed_sum) <= EPSILON] = 0.0
    return stability.astype(float), direction.astype(int)


def build_cc_prior(
    grouping_result,
    dimension,
    overlap_hypergraph,
    var_contribution,
    var_stability,
    var_direction,
    config: InfoAwareNDAConfig,
):
    normalized_config = config.normalized()
    dimension = int(dimension)
    var_contribution = np.asarray(var_contribution, dtype=float).reshape(-1)
    var_stability = np.asarray(var_stability, dtype=float).reshape(-1)
    var_direction = np.asarray(var_direction, dtype=int).reshape(-1)

    if var_contribution.size != dimension:
        var_contribution = np.full(dimension, 1.0, dtype=float)
    if var_stability.size != dimension:
        var_stability = np.full(dimension, 1.0, dtype=float)
    if var_direction.size != dimension:
        var_direction = np.zeros(dimension, dtype=int)

    var_to_groups = overlap_hypergraph.get("var_to_groups", {})
    overlap_degree = np.zeros(dimension, dtype=float)
    for var_id, group_ids in var_to_groups.items():
        if 0 <= int(var_id) < dimension:
            overlap_degree[int(var_id)] = float(len(group_ids))

    conflict_prior = np.where(overlap_degree > 1.0, 1.0 - np.clip(var_stability, 0.0, 1.0), 0.0)
    group_priority = np.ones(len(grouping_result), dtype=float)
    top_priority_groups = []
    overlap_ratios = []

    for group_id, group in enumerate(grouping_result):
        group = [int(var_id) for var_id in group]
        if not group:
            group_priority[group_id] = 1.0
            overlap_ratios.append(0.0)
            continue
        group_indices = np.asarray(group, dtype=int)
        overlap_mask = overlap_degree[group_indices] > 1.0
        overlap_vars = group_indices[overlap_mask]
        overlap_ratio = float(overlap_vars.size / max(1, len(group)))
        overlap_ratios.append(overlap_ratio)
        contribution_term = float(np.mean(var_contribution[group_indices]))
        conflict_term = float(np.mean(conflict_prior[overlap_vars])) if overlap_vars.size > 0 else 0.0
        priority_value = contribution_term
        priority_value += float(normalized_config.priority_alpha_conflict) * conflict_term
        priority_value += float(normalized_config.priority_beta_overlap) * overlap_ratio
        group_priority[group_id] = float(priority_value)

    group_priority = _safe_normalize_mean(group_priority, default_value=1.0)
    sorted_group_ids = list(np.argsort(-group_priority, kind="stable"))
    for group_id in sorted_group_ids[:5]:
        top_priority_groups.append(
            {
                "group_id": int(group_id),
                "priority": float(group_priority[group_id]),
                "group_size": int(len(grouping_result[group_id])),
                "overlap_ratio": float(overlap_ratios[group_id]),
            }
        )

    diagnostics = {
        "group_priority_min": float(np.min(group_priority)) if group_priority.size else 0.0,
        "group_priority_max": float(np.max(group_priority)) if group_priority.size else 0.0,
        "group_priority_mean": float(np.mean(group_priority)) if group_priority.size else 0.0,
        "group_priority_std": float(np.std(group_priority)) if group_priority.size else 0.0,
        "top_priority_groups": top_priority_groups,
    }

    return CCPrior(
        group_priority=group_priority.astype(float),
        variable_contribution=var_contribution.astype(float),
        variable_stability=np.clip(var_stability.astype(float), 0.0, 1.0),
        variable_direction=var_direction.astype(int),
        conflict_prior=conflict_prior.astype(float),
        overlap_degree=overlap_degree.astype(float),
        diagnostics=diagnostics,
    )


def build_priority_order(group_priority):
    priorities = np.asarray(group_priority, dtype=float).reshape(-1)
    if priorities.size == 0:
        return []
    return [int(index) for index in np.argsort(-priorities, kind="stable")]


def allocate_priority_budgets(remaining_fes, group_priority, min_budget=1):
    remaining_fes = max(0, int(remaining_fes))
    priorities = np.asarray(group_priority, dtype=float).reshape(-1)
    group_count = priorities.size
    if group_count == 0 or remaining_fes == 0:
        return []
    min_budget = max(1, int(min_budget))
    if remaining_fes <= min_budget * group_count:
        budgets = np.zeros(group_count, dtype=int)
        budgets[:remaining_fes] = 1
        return budgets.tolist()

    safe_priorities = np.where(np.isfinite(priorities) & (priorities > 0.0), priorities, 1.0)
    normalized = safe_priorities / np.sum(safe_priorities)
    budgets = np.full(group_count, min_budget, dtype=int)
    leftover = int(remaining_fes - np.sum(budgets))
    if leftover <= 0:
        return budgets.tolist()

    raw_extra = leftover * normalized
    extra = np.floor(raw_extra).astype(int)
    budgets += extra
    remainder = int(leftover - np.sum(extra))
    if remainder > 0:
        fractional_order = np.argsort(-(raw_extra - extra), kind="stable")
        for index in fractional_order[:remainder]:
            budgets[int(index)] += 1
    return budgets.tolist()


def merge_warnings(*warning_lists):
    merged = []
    seen = set()
    for warning_list in warning_lists:
        for warning in warning_list or []:
            warning_text = str(warning)
            if warning_text not in seen:
                seen.add(warning_text)
                merged.append(warning_text)
    return merged


def _json_safe_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    return value


def _json_safe_mapping(mapping):
    return {str(key): _json_safe_scalar(value) for key, value in dict(mapping).items()}


def _rankdata_stable(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    for rank, index in enumerate(order, start=1):
        ranks[int(index)] = float(rank)
    return ranks


def _pearson_correlation(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    if x_values.size == 0 or y_values.size == 0 or x_values.size != y_values.size:
        return None
    if x_values.size < 2:
        return None
    if np.std(x_values) <= EPSILON or np.std(y_values) <= EPSILON:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def compute_priority_audit(group_trace_rows, topk=3):
    rows = list(group_trace_rows or [])
    total_count = len(rows)
    warnings = []
    if total_count == 0:
        return {
            "priority_delta_spearman": None,
            "priority_delta_pearson": None,
            "topk_priority_hit_rate": None,
            "topk_priority_mean_delta": None,
            "bottomk_priority_mean_delta": None,
            "positive_delta_rate_topk": None,
            "positive_delta_rate_all": None,
            "group_trace_count": 0,
            "audit_warnings": ["no_group_trace_rows"],
        }

    valid_rows = []
    for row in rows:
        priority = row.get("priority")
        actual_delta = row.get("actual_delta")
        if priority is None or actual_delta is None:
            continue
        if not np.isfinite(float(priority)) or not np.isfinite(float(actual_delta)):
            continue
        valid_rows.append(
            {
                "priority": float(priority),
                "actual_delta": float(actual_delta),
                "original_group_id": int(row.get("original_group_id", 0)),
            }
        )

    if len(valid_rows) < total_count:
        warnings.append("filtered_non_finite_group_trace_rows")
    if len(valid_rows) < 3:
        warnings.append("insufficient_valid_groups_for_correlation")

    priorities = np.asarray([row["priority"] for row in valid_rows], dtype=float)
    deltas = np.asarray([row["actual_delta"] for row in valid_rows], dtype=float)

    if len(valid_rows) >= 3 and np.std(priorities) > EPSILON and np.std(deltas) > EPSILON:
        priority_delta_pearson = _pearson_correlation(priorities, deltas)
        priority_delta_spearman = _pearson_correlation(_rankdata_stable(priorities), _rankdata_stable(deltas))
    else:
        priority_delta_pearson = None
        priority_delta_spearman = None
        if len(valid_rows) >= 3:
            warnings.append("priority_or_delta_constant")

    if not valid_rows:
        return {
            "priority_delta_spearman": priority_delta_spearman,
            "priority_delta_pearson": priority_delta_pearson,
            "topk_priority_hit_rate": None,
            "topk_priority_mean_delta": None,
            "bottomk_priority_mean_delta": None,
            "positive_delta_rate_topk": None,
            "positive_delta_rate_all": None,
            "group_trace_count": total_count,
            "audit_warnings": warnings or ["no_valid_priority_delta_rows"],
        }

    topk = max(1, min(int(topk), len(valid_rows)))
    priority_order = list(np.argsort(-priorities, kind="stable"))
    delta_order = list(np.argsort(-deltas, kind="stable"))
    top_priority_set = {int(valid_rows[index]["original_group_id"]) for index in priority_order[:topk]}
    top_delta_set = {int(valid_rows[index]["original_group_id"]) for index in delta_order[:topk]}
    bottom_priority_indices = priority_order[-topk:]

    topk_priority_hit_rate = float(len(top_priority_set.intersection(top_delta_set)) / float(topk))
    topk_priority_mean_delta = float(np.mean(deltas[priority_order[:topk]]))
    bottomk_priority_mean_delta = float(np.mean(deltas[bottom_priority_indices]))
    positive_delta_rate_topk = float(np.mean(deltas[priority_order[:topk]] > 0.0))
    positive_delta_rate_all = float(np.mean(deltas > 0.0))

    return {
        "priority_delta_spearman": priority_delta_spearman,
        "priority_delta_pearson": priority_delta_pearson,
        "topk_priority_hit_rate": topk_priority_hit_rate,
        "topk_priority_mean_delta": topk_priority_mean_delta,
        "bottomk_priority_mean_delta": bottomk_priority_mean_delta,
        "positive_delta_rate_topk": positive_delta_rate_topk,
        "positive_delta_rate_all": positive_delta_rate_all,
        "group_trace_count": total_count,
        "audit_warnings": warnings,
    }


def build_info_aware_diagnostics_payload(
    config: InfoAwareNDAConfig,
    nda_info: Optional[NDAInfo],
    cc_prior: Optional[CCPrior],
    total_max_fes,
    execution_order=None,
    warnings=None,
    priority_mode_effective="off",
    sort_dangerous_ablation_changed_order=False,
    group_trace_rows=None,
    priority_audit=None,
):
    normalized_config = config.normalized()
    nda_info = nda_info or NDAInfo()
    warnings = merge_warnings(nda_info.warnings, warnings)
    contribution = nda_info.var_contribution if nda_info.var_contribution is not None else np.empty((0,), dtype=float)
    stability = nda_info.var_stability if nda_info.var_stability is not None else np.empty((0,), dtype=float)
    conflict_prior = cc_prior.conflict_prior if cc_prior is not None else np.empty((0,), dtype=float)
    group_priority = cc_prior.group_priority if cc_prior is not None else np.empty((0,), dtype=float)
    group_trace_rows = list(group_trace_rows or [])
    priority_audit = priority_audit or {}
    priority_audit_warnings = list(priority_audit.get("audit_warnings", []))
    group_trace_sample = [_json_safe_mapping(row) for row in group_trace_rows[:10]]

    payload = {
        "info_aware_nda_enabled": bool(normalized_config.enable),
        "config": normalized_config.to_dict(),
        "total_max_fes": int(total_max_fes),
        "nda_fe_used": int(nda_info.fe_used),
        "nda_fe_ratio": float(nda_info.fe_used / max(1, int(total_max_fes))),
        "early_switch_triggered": bool(nda_info.early_switch_triggered),
        "early_switch_reason": str(nda_info.early_switch_reason or ""),
        "best_before_nda": _json_safe_scalar(nda_info.best_before_nda),
        "best_after_nda": _json_safe_scalar(nda_info.best_after_nda),
        "improvement_rate_last": _json_safe_scalar(nda_info.improvement_rate_history[-1] if nda_info.improvement_rate_history else None),
        "center_shift_last": _json_safe_scalar(nda_info.center_shift_history[-1] if nda_info.center_shift_history else None),
        "diversity_last": _json_safe_scalar(nda_info.diversity_history[-1] if nda_info.diversity_history else None),
        "success_step_count": int(nda_info.success_gains.size if nda_info.success_gains is not None else 0),
        "var_contribution_mean": _json_safe_scalar(_finite_mean(contribution, 0.0)),
        "var_contribution_std": _json_safe_scalar(float(np.std(contribution)) if contribution.size else 0.0),
        "var_stability_mean": _json_safe_scalar(_finite_mean(stability, 0.0)),
        "var_stability_std": _json_safe_scalar(float(np.std(stability)) if stability.size else 0.0),
        "conflict_prior_mean": _json_safe_scalar(_finite_mean(conflict_prior, 0.0)),
        "conflict_prior_max": _json_safe_scalar(float(np.max(conflict_prior)) if conflict_prior.size else 0.0),
        "group_priority_mean": _json_safe_scalar(_finite_mean(group_priority, 0.0)),
        "group_priority_std": _json_safe_scalar(float(np.std(group_priority)) if group_priority.size else 0.0),
        "top_priority_groups": cc_prior.diagnostics.get("top_priority_groups", []) if cc_prior is not None else [],
        "priority_mode_effective": str(priority_mode_effective or "off"),
        "sort_dangerous_ablation_changed_order": bool(sort_dangerous_ablation_changed_order),
        "group_trace_count": int(len(group_trace_rows)),
        "priority_delta_spearman": _json_safe_scalar(priority_audit.get("priority_delta_spearman")),
        "priority_delta_pearson": _json_safe_scalar(priority_audit.get("priority_delta_pearson")),
        "topk_priority_hit_rate": _json_safe_scalar(priority_audit.get("topk_priority_hit_rate")),
        "topk_priority_mean_delta": _json_safe_scalar(priority_audit.get("topk_priority_mean_delta")),
        "bottomk_priority_mean_delta": _json_safe_scalar(priority_audit.get("bottomk_priority_mean_delta")),
        "positive_delta_rate_topk": _json_safe_scalar(priority_audit.get("positive_delta_rate_topk")),
        "positive_delta_rate_all": _json_safe_scalar(priority_audit.get("positive_delta_rate_all")),
        "group_trace_sample": group_trace_sample,
        "priority_audit_warnings": priority_audit_warnings,
        "execution_order": [int(group_id) for group_id in (execution_order or [])],
        "warnings": warnings,
    }
    if cc_prior is not None:
        payload["cc_prior"] = {
            "group_priority": [float(value) for value in np.asarray(cc_prior.group_priority, dtype=float).tolist()],
            "conflict_prior": [float(value) for value in np.asarray(cc_prior.conflict_prior, dtype=float).tolist()],
            "overlap_degree": [float(value) for value in np.asarray(cc_prior.overlap_degree, dtype=float).tolist()],
            "diagnostics": cc_prior.diagnostics,
        }
    return payload


def save_info_aware_diagnostics(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_info_aware_nda_config(config_path=None, enable=False):
    if config_path is None and not enable:
        return None
    config_data = {}
    if config_path is not None:
        with Path(config_path).open(encoding="utf-8") as handle:
            config_data = json.load(handle)
    config = InfoAwareNDAConfig(**config_data)
    if enable:
        config.enable = True
    return config.normalized()


class AdaptiveInfoAwareMMES(MMES):
    def __init__(self, problem, options, config: InfoAwareNDAConfig, total_max_fes, original_max_nda_fes):
        self.info_aware_config = config.normalized()
        self.total_max_fes = int(total_max_fes)
        self.original_max_nda_fes = int(original_max_nda_fes)
        self.min_nda_fes, self.max_nda_fes = resolve_nda_budget(total_max_fes, original_max_nda_fes, self.info_aware_config)
        super().__init__(problem, options)
        self.max_function_evaluations = min(int(self.max_function_evaluations), int(self.max_nda_fes))
        self._best_history = []
        self._mean_history = []
        self._diversity_history = []
        self._center_shift_history = []
        self._improvement_history = []
        self._success_steps = []
        self._success_gains = []
        self._previous_mean = None
        self._stagnant_counter = 0
        self._early_switch_triggered = False
        self._early_switch_reason = ""
        self._warnings = []
        self._best_before_nda = float("nan")

    def _record_success(self, previous_best_x, previous_best_y):
        if previous_best_x is None or self.best_so_far_x is None:
            return
        previous_best_y = float(previous_best_y)
        current_best_y = float(self.best_so_far_y)
        if not np.isfinite(previous_best_y) or not np.isfinite(current_best_y):
            return
        if current_best_y >= previous_best_y - EPSILON:
            return
        gain = float(previous_best_y - current_best_y)
        if gain <= float(self.info_aware_config.min_success_gain):
            return
        step = np.asarray(self.best_so_far_x, dtype=float) - np.asarray(previous_best_x, dtype=float)
        self._success_steps.append(step.astype(float))
        self._success_gains.append(float(gain))
        _trim_list_inplace(self._success_steps, self.info_aware_config.max_history_size)
        _trim_list_inplace(self._success_gains, self.info_aware_config.max_history_size)

    def _append_snapshot(self, mean, population=None):
        mean = np.asarray(mean, dtype=float).reshape(-1)
        best_value = float(self.best_so_far_y) if np.isfinite(self.best_so_far_y) else float("nan")
        self._best_history.append(best_value)
        self._mean_history.append(mean.copy())
        diversity = compute_population_diversity(
            population,
            self.lower_boundary,
            self.upper_boundary,
            normalize_by_bounds=self.info_aware_config.normalize_by_bounds,
        )
        self._diversity_history.append(float(diversity))

        if self._previous_mean is None:
            center_shift = 0.0
        else:
            center_shift = compute_center_shift(
                self._previous_mean,
                mean,
                self.lower_boundary,
                self.upper_boundary,
                normalize_by_bounds=self.info_aware_config.normalize_by_bounds,
            )
        self._center_shift_history.append(float(center_shift))
        self._improvement_history.append(
            float(compute_window_improvement(self._best_history, self.info_aware_config.window_size))
        )
        self._previous_mean = mean.copy()

        _trim_list_inplace(self._best_history, self.info_aware_config.max_history_size)
        _trim_list_inplace(self._mean_history, self.info_aware_config.max_history_size)
        _trim_list_inplace(self._diversity_history, self.info_aware_config.max_history_size)
        _trim_list_inplace(self._center_shift_history, self.info_aware_config.max_history_size)
        _trim_list_inplace(self._improvement_history, self.info_aware_config.max_history_size)

    def _build_nda_info(self):
        if self._success_steps:
            success_steps = np.vstack(self._success_steps).astype(float)
            success_gains = np.asarray(self._success_gains, dtype=float)
        else:
            success_steps = np.empty((0, self.ndim_problem), dtype=float)
            success_gains = np.empty((0,), dtype=float)

        if self.info_aware_config.enable_trajectory_distill:
            var_contribution = compute_variable_contribution(
                success_steps,
                success_gains,
                self.lower_boundary,
                self.upper_boundary,
                normalize_by_bounds=self.info_aware_config.normalize_by_bounds,
                default_value=1.0,
            )
            var_stability, var_direction = compute_variable_stability(
                success_steps,
                success_gains,
                self.lower_boundary,
                self.upper_boundary,
                normalize_by_bounds=self.info_aware_config.normalize_by_bounds,
                default_stability=1.0,
            )
            if success_gains.size == 0:
                self._warnings.append("no_success_steps_recorded")
        else:
            var_contribution = np.ones(self.ndim_problem, dtype=float)
            var_stability = np.ones(self.ndim_problem, dtype=float)
            var_direction = np.zeros(self.ndim_problem, dtype=int)
            self._warnings.append("trajectory_distillation_disabled")

        return NDAInfo(
            fe_used=int(self.n_function_evaluations),
            early_switch_triggered=bool(self._early_switch_triggered),
            early_switch_reason=str(self._early_switch_reason or ""),
            best_history=[float(value) for value in self._best_history],
            mean_x_history=[np.asarray(value, dtype=float).tolist() for value in self._mean_history],
            diversity_history=[float(value) if np.isfinite(value) else float("nan") for value in self._diversity_history],
            center_shift_history=[float(value) for value in self._center_shift_history],
            improvement_rate_history=[float(value) for value in self._improvement_history],
            success_steps=success_steps,
            success_gains=success_gains,
            var_contribution=var_contribution.astype(float),
            var_stability=np.clip(var_stability.astype(float), 0.0, 1.0),
            var_direction=var_direction.astype(int),
            best_before_nda=float(self._best_before_nda),
            best_after_nda=float(self.best_so_far_y),
            warnings=merge_warnings(self._warnings),
        )

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        self._best_before_nda = float(self.best_so_far_y)
        self._append_snapshot(mean, None)
        self._print_verbose_info(fitness, y[0])

        while not self.termination_signal:
            y_bak = np.copy(y)
            previous_best_x = None if self.best_so_far_x is None else np.asarray(self.best_so_far_x, dtype=float).copy()
            previous_best_y = float(self.best_so_far_y)
            x, y = self.iterate(x, mean, q, v, y, args)
            self._record_success(previous_best_x, previous_best_y)

            if self._check_terminations():
                self._append_snapshot(mean, x)
                if int(self.n_function_evaluations) >= int(self.max_nda_fes):
                    self._early_switch_reason = "max_nda_fes_reached"
                break

            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._append_snapshot(mean, x)
            self._print_verbose_info(fitness, y)

            if self.info_aware_config.enable_early_switch:
                switch_state = evaluate_early_switch(
                    self._best_history,
                    self._center_shift_history,
                    self._diversity_history,
                    self.n_function_evaluations,
                    self.total_max_fes,
                    self.original_max_nda_fes,
                    self.info_aware_config,
                    self._stagnant_counter,
                )
                self._stagnant_counter = int(switch_state["stagnant_counter"])
                if switch_state["should_stop"]:
                    self._early_switch_reason = str(switch_state["reason"])
                    self._early_switch_triggered = self._early_switch_reason == "early_stagnation"
                    break

            x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                args, x, mean, p, w, q, t, v, y, fitness
            )

        results = self._collect(fitness, y, mean)
        results["p"] = p
        results["w"] = w
        results["nda_info"] = self._build_nda_info()
        return results


def run_adaptive_info_aware_nda(problem, options, config: InfoAwareNDAConfig, total_max_fes, original_max_nda_fes):
    optimizer = AdaptiveInfoAwareMMES(problem, options, config, total_max_fes, original_max_nda_fes)
    return optimizer.optimize()
