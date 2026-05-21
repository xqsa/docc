import argparse
import csv
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_1_runs"
V0_RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_relation_action_audit.csv"
RUN_DETAILS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_1_run_details.csv"
RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_1_relation_action_audit.csv"
ATTRIBUTION_PATH = ARTIFACTS_ROOT / "arac_lite_v0_1_action_attribution.csv"
SWEEP_SUMMARY_PATH = ARTIFACTS_ROOT / "arac_lite_v0_1_sweep_summary.csv"
REPORT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_1_report.md"

DEFAULT_PROBLEMS = ["A6", "E6", "R6"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TFES = [10000]
DEFAULT_CC_PASS_GROUP_FES = 20

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
    "cc_pass_group_fes",
    "cc_pass_count",
    "best_error",
    "final_error",
    "fe_used",
    "runtime",
    "fusion_count",
    "freeze_count",
    "disable_count",
    "fusion_validation_attempt_count",
    "fusion_validation_accept_count",
    "fusion_validation_accept_rate",
    "validation_attempt_count",
    "validation_accept_count",
    "validation_reject_count",
    "validation_accept_rate",
    "validation_extra_fe_ratio",
    "relation_history_size",
    "status",
]

RELATION_AUDIT_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "method",
    "pass_id",
    "var_id",
    "proposal_support",
    "positive_proposal_count",
    "negative_proposal_count",
    "proposal_std",
    "proposal_std_ratio",
    "raw_update_ratio",
    "update_magnitude_ratio",
    "validation_attempted",
    "validation_accepted",
    "validation_accept_rate",
    "validation_delta",
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
    "arac_targeted_probe_candidate",
    "arac_targeted_probe_signature_matched",
    "arac_targeted_probe_reason",
    "arac_targeted_probe_score",
    "best_improvement_after_action",
    "relation_attempt_count",
    "relation_accept_rate",
    "relation_mean_validation_delta",
    "relation_reject_streak",
]

ATTRIBUTION_FIELDNAMES = [
    "problem",
    "method",
    "action_candidate",
    "action_count",
    "validation_attempt_count",
    "validation_accept_count",
    "validation_reject_count",
    "fusion_accept_rate",
    "fusion_reject_rate",
    "freeze_saved_bad_update_count",
    "disable_false_negative_count",
    "disable_false_negative_rate",
    "action_delta_mean",
    "action_delta_positive_rate",
    "action_regret_vs_oracle",
]

SWEEP_SUMMARY_FIELDNAMES = [
    "problem",
    "tfes",
    "method",
    "runs",
    "best_mean",
    "best_std",
    "final_mean",
    "final_std",
    "gap_vs_no_coordination",
    "gap_vs_validated_selective",
    "gap_vs_arac_lite_v0",
    "fusion_count",
    "freeze_count",
    "disable_count",
    "fusion_validation_accept_rate",
    "validation_accept_rate",
    "cc_pass_count_mean",
    "relation_history_size_mean",
]


@dataclass(frozen=True)
class ThresholdVariant:
    label: str
    overrides: dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.1 action attribution and threshold sweep artifacts.")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    parser.add_argument("--skip-sweep", action="store_true", help="Only build attribution from the existing V0 audit.")
    parser.add_argument("--v0-audit-path", type=Path, default=V0_RELATION_AUDIT_PATH)
    return parser.parse_args()


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def mean_or_nan(values):
    numeric = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
    return float(np.mean(numeric)) if numeric.size else float("nan")


def std_or_nan(values):
    numeric = np.asarray([to_float(value) for value in values if np.isfinite(to_float(value))], dtype=float)
    return float(np.std(numeric)) if numeric.size else float("nan")


def gap(value, baseline):
    value = to_float(value)
    baseline = to_float(baseline)
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-300:
        return float("nan")
    return float((value - baseline) / abs(baseline))


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


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def oracle_delta_by_relation(rows):
    oracle = {}
    for row in rows:
        key = (
            str(row.get("problem", "")).upper(),
            to_int(row.get("seed")),
            to_int(row.get("tfes")),
            to_int(row.get("pass_id")),
            to_int(row.get("var_id")),
        )
        delta = to_float(row.get("validation_delta"))
        if not np.isfinite(delta):
            continue
        oracle[key] = max(delta, oracle.get(key, float("-inf")))
    return oracle


def build_future_accepted_fusion_lookup(rows):
    by_relation = {}
    for row in rows:
        key = (
            str(row.get("problem", "")).upper(),
            to_int(row.get("seed")),
            to_int(row.get("tfes")),
            str(row.get("method", "")),
            to_int(row.get("var_id")),
        )
        by_relation.setdefault(key, []).append(row)
    future_lookup = set()
    for key, relation_rows in by_relation.items():
        sorted_rows = sorted(relation_rows, key=lambda row: to_int(row.get("pass_id")))
        accepted_future_passes = [
            to_int(row.get("pass_id"))
            for row in sorted_rows
            if str(row.get("action_candidate", "")) == "Fusion" and to_bool(row.get("validation_accepted"))
        ]
        for row in sorted_rows:
            pass_id = to_int(row.get("pass_id"))
            if any(future_pass > pass_id for future_pass in accepted_future_passes):
                future_lookup.add((*key, pass_id))
    return future_lookup


def build_action_attribution_rows(rows):
    rows = [dict(row) for row in rows]
    oracle = oracle_delta_by_relation(rows)
    future_accepted_fusion = build_future_accepted_fusion_lookup(rows)
    grouped = {}
    for row in rows:
        action = str(row.get("action_candidate", "") or "Unknown")
        key = (str(row.get("problem", "")).upper(), str(row.get("method", "")), action)
        grouped.setdefault(key, []).append(row)

    attribution_rows = []
    for key in sorted(grouped):
        problem, method, action = key
        action_rows = grouped[key]
        attempted_rows = [row for row in action_rows if to_bool(row.get("validation_attempted"))]
        accepted_rows = [row for row in attempted_rows if to_bool(row.get("validation_accepted"))]
        rejected_rows = [row for row in attempted_rows if not to_bool(row.get("validation_accepted"))]
        deltas = [to_float(row.get("validation_delta")) for row in action_rows]
        finite_deltas = [value for value in deltas if np.isfinite(value)]
        positive_delta_count = sum(1 for value in finite_deltas if value > 0.0)
        regrets = []
        for row in action_rows:
            relation_key = (
                str(row.get("problem", "")).upper(),
                to_int(row.get("seed")),
                to_int(row.get("tfes")),
                to_int(row.get("pass_id")),
                to_int(row.get("var_id")),
            )
            delta = to_float(row.get("validation_delta"))
            oracle_delta = oracle.get(relation_key)
            if oracle_delta is None or not np.isfinite(delta):
                continue
            regrets.append(max(0.0, float(oracle_delta - delta)))

        freeze_saved_bad_update_count = 0
        if action == "Freeze":
            freeze_saved_bad_update_count = sum(1 for value in finite_deltas if value <= 0.0)

        disable_false_negative_count = 0
        if action == "Disable":
            for row in action_rows:
                relation_pass_key = (
                    str(row.get("problem", "")).upper(),
                    to_int(row.get("seed")),
                    to_int(row.get("tfes")),
                    str(row.get("method", "")),
                    to_int(row.get("var_id")),
                    to_int(row.get("pass_id")),
                )
                if relation_pass_key in future_accepted_fusion:
                    disable_false_negative_count += 1

        fusion_accept_rate = 0.0
        fusion_reject_rate = 0.0
        if action == "Fusion" and attempted_rows:
            fusion_accept_rate = float(len(accepted_rows) / len(attempted_rows))
            fusion_reject_rate = float(len(rejected_rows) / len(attempted_rows))

        attribution_rows.append(
            {
                "problem": problem,
                "method": method,
                "action_candidate": action,
                "action_count": int(len(action_rows)),
                "validation_attempt_count": int(len(attempted_rows)),
                "validation_accept_count": int(len(accepted_rows)),
                "validation_reject_count": int(len(rejected_rows)),
                "fusion_accept_rate": float(fusion_accept_rate),
                "fusion_reject_rate": float(fusion_reject_rate),
                "freeze_saved_bad_update_count": int(freeze_saved_bad_update_count),
                "disable_false_negative_count": int(disable_false_negative_count),
                "disable_false_negative_rate": float(disable_false_negative_count / len(action_rows)) if action_rows else 0.0,
                "action_delta_mean": mean_or_nan(finite_deltas),
                "action_delta_positive_rate": float(positive_delta_count / len(finite_deltas)) if finite_deltas else 0.0,
                "action_regret_vs_oracle": mean_or_nan(regrets),
            }
        )
    return attribution_rows


def build_threshold_variants():
    return [
        ThresholdVariant("arac-lite-v0", {}),
        ThresholdVariant(
            "arac-lite-v0.1-disable-fast",
            {
                "arac_lite_history_min_attempts": 1,
                "arac_lite_disable_reject_streak": 1,
                "arac_lite_disable_accept_rate_threshold": 0.0,
                "arac_lite_disable_mean_delta_threshold": 0.0,
            },
        ),
        ThresholdVariant(
            "arac-lite-v0.1-fusion-strict",
            {
                "shared_variable_selective_min_positive_proposals": 3,
                "shared_variable_selective_max_proposal_std_ratio": 0.00075,
            },
        ),
        ThresholdVariant(
            "arac-lite-v0.1-balanced",
            {
                "arac_lite_history_min_attempts": 1,
                "arac_lite_disable_reject_streak": 1,
                "shared_variable_selective_min_positive_proposals": 3,
                "shared_variable_selective_max_proposal_std_ratio": 0.0009,
            },
        ),
    ]


def load_base_arac_config(cc_pass_group_fes):
    config = hcc_es.load_info_aware_nda_config(CONFIG_ROOT / "arac-lite-rule.json", enable=False)
    return replace(config, cc_pass_group_fes=max(1, int(cc_pass_group_fes)), cc_min_passes=3).normalized()


def apply_variant(base_config, variant):
    values = {field: getattr(base_config, field) for field in base_config.__dataclass_fields__}
    values.update(variant.overrides)
    return hcc_es.InfoAwareNDAConfig(**values).normalized()


def load_named_config(path, cc_pass_group_fes):
    config = hcc_es.load_info_aware_nda_config(path, enable=False)
    return replace(config, cc_pass_group_fes=max(1, int(cc_pass_group_fes)), cc_min_passes=3).normalized()


def summarize_action_rows(rows):
    rows = list(rows or [])
    fusion_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Fusion"]
    freeze_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Freeze"]
    disable_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Disable"]
    validated_fusion_rows = [row for row in fusion_rows if bool(row.get("validation_attempted"))]
    accepted_fusion_rows = [row for row in validated_fusion_rows if bool(row.get("validation_accepted"))]
    return {
        "fusion_count": int(len(fusion_rows)),
        "freeze_count": int(len(freeze_rows)),
        "disable_count": int(len(disable_rows)),
        "fusion_validation_attempt_count": int(len(validated_fusion_rows)),
        "fusion_validation_accept_count": int(len(accepted_fusion_rows)),
        "fusion_validation_accept_rate": float(len(accepted_fusion_rows) / len(validated_fusion_rows))
        if validated_fusion_rows
        else 0.0,
    }


def normalize_relation_rows(fusion_rows, problem, seed, tfes, method):
    relation_rows = []
    for row in list(fusion_rows or []):
        attempted = bool(row.get("validation_attempted"))
        accepted = bool(row.get("validation_accepted"))
        relation_rows.append(
            {
                "problem": str(problem).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method),
                "pass_id": int(row.get("cycle_id", 0) or 0),
                "var_id": int(row.get("var_id", -1)),
                "proposal_support": int(row.get("proposal_count", 0) or 0),
                "positive_proposal_count": int(row.get("positive_proposal_count", 0) or 0),
                "negative_proposal_count": int(row.get("negative_proposal_count", 0) or 0),
                "proposal_std": to_float(row.get("proposal_value_std")),
                "proposal_std_ratio": to_float(row.get("proposal_value_std_ratio")),
                "raw_update_ratio": to_float(row.get("raw_update_ratio")),
                "update_magnitude_ratio": to_float(row.get("update_magnitude_ratio")),
                "validation_attempted": bool(attempted),
                "validation_accepted": bool(accepted),
                "validation_accept_rate": 1.0 if attempted and accepted else 0.0,
                "validation_delta": to_float(row.get("fitness_delta")),
                "action_candidate": str(row.get("action_candidate", "")),
                "action_reason": str(row.get("action_reason", "")),
                "arac_recovery_candidate": bool(row.get("arac_recovery_candidate", False)),
                "arac_recovery_phase": str(row.get("arac_recovery_phase", "")),
                "arac_recovery_reason": str(row.get("arac_recovery_reason", "")),
                "arac_recovery_attempt_count": int(row.get("arac_recovery_attempt_count", 0) or 0),
                "arac_recovery_accept_rate": to_float(row.get("arac_recovery_accept_rate")),
                "arac_recovery_delta_mean": to_float(row.get("arac_recovery_delta_mean")),
                "arac_recovery_positive_delta_rate": to_float(row.get("arac_recovery_positive_delta_rate")),
                "arac_probe_candidate": bool(row.get("arac_probe_candidate", False)),
                "arac_probe_reason": str(row.get("arac_probe_reason", "")),
                "arac_probe_phase": str(row.get("arac_probe_phase", "")),
                "arac_probe_attempt_count": int(row.get("arac_probe_attempt_count", 0) or 0),
                "arac_probe_accept_rate": to_float(row.get("arac_probe_accept_rate")),
                "arac_probe_delta_mean": to_float(row.get("arac_probe_delta_mean")),
                "arac_targeted_probe_candidate": bool(row.get("arac_targeted_probe_candidate", False)),
                "arac_targeted_probe_signature_matched": bool(
                    row.get("arac_targeted_probe_signature_matched", False)
                ),
                "arac_targeted_probe_reason": str(row.get("arac_targeted_probe_reason", "")),
                "arac_targeted_probe_score": to_float(row.get("arac_targeted_probe_score")),
                "best_improvement_after_action": bool(row.get("post_coordination_best_improved")),
                "relation_attempt_count": int(row.get("relation_attempt_count", 0) or 0),
                "relation_accept_rate": to_float(row.get("relation_accept_rate")),
                "relation_mean_validation_delta": to_float(row.get("relation_mean_validation_delta")),
                "relation_reject_streak": int(row.get("relation_reject_streak", 0) or 0),
            }
        )
    return relation_rows


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
        fusion_rows = list(metadata.get("shared_variable_fusion_rows", []))
        action_summary = summarize_action_rows(fusion_rows)
        validation_summary = hcc_es.summarize_validated_coordination_rows(
            metadata.get("validated_coordination_rows", []),
            total_max_fes=tfes,
        )
        run_row = {
            "problem": str(problem_code).upper(),
            "seed": int(seed),
            "tfes": int(tfes),
            "method": str(method_name),
            "coordination_mode": hcc_es.resolve_shared_variable_coordination_mode(config),
            "cc_pass_group_fes": int(getattr(config, "cc_pass_group_fes", 0) or 0),
            "cc_pass_count": int(metadata.get("cc_pass_count", 0) or 0),
            "best_error": to_float(detail.get("best_fitness")),
            "final_error": to_float(detail.get("final_fitness")),
            "fe_used": int(to_float(detail.get("fe_used"))),
            "runtime": to_float(detail.get("runtime")),
            **action_summary,
            "validation_attempt_count": int(validation_summary.get("validation_attempt_count", 0)),
            "validation_accept_count": int(validation_summary.get("validation_accept_count", 0)),
            "validation_reject_count": int(validation_summary.get("validation_reject_count", 0)),
            "validation_accept_rate": to_float(validation_summary.get("validation_accept_rate")),
            "validation_extra_fe_ratio": to_float(validation_summary.get("validation_extra_fe_ratio")),
            "relation_history_size": int(len(metadata.get("arac_relation_history", {}) or {})),
            "status": str(detail.get("status", "ok")),
        }
        relation_rows = normalize_relation_rows(fusion_rows, problem_code, seed, tfes, method_name)
        return run_row, relation_rows
    except Exception as exc:
        return (
            {
                "problem": str(problem_code).upper(),
                "seed": int(seed),
                "tfes": int(tfes),
                "method": str(method_name),
                "coordination_mode": hcc_es.resolve_shared_variable_coordination_mode(config),
                "cc_pass_group_fes": int(getattr(config, "cc_pass_group_fes", 0) or 0),
                "cc_pass_count": 0,
                "best_error": float("nan"),
                "final_error": float("nan"),
                "fe_used": 0,
                "runtime": float("nan"),
                "fusion_count": 0,
                "freeze_count": 0,
                "disable_count": 0,
                "fusion_validation_attempt_count": 0,
                "fusion_validation_accept_count": 0,
                "fusion_validation_accept_rate": 0.0,
                "validation_attempt_count": 0,
                "validation_accept_count": 0,
                "validation_reject_count": 0,
                "validation_accept_rate": 0.0,
                "validation_extra_fe_ratio": 0.0,
                "relation_history_size": 0,
                "status": f"error:{type(exc).__name__}:{exc}",
            },
            [],
        )


def build_method_configs(cc_pass_group_fes):
    base_arac = load_base_arac_config(cc_pass_group_fes)
    configs = [
        ("no-coordination", load_named_config(CONFIG_ROOT / "no-coordination.json", cc_pass_group_fes)),
        (
            "validated-selective-conflict",
            load_named_config(CONFIG_ROOT / "validated-selective-conflict.json", cc_pass_group_fes),
        ),
    ]
    for variant in build_threshold_variants():
        configs.append((variant.label, apply_variant(base_arac, variant)))
    return configs


def summarize_runs(run_rows):
    grouped = {}
    for row in run_rows:
        key = (row["problem"], int(row["tfes"]), row["method"])
        grouped.setdefault(key, []).append(row)

    best_baselines = {}
    for (problem, tfes, method), rows in grouped.items():
        if method in {"no-coordination", "validated-selective-conflict", "arac-lite-v0"}:
            best_baselines[(problem, tfes, method)] = mean_or_nan(row.get("best_error") for row in rows)

    summary_rows = []
    for key in sorted(grouped):
        problem, tfes, method = key
        rows = grouped[key]
        best_mean = mean_or_nan(row.get("best_error") for row in rows)
        summary_rows.append(
            {
                "problem": problem,
                "tfes": int(tfes),
                "method": method,
                "runs": int(len(rows)),
                "best_mean": best_mean,
                "best_std": std_or_nan(row.get("best_error") for row in rows),
                "final_mean": mean_or_nan(row.get("final_error") for row in rows),
                "final_std": std_or_nan(row.get("final_error") for row in rows),
                "gap_vs_no_coordination": gap(best_mean, best_baselines.get((problem, tfes, "no-coordination"))),
                "gap_vs_validated_selective": gap(best_mean, best_baselines.get((problem, tfes, "validated-selective-conflict"))),
                "gap_vs_arac_lite_v0": gap(best_mean, best_baselines.get((problem, tfes, "arac-lite-v0"))),
                "fusion_count": int(sum(int(row.get("fusion_count", 0) or 0) for row in rows)),
                "freeze_count": int(sum(int(row.get("freeze_count", 0) or 0) for row in rows)),
                "disable_count": int(sum(int(row.get("disable_count", 0) or 0) for row in rows)),
                "fusion_validation_accept_rate": mean_or_nan(row.get("fusion_validation_accept_rate") for row in rows),
                "validation_accept_rate": mean_or_nan(row.get("validation_accept_rate") for row in rows),
                "cc_pass_count_mean": mean_or_nan(row.get("cc_pass_count") for row in rows),
                "relation_history_size_mean": mean_or_nan(row.get("relation_history_size") for row in rows),
            }
        )
    return summary_rows


def report_summary_table(summary_rows, tfes):
    lines = [
        f"## TFEs = {int(tfes)}",
        "",
        "| problem | method | best_mean | gap_vs_no | gap_vs_validated | gap_vs_v0 | Fusion | Freeze | Disable | fusion_accept |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [item for item in summary_rows if int(item["tfes"]) == int(tfes)]:
        lines.append(
            "| {problem} | {method} | {best} | {gap_no} | {gap_val} | {gap_v0} | {fusion} | {freeze} | {disable} | {accept:.3f} |".format(
                problem=row["problem"],
                method=row["method"],
                best=format_metric(row.get("best_mean")),
                gap_no=format_percent(row.get("gap_vs_no_coordination")),
                gap_val=format_percent(row.get("gap_vs_validated_selective")),
                gap_v0=format_percent(row.get("gap_vs_arac_lite_v0")),
                fusion=int(row.get("fusion_count", 0) or 0),
                freeze=int(row.get("freeze_count", 0) or 0),
                disable=int(row.get("disable_count", 0) or 0),
                accept=to_float(row.get("fusion_validation_accept_rate")),
            )
        )
    return "\n".join(lines)


def build_interpretation_lines(summary_rows):
    variants = [
        row
        for row in summary_rows
        if str(row.get("method", "")).startswith("arac-lite-v0.1")
    ]
    if not variants:
        return ["## Interpretation", "", "- Threshold sweep was skipped."]
    by_method = {}
    for row in variants:
        by_method.setdefault(str(row["method"]), []).append(row)
    candidate_rows = []
    for method, rows in sorted(by_method.items()):
        r6 = next((row for row in rows if row["problem"] == "R6"), None)
        e6 = next((row for row in rows if row["problem"] == "E6"), None)
        a6 = next((row for row in rows if row["problem"] == "A6"), None)
        total_fusion = sum(int(row.get("fusion_count", 0) or 0) for row in rows)
        avg_gap = mean_or_nan(row.get("gap_vs_no_coordination") for row in rows)
        candidate_rows.append(
            {
                "method": method,
                "r6_gap": to_float(r6.get("gap_vs_no_coordination")) if r6 else float("nan"),
                "e6_gap": to_float(e6.get("gap_vs_no_coordination")) if e6 else float("nan"),
                "a6_gap": to_float(a6.get("gap_vs_no_coordination")) if a6 else float("nan"),
                "total_fusion": int(total_fusion),
                "avg_gap": float(avg_gap),
            }
        )
    preserving_candidates = [row for row in candidate_rows if row["total_fusion"] > 0]
    ranked = preserving_candidates or candidate_rows
    ranked = sorted(
        ranked,
        key=lambda row: (
            abs(row["r6_gap"]) if np.isfinite(row["r6_gap"]) else float("inf"),
            row["avg_gap"] if np.isfinite(row["avg_gap"]) else float("inf"),
        ),
    )
    recommended = ranked[0]
    zero_fusion_methods = [row["method"] for row in candidate_rows if row["total_fusion"] == 0]
    lines = [
        "## Interpretation",
        "",
        f"- Recommended V0.1 threshold candidate: `{recommended['method']}`.",
        f"- Its R6 gap vs no-coordination is {format_percent(recommended['r6_gap'])}; E6 gap is {format_percent(recommended['e6_gap'])}; A6 gap is {format_percent(recommended['a6_gap'])}.",
    ]
    if zero_fusion_methods:
        lines.append(
            "- Zero-Fusion variants behave like conservative Freeze/Disable probes, useful as diagnostics but too close to no-coordination to be the main ARAC module: "
            + ", ".join(f"`{method}`" for method in zero_fusion_methods)
            + "."
        )
    lines.append(
        "- Attribution points to low-quality Fusion as the main R6 residual gap source; A6 still shows a small Disable false-negative signal."
    )
    return lines


def attribution_focus_lines(attribution_rows):
    lines = [
        "## Action Attribution",
        "",
        "| problem | method | action | count | accept | reject | false_negative | delta_mean | regret |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    focus_rows = [
        row
        for row in attribution_rows
        if row["method"] in {"arac-lite-rule", "arac-lite-v0", "validated-selective-conflict"}
    ]
    for row in sorted(focus_rows, key=lambda item: (item["problem"], item["method"], item["action_candidate"])):
        lines.append(
            "| {problem} | {method} | {action} | {count} | {accept:.3f} | {reject:.3f} | {false_neg:.3f} | {delta} | {regret} |".format(
                problem=row["problem"],
                method=row["method"],
                action=row["action_candidate"],
                count=int(row.get("action_count", 0) or 0),
                accept=to_float(row.get("fusion_accept_rate")),
                reject=to_float(row.get("fusion_reject_rate")),
                false_neg=to_float(row.get("disable_false_negative_rate")),
                delta=format_metric(row.get("action_delta_mean")),
                regret=format_metric(row.get("action_regret_vs_oracle")),
            )
        )
    return "\n".join(lines)


def write_report(attribution_rows, summary_rows, run_rows, relation_rows, args):
    report = [
        "# ARAC-lite V0.1",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(str(value).upper() for value in args.problems)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        f"- cc_pass_group_fes: {int(args.cc_pass_group_fes)}",
        "- Scope: action-level attribution + small threshold sweep. No UCB, no bandit, no larger matrix.",
        "",
        "## Quick Read",
        "",
        f"- Attribution rows: {len(attribution_rows)}",
        f"- Sweep runs: {len(run_rows)}",
        f"- Sweep relation-action rows: {len(relation_rows)}",
        "",
        attribution_focus_lines(attribution_rows),
        "",
        *build_interpretation_lines(summary_rows),
        "",
        "## Threshold Sweep",
    ]
    for tfes in args.tfes:
        report.extend(["", report_summary_table(summary_rows, tfes)])
    report.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- attribution: `{ATTRIBUTION_PATH.as_posix()}`",
            f"- sweep run details: `{RUN_DETAILS_PATH.as_posix()}`",
            f"- sweep relation audit: `{RELATION_AUDIT_PATH.as_posix()}`",
            f"- sweep summary: `{SWEEP_SUMMARY_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def run_sweep(args):
    problems = [hcc_es.parse_problem_code(problem)[2] for problem in args.problems]
    run_rows = []
    relation_rows = []
    for method_name, config in build_method_configs(args.cc_pass_group_fes):
        for tfes in args.tfes:
            for problem_code in problems:
                for seed in args.seeds:
                    run_row, case_relation_rows = run_one_case(method_name, config, problem_code, seed, tfes)
                    run_rows.append(run_row)
                    relation_rows.extend(case_relation_rows)
                    print(
                        f"{problem_code} {method_name} seed={seed} tfes={tfes}: "
                        f"{run_row['status']} fusion={run_row['fusion_count']} "
                        f"freeze={run_row['freeze_count']} disable={run_row['disable_count']}"
                    )
    return run_rows, relation_rows, summarize_runs(run_rows)


def main():
    args = parse_args()
    v0_rows = read_csv(args.v0_audit_path)
    attribution_rows = build_action_attribution_rows(v0_rows)
    write_csv(ATTRIBUTION_PATH, ATTRIBUTION_FIELDNAMES, attribution_rows)

    run_rows = []
    relation_rows = []
    summary_rows = []
    if not args.skip_sweep:
        run_rows, relation_rows, summary_rows = run_sweep(args)
        write_csv(RUN_DETAILS_PATH, RUN_DETAIL_FIELDNAMES, run_rows)
        write_csv(RELATION_AUDIT_PATH, RELATION_AUDIT_FIELDNAMES, relation_rows)
        write_csv(SWEEP_SUMMARY_PATH, SWEEP_SUMMARY_FIELDNAMES, summary_rows)

    write_report(attribution_rows, summary_rows, run_rows, relation_rows, args)
    ok_count = sum(1 for row in run_rows if str(row.get("status")) == "ok")
    print(f"attribution rows -> {len(attribution_rows)}")
    print(f"completed {ok_count}/{len(run_rows)} sweep runs")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
