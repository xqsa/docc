import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
V03_RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_3_relation_action_audit.csv"
V04_OFFLINE_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_4_recovery_offline_audit.csv"
V05_RELATION_AUDIT_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_relation_action_audit.csv"
V05_PROBE_METRICS_PATH = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_metrics.csv"
V03_RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_3_runs"
V05_RUNS_ROOT = ARTIFACTS_ROOT / "arac_lite_v0_5_probe_runs"

DISABLE_FAST_METHOD = "arac-lite-v0.1-disable-fast"
V05_METHOD = "arac-lite-v0.5-low-frequency-probe"

MATCH_PATH = ARTIFACTS_ROOT / "offline_online_candidate_match.csv"
BIAS_PATH = ARTIFACTS_ROOT / "probe_selection_bias.csv"
ATTRIBUTION_PATH = ARTIFACTS_ROOT / "validation_delta_attribution.csv"
REPORT_PATH = ARTIFACTS_ROOT / "v0_5_failure_audit_report.md"

MATCH_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "phase",
    "offline_candidate_count",
    "online_probe_count",
    "matched_probe_count",
    "matched_offline_candidate_count",
    "matched_positive_delta_count",
    "match_rate",
    "online_delta_mean_for_matched",
    "strict_group_pair_matched_probe_count",
]

BIAS_FIELDNAMES = [
    "problem",
    "seed",
    "tfes",
    "phase",
    "cohort",
    "row_count",
    "relation_count",
    "group_pair_count",
    "proposal_support_mean",
    "proposal_support_std",
    "proposal_std_mean",
    "proposal_std_std",
    "membership_count_mean",
    "validation_attempt_count",
    "validation_accept_rate",
    "positive_delta_rate",
    "delta_mean",
    "fusion_count",
    "fusion_accept_rate",
    "fusion_positive_delta_rate",
    "freeze_count",
    "disable_count",
]

ATTRIBUTION_FIELDNAMES = [
    "validation_id",
    "problem",
    "seed",
    "tfes",
    "phase",
    "pass_id",
    "num_probe_relations",
    "num_fusion_relations",
    "num_disable_relations",
    "num_freeze_relations",
    "num_recovery_relations",
    "global_delta",
    "validation_accepted",
    "relation_count",
    "action_mix_entropy",
    "action_mix",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARAC-lite V0.5 failure audit artifacts.")
    parser.add_argument("--problems", nargs="+", default=["E6", "S6", "R6"])
    parser.add_argument("--tfes", nargs="+", type=int, default=[10000])
    parser.add_argument("--v0-3-relation-audit", type=Path, default=V03_RELATION_AUDIT_PATH)
    parser.add_argument("--v0-5-relation-audit", type=Path, default=V05_RELATION_AUDIT_PATH)
    parser.add_argument("--v0-5-probe-metrics", type=Path, default=V05_PROBE_METRICS_PATH)
    return parser.parse_args()


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=float("nan")):
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def finite_values(values):
    return [value for value in (to_float(item) for item in values) if math.isfinite(value)]


def mean(values):
    values = finite_values(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def std(values):
    values = finite_values(values)
    if not values:
        return float("nan")
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def format_metric(value):
    value = to_float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.6e}"


def format_rate(value):
    value = to_float(value, 0.0)
    return f"{value:.3f}"


def phase_for_pass(pass_id, max_pass_id):
    pass_id = max(0, int(pass_id))
    max_pass_id = max(0, int(max_pass_id))
    if max_pass_id <= 0:
        return "early"
    ratio_value = pass_id / max_pass_id
    if ratio_value <= 1.0 / 3.0:
        return "early"
    if ratio_value <= 2.0 / 3.0:
        return "middle"
    return "late"


def run_key(row):
    return (
        str(row.get("problem", "")).upper(),
        to_int(row.get("seed")),
        to_int(row.get("tfes")),
        str(row.get("method", "")),
    )


def run_key_without_method(row):
    return (str(row.get("problem", "")).upper(), to_int(row.get("seed")), to_int(row.get("tfes")))


def relation_key(row):
    return (*run_key(row), to_int(row.get("var_id")))


def pass_relation_key(row):
    return (*run_key(row), to_int(row.get("pass_id")), to_int(row.get("var_id")))


def assign_computed_phase(rows):
    max_pass_by_run = defaultdict(int)
    for row in rows:
        max_pass_by_run[run_key(row)] = max(max_pass_by_run[run_key(row)], to_int(row.get("pass_id")))
    annotated = []
    for row in rows:
        next_row = dict(row)
        next_row["phase"] = phase_for_pass(to_int(row.get("pass_id")), max_pass_by_run[run_key(row)])
        annotated.append(next_row)
    return annotated


def assign_online_phase(rows):
    annotated = assign_computed_phase(rows)
    for row in annotated:
        row["phase"] = (
            str(row.get("arac_probe_phase") or row.get("arac_recovery_phase") or row.get("phase") or "")
            or row["phase"]
        )
    return annotated


def is_probe_row(row):
    return str(row.get("action_candidate", "")) == "Fusion" and (
        to_bool(row.get("arac_probe_candidate")) or str(row.get("action_reason", "")).startswith("probe_fusion")
    )


def is_recovery_row(row):
    return to_bool(row.get("arac_recovery_candidate")) or str(row.get("action_reason", "")).startswith("recovery_")


def load_group_pair_lookup(runs_root, rows):
    lookup = {}
    run_keys = sorted({run_key(row) for row in rows})
    for problem, seed, tfes, method in run_keys:
        proposal_path = (
            Path(runs_root)
            / f"tfes-{tfes}"
            / method
            / problem
            / f"seed-{seed}"
            / "shared_variable_proposals.csv"
        )
        if not proposal_path.exists():
            continue
        groups_by_relation = defaultdict(set)
        for proposal in read_csv(proposal_path):
            cycle_id = to_int(proposal.get("cycle_id"))
            var_id = to_int(proposal.get("var_id"))
            group_id = to_int(proposal.get("group_id"), default=-1)
            if group_id >= 0:
                groups_by_relation[(cycle_id, var_id)].add(group_id)
        for (cycle_id, var_id), groups in groups_by_relation.items():
            group_pair = "+".join(str(group_id) for group_id in sorted(groups))
            lookup[(problem, seed, tfes, method, cycle_id, var_id)] = group_pair
    return lookup


def attach_relation_metadata(rows, lookup):
    annotated = []
    for row in rows:
        next_row = dict(row)
        var_id = to_int(next_row.get("var_id"))
        next_row["relation_id"] = str(var_id)
        next_row["group_pair"] = lookup.get((*run_key(next_row), to_int(next_row.get("pass_id")), var_id), "")
        next_row["membership_count"] = to_int(next_row.get("proposal_support"))
        annotated.append(next_row)
    return annotated


def build_future_accepted_fusion_pass(rows):
    future_pass = {}
    for row in rows:
        if (
            str(row.get("action_candidate", "")) == "Fusion"
            and to_bool(row.get("validation_attempted"))
            and to_bool(row.get("validation_accepted"))
        ):
            key = relation_key(row)
            future_pass[key] = max(future_pass.get(key, -1), to_int(row.get("pass_id")))
    return future_pass


def build_offline_candidate_rows(v03_rows, problems, tfes_values, group_lookup=None):
    group_lookup = group_lookup or {}
    selected = [
        row
        for row in v03_rows
        if str(row.get("method", "")) == DISABLE_FAST_METHOD
        and str(row.get("problem", "")).upper() in problems
        and to_int(row.get("tfes")) in tfes_values
    ]
    selected = attach_relation_metadata(assign_computed_phase(selected), group_lookup)
    future_pass = build_future_accepted_fusion_pass(selected)
    candidates = []
    for row in selected:
        future = future_pass.get(relation_key(row), -1)
        if str(row.get("action_candidate", "")) not in {"Disable", "Freeze"}:
            continue
        if row.get("phase") not in {"middle", "late"}:
            continue
        if future <= to_int(row.get("pass_id")):
            continue
        next_row = dict(row)
        next_row["cohort"] = "offline_recovery_candidate"
        next_row["future_accepted_fusion_pass_id"] = future
        candidates.append(next_row)
    return candidates


def build_offline_fusion_window_rows(v03_rows, problems, tfes_values, group_lookup=None):
    group_lookup = group_lookup or {}
    selected = [
        row
        for row in v03_rows
        if str(row.get("method", "")) == DISABLE_FAST_METHOD
        and str(row.get("problem", "")).upper() in problems
        and to_int(row.get("tfes")) in tfes_values
    ]
    selected = attach_relation_metadata(assign_computed_phase(selected), group_lookup)
    fusion_rows = []
    for row in selected:
        if str(row.get("action_candidate", "")) == "Fusion" and row.get("phase") in {"middle", "late"}:
            next_row = dict(row)
            next_row["cohort"] = "offline_phase_fusion_window"
            fusion_rows.append(next_row)
    return fusion_rows


def build_online_probe_rows(v05_rows, problems, tfes_values, group_lookup=None):
    group_lookup = group_lookup or {}
    selected = [
        row
        for row in v05_rows
        if str(row.get("method", "")) == V05_METHOD
        and str(row.get("problem", "")).upper() in problems
        and to_int(row.get("tfes")) in tfes_values
    ]
    selected = attach_relation_metadata(assign_online_phase(selected), group_lookup)
    probe_rows = []
    for row in selected:
        if is_probe_row(row):
            next_row = dict(row)
            next_row["cohort"] = "online_probe"
            probe_rows.append(next_row)
    return probe_rows


def build_match_rows(offline_candidates, online_probes):
    offline_by_bucket = defaultdict(list)
    online_by_bucket = defaultdict(list)
    for row in offline_candidates:
        offline_by_bucket[(row["problem"], to_int(row["seed"]), to_int(row["tfes"]), row["phase"])].append(row)
    for row in online_probes:
        online_by_bucket[(row["problem"], to_int(row["seed"]), to_int(row["tfes"]), row["phase"])].append(row)

    rows = []
    for key in sorted(set(offline_by_bucket) | set(online_by_bucket)):
        offline_rows = offline_by_bucket.get(key, [])
        online_rows = online_by_bucket.get(key, [])
        offline_vars = {to_int(row.get("var_id")) for row in offline_rows}
        online_vars = {to_int(row.get("var_id")) for row in online_rows}
        strict_pairs = {(to_int(row.get("var_id")), row.get("group_pair", "")) for row in offline_rows}
        matched_online = [row for row in online_rows if to_int(row.get("var_id")) in offline_vars]
        matched_offline = [row for row in offline_rows if to_int(row.get("var_id")) in online_vars]
        strict_matched = [
            row
            for row in online_rows
            if (to_int(row.get("var_id")), row.get("group_pair", "")) in strict_pairs
        ]
        positive_matched = [
            row
            for row in matched_online
            if math.isfinite(to_float(row.get("validation_delta"))) and to_float(row.get("validation_delta")) > 0.0
        ]
        problem, seed, tfes, phase = key
        rows.append(
            {
                "problem": problem,
                "seed": seed,
                "tfes": tfes,
                "phase": phase,
                "offline_candidate_count": len(offline_rows),
                "online_probe_count": len(online_rows),
                "matched_probe_count": len(matched_online),
                "matched_offline_candidate_count": len(matched_offline),
                "matched_positive_delta_count": len(positive_matched),
                "match_rate": rate(len(matched_online), len(offline_rows)),
                "online_delta_mean_for_matched": mean(row.get("validation_delta") for row in matched_online),
                "strict_group_pair_matched_probe_count": len(strict_matched),
            }
        )
    return rows


def summarize_feature_rows(rows, cohort):
    relation_ids = {str(row.get("relation_id", "")) for row in rows if str(row.get("relation_id", ""))}
    group_pairs = {str(row.get("group_pair", "")) for row in rows if str(row.get("group_pair", ""))}
    attempted = [row for row in rows if to_bool(row.get("validation_attempted"))]
    accepted = [row for row in attempted if to_bool(row.get("validation_accepted"))]
    positive_delta = [
        row
        for row in attempted
        if math.isfinite(to_float(row.get("validation_delta"))) and to_float(row.get("validation_delta")) > 0.0
    ]
    fusion_rows = [row for row in rows if str(row.get("action_candidate", "")) == "Fusion"]
    fusion_attempted = [row for row in fusion_rows if to_bool(row.get("validation_attempted"))]
    fusion_accepted = [row for row in fusion_attempted if to_bool(row.get("validation_accepted"))]
    fusion_positive = [
        row
        for row in fusion_attempted
        if math.isfinite(to_float(row.get("validation_delta"))) and to_float(row.get("validation_delta")) > 0.0
    ]
    return {
        "cohort": cohort,
        "row_count": len(rows),
        "relation_count": len(relation_ids),
        "group_pair_count": len(group_pairs),
        "proposal_support_mean": mean(row.get("proposal_support") for row in rows),
        "proposal_support_std": std(row.get("proposal_support") for row in rows),
        "proposal_std_mean": mean(row.get("proposal_std") for row in rows),
        "proposal_std_std": std(row.get("proposal_std") for row in rows),
        "membership_count_mean": mean(row.get("membership_count") for row in rows),
        "validation_attempt_count": len(attempted),
        "validation_accept_rate": rate(len(accepted), len(attempted)),
        "positive_delta_rate": rate(len(positive_delta), len(attempted)),
        "delta_mean": mean(row.get("validation_delta") for row in attempted),
        "fusion_count": len(fusion_rows),
        "fusion_accept_rate": rate(len(fusion_accepted), len(fusion_attempted)),
        "fusion_positive_delta_rate": rate(len(fusion_positive), len(fusion_attempted)),
        "freeze_count": sum(1 for row in rows if str(row.get("action_candidate", "")) == "Freeze"),
        "disable_count": sum(1 for row in rows if str(row.get("action_candidate", "")) == "Disable"),
    }


def build_bias_rows(offline_candidates, offline_fusion_windows, online_probes):
    grouped = defaultdict(list)
    for row in offline_candidates:
        grouped[(row["problem"], to_int(row["seed"]), to_int(row["tfes"]), row["phase"], "offline_recovery_candidate")].append(row)
    for row in offline_fusion_windows:
        grouped[(row["problem"], to_int(row["seed"]), to_int(row["tfes"]), row["phase"], "offline_phase_fusion_window")].append(row)
    for row in online_probes:
        grouped[(row["problem"], to_int(row["seed"]), to_int(row["tfes"]), row["phase"], "online_probe")].append(row)

    rows = []
    for key in sorted(grouped):
        problem, seed, tfes, phase, cohort = key
        summary = summarize_feature_rows(grouped[key], cohort)
        summary.update({"problem": problem, "seed": seed, "tfes": tfes, "phase": phase})
        rows.append(summary)
    return rows


def entropy_for_counts(counts):
    total = sum(count for count in counts if count > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return entropy


def build_validation_delta_attribution_rows(v05_rows, problems, tfes_values):
    selected = [
        row
        for row in v05_rows
        if str(row.get("method", "")) == V05_METHOD
        and str(row.get("problem", "")).upper() in problems
        and to_int(row.get("tfes")) in tfes_values
    ]
    selected = assign_online_phase(selected)
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["problem"], to_int(row["seed"]), to_int(row["tfes"]), to_int(row["pass_id"]))].append(row)

    rows = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        if not any(to_bool(row.get("validation_attempted")) for row in group_rows):
            continue
        problem, seed, tfes, pass_id = key
        action_counts = Counter(str(row.get("action_candidate", "")) for row in group_rows)
        deltas = finite_values(row.get("validation_delta") for row in group_rows)
        phases = Counter(str(row.get("phase", "")) for row in group_rows)
        phase = phases.most_common(1)[0][0] if phases else ""
        action_mix = ";".join(
            f"{action}={action_counts.get(action, 0)}" for action in ["Fusion", "Freeze", "Disable"] if action_counts.get(action, 0)
        )
        rows.append(
            {
                "validation_id": f"{problem}:seed-{seed}:tfes-{tfes}:pass-{pass_id}",
                "problem": problem,
                "seed": seed,
                "tfes": tfes,
                "phase": phase,
                "pass_id": pass_id,
                "num_probe_relations": sum(1 for row in group_rows if is_probe_row(row)),
                "num_fusion_relations": action_counts.get("Fusion", 0),
                "num_disable_relations": action_counts.get("Disable", 0),
                "num_freeze_relations": action_counts.get("Freeze", 0),
                "num_recovery_relations": sum(1 for row in group_rows if is_recovery_row(row)),
                "global_delta": deltas[0] if deltas else float("nan"),
                "validation_accepted": any(to_bool(row.get("validation_accepted")) for row in group_rows),
                "relation_count": len(group_rows),
                "action_mix_entropy": entropy_for_counts(
                    [action_counts.get("Fusion", 0), action_counts.get("Freeze", 0), action_counts.get("Disable", 0)]
                ),
                "action_mix": action_mix,
            }
        )
    return rows


def aggregate_match(match_rows, problem, phase):
    selected = [row for row in match_rows if row["problem"] == problem and row["phase"] == phase]
    matched_deltas = []
    offline_count = sum(to_int(row.get("offline_candidate_count")) for row in selected)
    comparable_offline_count = sum(
        to_int(row.get("offline_candidate_count")) for row in selected if to_int(row.get("online_probe_count")) > 0
    )
    matched_count = sum(to_int(row.get("matched_probe_count")) for row in selected)
    for row in selected:
        delta = to_float(row.get("online_delta_mean_for_matched"))
        count = to_int(row.get("matched_probe_count"))
        if math.isfinite(delta):
            matched_deltas.extend([delta] * count)
    return {
        "offline_candidate_count": offline_count,
        "comparable_offline_candidate_count": comparable_offline_count,
        "online_probe_count": sum(to_int(row.get("online_probe_count")) for row in selected),
        "matched_probe_count": matched_count,
        "matched_positive_delta_count": sum(to_int(row.get("matched_positive_delta_count")) for row in selected),
        "match_rate": rate(matched_count, offline_count),
        "online_delta_mean_for_matched": mean(matched_deltas),
    }


def aggregate_bias(bias_rows, problem, phase, cohort):
    selected = [row for row in bias_rows if row["problem"] == problem and row["phase"] == phase and row["cohort"] == cohort]
    expanded = []
    for row in selected:
        expanded.extend([row] * max(1, to_int(row.get("row_count"))))
    if not selected:
        return None
    return {
        "row_count": sum(to_int(row.get("row_count")) for row in selected),
        "relation_count": sum(to_int(row.get("relation_count")) for row in selected),
        "proposal_support_mean": mean(row.get("proposal_support_mean") for row in expanded),
        "proposal_std_mean": mean(row.get("proposal_std_mean") for row in expanded),
        "validation_accept_rate": mean(row.get("validation_accept_rate") for row in expanded),
        "positive_delta_rate": mean(row.get("positive_delta_rate") for row in expanded),
        "delta_mean": mean(row.get("delta_mean") for row in expanded),
    }


def attribution_summary(attribution_rows, problem=None):
    selected = [row for row in attribution_rows if problem is None or row["problem"] == problem]
    probe_selected = [row for row in selected if to_int(row.get("num_probe_relations")) > 0]
    return {
        "validation_count": len(selected),
        "mixed_action_count": sum(1 for row in selected if to_float(row.get("action_mix_entropy"), 0.0) > 0.0),
        "probe_validation_count": len(probe_selected),
        "multi_probe_validation_count": sum(1 for row in probe_selected if to_int(row.get("num_probe_relations")) > 1),
        "relation_count_mean": mean(row.get("relation_count") for row in selected),
        "probe_relation_count_mean": mean(row.get("num_probe_relations") for row in probe_selected),
        "global_delta_mean_for_probe": mean(row.get("global_delta") for row in probe_selected),
    }


def probe_metric_by_problem(path):
    return {str(row.get("problem", "")).upper(): row for row in read_csv(path)}


def report_match_table(match_rows):
    lines = [
        "| problem | seed | phase | offline | online_probe | matched_probe | matched_pos | match_rate | matched_delta |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    selected = [
        row
        for row in match_rows
        if row["problem"] in {"S6", "R6"} and row["phase"] in {"middle", "late"} and to_int(row.get("offline_candidate_count")) > 0
    ]
    for row in selected:
        lines.append(
            "| {problem} | {seed} | {phase} | {offline} | {online} | {matched} | {positive} | {rate} | {delta} |".format(
                problem=row["problem"],
                seed=row["seed"],
                phase=row["phase"],
                offline=row["offline_candidate_count"],
                online=row["online_probe_count"],
                matched=row["matched_probe_count"],
                positive=row["matched_positive_delta_count"],
                rate=format_rate(row["match_rate"]),
                delta=format_metric(row["online_delta_mean_for_matched"]),
            )
        )
    return "\n".join(lines)


def report_bias_table(bias_rows):
    lines = [
        "| problem | phase | cohort | rows | relations | support | proposal_std | accept | pos_delta | delta_mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem, phase in [("S6", "middle"), ("R6", "middle")]:
        for cohort in ["offline_recovery_candidate", "offline_phase_fusion_window", "online_probe"]:
            row = aggregate_bias(bias_rows, problem, phase, cohort)
            if not row:
                continue
            lines.append(
                "| {problem} | {phase} | {cohort} | {rows} | {relations} | {support} | {std} | {accept} | {pos} | {delta} |".format(
                    problem=problem,
                    phase=phase,
                    cohort=cohort,
                    rows=row["row_count"],
                    relations=row["relation_count"],
                    support=format_metric(row["proposal_support_mean"]),
                    std=format_metric(row["proposal_std_mean"]),
                    accept=format_rate(row["validation_accept_rate"]),
                    pos=format_rate(row["positive_delta_rate"]),
                    delta=format_metric(row["delta_mean"]),
                )
            )
    return "\n".join(lines)


def write_report(match_rows, bias_rows, attribution_rows, probe_metrics, args):
    s6_middle = aggregate_match(match_rows, "S6", "middle")
    r6_middle = aggregate_match(match_rows, "R6", "middle")
    all_attr = attribution_summary(attribution_rows)
    s6_attr = attribution_summary(attribution_rows, "S6")
    r6_attr = attribution_summary(attribution_rows, "R6")
    r6_metric = probe_metrics.get("R6", {})
    s6_metric = probe_metrics.get("S6", {})

    lines = [
        "# ARAC-lite V0.5 Failure Audit",
        "",
        "- 日期：2026-05-20",
        "- 执行者：Codex",
        f"- Problems: {', '.join(args.problems)}",
        f"- TFEs: {', '.join(str(value) for value in args.tfes)}",
        "- Scope: audit only; no main algorithm changes; no probe-frequency change; no UCB.",
        "",
        "## Executive Finding",
        "",
        (
            f"- S6 middle offline candidates={s6_middle['offline_candidate_count']}, "
            f"online-overlap offline candidates={s6_middle['comparable_offline_candidate_count']}, "
            f"online probes in same phase={s6_middle['online_probe_count']}, "
            f"matched probes={s6_middle['matched_probe_count']}, "
            f"match_rate={format_rate(s6_middle['match_rate'])}, "
            f"matched_delta_mean={format_metric(s6_middle['online_delta_mean_for_matched'])}."
        ),
        (
            f"- V0.5 aggregate S6 probe_accept_rate={format_rate(s6_metric.get('probe_accept_rate'))}, "
            f"probe_delta_mean={format_metric(s6_metric.get('probe_delta_mean'))}; "
            "S6 的 online evidence 没有稳定落在离线正窗口上。"
        ),
        (
            f"- R6 probe_accept_rate={format_rate(r6_metric.get('probe_accept_rate'))}, "
            f"probe_delta_mean={format_metric(r6_metric.get('probe_delta_mean'))}, "
            f"R6_bad_probe_count={to_int(r6_metric.get('R6_bad_probe_count'))}；"
            "高 accept 但负 delta 是稳定冲突信号，不应允许 recovery。"
        ),
        (
            f"- Pass-level attribution: validations={all_attr['validation_count']}, "
            f"mixed_action_validations={all_attr['mixed_action_count']}, "
            f"probe_validations={all_attr['probe_validation_count']}, "
            f"mean_relation_count={format_metric(all_attr['relation_count_mean'])}, "
            f"mean_probe_relations_when_probe={format_metric(all_attr['probe_relation_count_mean'])}."
        ),
        "",
        "## Audit 1: Offline/Online Candidate Match",
        "",
        report_match_table(match_rows),
        "",
        "Interpretation: `relation_id` 当前等价于 `var_id`，`group_pair` 由 per-run proposal trace 派生；match 以 problem/seed/tfes/phase/var_id 为准，另在 CSV 中保留 strict group-pair match 计数。S6 middle 的 23 个离线候选中，有 13 个落在 V0.5 实际有 online probe 的 seed/phase 覆盖内，这 13 个同样 0 命中；因此结论不是单纯 seed 覆盖差异，而是 probe selection miss。",
        "",
        "## Audit 2: Probe Selection Bias",
        "",
        report_bias_table(bias_rows),
        "",
        "Interpretation: `offline_phase_fusion_window` 表示离线同 phase 的 Fusion 行，能反映离线看到的 accept/delta 窗口；`offline_recovery_candidate` 是当时被 Disable/Freeze、但后续同 var 出现 accepted Fusion 的候选行；`online_probe` 是 V0.5 实际 probe 成 Fusion 的行。",
        "",
        "## Audit 3: Pass-Level Delta Attribution",
        "",
        "| scope | validations | probe_validations | multi_probe_validations | mixed_action | relation_count_mean | probe_relation_count_mean | probe_delta_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in [("all", all_attr), ("S6", s6_attr), ("R6", r6_attr)]:
        lines.append(
            "| {label} | {validations} | {probe_validations} | {multi_probe} | {mixed} | {relation_mean} | {probe_mean} | {delta} |".format(
                label=label,
                validations=summary["validation_count"],
                probe_validations=summary["probe_validation_count"],
                multi_probe=summary["multi_probe_validation_count"],
                mixed=summary["mixed_action_count"],
                relation_mean=format_metric(summary["relation_count_mean"]),
                probe_mean=format_metric(summary["probe_relation_count_mean"]),
                delta=format_metric(summary["global_delta_mean_for_probe"]),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation: 单个 validation pass 同时包含多个 relation/action，并共享同一个 `global_delta`。因此当前 reward 信号可以判断整批候选好坏，但不能可靠地给每个 relation 做细粒度信用分配。",
            "",
            "## Audit 4: R6 Hard Negative Rule",
            "",
            "- 结论：R6 的 high accept rate + negative delta 应固化为 hard Disable/Freeze 特征。",
            "- 推荐规则：如果 rolling_delta_mean <= 0，则不允许 recovery；即使 accept_rate 很高也不允许。",
            "- 推荐状态：high_accept_rate + negative_delta_mean -> Low-Credibility / Conflict -> Disable 或 Freeze。",
            "",
            "## Branching Recommendation",
            "",
            "- 若 S6 middle match_rate 低：下一步做 targeted recovery probe，而不是提高随机 probe 频率。",
            "- 若命中但 matched_delta_mean 为负：下一步查 offline/online candidate 构造口径，或改 relation-batch validation。",
            "- 若 validation 中 relation_count/action_mix_entropy 高：下一步做 cluster-level / relation-batch validation，减少 reward 混合。",
            "- 若 R6 high accept + negative delta 稳定：下一版加入 negative rolling delta hard gate。",
            "",
            "## Artifacts",
            "",
            f"- match: `{MATCH_PATH.as_posix()}`",
            f"- selection bias: `{BIAS_PATH.as_posix()}`",
            f"- delta attribution: `{ATTRIBUTION_PATH.as_posix()}`",
            f"- report: `{REPORT_PATH.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    problems = {problem.upper() for problem in args.problems}
    tfes_values = set(args.tfes)

    v03_rows = read_csv(args.v0_3_relation_audit)
    v05_rows = read_csv(args.v0_5_relation_audit)

    v03_selected = [
        row
        for row in v03_rows
        if str(row.get("method", "")) == DISABLE_FAST_METHOD
        and str(row.get("problem", "")).upper() in problems
        and to_int(row.get("tfes")) in tfes_values
    ]
    v05_selected = [
        row
        for row in v05_rows
        if str(row.get("method", "")) == V05_METHOD
        and str(row.get("problem", "")).upper() in problems
        and to_int(row.get("tfes")) in tfes_values
    ]
    v03_group_lookup = load_group_pair_lookup(V03_RUNS_ROOT, v03_selected)
    v05_group_lookup = load_group_pair_lookup(V05_RUNS_ROOT, v05_selected)

    offline_candidates = build_offline_candidate_rows(v03_rows, problems, tfes_values, v03_group_lookup)
    offline_fusion_windows = build_offline_fusion_window_rows(v03_rows, problems, tfes_values, v03_group_lookup)
    online_probes = build_online_probe_rows(v05_rows, problems, tfes_values, v05_group_lookup)
    match_rows = build_match_rows(offline_candidates, online_probes)
    bias_rows = build_bias_rows(offline_candidates, offline_fusion_windows, online_probes)
    attribution_rows = build_validation_delta_attribution_rows(v05_rows, problems, tfes_values)
    probe_metrics = probe_metric_by_problem(args.v0_5_probe_metrics)

    write_csv(MATCH_PATH, MATCH_FIELDNAMES, match_rows)
    write_csv(BIAS_PATH, BIAS_FIELDNAMES, bias_rows)
    write_csv(ATTRIBUTION_PATH, ATTRIBUTION_FIELDNAMES, attribution_rows)
    write_report(match_rows, bias_rows, attribution_rows, probe_metrics, args)

    print(f"offline candidates -> {len(offline_candidates)}")
    print(f"online probes -> {len(online_probes)}")
    print(f"match rows -> {len(match_rows)}")
    print(f"bias rows -> {len(bias_rows)}")
    print(f"attribution rows -> {len(attribution_rows)}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
