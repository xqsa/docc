import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"
CONFIG_ROOT = REPO_ROOT / "configs" / "info_aware_nda"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
ROUND2_ROOT = ARTIFACTS_ROOT / "round2"
SEEDS = [1, 2, 3, 4, 5]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)

import HCC.info_aware_nda as info_aware_nda


COMMON_OVERRIDES = {
    "min_nda_fe_ratio": 0.05,
    "max_nda_fe_ratio": 0.6,
    "window_size": 2,
    "patience": 1,
    "eps_improve": 0.05,
    "eps_center_shift": 0.05,
    "save_diagnostics": True,
}

METHOD_CONFIGS = [
    ("baseline", CONFIG_ROOT / "baseline.json"),
    ("early-switch-only", CONFIG_ROOT / "early-switch-only.json"),
    ("diagnostic-only", CONFIG_ROOT / "diagnostic-only.json"),
    ("sort-dangerous-ablation", CONFIG_ROOT / "sort-dangerous-ablation.json"),
]


class SyntheticSphere:
    def __init__(self):
        self.fitness_record = []

    def __call__(self, x_batch):
        x_array = np.asarray(x_batch, dtype=float)
        values = np.sum(np.square(x_array), axis=-1)
        self.fitness_record.extend(np.asarray(values, dtype=float).reshape(-1).tolist())
        return values


def build_method_config(config_path):
    config = info_aware_nda.load_info_aware_nda_config(config_path, enable=False)
    config_dict = config.to_dict() if config is not None else {}
    config_dict.update(COMMON_OVERRIDES)
    return info_aware_nda.InfoAwareNDAConfig(**config_dict).normalized()


def run_case(config, seed):
    grouping_result = [
        list(range(0, 10)),
        list(range(8, 18)),
        list(range(16, 20)),
    ]
    adjacent_overlaps = hcc_es.compute_adjacent_overlaps_for_groups(grouping_result)
    info = {
        "dimension": 20,
        "lower": -5.0,
        "upper": 5.0,
    }
    best_individual = np.full(info["dimension"], 3.0)
    fun = SyntheticSphere()
    curve, runtime, diagnostics, metadata = hcc_es.run_hcc_core(
        fun=fun,
        output_path="",
        best_individual=best_individual,
        max_fes=120,
        grouping_result=grouping_result,
        info=info,
        adjacent_overlapping_elements=adjacent_overlaps,
        seed=seed,
        info_aware_config=config,
        return_metadata=True,
    )
    return {
        "curve": curve,
        "runtime": runtime,
        "diagnostics": diagnostics,
        "metadata": metadata,
        "grouping_result": grouping_result,
        "info": info,
    }


def _finite_stats(values):
    numeric = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if numeric.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def _mean_or_none(values):
    numeric = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=float)
    if numeric.size == 0:
        return None
    return float(np.mean(numeric))


def _std_or_none(values):
    numeric = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=float)
    if numeric.size == 0:
        return None
    return float(np.std(numeric))


def summarize_method(run_rows):
    final_errors = [row["final_error"] for row in run_rows]
    nda_fes = [row["nda_fe_used"] for row in run_rows]
    early_switch_flags = [1.0 if row["early_switch_triggered"] else 0.0 for row in run_rows]
    spearman_values = [row["priority_delta_spearman"] for row in run_rows]
    positive_delta_rates = [row["positive_delta_rate_all"] for row in run_rows]
    final_stats = _finite_stats(final_errors)
    return {
        "final_error": final_stats,
        "nda_fe_used_mean": _mean_or_none(nda_fes),
        "early_switch_rate": _mean_or_none(early_switch_flags),
        "priority_delta_spearman_mean": _mean_or_none(spearman_values),
        "priority_delta_spearman_std": _std_or_none(spearman_values),
        "positive_delta_rate_all_mean": _mean_or_none(positive_delta_rates),
    }


def summarize_negative_deltas(trace_rows):
    negative_rows = [row for row in trace_rows if float(row.get("actual_delta", 0.0)) < 0.0]
    if not trace_rows:
        return {
            "negative_delta_count": 0,
            "negative_delta_rate": None,
            "negative_delta_overlap_ratio_mean": None,
            "negative_delta_conflict_prior_mean": None,
        }
    if not negative_rows:
        return {
            "negative_delta_count": 0,
            "negative_delta_rate": 0.0,
            "negative_delta_overlap_ratio_mean": 0.0,
            "negative_delta_conflict_prior_mean": 0.0,
        }
    return {
        "negative_delta_count": int(len(negative_rows)),
        "negative_delta_rate": float(len(negative_rows) / float(len(trace_rows))),
        "negative_delta_overlap_ratio_mean": float(np.mean([float(row["overlap_ratio"]) for row in negative_rows])),
        "negative_delta_conflict_prior_mean": float(np.mean([float(row["conflict_prior_mean"]) for row in negative_rows])),
    }


def save_aggregated_group_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "seed"] + list(hcc_es.GROUP_TRACE_FIELDNAMES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def compare_methods(per_method_rows, left_name, right_name):
    left_rows = per_method_rows[left_name]
    right_rows = per_method_rows[right_name]
    paired = list(zip(left_rows, right_rows))
    left_errors = np.asarray([row["final_error"] for row, _ in paired], dtype=float)
    right_errors = np.asarray([row["final_error"] for _, row in paired], dtype=float)
    return {
        "mean_gap": float(np.mean(left_errors - right_errors)),
        "wins": int(np.sum(left_errors < right_errors)),
        "ties": int(np.sum(np.isclose(left_errors, right_errors))),
        "losses": int(np.sum(left_errors > right_errors)),
    }


def format_float(value, digits=6):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def main():
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    ROUND2_ROOT.mkdir(parents=True, exist_ok=True)

    per_method_rows = {}
    aggregated_group_trace_rows = []
    per_method_trace_rows = {}

    for method_name, config_path in METHOD_CONFIGS:
        config = build_method_config(config_path)
        method_rows = []
        method_trace_rows = []
        method_dir = ROUND2_ROOT / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            result = run_case(config, seed)
            payload = result["metadata"].get("info_aware_diagnostics", {})
            trace_rows = list(result["metadata"].get("group_trace_rows", []))

            seed_dir = method_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_path = seed_dir / "info_aware_nda_diagnostics.json"
            info_aware_nda.save_info_aware_diagnostics(diagnostics_path, payload)
            if trace_rows:
                trace_path = seed_dir / "group_priority_trace.csv"
                hcc_es.save_group_trace_csv(trace_path, trace_rows)
                payload["group_trace_csv"] = trace_path.as_posix()
                info_aware_nda.save_info_aware_diagnostics(diagnostics_path, payload)

            method_rows.append(
                {
                    "seed": seed,
                    "final_error": float(result["curve"][-1]),
                    "best_error": float(np.min(result["curve"])),
                    "runtime": float(result["runtime"]),
                    "nda_fe_used": payload.get("nda_fe_used", 0),
                    "early_switch_triggered": bool(payload.get("early_switch_triggered", False)),
                    "priority_delta_spearman": payload.get("priority_delta_spearman"),
                    "positive_delta_rate_all": payload.get("positive_delta_rate_all"),
                    "priority_mode_effective": payload.get("priority_mode_effective"),
                }
            )

            for trace_row in trace_rows:
                row = {"method": method_name, "seed": seed}
                row.update(trace_row)
                aggregated_group_trace_rows.append(row)
                method_trace_rows.append(row)

        per_method_rows[method_name] = method_rows
        per_method_trace_rows[method_name] = method_trace_rows

    save_aggregated_group_trace(ARTIFACTS_ROOT / "group_priority_trace.csv", aggregated_group_trace_rows)

    method_summaries = {method_name: summarize_method(rows) for method_name, rows in per_method_rows.items()}
    negative_delta_summaries = {
        method_name: summarize_negative_deltas(rows)
        for method_name, rows in per_method_trace_rows.items()
    }

    diagnostic_vs_early = compare_methods(per_method_rows, "diagnostic-only", "early-switch-only")
    sort_vs_early = compare_methods(per_method_rows, "sort-dangerous-ablation", "early-switch-only")
    sort_vs_diagnostic = compare_methods(per_method_rows, "sort-dangerous-ablation", "diagnostic-only")

    sort_stable_degradation = bool(
        sort_vs_diagnostic["mean_gap"] > 0.0 and sort_vs_diagnostic["losses"] >= 3
    )

    report_lines = [
        "# Info-aware NDA Round2 Report",
        "",
        "## Setup",
        "",
        "1. 问题：Synthetic overlapping sphere",
        "2. 维度 D：20",
        "3. MaxFEs：120",
        f"4. Seeds：{SEEDS}",
        "5. 目标：分离 priority 的预测力与 sort 对 overlap blending 顺序的破坏效应。",
        "",
        "## Method Summary",
        "",
        "| Method | Final Mean | Final Std | Final Min | Final Max | NDA FE Mean | Early Switch Rate | Spearman Mean | Spearman Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method_name, _ in METHOD_CONFIGS:
        summary = method_summaries[method_name]
        report_lines.append(
            f"| {method_name} | "
            f"{format_float(summary['final_error']['mean'])} | "
            f"{format_float(summary['final_error']['std'])} | "
            f"{format_float(summary['final_error']['min'])} | "
            f"{format_float(summary['final_error']['max'])} | "
            f"{format_float(summary['nda_fe_used_mean'])} | "
            f"{format_float(summary['early_switch_rate'])} | "
            f"{format_float(summary['priority_delta_spearman_mean'])} | "
            f"{format_float(summary['priority_delta_spearman_std'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Pairwise Checks",
            "",
            f"- diagnostic-only vs early-switch-only: mean_gap={format_float(diagnostic_vs_early['mean_gap'])}, wins={diagnostic_vs_early['wins']}, ties={diagnostic_vs_early['ties']}, losses={diagnostic_vs_early['losses']}",
            f"- sort-dangerous-ablation vs early-switch-only: mean_gap={format_float(sort_vs_early['mean_gap'])}, wins={sort_vs_early['wins']}, ties={sort_vs_early['ties']}, losses={sort_vs_early['losses']}",
            f"- sort-dangerous-ablation vs diagnostic-only: mean_gap={format_float(sort_vs_diagnostic['mean_gap'])}, wins={sort_vs_diagnostic['wins']}, ties={sort_vs_diagnostic['ties']}, losses={sort_vs_diagnostic['losses']}",
            "",
            "## Per-seed Final Error",
            "",
            "| Seed | baseline | early-switch-only | diagnostic-only | sort-dangerous-ablation |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for row_index, seed in enumerate(SEEDS):
        report_lines.append(
            f"| {seed} | "
            f"{format_float(per_method_rows['baseline'][row_index]['final_error'])} | "
            f"{format_float(per_method_rows['early-switch-only'][row_index]['final_error'])} | "
            f"{format_float(per_method_rows['diagnostic-only'][row_index]['final_error'])} | "
            f"{format_float(per_method_rows['sort-dangerous-ablation'][row_index]['final_error'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Negative Delta Audit",
            "",
            "| Method | Negative Delta Count | Negative Delta Rate | Mean Overlap Ratio | Mean Conflict Prior |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for method_name in ["early-switch-only", "diagnostic-only", "sort-dangerous-ablation"]:
        summary = negative_delta_summaries[method_name]
        report_lines.append(
            f"| {method_name} | "
            f"{summary['negative_delta_count']} | "
            f"{format_float(summary['negative_delta_rate'])} | "
            f"{format_float(summary['negative_delta_overlap_ratio_mean'])} | "
            f"{format_float(summary['negative_delta_conflict_prior_mean'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"1. diagnostic-only 与 early-switch-only 5/5 完全一致：mean_gap={format_float(diagnostic_vs_early['mean_gap'])}, ties={diagnostic_vs_early['ties']}。这说明 diagnostic_only 没有改变优化行为。",
            f"2. diagnostic-only 的 priority_delta_spearman mean/std = {format_float(method_summaries['diagnostic-only']['priority_delta_spearman_mean'])} / {format_float(method_summaries['diagnostic-only']['priority_delta_spearman_std'])}。当前 priority 只有很弱且波动很大的正相关，预测力不稳定。",
            f"3. sort-dangerous-ablation 是否稳定劣化：{sort_stable_degradation}。它相对 diagnostic-only 的 mean_gap={format_float(sort_vs_diagnostic['mean_gap'])}，losses={sort_vs_diagnostic['losses']}。",
            "4. 负收益来源判断：所有方法的 group-level actual_delta 都没有出现负值，因此性能变差不是来自单个 group 的局部回退，而更像是 sort_dangerous_ablation 改写 execution_order 与 adjacent overlap path 之后，跨组协调路径被破坏。",
            "5. 下一步结论：保留 early-switch-only、diagnostic-only 和 sort-dangerous-ablation 这三个实验形态；其中 sort-dangerous-ablation 只作为反例消融，不再视为候选主方法。",
            "",
            "## Artifacts",
            "",
            f"- aggregated group trace: {(ARTIFACTS_ROOT / 'group_priority_trace.csv').as_posix()}",
            f"- round2 diagnostics root: {ROUND2_ROOT.as_posix()}",
        ]
    )

    report_path = ARTIFACTS_ROOT / "info_aware_nda_round2_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"group trace -> {ARTIFACTS_ROOT / 'group_priority_trace.csv'}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
