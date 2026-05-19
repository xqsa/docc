import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "HCC_SRC"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


spec = importlib.util.spec_from_file_location("hcc_es_module", SOURCE_ROOT / "HCC-ES.py")
hcc_es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hcc_es)

import HCC.info_aware_nda as info_aware_nda


class SyntheticSphere:
    def __init__(self):
        self.fitness_record = []

    def __call__(self, x_batch):
        x_array = np.asarray(x_batch, dtype=float)
        values = np.sum(np.square(x_array), axis=-1)
        self.fitness_record.extend(np.asarray(values, dtype=float).reshape(-1).tolist())
        return values


def run_case(config):
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
        seed=7,
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


def main():
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = info_aware_nda.InfoAwareNDAConfig(enable=False)
    early_switch_config = info_aware_nda.InfoAwareNDAConfig(
        enable=True,
        enable_trajectory_distill=False,
        enable_group_priority=False,
        priority_mode="off",
        min_nda_fe_ratio=0.05,
        max_nda_fe_ratio=0.6,
        window_size=2,
        patience=1,
        eps_improve=0.05,
        eps_center_shift=0.05,
    )
    diagnostic_config = info_aware_nda.InfoAwareNDAConfig(
        enable=True,
        enable_trajectory_distill=True,
        enable_group_priority=True,
        priority_mode="diagnostic_only",
        min_nda_fe_ratio=0.05,
        max_nda_fe_ratio=0.6,
        window_size=2,
        patience=1,
        eps_improve=0.05,
        eps_center_shift=0.05,
    )
    sort_ablation_config = info_aware_nda.InfoAwareNDAConfig(
        enable=True,
        enable_trajectory_distill=True,
        enable_group_priority=True,
        priority_mode="sort_dangerous_ablation",
        min_nda_fe_ratio=0.05,
        max_nda_fe_ratio=0.6,
        window_size=2,
        patience=1,
        eps_improve=0.05,
        eps_center_shift=0.05,
    )

    baseline = run_case(baseline_config)
    early_switch = run_case(early_switch_config)
    diagnostic_only = run_case(diagnostic_config)
    sort_ablation = run_case(sort_ablation_config)

    diagnostic_payload = diagnostic_only["metadata"]["info_aware_diagnostics"]
    sort_ablation_payload = sort_ablation["metadata"]["info_aware_diagnostics"]
    diagnostics_path = artifacts_dir / "info_aware_nda_diagnostics.json"
    info_aware_nda.save_info_aware_diagnostics(diagnostics_path, diagnostic_payload)

    report_path = artifacts_dir / "info_aware_nda_minimal_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Info-aware NDA Minimal Report",
                "",
                "1. 使用的问题：Synthetic overlapping sphere",
                f"2. 维度 D：{diagnostic_only['info']['dimension']}",
                "3. max_fes：120",
                "4. seed：7",
                f"5. baseline final error：{baseline['curve'][-1]:.12f}",
                f"6. early switch final error：{early_switch['curve'][-1]:.12f}",
                f"7. diagnostic-only final error：{diagnostic_only['curve'][-1]:.12f}",
                f"8. sort-dangerous-ablation final error：{sort_ablation['curve'][-1]:.12f}",
                f"9. NDA used FEs：{diagnostic_payload['nda_fe_used']}",
                f"10. NDA used ratio：{diagnostic_payload['nda_fe_ratio']:.12f}",
                f"11. 是否触发 early switch：{diagnostic_payload['early_switch_triggered']} ({diagnostic_payload['early_switch_reason']})",
                f"12. diagnostic-only group priority top-5：{diagnostic_payload['top_priority_groups']}",
                f"13. sort-dangerous-ablation 是否改变顺序：{sort_ablation_payload['sort_dangerous_ablation_changed_order']}",
                "14. 观察到的问题：diagnostic-only 与 early switch-only 不应改变优化行为，而 sort-dangerous-ablation 只保留为反例消融，用来展示重排 CC 顺序会破坏原始 overlap blending 路径。",
                "15. 下一步建议：保留 diagnostic-only 作为默认审计模式，不再把危险重排视为候选主方法。",
                "",
                f"- diagnostics: {diagnostics_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"diagnostics -> {diagnostics_path}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
