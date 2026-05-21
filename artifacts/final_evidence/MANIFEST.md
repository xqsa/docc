# ARAC-lite Final Evidence Manifest

- 日期：2026-05-21
- 执行者：Codex
- 生成时间：2026-05-21T10:50:35

## Package Policy

- 轻量 summary、robustness、action distribution、probe metrics 和报告文件复制到 `artifacts/final_evidence/`。
- relation-action audit CSV 为数百 MB 级，默认只在 manifest 中引用原路径；需要完整复制时运行 `--include-large-audits`。
- 该证据包不移动、不删除、不重写原始 V0.7/V0.8 产物。

## Copied Files

| category | file | bytes |
| --- | --- | ---: |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_7_generalization_report.md` | 7812 |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_7_generalization_summary.csv` | 27614 |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_7_generalization_run_details.csv` | 168227 |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_7_generalization_rank_summary.csv` | 7096 |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_8_mechanism_ablation_report.md` | 9567 |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_8_mechanism_ablation_summary.csv` | 27740 |
| main_performance | `artifacts\final_evidence\main_performance\arac_lite_v0_8_mechanism_ablation_run_details.csv` | 169115 |
| paired_robustness | `artifacts\final_evidence\paired_robustness\arac_lite_v0_7_generalization_robustness.csv` | 30672 |
| paired_robustness | `artifacts\final_evidence\paired_robustness\arac_lite_v0_8_mechanism_ablation_robustness.csv` | 38813 |
| paired_robustness | `artifacts\final_evidence\paired_robustness\arac_lite_v0_8_delta_stress_robustness.csv` | 3277 |
| action_distribution | `artifacts\final_evidence\action_distribution\arac_lite_v0_7_generalization_action_distribution.csv` | 25997 |
| action_distribution | `artifacts\final_evidence\action_distribution\arac_lite_v0_8_mechanism_ablation_action_distribution.csv` | 26278 |
| delta_gate_stress | `artifacts\final_evidence\delta_gate_stress\arac_lite_v0_8_delta_stress_run_details.csv` | 37633 |
| delta_gate_stress | `artifacts\final_evidence\delta_gate_stress\arac_lite_v0_8_delta_stress_summary.csv` | 5530 |
| delta_gate_stress | `artifacts\final_evidence\delta_gate_stress\arac_lite_v0_8_delta_stress_robustness.csv` | 3277 |
| delta_gate_stress | `artifacts\final_evidence\delta_gate_stress\arac_lite_v0_8_delta_stress_probe_metrics.csv` | 2190 |
| delta_gate_stress | `artifacts\final_evidence\delta_gate_stress\arac_lite_v0_8_delta_stress_recovery_metrics.csv` | 1918 |
| recovery_probe | `artifacts\final_evidence\recovery_probe\arac_lite_v0_7_generalization_probe_metrics.csv` | 3424 |
| recovery_probe | `artifacts\final_evidence\recovery_probe\arac_lite_v0_8_mechanism_ablation_probe_metrics.csv` | 3422 |
| recovery_probe | `artifacts\final_evidence\recovery_probe\arac_lite_v0_8_mechanism_ablation_budget_alignment.csv` | 810 |
| caveats | `artifacts\final_evidence\caveats\arac_lite_v0_8_mechanism_ablation_report.md` | 9567 |

## Referenced Large Files

| category | source | bytes |
| --- | --- | ---: |
| paired_robustness | `artifacts\arac_lite_v0_7_generalization_relation_action_audit.csv` | 880937663 |
| paired_robustness | `artifacts\arac_lite_v0_8_mechanism_ablation_relation_action_audit.csv` | 881372157 |
| delta_gate_stress | `artifacts\arac_lite_v0_8_delta_stress_relation_action_audit.csv` | 248572618 |

## Generated Files

| category | file | bytes |
| --- | --- | ---: |
| caveats | `artifacts\final_evidence\caveats\CAVEATS.md` | 565 |
