# HCC-main Project Map

- 日期：2026-05-21
- 执行者：Codex
- 仓库根目录：`C:/Users/83718/Desktop/HCC/HCC-main`

## 读图原则

当前项目已经从探索阶段进入 ARAC-lite 主方法收敛阶段。整理策略是非破坏式：

- 原始源码、测试、V0.7/V0.8 产物保持原路径；
- `docs/` 提供论文主线地图；
- `configs/final/` 提供主方法和关键 baseline 配置；
- `scripts/final/` 提供复现包装入口；
- `artifacts/final_evidence/` 提供最终证据包；
- `archive/` 目录用于分类，但本轮不移动会破坏旧命令的文件。

## Core Source

| 路径 | 作用 | 状态 |
| --- | --- | --- |
| `HCC_SRC/HCC-ES.py` | HCC-ES 主入口、CC pass、shared-variable coordination、ARAC-lite action/probe/recovery 执行位置 | 保留，未在本次整理中改算法行为 |
| `HCC_SRC/HCC/info_aware_nda.py` | `InfoAwareNDAConfig` 配置 schema、归一化、NDA 诊断与辅助结构 | 保留，final JSON 必须只使用该 dataclass 支持的字段 |
| `HCC_SRC/HCC/OPT/CMAES/cmaes.py` | CMA-ES 子优化器 | 归属原 HCC 复现，不纳入本次整理改动 |
| `HCC_SRC/experiment_protocols.py` | 实验协议辅助入口 | 保留 |

## Final Configs

| 路径 | 方法名 | 用途 |
| --- | --- | --- |
| `configs/final/arac-lite-final.json` | ARAC-lite Final | 冻结当前代码级候选配置：relation-to-action + disable-fast + low-cost probe + delta gate |
| `configs/final/no-coordination.json` | no-coordination | 无共享变量协调 baseline |
| `configs/final/validated-selective-conflict.json` | validated-selective-conflict | pass-end validated selective hypergraph baseline |
| `configs/final/disable-fast.json` | arac-lite-v0.1-disable-fast | 防御底盘 baseline |

注意：`configs/final/arac-lite-final.json` 保留 V0.6 targeted-probe 的代码级候选设置，是为了和 V0.7/V0.8 证据链对齐；论文贡献不能写成 targeted selection。

## Final Scripts

| 路径 | 作用 |
| --- | --- |
| `scripts/final/run_arac_lite_final.py` | final workflow 包装入口，可执行 V0.7、V0.8 或 evidence package |
| `scripts/final/generate_final_evidence_package.py` | 复制轻量最终证据，并引用超大 relation-action audit 文件 |

推荐入口：

```powershell
python scripts\final\run_arac_lite_final.py --stage package
python scripts\final\run_arac_lite_final.py --stage v0.7 --workers 12 --resume
python scripts\final\run_arac_lite_final.py --stage v0.8 --workers 12 --resume
```

## Main Evidence Artifacts

| 路径 | 内容 |
| --- | --- |
| `artifacts/arac_lite_v0_7_generalization_report.md` | V0.7 固定候选泛化报告 |
| `artifacts/arac_lite_v0_7_generalization_run_details.csv` | V0.7 900-run 明细 |
| `artifacts/arac_lite_v0_7_generalization_summary.csv` | V0.7 performance summary |
| `artifacts/arac_lite_v0_7_generalization_robustness.csv` | V0.7 paired robustness |
| `artifacts/arac_lite_v0_7_generalization_action_distribution.csv` | V0.7 action distribution |
| `artifacts/arac_lite_v0_7_generalization_probe_metrics.csv` | V0.7 probe metrics |
| `artifacts/arac_lite_v0_8_mechanism_ablation_report.md` | V0.8 mechanism ablation 报告 |
| `artifacts/arac_lite_v0_8_mechanism_ablation_budget_alignment.csv` | same-budget random 对齐证据 |
| `artifacts/arac_lite_v0_8_delta_stress_recovery_metrics.csv` | delta gate stress recovery 指标 |

超大审计文件保留原路径，不默认复制：

- `artifacts/arac_lite_v0_7_generalization_relation_action_audit.csv`
- `artifacts/arac_lite_v0_8_mechanism_ablation_relation_action_audit.csv`
- `artifacts/arac_lite_v0_8_delta_stress_relation_action_audit.csv`

## Experiment Script Families

| 家族 | 代表脚本 | 当前归属 |
| --- | --- | --- |
| ARAC-lite final | `scripts/generate_arac_lite_v0_7_generalization.py`, `scripts/generate_arac_lite_v0_8_mechanism_ablation.py` | 主线证据 |
| ARAC-lite exploration V0-V0.6 | `scripts/generate_arac_lite_v0_artifacts.py` 到 `scripts/generate_arac_lite_v0_6_targeted_probe_artifacts.py` | 归档支线，保留复现 |
| HCC paper-fidelity | `scripts/generate_hcc_paper_fidelity_audit.py`, `scripts/generate_hcc_fidelity_patch_v1*.py`, `scripts/generate_hcc_fidelity_patch_v2_omega_artifacts.py` | 归档支线，保留复现 |
| hypergraph / validated coordination | `scripts/generate_hypergraph_shared_variable_artifacts.py`, `scripts/generate_validated_hypergraph_coordination_artifacts.py` | baseline/支线 |
| owner_soft / selector | `scripts/sweep_dynamic_soft_ownership.py`, `scripts/validate_owner_soft_longfe.py`, `scripts/generate_coordination_selector_artifacts.py` | 归档支线 |

## Tests

关键测试全部保留：

- `tests/test_arac_lite_v0_1.py` 到 `tests/test_arac_lite_v0_8.py`
- `tests/test_info_aware_nda.py`
- `tests/test_hcc_es_none_guard.py`
- `tests/test_experiment_protocols.py`
- external baseline / PyPop7 adapter tests

本轮整理不删除测试、不降低测试覆盖、不修改算法默认行为。

