# Experiment Ledger

- 日期：2026-05-21
- 执行者：Codex
- 目的：把探索线、归档线和论文主线分开，避免后续把支线结果误写成主方法。

## Mainline

| 阶段 | 脚本 / 产物 | 状态 | 结论 |
| --- | --- | --- | --- |
| ARAC-lite V0.7 fixed candidate generalization | `scripts/generate_arac_lite_v0_7_generalization.py`; `artifacts/arac_lite_v0_7_generalization_report.md` | 主线证据 | V0.6 candidate 相对 disable-fast 在 6/6 问题 majority non-worse，额外 FE 低，Fusion 通道未清零 |
| ARAC-lite V0.8 same-budget random ablation | `scripts/generate_arac_lite_v0_8_mechanism_ablation.py`; `artifacts/arac_lite_v0_8_mechanism_ablation_report.md` | 主线 caveat | targeted probe 没有明显优于 same-budget random，不能作为主贡献 |
| ARAC-lite V0.8 delta gate stress | `artifacts/arac_lite_v0_8_delta_stress_recovery_metrics.csv` | 主线机制证据 | accept-only 在 R6 上显著放大 bad recovery；delta gate 是关键保护信号 |

## ARAC-lite Exploration

| 阶段 | 代表产物 | 归属 | 结论 |
| --- | --- | --- | --- |
| V0 | `scripts/generate_arac_lite_v0_artifacts.py` | 归档 | 初始 relation-to-action audit |
| V0.1 disable-fast | `scripts/generate_arac_lite_v0_1_artifacts.py` | baseline | 形成快速关闭坏 Fusion 的防御底盘 |
| V0.2/V0.3 robustness | `scripts/generate_arac_lite_v0_2_artifacts.py`, `scripts/generate_arac_lite_v0_3_artifacts.py` | 归档 | 扩展 paired robustness / rank / action 统计口径 |
| V0.4 recovery | `scripts/generate_arac_lite_v0_4_recovery_artifacts.py` | 归档 | recovery 候选分析 |
| V0.5 low-frequency probe | `scripts/generate_arac_lite_v0_5_probe_artifacts.py`, `scripts/generate_arac_lite_v0_5_failure_audit.py` | failure audit | 成本低但收益失败，R6 bad probe 暴露 accept-rate 不可靠 |
| V0.6 targeted probe | `scripts/generate_arac_lite_v0_6_targeted_probe_artifacts.py` | 候选来源 | 小矩阵显示能恢复部分 S6/E6 关系，但 V0.8 后不再把 targeted selection 作为贡献 |

## HCC Fidelity Line

| 阶段 | 代表脚本 / 配置 | 归属 | 结论 |
| --- | --- | --- | --- |
| paper-fidelity audit | `scripts/generate_hcc_paper_fidelity_audit.py` | 归档支线 | 审计 RDDSM order、AOB order visibility、Equation 8 语义 |
| Patch V1 | `scripts/generate_hcc_fidelity_patch_v1*.py` | 归档支线 | topology-preserving ordering 与 Equation 8 修复线 |
| Patch V2 omega | `scripts/generate_hcc_fidelity_patch_v2_omega_artifacts.py` | 归档支线 | omega_i / persistent_mean 同步作为独立消融，不混入 ARAC-lite Final |

## Coordination Exploration

| 线 | 代表脚本 / 配置 | 归属 | 结论 |
| --- | --- | --- | --- |
| hypergraph shared-variable coordination | `scripts/generate_hypergraph_shared_variable_artifacts.py` | 归档支线 | 保留为结构协调探索，不是 final 方法 |
| validated coordination | `scripts/generate_validated_hypergraph_coordination_artifacts.py`; `configs/info_aware_nda/validated-selective-conflict.json` | baseline | 作为 V0.7/V0.8 对照方法 |
| owner_soft | `scripts/sweep_dynamic_soft_ownership.py`, `scripts/validate_owner_soft_longfe.py` | 归档支线 | 不进入当前 ARAC-lite Final |
| coordination selector | `scripts/generate_coordination_selector_artifacts.py`; `configs/info_aware_nda/coordination-selector.json` | 归档支线 | 暂停，不上 UCB/bandit |

## Final Evidence Directory

最终证据包由以下命令生成：

```powershell
python scripts\final\generate_final_evidence_package.py
```

该命令会复制轻量证据并生成：

- `artifacts/final_evidence/MANIFEST.csv`
- `artifacts/final_evidence/MANIFEST.md`
- `artifacts/final_evidence/caveats/CAVEATS.md`

超大 audit CSV 默认只引用，不复制。

