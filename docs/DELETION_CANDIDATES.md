# Deletion Candidates

- 日期：2026-05-21
- 执行者：Codex
- 状态：仅列候选，本轮没有删除任何文件。

## 禁止删除

以下内容不进入删除候选：

- `HCC_SRC/` 源码；
- `tests/` 测试；
- `configs/final/`；
- `scripts/final/`；
- `docs/` 本轮新增文档；
- `artifacts/final_evidence/`；
- `artifacts/arac_lite_v0_7_generalization_*`；
- `artifacts/arac_lite_v0_8_*`；
- V0.7/V0.8 case cache 目录；
- 任何能够支撑正式报告、复现或审计的 CSV/MD。

## 当前不建议立即删除

当前仓库处于论文证据整理阶段，很多临时日志和旧实验产物仍可能用于追溯。建议暂不删除，先完成论文主线草稿后再清理。

## 需要用户确认后才可删除的候选

| 候选类型 | 示例 | 风险 | 建议 |
| --- | --- | --- | --- |
| 过期 smoke 日志 | `.codex/*smoke*.log` | 可能丢失失败链上下文 | 论文主线稳定后再删 |
| 重复的临时输出目录 | `tmp/` 下的手工 smoke 输出 | 可能包含未汇总的 debug 结果 | 先列出具体路径再确认 |
| 已迁移的轻量副本 | 如果未来将某些 artifact 复制到 `artifacts/archive/` 后产生重复 | 可能破坏旧文档链接 | 先更新 reproduction 路径 |
| 空 archive 分类目录 | `configs/archive/*`、`scripts/archive/*`、`artifacts/archive/*` | 目录本身无害 | 保留可作为整理结构 |

## 大体积候选路径

以下路径是当前工作区 Git 噪声和磁盘占用的主要来源。本轮仅通过 `.gitignore` 让它们不再默认进入 Git 状态，没有删除任何文件。

| 候选路径 | 当前用途 | 删除风险 | 建议 |
| --- | --- | --- | --- |
| `artifacts/arac_lite_final_3e6_gate_runs/` | 3e6 gate 原始运行目录 | 可能丢失 long-budget failure audit 的原始 trace | 保留到 long-budget protocol 重定义完成后再评估 |
| `artifacts/arac_lite_final_3e6_gate_cases/relation_action_audit/` | 3e6 gate case 级 relation-action audit | 文件很大，但可支撑协议失败追溯 | 保留，除非已压缩或确认不再需要 |
| `artifacts/arac_lite_final_3e6_fast_gate_*` | 已中止/加速尝试的 fast-gate 聚合文件 | 主要是调度和资源探索记录 | 可在写入资源审计摘要后删除 |
| `artifacts/*_runs/` | 各实验线原始 run outputs | 删除后难以复查单 seed 细节 | 论文主线稳定前不建议删 |
| `artifacts/*_cases/` | case-level cache/resume 目录 | 删除后会失去 resume 能力 | 对已完成并汇总的旧实验，可确认后删除 |
| `artifacts/*relation_action_audit.csv` | 大型 relation/action 明细 | 很占空间，但支撑机制审计 | 先保留 V0.7/V0.8，旧版本可在 manifest 记录后再删 |
| `external_baselines/` | 外部 C++ baseline 快照 | 删除后需重新获取外部参考 | 当前仅忽略 Git，不建议删除 |

## 当前 Git 忽略策略

- `artifacts/*` 默认忽略，避免 raw experiment outputs 刷屏。
- `artifacts/final_evidence/**` 保持可追踪。
- `artifacts/arac_lite_final_3e6_short_audit.md`、`artifacts/arac_lite_final_3e6_short_audit_summary.csv`、`artifacts/arac_lite_v0_7_generalization_report.md`、`artifacts/arac_lite_v0_8_mechanism_ablation_report.md` 保持可追踪。
- `external_baselines/` 默认忽略，作为本地参考快照保留。

## 删除审批流程

1. 先生成具体路径清单；
2. 标注每个路径是否有复现引用；
3. 标注是否影响 V0.7/V0.8；
4. 用户确认后再删除；
5. 删除后运行本地验证。

本轮没有执行删除命令。
