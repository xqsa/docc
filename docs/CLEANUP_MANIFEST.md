# Cleanup Manifest

- 日期：2026-05-21
- 执行者：Codex
- 策略：非破坏式整理，不直接删除。

## 本轮新增

| 路径 | 用途 |
| --- | --- |
| `docs/PROJECT_MAP.md` | 项目地图 |
| `docs/FINAL_METHOD.md` | 主方法定义 |
| `docs/REPRODUCTION.md` | 复现命令 |
| `docs/EXPERIMENT_LEDGER.md` | 实验账本 |
| `docs/CLEANUP_MANIFEST.md` | 清理策略 |
| `docs/DELETION_CANDIDATES.md` | 删除候选清单 |
| `configs/final/` | 主方法和核心 baseline 配置 |
| `scripts/final/` | final workflow 包装入口 |
| `artifacts/final_evidence/` | final evidence 证据包输出目录 |
| `configs/archive/`, `scripts/archive/`, `artifacts/archive/` | 归档分类目录，当前不强制搬动旧文件 |

## 必须保留

- `HCC_SRC/` 全部源码；
- `tests/` 全部测试；
- `configs/info_aware_nda/` 既有配置；
- `scripts/generate_arac_lite_v0_7_generalization.py`；
- `scripts/generate_arac_lite_v0_8_mechanism_ablation.py`；
- `scripts/generate_arac_lite_v0_*.py` 旧版本复现脚本；
- `artifacts/arac_lite_v0_7_generalization_*`；
- `artifacts/arac_lite_v0_8_*`；
- `verification.md` 和 `.codex/testing.md`；
- `.codex-tasks/` 中既有任务记录。

## 归档分类

当前仅创建分类目录，不移动旧文件。后续如要移动，应先确认移动不会破坏文档中的复现命令。

| 分类 | 目标目录 | 典型内容 |
| --- | --- | --- |
| HCC fidelity | `configs/archive/hcc_fidelity/`, `scripts/archive/hcc_fidelity/`, `artifacts/archive/hcc_fidelity/` | paper-fidelity audit、Patch V1/V2、omega_i |
| owner_soft | `configs/archive/owner_soft/`, `scripts/archive/owner_soft/`, `artifacts/archive/owner_soft/` | soft ownership sweep 和长 FE 验证 |
| selector | `configs/archive/selector/`, `scripts/archive/selector/`, `artifacts/archive/selector/` | coordination selector |
| hypergraph | `configs/archive/hypergraph/` | hypergraph/pass-end/shared-variable 探索 |
| old ARAC versions | `configs/archive/old_arac_versions/`, `scripts/archive/arac_exploration/`, `artifacts/archive/arac_v0_1_to_v0_6/` | V0 到 V0.6 探索线 |

## 本轮未执行

- 未删除任何文件；
- 未移动 V0.7/V0.8 产物；
- 未移动或删除测试；
- 未改算法默认行为；
- 未把探索性分支改名为主方法；
- 未复制数百 MB 级 relation-action audit 到 final_evidence，默认在 manifest 中引用原路径。

## 后续清理建议

1. 先用 `docs/DELETION_CANDIDATES.md` 审批删除候选；
2. 对需要移动的支线 artifact 先写迁移映射；
3. 更新 `docs/REPRODUCTION.md` 后再移动；
4. 移动后运行 `python -m pytest -q` 和 final evidence package；
5. 确认 V0.7/V0.8 原始证据路径仍可追溯。

