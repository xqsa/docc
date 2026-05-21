# ARAC-lite Final Method

- 日期：2026-05-21
- 执行者：Codex
- 仓库：`C:/Users/83718/Desktop/HCC/HCC-main`

## 方法名

**ARAC-lite: Delta-gated Relation-to-Action Coordination**

## 定位

ARAC-lite Final 是当前仓库中用于论文主线的冻结候选。它不是完整 ARAC，不包含 UCB/bandit，也不把 targeted probe selection 作为主贡献。当前代码级候选继承 V0.6 targeted-probe 的实现路径，是因为 V0.7/V0.8 的完整矩阵都围绕该候选展开；但 V0.8 已显示 targeted probe 没有明显优于 same-budget random probe，因此论文叙事应把 targeted selection 作为实现细节和 caveat，而不是机制贡献。

最终可主张的机制是：

1. relation-to-action mapping；
2. disable-fast defensive base；
3. low-cost recovery probe；
4. delta-gated recovery。

## 核心思想

现有 overlapping LSGO 方法往往默认识别出的重叠变量应该被协调，或仅依赖局部贡献度进行共享变量处理。ARAC-lite 将重叠关系视为需要动作决策的优化对象，根据关系状态将其映射到 Fusion、Freeze、Disable，并通过低成本 recovery probe 与 global/rolling delta gate 决定是否允许恢复协调。

换句话说，ARAC-lite 不再问“是否统一融合所有重叠关系”，而是问：

```text
当前关系状态支持什么动作？
```

## Relation Diagnosis

每个共享变量或重叠关系在 pass-level coordination 中积累诊断状态：

- proposal support：有多少组对同一个共享变量提出更新；
- proposal agreement / conflict：提案方向、幅度、方差和正向提案数量；
- validation accept / reject：Fusion 候选经过 pass-end full-fitness validation 后是否改善；
- global or rolling delta：近期 validation delta 是否为正，是否存在高 accept 但负 delta 的虚假恢复。

这些状态共同决定关系是应当继续 Fusion、暂时 Freeze，还是快速 Disable。

## Action Mapping

ARAC-lite 的最小动作集合是：

| Action | 语义 | 当前用途 |
| --- | --- | --- |
| Fusion | 允许多个重叠关系的更新融合并进入 validation | 只在有足够支持、冲突低且 delta gate 不阻止时使用 |
| Freeze | 暂停不确定关系，不提交潜在坏更新 | 用于证据不足或提案冲突但尚未确认应关闭的关系 |
| Disable | 快速关闭坏关系 | disable-fast base，用于保护 R6 等对错误协调敏感的问题 |

原始 ARAC 设想中的 Owner 动作当前不作为主方法的一部分。owner_soft 属于归档支线，不进入 ARAC-lite Final 贡献链。

## Defensive Base

disable-fast 是 ARAC-lite Final 的防御底盘。它使用非常保守的历史门槛：

```text
arac_lite_history_min_attempts = 1
arac_lite_disable_reject_streak = 1
arac_lite_disable_accept_rate_threshold = 0.0
arac_lite_disable_mean_delta_threshold = 0.0
```

它的作用不是追求激进提升，而是在发现坏 Fusion 信号后快速关闭该关系，避免默认协调在 R6 这类问题上扩大损失。V0.7 的证据显示，V0.6 candidate 相对 disable-fast 在 6/6 问题上 majority non-worse，因此 recovery 通道没有破坏防御底盘。

## Recovery

Recovery 分两层：

1. low-cost recovery probe：在极低额外 FE 成本下，允许少量被关闭或冻结的关系重新进入 validation；
2. delta-gated recovery：只有当 recovery 关系的 global/rolling delta 不显示负向风险时，才允许恢复 Fusion。

当前 final 配置 `configs/final/arac-lite-final.json` 保留 V0.6 的低频 middle-phase probe 设置，并维持 delta threshold 为 `0.0`。这意味着恢复机制必须跨过“非负 delta”门槛，不能只依赖 accept rate。

## Evidence Snapshot

V0.7 fixed-candidate generalization：

- 矩阵：E4/E6/S4/S6/A6/R6 × TFEs 5000/10000/20000 × seeds 1..10；
- 运行：900/900 ok；
- vs disable-fast：6/6 problems majority non-worse；
- V0.6 candidate Fusion count：32832；
- max extra FE ratio：+0.184%；
- targeted probes：34；
- matched probes：13；
- recovered Fusion：9；
- bad probes：2；
- R6 bad_probe_count：2。

V0.8 mechanism ablation：

- main matrix：900/900 ok；
- delta stress：180/180 ok；
- same-budget random budget alignment：target total 34，random total 34，max_abs_budget_gap 0；
- targeted vs random 多数 paired case 打平，targeted 没有明显优势；
- recovery totals：accept+delta 将 R6 bad recovery 控制在 523，accept-only 放大到 1352。

## Caveats

- targeted probe 在 V0.8 中没有显著优于 same-budget random；不能作为主贡献；
- V0.6 targeted-probe 是当前完整矩阵验证过的代码级候选，但论文机制归因应落在 low-cost recovery 与 delta gate；
- delta gate stress test 显示出明显必要性：accept-only recovery 会显著放大 R6 上高 accept 但负 global delta 的坏恢复；
- 当前版本定位为 ARAC-lite，不是完整 ARAC，不包含 UCB/bandit，也不声称解决所有 overlapping LSGO 问题；
- 相对 no-coordination，ARAC-lite 不是全面赢家；更准确的定位是 risk control + low-cost recovery。

## Main Files

- Source implementation: `HCC_SRC/HCC-ES.py`
- Config schema and normalization: `HCC_SRC/HCC/info_aware_nda.py`
- Final config: `configs/final/arac-lite-final.json`
- V0.7 generalization script: `scripts/generate_arac_lite_v0_7_generalization.py`
- V0.8 mechanism ablation script: `scripts/generate_arac_lite_v0_8_mechanism_ablation.py`
- Final evidence packager: `scripts/final/generate_final_evidence_package.py`

