# Info-aware NDA Round2 Report

## Setup

1. 问题：Synthetic overlapping sphere
2. 维度 D：20
3. MaxFEs：120
4. Seeds：[1, 2, 3, 4, 5]
5. 目标：分离 priority 的预测力与 sort 对 overlap blending 顺序的破坏效应。

## Method Summary

| Method | Final Mean | Final Std | Final Min | Final Max | NDA FE Mean | Early Switch Rate | Spearman Mean | Spearman Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 102.220715 | 10.886480 | 87.010649 | 120.043935 | 0.000000 | 0.000000 | n/a | n/a |
| early-switch-only | 87.575790 | 5.899982 | 80.804224 | 97.986071 | 19.000000 | 0.800000 | -0.900000 | 0.200000 |
| diagnostic-only | 87.575790 | 5.899982 | 80.804224 | 97.986071 | 19.000000 | 0.800000 | 0.100000 | 0.734847 |
| sort-dangerous-ablation | 95.838471 | 12.504533 | 81.140209 | 111.835817 | 19.000000 | 0.800000 | 0.500000 | 0.547723 |

## Pairwise Checks

- diagnostic-only vs early-switch-only: mean_gap=0.000000, wins=0, ties=5, losses=0
- sort-dangerous-ablation vs early-switch-only: mean_gap=8.262682, wins=1, ties=1, losses=3
- sort-dangerous-ablation vs diagnostic-only: mean_gap=8.262682, wins=1, ties=1, losses=3

## Per-seed Final Error

| Seed | baseline | early-switch-only | diagnostic-only | sort-dangerous-ablation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 120.043935 | 88.757279 | 88.757279 | 109.692226 |
| 2 | 87.010649 | 87.010649 | 87.010649 | 87.010649 |
| 3 | 96.146256 | 83.320725 | 83.320725 | 89.513456 |
| 4 | 104.532149 | 80.804224 | 80.804224 | 111.835817 |
| 5 | 103.370588 | 97.986071 | 97.986071 | 81.140209 |

## Negative Delta Audit

| Method | Negative Delta Count | Negative Delta Rate | Mean Overlap Ratio | Mean Conflict Prior |
| --- | ---: | ---: | ---: | ---: |
| early-switch-only | 0 | 0.000000 | 0.000000 | 0.000000 |
| diagnostic-only | 0 | 0.000000 | 0.000000 | 0.000000 |
| sort-dangerous-ablation | 0 | 0.000000 | 0.000000 | 0.000000 |

## Conclusions

1. diagnostic-only 与 early-switch-only 5/5 完全一致：mean_gap=0.000000, ties=5。这说明 diagnostic_only 没有改变优化行为。
2. diagnostic-only 的 priority_delta_spearman mean/std = 0.100000 / 0.734847。当前 priority 只有很弱且波动很大的正相关，预测力不稳定。
3. sort-dangerous-ablation 是否稳定劣化：True。它相对 diagnostic-only 的 mean_gap=8.262682，losses=3。
4. 负收益来源判断：所有方法的 group-level actual_delta 都没有出现负值，因此性能变差不是来自单个 group 的局部回退，而更像是 sort_dangerous_ablation 改写 execution_order 与 adjacent overlap path 之后，跨组协调路径被破坏。
5. 下一步结论：保留 early-switch-only、diagnostic-only 和 sort-dangerous-ablation 这三个实验形态；其中 sort-dangerous-ablation 只作为反例消融，不再视为候选主方法。

## Artifacts

- aggregated group trace: C:/Users/83718/Desktop/HCC/HCC-main/artifacts/group_priority_trace.csv
- round2 diagnostics root: C:/Users/83718/Desktop/HCC/HCC-main/artifacts/round2