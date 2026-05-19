# Info-aware NDA Minimal Report

1. 使用的问题：Synthetic overlapping sphere
2. 维度 D：20
3. max_fes：120
4. seed：7
5. baseline final error：94.786071805075
6. early switch final error：88.789329152591
7. diagnostic-only final error：88.789329152591
8. sort-dangerous-ablation final error：103.672120116551
9. NDA used FEs：13
10. NDA used ratio：0.108333333333
11. 是否触发 early switch：True (early_stagnation)
12. diagnostic-only group priority top-5：[{'group_id': 1, 'priority': 1.0815754473604517, 'group_size': 10, 'overlap_ratio': 0.4}, {'group_id': 0, 'priority': 0.9698141262492451, 'group_size': 10, 'overlap_ratio': 0.2}, {'group_id': 2, 'priority': 0.9486104263903036, 'group_size': 4, 'overlap_ratio': 0.5}]
13. sort-dangerous-ablation 是否改变顺序：True
14. 观察到的问题：diagnostic-only 与 early switch-only 不应改变优化行为，而 sort-dangerous-ablation 只保留为反例消融，用来展示重排 CC 顺序会破坏原始 overlap blending 路径。
15. 下一步建议：保留 diagnostic-only 作为默认审计模式，不再把危险重排视为候选主方法。

- diagnostics: C:/Users/83718/Desktop/HCC/HCC-main/artifacts/info_aware_nda_diagnostics.json