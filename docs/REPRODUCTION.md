# ARAC-lite Reproduction Guide

- 日期：2026-05-21
- 执行者：Codex
- 平台口径：Windows PowerShell，本地 Python 环境

## Quick Validation

```powershell
python -m pytest -q
```

该命令用于确认当前仓库测试面未被整理动作破坏。不要用 CI 或人工外包验证替代本地执行。

## Final Evidence Package

生成轻量 final evidence 包：

```powershell
python scripts\final\generate_final_evidence_package.py
```

或通过 final wrapper：

```powershell
python scripts\final\run_arac_lite_final.py --stage package
```

输出目录：

```text
artifacts/final_evidence/
  main_performance/
  paired_robustness/
  action_distribution/
  delta_gate_stress/
  recovery_probe/
  caveats/
  MANIFEST.csv
  MANIFEST.md
```

默认不会复制数百 MB 级 relation-action audit CSV；manifest 会引用原路径。确实需要完整复制时：

```powershell
python scripts\final\generate_final_evidence_package.py --include-large-audits
```

## V0.7 Fixed Candidate Generalization

正式矩阵：

```powershell
python scripts\generate_arac_lite_v0_7_generalization.py --problems E4 E6 S4 S6 A6 R6 --seeds 1 2 3 4 5 6 7 8 9 10 --tfes 5000 10000 20000 --cc-pass-group-fes 20 --workers 12 --resume *> .codex\arac_lite_v0_7_full_workers12.log
```

等价 wrapper：

```powershell
python scripts\final\run_arac_lite_final.py --stage v0.7 --workers 12 --resume
```

已记录结果：

```text
900/900 ok
relation-action audit rows: 2,509,425
probe metric rows: 36
vs disable-fast majority non-worse: 6/6 problems
max extra FE ratio: +0.184%
```

## V0.8 Mechanism Ablation

正式矩阵：

```powershell
python scripts\generate_arac_lite_v0_8_mechanism_ablation.py --problems E4 E6 S4 S6 A6 R6 --stress-problems R6 S6 E6 --seeds 1 2 3 4 5 6 7 8 9 10 --tfes 5000 10000 20000 --cc-pass-group-fes 20 --workers 12 --resume *> .codex\arac_lite_v0_8_full_workers12.log
```

等价 wrapper：

```powershell
python scripts\final\run_arac_lite_final.py --stage v0.8 --workers 12 --resume
```

已记录结果：

```text
main matrix: 900/900 ok
delta stress: 180/180 ok
budget target total: 34
random probe total: 34
max_abs_budget_gap: 0
```

## Final Config Smoke

配置文件可通过 `InfoAwareNDAConfig` loader 读取：

```powershell
python -m py_compile scripts\final\run_arac_lite_final.py scripts\final\generate_final_evidence_package.py
```

如需直接跑 HCC-ES 单方法实验，可使用：

```powershell
python HCC_SRC\HCC-ES.py --problem E6 --tfes 5000 --cycle-num 1 --enable-info-aware-nda --info-aware-nda-config configs\final\arac-lite-final.json --output-dir tmp\arac_lite_final_smoke
```

注意：正式论文证据仍以 V0.7/V0.8 生成器产物为准。

## Interpretation Rules

- 不要把 targeted probe 写成主贡献；
- same-budget random 结果必须作为 caveat；
- delta gate stress 是机制必要性的主要证据；
- 相对 no-coordination 的不全面优势必须诚实保留；
- V0.7/V0.8 原始 artifact 路径是复现锚点，不要移动或删除。

