# ARAC-lite-final 3e6 Gate Short Audit

- Date: 2026-05-21
- Executor: Codex
- Repo: `C:/Users/83718/Desktop/HCC/HCC-main`
- Scope: pause the large 3e6 gate run and audit configuration identity, metric semantics, long-budget protocol, `final_error` drift, and one small baseline calibration.

## Executive Finding

The cached 3e6 rows are genuine `ARAC-lite-final` runs, but this gate is not directly comparable to the paper HCC-ES reference table. The decisive issue is the carried-over small-budget protocol: `cc_pass_group_fes=20` turns the 3e6 run into thousands of tiny 20-FE subspace CMAES updates. A one-seed repo-default HCC-ES rerun on E4 reaches the paper scale, while the `cc_pass_group_fes=20` ARAC-lite-final row is more than one order of magnitude worse.

Therefore, the current 3e6 gate should be treated as a protocol-failure audit, not as evidence that the ARAC-lite relation-to-action rule itself fails at long budget.

## 1. Run Identity

Direct scan of case-level CSV files under:

```text
artifacts/arac_lite_final_3e6_gate_cases/run_details/*.csv
```

shows:

| field | value |
| --- | --- |
| 3e6 case rows | 59 |
| ok rows | 47 |
| error rows | 12 `error:MemoryError:` |
| method among ok rows | `arac-lite-final` only |
| coordination_mode among ok rows | `arac_lite_rule` only |
| cc_pass_group_fes among ok rows | `20` only |
| fe_used among ok rows | `3000000` only |

Conclusion: the successful cached rows really are `ARAC-lite-final` with `cc_pass_group_fes=20`.

## 2. Current Gate Summary

This table uses direct case CSV scanning, not the stale/heavy aggregate relation-action audit.

| problem | ok seeds | missing seeds | best_mean | best_median | paper HCC mean | gap vs paper | median final/best | mean cc passes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4 | 10 | none | `9.655486e+08` | `6.880867e+08` | `2.260000e+07` | `+4172.339%` | `8.712` | `6302.4` |
| E6 | 10 | none | `1.095242e+09` | `6.049578e+08` | `4.320000e+07` | `+2435.282%` | `15.598` | `6214.1` |
| S4 | 3 | 4 5 6 7 8 9 10 | `3.240503e+06` | `2.881152e+06` | `1.240000e+03` | `+261230.884%` | `1.411` | `7002.7` |
| S6 | 10 | none | `3.781615e+06` | `3.951306e+06` | `6.650000e+04` | `+5586.639%` | `1.315` | `6997.5` |
| A6 | 4 | 5 6 7 8 9 10 | `8.058691e+04` | `8.073893e+04` | `7.800000e+04` | `+3.317%` | `1.037` | `7126.0` |
| R6 | 10 | none | `2.993188e+06` | `3.000060e+06` | `8.150000e+05` | `+267.262%` | `1.414` | `6793.2` |

The gate signal is negative if interpreted literally, but the protocol audit below shows that literal interpretation is not valid.

## 3. Metric Semantics

Relevant code:

- `HCC_SRC/HCC-ES.py:1200` builds run detail rows.
- `HCC_SRC/HCC-ES.py:1215` sets `final_fitness = curve[-1]`.
- `HCC_SRC/HCC-ES.py:1216` sets `best_fitness = np.min(curve)`.
- `HCC_SRC/HCC-ES.py:4170` stores `metadata["final_selected_fitness"] = current_best_fitness`.
- `HCC_SRC/HCC-ES.py:4479` overwrites `detail_row["final_fitness"]` with that final selected/current fitness.

Interpretation:

- `best_error` is the historical minimum over all full evaluations recorded in the run curve.
- `final_error` is the final selected/current state fitness, not guaranteed to equal historical best-so-far.
- For paper-style performance comparison, `best_error` is the closer metric.
- `final_error > best_error` is therefore expected when later optimization steps move the current state away from an earlier best evaluated point.

The median `final_error / best_error` ratios are large on E4/E6 especially (`8.712` and `15.598`), so final-state retention is a separate concern, but it should not replace `best_error` for paper-table comparison.

## 4. Long-Budget Protocol Fragmentation

Relevant code:

- `HCC_SRC/HCC-ES.py:550` computes the original global-stage FE budget as `0.2 + 0.8 * overlap_ratio`.
- `HCC_SRC/HCC-ES.py:3296` resolves CC subspace schedules.
- `HCC_SRC/HCC-ES.py:3312` computes the raw uniform subspace budget from remaining FE divided by group count.
- `HCC_SRC/HCC-ES.py:3313` reads `cc_pass_group_fes`.
- `HCC_SRC/HCC-ES.py:3315` caps every group budget with `min(raw_uniform_subfes, cc_pass_group_fes)`.
- `HCC_SRC/HCC-ES.py:1925` estimates ARAC-lite pass count from `group_count * cc_pass_group_fes`, plus validation FE when enabled.

At `TFEs=3,000,000`, the expected first-pass raw subspace budget is roughly `97,200` to `108,600` FEs per group, but the gate caps it to `20`.

| problem | group_count | overlap_vars | overlap_ratio | original_glofes | CC remaining | raw first-pass group FE | cap20 pass FE plus validation | estimated cap20 passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E4 | 20 | 95 | `0.095` | 828000 | 2172000 | 108600 | 401 | 5417 |
| E6 | 20 | 190 | `0.190` | 1056000 | 1944000 | 97200 | 401 | 4848 |
| S4 | 20 | 95 | `0.095` | 828000 | 2172000 | 108600 | 401 | 5417 |
| S6 | 20 | 190 | `0.190` | 1056000 | 1944000 | 97200 | 401 | 4848 |
| A6 | 20 | 190 | `0.190` | 1056000 | 1944000 | 97200 | 401 | 4848 |
| R6 | 20 | 190 | `0.190` | 1056000 | 1944000 | 97200 | 401 | 4848 |

Observed successful ARAC-lite-final runs actually reached about 5.4k to 7.1k passes. This means the long-budget run is dominated by thousands of tiny subspace restarts, not by paper-like long subspace optimization.

## 5. Baseline Calibration

I ran one small repo-default calibration:

```text
python HCC_SRC/HCC-ES.py --problems E4 --seeds 1 --tfes 3000000 --workers 1 --summary-refresh-every 1 --output-dir artifacts/arac_lite_final_3e6_short_audit_baselines/repo_default_e4_seed1
```

Result:

| metric | ARAC-lite-final cap20 E4 seed 1 | repo default HCC-ES E4 seed 1 | paper HCC E4 mean |
| --- | ---: | ---: | ---: |
| best | `7.552043576e+08` | `2.777980315e+07` | `2.260000000e+07` |
| final | `7.454588678e+09` | `6.387777734e+08` | n/a |
| best gap vs paper | `+3241.612%` | `+22.919%` | n/a |
| final/best | `9.871` | `22.994` | n/a |
| runtime seconds | `1440.3` | `410.2` | n/a |
| fe_used | `3000000` | `3000000` | n/a |

This is strong evidence that the current code can still reach the paper scale under repo-default long-budget HCC-ES, while the ARAC-lite-final gate protocol with `cc_pass_group_fes=20` cannot be compared directly.

## 6. Why final_error Can Be Much Worse Than best_error

There are two separate reasons:

1. Reporting semantics: `best_error` is historical best-so-far, while `final_error` is final current selected state.
2. Optimization dynamics: repeated subspace updates can move the current individual away from an earlier globally better evaluated state. The code records the historical best in the evaluation curve, but the final selected state is not forced to roll back to that historical best at the end.

The repo-default E4 calibration also shows this behavior: `best_fitness=2.777980315e+07`, but `final_fitness=6.387777734e+08`. So `final_error >> best_error` is not unique to ARAC-lite-final.

## Decision

Do not continue the current `cc_pass_group_fes=20` 3e6 gate matrix. It is too expensive and not paper-comparable.

Recommended next step:

1. Keep V0.7/V0.8 small-budget mechanism evidence as-is.
2. Define a separate long-budget evaluation protocol for final performance:
   - repo-default HCC-ES baseline,
   - no-coordination under the same long-budget protocol,
   - ARAC-lite-final with no tiny cap or with a scaled cap appropriate for 3e6.
3. First run 1-2 seeds on E4/S6/R6 before any full matrix.
4. Report `best_error` as the paper-comparison metric, with `final_error` as a state-retention diagnostic.

The current failed gate is best described as:

```text
ARAC-lite-final @ 3e6 with small-budget cc_pass_group_fes=20 is not a valid paper-comparable gate.
```

It should not be used to reject ARAC-lite-final as a method.
