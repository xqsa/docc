# ARAC-lite V0.7 Fixed Candidate Generalization

- 日期：2026-05-21
- 执行者：Codex
- 结论边界：冻结 V0.6 targeted-probe，不继续调 probe 阈值或动作规则。
- Problems: E4, E6, S4, S6, A6, R6
- Seeds: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- TFEs: 5000, 10000, 20000
- cc_pass_group_fes: 20
- workers: 12
- resume: True
- case cache: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_cases`

## Matrix

- Runs: 900/900 ok
- Methods: no-coordination, validated-selective-conflict, arac-lite-v0.1-disable-fast, arac-lite-v0.6-targeted-probe, arac-lite-v0.6-no-delta-hard-block。
- Ablation: no-delta-hard-block 放宽 targeted relation delta 类 hard gate，用于判断 delta rule 是否必要。

## Acceptance Snapshot

- vs disable-fast majority non-worse problems: 6/6。
- max extra FE ratio: +0.184%。
- V0.6 Fusion count: 32832。
- R6 bad_probe_count: 2。

## Paired Robustness

下表是 paired win/loss/tie 汇总，gap 使用同 problem/tfes/seed 下 candidate 相对 baseline 的 best_error。

| problem | tfes | candidate | baseline | paired | W/L/T | non_worse | median_gap | worst_gap | IQR |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A6 | all | arac-lite-v0.6-targeted-probe | arac-lite-v0.1-disable-fast | 30 | 0/0/30 | 1.000 | +0.000% | +0.000% | +0.000% |
| A6 | all | arac-lite-v0.6-targeted-probe | no-coordination | 30 | 0/24/6 | 0.200 | +0.036% | +0.563% | +0.086% |
| A6 | all | arac-lite-v0.6-targeted-probe | validated-selective-conflict | 30 | 0/1/29 | 0.967 | +0.000% | +0.053% | +0.000% |
| E4 | all | arac-lite-v0.6-targeted-probe | arac-lite-v0.1-disable-fast | 30 | 0/2/28 | 0.933 | +0.000% | +0.002% | +0.000% |
| E4 | all | arac-lite-v0.6-targeted-probe | no-coordination | 30 | 13/16/1 | 0.467 | +0.030% | +2.397% | +1.010% |
| E4 | all | arac-lite-v0.6-targeted-probe | validated-selective-conflict | 30 | 20/10/0 | 0.667 | -0.175% | +1.688% | +1.405% |
| E6 | all | arac-lite-v0.6-targeted-probe | arac-lite-v0.1-disable-fast | 30 | 3/2/25 | 0.933 | +0.000% | +0.004% | +0.000% |
| E6 | all | arac-lite-v0.6-targeted-probe | no-coordination | 30 | 10/19/1 | 0.367 | +0.150% | +3.048% | +0.775% |
| E6 | all | arac-lite-v0.6-targeted-probe | validated-selective-conflict | 30 | 16/14/0 | 0.533 | -0.001% | +1.976% | +0.672% |
| R6 | all | arac-lite-v0.6-targeted-probe | arac-lite-v0.1-disable-fast | 30 | 0/1/29 | 0.967 | +0.000% | +0.008% | +0.000% |
| R6 | all | arac-lite-v0.6-targeted-probe | no-coordination | 30 | 10/18/2 | 0.400 | +0.074% | +4.979% | +0.439% |
| R6 | all | arac-lite-v0.6-targeted-probe | validated-selective-conflict | 30 | 19/11/0 | 0.633 | -0.142% | +3.959% | +1.636% |
| S4 | all | arac-lite-v0.6-targeted-probe | arac-lite-v0.1-disable-fast | 30 | 2/0/28 | 1.000 | +0.000% | +0.000% | +0.000% |
| S4 | all | arac-lite-v0.6-targeted-probe | no-coordination | 30 | 13/12/5 | 0.600 | +0.000% | +4.186% | +0.883% |
| S4 | all | arac-lite-v0.6-targeted-probe | validated-selective-conflict | 30 | 19/11/0 | 0.633 | -0.239% | +3.181% | +1.284% |
| S6 | all | arac-lite-v0.6-targeted-probe | arac-lite-v0.1-disable-fast | 30 | 2/0/28 | 1.000 | +0.000% | +0.000% | +0.000% |
| S6 | all | arac-lite-v0.6-targeted-probe | no-coordination | 30 | 8/21/1 | 0.300 | +0.303% | +3.063% | +1.342% |
| S6 | all | arac-lite-v0.6-targeted-probe | validated-selective-conflict | 30 | 14/15/1 | 0.500 | +0.002% | +3.071% | +1.572% |

## Probe Metrics

| problem | tfes | method | targeted | matched | recovered | bad | delta | extra_fe |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A6 | 5000 | arac-lite-v0.6-no-delta-hard-block | 2 | 0 | 0 | 0 | -3.181734e+02 | +0.184% |
| A6 | 5000 | arac-lite-v0.6-targeted-probe | 2 | 0 | 0 | 0 | -3.181734e+02 | +0.184% |
| A6 | 10000 | arac-lite-v0.6-no-delta-hard-block | 6 | 2 | 2 | 0 | 2.118887e+01 | +0.168% |
| A6 | 10000 | arac-lite-v0.6-targeted-probe | 6 | 2 | 2 | 0 | 2.118887e+01 | +0.168% |
| A6 | 20000 | arac-lite-v0.6-no-delta-hard-block | 4 | 1 | 1 | 0 | -7.392944e-01 | +0.111% |
| A6 | 20000 | arac-lite-v0.6-targeted-probe | 4 | 1 | 1 | 0 | -7.392944e-01 | +0.111% |
| E4 | 5000 | arac-lite-v0.6-no-delta-hard-block | 1 | 0 | 0 | 0 | -2.381950e+07 | +0.152% |
| E4 | 5000 | arac-lite-v0.6-targeted-probe | 1 | 0 | 0 | 0 | -2.381950e+07 | +0.152% |
| E4 | 10000 | arac-lite-v0.6-no-delta-hard-block | 2 | 1 | 0 | 0 | -5.461373e+07 | +0.086% |
| E4 | 10000 | arac-lite-v0.6-targeted-probe | 2 | 1 | 0 | 0 | -5.461373e+07 | +0.086% |
| E4 | 20000 | arac-lite-v0.6-no-delta-hard-block | 0 | 0 | 0 | 0 | n/a | +0.044% |
| E4 | 20000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.044% |
| E6 | 5000 | arac-lite-v0.6-no-delta-hard-block | 2 | 1 | 1 | 0 | -2.474863e+08 | +0.152% |
| E6 | 5000 | arac-lite-v0.6-targeted-probe | 2 | 1 | 1 | 0 | -2.474863e+08 | +0.152% |
| E6 | 10000 | arac-lite-v0.6-no-delta-hard-block | 3 | 2 | 1 | 0 | -2.164085e+07 | +0.081% |
| E6 | 10000 | arac-lite-v0.6-targeted-probe | 3 | 2 | 1 | 0 | -2.164085e+07 | +0.081% |
| E6 | 20000 | arac-lite-v0.6-no-delta-hard-block | 2 | 1 | 1 | 0 | -5.062947e+07 | +0.051% |
| E6 | 20000 | arac-lite-v0.6-targeted-probe | 2 | 1 | 1 | 0 | -5.062947e+07 | +0.051% |
| R6 | 5000 | arac-lite-v0.6-no-delta-hard-block | 1 | 1 | 1 | 0 | 3.807654e+07 | +0.138% |
| R6 | 5000 | arac-lite-v0.6-targeted-probe | 1 | 1 | 1 | 0 | 3.807654e+07 | +0.138% |
| R6 | 10000 | arac-lite-v0.6-no-delta-hard-block | 0 | 0 | 0 | 0 | n/a | +0.070% |
| R6 | 10000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.070% |
| R6 | 20000 | arac-lite-v0.6-no-delta-hard-block | 2 | 0 | 0 | 2 | -2.795264e+04 | +0.044% |
| R6 | 20000 | arac-lite-v0.6-targeted-probe | 2 | 0 | 0 | 2 | -2.795264e+04 | +0.044% |
| S4 | 5000 | arac-lite-v0.6-no-delta-hard-block | 3 | 0 | 1 | 0 | -4.061666e+09 | +0.138% |
| S4 | 5000 | arac-lite-v0.6-targeted-probe | 3 | 0 | 1 | 0 | -4.061666e+09 | +0.138% |
| S4 | 10000 | arac-lite-v0.6-no-delta-hard-block | 0 | 0 | 0 | 0 | n/a | +0.071% |
| S4 | 10000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.071% |
| S4 | 20000 | arac-lite-v0.6-no-delta-hard-block | 0 | 0 | 0 | 0 | n/a | +0.047% |
| S4 | 20000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.047% |
| S6 | 5000 | arac-lite-v0.6-no-delta-hard-block | 2 | 0 | 0 | 0 | -1.149520e+10 | +0.154% |
| S6 | 5000 | arac-lite-v0.6-targeted-probe | 2 | 0 | 0 | 0 | -1.149520e+10 | +0.154% |
| S6 | 10000 | arac-lite-v0.6-no-delta-hard-block | 4 | 4 | 1 | 0 | 1.507202e+07 | +0.090% |
| S6 | 10000 | arac-lite-v0.6-targeted-probe | 4 | 4 | 1 | 0 | 1.507202e+07 | +0.090% |
| S6 | 20000 | arac-lite-v0.6-no-delta-hard-block | 0 | 0 | 0 | 0 | n/a | +0.039% |
| S6 | 20000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.039% |

## Artifacts

- run details: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_run_details.csv`
- relation-action audit: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_relation_action_audit.csv`
- summary: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_summary.csv`
- paired robustness: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_robustness.csv`
- rank summary: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_rank_summary.csv`
- action distribution: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_action_distribution.csv`
- probe metrics: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_probe_metrics.csv`
- report: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_7_generalization_report.md`
