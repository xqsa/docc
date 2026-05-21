# ARAC-lite V0.8 Mechanism Ablation

- 日期：2026-05-21
- 执行者：Codex
- 结论边界：冻结 V0.6 targeted-probe，不继续调 probe 阈值或动作规则。
- Problems: E4, E6, S4, S6, A6, R6
- Stress problems: R6, S6, E6
- TFEs: 5000, 10000, 20000
- Seeds: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- cc_pass_group_fes: 20
- workers: 12
- reuse_v0_7: True

## Main Matrix

- Runs: 900/900 ok
- Methods: no-coordination, validated-selective-conflict, arac-lite-v0.1-disable-fast, arac-lite-v0.6-targeted-probe, arac-lite-v0.6-random-probe-same-budget。
- Max absolute random budget gap: 0

## Targeted vs Same-Budget Random

| problem | paired | W/L/T | median_gap | worst_gap | IQR |
| --- | ---: | ---: | ---: | ---: | ---: |
| A6 | 30 | 0/0/30 | +0.000% | +0.000% | +0.000% |
| E4 | 30 | 0/2/28 | +0.000% | +0.071% | +0.000% |
| E6 | 30 | 4/2/24 | +0.000% | +0.004% | +0.000% |
| R6 | 30 | 2/1/27 | +0.000% | +0.000% | +0.000% |
| S4 | 30 | 1/1/28 | +0.000% | +0.150% | +0.000% |
| S6 | 30 | 4/1/25 | +0.000% | +0.880% | +0.000% |

## Probe Metrics

| problem | tfes | method | probes | matched | recovered | bad | delta | extra_fe |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A6 | 5000 | arac-lite-v0.6-random-probe-same-budget | 2 | 0 | 1 | 0 | -2.990053e+02 | +0.184% |
| A6 | 5000 | arac-lite-v0.6-targeted-probe | 2 | 0 | 0 | 0 | -3.181734e+02 | +0.184% |
| A6 | 10000 | arac-lite-v0.6-random-probe-same-budget | 6 | 0 | 3 | 0 | 2.121828e+01 | +0.167% |
| A6 | 10000 | arac-lite-v0.6-targeted-probe | 6 | 2 | 2 | 0 | 2.118887e+01 | +0.168% |
| A6 | 20000 | arac-lite-v0.6-random-probe-same-budget | 4 | 0 | 0 | 0 | -3.902102e+01 | +0.112% |
| A6 | 20000 | arac-lite-v0.6-targeted-probe | 4 | 1 | 1 | 0 | -7.392944e-01 | +0.111% |
| E4 | 5000 | arac-lite-v0.6-random-probe-same-budget | 1 | 0 | 1 | 0 | 3.838164e+06 | +0.152% |
| E4 | 5000 | arac-lite-v0.6-targeted-probe | 1 | 0 | 0 | 0 | -2.381950e+07 | +0.152% |
| E4 | 10000 | arac-lite-v0.6-random-probe-same-budget | 2 | 0 | 0 | 0 | -3.425047e+07 | +0.091% |
| E4 | 10000 | arac-lite-v0.6-targeted-probe | 2 | 1 | 0 | 0 | -5.461373e+07 | +0.086% |
| E4 | 20000 | arac-lite-v0.6-random-probe-same-budget | 0 | 0 | 0 | 0 | n/a | +0.044% |
| E4 | 20000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.044% |
| E6 | 5000 | arac-lite-v0.6-random-probe-same-budget | 2 | 0 | 2 | 0 | 1.639476e+08 | +0.152% |
| E6 | 5000 | arac-lite-v0.6-targeted-probe | 2 | 1 | 1 | 0 | -2.474863e+08 | +0.152% |
| E6 | 10000 | arac-lite-v0.6-random-probe-same-budget | 3 | 0 | 2 | 0 | -2.778865e+07 | +0.084% |
| E6 | 10000 | arac-lite-v0.6-targeted-probe | 3 | 2 | 1 | 0 | -2.164085e+07 | +0.081% |
| E6 | 20000 | arac-lite-v0.6-random-probe-same-budget | 2 | 0 | 0 | 0 | -1.063619e+07 | +0.051% |
| E6 | 20000 | arac-lite-v0.6-targeted-probe | 2 | 1 | 1 | 0 | -5.062947e+07 | +0.051% |
| R6 | 5000 | arac-lite-v0.6-random-probe-same-budget | 1 | 0 | 1 | 0 | 4.059554e+08 | +0.138% |
| R6 | 5000 | arac-lite-v0.6-targeted-probe | 1 | 1 | 1 | 0 | 3.807654e+07 | +0.138% |
| R6 | 10000 | arac-lite-v0.6-random-probe-same-budget | 0 | 0 | 0 | 0 | n/a | +0.070% |
| R6 | 10000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.070% |
| R6 | 20000 | arac-lite-v0.6-random-probe-same-budget | 2 | 0 | 1 | 1 | 1.064231e+04 | +0.044% |
| R6 | 20000 | arac-lite-v0.6-targeted-probe | 2 | 0 | 0 | 2 | -2.795264e+04 | +0.044% |
| S4 | 5000 | arac-lite-v0.6-random-probe-same-budget | 3 | 0 | 1 | 0 | -3.416896e+09 | +0.138% |
| S4 | 5000 | arac-lite-v0.6-targeted-probe | 3 | 0 | 1 | 0 | -4.061666e+09 | +0.138% |
| S4 | 10000 | arac-lite-v0.6-random-probe-same-budget | 0 | 0 | 0 | 0 | n/a | +0.071% |
| S4 | 10000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.071% |
| S4 | 20000 | arac-lite-v0.6-random-probe-same-budget | 0 | 0 | 0 | 0 | n/a | +0.047% |
| S4 | 20000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.047% |
| S6 | 5000 | arac-lite-v0.6-random-probe-same-budget | 2 | 0 | 2 | 0 | 5.088049e+09 | +0.154% |
| S6 | 5000 | arac-lite-v0.6-targeted-probe | 2 | 0 | 0 | 0 | -1.149520e+10 | +0.154% |
| S6 | 10000 | arac-lite-v0.6-random-probe-same-budget | 4 | 0 | 3 | 0 | 6.819060e+07 | +0.088% |
| S6 | 10000 | arac-lite-v0.6-targeted-probe | 4 | 4 | 1 | 0 | 1.507202e+07 | +0.090% |
| S6 | 20000 | arac-lite-v0.6-random-probe-same-budget | 0 | 0 | 0 | 0 | n/a | +0.039% |
| S6 | 20000 | arac-lite-v0.6-targeted-probe | 0 | 0 | 0 | 0 | n/a | +0.039% |

## Delta Gate Stress

- Stress runs: 180/180 ok

| problem | tfes | method | probes | matched | recovered | bad | delta | extra_fe |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E6 | 5000 | arac-lite-v0.6-accept-delta-recovery | 30 | 0 | 11 | 0 | -1.331595e+08 | +0.174% |
| E6 | 5000 | arac-lite-v0.6-accept-only-recovery | 30 | 0 | 14 | 0 | -4.237628e+07 | +0.176% |
| E6 | 10000 | arac-lite-v0.6-accept-delta-recovery | 60 | 0 | 20 | 0 | -2.531869e+07 | +0.148% |
| E6 | 10000 | arac-lite-v0.6-accept-only-recovery | 60 | 0 | 18 | 0 | -4.789886e+07 | +0.161% |
| E6 | 20000 | arac-lite-v0.6-accept-delta-recovery | 110 | 0 | 50 | 0 | -3.226395e+06 | +0.140% |
| E6 | 20000 | arac-lite-v0.6-accept-only-recovery | 110 | 0 | 44 | 0 | -1.109966e+07 | +0.146% |
| R6 | 5000 | arac-lite-v0.6-accept-delta-recovery | 30 | 0 | 12 | 18 | -2.018482e+08 | +0.168% |
| R6 | 5000 | arac-lite-v0.6-accept-only-recovery | 30 | 0 | 11 | 19 | -2.345286e+08 | +0.172% |
| R6 | 10000 | arac-lite-v0.6-accept-delta-recovery | 60 | 0 | 30 | 30 | -3.853446e+05 | +0.153% |
| R6 | 10000 | arac-lite-v0.6-accept-only-recovery | 60 | 0 | 30 | 30 | -4.188474e+05 | +0.157% |
| R6 | 20000 | arac-lite-v0.6-accept-delta-recovery | 110 | 0 | 38 | 72 | -3.271812e+04 | +0.139% |
| R6 | 20000 | arac-lite-v0.6-accept-only-recovery | 110 | 0 | 31 | 79 | -4.862827e+04 | +0.144% |
| S6 | 5000 | arac-lite-v0.6-accept-delta-recovery | 30 | 0 | 7 | 0 | -2.648118e+09 | +0.178% |
| S6 | 5000 | arac-lite-v0.6-accept-only-recovery | 30 | 0 | 8 | 0 | -3.174976e+09 | +0.178% |
| S6 | 10000 | arac-lite-v0.6-accept-delta-recovery | 60 | 0 | 30 | 0 | -3.804451e+06 | +0.154% |
| S6 | 10000 | arac-lite-v0.6-accept-only-recovery | 60 | 0 | 23 | 0 | -2.108159e+07 | +0.164% |
| S6 | 20000 | arac-lite-v0.6-accept-delta-recovery | 110 | 0 | 48 | 0 | -4.094786e+04 | +0.133% |
| S6 | 20000 | arac-lite-v0.6-accept-only-recovery | 110 | 0 | 42 | 0 | -7.605709e+04 | +0.139% |

## Delta Gate Recovery Metrics

| problem | tfes | method | candidates | recovery_fusion | accepted | bad | delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| E6 | 5000 | arac-lite-v0.6-accept-delta-recovery | 146 | 84 | 27 | 0 | -9.242131e+07 |
| E6 | 5000 | arac-lite-v0.6-accept-only-recovery | 567 | 328 | 150 | 0 | 2.250830e+06 |
| E6 | 10000 | arac-lite-v0.6-accept-delta-recovery | 937 | 584 | 146 | 0 | -1.059759e+08 |
| E6 | 10000 | arac-lite-v0.6-accept-only-recovery | 2495 | 1517 | 534 | 0 | -8.202504e+07 |
| E6 | 20000 | arac-lite-v0.6-accept-delta-recovery | 785 | 436 | 206 | 0 | -6.086272e+06 |
| E6 | 20000 | arac-lite-v0.6-accept-only-recovery | 2959 | 1780 | 703 | 0 | -1.183587e+07 |
| R6 | 5000 | arac-lite-v0.6-accept-delta-recovery | 132 | 64 | 15 | 49 | -4.402172e+08 |
| R6 | 5000 | arac-lite-v0.6-accept-only-recovery | 742 | 428 | 64 | 364 | -8.207886e+08 |
| R6 | 10000 | arac-lite-v0.6-accept-delta-recovery | 299 | 186 | 53 | 133 | -2.532784e+06 |
| R6 | 10000 | arac-lite-v0.6-accept-only-recovery | 463 | 295 | 88 | 207 | -2.144099e+06 |
| R6 | 20000 | arac-lite-v0.6-accept-delta-recovery | 772 | 492 | 151 | 341 | -4.456258e+04 |
| R6 | 20000 | arac-lite-v0.6-accept-only-recovery | 1554 | 976 | 195 | 781 | -7.675106e+04 |
| S6 | 5000 | arac-lite-v0.6-accept-delta-recovery | 191 | 101 | 14 | 0 | -1.055940e+10 |
| S6 | 5000 | arac-lite-v0.6-accept-only-recovery | 834 | 486 | 93 | 0 | -7.223336e+09 |
| S6 | 10000 | arac-lite-v0.6-accept-delta-recovery | 218 | 135 | 59 | 0 | -7.544445e+06 |
| S6 | 10000 | arac-lite-v0.6-accept-only-recovery | 2481 | 1543 | 398 | 0 | -7.064896e+07 |
| S6 | 20000 | arac-lite-v0.6-accept-delta-recovery | 827 | 534 | 127 | 0 | -1.641775e+05 |
| S6 | 20000 | arac-lite-v0.6-accept-only-recovery | 2178 | 1354 | 408 | 0 | -3.184133e+05 |

## Artifacts

- main run details: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_run_details.csv`
- main relation-action audit: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_relation_action_audit.csv`
- main summary: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_summary.csv`
- main robustness: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_robustness.csv`
- main probe metrics: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_probe_metrics.csv`
- budget alignment: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_budget_alignment.csv`
- stress run details: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_delta_stress_run_details.csv`
- stress robustness: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_delta_stress_robustness.csv`
- stress recovery metrics: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_delta_stress_recovery_metrics.csv`
- report: `C:/Users/83718/Desktop/HCC/HCC-main/artifacts/arac_lite_v0_8_mechanism_ablation_report.md`
