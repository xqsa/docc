# Verification

## 2026-05-20 HCC-Fidelity Patch V1.1: Group Order Role Decoupling

Current verification status for separating CMA-ES execution order from adjacent coordination order:

- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_hcc_fidelity_patch_v1_1_artifacts.py` passes.
- Focused V1.1 tests pass after red-green implementation:
  - `test_group_order_plan_can_decouple_execution_and_coordination_orders`
  - `test_coordination_edges_wait_until_both_groups_have_run`
  - `test_decoupled_coordination_blend_uses_both_endpoint_proposals`
  - `test_run_hcc_core_decouples_rddsm_execution_from_aob_coordination_order`
- `python -m pytest tests/test_hcc_es_none_guard.py tests/test_info_aware_nda.py -q` passes with `59` tests.
- `python -m pytest -q` passes with `83` tests.
- `python scripts\generate_hcc_fidelity_patch_v1_1_artifacts.py --problems E4 E6 S4 S6 A6 R6 --seeds 1 2 3 4 5 --tfes 1000 5000 10000` completes with `720/720` runs in `ok` status.

Generated V1.1 artifacts:

- `artifacts/hcc_fidelity_patch_v1_1_visibility.csv`
- `artifacts/hcc_fidelity_patch_v1_1_run_details.csv`
- `artifacts/hcc_fidelity_patch_v1_1_summary.csv`
- `artifacts/hcc_fidelity_patch_v1_1_report.md`

Key V1.1 result:

- `rddsm-exec-aob-coord-*` proves execution order and coordination visibility can be decoupled: execution visibility remains `0.000..0.105`, while coordination visibility becomes `1.000`.
- This decoupling does not produce a stable performance win over old RDDSM. At `TFEs=10000`, `rddsm-exec-aob-coord-original` has gap vs old of A6 `-0.001%`, E4 `+1.414%`, E6 `+1.170%`, R6 `+2.923%`, S4 `+2.102%`, S6 `+3.724%`.
- `aob-exec-rddsm-coord-original` is often much better than AOB full order, so V1's degradation is strongly tied to high-visibility AOB adjacent coordination itself, not only to AOB execution order.
- `equation8_correct` is not a safe default under the current non-persistent `omega_i` implementation; it frequently worsens the restored-coordination variants.
- V1.1 run details and summary include `group_delta_mean`, `positive_delta_rate`, `early_delta_mean`, and `early_positive_delta_rate` for group-delta distribution and early subspace improvement auditing.

## 2026-05-20 HCC-Fidelity Patch V1

Current verification status for topology-preserving group ordering and Equation 8 correct blend:

- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_hcc_fidelity_patch_v1_artifacts.py` passes.
- `python -m pytest tests/test_hcc_es_none_guard.py tests/test_info_aware_nda.py -q` passes with `55` tests.
- `python scripts\generate_hcc_fidelity_patch_v1_artifacts.py --problems E4 E6 S4 S6 A6 R6 --seeds 1 2 3 4 5 --tfes 1000 5000 10000` completes with `450/450` runs in `ok` status.
- `python -m pytest -q` passes with `79` tests.

Generated V1 artifacts:

- `artifacts/hcc_fidelity_patch_v1_visibility.csv`
- `artifacts/hcc_fidelity_patch_v1_run_details.csv`
- `artifacts/hcc_fidelity_patch_v1_summary.csv`
- `artifacts/hcc_fidelity_patch_v1_report.md`

Key V1 result:

- `aob_topology` restores adjacent overlap visibility to `1.000` for all audited problems.
- RDDSM order remains `0.000` for `E4/S4` and `0.105` for `E6/S6/A6/R6`.
- The implementation now supports `overlap_blend_mode=equation8_correct`, where the current group value is weighted by `current_delta / (previous_delta + current_delta)`.
- In the completed 1000/5000/10000 FE matrix, topology order improves the paper-fidelity visibility condition but mostly worsens best/final fitness versus old RDDSM order and no-coordination; this means the fidelity repair is structurally active but not yet a performance win.

## 2026-05-20 Coordination Selector

Current verification status for the stage-level coordination selector:

- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_coordination_selector_artifacts.py` passes.
- `python -m pytest tests/test_info_aware_nda.py -k coordination_selector -q` passes with `2` tests.
- `python -m pytest tests/test_info_aware_nda.py tests/test_hcc_es_none_guard.py -q` passes with `50` tests.
- `python -m pytest -q` passes with `74` tests.
- `python scripts\generate_coordination_selector_artifacts.py` completes the requested matrix with `120/120` runs in `ok` status.

Generated selector artifacts:

- `artifacts/coordination_selector_run_details.csv`
- `artifacts/coordination_selector_trace.csv`
- `artifacts/coordination_selector_report.md`

Key selector result:

- R6 automatically closes coordination in `10/10` selector runs.
- A6 keeps validated coordination in `5/10` selector runs.
- E6 keeps validated coordination in `4/10` selector runs.
- Maximum validation extra FE ratio is `0.001`.
- Current 1000/5000 FE protocol records only one selector trace row per run, so the selector makes a decision but has no later recorded pass where off/on can materially diverge from always-on validated coordination.

## 2026-05-20 Coordination Selector Multi-pass Continuation

Current multi-pass verification status:

- `python scripts\generate_coordination_selector_artifacts.py --tfes 10000 --cc-pass-group-fes 20` completes with `45/45` runs in `ok` status.
- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_coordination_selector_artifacts.py` passes.
- `python -m pytest -q` passes with `75` tests.

Generated selector artifacts now reflect the `10000` FE multi-pass continuation:

- `artifacts/coordination_selector_run_details.csv`
- `artifacts/coordination_selector_trace.csv`
- `artifacts/coordination_selector_report.md`

Key selector result:

- R6 automatically closes coordination in `5/5` selector runs.
- A6 automatically closes coordination in `5/5` selector runs.
- E6 keeps validated coordination in `1/5` selector runs.
- Selector run `cc_pass_count` is `17..23`; `post_probe_pass_count` is `16..22`.
- Maximum validation extra FE ratio is `0.002`.
- Under this multi-pass protocol, selector materially protects R6 by matching no-coordination and avoiding always-on validated degradation, but it also closes A6 completely and preserves E6 only sparsely.

This workspace has been rolled back to the original `hcc_es_original` implementation.

Current verification status:

- `python -m py_compile HCC_SRC/HCC-ES.py HCC_SRC/run_pypop7_baselines.py HCC_SRC/experiment_protocols.py` passes.
- `python -m unittest discover -s tests -p 'test_*.py' -v` passes with `26` tests.
- `python HCC_SRC/HCC-ES.py --problems E4 --seeds 1 --tfes 20 --method hcc_es_original --output-dir tmp/hcc-rollback-smoke` completes successfully and writes `summary.csv`, `run_details.csv`, and `diagnostics.csv`.

Removed from the runnable codebase:

- `CHCFR`
- `OHR`
- `OHBCD`
- `U-OHBCD`
- risk-gated ownership variants
- graph-risk-gated variants

Historical experiment outputs may still exist under `HCC_SRC/result/` as archived artifacts, but they are no longer part of the active implementation surface.

## 2026-05-20 HCC Paper-Fidelity Audit

Current verification status:

- `python -m py_compile scripts\generate_hcc_paper_fidelity_audit.py` passes.
- `python scripts\generate_hcc_paper_fidelity_audit.py --skip-protocol-runs` passes.
- `python scripts\generate_hcc_paper_fidelity_audit.py --protocol-problems A6 E6 R6 --seeds 1 2 3 4 5 --tfes 10000` completes with `30/30` protocol audit runs in `ok` status.
- `python -m pytest -q` passes with `75` tests.

Generated audit artifacts:

- `artifacts/hcc_paper_fidelity_group_order_audit.csv`
- `artifacts/hcc_paper_fidelity_protocol_audit.csv`
- `artifacts/hcc_paper_fidelity_protocol_summary.csv`
- `artifacts/hcc_paper_fidelity_audit.md`

Key audit result:

- AOB natural order has adjacent overlap visibility `1.000` for all audited problems.
- RDDSM order has adjacent overlap visibility `0.000` for `E4/S4` and `0.105` for `E6/S6/A6/R6`.
- Original adjacent Equation 8 blend direction is not paper-faithful.
- Persistent per-subspace `omega_i` synchronization is not implemented.
- Paper-like protocol has no post-probe room, while selector-friendly protocol creates post-probe passes.

## 2026-05-20 HCC-Fidelity V2: omega_i / optimizer-state synchronization

Current V2 verification status:

- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_hcc_fidelity_patch_v2_omega_artifacts.py` passes.
- `python -m pytest tests/test_info_aware_nda.py tests/test_hcc_es_none_guard.py -q` passes with `62` tests.
- `python -m pytest -q` passes with `86` tests.
- `python scripts\generate_hcc_fidelity_patch_v2_omega_artifacts.py --problems A6 E6 R6 --seeds 1 2 3 4 5 --tfes 10000` completes with `90/90` runs in `ok` status.
- Artifact sanity-check passes:
  - `run_rows=90`
  - `ok_rows=90`
  - `state_rows=7500`
  - `summary_rows=18`
  - `persistent_runs=15`
  - `persistent_sync_rows=5700`

Generated V2 artifacts:

- `artifacts/hcc_fidelity_patch_v2_omega_run_details.csv`
- `artifacts/hcc_fidelity_patch_v2_omega_state_audit.csv`
- `artifacts/hcc_fidelity_patch_v2_omega_summary.csv`
- `artifacts/hcc_fidelity_patch_v2_omega_report.md`
- `artifacts/hcc_fidelity_patch_v2_omega_runs/`

Key V2 result:

- `optimizer_state_mode` now supports default `ephemeral` and switchable `persistent_mean`.
- `persistent_mean` stores a per-source-group mean cache and synchronizes coordinated overlap variables into cached group means only; sigma, covariance, and evolution paths remain intentionally ephemeral.
- State audit confirms the synchronization mechanism is active: persistent_mean runs record `omega_sync_count=380` and `mean_coord_mismatch_after_sync_max=0.0`.
- In the current 10000 FE paper-like protocol, all V2 methods have `cc_pass_count_mean=1.0`. This means synchronized omega/mean values have no later subspace optimizer initialization where they can change search dynamics.
- Therefore the identical best/final means between `rddsm-exec-aob-coord-eq8-correct` and `rddsm-exec-aob-coord-eq8-correct-persistent-mean` should not be interpreted as state synchronization being ineffective. The cleaner interpretation is that the single-pass protocol gives optimizer-state synchronization no downstream opportunity.
- R6 remains best under `no-coordination`, supporting a future relation-to-action mapping where noisy or conflict-heavy overlap relations can map to Disable / Freeze instead of forced coordination.

## 2026-05-20 HCC-Fidelity V2.1: controlled multi-pass probe

Current V2.1 verification status:

- `python -m py_compile scripts\generate_hcc_fidelity_v2_1_multipass_probe.py` passes.
- `python scripts\generate_hcc_fidelity_v2_1_multipass_probe.py --problems A6 E6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20` completes with `45/45` runs in `ok` status.
- Artifact sanity-check passes: `rows=45; bad_status=0; cc_pass_le_1=0; persistent_post_sync_le_0=0`.
- `python -m pytest tests\test_info_aware_nda.py tests\test_hcc_es_none_guard.py -q` passes with `62` tests.

Generated V2.1 artifacts:

- `artifacts/hcc_fidelity_v2_1_multipass_run_details.csv`
- `artifacts/hcc_fidelity_v2_1_multipass_state_audit.csv`
- `artifacts/hcc_fidelity_v2_1_multipass_summary.csv`
- `artifacts/hcc_fidelity_v2_1_multipass_report.md`
- `artifacts/hcc_fidelity_v2_1_multipass_runs/`

Key V2.1 result:

- The protocol now gives persistent_mean a real downstream opportunity: `cc_pass_count_mean=17.0` and persistent runs have `post_sync_init_count_mean=320.0`.
- State absorption is confirmed: persistent runs have `omega_sync_count_mean=6460.0`, nonzero `mean_init_shift_after_sync`, and `mean_coord_mismatch_after_sync_max_mean=0.0`.
- persistent_mean is not a performance win in this controlled multi-pass probe. It ties ephemeral Eq8 on A6 best_mean, worsens A6 final_mean, and worsens E6/R6 best/final.
- R6 remains strongly in favor of `no-coordination`: persistent_mean is `+23.033%` worse than no-coordination on best_mean and `+11.537%` worse than ephemeral Eq8.
- Interpretation: optimizer mean synchronization is not the main bottleneck as a universal repair. It can faithfully propagate coordination state, but if the relation action is harmful, state consistency amplifies or preserves the wrong action. This supports returning to ARAC-lite relation-to-action mapping, especially Disable/Freeze for R6-like relations.

## 2026-05-20 ARAC-lite V0: relation-to-action audit

Current ARAC-lite V0 verification status:

- `python scripts\generate_arac_lite_v0_artifacts.py --problems A6 E6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20` completes with `60/60` runs in `ok` status.
- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_arac_lite_v0_artifacts.py` passes.
- `python -m pytest tests\test_info_aware_nda.py tests\test_hcc_es_none_guard.py -k "arac_lite" -q` passes with `5` tests.
- `python -m pytest tests\test_info_aware_nda.py tests\test_hcc_es_none_guard.py -q` passes with `67` tests.
- `python -m pytest -q` passes with `91` tests.

Generated ARAC-lite V0 artifacts:

- `artifacts/arac_lite_v0_run_details.csv`
- `artifacts/arac_lite_v0_relation_action_audit.csv`
- `artifacts/arac_lite_v0_summary.csv`
- `artifacts/arac_lite_v0_report.md`
- `artifacts/arac_lite_v0_runs/`

Key ARAC-lite V0 result:

- Relation-action audit contains `107350` rows.
- R6 is mostly mapped away from Fusion: `Fusion=2053`, `Freeze=1177`, `Disable=13870`.
- R6 best_mean gap vs no-coordination is `+0.786%`, compared with eq8-correct `+10.307%` and validated-selective-conflict `+1.377%`.
- E6 best_mean gap vs no-coordination is `+0.805%`, slightly behind validated-selective-conflict `+0.666%` but better than eq8-correct `+3.495%`.
- A6 ARAC-lite best_mean matches validated-selective-conflict and is `+0.120%` vs no-coordination.
- Interpretation: V0 supports the ARAC hypothesis. The main bottleneck is relation-to-action selection, not stronger or more visible coordination. The first rule version protects R6, but still leaves tuning room for E6/A6 and for further reducing harmful R6 Fusion.

## 2026-05-20 ARAC-lite V0.1: attribution and threshold sweep

Current ARAC-lite V0.1 verification status:

- `python scripts\generate_arac_lite_v0_1_artifacts.py --skip-sweep` completes and generates `18` action-attribution rows.
- `python scripts\generate_arac_lite_v0_1_artifacts.py --problems A6 E6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20` completes with `90/90` runs in `ok` status.
- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_arac_lite_v0_1_artifacts.py` passes.
- `python -m pytest tests\test_arac_lite_v0_1.py -q` passes with `2` tests.
- `python -m pytest tests\test_info_aware_nda.py tests\test_hcc_es_none_guard.py tests\test_arac_lite_v0_1.py -q` passes with `69` tests.
- `python -m pytest -q` passes with `93` tests.

Generated ARAC-lite V0.1 artifacts:

- `artifacts/arac_lite_v0_1_action_attribution.csv`
- `artifacts/arac_lite_v0_1_run_details.csv`
- `artifacts/arac_lite_v0_1_relation_action_audit.csv`
- `artifacts/arac_lite_v0_1_sweep_summary.csv`
- `artifacts/arac_lite_v0_1_report.md`
- `artifacts/arac_lite_v0_1_runs/`

Key ARAC-lite V0.1 result:

- V0 action attribution shows R6 Fusion is low quality: accept rate `0.114`, reject rate `0.886`.
- V0 action attribution shows A6 has a small Disable false-negative signal: `0.032`.
- `arac-lite-v0.1-disable-fast` is the best current rule candidate because it preserves some Fusion while improving all main gaps:
  - R6 gap vs no-coordination improves from V0 `+0.786%` to `+0.203%`.
  - E6 gap vs no-coordination improves from V0 `+0.805%` to `+0.561%`, better than validated-selective-conflict `+0.666%`.
  - A6 best_mean stays tied with V0 and validated-selective-conflict.
- `arac-lite-v0.1-balanced` and `arac-lite-v0.1-fusion-strict` drive Fusion to `0`; they are useful conservative diagnostics, but too close to no-coordination to serve as the primary ARAC module.

## 2026-05-20 ARAC-lite V0.4: Fusion Recovery audit and conservative gate

Current ARAC-lite V0.4 verification status:

- `python scripts\generate_arac_lite_v0_4_recovery_artifacts.py --offline-only` completes and generates `54` offline recovery audit rows.
- `python scripts\generate_arac_lite_v0_4_recovery_artifacts.py --problems E6 S6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20` completes with `60/60` runs in `ok` status.
- Artifact sanity-check passes:
  - run details rows: `60`; ok rows: `60`.
  - summary rows: `12`.
  - relation-action audit rows: `151050`.
  - V0.4 relation rows: `51300`.
  - V0.4 recovery candidate rows: `0`.
  - recovery blocking reasons: `phase_not_recoverable=17100`, `rolling_attempts_below_min=34200`.
- `python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_arac_lite_v0_4_recovery_artifacts.py` passes.
- `python -m pytest tests\test_arac_lite_v0_4.py tests\test_info_aware_nda.py tests\test_hcc_es_none_guard.py -q` passes with `71` tests.
- `python -m pytest -q` passes with `104` tests.

Generated ARAC-lite V0.4 artifacts:

- `artifacts/arac_lite_v0_4_recovery_offline_audit.csv`
- `artifacts/arac_lite_v0_4_recovery_run_details.csv`
- `artifacts/arac_lite_v0_4_recovery_relation_action_audit.csv`
- `artifacts/arac_lite_v0_4_recovery_action_audit.csv`
- `artifacts/arac_lite_v0_4_recovery_summary.csv`
- `artifacts/arac_lite_v0_4_recovery_robustness.csv`
- `artifacts/arac_lite_v0_4_recovery_phase_summary.csv`
- `artifacts/arac_lite_v0_4_recovery_report.md`

Key V0.4 result:

- Offline audit confirms that `S6 middle` is the only strong recovery window among the inspected signals: Fusion accept rate `0.629`, fusion delta mean `2.872092e+06`, positive-delta rate `0.629`, recovery candidates `23`.
- `R6 middle` shows why accept rate alone is unsafe: accept rate `0.286`, but fusion delta mean `-3.211746e+06`.
- `E6 middle/late` has some accept-rate signal, but fusion delta remains negative, so it should not be recovered by a broad fixed rule.
- Online V0.4 does not improve over `arac-lite-v0.1-disable-fast` because no recovery candidate is triggered. E6/S6/R6 all have gap vs disable-fast `+0.000%`.
- This is a useful negative result: a conservative gate without exploration/probing cannot recover relations once disable-fast stops producing enough rolling Fusion attempts. The next small step should be a low-frequency recovery probe, not UCB and not global threshold tuning.
## 2026-05-20 ARAC-lite V0.2 fixed-rule generalization

- 执行者：Codex
- 范围：冻结 `arac-lite-v0.1-disable-fast`，对比 `no-coordination`、`eq8-correct`、`validated-selective-conflict`、`arac-lite-v0`、`arac-lite-v0.1-disable-fast`。
- 协议：`E4/E6/S4/S6/A6/R6`，`TFEs=10000`，`seeds=1..5`，`cc_pass_group_fes=20`。

### 本地验证命令与结果

```powershell
python -m py_compile scripts\generate_arac_lite_v0_2_artifacts.py
```

结果：通过。

```powershell
python -m pytest tests\test_arac_lite_v0_2.py -q
```

结果：通过，`3 passed in 3.05s`。

```powershell
python scripts\generate_arac_lite_v0_2_artifacts.py --problems E4 E6 S4 S6 A6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20
```

结果：通过，`150/150` run 状态为 `ok`。

```powershell
python -m pytest -q
```

结果：通过；记录补充后再次执行的新鲜验证为 `96 passed in 12.00s`。

### Artifact 检查

- `artifacts/arac_lite_v0_2_run_details.csv`: `150` rows，`bad=0`。
- `artifacts/arac_lite_v0_2_summary.csv`: `30` rows。
- `artifacts/arac_lite_v0_2_action_audit.csv`: `53` rows。
- `artifacts/arac_lite_v0_2_relation_action_audit.csv`: `269040` rows。
- held-out candidate rows: `3`。
- held-out Fusion count: `2383`。

### 结论

`arac-lite-v0.1-disable-fast` 在 held-out `E4/S4/S6` 上没有崩：相对 V0 为 `2/3` non-worse，相对 validated 为 `2/3` non-worse，且 held-out 平均 `gap_vs_no=-0.184%`。Fusion 没有被清零，Disable false-negative 在 held-out 上为 `0.0`，因此本轮支持把 disable-fast 作为第一个候选主版本继续推进到更大预算与更多 seeds。

## 2026-05-20 ARAC-lite V0.3 robustness and pass dynamics

- 执行者：Codex
- 范围：冻结 `arac-lite-v0.1-disable-fast`，比较 `no-coordination`、`validated-selective-conflict`、`arac-lite-v0`、`arac-lite-v0.1-disable-fast`。
- 协议：`E4/E6/S4/S6/A6/R6`，`TFEs=5000/10000/20000`，`seeds=1..10`，`cc_pass_group_fes=20`。

### 本地验证命令与结果

```powershell
python -m py_compile scripts\generate_arac_lite_v0_3_artifacts.py
```

结果：通过。

```powershell
python -m pytest tests\test_arac_lite_v0_3.py -q
```

结果：通过，报告修正后 `4 passed`。

```powershell
python scripts\generate_arac_lite_v0_3_artifacts.py --problems E4 E6 S4 S6 A6 R6 --seeds 1 2 3 4 5 6 7 8 9 10 --tfes 5000 10000 20000 --cc-pass-group-fes 20 *> tmp\arac_lite_v0_3_full.log
```

结果：通过，`720/720` run 状态为 `ok`。

```powershell
python -m pytest -q
```

结果：通过，`100 passed in 12.31s`。

### Artifact 检查

- `artifacts/arac_lite_v0_3_run_details.csv`: `720` rows，`bad=0`。
- 四方法各 `180` run。
- `artifacts/arac_lite_v0_3_relation_action_audit.csv`: `1873970` rows。
- `artifacts/arac_lite_v0_3_robustness.csv`: `72` rows。
- `artifacts/arac_lite_v0_3_phase_summary.csv`: `446` rows。

### 结论

`arac-lite-v0.1-disable-fast` 相对 V0 和 validated 都满足稳健性验收：两者均为 `6/6` problems 达到 >=50% paired non-worse。相对 no-coordination 只有 `1/6`，说明 fixed rule 仍不是“全局优于无协调”，但它稳定优于前一版关系动作映射与 validated baseline。S6 仍是系统性观察点：vs no median gap `+0.430%`，vs validated median gap `+0.002%`，vs V0 median gap `-0.008%`。

阶段动态审计显示 E6/R6 middle/late Fusion accept 有上升，但样本量小且 action_delta 多为负；S6 middle 有 recovery 信号但 late 不稳定。因此当前不建议直接进入非平稳 UCB，更适合保留固定规则主线，同时单独审计 S6/E6 的中后期 Fusion recovery。

## 2026-05-20 ARAC-lite V0.5 low-frequency recovery probe

- 执行者：Codex
- 范围：实现 V0.5 low-frequency recovery probe，并在 `E6/S6/R6`、`TFEs=10000`、`seeds=1..5`、`cc_pass_group_fes=20` 上对比四方法。

### 本地验证命令与结果

```powershell
python -m pytest tests\test_arac_lite_v0_5.py -q
```

结果：最终 focused V0.5 测试通过，`8 passed in 2.25s`。测试覆盖配置归一化、middle probe、early 禁止 probe、probe conflict freeze 不计入实际 probe、probe evidence recovery、probe metrics 和 report acceptance。

```powershell
python -m pytest tests\test_arac_lite_v0_5.py tests\test_arac_lite_v0_4.py tests\test_info_aware_nda.py tests\test_hcc_es_none_guard.py -q
```

结果：通过，`79 passed in 4.57s`。

```powershell
python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_arac_lite_v0_5_probe_artifacts.py scripts\generate_arac_lite_v0_1_artifacts.py
```

结果：通过，无编译错误。

```powershell
python scripts\generate_arac_lite_v0_5_probe_artifacts.py --problems E6 S6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20
```

结果：通过，正式矩阵 `60/60` run 状态为 `ok`。

```powershell
python -m pytest -q
```

结果：通过，`112 passed in 9.58s`。

### Artifact 检查

- `artifacts/arac_lite_v0_5_probe_run_details.csv`: `60` rows，`60` ok。
- `artifacts/arac_lite_v0_5_probe_summary.csv`: `12` rows。
- `artifacts/arac_lite_v0_5_probe_relation_action_audit.csv`: `148200` rows。
- `artifacts/arac_lite_v0_5_probe_metrics.csv`: `3` rows。
- `artifacts/arac_lite_v0_5_probe_report.md`: 包含 Acceptance 段。

### 关键结果

- 成本验收：通过；max extra FE ratio `0.136%`，低于 `1%`。
- 收益验收：不通过。
- E6：V0.5 best_mean `2.118185e+12`，gap vs no-coordination `+0.672%`，gap vs disable-fast `+0.111%`；probe_count `150`，probe_accept_rate `0.300`，probe_delta_mean `-6.404831e+07`。
- S6：V0.5 best_mean `3.309731e+11`，gap vs no-coordination `+0.843%`，gap vs disable-fast `+0.472%`；probe_count `150`，probe_accept_rate `0.367`，probe_delta_mean `-1.547079e+07`，bad_recovery_count `1`。
- R6：V0.5 best_mean `3.062238e+10`，gap vs no-coordination `+0.936%`，gap vs disable-fast `+0.731%`；probe_count `150`，probe_accept_rate `0.633`，probe_delta_mean `-1.315273e+05`，R6_bad_probe_count `55`。

### 结论

V0.5 是有价值的负结果。它证明低频主动 probe 可以在低成本下恢复在线 evidence，但当前固定恢复门控仍会误放坏关系：S6 没有恢复，R6 被拉坏。下一步应做 failure audit，而不是直接加大 probe 频率、放宽阈值或上 UCB。
# 2026-05-20 ARAC-lite V0.5 Failure Audit Verification

- 执行者：Codex
- 任务：生成 V0.5 failure audit，不修改主算法。

## 本地验证

```text
python -m pytest tests/test_arac_lite_v0_5_failure_audit.py
```

结果：3 passed in 0.02s。

```text
python scripts/generate_arac_lite_v0_5_failure_audit.py
```

结果：

```text
offline candidates -> 44
online probes -> 450
match rows -> 35
bias rows -> 71
attribution rows -> 195
report -> C:\Users\83718\Desktop\HCC\HCC-main\artifacts\v0_5_failure_audit_report.md
```

```text
python -m py_compile scripts/generate_arac_lite_v0_5_failure_audit.py
```

结果：exit 0。

```text
python -m pytest tests/test_arac_lite_v0_5_failure_audit.py tests/test_arac_lite_v0_5.py tests/test_arac_lite_v0_4.py
```

结果：15 passed in 2.45s。

```text
python -m pytest
```

结果：115 passed in 10.22s。

## 审计产物

- `artifacts/offline_online_candidate_match.csv`
- `artifacts/probe_selection_bias.csv`
- `artifacts/validation_delta_attribution.csv`
- `artifacts/v0_5_failure_audit_report.md`

## 风险评估

- 本次仅新增审计脚本与报告产物，不修改 ARAC-lite 主算法，因此运行时行为风险低。
- 现有 relation_id 在 ARAC-lite history 中等价于 var_id；group_pair 由 proposal trace 派生，用于诊断分布，不作为主匹配键。

# 2026-05-20 ARAC-lite V0.6 Targeted Recovery Probe Verification

- 执行者：Codex
- 任务：实现并验证 V0.6 targeted recovery probe。

## 本地验证

```text
python -m pytest tests/test_arac_lite_v0_6.py tests/test_arac_lite_v0_5_failure_audit.py -q
```

结果：10 passed in 2.32s。

```text
python scripts/generate_arac_lite_v0_6_targeted_probe_artifacts.py --problems E6 S6 R6 --seeds 1 2 3 4 5 --tfes 10000 --cc-pass-group-fes 20 *> .codex/arac_lite_v0_6_matrix.log
```

结果：

```text
completed 75/75 runs
relation-action audit rows -> 199310
targeted metric rows -> 3
report -> C:\Users\83718\Desktop\HCC\HCC-main\artifacts\arac_lite_v0_6_targeted_probe_report.md
```

```text
python -m py_compile HCC_SRC/HCC-ES.py HCC_SRC/HCC/info_aware_nda.py scripts/generate_arac_lite_v0_6_targeted_probe_artifacts.py scripts/generate_arac_lite_v0_5_failure_audit.py
```

结果：通过，无编译错误。

```text
python -m pytest tests/test_arac_lite_v0_6.py tests/test_arac_lite_v0_5_failure_audit.py tests/test_arac_lite_v0_5.py tests/test_info_aware_nda.py tests/test_hcc_es_none_guard.py -q
```

结果：85 passed in 4.52s。

```text
python -m pytest -q
```

结果：122 passed in 10.04s。

## 验收结果

- 机制验收：通过。
- S6 middle matched_probe_count：`3`，从 V0.5 的 `0` 提升为 `>0`。
- S6 targeted_probe_delta_mean：`2.375493e+07`，优于 V0.5 random probe `-1.547079e+07`。
- R6 bad_probe_count：`0`，未上升且低于 V0.5 的 `55`。
- max extra_fe_ratio：`0.098%`，低于 `1%`。
- 性能验收：通过；S6 gap vs disable-fast `-0.121%`，R6 `+0.000%`，E6 `-0.034%`。

## 产物

- `artifacts/arac_lite_v0_6_targeted_probe_run_details.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_summary.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_relation_action_audit.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_action_audit.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_phase_summary.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_metrics.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_offline_match.csv`
- `artifacts/arac_lite_v0_6_targeted_probe_report.md`

## 风险评估

- V0.6 小矩阵仍限于 E6/S6/R6、TFEs=10000、seeds=1..5；结论是当前 targeted probe 策略在这一组验证上的通过结果。
- targeted probe 使用 pass-level validation delta 作为在线反馈；该口径与 V0.5 保持一致，便于比较，但仍不是变量独立贡献估计。

# 2026-05-21 ARAC-lite V0.7 Fixed Candidate Generalization Verification

- 执行者：Codex
- 任务：冻结 V0.6 targeted-probe，新增 V0.7 泛化验证和 no-delta-hard-block 消融。

## 本地验证

```text
python -m pytest tests\test_arac_lite_v0_7.py -q
```

结果：先 RED 后 GREEN；最终 `6 passed in 2.51s`。

```text
python -m py_compile scripts\generate_arac_lite_v0_7_generalization.py
```

结果：通过，无编译错误。

```text
python scripts\generate_arac_lite_v0_7_generalization.py --problems S6 R6 --seeds 1 2 --tfes 5000 --cc-pass-group-fes 20 *> .codex\arac_lite_v0_7_smoke.log
```

结果：

```text
completed 20/20 runs
relation-action audit rows -> 29070
probe metric rows -> 4
report -> C:\Users\83718\Desktop\HCC\HCC-main\artifacts\arac_lite_v0_7_generalization_report.md
```

```text
python -m pytest tests\test_arac_lite_v0_7.py tests\test_arac_lite_v0_6.py -q
```

结果：`13 passed in 2.55s`。

```text
python -m pytest -q
```

结果：`128 passed in 9.83s`。

## 产物

- `scripts/generate_arac_lite_v0_7_generalization.py`
- `tests/test_arac_lite_v0_7.py`
- `artifacts/arac_lite_v0_7_generalization_run_details.csv`
- `artifacts/arac_lite_v0_7_generalization_relation_action_audit.csv`
- `artifacts/arac_lite_v0_7_generalization_summary.csv`
- `artifacts/arac_lite_v0_7_generalization_robustness.csv`
- `artifacts/arac_lite_v0_7_generalization_rank_summary.csv`
- `artifacts/arac_lite_v0_7_generalization_action_distribution.csv`
- `artifacts/arac_lite_v0_7_generalization_probe_metrics.csv`
- `artifacts/arac_lite_v0_7_generalization_report.md`

## 风险评估

- 本次只运行 S6/R6、seeds 1..2、TFEs=5000 的 smoke 矩阵，用于验证脚本与指标链路；尚未执行用户要求的完整 6 problems x 3 TFEs x 10 seeds 正式矩阵。
- no-delta-hard-block 消融已放宽 relation_delta_mean 与 positive_delta_rate 两个 delta 类 gate，但仍保留 phase/support/std/accept-rate 条件，避免退化为无选择随机 probe。

# 2026-05-21 ARAC-lite V0.7 Efficiency Verification

- 执行者：Codex
- 任务：提高 V0.7 固定候选泛化验证的运行效率。

## 本地验证

```text
python -m pytest tests\test_arac_lite_v0_7.py -q
```

结果：先 RED 后 GREEN；最终 `8 passed in 2.74s`。RED 原因是 `build_case_tasks` / `run_tasks` 尚不存在，修复后覆盖 case 展开、resume skip 和 executor 路径。

```text
python -m py_compile scripts\generate_arac_lite_v0_7_generalization.py
```

结果：通过，无编译错误。

```text
python -m pytest tests\test_arac_lite_v0_7.py tests\test_arac_lite_v0_6.py -q
```

结果：`15 passed in 2.34s`。

```text
Measure-Command { python scripts\generate_arac_lite_v0_7_generalization.py --problems S6 R6 --seeds 1 2 --tfes 5000 --cc-pass-group-fes 20 --workers 4 *> .codex\arac_lite_v0_7_parallel_smoke.log } | Select-Object TotalSeconds
```

结果：

```text
TotalSeconds: 27.63
V0.7 requested cases=20 pending=20 workers=4 resume=False
completed 20/20 runs
relation-action audit rows -> 29070
probe metric rows -> 4
```

```text
Measure-Command { python scripts\generate_arac_lite_v0_7_generalization.py --problems S6 R6 --seeds 1 2 --tfes 5000 --cc-pass-group-fes 20 --workers 4 --resume *> .codex\arac_lite_v0_7_resume_smoke.log } | Select-Object TotalSeconds
```

结果：

```text
TotalSeconds: 12.11
V0.7 requested cases=20 pending=0 workers=4 resume=True
completed 20/20 runs
relation-action audit rows -> 29070
probe metric rows -> 4
```

```text
python -m pytest -q
```

结果：`130 passed in 9.55s`。

## 产物

- `scripts/generate_arac_lite_v0_7_generalization.py`
- `tests/test_arac_lite_v0_7.py`
- `artifacts/arac_lite_v0_7_generalization_cases/run_details/`
- `artifacts/arac_lite_v0_7_generalization_cases/relation_action_audit/`
- `.codex/arac_lite_v0_7_parallel_smoke.log`
- `.codex/arac_lite_v0_7_resume_smoke.log`

## 风险评估

- 本次仍只用 S6/R6、seeds 1..2、TFEs=5000 做提效 smoke；它证明调度链路、缓存和 resume 可用，不构成正式 V0.7 泛化结论。
- `--workers` 使用进程并行，Windows 上不建议盲目拉满；正式矩阵建议先使用 `--workers 4 --resume`，资源余量足够再尝试 `6`。
- `--resume` 只跳过 `status == ok` 的 case；若某个 case 写出 error 状态，会在下一轮作为 pending 重跑。

# 2026-05-21 ARAC-lite V0.7 Full Matrix Verification

- 执行者：Codex
- 任务：按用户要求使用 `workers=12` 跑正式 V0.7 全矩阵并汇总结果。

## 正式命令

```text
python scripts\generate_arac_lite_v0_7_generalization.py --problems E4 E6 S4 S6 A6 R6 --seeds 1 2 3 4 5 6 7 8 9 10 --tfes 5000 10000 20000 --cc-pass-group-fes 20 --workers 12 --resume *> .codex\arac_lite_v0_7_full_workers12.log
```

## 运行结果

```text
V0.7 requested cases=900 pending=880 workers=12 resume=True
completed 900/900 runs
relation-action audit rows -> 2509425
probe metric rows -> 36
report -> C:\Users\83718\Desktop\HCC\HCC-main\artifacts\arac_lite_v0_7_generalization_report.md
```

- `.codex\arac_lite_v0_7_full_workers12.err.log`: 0 字节。
- `artifacts/arac_lite_v0_7_generalization_run_details.csv`: 900 rows，900 ok。
- `artifacts/arac_lite_v0_7_generalization_summary.csv`: 90 rows。
- `artifacts/arac_lite_v0_7_generalization_robustness.csv`: 144 rows。
- `artifacts/arac_lite_v0_7_generalization_rank_summary.csv`: 120 rows。
- `artifacts/arac_lite_v0_7_generalization_action_distribution.csv`: 215 rows。
- `artifacts/arac_lite_v0_7_generalization_probe_metrics.csv`: 36 rows。

## 关键结果

- vs disable-fast：6/6 problems majority non-worse。
- vs validated：A6/E4/E6/R6/S4 majority non-worse，S6 为 `14/15/1`，non-worse `0.500`。
- vs no-coordination：不要求全面赢，本轮也未全面赢；S4 non-worse `0.600`，E4 `0.467`，R6 `0.400`，E6 `0.367`，S6 `0.300`，A6 `0.200`。
- V0.6 totals：Fusion `32,832`，Freeze `37,320`，Disable `562,833`。
- Extra FE：candidate max `+0.184%`，mean `+0.101%`，低于 `1%`。
- Probe totals：targeted `34`，matched `13`，recovered `9`，bad `2`。
- R6 bad_probe_count：`2`，不是 V0.6 小矩阵的 `0`，但整体仍低；vs disable-fast 为 `0/1/29`，non-worse `0.967`，median gap `0`，worst gap `+0.008%`。
- Ablation：no-delta-hard-block 与主方法在 900 个 case 上无差异，`diff_cases=0`，本轮不能证明 delta gate 必要性。

## 结论

V0.7 泛化验证支持“V0.6-targeted-probe 可作为稳定主候选”：它相对 disable-fast 在 6 个问题上多数 non-worse，extra FE 远低于 1%，Fusion 没有清零，并且 R6 坏 probe 保持极低。主要 caveat 是 no-delta-hard-block 消融未产生分叉，需要换更敏感的消融或补 random-probe-same-budget 才能证明 delta gate / targeted selection 的必要性。

# 2026-05-21 ARAC-lite V0.8 Mechanism Ablation Verification

- 执行者：Codex
- 任务：冻结 V0.6 targeted-probe，补 `same-budget random probe` 和 `accept-only vs accept+delta` stress test。

## RED/GREEN 与本地验证

```text
python -m pytest tests\test_arac_lite_v0_8.py -q
```

RED：`7 failed`，失败原因是 random same-budget 配置字段、算法参数和 V0.8 生成器尚不存在。

GREEN：

```text
7 passed in 2.47s
```

```text
python -m pytest tests\test_arac_lite_v0_8.py tests\test_arac_lite_v0_7.py tests\test_arac_lite_v0_6.py -q
```

结果：

```text
24 passed in 2.95s
```

```text
python -m py_compile HCC_SRC\HCC-ES.py HCC_SRC\HCC\info_aware_nda.py scripts\generate_arac_lite_v0_8_mechanism_ablation.py
```

结果：通过，无编译错误。

## Smoke

```text
python scripts\generate_arac_lite_v0_8_mechanism_ablation.py --problems S6 R6 --stress-problems S6 R6 --seeds 1 2 --tfes 5000 --cc-pass-group-fes 20 --workers 4 --resume *> .codex\arac_lite_v0_8_smoke.log
```

结果：

```text
main 20/20 ok
stress 8/8 ok
budget target total 1, random total 1, max_abs_gap 0
```

## 正式 V0.8 全矩阵

```text
python scripts\generate_arac_lite_v0_8_mechanism_ablation.py --problems E4 E6 S4 S6 A6 R6 --stress-problems R6 S6 E6 --seeds 1 2 3 4 5 6 7 8 9 10 --tfes 5000 10000 20000 --cc-pass-group-fes 20 --workers 12 --resume *> .codex\arac_lite_v0_8_full_workers12.log
```

结果：

```text
completed main 900/900 runs
main relation-action audit rows -> 2509615
main probe metric rows -> 36
stress runs -> 180
```

CSV 审计：

```text
artifacts/arac_lite_v0_8_mechanism_ablation_run_details.csv: 900 rows, 900 ok
artifacts/arac_lite_v0_8_delta_stress_run_details.csv: 180 rows, 180 ok
artifacts/arac_lite_v0_8_mechanism_ablation_budget_alignment.csv: 30 rows
budget target total 34, random total 34, max_abs_gap 0, nonzero_gaps 0
```

## 关键结果

Targeted vs same-budget random，按 problem 聚合到 all TFEs：

```text
A6  paired 30  W/L/T 0/0/30  median +0.000%  worst +0.000%
E4  paired 30  W/L/T 0/2/28  median +0.000%  worst +0.071%
E6  paired 30  W/L/T 4/2/24  median +0.000%  worst +0.004%
R6  paired 30  W/L/T 2/1/27  median +0.000%  worst +0.000%
S4  paired 30  W/L/T 1/1/28  median +0.000%  worst +0.150%
S6  paired 30  W/L/T 4/1/25  median +0.000%  worst +0.880%
```

Probe totals：

```text
targeted: probes 34, matched 13, recovered 9, bad 2, extra max +0.184%, extra mean +0.101%
random:   probes 34, matched 0,  recovered 17, bad 1, extra max +0.184%, extra mean +0.102%
```

Delta stress recovery totals：

```text
accept+delta: candidates 4307, fusion 2616, accepted 798, R6 bad_recovery 523
accept-only:  candidates 14273, fusion 8707, accepted 2633, R6 bad_recovery 1352
```

R6 recovery by TFE：

```text
5000  accept+delta cand 132 fusion 64  accepted 15  bad 49   delta -4.402172e+08
5000  accept-only  cand 742 fusion 428 accepted 64  bad 364  delta -8.207886e+08
10000 accept+delta cand 299 fusion 186 accepted 53  bad 133  delta -2.532784e+06
10000 accept-only  cand 463 fusion 295 accepted 88  bad 207  delta -2.144099e+06
20000 accept+delta cand 772 fusion 492 accepted 151 bad 341  delta -4.456258e+04
20000 accept-only  cand 1554 fusion 976 accepted 195 bad 781 delta -7.675106e+04
```

## 结论

V0.8 的 random-probe-same-budget 消融没有支持“targeted selection 明显优于同预算随机 probe”的强结论；在 final best-error 的 paired 口径上多数 case 打平，random 甚至有更多 recovered probe。这个结果应作为 caveat 写入论文线索：V0.6 的稳定性成立，但 targeted selection 的必要性证据不强。

delta gate stress test 更有说服力：accept-only 会显著放大 recovery candidate / recovery fusion 数量，并在 R6 上把 bad recovery 从 `523` 推到 `1352`。因此当前机制必要性证据更支持“delta gate 是防止高 accept/负 delta 虚假恢复的关键保护信号”。

# 2026-05-21 ARAC-lite Final Freeze

- 执行者：Codex
- 目标：非破坏式整理项目，冻结主方法为 `ARAC-lite: Delta-gated Relation-to-Action Coordination`。

## 验证命令与结果

```text
python -m py_compile scripts\final\run_arac_lite_final.py scripts\final\generate_final_evidence_package.py
```

结果：通过，无编译错误。

```text
PYTHONPATH=HCC_SRC; load configs/final/*.json through HCC-ES.py load_info_aware_nda_config
```

结果：

```text
arac-lite-final.json: enable=True mode=arac_lite_rule cc_pass_group_fes=20 cc_min_passes=3
disable-fast.json: enable=True mode=arac_lite_rule cc_pass_group_fes=20 cc_min_passes=3
no-coordination.json: enable=False mode=no_coordination cc_pass_group_fes=20 cc_min_passes=3
validated-selective-conflict.json: enable=True mode=selective_hypergraph_pass_end cc_pass_group_fes=20 cc_min_passes=3
```

```text
python scripts\final\generate_final_evidence_package.py
```

结果：

```text
manifest csv -> C:\Users\83718\Desktop\HCC\HCC-main\artifacts\final_evidence\MANIFEST.csv
manifest md -> C:\Users\83718\Desktop\HCC\HCC-main\artifacts\final_evidence\MANIFEST.md
copied/referenced rows -> 25
```

```text
python -m pytest -q
```

结果：

```text
139 passed in 12.88s
```

## 验证结论

- `configs/final/` 配置均可由现有 dataclass loader 读取。
- `scripts/final/` wrapper 和 evidence packager 可编译。
- `artifacts/final_evidence/` 已生成轻量证据包；超大 relation-action audit 在 manifest 中引用原路径。
- 全量测试通过。
- 本轮未删除文件、未移动 V0.7/V0.8 产物、未修改算法默认行为。
# 2026-05-21 ARAC-lite-final 3e6 Short Audit

- 执行者：Codex
- 任务：暂停继续跑 3e6 gate，审计 ARAC-lite-final 配置真实性、`best_error` / `final_error` 口径、`cc_pass_group_fes=20` multi-pass 长预算碎片化、`final_error` 远大于 `best_error` 原因，并做小量 baseline/protocol 对照。
- 进程检查：`Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell' -and $_.CommandLine -match 'run_arac_lite_3e6_gate|arac_lite_final_3e6' }`，结果为空，确认没有残留 gate runner。
- Case cache scan：直接扫描 `artifacts/arac_lite_final_3e6_gate_cases/run_details/*.csv`，得到 `59` 个 3e6 rows，其中 `47` 个 `ok`、`12` 个 `error:MemoryError:`；47 个 ok rows 均为 `method=arac-lite-final`、`coordination_mode=arac_lite_rule`、`cc_pass_group_fes=20`、`fe_used=3000000`。
- 指标口径检查：`HCC_SRC/HCC-ES.py` 中 `build_run_detail_row()` 使用 `np.min(curve)` 作为 `best_fitness`，后续将 `metadata["final_selected_fitness"]` 覆盖到 `final_fitness`；因此 `best_error` 更接近 paper comparison 的 best-so-far，`final_error` 是 final selected/current state。
- 调度口径检查：`cc_pass_group_fes=20` 在 3e6 下把 raw first-pass group FE 约 `97200` 到 `108600` cap 成 `20`；observed successful runs 的 `cc_pass_count` 范围约 `5431` 到 `7126`。
- Baseline calibration：执行 `python HCC_SRC\HCC-ES.py --problems E4 --seeds 1 --tfes 3000000 --workers 1 --summary-refresh-every 1 --output-dir artifacts\arac_lite_final_3e6_short_audit_baselines\repo_default_e4_seed1`，exit code `0`；结果 `best_fitness=2.777980315e+07`、`final_fitness=6.387777734e+08`、`runtime=410.2396s`。
- 对照结论：同一 E4 seed=1 下，ARAC-lite-final cap20 的 `best_error=7.552043576e+08`，repo-default HCC-ES 的 `best_fitness=2.777980315e+07`，paper HCC E4 mean 为 `2.26e+07`；说明当前 cap20 3e6 gate 不是 paper-comparable 协议，不应继续扩大或据此否定 ARAC-lite-final。
- 新增产物：`artifacts/arac_lite_final_3e6_short_audit.md`、`artifacts/arac_lite_final_3e6_short_audit_summary.csv`、`artifacts/arac_lite_final_3e6_short_audit_baselines/repo_default_e4_seed1/run_details.csv`、`artifacts/arac_lite_final_3e6_short_audit_baselines/repo_default_e4_seed1/summary.csv`。
