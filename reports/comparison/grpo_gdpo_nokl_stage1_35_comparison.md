# ProjectB Stage 1: GRPO-original, GDPO-current, and GRPO-noKL

## Scope

All three runs are independent 35-step Stage 1 runs from RL_INIT_V1, using the frozen formal_validation_80 endpoint at steps 0/7/14/21/28/35. Primary metrics are raw per-example validation scores over 80 rows. Sensitivity metrics are a post-hoc sensitivity analysis using a fixed source_id 3814 exclusion rule applied identically across algorithms and steps to the same 80 outputs, yielding 79 rows; there is no resampling.

## Primary validation curve: 80 rows

| step | GRPO total | GDPO total | noKL total | GRPO accuracy | GDPO accuracy | noKL accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 2.400171 | 2.400171 | 1.450171 | 1.450171 | 1.450171 |
| 7 | 2.717582 | 2.611385 | 2.588245 | 1.767582 | 1.648885 | 1.638245 |
| 14 | 2.799781 | 2.798260 | 2.757760 | 1.849781 | 1.835760 | 1.807760 |
| 21 | 2.785379 | 2.768737 | 2.784451 | 1.835379 | 1.806237 | 1.821951 |
| 28 | 2.831808 | 2.781237 | 2.781237 | 1.881808 | 1.818737 | 1.818737 |
| 35 | 2.799721 | 2.820522 | 2.818737 | 1.849721 | 1.858022 | 1.856237 |

| step | GRPO strict | GDPO strict | noKL strict | GRPO tool parse | GDPO tool parse | noKL tool parse | GRPO wrapper | GDPO wrapper | noKL wrapper |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76 (95.00%) | 76 (95.00%) | 76 (95.00%) | 73 (91.25%) | 73 (91.25%) | 73 (91.25%) | 5 (6.25%) | 5 (6.25%) | 5 (6.25%) |
| 7 | 76 (95.00%) | 77 (96.25%) | 76 (95.00%) | 74 (92.50%) | 74 (92.50%) | 73 (91.25%) | 5 (6.25%) | 4 (5.00%) | 5 (6.25%) |
| 14 | 76 (95.00%) | 77 (96.25%) | 76 (95.00%) | 74 (92.50%) | 75 (93.75%) | 74 (92.50%) | 5 (6.25%) | 4 (5.00%) | 5 (6.25%) |
| 21 | 76 (95.00%) | 77 (96.25%) | 77 (96.25%) | 74 (92.50%) | 75 (93.75%) | 75 (93.75%) | 5 (6.25%) | 4 (5.00%) | 4 (5.00%) |
| 28 | 76 (95.00%) | 77 (96.25%) | 77 (96.25%) | 74 (92.50%) | 75 (93.75%) | 75 (93.75%) | 5 (6.25%) | 4 (5.00%) | 4 (5.00%) |
| 35 | 76 (95.00%) | 77 (96.25%) | 77 (96.25%) | 74 (92.50%) | 75 (93.75%) | 75 (93.75%) | 5 (6.25%) | 4 (5.00%) | 4 (5.00%) |

Format reward is included in the machine-readable JSON and in each run's report; it remains an output-contract metric, not a training reward mean.

## Sensitivity validation curve: same 79 rows

The 79-row sensitivity is derived offline by removing source_id 3814 (validation manifest row index 56) from each corresponding persisted output file. The complete values are in the machine-readable comparison JSON. At step 35, raw total reward is: GRPO 2.809844, GDPO 2.830909, noKL 2.829100.

## Training diagnostics

- actor loss: GRPO-original: mean 0.012533, range -0.008118..0.036012; GDPO-current: mean -0.003111, range -0.017813..0.005842; GRPO-noKL: mean 0.007041, range -0.007567..0.033520
- grad norm: GRPO-original: mean 0.411523, range 0.329626..0.617412; GDPO-current: mean 0.595946, range 0.461149..0.897989; GRPO-noKL: mean 0.314941, range 0.217709..0.616645
- actor PPO KL: GRPO-original: mean 0.000032, range -0.000057..0.000214; GDPO-current: mean 0.000058, range -0.000072..0.000487; GRPO-noKL: mean 0.000032, range -0.000064..0.000274
- step time (s): GRPO-original: mean 1089.529834, range 1073.699350..1118.507512; GDPO-current: mean 1078.800352, range 1055.502614..1111.230258; GRPO-noKL: mean 905.334667, range 880.475636..941.109704

These are diagnostics from persisted training logs. They are not the primary cross-algorithm endpoint. NoKL has no reference-policy path under both KL flags false, so wall-clock and GPU-efficiency comparisons against the KL-enabled runs are not interpreted as fair algorithm comparisons.

## Pairwise interpretation

- GRPO-original versus GDPO-current retains the known KL-treatment and reward-dimension confound; it cannot support a pure advantage-estimator causal conclusion.
- GRPO-original versus GRPO-noKL is the KL ablation comparison, with dataset/order/seed/batch/rollout/learning rate/lengths and validation frozen. Removing reward-side KL also removes the reference-policy runtime path under the audited official v0.9.1 semantics.
- GDPO-current versus GRPO-noKL aligns objective-level reward-side KL treatment. It compares complete advantage constructions: separate component normalization, aggregation and batch whitening versus aggregate normalization. Those are method differences, not evidence isolating per-dimension normalization alone; reference-policy runtime still differs.
- Results are single-seed and 35-step Stage 1 evidence only; they do not establish 70/105-step behavior or generalization.

## Evidence

- Run reports: [original](../grpo/stage1_35/grpo_stage1_report.md), [GDPO](../gdpo/stage1_35/gdpo_stage1_report.md), [noKL canonical report](../grpo_nokl/stage1_35/grpo_nokl_stage1_report.md).
- Canonical metrics: [original](../grpo/grpo_stage1_metrics_canonical.json), [GDPO](../gdpo/gdpo_stage1_metrics_canonical.json), [noKL](../grpo/grpo_nokl_stage1_metrics_canonical.json).
- [Paired bootstrap](paired_bootstrap_stage1_35.json) and [80/79 sensitivity](primary80_vs_sensitivity79_stage1_35.md).

## Current interpretation and project closeout (2026-10-03)

All three observed step35 scores exceed their common initialization. Original/noKL has no clear separable final validation benefit in the archived paired interval; this is not proof of no KL effect. GDPO/noKL endpoints are nearly identical, so these data do not establish a clear downstream advantage for GDPO. Equal or close scores do not establish equal outputs, weights or optimization trajectories.

**Direct evidence:** the sampled pilots show reward variation; [shared-rollout records and correction](grpo_gdpo_shared_rollout_diagnostic.md) show magnitude/centering/sample-weighting changes but no substantive within-group ranking disagreement under 1e-8; [fresh-holdout records](../sft/fresh_holdout_reward_variation.md) contain 8 opposing-centered-sign trajectories in 6/32 groups. Formal greedy validation format is 95%–96.25%; remaining total-score movement mostly comes from accuracy reward.

**Plausible hypothesis, not an established causal explanation:** greedy validation format is near its observed ceiling, and reward conflict may not be frequent/strong enough in this setting for the changed advantage construction to create a clear endpoint gain. Conflict frequency/strength during formal training was not systematically measured; sampled training-format saturation is not established. The eight-prompt diagnostic supplies no gradient or performance measurement.

The primary80 results and archived bootstrap JSON are preserved. Sensitivity79 is post-hoc and its fixed exclusion does not materially change the current interpretation. This documentation revision cross-checks public aggregates but does not independently recompute the unavailable formal-validation JSONL bootstrap.

The [closeout decision](../../docs/experiment_design/stage1_closeout_decision.md) now freezes the current project's training budget at 35; this is not convergence. The [final-test protocol](../../docs/experiment_design/final_test_protocol.md) fixes RL_INIT_V1 and the three step35 models and is NOT EXECUTED. No server or GPU task is started by this update.
