# ProjectB Stage 1: GRPO-original, GDPO-current, and GRPO-noKL

## Scope

All three runs are independent 35-step Stage 1 runs from RL_INIT_V1, using the frozen formal_validation_80 endpoint at steps 0/7/14/21/28/35. Primary metrics are raw per-example validation scores over 80 rows. Sensitivity metrics remove the pre-registered source_id 3814 row from the same 80 outputs, yielding 79 rows; there is no resampling.

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
- GDPO-current versus GRPO-noKL is informative for training-effect comparison, but still differs in advantage construction and reward dimensions; it is not a single-variable causal comparison.
- Results are single-seed and 35-step Stage 1 evidence only; they do not establish 70/105-step behavior or generalization.

## Evidence

- GRPO report: reports/grpo/stage1_35/grpo_stage1_report.md
- GDPO report: reports/gdpo/stage1_35/gdpo_stage1_report.md
- noKL report: reports/grpo/stage1_35/grpo_nokl_stage1_report.md
- canonical metrics: reports/grpo/grpo_stage1_metrics_canonical.json, reports/gdpo/gdpo_stage1_metrics_canonical.json, reports/grpo/grpo_nokl_stage1_metrics_canonical.json
- paired bootstrap: reports/comparison/paired_bootstrap_stage1_35.json
