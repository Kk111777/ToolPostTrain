# GRPO vs GDPO Stage 1: 35-step comparison

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Original two-run comparison, superseded as the current result overview by the three-run report. A KL-aligned noKL run has since completed; the historical A/B confound remains. Original facts and values below are preserved. Current artifacts: [three-run completion](final_stage1_ablation_completion_report.md); current budget: [35-step decision](../../docs/experiment_design/stage1_closeout_decision.md).

## Scope

- Both algorithms start independently from RL_INIT_V1.
- Both use the same 3901 eligible-prompt pool, source order, seed, 512-prompt batch, rollout.n=4, five epochs, and 35 optimizer steps.
- Both use the same greedy n=1 formal_validation_80 at steps 0/7/14/21/28/35.
- test.parquet was not used for intermediate validation.
- This is a controlled Stage 1 comparison, not a full 70/105-step reproduction.

## Validation learning curve

| step | GRPO total | GDPO total | GRPO format | GDPO format | GRPO accuracy | GDPO accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 2.400171 | 0.950000 | 0.950000 | 1.450171 | 1.450171 |
| 7 | 2.717582 | 2.611385 | 0.950000 | 0.962500 | 1.767582 | 1.648885 |
| 14 | 2.799781 | 2.798260 | 0.950000 | 0.962500 | 1.849781 | 1.835760 |
| 21 | 2.785379 | 2.768737 | 0.950000 | 0.962500 | 1.835379 | 1.806237 |
| 28 | 2.831808 | 2.781237 | 0.950000 | 0.962500 | 1.881808 | 1.818737 |
| 35 | 2.799721 | 2.820522 | 0.950000 | 0.962500 | 1.849721 | 1.858022 |

## Output contract metrics

| step | GRPO tool parse | GDPO tool parse | GRPO response wrapper | GDPO response wrapper |
|---:|---:|---:|---:|---:|
| 0 | 73/80 (91.25%) | 73/80 (91.25%) | 5/80 (6.25%) | 5/80 (6.25%) |
| 7 | 74/80 (92.50%) | 74/80 (92.50%) | 5/80 (6.25%) | 4/80 (5.00%) |
| 14 | 74/80 (92.50%) | 75/80 (93.75%) | 5/80 (6.25%) | 4/80 (5.00%) |
| 21 | 74/80 (92.50%) | 75/80 (93.75%) | 5/80 (6.25%) | 4/80 (5.00%) |
| 28 | 74/80 (92.50%) | 75/80 (93.75%) | 5/80 (6.25%) | 4/80 (5.00%) |
| 35 | 74/80 (92.50%) | 75/80 (93.75%) | 5/80 (6.25%) | 4/80 (5.00%) |

## Training metric summary

| metric | GRPO mean | GRPO min-max | GDPO mean | GDPO min-max |
|---|---:|---:|---:|---:|
| actor/loss | 0.012533 | -0.008118 to 0.036012 | -0.003111 | -0.017813 to 0.005842 |
| actor/grad_norm | 0.411523 | 0.329626 to 0.617412 | 0.595946 | 0.461149 to 0.897989 |
| actor/ppo_kl | 0.000032 | -0.000057 to 0.000214 | 0.000058 | -0.000072 to 0.000487 |
| perf/time_per_step | 1089.529834 | 1073.699350 to 1118.507512 | 1078.800352 | 1055.502614 to 1111.230258 |
| critic/score/mean | 2.356651 | 1.221488 to 2.789111 | 2.417724 | 1.243984 to 2.854196 |

## KL-treatment confound

The two runs are not identical except for the advantage estimator. The GRPO path enables reward-side KL treatment (use_kl_in_reward=true), while the GDPO path uses the component reward dimensions (accuracy_reward, format_reward) and keeps actor KL loss disabled. Therefore any GRPO/GDPO difference here cannot be attributed purely to the advantage-estimator change. A future KL-aligned ablation is required for that claim.

## Evidence

- GRPO report: reports/grpo/stage1_35/grpo_stage1_report.md
- GDPO report: reports/gdpo/stage1_35/gdpo_stage1_report.md
- Machine-readable comparison: reports/comparison/grpo_gdpo_stage1_35_comparison.json
- GDPO validation curve: reports/gdpo/stage1_35/validation_learning_curve.json
