# GRPO-original vs GRPO-noKL

This is a descriptive 35-step, single-seed comparison on the frozen formal_validation_80 endpoint.
The paired bootstrap is right minus left over the same validation row order; it is not a population-significance claim.

| step | left total | right total | left accuracy | right accuracy | left format | right format |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 2.400171 | 1.450171 | 1.450171 | 0.950000 | 0.950000 |
| 7 | 2.717582 | 2.588245 | 1.767582 | 1.638245 | 0.950000 | 0.950000 |
| 14 | 2.799781 | 2.757760 | 1.849781 | 1.807760 | 0.950000 | 0.950000 |
| 21 | 2.785379 | 2.784451 | 1.835379 | 1.821951 | 0.950000 | 0.962500 |
| 28 | 2.831808 | 2.781237 | 1.881808 | 1.818737 | 0.950000 | 0.962500 |
| 35 | 2.799721 | 2.818737 | 1.849721 | 1.856237 | 0.950000 | 0.962500 |

## Sensitivity at step 35

Removing the pre-registered source_id 3814 row from the same 80 outputs gives 79 rows: left total 2.809844; right total 2.829100.

## Boundary

- GRPO-original vs GDPO-current retains the known KL-treatment and reward-dimension confound.
- GRPO-noKL has both reward-side KL and actor KL loss disabled, so its reference-policy runtime is different; do not compare wall-clock/GPU efficiency as if it were fair.
- These results are single-seed Stage 1 evidence and do not establish 70/105-step behavior.

Paired bootstrap details: reports/comparison/paired_bootstrap_stage1_35.json, key GRPO-noKL_vs_GRPO-original.
