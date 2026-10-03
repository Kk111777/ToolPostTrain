# GRPO-original vs GDPO-current

This is a descriptive 35-step, single-seed comparison on the frozen formal_validation_80 endpoint.
The paired bootstrap is right minus left over the same validation row order; it is not a population-significance claim.

| step | left total | right total | left accuracy | right accuracy | left format | right format |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 2.400171 | 1.450171 | 1.450171 | 0.950000 | 0.950000 |
| 7 | 2.717582 | 2.611385 | 1.767582 | 1.648885 | 0.950000 | 0.962500 |
| 14 | 2.799781 | 2.798260 | 1.849781 | 1.835760 | 0.950000 | 0.962500 |
| 21 | 2.785379 | 2.768737 | 1.835379 | 1.806237 | 0.950000 | 0.962500 |
| 28 | 2.831808 | 2.781237 | 1.881808 | 1.818737 | 0.950000 | 0.962500 |
| 35 | 2.799721 | 2.820522 | 1.849721 | 1.858022 | 0.950000 | 0.962500 |

## Sensitivity at step 35

A post-hoc sensitivity analysis removes source_id 3814 from the same 80 outputs, using a fixed exclusion rule applied identically across algorithms and steps. This gives 79 rows: left total 2.809844; right total 2.830909.

## Boundary

- GRPO-original vs GDPO-current retains the known KL-treatment and reward-dimension confound.
- GRPO-noKL has both reward-side KL and actor KL loss disabled, so its reference-policy runtime is different; do not compare wall-clock/GPU efficiency as if it were fair.
- These results are single-seed Stage 1 evidence and do not establish 70/105-step behavior.

Paired bootstrap: [JSON](paired_bootstrap_stage1_35.json), key `GRPO-original_vs_GDPO-current`. Table left=original, right=GDPO; JSON uses GDPO−original. At step35: mean +0.020801, 95% interval [-0.122012, 0.169623]. These are archived estimates, not independently recomputed from formal validation JSONL during this revision.
