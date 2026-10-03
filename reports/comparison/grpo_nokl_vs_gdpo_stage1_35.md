# GRPO-noKL vs GDPO-current

This is a descriptive 35-step, single-seed comparison on the frozen formal_validation_80 endpoint.
The table uses left=noKL, right=GDPO. The archived bootstrap JSON uses the reverse orientation (noKL−GDPO); see the explicit conversion below. It is not a population-significance claim.

| step | left total | right total | left accuracy | right accuracy | left format | right format |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 2.400171 | 1.450171 | 1.450171 | 0.950000 | 0.950000 |
| 7 | 2.588245 | 2.611385 | 1.638245 | 1.648885 | 0.950000 | 0.962500 |
| 14 | 2.757760 | 2.798260 | 1.807760 | 1.835760 | 0.950000 | 0.962500 |
| 21 | 2.784451 | 2.768737 | 1.821951 | 1.806237 | 0.962500 | 0.962500 |
| 28 | 2.781237 | 2.781237 | 1.818737 | 1.818737 | 0.962500 | 0.962500 |
| 35 | 2.818737 | 2.820522 | 1.856237 | 1.858022 | 0.962500 | 0.962500 |

## Sensitivity at step 35

A post-hoc sensitivity analysis removes source_id 3814 from the same 80 outputs, using a fixed exclusion rule applied identically across algorithms and steps. This gives 79 rows: left total 2.829100; right total 2.830909.

## Boundary

- GDPO uses raw component rewards and no actor KL loss; noKL is objective-level KL-aligned. This compares complete advantage constructions (including GDPO batch whitening), not per-dimension normalization alone. The nearly identical endpoints do not establish a clear GDPO downstream advantage.
- GRPO-noKL has both reward-side KL and actor KL loss disabled, so its reference-policy runtime is different; do not compare wall-clock/GPU efficiency as if it were fair.
- These results are single-seed Stage 1 evidence and do not establish 70/105-step behavior.

Paired bootstrap: [JSON](paired_bootstrap_stage1_35.json), key `GDPO-current_vs_GRPO-noKL`, which reports noKL−GDPO: step35 mean −0.001786, 95% interval [-0.005357, 0]. For this table's GDPO−noKL direction, negate the mean and reverse/negate interval endpoints: +0.001786, [0, 0.005357]. This algebraic conversion does not rerun the bootstrap or establish GDPO superiority.
