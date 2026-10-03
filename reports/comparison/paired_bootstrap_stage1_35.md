# Pairwise bootstrap: ProjectB Stage 1

The estimates below are paired differences in raw per-example validation score, right minus left, using the same validation row order. The bootstrap is deterministic (10,000 resamples, seed 42) and is descriptive, not a claim of population-level significance.

| pair | subset | step | n | mean difference | 95% bootstrap interval | approx two-sided tail |
|---|---|---:|---:|---:|---:|---:|
| GRPO-original_vs_GDPO-current | primary_80 | 0 | 80 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GRPO-original_vs_GDPO-current | primary_80 | 7 | 80 | -0.106196 | [-0.301826, 0.045375] | 0.4669 |
| GRPO-original_vs_GDPO-current | primary_80 | 14 | 80 | -0.001520 | [-0.100740, 0.095801] | 0.9749 |
| GRPO-original_vs_GDPO-current | primary_80 | 21 | 80 | -0.016643 | [-0.154648, 0.111357] | 0.8023 |
| GRPO-original_vs_GDPO-current | primary_80 | 28 | 80 | -0.050571 | [-0.201077, 0.092645] | 0.5715 |
| GRPO-original_vs_GDPO-current | primary_80 | 35 | 80 | 0.020801 | [-0.122012, 0.169623] | 0.7769 |
| GRPO-original_vs_GDPO-current | sensitivity_79 | 0 | 79 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GRPO-original_vs_GDPO-current | sensitivity_79 | 7 | 79 | -0.107541 | [-0.306148, 0.045497] | 0.4753 |
| GRPO-original_vs_GDPO-current | sensitivity_79 | 14 | 79 | -0.001539 | [-0.101503, 0.096943] | 0.9767 |
| GRPO-original_vs_GDPO-current | sensitivity_79 | 21 | 79 | -0.016854 | [-0.154430, 0.114143] | 0.8061 |
| GRPO-original_vs_GDPO-current | sensitivity_79 | 28 | 79 | -0.051212 | [-0.201814, 0.092445] | 0.5749 |
| GRPO-original_vs_GDPO-current | sensitivity_79 | 35 | 79 | 0.021065 | [-0.119209, 0.171694] | 0.7779 |
| GRPO-original_vs_GRPO-noKL | primary_80 | 0 | 80 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GRPO-original_vs_GRPO-noKL | primary_80 | 7 | 80 | -0.129337 | [-0.316837, 0.000000] | 0.4776 |
| GRPO-original_vs_GRPO-noKL | primary_80 | 14 | 80 | -0.042020 | [-0.134821, 0.015011] | 0.4466 |
| GRPO-original_vs_GRPO-noKL | primary_80 | 21 | 80 | -0.000929 | [-0.129643, 0.123000] | 0.9886 |
| GRPO-original_vs_GRPO-noKL | primary_80 | 28 | 80 | -0.050571 | [-0.198071, 0.090500] | 0.5717 |
| GRPO-original_vs_GRPO-noKL | primary_80 | 35 | 80 | 0.019016 | [-0.125000, 0.165516] | 0.7939 |
| GRPO-original_vs_GRPO-noKL | sensitivity_79 | 0 | 79 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GRPO-original_vs_GRPO-noKL | sensitivity_79 | 7 | 79 | -0.130974 | [-0.320847, 0.000000] | 0.4784 |
| GRPO-original_vs_GRPO-noKL | sensitivity_79 | 14 | 79 | -0.042552 | [-0.134721, 0.015201] | 0.4454 |
| GRPO-original_vs_GRPO-noKL | sensitivity_79 | 21 | 79 | -0.000940 | [-0.134615, 0.125935] | 0.9890 |
| GRPO-original_vs_GRPO-noKL | sensitivity_79 | 28 | 79 | -0.051212 | [-0.202749, 0.090128] | 0.5694 |
| GRPO-original_vs_GRPO-noKL | sensitivity_79 | 35 | 79 | 0.019256 | [-0.125047, 0.168636] | 0.7974 |
| GDPO-current_vs_GRPO-noKL | primary_80 | 0 | 80 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GDPO-current_vs_GRPO-noKL | primary_80 | 7 | 80 | -0.023140 | [-0.157929, 0.092893] | 0.7226 |
| GDPO-current_vs_GRPO-noKL | primary_80 | 14 | 80 | -0.040500 | [-0.166013, 0.071012] | 0.5693 |
| GDPO-current_vs_GRPO-noKL | primary_80 | 21 | 80 | 0.015714 | [-0.067857, 0.102500] | 0.7115 |
| GDPO-current_vs_GRPO-noKL | primary_80 | 28 | 80 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GDPO-current_vs_GRPO-noKL | primary_80 | 35 | 80 | -0.001786 | [-0.005357, 0.000000] | 0.6437 |
| GDPO-current_vs_GRPO-noKL | sensitivity_79 | 0 | 79 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GDPO-current_vs_GRPO-noKL | sensitivity_79 | 7 | 79 | -0.023433 | [-0.153127, 0.095391] | 0.7154 |
| GDPO-current_vs_GRPO-noKL | sensitivity_79 | 14 | 79 | -0.041013 | [-0.168608, 0.073418] | 0.5551 |
| GDPO-current_vs_GRPO-noKL | sensitivity_79 | 21 | 79 | 0.015913 | [-0.068716, 0.103797] | 0.7118 |
| GDPO-current_vs_GRPO-noKL | sensitivity_79 | 28 | 79 | 0.000000 | [0.000000, 0.000000] | 1.0000 |
| GDPO-current_vs_GRPO-noKL | sensitivity_79 | 35 | 79 | -0.001808 | [-0.005425, 0.000000] | 0.6372 |

Interpretation remains bounded by the known KL-treatment/reward-dimension confounds and the single-seed, 35-step Stage 1 budget. This table does not justify a pure causal claim.

## Interpretation and revision boundary (2026-10-03)

Pair keys encode left_vs_right; `right_minus_left` in the [JSON](paired_bootstrap_stage1_35.json) is authoritative. The noKL-vs-GDPO narrative table uses the reverse orientation and [explicitly converts](grpo_nokl_vs_gdpo_stage1_35.md) the archived estimates. No bootstrap result was regenerated in this documentation revision because the formal per-row JSONL is not public.

Sensitivity79 is post-hoc with one fixed exclusion applied identically to all runs and steps. The percentile intervals describe prompt-level variation conditional on these single-seed trained models; they do not estimate training-seed variability or algorithm equivalence. The `approx two-sided tail` column is archived diagnostic output, not a calibrated population-level significance test. Cross-algorithm same-step intervals are not within-run step28→35 continuation statistics. See the [35-step budget decision](../../docs/experiment_design/stage1_closeout_decision.md).
