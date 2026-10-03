# ProjectB shared-rollout GRPO/GDPO advantage diagnostic

No backward pass or optimizer.step() was executed in this diagnostic.

## Configuration

- Frozen initialization: `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged` (RL_INIT_V1)
- Dataset: `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet`
- Fixed seed: `20261001`
- Selected source rows: `[978, 1321, 1481, 3460, 2175, 2378, 2940, 2598]`
- Rollout: `vLLM`, TP=1, n=4, temperature=1.0, max_tokens=1024
- Reward manager: official `GDPORewardManager` with the existing `rlla.compute_score`.
- KL: unavailable in this standalone shared-rollout pass; the live trainer path is recorded in `kl_path_audit.md` and Parts B/C.

## Advantage summary (archived raw diagnostics)

- GRPO valid-response-token mean/std: `0.010018` / `0.500527`
- GDPO valid-response-token mean/std: `-0.000000` / `1.000000`
- All advantages finite: `True`
- Raw exact-float sign disagreement: `16/32`
- Mean |GRPO-GDPO|: `0.387903`
- Raw exact-float within-group ranking disagreement pairs: `11`

## Interpretation correction (2026-10-03)

The [original JSON](grpo_gdpo_shared_rollout_diagnostic.json), including `sign_disagreement_count=16` and `ranking_disagreement_pair_count=11`, is unchanged. The revision recomputed the following **from its public per-trajectory advantage values**, without generation, backward or optimizer updates:

| diagnostic | definition | result |
|---|---|---:|
| raw sign mismatch | exact sign(GRPO A) != sign(GDPO A) | 16/32 |
| positive↔negative advantage reversal | GRPO A × GDPO A < 0 | 0/32 |
| raw within-group ranking mismatch | exact sign(A_i−A_j) differs, among 48 within-prompt pairs | 11/48 |
| substantive within-group ranking disagreement | same comparison with absolute tolerance 1e-8 | 0/48 |
| strict within-group rank reversal | both pairwise differences exceed1e-8 and have opposite signs | 0/48 |

All 16 raw sign mismatches are GRPO zero → GDPO approximately −0.01672776 in four equal-reward groups. They are zero-to-nonzero shifts, not positive↔negative reversals. That centering shift is larger than the numerical tolerance and remains a real advantage-value difference.

All 11 raw ranking mismatches are a GRPO tie versus a tiny GDPO pairwise difference (maximum 3.725290298461914e-09, below 1e-8). They provide no evidence of meaningful within-group preference reordering. Here sign_tau(x)=0 for |x|≤1e-8, +1 above the tolerance and−1 below it. This distinguishes raw exact-float diagnostics from numerically meaningful diagnostics; it does not erase the magnitude/centering differences.

The archived mean/std summarize valid response-token advantages, not an unweighted set of 32 trajectory scalars. Longer responses contribute more tokens. Per-trajectory averages therefore need not have the same mean, and token-level batch whitening can shift formerly zero trajectory advantages. GRPO and GDPO also differ in scaling/weighting across samples and groups. **GDPO changes advantage scale/centering/sample weighting, but this shared rollout does not show substantive within-group preference reversals under a numerical tolerance.** It contains no gradient-direction or downstream-performance measurement.

Reproduction rule: group records by `prompt_group_id`; compare every unordered within-group pair; compute signs of each algorithm's advantage difference with the above tolerance. Public input counts and raw fields are preserved. See the [revision audit](../../docs/diagnostics/documentation_revision_audit_20261003.md) for the checked values and [fresh-holdout conflict](../sft/fresh_holdout_reward_variation.md) for a separate sample that does contain opposing reward dimensions.

## Reward dimensions

- Format mean/std: `0.750000` / `0.433013`
- Accuracy mean/std: `0.034524` / `1.906943`
- Total/raw mean/std: `0.784524` / `2.142707`

## Per-group variance

| group | source row | Var(format) | Var(accuracy) | Var(total) | Var(GRPO A) | Var(GDPO A) |
|---|---:|---:|---:|---:|---:|---:|
| 978 | 978 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1321 | 1321 | 0.187500 | 4.913265 | 6.922194 | 0.749999 | 3.783586 |
| 1481 | 1481 | 0.000000 | 1.800000 | 1.800000 | 0.749999 | 0.970725 |
| 3460 | 3460 | 0.250000 | 6.030000 | 8.380000 | 0.750000 | 3.601749 |
| 2175 | 2175 | 0.187500 | 6.750000 | 9.187500 | 0.750000 | 3.882897 |
| 2378 | 2378 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2940 | 2940 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2598 | 2598 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Reward-dimension collision examples

A collision is a rollout whose format and accuracy group-normalized rewards have opposite signs; these are the cases where decoupled normalization can disagree with total-reward normalization.

| group | rollout | format | accuracy | total | format z | accuracy z | GRPO A | GDPO A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
No opposite-sign per-dimension group-normalized reward example occurred in this batch.

## Per-rollout records

The JSON report retains the full prompt/response token IDs and response text.

| source row | group | rollout | format | accuracy | total | GRPO A | GDPO A |
|---:|---|---:|---:|---:|---:|---:|---:|
| 978 | 978 | 0 | 1.000 | -2.167 | -1.167 | 0.000000 | -0.016728 |
| 978 | 978 | 1 | 1.000 | -2.167 | -1.167 | 0.000000 | -0.016728 |
| 978 | 978 | 2 | 1.000 | -2.167 | -1.167 | 0.000000 | -0.016728 |
| 978 | 978 | 3 | 1.000 | -2.167 | -1.167 | 0.000000 | -0.016728 |
| 1321 | 1321 | 0 | 1.000 | 1.286 | 2.286 | 0.293894 | 0.837852 |
| 1321 | 1321 | 1 | 1.000 | 1.286 | 2.286 | 0.293894 | 0.837852 |
| 1321 | 1321 | 2 | 1.000 | 3.000 | 4.000 | 0.858170 | 1.599837 |
| 1321 | 1321 | 3 | 0.000 | -3.000 | -3.000 | -1.445958 | -3.342453 |
| 1481 | 1481 | 0 | 1.000 | -0.600 | 0.400 | -1.161894 | -1.338584 |
| 1481 | 1481 | 1 | 1.000 | 1.800 | 2.800 | 0.387298 | 0.423891 |
| 1481 | 1481 | 2 | 1.000 | 3.000 | 4.000 | 1.161894 | 1.305129 |
| 1481 | 1481 | 3 | 1.000 | 0.600 | 1.600 | -0.387298 | -0.457347 |
| 3460 | 3460 | 0 | 0.000 | -3.000 | -3.000 | -0.777825 | -1.844556 |
| 3460 | 3460 | 1 | 1.000 | -0.600 | 0.400 | 0.239331 | 1.088893 |
| 3460 | 3460 | 2 | 0.000 | -3.000 | -3.000 | -0.777825 | -1.844556 |
| 3460 | 3460 | 3 | 1.000 | 3.000 | 4.000 | 1.316319 | 2.533308 |
| 2175 | 2175 | 0 | 0.000 | -3.000 | -3.000 | -1.500000 | -3.429746 |
| 2175 | 2175 | 1 | 1.000 | 3.000 | 4.000 | 0.500000 | 1.120945 |
| 2175 | 2175 | 2 | 1.000 | 3.000 | 4.000 | 0.500000 | 1.120945 |
| 2175 | 2175 | 3 | 1.000 | 3.000 | 4.000 | 0.500000 | 1.120945 |
| 2378 | 2378 | 0 | 0.000 | 0.000 | 0.000 | 0.000000 | -0.016728 |
| 2378 | 2378 | 1 | 0.000 | 0.000 | 0.000 | 0.000000 | -0.016728 |
| 2378 | 2378 | 2 | 0.000 | 0.000 | 0.000 | 0.000000 | -0.016728 |
| 2378 | 2378 | 3 | 0.000 | 0.000 | 0.000 | 0.000000 | -0.016728 |
| 2940 | 2940 | 0 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2940 | 2940 | 1 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2940 | 2940 | 2 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2940 | 2940 | 3 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2598 | 2598 | 0 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2598 | 2598 | 1 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2598 | 2598 | 2 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
| 2598 | 2598 | 3 | 1.000 | 0.000 | 1.000 | 0.000000 | -0.016728 |
