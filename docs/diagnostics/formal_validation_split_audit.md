# Formal validation split audit

> **Historical row-disjoint split audit; content-isolation correction added 2026-10-03.** The original source-row intersections below remain valid, but “clean” did not establish content disjointness: validation source3814 duplicates train/SFT source527 prompt and ground truth. Preserve primary80 and use a fixed post-hoc sensitivity79 exclusion across algorithms/steps; see the [analysis plan](../../reports/comparison/grpo_gdpo_primary_sensitivity_analysis_plan.md). Current outcomes: [three-run completion](../../reports/comparison/final_stage1_ablation_completion_report.md).

This audit freezes `formal_validation_80` before Stage 1. It does not start training or modify the parquet datasets, reward manager, algorithm, or environment.

## Source and exact split

- Source: `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet`
- Raw train rows: **3920**
- Removed by the already-audited exact overlong rule: **19**
- Eligible rows: **3901**
- Actual RL optimizer sequence per epoch: `eligible[0:3584]` = **3584** prompts
- Permanent `drop_last=True` tail: `eligible[3584:3901]` = **317** prompts
- Clean tail after all exclusions: **188** prompts
- Selected formal validation: **80** prompts
- Selection: `numpy.default_rng(42)` samples 80 from the clean tail; selected source rows are then sorted for a stable manifest.
- Selected prompt token length under the exact RL tokenizer/chat template: **428–1664**; all 80 are <=2048.
- The selected rows remain outside the optimizer because they are in the permanent drop-last tail.

## Exclusion accounting

| set | total unique rows | overlap with tail |
|---|---:|---:|
| SFT v1 | 800 | 54 |
| SFT v2 | 800 | 60 |
| SFT union | 993 | 87 |
| fixed 16 validation | 16 | 1 |
| constrained-generation diagnostic | 16 | 1 |
| fresh holdout pilot | 32 | 1 |
| shared-rollout diagnostic | 8 | 0 |
| capacity smoke | 512 | 52 |
| known diagnostic union | 560 | 54 |
| all exclusions (SFT union + diagnostics) | 1425 | 129 |

Known diagnostic source artifacts were excluded conservatively when an explicit source-row manifest was available. The capacity-smoke manifest is included, as are the fixed 16-row reports, constrained-generation report, fresh-holdout pilot, and shared-rollout advantage diagnostic.

## Selected manifest and separation proofs

- Manifest: `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json`
- Runtime validation parquet (derived view, not a source-dataset modification): `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet`
- Materialized parquet row count/order matches the manifest source rows: **PASS (80/80)**.
- Validation count: **80** (asserted)
- Validation ∩ actual RL train prefix: **0**
- Validation ∩ SFT v1/v2 source union: **0**
- Validation ∩ known diagnostic union: **0**
- Validation ∩ eligible tail: **80**
- Actual RL train source sequence SHA-256: `972f09807948b426f873d523b5dead2de92fe3d9b58ed7ca109bf34ffd0aea7b`
- Validation source sequence SHA-256: `24692b6ddbf8573a600feb71142bd412e3250aeac6cca4525a5f6931ec3aa07e`

Selected source rows:
```text
[3616, 3621, 3622, 3623, 3625, 3627, 3631, 3644, 3645, 3647, 3657, 3660, 3670, 3672, 3679, 3680, 3687, 3688, 3691, 3692, 3693, 3696, 3700, 3705, 3707, 3711, 3713, 3714, 3721, 3722, 3728, 3730, 3735, 3738, 3740, 3743, 3745, 3746, 3747, 3748, 3750, 3752, 3762, 3774, 3777, 3780, 3787, 3790, 3791, 3797, 3803, 3805, 3806, 3807, 3810, 3811, 3814, 3815, 3819, 3830, 3832, 3837, 3839, 3847, 3862, 3863, 3864, 3865, 3866, 3880, 3881, 3884, 3887, 3889, 3896, 3909, 3910, 3915, 3918, 3919]
```

## Shared evaluation contract

- GRPO manifest: `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json`
- GDPO manifest: `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json`
- Both algorithms must use exactly this manifest, the same greedy `n=1` evaluation settings, and the same reward manager.
- Evaluation points: steps `0, 7, 14, 21, 28, 35` for Stage 1; evaluation never calls `optimizer.step`.

## Test-set firewall

- `test.parquet` remains at `/root/autodl-tmp/ProjectB/repo/dataset/rlla_4k/test.parquet`.
- It is removed from intermediate validation and cannot be used for early stopping, stage continuation, hyperparameter selection, or checkpoint selection.
- After the common final budget is frozen, each final GRPO and GDPO checkpoint may receive one final evaluation on `test.parquet`.

## Paired budget-decision statistic

- Compare step 28 versus step 35 on the same 80 source rows, paired by source row.
- For total reward and accuracy reward separately, report the mean paired difference, the fraction of prompts improved, and a deterministic paired bootstrap 95% percentile interval (seed 42; 10,000 resamples).
- If the interval crosses zero and the improvement is small, treat it as no clear continuation benefit. Continue only for stable positive improvement with no material format/parse/KL regression.

## Result

**PASS: 80 clean formal-validation rows were constructed from the permanent drop-last tail; the selected set is disjoint from the actual RL train prefix, SFT sources, and all explicitly tracked diagnostic rows.**

## Current test and decision boundary

The test-firewall rule below governs formal Stage1, not all historical scripts. The legacy [single-GPU smoke](../../scripts/run_single_gpu_smoke.sh) configured test as validation; whether that path executed remains a pending server-log audit. The canonical documented path is `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/test.parquet`; the earlier shorter path in this historical report is not execution-time proof. Follow the [final-test protocol](../experiment_design/final_test_protocol.md) before any inference. The current [35-step closeout](../experiment_design/stage1_closeout_decision.md) supersedes automatic continuation criteria.
