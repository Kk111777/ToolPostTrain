# ProjectB final Stage 1 ablation completion report

Generated from persisted artifacts after the three 35-step runs. No retraining, resume, new inference, or new GPU experiment was started during closeout.

## Run status

| run | status | initialization | final step | validation |
|---|---|---|---:|---|
| GRPO-original35 | PASS | RL_INIT_V1 | 35 | formal_validation_80 |
| GDPO-current35 | PASS | RL_INIT_V1 | 35 | formal_validation_80 |
| GRPO-noKL35 | PASS | RL_INIT_V1 | 35 | formal_validation_80 |

All three runs use the frozen 0/7/14/21/28/35 validation schedule. test.parquet was not used for intermediate validation. No 70/105-step run was started.

## noKL completion

- Completion marker: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/STAGE_COMPLETE_OK
- Full resume: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/checkpoints/global_step_35
- Model-only: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/model_only/step_35
- Final report: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/grpo_nokl_stage1_report.md
- Canonical metrics: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/metrics/stage1_metrics_canonical.json
- Verified checkpoint manifest: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/checkpoint_manifest_verified.json
- Old rolling checkpoint entries were cleaned only after step35 verification; step35 full and model-only were retained.

## Analysis outputs

- Three-run comparison: /root/autodl-tmp/ProjectB/repo/reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md
- GRPO-original vs noKL: /root/autodl-tmp/ProjectB/repo/reports/comparison/grpo_original_vs_nokl_stage1_35.md
- noKL vs GDPO: /root/autodl-tmp/ProjectB/repo/reports/comparison/grpo_nokl_vs_gdpo_stage1_35.md
- GRPO-original vs GDPO: /root/autodl-tmp/ProjectB/repo/reports/comparison/grpo_original_vs_gdpo_stage1_35.md
- Primary 80 vs sensitivity 79: /root/autodl-tmp/ProjectB/repo/reports/comparison/primary80_vs_sensitivity79_stage1_35.md
- Paired bootstrap: /root/autodl-tmp/ProjectB/repo/reports/comparison/paired_bootstrap_stage1_35.md
- Canonical noKL metrics: /root/autodl-tmp/ProjectB/repo/reports/grpo/grpo_nokl_stage1_metrics_canonical.json

The three-way interpretation retains the known KL-treatment/reward-dimension confound. NoKL also removes the reference-policy runtime path under the audited official v0.9.1 semantics, so wall-clock and GPU-efficiency comparisons are not treated as fair endpoints. Results are single-seed, 35-step Stage 1 evidence only.

## Storage and next boundary

- At closeout: /root/autodl-tmp is 200G total with about 88G free; GPU memory is 0 MiB.
- GRPO-original35 and GDPO-current35 assets remain retained.
- The three step35 full resume checkpoints and model-only artifacts are retained for explicit future continuation decisions.
- No automatic 70/105 continuation is authorized.

## Git synchronization

- results_commit: PENDING
- verification_commit: PENDING
- push_status: PENDING
