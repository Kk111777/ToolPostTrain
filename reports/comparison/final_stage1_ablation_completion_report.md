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
- Old rolling checkpoint entries for all three runs were cleaned only after step35 verification; each step35 full and model-only artifact was retained. Cleanup evidence is recorded in the three checkpoint_cleanup_manifest.json files.
- GPU memory at closeout: 0 MiB.

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

- At closeout: /root/autodl-tmp is 200G total with about 88G free (57% used).
- ProjectB occupies about 113G; the shared checkpoints directory about 3.5G; each of the three stage run directories about 22G.
- GRPO-original35 and GDPO-current35 assets remain retained.
- The three step35 full resume checkpoints and model-only artifacts are retained for explicit future continuation decisions.
- No automatic 70/105 continuation is authorized.

## Git synchronization

- Results commit: ff01509d97e50c37a9ac8cc4a00e0c3d164b031b
- Commit message: projectb: add GRPO-noKL ablation and three-run comparison
- Push status: PASS; origin/main was verified at ff01509d97e50c37a9ac8cc4a00e0c3d164b031b and the curated report/metrics/comparison files are present on the remote branch.
- Checkpoint/model weights, optimizer state, large logs, caches, and secrets were not committed.
- Curated cleanup manifests were added for the GRPO-original and GDPO-current runs after the final integrity check.
- This report update is a closeout verification record; it does not change any training result.

## Documentation closeout decision (2026-10-03)

The run-completion and Git/storage records above describe their recorded closeout time and were not reverified on the server during this revision. The accepted [budget decision](../../docs/experiment_design/stage1_closeout_decision.md) now freezes the current project at 35 training steps; this does not establish convergence or authorize checkpoint deletion.

The original A/B comparison retains its objective-treatment/advantage confound. The completed noKL reference aligns objective-level reward-side KL with GDPO, while comparing the complete advantage construction and differing reference-policy runtime. Current nearly identical GDPO/noKL validation endpoints do not establish a clear downstream advantage for GDPO. Sensitivity79 is post-hoc, with the same fixed exclusion across runs/steps. See the [current three-run interpretation](grpo_gdpo_nokl_stage1_35_comparison.md).

The [final-test protocol](../../docs/experiment_design/final_test_protocol.md) includes RL_INIT_V1 and all three step35 models. It is NOT EXECUTED; historical test provenance, actual hashes/configs and analysis-program freeze remain pending. No server startup, GPU work or final inference occurred during documentation revision.
