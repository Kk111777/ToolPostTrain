# Results & Technical Evidence

Completed 2026-10-03 (Asia/Shanghai): three 35-step training runs and **432/432** final internal-holdout outputs. Start with the seven reports below; machine-readable artifacts and historical records follow by topic.

## Start Here

| Read | What it answers |
|---|---|
| [Final holdout completion and paired results](final_holdout_v1/completion_report.md) | What did the four fixed models achieve, and how uncertain are the differences? |
| [Three-run training comparison](comparison/grpo_gdpo_nokl_stage1_35_comparison.md) | How did the shared 35-step validation curves evolve? |
| [Final KL / objective-path audit](../docs/diagnostics/gdpo_hidden_kl_use_final_audit.md) | Why did matching KL flags conceal different objectives? |
| [Original vs noKL configuration diff](../docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md) | What was deliberately changed in the matched control? |
| [Format-SFT before/after diagnostic](sft/sft_v2_after_reward_report.md) | Why was an output-contract warmup needed? |
| [Shared-rollout advantage diagnostic](comparison/grpo_gdpo_shared_rollout_diagnostic.md) | What changed in magnitude/centering, and what did not change in ranking? |
| [Sampled reward-conflict diagnostic](sft/fresh_holdout_reward_variation.md) | Where did accuracy and format signals conflict? |

## 1. Main Results

- **Final endpoint:** [completion report](final_holdout_v1/completion_report.md) and [all six frozen comparisons / component metrics](final_holdout_v1/metrics_and_paired_comparisons.json). N=108 per model; raw total reward is the primary metric.
- **Training validation:** [three-run comparison](comparison/grpo_gdpo_nokl_stage1_35_comparison.md) · [JSON](comparison/grpo_gdpo_nokl_stage1_35_comparison.json); fixed primary80 at steps 0/7/14/21/28/35.
- **Validation pairwise reports:** [original vs noKL](comparison/grpo_original_vs_nokl_stage1_35.md), [noKL vs GDPO](comparison/grpo_nokl_vs_gdpo_stage1_35.md), [original vs GDPO](comparison/grpo_original_vs_gdpo_stage1_35.md).
- **Validation uncertainty / sensitivity:** [archived paired bootstrap](comparison/paired_bootstrap_stage1_35.md) · [JSON](comparison/paired_bootstrap_stage1_35.json); [primary80 vs sensitivity79](comparison/primary80_vs_sensitivity79_stage1_35.md) · [post-hoc plan](comparison/grpo_gdpo_primary_sensitivity_analysis_plan.md).

The final endpoint supports fixed-checkpoint RL increments over RL_INIT. All RL-versus-RL primary intervals include zero. This does not establish GDPO superiority, algorithm equivalence or seed robustness.

## 2. Training Runs

| Stage | Report | Completion / canonical evidence |
|---|---|---|
| Format SFT v2 | [800-example / 50-update report](sft/sft_v2_train_report.md) | [Data report](sft/format_sft_v2_data_report.md) · [Merged RL_INIT checks](comparison/rl_init_merge_validation.md) |
| GRPO-original35 | [Training report](grpo/stage1_35/grpo_stage1_report.md) | [PASS marker](grpo/stage1_35/STAGE_COMPLETE_OK) · [Canonical metrics](grpo/grpo_stage1_metrics_canonical.json) |
| GDPO-current35 | [Training report](gdpo/stage1_35/gdpo_stage1_report.md) | [PASS marker](gdpo/stage1_35/STAGE_COMPLETE_OK) · [Canonical metrics](gdpo/gdpo_stage1_metrics_canonical.json) |
| GRPO-noKL35 | [Training report](grpo_nokl/stage1_35/grpo_nokl_stage1_report.md) | [PASS marker](grpo_nokl/stage1_35/STAGE_COMPLETE_OK) · [Canonical metrics](grpo/grpo_nokl_stage1_metrics_canonical.json) |

[Three-run closeout](comparison/final_stage1_ablation_completion_report.md) records completed training and retained artifacts. [35-step decision](../docs/experiment_design/stage1_closeout_decision.md) records the budget boundary; it is not a convergence finding.

Effective configurations, checkpoint identities and cleanup manifests remain under each run's `stage1_35/` directory. [noKL verified model/checkpoint manifest](grpo_nokl/stage1_35/checkpoint_manifest_verified.json) distinguishes the performed checks. File presence, model-only loading and full optimizer/RNG/dataloader restoration are different levels of evidence.

## 3. KL / Objective Audit

- [Final hidden-KL audit](../docs/diagnostics/gdpo_hidden_kl_use_final_audit.md): raw component rewards versus KL-adjusted aggregate rewards, advantage dispatch and policy-loss consumers.
- [Original/noKL effective-config diff](../docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md) · [JSON](comparison/grpo_original_vs_grpo_nokl_effective_config_diff.json): one planned reward-side KL change plus isolated paths.
- [noKL prelaunch audit](comparison/grpo_nokl_prelaunch_audit.json): frozen controls checked before its formal run.
- [Initial KL audit](comparison/kl_path_audit.md): historical source tracing; the final audit governs the current objective interpretation.

The noKL run removes reference-policy computation as a consequence of the framework flags. It is an objective control, not a clean wall-clock estimator-efficiency benchmark.

## 4. Mechanism Diagnostics

- [Shared-rollout report and correction](comparison/grpo_gdpo_shared_rollout_diagnostic.md) · [original JSON](comparison/grpo_gdpo_shared_rollout_diagnostic.json): 8 × 4 trajectories; magnitude/centering differences, no substantive within-group ranking reversals under tolerance.
- [Fresh sampled reward variation / conflict](sft/fresh_holdout_reward_variation.md) · [JSON](sft/fresh_holdout_reward_variation.json): 8 conflict trajectories across 6/32 groups; a diagnostic sample, not formal-training conflict frequency.
- [Format-SFT before/after](sft/sft_v2_after_reward_report.md), [reward signal](../docs/diagnostics/reward_signal_report.md), [prompt contract](../docs/diagnostics/prompt_contract_report.md) and [SFT target audit](../docs/diagnostics/format_sft_target_audit.md): contract alignment and remaining sampled variation.
- [Documentation correction audit](../docs/diagnostics/documentation_revision_audit_20261003.md): public per-trajectory numerical checks behind the interpretation corrections.

Format saturation or insufficient conflict as the cause of small endpoint differences remains a **plausible mechanism hypothesis**, not an established causal mechanism.

## 5. Final Evaluation

| Question | Evidence |
|---|---|
| What protocol and endpoint were frozen? | [Protocol](../docs/experiment_design/final_test_protocol.md) · [Freeze audit](../docs/diagnostics/final_holdout_v1_freeze_audit.md) |
| Which rows, models and scorer? | [Endpoint](../manifests/final_holdout_v1_manifest.json) · [Runtime mapping](../manifests/final_holdout_v1_runtime_mapping.json) · [Models](../manifests/final_test_models_manifest.json) · [Scorer](../manifests/final_test_scorer_manifest.json) |
| Did every output pass? | [432/432 completion gate](final_holdout_v1/completion_gate.json) |
| What actual config and identity checks passed? | [RL_INIT](final_holdout_v1/01_RL_INIT_V1_retry_02_integrity.json) · [original](final_holdout_v1/02_GRPO_original35_integrity.json) · [GDPO](final_holdout_v1/03_GDPO_current35_integrity.json) · [noKL](final_holdout_v1/04_GRPO_noKL35_integrity.json) |
| What is the complete frozen result? | [All comparisons and strata](final_holdout_v1/metrics_and_paired_comparisons.json) · [Completion report](final_holdout_v1/completion_report.md) |
| How were execution failures handled? | [Preserved failure and engineering repairs](final_holdout_v1/completion_report.md#preserved-failure-provenance-and-engineering-repairs) · [Original false gate](final_holdout_v1/RL_INIT_V1_initial_validator_failure.json) |
| Where is private archival verification recorded? | [Archive receipt](final_holdout_v1/archive_receipt.json) |

FINAL_HOLDOUT_V1 is a post-hoc internal corroborative endpoint under the recoverable exposure audit. It shares the original train-split tail source with validation. [Historical official-test usage](../docs/diagnostics/final_test_historical_usage_audit.md) was audited; no official unseen-test claim is made.

## 6. Reproducibility / Infrastructure

- **Runtime:** [final runtime versions](final_holdout_v1/runtime_environment.json), [earlier SM120/runtime verification](runtime_verification_20261002.md), [environment setup report](../configs/environment_report.md).
- **Single-GPU capacity:** [GRPO](grpo/grpo_official_scale_capacity_report.md) / [GDPO](gdpo/gdpo_official_scale_capacity_report.md); Config A [GRPO](grpo/grpo_throughput_config_a_report.md) / [GDPO](gdpo/gdpo_throughput_config_a_report.md).
- **Static final-evaluation checks:** [launcher](../docs/diagnostics/final_test_launcher_static_audit.md), [metric definition](../docs/diagnostics/final_test_metric_definition_audit.md), [analysis script](../docs/diagnostics/final_test_analysis_script_validation.md), [interface repair](../docs/diagnostics/final_eval_interface_fix_audit_20261003.md), [actor config repair](../docs/diagnostics/final_eval_actor_config_fix_audit_20261003.md).
- **Audit trail:** [server artifact manifest](server_artifact_manifest.json), [documentation audit](../docs/diagnostics/documentation_revision_audit_20261003.md), [presentation claim audit](../docs/diagnostics/github_presentation_claim_audit_20261003.md).

Raw formal JSONL, parquet, weights, optimizer state and large runtime logs are not published. The final scorer recomputation and frozen analysis were performed during experiment closeout; this presentation revision cross-checks public artifacts without rerunning them. Validation bootstrap values remain archived estimates, not independently recomputed during this revision.

<details>
<summary>Historical snapshots and engineering incidents</summary>

- [Original two-run completion](comparison/stage1_35x2_completion_report.md) and [preflight report](comparison/final_test_preflight_report.md) describe their original stages, not the final four-model status.
- [GRPO gate failure resolution](grpo/stage1_35/STAGE_FAILED.resolution.md), [historical 0/105 launch failure](grpo/historical_grpo_formal_run_report.md), [old resume smoke](../docs/diagnostics/checkpoint_resume_smoke.md) and the original environment failure remain preserved.
- The original GRPO report's `step_time=False` reflected a check-key mismatch; its 35 native `perf/time_per_step` values are finite. Canonical metrics govern current comparisons.
- The final completion gate's shutdown field is a pre-shutdown snapshot. Publication, private archive verification and successful shutdown/SSH-unreachable receipts were subsequently saved in the local controlled archive.

</details>
