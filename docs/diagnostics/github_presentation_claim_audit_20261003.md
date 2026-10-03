# GitHub presentation claim audit

Date: 2026-10-03 (Asia/Shanghai). Scope: README, reports index, resume summary and the principal result/diagnostic reports. Verification is source-bound to the public artifacts at result commit `a93d4298b25470ede7ac6906827afaa7e71220a0`; it does not rerun training, inference, scoring or bootstrap.

## Material claims and governing evidence

| Claim / wording | Source | Verdict and boundary |
|---|---|---|
| Real single-GPU post-training on Qwen2.5-1.5B / SM120 | [Runtime](../../reports/runtime_verification_20261002.md), [three-run closeout](../../reports/comparison/final_stage1_ablation_completion_report.md) | Supported. Adaptation and execution, not a new algorithm or full original-paper setup. |
| Base protocol failure motivated Format SFT | [Prompt contract](prompt_contract_report.md), [before/after diagnostic](../../reports/sft/sft_v2_after_reward_report.md) | Supported on the recorded diagnostic sample. No external benchmark claim. |
| 800 SFT examples, one epoch, 50 updates; shared merged initialization | [SFT report](../../reports/sft/sft_v2_train_report.md), [merge validation](../../reports/comparison/rl_init_merge_validation.md) | Supported. Adapter/merged bitwise identity is not claimed. |
| Three completed 35-step runs with matched settings | [Closeout](../../reports/comparison/final_stage1_ablation_completion_report.md), [config diff](grpo_original_vs_grpo_nokl_effective_config_diff.md) | Supported. Each run starts separately from RL_INIT; “independent” does not mean multiple seeds. |
| KL confound despite matching flags | [Final objective-path audit](gdpo_hidden_kl_use_final_audit.md) | Supported for pinned veRL v0.9.1 and the configured GDPO reward keys. Not a universal GDPO property. |
| noKL is an objective-level KL-aligned reference | [Config/runtime diff](grpo_original_vs_grpo_nokl_effective_config_diff.md) | Supported. Reference computation changes as a flag consequence; no pure efficiency ranking. |
| 108 rows × four models = 432 outputs; exact scorer agreement | [Completion gate](../../reports/final_holdout_v1/completion_gate.json), [completion report](../../reports/final_holdout_v1/completion_report.md) | Supported by the completed execution evidence. No new recomputation during presentation. |
| Four means, reward components, deltas and intervals | [Canonical final results](../../reports/final_holdout_v1/metrics_and_paired_comparisons.json) | Supported. README values rounded to six decimals; Δ is explicitly right minus left. |
| RL gain over initialization, no clear GDPO/noKL advantage | [All six primary comparisons](../../reports/final_holdout_v1/completion_report.md#all-predeclared-paired-comparisons) | RL−RL_INIT intervals above zero; all RL−RL intervals contain zero. No algorithm equivalence, hypothesis-test significance or general superiority claim. |
| Advantage magnitude/centering differences, no substantive ranking reversal | [Shared-rollout correction](../../reports/comparison/grpo_gdpo_shared_rollout_diagnostic.md) | Supported on 32 trajectories at documented tolerance. Raw zero-to-negative shifts are not positive/negative reversals. |
| 8/128 conflict trajectories across 6/32 groups | [Conflict diagnostic](../../reports/sft/fresh_holdout_reward_variation.md) | Supported as a post-hoc diagnostic count, not formal-training conflict frequency. |
| High format scores / limited conflict may explain small endpoint gain | [Shared diagnostic](../../reports/comparison/grpo_gdpo_shared_rollout_diagnostic.md), [sampled variation](../../reports/sft/fresh_holdout_reward_variation.md) | Plausible mechanism hypothesis only. Sampled-training format saturation and causal mediation are unproved. |
| Post-hoc internal holdout, exposure-audited | [Freeze audit](final_holdout_v1_freeze_audit.md), [historical usage](final_test_historical_usage_audit.md) | Audit-bounded internal endpoint from the optimizer-unused tail; not an official unseen test, independent source distribution or external benchmark. |

## Claim-language review

The reviewed principal reports are: final holdout completion, three-run comparison, three validation pairwise reports, paired bootstrap, primary80/sensitivity79, SFT v2 training/after-reward, shared rollout, fresh conflict and KL-path reports. Historical statements retain their timestamps and prelaunch annotations.

| Term family | Accepted use / action |
|---|---|
| improve / improvement | Fixed-checkpoint increments on the stated internal endpoint, supported by archived deltas/intervals. Never a general algorithm claim. |
| outperform / significant | No affirmative GDPO superiority or hypothesis-test significance claim in the new presentation. |
| robust / generalize | Training-seed robustness and generalization remain unsupported. Existing 80/79 “robustness” refers only to one identified duplicate-content exclusion. |
| unseen / unexposed | Official unseen-test wording is rejected. Final report's “previously-unexposed” is explicitly qualified by the persisted-artifact audit. |
| independent | Separate runs, model loading or closeout scorer checks where specified. This revision does not claim independent bootstrap reconstruction from raw validation JSONL. |
| equivalent / saturation | No algorithm equivalence, adapter bitwise equivalence or sampled-training format saturation claim. |

The README features four of the six frozen primary comparisons for readability and links all six. No comparison, endpoint, metric, bootstrap seed/index, row set or analysis artifact is changed.

## Presentation and repository boundary

- Reordered README around contributions, results and the objective-path finding; moved provenance detail behind technical content.
- Reorganized the reports index into six reader-oriented categories and preserved access to historical failure records.
- Added a personal interview-preparation summary in `docs/`; no personal contact details or secrets added.
- Preserved upstream attribution, licenses, directory layout and reproduction paths. No existing file deleted or moved.
- No scripts, configs, manifests, result JSON, bootstrap definitions or formal answer files changed.
- The upstream Pages workflow targets the `pages` branch and is not presented as experiment CI. No CI-passing badge is added.
- Use “35 training steps” for the trainer's batch/global-step budget, rather than equating it to 35 optimizer calls; PPO minibatches are a separate update-count layer.
- Repository About and all ten requested topics were updated and read back through GitHub's API; the default branch remains `main`.
- Verification checked rounded result/component means and the four displayed paired intervals against the frozen JSON, resolved all local presentation links, and byte-compared 736 existing evidence/code files outside the two rewritten README files against the result commit.
