# ProjectB KL path audit

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Initial two-run source/smoke audit; its not-yet-ablated statement predates completed noKL35. Original facts and values below are preserved. Current artifacts: [three-run completion](final_stage1_ablation_completion_report.md); current budget: [35-step decision](../../docs/experiment_design/stage1_closeout_decision.md).

This is a read-only source audit paired with the two controlled one-step runtime logs. No KL implementation or configuration was changed during the audit.

## Runtime configuration

- Both runs started from the same frozen RL_INIT_V1 checkpoint: `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged`.
- The smoke driver set `algorithm.use_kl_in_reward=True`, `algorithm.kl_penalty=kl`, and `algorithm.kl_ctrl.kl_coef=0.001`.
- It set `actor_rollout_ref.actor.use_kl_loss=False`; therefore this audit concerns the reward-side KL path, not an auxiliary actor KL loss.
- Runtime `actor/reward_kl_penalty`: GRPO `0`, GDPO `0`.
- Runtime `actor/reward_kl_penalty_coeff`: GRPO `0.001`, GDPO `0.001`.
- `rollout_corr/kl` was GRPO `0.000652491639` and GDPO `0.0010771919`. This is rollout-vs-training correction telemetry, not the reward-side `apply_kl_penalty` value.

## Actual tensor path

1. `GDPORewardManager.run_single()` decodes the response and calls `rlla.compute_score`; when the result is a dict it returns scalar `score` plus `reward_extra_info` containing `format_reward` and `accuracy_reward` (source `verl/experimental/reward_loop/reward_manager/gdpo.py:79-90`).
2. The trainer stores scalar reward in `batch.batch["token_level_scores"]` and copies reward extras into `batch.non_tensor_batch` (source `verl/trainer/ppo/ray_trainer.py:1635-1636`).
3. Because `use_kl_in_reward=True`, `apply_kl_penalty()` computes KL from `old_log_probs` and `ref_log_prob`, then writes `batch.batch["token_level_rewards"] = token_level_scores - beta * kld` (source `ray_trainer.py:94-113`).
4. GRPO calls `compute_grpo_outcome_advantage()` with `data.batch["token_level_rewards"]`; it sums this tensor per response and group-normalizes it (source `ray_trainer.py:235-247`, `core_algos.py:304-329`). Therefore current GRPO uses KL-penalized total/token rewards.
5. GDPO passes both `non_tensor_batch` and `batch` to `compute_gdpo_outcome_advantage()` (source `ray_trainer.py:248-263`). With `gdpo_reward_keys=["accuracy_reward","format_reward"]`, GDPO reads those raw non-tensor component arrays, places them at the final response token, normalizes each component independently, sums them, and batch-whitens the result (source `core_algos.py:412-466`). It does not use the KL-adjusted `token_level_rewards` when component keys are present.

## Conclusion

- GRPO KL treatment: **KL-penalized aggregate/token-level reward tensor**.
- GDPO KL treatment: **raw per-dimension reward tensors; KL penalty is not injected before component normalization**.
- Current implementation has a **real KL-path inconsistency** between GRPO and GDPO. This is the previously identified confound and was only recorded, not corrected or ablated.
- In these two initial-policy smoke runs the reward-side KL metric happened to be `0.0`, but that numerical value does not remove the code-path inconsistency for later policy updates.
