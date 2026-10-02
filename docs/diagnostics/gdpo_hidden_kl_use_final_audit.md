# GDPO hidden KL-use final audit

- Audit status: PASS (read-only source audit; no model was loaded and no GPU training was started).
- Source tree: /root/autodl-tmp/ProjectB/upstream/verl-v0.9.1.
- Formal GDPO configuration: algorithm.adv_estimator=gdpo, algorithm.gdpo_reward_keys=[accuracy_reward,format_reward], algorithm.use_kl_in_reward=True, actor_rollout_ref.actor.use_kl_loss=False.
- Decision: GDPO_OBJECTIVE_REWARD_KL_FREE = PASS.

## Verified data flow

compute_score / reward manager
  -> token_level_scores = aggregate reward tensor
  -> non_tensor_batch[accuracy_reward] and non_tensor_batch[format_reward]

use_kl_in_reward=True
  -> token_level_rewards = token_level_scores - beta * KL(old_log_probs, ref_log_prob)

GDPO advantage dispatch
  -> gdpo_reward_keys -> raw accuracy_reward and format_reward
  -> per-component group normalization
  -> sum / batch masked whitening
  -> advantages
  -> actor policy loss -> backward -> optimizer.step

KL-adjusted token_level_rewards
  -> retained for reward/sequence statistics; not consumed by the GDPO
     component-advantage branch above.

## Code evidence

1. verl/trainer/ppo/ray_trainer.py:1633-1645 stores the reward tensor as token_level_scores; with use_kl_in_reward=True, apply_kl_penalty writes KL-adjusted token_level_rewards.
2. verl/trainer/ppo/ray_trainer.py:1635-1636 copies reward-manager extras into batch.non_tensor_batch.
3. verl/trainer/ppo/ray_trainer.py:1662-1675 calls compute_advantage after reward construction.
4. verl/trainer/ppo/core_algos.py:361-468 implements GDPO. With the formal reward keys, lines 412-433 build score_list from raw non-tensor reward components; token_level_rewards is only a fallback when component keys are absent. Lines 451-468 normalize those component scores and return the advantages used by the policy update.
5. verl/workers/utils/losses.py policy-loss path consumes advantages, old_log_probs, and response_mask; it does not read token_level_rewards. The actor KL term is conditional on actor.use_kl_loss, which is false in this GDPO run.
6. verl/trainer/ppo/metric_utils.py:453-454 reads token_level_scores and token_level_rewards for sequence metrics. This is logging/statistics, not the optimizer objective.
7. The formal effective configuration has no enabled algorithm.filter_groups and no rollout-correction configuration. The audited correction/rejection-sampling paths operate on token_level_scores/importance quantities, not KL-adjusted token_level_rewards; no KL-reward sample filter, loss weighting, mask, group selection, or optimizer weighting is active in this run.

## Explicit answers to hidden-influence checks

| Path | KL-adjusted token_level_rewards affects optimizer update? | Evidence |
|---|---:|---|
| GDPO component advantage | No | raw non_tensor_batch reward keys are used when configured |
| actor policy loss | No | loss consumes advantages; no reward tensor read |
| loss weighting / masks | No | current policy loss uses response/rollout masks and advantages |
| sample filtering / group filtering | No | no such formal feature enabled; audited active correction path uses scores/weights |
| importance weighting | No | no active KL-reward importance path; correction uses rollout probability quantities |
| optimizer step | No direct path | optimizer receives gradients from actor loss |
| logging / sequence reward statistics | Yes, statistics only | metric_utils.py |

## Reference-policy/runtime audit for noKL

verl/trainer/ppo/utils.py:78-79 defines need_reference_policy(config) as algorithm.use_kl_in_reward OR actor_rollout_ref.actor.use_kl_loss. Therefore, with both values false:

- need_reference_policy=False;
- the reference-policy role is not created by the trainer;
- the training loop does not call _compute_ref_log_prob (ray_trainer.py:1618-1622 is conditional on self.use_reference_policy);
- the actor loss does not require ref_log_prob because the actor KL-loss flag is false.

This is the intended no-reference runtime semantics of the noKL ablation, not an accidental third training variable. It also means GRPO-noKL and GDPO-current must not be compared for wall-clock time, GPU memory, or reference-policy efficiency; the planned comparison is training effect on the frozen validation endpoint.

The one-step GPU smoke must still verify: no missing ref_log_prob error, no reference worker/model initialization, finite reward/advantage/loss/grad, and a real optimizer step. The smoke checkpoint must not be reused for formal training.

## Boundary

This PASS means only that, in official v0.9.1 and the frozen GDPO config, KL-adjusted aggregate rewards do not enter the GDPO optimizer objective through an unexamined hidden path. KL still causes reference-policy computation and affects logged KL/reward statistics in the existing GDPO run. The current GRPO-vs-GDPO results therefore retain a KL-treatment/reward-dimension confound and cannot support a pure advantage-estimator causal claim.
