# GRPO-original vs GRPO-noKL final effective-config diff

Status: PASS. Static audit only: no noKL model was loaded and no GPU training was started.

## Machine-readable evidence

- Baseline: /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/effective_config_prelaunch.yaml
- Proposed: /root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/effective_config_prelaunch.yaml
- Hydra mode: --cfg job --resolve
- Hydra result: PASS
- Unexpected differences: 0

## Allowed differences

| Path | GRPO-original35 | GRPO-noKL35 | Meaning |
|---|---|---|---|
| algorithm.use_kl_in_reward | True | False | the sole planned reward-side KL ablation |
| trainer.default_local_dir | '/root/autodl-tmp/ProjectB/runs/grpo_stage1_35/checkpoints' | '/root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/checkpoints' | isolated run/output path |
| trainer.rollout_data_dir | '/root/autodl-tmp/ProjectB/runs/grpo_stage1_35/.rollout_capture' | '/root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/.rollout_capture' | isolated run/output path |
| trainer.validation_data_dir | '/root/autodl-tmp/ProjectB/runs/grpo_stage1_35/eval_generations' | '/root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/eval_generations' | isolated run/output path |

## Frozen fields verified equal

- RL_INIT_V1 path and step-0 initialization.
- algorithm.adv_estimator=grpo.
- actor_rollout_ref.actor.use_kl_loss=False.
- Dataset, 3901 eligible prompts, source order, shuffle/drop_last semantics, batch 512, rollout.n=4, PPO minibatch 128, physical microbatch 1.
- Learning rate 1e-6, lengths 2048/1024, seed 42, formal_validation_80, greedy n=1 at 0/7/14/21/28/35.
- vLLM Config A and reward manager/parser.
- Legal trainer.experiment_name=qwen2p5_1p5b_grpo_one_step.

## Reference-policy runtime

Official v0.9.1 defines need_reference_policy as algorithm.use_kl_in_reward OR actor_rollout_ref.actor.use_kl_loss.
Therefore original needs a reference-policy path, while noKL with both flags false does not instantiate or call the reference-policy/ref_log_prob path. This is the expected runtime consequence of the planned ablation. It makes wall-clock and GPU-efficiency comparisons with GDPO-current invalid; the intended comparison is training effect.

A one-step smoke is still required before formal training to verify no missing ref_log_prob error, no reference worker/model initialization, finite reward/advantage/loss/grad, and a real optimizer step. The smoke checkpoint must not be reused for the formal 35-step run.
