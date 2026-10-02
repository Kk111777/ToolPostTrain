# GRPO-noKL Stage 1: 35-step report

## Status

- Completion: 35/35 optimizer steps completed; run_status.json reports phase=completed and exit_code=0.
- Initialization: /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged (independent step 0).
- Budget: 5 epochs, 35 steps; this is Stage 1 and is not a final-budget decision.
- Validation: /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet, greedy n=1, steps 0/7/14/21/28/35.
- Test policy: test.parquet was not used for intermediate validation.
- Training data semantics: raw 3920 -> 3901 eligible; shuffle=false; batch 512; 7 full batches/epoch; drop-last semantics; 35 expected optimizer steps.

## Frozen configuration summary

- algorithm.adv_estimator=grpo
- algorithm.use_kl_in_reward=false
- actor_rollout_ref.actor.use_kl_loss=false
- RL_INIT_V1, seed 42, source order unchanged
- rollout n=4; PPO minibatch 128; physical microbatch 1
- dynamic token budget 6144; prompt/response limits 2048/1024; learning rate 1e-6
- Config A vLLM: TP=1, GPU utilization 0.40, max sequences 8, max batched tokens 6144
- parser/reward: qwen2p5_1p5b_grpo_one_step, rlla.py, reward.manager=gdpo

## Formal validation learning curve (primary 80)

| step | total reward | accuracy reward | format reward | strict format | tool parse | response wrapper |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 1.450171 | 0.950000 | 76/80 (95.00%) | 73/80 (91.25%) | 5/80 (6.25%) |
| 7 | 2.588245 | 1.638245 | 0.950000 | 76/80 (95.00%) | 73/80 (91.25%) | 5/80 (6.25%) |
| 14 | 2.757760 | 1.807760 | 0.950000 | 76/80 (95.00%) | 74/80 (92.50%) | 5/80 (6.25%) |
| 21 | 2.784451 | 1.821951 | 0.962500 | 77/80 (96.25%) | 75/80 (93.75%) | 4/80 (5.00%) |
| 28 | 2.781237 | 1.818737 | 0.962500 | 77/80 (96.25%) | 75/80 (93.75%) | 4/80 (5.00%) |
| 35 | 2.818737 | 1.856237 | 0.962500 | 77/80 (96.25%) | 75/80 (93.75%) | 4/80 (5.00%) |

The values above are raw per-example validation means from persisted eval_generations/*.jsonl; they are not training reward means.

## Training diagnostics (steps 1-35)

| metric | mean | min | max | samples | finite |
|---|---:|---:|---:|---:|:---:|
| actor loss | 0.007041 | -0.007567 | 0.033520 | 35 | PASS |
| grad norm | 0.314941 | 0.217709 | 0.616645 | 35 | PASS |
| actor PPO KL | 0.000032 | -0.000064 | 0.000274 | 35 | PASS |
| step time (s) | 905.334667 | 880.475636 | 941.109704 | 35 | PASS |

## Integrity and failure checks

- Reward, accuracy, format, actor loss, grad norm, actor PPO KL, and step time were finite in the persisted records: PASS.
- OOM/NaN/Inf/CUDA/parser/checkpoint fatal scan: PASS.
- Trainer completion state: PASS (phase=completed, exit code 0).
- Step35 full resume structural files: PASS; verified size 19469700977 bytes.
- Step35 model-only files: PASS; verified size 3098893848 bytes; optimizer is not included.
- Diagnostics manifest: diagnostics/manifest.json reports no compactor errors and retained steps 1/20/35.
- Artifact reconciliation: the launcher-emitted manifest was preserved as checkpoint_manifest_launcher_original.json; it named the GRPO run path, so checkpoint_manifest.json was replaced with the verified noKL-specific manifest without changing any checkpoint bytes.

## Continuation boundary

The step35 full resume checkpoint is retained and can support a future 35->70 continuation after an explicit user decision and a fresh preflight. This Stage 1 completion does not authorize automatic 70/105 training.
