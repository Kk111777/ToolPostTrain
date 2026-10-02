# GDPO Stage 1 - 35-step report

## A. Completion status

- Status: PASS; global_step=35; trainer exited with code 0.
- This is a staged 35-step run. No automatic continuation to 70 or 105 steps was started.
- The step-35 checkpoint is retained as a future resume candidate; this report does not claim that 35 is the final research budget.

## B. Effective configuration

- Initialization: /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged (RL_INIT_V1).
- Algorithm: GDPO; reward dimensions: accuracy_reward, format_reward.
- Train data: raw 3920; eligible 3901 after the official overlength filter; shuffle=false; drop_last=true; batch 512; 7 full batches/epoch; 5 epochs; 35 optimizer steps.
- Validation: /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet; greedy n=1 at steps 0/7/14/21/28/35; not used by the optimizer.
- No test.parquet was used for intermediate validation.
- Lengths: prompt 2048, response 1024; learning rate 1e-6; rollout n=4.
- Execution: vLLM TP=1, GPU memory utilization 0.40, max_num_seqs=8, max_num_batched_tokens=6144; PPO mini-batch 128; physical microbatch 1.

## C. Formal validation learning curve

| step | total reward mean | format reward mean | accuracy reward mean | tool parse | response wrapper |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.400171 | 0.950000 | 1.450171 | 73/80 (91.25%) | 5/80 (6.25%) |
| 7 | 2.611385 | 0.962500 | 1.648885 | 74/80 (92.50%) | 4/80 (5.00%) |
| 14 | 2.798260 | 0.962500 | 1.835760 | 75/80 (93.75%) | 4/80 (5.00%) |
| 21 | 2.768737 | 0.962500 | 1.806237 | 75/80 (93.75%) | 4/80 (5.00%) |
| 28 | 2.781237 | 0.962500 | 1.818737 | 75/80 (93.75%) | 4/80 (5.00%) |
| 35 | 2.820522 | 0.962500 | 1.858022 | 75/80 (93.75%) | 4/80 (5.00%) |

All validation reward fields were finite.

## D. Training metrics

The compact metric artifact is metrics/stage1_metrics.json. The recorded training metric ranges are:
- critic/score/mean: count=35, min=1.2439838647842407, max=2.8541955947875977, mean=2.417723992892674, finite=True
- actor/loss: count=35, min=-0.017813362181186676, max=0.005842055659741163, mean=-0.003110953067828502, finite=True
- actor/grad_norm: count=35, min=0.4611491188406944, max=0.89798903465271, mean=0.5959462798067502, finite=True
- actor/ppo_kl: count=35, min=-7.160778313424068e-05, max=0.00048732771999015866, mean=5.77106192780651e-05, finite=True
- perf/time_per_step: count=35, min=1055.50261386903, max=1111.2302577069495, mean=1078.80035222983, finite=True
- gdpo/accuracy_reward/mean: count=35, min=0.4432026147842407, max=1.9142541885375977, mean=1.5004528062684195, finite=True
- gdpo/format_reward/mean: count=35, min=0.80078125, max=0.95068359375, mean=0.9172712053571429, finite=True

## E. Stability and artifacts

- No OOM, fatal CUDA error, parser fatal error, NaN, or Inf was found in the completed run logs; process exit code was 0.
- Full resume checkpoint: /root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/checkpoints/global_step_35.
- Model-only checkpoint: /root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/model_only/step_35.
- Checkpoint manifest: checkpoint_manifest.json; diagnostic manifest: diagnostics/manifest.json; compact metrics: metrics/stage1_metrics.json.
- The full step-35 checkpoint and model-only export were produced after training completion; GPU was released afterward.

## F. Next-stage note

The run is intentionally stopped at Stage 1, 35 steps. No 70/105-step training, GRPO-noKL run, or third GPU experiment was started. Any continuation requires a later explicit decision.
