# GRPO Stage 1 (35-step) completion report

Generated from persisted artifacts at 2026-10-01T23:10:26.715530+00:00. No training was rerun.

## A. Formal configuration summary

- Initialization: /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged (RL_INIT_V1).
- Raw train rows: 3920; eligible after the official overlong filter: 3901.
- shuffle=false, drop_last=true, train batch 512 prompts, 7 batches/epoch.
- Stage budget: 5 epochs / 35 optimizer steps; rollout.n=4; PPO minibatch 128; physical microbatch 1; dynamic token budget 6144.
- Prompt/response limits: 2048 / 1024; learning rate 1e-6.
- vLLM Config A: TP=1, gpu_memory_utilization=0.40, max_num_seqs=8, max_num_batched_tokens=6144.
- Intermediate validation: /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet; test.parquet was not used.

## B. Completion status

- run_status.json: phase=completed, exit_code=0.
- Global step reached 35/35 and the trainer exited normally.
- No optimizer rerun or resume was performed while generating this report.

## C. Formal validation learning curve

| step | total reward mean | format reward mean | accuracy reward mean |
|---:|---:|---:|---:|
| 0 | 2.400171 | 0.950000 | 1.450171 |
| 7 | 2.717582 | 0.950000 | 1.767582 |
| 14 | 2.799781 | 0.950000 | 1.849781 |
| 21 | 2.785379 | 0.950000 | 1.835379 |
| 28 | 2.831808 | 0.950000 | 1.881808 |
| 35 | 2.799721 | 0.950000 | 1.849721 |

- Validation artifacts are the six persisted eval_generations/{0,7,14,21,28,35}.jsonl files.
- The native validation logger emitted these reward dimensions; no unrecorded validation statistic was inferred.

## D. Training reward and optimization metrics

- Training score mean (35 persisted metric records): n=35, min=1.22149, max=2.78911, mean=2.35665.
- Actor loss: n=35, min=-0.00811792, max=0.0360122, mean=0.0125332; finite=True.
- Actor grad norm: n=35, min=0.329626, max=0.617412, mean=0.411523; finite=True.
- Actor PPO KL: n=35, min=-5.72039e-05, max=0.000213699, mean=3.24737e-05; finite=True.
- Step time: n=35, min=1073.7, max=1118.51, mean=1089.53 seconds.
- Native logs did not provide a separate compact per-step format/accuracy reward aggregate for the training batches; the report does not fabricate one.
- Native logs recorded actor memory up to approximately 29.435 GiB allocated and 45.918 GiB reserved.

## E. Error status

- Fatal-pattern scan of persisted training stdout: no matching fatal pattern.
- Finite metric checks: loss=True, grad_norm=True, ppo_kl=True, step_time=False, reward=True.
- Recorded run status and checkpoint artifacts indicate no OOM, NaN/Inf, CUDA fatal error, or parser fatal error.

## F. Checkpoints and artifacts

- Full resume: /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/checkpoints/global_step_35; exists=True; bytes=19469701041.
- Model-only: /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/model_only/step_35; exists=True; bytes=3098893848.
- Independent load/tokenizer/chat-template checks: True/True/True.
- Diagnostic manifest: /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/diagnostics/manifest.json; content={"errors": [], "kept_steps": [1, 20, 35], "max_rows_per_step": 64}.
- The old step10/20/30 directories are only 4 KiB metadata markers; the retained full resume is step35.

## G. Continuation policy

- This is a valid Stage 1 35-step result and may be used as a resume point for a future 35->70 continuation.
- 35 steps is NOT the FINAL_BUDGET; no automatic continuation to 70 or 105 is implied.
- GDPO must remain independently initialized from RL_INIT_V1, not from the GRPO step35 model.

