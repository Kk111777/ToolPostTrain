# ProjectB staged formal experiment configuration

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Pre-run design snapshot; NOT STARTED and the original continuation/retention rules describe that point in time. Original facts and values below are preserved. Current artifacts: [three-run completion](../reports/comparison/final_stage1_ablation_completion_report.md); current budget: [35-step decision](../docs/experiment_design/stage1_closeout_decision.md).

## Current status

**STAGE 1 / NOT STARTED.** The first formal budget is now 35 optimizer
steps, not 105. A previous 105-step supervisor was stopped because the
experiment budget changed; it had not emitted a step-1 trainer metric and
had saved no checkpoint. Its log remains under
/root/autodl-tmp/ProjectB/runs/grpo_formal_v1 for audit and is not an
initialization checkpoint.

The 105-step budget remains the maximum/full-reproduction budget. It is not
the default first-stage budget.

## Frozen common initialization and semantics

GRPO and GDPO both start from the exact same RL_INIT_V1:

- /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged

The following remain unchanged:

- raw train rows 3920; overlength rows removed 19; eligible rows 3901
- tokenizer/chat-template filtering with max prompt length 2048
- shuffle=false and drop_last=true
- 512 prompts per algorithm batch; 7 complete batches per epoch
- the optimizer sees the fixed first 3584 eligible prompts per epoch; the 317-row tail is never in the optimizer
- rollout.n=4, giving 2048 trajectories per optimizer step
- PPO mini-batch 128 prompts
- physical microbatch 1
- dynamic token budget 6144
- max prompt/response 2048/1024
- AdamW learning rate 1e-6
- FSDP actor, actor offload=false, reference parameter offload=true
- vLLM Config A: TP=1, gpu_memory_utilization=0.40,
  max_num_seqs=8, max_num_batched_tokens=6144
- seed 42 and sampler/data-loader order
- official verl commit 1876b06d0a3e4e71e06230be10af14492ca8a75b
- ProjectB commit c105a5e8e009339849d9a2247209ea1494a4043d

The only algorithm-level comparison remains GRPO versus GDPO advantage
estimation. The legal parser experiment names remain
qwen2p5_1p5b_grpo_one_step and qwen2p5_1p5b_gdpo_one_step.

## Staged budget

| stage | epochs | steps/epoch | total steps | interpretation |
|---|---:|---:|---:|---|
| Stage 1 | 5 | 7 | 35 | initial learning-curve and feasibility result |
| Stage 2 | 10 | 7 | 70 | extend only if Stage 1 still improves |
| Stage 3 | 15 | 7 | 105 | maximum/full reproduction budget |

GRPO and GDPO must use the same stage budget. A 35-step GRPO result
cannot be compared as the primary result against a 70- or 105-step GDPO
result.

## Train/validation/test separation

The actual optimizer sequence is frozen and unchanged:

- TRAIN: the first `3584` prompts of the `3901` eligible train rows after the
  exact tokenizer/chat-template overlength filter; `drop_last=true` makes
  this `7 * 512` prompts per epoch.
- VALIDATION: the native-loader view
  `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet`, whose
  exact source-row manifest is
  `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json`.
  It contains exactly 80 rows sampled from the permanent 317-row drop-last tail. These
  rows are not passed to the optimizer and are used at steps `0, 7, 14, 21,
  28, 35`.
- TEST: `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/test.parquet`,
  exactly 80 rows, held back from all intermediate evaluation and all budget,
  hyperparameter, early-stopping, and checkpoint decisions. It is evaluated
  once per final GRPO/GDPO checkpoint only after the common budget is frozen.

The split proof and exclusion accounting are in:

- `/root/autodl-tmp/ProjectB/env-modern/formal_validation_split_audit.md`
- `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json`
- `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet` (derived runtime view)

GRPO and GDPO must use the same formal-validation manifest and the same
validation decoding/reward settings:

- `n=1`
- greedy decoding: `temperature=0`, `do_sample=false`, `top_k=-1`,
  `top_p=1.0`
- same tokenizer/chat template, vLLM backend, reward manager, and max
  prompt/response lengths `2048/1024`

Each validation point records accuracy/correctness reward, format reward,
total reward, tool-call parse rate, response-only wrapper rate, generation
length, and available native validation metrics. The current validation path
does not compute actor/reference KL; training `actor/ppo_kl` remains a
separate training metric and is not mislabeled as held-out validation KL.
Validation never calls `optimizer.step`.

For the Stage-1 budget decision, compare step `28` and step `35` paired by
`source_row`. For total reward and accuracy reward separately, report the
mean paired difference, the fraction of prompts improved, and a deterministic
paired bootstrap percentile 95% interval using seed `42` and `10,000`
resamples. A small change whose interval crosses zero is no clear
continuation benefit. Continue only for stable positive improvement without
material format/parse/KL regression.

## Checkpoint and disk policy

For Stage 1, save full resumable checkpoints at steps 10, 20, 30, and the
last step 35. Keep only the newest full checkpoint after a successful save;
the old one may coexist only while the new one is being validated. The
step-35 full checkpoint is retained temporarily until the Stage 1 decision:

- if stopping at 35, validate model-only step35, then delete full step35;
- if continuing to 70, retain validated full step35 as the resume point;
- future 70 and 105 boundaries use the same policy.

Save permanent model-only step35. Do not create step100/105 model-only
snapshots during Stage 1. At later boundaries, model-only snapshots are
step70 and step105 only.

Measured sizes:

- one full resume checkpoint: 19,469,701,513 bytes / 18.132573 GiB
- one model-only checkpoint: 3,098,893,848 bytes / 2.886070 GiB

At the Stage 1 decision, each completed run retains at most one
model-only snapshot and, only if continuing, one full resume checkpoint.
GRPO and GDPO runs should be kept in separate directories and executed
sequentially. This leaves ample disk headroom on the current 100 GiB data
volume; no important existing artifact is to be deleted automatically.

## Time estimate

Measured execution-only Config A is about 1117.1 seconds per optimizer
step.

- 35 steps, compute only: 39,098.5 seconds = 10.86 hours per run.
- Add five intermediate evals plus step-0 baseline: approximately 10-25
  minutes per run under the fixed 80-row greedy evaluation.
- Add checkpoint/model-only export overhead: approximately 3-10 minutes
  per run.
- Practical estimate: **11.1-11.5 hours per GRPO or GDPO Stage-1 run**.
- Practical estimate for both sequential runs: **22.2-23.0 hours**.

Stage 2 and Stage 3 use the same per-step rate; their compute-only
estimates are approximately 21.7 hours and 32.6 hours per run,
respectively.

## Continuation gates

Continue from 35 to 70 only if the predeclared step-28 to step-35
comparison shows real improvement on the paired held-out samples:

1. accuracy/correctness and total reward improve, with the paired bootstrap
   interval or an equivalent fixed-sample uncertainty check supporting a
   non-noise improvement;
2. format reward and parse/wrapper rates do not materially regress
   (use a 5 percentage-point drop as the warning threshold);
3. all training metrics remain finite and KL is not unstable.

Stop at 35 if the curve is flat within held-out noise, format/parse quality
regresses, or the run has numerical instability.

At 70, continue to 105 only if the last evaluation interval still improves,
the curve has not stabilized, or a full 105-step reproduction is explicitly
needed. Stop at 70 when reward and format behavior have plateaued over the
last two evaluation intervals without instability.

## Interpretation boundary

A 35-step result is a staged learning-curve/mechanism result. It is useful
for deciding whether additional budget is justified, but it must not be
described as a complete 105-step reproduction, final benchmark, or
converged training result.
