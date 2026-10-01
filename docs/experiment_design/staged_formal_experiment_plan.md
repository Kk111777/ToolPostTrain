# ProjectB staged formal experiment plan

## 1. Why Stage 1 is 35 steps

The measured Config A cost is about 1117.1 seconds per optimizer step. A
105-step GRPO run is therefore roughly 32.6 hours before checkpoint and
evaluation overhead, and a paired GRPO/GDPO study is roughly 65 hours.
A 35-step first stage covers five complete data epochs (7 steps each),
produces a meaningful early learning curve, and limits wasted compute if
the held-out signal does not improve.

This is a budget decision only. It does not change the model, data,
reward, optimizer, rollout, or advantage semantics.

## 2. Why 35 -> 70 -> 105

The budgets are aligned to complete epochs:

- 35 steps = 5 epochs
- 70 steps = 10 epochs
- 105 steps = 15 epochs and the maximum/full reproduction budget

Using complete epochs avoids stopping in the middle of the fixed
3901-row, drop_last=true schedule. GRPO and GDPO always use the same
selected budget.

## 3. Train, validation, and test protocol

The optimizer sequence is fixed before Stage 1:

- TRAIN: the first `3584` eligible prompts per epoch (`7 * 512`), with
  `shuffle=false` and `drop_last=true`.
- VALIDATION: the derived runtime view
  `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet`, fixed by
  `/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json`,
  from the permanent 317-row drop-last tail. It is disjoint from the actual
  RL train prefix, both format-SFT source unions, and all explicitly tracked
  diagnostic rows. The split proof is in
  `/root/autodl-tmp/ProjectB/env-modern/formal_validation_split_audit.md`.
- TEST: `test.parquet` is a final-only holdout and is not used for any
  intermediate evaluation or budget/model-selection decision.

Use the same 80 validation rows in fixed manifest order for both algorithms.
Evaluate at step 0, 7, 14, 21, 28, and 35 using:

- `n=1`
- greedy decoding: `temperature=0`, `do_sample=false`, `top_k=-1`,
  `top_p=1.0`
- the same tokenizer/chat template, vLLM backend, reward manager, max
  lengths, and validation manifest

Record accuracy/correctness reward, format reward, total reward, tool-call
parse rate, response-only wrapper rate, generation length, and all available
native validation metrics. The current validation path does not produce
actor/reference KL; report training `actor/ppo_kl` separately. Validation has
no `optimizer.step`.

If Stage 2 is approved, extend the same formal-validation schedule with
steps 42, 49, 56, 63, and 70. If Stage 3 is approved, continue with the same
7-step cadence through 105.

## 4. Paired budget-selection statistic

After both algorithms complete the same stage budget, compare step 28 versus
step 35 on the same 80 prompts, paired by `source_row`. For total reward and
accuracy reward separately, record:

- mean paired difference;
- fraction of prompts with an improvement;
- deterministic paired-bootstrap percentile 95% interval, with seed 42 and
  10,000 resamples.

A small improvement whose interval crosses zero is treated as no clear
continuation benefit. Continue only when the paired improvement is stably
positive and format reward, parse/wrapper rates, and KL show no material
regression. The identical rule is applied to the 70 -> 105 decision.

## 5. Test-set firewall

Before the final common budget is frozen, `test.parquet` is forbidden for:

- early stopping;
- deciding whether to continue from 35 to 70 or 70 to 105;
- hyperparameter or execution-configuration selection;
- checkpoint selection.

After the final budget is fixed, run exactly one formal final evaluation on
`test.parquet` for the final GRPO checkpoint and one for the final GDPO
checkpoint. If a KL-aligned ablation is later added, it follows the same
final-only test rule.

## 6. Fair GRPO/GDPO comparison

For every stage:

- start both runs from RL_INIT_V1, not from the other algorithm;
- use the same 512-prompt batches, filter result, order, seed, rollout.n,
  sampler, max lengths, LR, microbatch, vLLM Config A, and reward manager;
- use the same frozen held-out 80 rows and greedy evaluation settings;
- change only the advantage estimator and GDPO reward-key path;
- compare only equal step budgets.

Separate run directories prevent one algorithm's optimizer state or
checkpoint from becoming the other's initialization.

## 7. Checkpoint and disk handling

Stage 1 full checkpoints are at steps 10, 20, 30, and the last step 35.
The official retention setting keeps one recent full checkpoint, with two
full directories allowed transiently during a safe replacement. Export
model-only step35 and validate it.

If the experiment stops at35, retain the step35 model-only snapshot and
delete the temporary full resume checkpoint after validation. If it
continues, retain the validated step35 full checkpoint and resume to the
common 70-step budget. The same pattern applies at70 and105.

## 8. How to describe a 35-step result

Correct: “Stage-1 35-step controlled GRPO/GDPO learning-curve result.”
Incorrect: “complete 105-step reproduction,” “converged final model,” or
“full benchmark result.”

A 35-step result answers whether the reward/format signal is moving under
the frozen setup. It does not establish final convergence or replace the
maximum 105-step reproduction.
