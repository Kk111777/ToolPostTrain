# GRPO throughput Config B report

**Status: PASS.** Exactly one official-scale GRPO step completed through
optimizer.step. Config B ran because Config A passed and actor update was the
largest stage. No packages, environment, code, dataset, reward logic,
algorithm logic, formal training, or checkpoint were modified.

## Parser path

Config B retained the legal baseline parser name:

```
trainer.experiment_name=qwen2p5_1p5b_grpo_one_step
```

The throughput label stayed in project/output/log paths; rlla used the same
Qwen branch as baseline and Config A.

## Configuration

Config B equals A except dynamic token budget 6144 -> **12288**.

| field | value |
|---|---|
| initialization / estimator | RL_INIT_V1 / GRPO |
| train prompts / rollout | 512 / n=4 = 2048 trajectories |
| PPO mini-batch / physical microbatch | 128 prompts / 1 |
| actor/old/ref dynamic token budget | **12288** |
| max prompt / response | 2048 / 1024 |
| lr / seed | 1e-6 / 42 |
| vLLM TP / utilization / max seqs | 1 / 0.40 / 8 |
| vLLM max_num_batched_tokens | 6144 |
| reward, model, data, offload, algorithm | same as baseline |

## Timings (trainer timing_s)

| stage | seconds |
|---|---:|
| rollout generation | 244.859458 |
| reward aggregation | 0.000026 |
| old log-probability | 191.568879 |
| reference log-probability | 194.042526 |
| advantage | 0.122520 |
| actor forward/backward/update | 687.690835 |
| weight update | 0.687538 |
| **total** | **1320.015385** |

Baseline total: 1350.202769 s. Config B saved 30.187383 s
(**2.2358%**), but was 202.882017 s (**18.1609%**) slower than A; actor
update was the main regression.

## Memory and metrics

External GPU monitor: rollout peak approximately **46.68 GiB**; pre-actor
log-probability peak **56.71 GiB**; actor/update and overall peak
**57.55 GiB** of 86 GiB. Trainer actor memory:
35.74 GiB allocated and 55.70 GiB reserved.

| metric | value |
|---|---:|
| reward mean | 1.185765 |
| reward std | not emitted; no full-batch reward dump |
| advantage mean | -0.023780 |
| advantage std | not emitted; trainer emitted mean/min/max only |
| actor loss | 0.013631 |
| grad norm | 0.625116 |
| PPO KL | 0.00004274 |
| NaN/Inf | none |
| optimizer.step | PASS |

## Decision

B is functionally valid but not selected. Recommended execution setting is A:
vLLM utilization 0.40, max_num_seqs 8, max_num_batched_tokens 6144,
actor/old/ref token budget 6144, physical microbatch 1, with all algorithm,
reward, model, data, seed, length, and optimizer fields unchanged.

Artifacts:
`/root/autodl-tmp/ProjectB/env-modern/throughput-tuning-logs/grpo_throughput_config_b.stdout.log`
and the corresponding `throughput-tuning-monitors/grpo_throughput_config_b.gpu.csv`.
