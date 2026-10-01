# GRPO throughput Config A report

**Status: PASS.** Exactly one official-scale GRPO step completed:
rollout -> reward -> old/reference log-probability -> advantage -> actor
backward -> optimizer.step. No packages, environment, code, dataset, reward
logic, algorithm logic, formal training, or checkpoint were modified.

## Phase 0: experiment-name audit

The failed attempt set `trainer.experiment_name=grpo_config_a`. In this
runtime `main_ppo.py` exports that field as Ray `EXPERIMENT_NAME`;
`verl/utils/reward_score/rlla.py::compute_score` reads it and dispatches
only names containing `qwen` or `llama`. Thus `grpo_config_a` raised
`NotImplementedError: Unknown model name: grpo_config_a`.

The legal baseline value was restored unchanged:

```
qwen2p5_1p5b_grpo_one_step
```

Throughput labels used independent project/output/stdout/monitor paths
(`ProjectB_throughput_tuning`, config_a output directory, and
`grpo_throughput_config_a.*` files), so no parser field was used as a label.

## Configuration

| field | value |
|---|---|
| initialization / estimator | RL_INIT_V1 / GRPO |
| train prompts / rollout | 512 / n=4 = 2048 trajectories |
| PPO mini-batch / physical microbatch | 128 prompts / 1 |
| actor/old/ref dynamic token budget | 6144 |
| max prompt / response | 2048 / 1024 |
| lr / seed | 1e-6 / 42 |
| reward | baseline gdpo manager + rlla |
| vLLM TP / utilization | 1 / **0.40** |
| vLLM max_num_seqs / max_num_batched_tokens | **8 / 6144** |
| offload | actor=false, reference=true |
| algorithm-level fields | unchanged |

## Timings (trainer timing_s)

| stage | seconds |
|---|---:|
| rollout generation | 246.667251 |
| reward aggregation | 0.000031 |
| old log-probability | 174.793992 |
| reference log-probability | 183.875845 |
| advantage | 0.151338 |
| actor forward/backward/update | 510.297686 |
| weight update | 0.506206 |
| **total** | **1117.133368** |

Baseline total: 1350.202769 s. Config A saved 233.069400 s
(**17.2618%**). Stage deltas: generation -229.398254 s, old log-prob
-0.773345 s, reference +0.112010 s, actor update -1.866913 s. Actor
update remained the largest stage, so Config B was tested.

## Memory and metrics

External GPU monitor: rollout peak approximately **46.68 GiB**; actor/update
and overall peak **52.27 GiB** of 86 GiB. Trainer actor memory:
28.84 GiB allocated and 39.97 GiB reserved.

| metric | value |
|---|---:|
| reward mean | 1.238398 |
| reward std | not emitted; no full-batch reward dump |
| advantage mean | -0.032605 |
| advantage std | not emitted; trainer emitted mean/min/max only |
| actor loss | 0.023394 |
| grad norm | 0.614580 |
| PPO KL | 0.00007405 |
| NaN/Inf | none |
| optimizer.step | PASS |

Partial per-worker reward prints were not treated as full-batch statistics.

Artifacts:
`/root/autodl-tmp/ProjectB/env-modern/throughput-tuning-logs/grpo_throughput_config_a.stdout.log`
and the corresponding `throughput-tuning-monitors/grpo_throughput_config_a.gpu.csv`.
