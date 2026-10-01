# GDPO throughput Config A report

## Result

**PASS.** Exactly one official-scale GDPO step completed:
rollout -> reward -> old log-prob -> reference log-prob -> GDPO advantage ->
actor forward/backward -> optimizer.step.

The Ray metrics emitted `training/global_step:1`, non-zero gradient norm,
and complete timing. No OOM, NaN, or Inf was observed. GPU was released.

## Configuration

| field | value |
|---|---|
| initialization / estimator | RL_INIT_V1 / GDPO |
| model | `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged` |
| train prompts / rollout | 512 / n=4 = 2048 trajectories |
| PPO mini-batch / physical microbatch | 128 prompts / 1 |
| actor/old/ref dynamic token budget | 6144 |
| max prompt / response | 2048 / 1024 |
| learning rate / seed | 1e-6 / 42 |
| vLLM TP / utilization | 1 / 0.40 |
| vLLM max_num_seqs / max_num_batched_tokens | 8 / 6144 |
| actor / reference offload | actor=false / reference parameter=true |
| reward manager / scorer | gdpo / rlla.py::compute_score |
| GDPO reward keys | accuracy_reward, format_reward |
| parser-compatible experiment name | qwen2p5_1p5b_gdpo_one_step |

The parser-compatible experiment name was unchanged. Throughput labels used
only project/output/log paths.

## Timing

| stage | seconds |
|---|---:|
| rollout generation | 250.044093 |
| reward aggregation | 0.000036 |
| old log-probability | 174.057091 |
| reference log-probability | 183.630640 |
| GDPO advantage | 0.203765 |
| actor forward/backward/update | 510.619594 |
| weight update | 0.471909 |
| other/unlisted trainer overhead | 0.885135 |
| **total step** | **1119.912262** |

The previous GRPO baseline was 1350.202769 s. GDPO Config A saved
230.290507 s (17.05599%). It was 2.778894 s slower than successful GRPO
Config A (1117.133368 s).

## Memory and numerical metrics

External monitoring recorded **47,566 MiB = 46.45 GiB used**; GPU capacity was
85,651 MiB. Trainer actor maximum allocation/reservation was 28.841 / 39.945
GiB.

| metric | value |
|---|---:|
| raw/total reward mean | 1.187858 |
| raw/total reward std | not emitted |
| accuracy reward mean / std | 0.386588 / 2.236181 |
| format reward mean / std | 0.801270 / 0.399045 |
| advantage mean | -2.09e-08 |
| advantage min / max | -2.524049 / 2.633273 |
| actor loss / policy loss | -0.015746 / -0.015746 |
| grad norm | 0.978452 |
| actor PPO KL | 6.752e-05 |
| rollout-correlation KL | 4.081e-04 |
| reward KL penalty metric | 0.0 |
| NaN/Inf | none |
| optimizer.step | PASS; global step=1 |

## Evidence

- Ray metrics: `/tmp/ray/session_latest/logs/worker-f1202efe4b3ac43258391c73bdac5290b70532397d4015ca18e92420-01000000-94080.out`
- stdout: `/root/autodl-tmp/ProjectB/env-modern/throughput-tuning-logs/gdpo_throughput_config_a.stdout.log`
- GPU monitor: `/root/autodl-tmp/ProjectB/env-modern/throughput-tuning-monitors/gdpo_throughput_config_a.gpu.csv`
- official verl commit: `1876b06d0a3e4e71e06230be10af14492ca8a75b`
