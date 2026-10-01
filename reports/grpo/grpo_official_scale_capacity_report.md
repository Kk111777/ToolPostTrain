# GRPO official-scale capacity report

Verdict: PASS
## Common configuration and result

- model: RL_INIT_V1 at /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged
- train batch: 512 eligible prompts
- rollout.n: 4; trajectories: 2048
- max prompt/response: 2048 / 1024
- learning rate: 1e-6
- PPO mini batch: 128 prompts, which is 512 trajectories per actor mini-update
- physical PPO microbatch: 1
- dynamic token budget: 6144
- vLLM: TP=1, gpu_memory_utilization=0.30, max_num_batched_tokens=6144, max_num_seqs=4
- actor FSDP offload: false; reference parameter offload: true
- reward manager, seed, tokenizer, dataset selection, KL path and optimizer were held fixed
- no checkpoint was saved and no second optimizer step was run

## Batch and data verification

- manifest selected_count: 512
- rollout dump count: 2048
- prompt groups: 512, group size range: 4..4
- the same temporary eligible-512 parquet was used; original train.parquet was not overwritten
- full filter audit: /root/autodl-tmp/ProjectB/env-modern/full_train_overlong_filter_audit.md

## Training metrics

- reward mean: 1.207838
- reward std (population, rollout dump): 2.471229
- advantage mean (trainer token-mask metric): -0.048617
- advantage max/min (trainer): 1.500000 / -1.500000
- advantage std (token-mask reconstruction from dump): 0.771184
- actor loss: 0.034090
- policy loss: 0.034090
- grad norm: 0.606180
- learning rate: 0.000001
- actor PPO KL: 0.000066
- rollout/reference KL metric: 0.000330
- reward KL penalty: 0.0 in this step
- NaN/Inf: none observed
- optimizer.step: PASS (global step=1, non-zero grad norm, return code=0)

## Timing and memory

- rollout time: 476.066 s
- old log-prob time: 175.567 s
- reference log-prob time: 183.764 s
- advantage time: 0.501 s
- actor update time: 512.165 s
- official trainer step time: 1350.203 s
- external supervisor elapsed: approximately 1428.916 s
- observed rollout-phase GPU peak: 38.42 GiB
- observed actor-phase GPU peak: 41.80 GiB
- overall external GPU peak: 44.01 GiB
- actor reported max allocated/reserved: 28.83 / 39.96 GiB
- host RAM peak: 227.33 GiB
- OOM: none

## Artifacts

- log: /root/autodl-tmp/ProjectB/env-modern/official-scale-capacity-logs/grpo_eligible512.stdout.log
- rollout dump: /root/autodl-tmp/ProjectB/env-modern/official-scale-rollout-dumps/eligible512_grpo/1.jsonl
- capacity manifest: /root/autodl-tmp/ProjectB/env-modern/capacity_smoke_512_manifest.json
