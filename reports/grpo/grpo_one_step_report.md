# ProjectB GRPO controlled 1-step validation

This is a one-step functional validation only. No formal multi-step training was run and no checkpoint was saved as a subsequent experiment start.

## Configuration

- Frozen initialization (RL_INIT_V1): `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged`
- Dataset: `repo/verl-GDPO/dataset/rlla_4k/train.parquet`
- Batch: `8` prompts × rollout.n `4` = `32` trajectories
- Actor: FSDP, BF16, gradient checkpointing, micro-batch 1, PPO mini-batch 8
- Rollout: official v0.9.1 async vLLM, TP=1, `gpu_memory_utilization=0.30`, max prompt/response 2048/1024
- Reward manager: official `GDPORewardManager` + existing `rlla.compute_score`
- Learning rate: `1e-6`; `use_kl_in_reward=True`, `kl_penalty=kl`, `kl_coef=0.001`
- `trainer.total_training_steps=1`, `trainer.save_freq=-1`, `trainer.resume_mode=disable`

## Chain result

- Overall driver status: **PASS**
- dataset → rollout → reward → advantage → actor loss → backward → optimizer.step: **PASS**
- optimizer-step marker: `True`
- finite non-zero grad norm: `True`
- parameter update: **True** (official FSDP optimizer path reached `self.optimizer.step()`; direct post-step checksum was not collected because checkpoint saving was disabled)
- NaN/Inf observed: `False`

## Runtime metrics

| metric | value |
|---|---:|
| actor/pg_loss | -0.0213974155 |
| actor/loss | -0.021397423 |
| reward mean | 1.38125002 |
| reward std | not emitted by the legacy trainer metric line |
| reward min / max | -3 / 4 |
| advantage mean | 0.02139741 |
| advantage std | not emitted by the legacy trainer metric line |
| advantage min / max | -1.49999964 / 1.48365033 |
| reward KL penalty | 0 |
| reward KL coefficient | 0.001 |
| rollout-correction KL (different metric) | 0.000652491639 |
| grad norm | 1.33733809 |
| learning rate | 1e-06 |
| step time (s) | 42.7325141 |
| GPU peak by nvidia-smi | 41618 MiB |
| actor peak allocated / reserved | 28.7717447 / 38.796875 GiB |
| total tokens | 29426 |
| mean prompt / response tokens | 837.625 / 81.9375 |

## Evidence boundary

The legacy trainer emits reward/advantage mean/min/max but not their standard deviations or a post-step parameter checksum. The report therefore leaves those two standard deviations explicitly un-substituted, while the finite non-zero gradient, optimizer marker, successful return code, and official FSDP `optimizer_step()` path establish the one-step update path.
