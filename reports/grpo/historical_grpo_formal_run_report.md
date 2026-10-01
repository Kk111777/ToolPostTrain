# Formal GRPO run report

- status: **FAIL**
- completed trainer metric steps: 0/105
- supervisor PID: 113223
- trainer PID: 113228
- exit code: 134
- wall time seconds: 675
- automatic restart: no

## Frozen initialization and configuration

- RL_INIT_V1: /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged
- effective configuration: /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/effective_config.yaml
- official verl commit: 1876b06d0a3e4e71e06230be10af14492ca8a75b
- ProjectB commit: c105a5e8e009339849d9a2247209ea1494a4043d

## Training curves from native trainer metrics

| metric | summary |
|---|---|
| critic/rewards/mean | not emitted |
| critic/advantages/mean | not emitted |
| actor/loss | not emitted |
| actor/grad_norm | not emitted |
| actor/ppo_kl | not emitted |
| actor/lr | not emitted |

## Timing

| stage | summary seconds |
|---|---|
| timing_s/gen | not emitted |
| timing_s/reward | not emitted |
| timing_s/old_log_prob | not emitted |
| timing_s/ref | not emitted |
| timing_s/adv | not emitted |
| timing_s/update_actor | not emitted |
| timing_s/update_weights | not emitted |
| timing_s/step | not emitted |

## Reward dimensions at required diagnostic milestones

The legacy GRPO logger emits total reward natively. Format/accuracy are preserved in the existing rollout reward-extra-info path and summarized from fixed 64-trajectory diagnostic artifacts.

| step | samples | total reward mean | format reward mean | accuracy reward mean |
|---:|---:|---:|---:|---:|

## Memory and integrity

- GPU monitor peak: 47800 MiB (46.68 GiB)
- NaN/Inf text scan: none observed
- optimizer.step: not confirmed
- model-only exports: /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/model_only/step_100 and step_105
- full resume checkpoint root: /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/checkpoints

## Logs and diagnostics

- native trainer log: /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/logs/grpo_train.log
- GPU monitor: /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/logs/gpu_monitor.csv
- fixed milestone diagnostics: /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/diagnostics/
- no GDPO run was started by this supervisor.
