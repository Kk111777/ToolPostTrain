# GDPO Stage 1 versus successful Config A

Status: **STATIC REVIEW BLOCKED; no GDPO training was started.**

## Evidence used

- Successful run summary: env-modern/gdpo_throughput_config_a_report.md
- Successful resolved/config dump: env-modern/throughput-tuning-logs/gdpo_throughput_config_a.stdout.log
- Stage 1 resolved config: runs/gdpo_stage1_35/effective_config_prelaunch.yaml
- Stage 1 launcher: scripts/train_gdpo_stage1.sh

The successful parser-compatible experiment name is
qwen2p5_1p5b_gdpo_one_step.

## Intentional GDPO/runtime changes

- algorithm.adv_estimator: gdpo and
  algorithm.gdpo_reward_keys=["accuracy_reward","format_reward"]
- parser-compatible trainer.experiment_name: GDPO one-step name above
- one optimizer step / one epoch -> Stage 1 35 steps / 5 epochs
- validation source: test.parquet in the one-step smoke ->
  formal_validation_80.parquet for Stage 1
- validation schedule: disabled in one-step smoke -> 0/7/14/21/28/35
- full checkpoint schedule: disabled in one-step smoke -> 10/20/30/35
- independent GDPO run/checkpoint/log/model-only paths

All listed execution settings remain Config A values: RL_INIT_V1,
batch 512, rollout.n 4, PPO mini-batch 128, physical microbatch 1,
dynamic token budget 6144, lengths 2048/1024, lr 1e-6, vLLM
TP=1/utilization=0.40/max_num_seqs=8/max_num_batched_tokens=6144,
actor offload disabled, reference parameter offload enabled.

## Additional data-scope difference requiring confirmation

The successful capacity Config A resolved data.train_max_samples=512.
The formal Stage 1 launcher resolves data.train_max_samples=-1 so the
full eligible training sequence (3901 rows, 7 complete 512-prompt batches
per epoch) is used. This is necessary for the formal 35-step schedule, but
it is not one of the explicitly listed Stage 1-only changes. The validation
batch also resolves to 80 because the frozen formal validation set has 80
