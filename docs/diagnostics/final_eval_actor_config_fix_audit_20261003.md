# FINAL_EVAL actor-config compatibility repair audit

Status: `ACTOR_CONFIG_FIX_PASS`

## Scope

This repair is based on commit
`e8c8ffef0027f75bf7b32017fb0c3124e4059428` and is isolated on branch
`final-eval-actor-config-fix-20261003`. No GPU smoke, model loading,
vLLM process, inference, scoring, training, optimizer step, or FINAL_HOLDOUT
evaluation was run in this repair pass.

The first infrastructure smoke remains preserved at
`/root/autodl-tmp/ProjectB/final-test/_infra-smoke/SMOKE_REPORT.md`.
It stopped before model loading with:

`AssertionError: [actor] Please set at least one of 'actor.ppo_micro_batch_size' or 'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled.`

## Root cause and exact repair

The frozen formal launcher omitted the two actor-schema settings required by
the pinned veRL revision `1876b06d0a3e4e71e06230be10af14492ca8a75b`.
The repair adds exactly:

```text
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_dynamic_bsz=True
```

No other formal launcher line changed. The same two settings are present in
the three successful Stage1 launchers:

- `scripts/train_grpo_stage1.sh` lines 320-321
- `scripts/train_gdpo_stage1.sh` lines 323-324
- `scripts/train_grpo_nokl_stage1.sh` lines 332-333

This is a schema/config compatibility repair before inference, not batch-size
tuning. The formal launcher still uses `trainer.val_only=true`,
`trainer.val_before_train=true`, `trainer.total_epochs=0`, and
`trainer.total_training_steps=0`; no PPO actor update or optimizer step is
enabled by this repair.

## CPU Hydra compose/resolve

The patched launcher was resolved with `CUDA_VISIBLE_DEVICES=""` and
Hydra `--cfg job --resolve`; `_validate()` and model loading were not called.
The clean resolved config is preserved at:

`/root/autodl-tmp/ProjectB/final-test/_infra-smoke/actor_config_fix_resolved_config.yaml`

Resolved values:

```text
actor.ppo_micro_batch_size: null
actor.ppo_micro_batch_size_per_gpu: 1
actor.use_dynamic_bsz: true
actor.use_kl_loss: false
trainer.val_only: true
trainer.val_before_train: true
trainer.total_epochs: 0
trainer.total_training_steps: 0
algorithm.use_kl_in_reward: false
```

The resolved validation path remains
`/root/autodl-tmp/ProjectB/env-modern/final_holdout_v1.parquet`.

## Formal generation identity

The generation and endpoint-related settings are unchanged:

```text
n=1
temperature=0
do_sample=false
top_k=-1
top_p=1.0
max_prompt_length=2048
max_response_length=1024
validation_shuffle=false
gpu_memory_utilization=0.40
max_num_seqs=8
max_num_batched_tokens=6144
max_model_len=3072
```

The scorer path/name, reward manager, model identity, tokenizer, endpoint,
runtime mapping, validation batch, and KL settings were not changed. The
working-tree diff contains only the two actor-schema additions.

Frozen identities rechecked:

```text
endpoint manifest: 160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16
runtime mapping:   ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd
holdout parquet:   e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22
ordered source IDs:75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad
rows: 108 (tool_only=101, response_only=7)
```

## Static checks

- `bash -n scripts/run_final_test.sh`: PASS
- CPU `final_eval_interface_static_gate.py`: PASS
- Hydra compose/resolve: PASS
- `git diff --check`: PASS
- new schema blocker: none
- model output generated during repair: NO
- optimizer/training executed during repair: NO
