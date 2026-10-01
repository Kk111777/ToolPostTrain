# Formal sampler determinism audit

## Verdict

**PASS for the frozen configuration.** GRPO and GDPO receive the same
prompt-group sequence when starting from RL_INIT_V1 with the same data tree.

Frozen sampler settings:

- `data.shuffle=False`
- `data.validation_shuffle=False`
- `actor_rollout_ref.actor.shuffle=False`
- `actor_rollout_ref.actor.data_loader_seed=42`
- same filtered dataset and source-order manifest
- training `drop_last=True`

## Source audit

Official verl v0.9.1:

- `verl/trainer/ppo/utils.py:140-166`: false data shuffle creates
  `SequentialSampler`; true creates stateful `RandomSampler` with a
  generator seeded from `data.seed`.
- `verl/trainer/main_ppo_v0.py:215`: creates the training sampler.
- `verl/trainer/ppo/ray_trainer.py:405-412`: creates
  `StatefulDataLoader` with the train batch size, explicit sampler, and
  `drop_last=True`.
- `verl/trainer/ppo/ray_trainer.py:418-424`: validation uses
  `drop_last=False`.
- `verl/workers/config/actor.py:178-179`: actor minibatch shuffle and
  data-loader seed are separate controls; validated actor shuffle is false.

With sequential sampling, the filtered dataset is traversed in source order.
A batch is yielded before rollout, reward, advantage, or CUDA actor work, so
those later operations cannot change the next sampler index. GRPO and GDPO
therefore see identical prompt-group IDs at each step. `rollout.n=4`
expands each group and does not reorder prompt groups.

This audit changed no source or dataset. If a future run enables data shuffle,
the sampler seed and stateful dataloader state must be frozen and compared.
