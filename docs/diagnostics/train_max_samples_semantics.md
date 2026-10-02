# train_max_samples semantic audit

Generated: 2026-10-01T23:13:27.169607+00:00

Evidence inspected in the official v0.9.1 source:

- upstream/verl-v0.9.1/verl/utils/dataset/rl_dataset.py: RLHFDataset receives max_samples=-1 by default and only selects a prefix/subset when max_samples > 0 and max_samples < total. With shuffle=false, a positive value selects the first rows; -1 leaves the complete dataset.
- upstream/verl-v0.9.1/verl/trainer/ppo/ray_trainer.py: the train dataset is constructed with config.data.get("train_max_samples", -1), and the train dataloader uses drop_last=true. Steps per epoch are the integer dataset length divided by train batch size.
- Current GDPO resolved config: runs/gdpo_stage1_35/effective_config_prelaunch.yaml contains train_max_samples: -1, train_batch_size: 512, filter_overlong_prompts: true, shuffle: false, 5 epochs and 35 total training steps.

Independent dry-run result:

- raw=3920
- eligible=3901
- batches_per_epoch=7
- total_steps=35

Therefore formal GDPO Stage 1 should retain train_max_samples=-1; changing it to 512 would train only one full batch per epoch and would not match the formal GRPO data semantics.
