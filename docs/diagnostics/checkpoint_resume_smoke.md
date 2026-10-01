# Checkpoint and resume smoke

## Result

**PASS.** A separate one-step GDPO run from RL_INIT_V1 saved a full
checkpoint. A second process loaded it in resume-only mode. The temporary
checkpoint was deleted after verification and was not used as a later
initialization.

## Saved checkpoint

Temporary path:
`/root/autodl-tmp/ProjectB/checkpoints/resume_smoke_tmp/global_step_1`

Measured full directory:

- 19,469,701,513 bytes
- `du -sh`: 19G
- approximately 18.13 GiB

Contents included model, optimizer, extra state (RNG and scheduler),
dataloader `data.pt`, latest-iteration metadata, HuggingFace config,
generation config, tokenizer, and chat template.

The save run emitted `training/global_step:1` and
`local_global_step_folder: .../global_step_1`.

## Resume-only validation

The second process used:

- `trainer.resume_mode=resume_path`
- `trainer.resume_from_path=.../global_step_1`
- `trainer.total_training_steps=1`
- `trainer.save_freq=-1`

The log confirms checkpoint discovery, `Setting global step to 1`, model
load, optimizer load, RNG load, scheduler load, and dataloader boundary
handling. Progress reached 100% immediately.

The resume log contains no new `training/global_step:`, `timing_s/step`,
actor loss, or optimizer-step record. The checkpoint timestamp was unchanged.
Thus no second update or overwrite occurred.

Resume log:
`/root/autodl-tmp/ProjectB/env-modern/checkpoint-resume-logs/gdpo_resume_only.stdout.log`

Cleanup result: temporary directory removed; GPU 0 MiB; free disk about 70G.
