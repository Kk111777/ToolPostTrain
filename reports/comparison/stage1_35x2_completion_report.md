# ProjectB Stage 1 GRPO + GDPO completion report

## Final status

- GRPO Stage 1: PASS, global_step=35, marker present.
- GDPO Stage 1: PASS, global_step=35, marker present.
- Both trainers exited normally with code 0; no 70/105-step training or GRPO-noKL run was started.
- GPU was released: 0 MiB in the final check.

## Shared validation and fairness

- Both stages use the same formal validation set: /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet.
- Intermediate validation steps: 0, 7, 14, 21, 28, 35.
- test.parquet was not used for intermediate validation.
- The shared RL initialization is /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged.
- The shared training semantics are raw3920 -> eligible3901, shuffle=false, drop_last=true, batch512, 7 batches/epoch, 5 epochs, 35 steps.

## Retained artifacts

| Stage | Full resume checkpoint | Model-only checkpoint | Report |
|---|---|---|---|
| GRPO | /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/checkpoints/global_step_35 | /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/model_only/step_35 | /root/autodl-tmp/ProjectB/runs/grpo_stage1_35/grpo_stage1_report.md |
| GDPO | /root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/checkpoints/global_step_35 | /root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/model_only/step_35 | /root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/gdpo_stage1_report.md |

- Both full checkpoints and both model-only checkpoints passed structural/readability checks.
- GDPO model-only passed independent CPU from_pretrained loading with 1,543,714,304 parameters and a usable chat template.
- Compact GDPO metrics: /root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/metrics/stage1_metrics.json.
- Checkpoint manifests, diagnostics, validation artifacts, logs, and stage markers are readable.
- Rolling step10/20/30 full payloads were cleaned only after later checkpoints were present; step35 full is retained for future continuation decisions.

## Final disk and shutdown state

- Final free space before shutdown: 27.12 GiB.
- ProjectB runs remain preserved; no model/checkpoint was deleted by this finalization.
- This report records completion of the current 35-step paired stage; it does not authorize future 70/105 or GRPO-noKL work.

Generated at UTC: 2026-10-02T09:53:07.735778+00:00
