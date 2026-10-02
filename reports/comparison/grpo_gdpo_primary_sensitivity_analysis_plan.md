# GRPO/GDPO Stage 1 primary and sensitivity analysis plan

## Frozen endpoints

- Primary endpoint: the same /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet, greedy n=1 at steps 0/7/14/21/28/35.
- Sensitivity endpoint: the same persisted 80-row outputs with source row 3814 excluded, yielding 79 rows. No resampling and no redefinition of validation.
- The excluded row is known to duplicate the prompt and prompt+ground-truth content of training/SFT source row 527; this is a post-hoc sensitivity analysis, not a new split.

## Metrics

For every step and algorithm, report raw total validation reward, accuracy reward, format reward, strict format, tool-call parse, and response wrapper. Do not use training reward means as the primary cross-algorithm endpoint.

Canonical files:

- reports/grpo/grpo_stage1_metrics_canonical.json
- reports/gdpo/gdpo_stage1_metrics_canonical.json

Both contain primary-80 and offline-derived sensitivity-79 values from the same persisted JSONL outputs.

## Inference and uncertainty

The final comparison should use paired per-prompt differences on common rows. Bootstrap resampling, if reported, must resample prompt pairs within each step and endpoint; it must not resample or redefine the validation split. Any causal statement must preserve the known KL-treatment/reward-dimension confound between the existing GRPO-original and GDPO-current runs.
