# Stage 1 closeout: freeze the current project at 35 training steps

Decision recorded: 2026-10-03 (Asia/Shanghai). This is an experiment-budget decision accepted by the project owner, not a claim of convergence. GRPO-original35, GDPO-current35 and GRPO-noKL35 are the frozen RL endpoints. No 70/105 continuation or final-test execution is authorized by this document.

## Evidence and decision

All three independent runs start from RL_INIT_V1 and complete five epochs, seven batches per epoch, 35 optimizer steps, with the common 0/7/14/21/28/35 validation schedule. [Three-run results](../../reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md) show positive observed step0-to-step35 changes but no clear downstream advantage of GDPO over the KL-aligned noKL reference.

| run | step28 total | step35 total | step35 minus step28 | observed mean seconds/step | projected additional 35-step GPU hours |
|---|---:|---:|---:|---:|---:|
| GRPO-original | 2.831808 | 2.799721 | -0.032087 | 1089.529834 | 10.59 |
| GDPO-current | 2.781237 | 2.820522 | +0.039286 | 1078.800352 | 10.49 |
| GRPO-noKL | 2.781237 | 2.818737 | +0.037500 | 905.334667 | 8.80 |

Projection: archived mean step time × 35 / 3600 = approximately **29.88 GPU hours** across three continuations, before initialization, evaluation, export and checkpoint overhead. This is resource budgeting, not a fair algorithm-speed comparison: noKL removes reference-policy computation. Future throughput and monetary cost are not measured or quoted here.

A rebound over one final interval does not establish a persistent upward trend. The archived cross-algorithm paired bootstrap does not estimate each run's step28-to-step35 improvement. This revision does not independently recompute those within-run intervals from unavailable formal validation JSONL.

## Research value and internship-project ROI

A matched continuation would answer a longer-budget question; it would still use one training seed and a small validation set. It would not establish training-seed robustness, isolate per-dimension normalization from GDPO's batch whitening, or validate the proposed causal mechanism. Repeated exposure to the same training prefix adds budget, not new evaluation evidence.

The current project already demonstrates reward-contract diagnosis, Format SFT/LoRA, modern veRL/vLLM training, objective-path audit, a KL ablation, mechanism inspection and bounded statistical interpretation. A frozen final test and consistent public evidence address the remaining project gap with greater expected value than approximately 30 additional GPU hours. No additional training experiment is required for closeout.

## Frozen next boundary

- Final inference model set: RL_INIT_V1, GRPO-original step35, GDPO-current step35, GRPO-noKL step35.
- Do not substitute validation-selected step28 or another intermediate checkpoint.
- Prepare and follow the [final-test protocol](final_test_protocol.md); currently it is documentation only and has not run.
- Retain existing step35 full/model-only artifacts as recorded by the [completion report](../../reports/comparison/final_stage1_ablation_completion_report.md). This budget decision does not authorize deletion or claim independently verified full training restore.
- Do not use final-test outcomes to extend to 70, tune rewards/hyperparameters or choose a different checkpoint.
- A later research project requires a new explicit scope; the present test is observed data after evaluation.

The original 35→70→105 plans and configuration snapshots remain historical evidence. Their continuation/retention rules are superseded for the current project by this decision; frozen effective training configurations are not rewritten.
