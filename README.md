# ToolPostTrain

**English** | [简体中文](README.zh-CN.md)

GRPO/GDPO post-training for tool calling on **Qwen2.5-1.5B-Instruct**, with an objective-path audit, a reward-side KL control and paired evaluation.

## Overview

This repository builds on [NVlabs/GDPO](https://github.com/NVlabs/GDPO), ToolRL and veRL. The experiment covers an 800-example Format-SFT warmup, three 35-step RL runs and evaluation of four fixed models on a 108-prompt internal holdout.

The initial GRPO/GDPO comparison had a KL confound: both configurations enabled the same flag, but their advantage paths consumed different rewards. A **GRPO-noKL** run was added to align reward-side KL treatment. All three trained checkpoints improved over the shared initialization on this endpoint; GDPO showed no clear additional gain over noKL.

**Stack:** PyTorch · veRL · vLLM · LoRA · FSDP · one RTX 6000D.

## Main Results

Four frozen models, the same **108 prompts**, one greedy answer per prompt: **432 outputs**.

| Model | Role in the comparison | Raw total reward | Δ vs RL_INIT |
|---|---|---:|---:|
| RL_INIT_V1 | Shared Format-SFT initialization | 2.417 | — |
| GRPO-original35 | Original GRPO reference; reward-side KL active during training | 2.946 | +0.529 |
| GDPO-current35 | Per-reward advantage construction | 2.956 | +0.540 |
| GRPO-noKL35 | Control for the reward-side KL difference | 2.945 | +0.529 |

All three RL-minus-initialization paired 95% intervals are above zero. **GDPO − GRPO-noKL is +0.011**, with a 95% interval of **[−0.097, +0.165]**. The small mean difference does not establish GDPO superiority or algorithm equivalence.

Values are rounded to three decimals. Raw total reward is accuracy reward + format reward, not an accuracy percentage. [Exact results and all six paired comparisons →](reports/final_holdout_v1/completion_report.md)

## Experimental Design

### Shared initialization

The base Qwen checkpoint frequently violated ToolRL's required output structure. This would mix format compliance with task quality in the RL comparison. An 800-example Format-SFT warmup, after checking the prompt/scorer contract, established a shared **RL_INIT_V1**. The LoRA checkpoint was merged before the three RL runs. [Before/after diagnostic](reports/sft/sft_v2_after_reward_report.md) · [SFT training report](reports/sft/sft_v2_train_report.md)

GRPO-original, GDPO and GRPO-noKL used the same initialization, data order, training budget and evaluation schedule. The noKL run removes reward-side KL from the GRPO update; its complete configuration diff is [recorded here](docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md).

### Changes from the upstream setup

| Area | Upstream foundation | Setup in this repository |
|---|---|---|
| Initialization | Qwen model; ToolRL data and reward function | Format SFT and a frozen shared RL_INIT_V1 |
| Objective audit | GRPO/GDPO implementations and training configurations | Reward → advantage → actor-loss tracing of the executed KL paths |
| Control | GRPO/GDPO training recipes | Additional GRPO-noKL run with aligned reward-side KL treatment |
| Diagnostics | Reward and advantage tensors | Shared-rollout advantage comparison and sampled reward-conflict analysis |
| Evaluation | Trainer validation support | Four frozen models, 108 internal-holdout prompts and paired bootstrap |

## Objective-Path Audit

Both initial runs set `use_kl_in_reward=true`. The flag alone did not describe the objective reaching the actor:

```text
GRPO: KL-adjusted token-level rewards → group normalization → policy update
GDPO: raw accuracy/format components → per-component normalization → whitening → policy update
```

In the configured GDPO branch, advantages were constructed from raw reward components. The original comparison therefore changed both **advantage construction and KL treatment**. GRPO-noKL provides a KL-aligned reference for comparing the complete advantage constructions.

The distinction is specific to the audited implementation and configuration. GDPO still performs reference-policy computation; noKL removes it as a consequence of the framework flags. Their runtime difference is therefore not a clean measure of estimator efficiency. [Source paths and loss consumers →](docs/diagnostics/gdpo_hidden_kl_use_final_audit.md)

## Optimization Diagnostics

On **8 prompts × 4 shared rollouts**, GDPO changed advantage scale, centering and sample weighting, without substantive within-group ranking reversals at the documented tolerance. A separate sampled diagnostic found opposing centered accuracy/format signals in **8/128 trajectories**, across **6/32 groups**. [Shared-rollout analysis](reports/comparison/grpo_gdpo_shared_rollout_diagnostic.md) · [Reward-conflict analysis](reports/sft/fresh_holdout_reward_variation.md)

The advantage differences did not translate into a clear final-score gain. High greedy-validation format scores and limited conflict in the diagnostic sample offer a **plausible mechanism hypothesis**; they do not prove training-time format saturation or a causal explanation.

## Reproduction Notes

Two evaluation issues are worth recording:

- **Validation-only still required actor configuration.** The first infrastructure smoke stopped before model loading because the launcher omitted the actor settings used by the successful training runs: `ppo_micro_batch_size_per_gpu=1` and `use_dynamic_bsz=True`. Restoring those fields satisfied the pinned veRL actor-schema check; the frozen generation settings and zero training budget remained unchanged. [Compatibility repair](docs/diagnostics/final_eval_actor_config_fix_audit_20261003.md)
- **A stale smoke gate rejected a complete output file.** RL_INIT generated all 108 formal answers, but a validator still expected 80 rows and an old mapping hash. The existing JSONL was preserved, the technical checks were corrected, and validation passed without regenerating answers. [Failure provenance](reports/final_holdout_v1/completion_report.md#preserved-failure-provenance-and-engineering-repairs)

The modern single-GPU environment differs from the archived upstream setup. Actual versions and capacity checks are recorded in the [runtime report](reports/final_holdout_v1/runtime_environment.json) and [GRPO](reports/grpo/grpo_official_scale_capacity_report.md) / [GDPO](reports/gdpo/gdpo_official_scale_capacity_report.md) capacity reports.

## Configuration and Local Checks

| Item | Setting |
|---|---|
| Model / initialization | Qwen2.5-1.5B-Instruct / merged Format SFT, 800 examples |
| RL variants | GRPO-original / GDPO / GRPO-noKL |
| Budget / rollouts | 35 training steps per run; 512 prompts/batch; 4 rollouts/prompt |
| Learning rate / training seed | 1e-6 / 42 |
| Runtime | veRL + vLLM; one RTX 6000D |
| Final evaluation | 108 prompts × 4 fixed models; greedy n=1 |

```bash
bash setup-local.sh
bash 运行本地验证.command
```

These run CPU checks. Historical GPU launchers depend on the recorded server layout and separately retained data/models.

`configs/` and `manifests/` hold settings and identities; `scripts/` holds run and analysis entrypoints; `reports/` holds results; `docs/` holds protocols and implementation audits. Raw outputs, datasets and weights remain outside Git.

## Limitations

- **One training seed:** paired intervals measure prompt-sample variation for fixed checkpoints, not variation across training seeds.
- **35-step budget:** the chosen training budget is not a convergence claim.
- **Post-hoc internal holdout:** selected from the exposure-audited, optimizer-unused train-split tail; it shares its source pool with validation.
- **Task scope:** tool-call output scoring, without live tool execution or an external benchmark.

## Reports and Attribution

- [Three-run training comparison](reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md): validation curves and the 35-step comparison.
- [Final evaluation report](reports/final_holdout_v1/completion_report.md): exact results, row-mapping checks, scorer recomputation and preserved execution failures.
- [Results and technical evidence](reports/README.md): configurations, protocols, identities, diagnostics and historical records.

Qwen, ToolRL, the GRPO/GDPO implementations and the veRL trainer come from upstream. This repository adds the initialization, objective audit, control run, diagnostics and evaluation described above. See the [upstream README](README.upstream.md), [LICENSE](LICENSE) and [third-party license](third_party_dependency.LICENSE).
