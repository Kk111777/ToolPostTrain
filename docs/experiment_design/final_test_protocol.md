# Frozen final-endpoint protocol: FINAL_HOLDOUT_V1, RL_INIT_V1 and three step35 models

Protocol recorded: 2026-10-03 (Asia/Shanghai). **NOT EXECUTED.** This document authorizes no server startup, GPU task, checkpoint restore, training or final inference. The current project training budget is [frozen at 35](stage1_closeout_decision.md).

## Current endpoint revision: FINAL_HOLDOUT_V1

The primary final endpoint is now `FINAL_HOLDOUT_V1`, stored on the server at
`/root/autodl-tmp/ProjectB/env-modern/final_holdout_v1.parquet` and frozen by
`manifests/final_holdout_v1_manifest.json`. It contains exactly 108 rows
(`tool_only=101`, `response_only=7`) selected in original `train.parquet`
source order from the optimizer-unseen `drop_last` tail after the previously
audited exposure, duplicate, and prompt-length exclusions. The manifest
records the ordered source IDs, four content hashes per row, tokenizer/chat
template identity, source and derived parquet SHA256 values, and the
construction commit; it intentionally contains no prompt or ground-truth
body text and no model outcomes.

`FINAL_HOLDOUT_V1` is a fresh previously-unexposed internal corroborative
endpoint under the same task distribution and scorer. It is not the official
ToolRL `test.parquet`, not an independent external benchmark, and not proof of
general algorithm superiority. The historical `test.parquet` remains on the
server as observed legacy data, but it is not the new primary endpoint and is
not used by the current launcher.

The primary metric is the raw total reward mean over all 108 rows. Secondary
metrics are accuracy reward, format reward, and strict format over all 108
rows. Tool-only metrics (N=101) and response-only metrics (N=7) are fixed
stratified auxiliaries; the response-only view is descriptive only and is not
used as a new primary endpoint. No type rebalancing or outcome-based row
deletion is permitted.

The final launcher must assert the frozen manifest SHA256, derived parquet
SHA256, row count, and ordered source-ID hash before starting any model or
vLLM process. Any path containing `test.parquet` is rejected by default. A
separate `formal_validation_80.parquet` infrastructure smoke must pass before
any future model evaluation; the four models are then evaluated once each on
`FINAL_HOLDOUT_V1` under the unchanged greedy generation/scoring contract.

## Research endpoint and fixed model set

Evaluate the following four models once on the same frozen `FINAL_HOLDOUT_V1` rows, with identical generation/scoring semantics:

| identifier | retained model-only source; to verify before execution |
|---|---|
| RL_INIT_V1 | `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged` |
| GRPO-original35 | `/root/autodl-tmp/ProjectB/runs/grpo_stage1_35/model_only/step_35` |
| GDPO-current35 | `/root/autodl-tmp/ProjectB/runs/gdpo_stage1_35/model_only/step_35` |
| GRPO-noKL35 | `/root/autodl-tmp/ProjectB/runs/grpo_nokl_stage1_35/model_only/step_35` |

No step28 or validation-selected alternative is eligible. Test all four under the protocol fixed before reading any new test-model outcome. The three RL-minus-initialization comparisons measure incremental RL behavior after Format SFT; they do not isolate the benefit of SFT itself.

## Historical test provenance (retained, not the current endpoint)

The old source `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/test.parquet` is retained as historically observed ToolRL data. CPU provenance audit confirmed historical loading/scoring before the current endpoint was frozen, so it cannot support a new “unseen test” claim. It is not used by `FINAL_HOLDOUT_V1` and should not be regenerated or substituted.

## Pre-execution provenance gate

Expected current source: `/root/autodl-tmp/ProjectB/env-modern/final_holdout_v1.parquet`, with the exact identity and hashes in `manifests/final_holdout_v1_manifest.json`. Verify the manifest, derived parquet, row count 108, schema, original source order, and stable ordered source IDs before any future inference. No parquet is committed to the public repository.

1. Register model/checkpoint identifiers and SHA256 of each weight shard, config, tokenizer, generation config and chat template. Record RL_INIT_V1 identity and the retained model-only integrity checks; full optimizer restore is unnecessary for inference.
2. Register the frozen FINAL_HOLDOUT_V1 parquet SHA256, stable original row IDs, ordered manifest, input/ground-truth hashes, prompt lengths and target categories. Check exact and normalized-content overlap with actual RL train, SFT sources, formal validation, historical test data, and known diagnostic evidence, using prompt and prompt+ground-truth hashes. Version the normalization rule: deterministic serialization of message role/content and Unicode NFC/newline normalization without erasing argument-value distinctions. Exact/normalized checks do not guarantee absence of semantic overlap.
3. Keep the historical test-use audit as disclosure only. [The legacy smoke script](../../scripts/run_single_gpu_smoke.sh) configured `data.val_files="$DATA_DIR/test.parquet"`; script existence did not prove execution, but the current CPU provenance audit confirmed historical loading/scoring. Therefore the old test cannot be called unseen and is not the current endpoint. Do not use any final-holdout outcome to delete rows or alter the frozen manifest.
4. Freeze and commit the model list, generation configuration, scorer, metric definitions and analysis-program revision before generating test outputs. Use synthetic/validation fixtures to verify the evaluation and analysis programs, not test outcomes to tune them.
5. If count/order/hash, prompt limits or provenance conflict with the frozen specification, stop at preflight and document the discrepancy. Do not silently drop rows or truncate prompts to fit. The 108-row endpoint has no source3814 sensitivity deletion; any separate sensitivity subset must be declared as a new analysis view without changing the frozen manifest.

## Generation contract

Match the saved formal-validation semantics in [noKL effective config](../../reports/grpo_nokl/stage1_35/effective_config_final.yaml):

| field | frozen value |
|---|---|
| n | 1 |
| temperature | 0 |
| do_sample | false |
| top_p / top_k | 1.0 / -1 |
| seed | 42 |
| max prompt / response tokens | 2048 / 1024 |

Record the **actual resolved generation configuration**, including engine translation of do_sample/temperature, max tokens, special tokens/EOS, stop strings/token IDs, penalties, prompt truncation policy, detokenization and template arguments. Match the archived formal-validation behavior rather than assuming temperature=0 alone specifies it. Any material mismatch requires a documented protocol revision before test output, not after comparing scores.

Use the same engine/version, precision, tokenizer/template behavior and batching settings for all four models; serialize model evaluation and preserve original row order. Record Python, torch/CUDA, transformers, vLLM, veRL commit, GPU and scorer environment. Greedy decoding targets deterministic semantics but does not promise bitwise reproducibility across hardware/runtime settings. No sampled retry, best-of-n or constrained decoding is introduced.

## Scoring and denominators

Freeze the actual formal modern-runtime `rlla.compute_score`/reward-manager implementation and hash, response preprocessing, compatible Qwen parser identity and all reward-affecting environment flags. Do not silently substitute the vendored legacy scorer or use run-specific parser behavior. Score raw outputs using the same fixed task rewards for all models; reward-side KL and actor/reference telemetry are not evaluation scores.

| metric | definition and denominator |
|---|---|
| **Primary: raw total reward mean** | arithmetic mean of raw per-row score over all N fixed rows, including invalid/truncated outputs |
| accuracy / format reward | separate per-row means over all N; correctness reward is a score, not accuracy percent |
| strict format | official target-aware format predicate; valid count/N |
| tool-call JSON parse | all required JSON-line calls parse with the frozen parser; denominator is all target rows requiring tool calls, including mixed targets; a wrong output kind is failure |
| response-only wrapper | frozen response-wrapper predicate on response-only targets; denominator is all response-only target rows; explicitly label tag presence versus strict format |
| auxiliary counts | optional output-type/tool-parse/wrapper counts over all rows, labelled **counts**, not conditioned success rates |
| diagnostics | generated length, finish reason, truncation/invalid-output rate; tool-call correctness-reward mean and full-reward count on tool targets |

Derive target categories from ground truth before inference. A zero-sized category is N/A, not 0% or 100%. Malformed model outputs retain their official penalty; infrastructure/scorer exceptions are separately logged, not silently converted to model failures or dropped. Tool JSON parsing does not imply schema validity, correct arguments, successful tool execution or end-to-end Agent success. Response-only correctness may be constant under the official reward; do not reinterpret it as semantic answer quality.

## Frozen paired analysis

Pair outputs by stable row ID **and** input hash; verify ground-truth hash, complete coverage and unique one-output-per-model-per-row mapping before statistics. Never rely on file position alone or bootstrap unpaired model samples.

Predeclared RL comparisons (right minus left):

- `GRPO-original35_vs_GRPO-noKL35`: noKL−original, KL ablation.
- `GRPO-original35_vs_GDPO-current35`: GDPO−original, retains the historical objective-treatment/advantage confound.
- `GRPO-noKL35_vs_GDPO-current35`: GDPO−noKL, compares the complete KL-aligned advantage constructions, not per-dimension normalization in isolation.

Also report each of the three RL endpoints minus RL_INIT_V1. For all six comparisons, report paired mean differences in total/accuracy/format, improved/declined/tied fractions and effect sizes in the original reward units. Set the reporting tie tolerance to 1e-8 and retain unrounded values.

Use 10,000 paired row bootstrap resamples, seed42 with a recorded NumPy RNG implementation/version, 2.5%/97.5% percentile interval. Draw common row-index resamples for all models/metrics in the overall analysis; conditioned metrics use their fixed eligible-row subset and separately recorded draws. Preserve pairing. Disclose whether prompts are independent sampling units; related/duplicate prompts limit the interpretation, and no training-seed uncertainty is estimated by this bootstrap.

Report all comparisons, not only favorable ones. No population-level significance, algorithm equivalence, or training-seed robustness claim is inferred from these intervals. This final test evaluates a small dataset with one trained seed, not BFCL or an executed-Agent benchmark.

## Artifact and failure policy

Save a dedicated final-test directory containing:

- the precommitted protocol/model/data manifest and actual resolved generation/scoring config;
- one JSONL per model with row ID, hashes, output text, token length, finish reason, raw reward components and parser outcomes;
- environment/code revisions, execution timestamps, logs, return status and SHA256 artifact inventory;
- complete paired analysis inputs, analysis script revision, summary JSON/Markdown and completion gate.

A technical failure may resume missing rows under exactly the same frozen configuration, preserving completed outputs and all attempts; do not regenerate and select the better answer. Any material runtime change invalidates comparability until reviewed. Scorer-only recomputation is traceable and uses persisted outputs; no outcome-driven metric change is allowed.

Public GitHub receives reviewed small reports, metric/manifest files and analysis code. Full raw outputs remain in controlled storage unless dataset redistribution and content review permit publication. Weights, optimizer states, parquet, large logs and secrets stay out of Git. Completion requires all four models' full coverage, consistent hashes/configs, readable outputs and analyses, and normal process exit; this protocol defines no automatic shutdown or training action.

## Test-use firewall

After test outcomes are observed, do not use them to choose step28/35, continue to 70, change reward, tune hyperparameters, retrain or present the same test as unseen again. If new research starts, treat this test as observed data and specify a new evaluation boundary. Negative or close results are published with the same reporting rules.

**Current status:** all server-dependent identity/provenance checks, analysis-program freeze and GPU evaluation are pending. This revision supplies the protocol only.
