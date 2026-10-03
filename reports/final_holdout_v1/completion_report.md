# FINAL_HOLDOUT_V1 completion report

Date: 2026-10-03 (Asia/Shanghai). Gate: **FINAL_HOLDOUT_V1_COMPLETE_PASS**.

The frozen four-model matrix completed with 432/432 unique paired outputs. Existing RL_INIT_V1_retry_02 answers were preserved and revalidated; RL_INIT was not regenerated. No model-quality decision was made until every model passed its technical gate.

## Execution and identity

- Execution launcher commit: `066e158639bcebdc79131e7ab0180d11ef021aaf`; upstream: `1876b06d0a3e4e71e06230be10af14492ca8a75b`.
- Exact frozen launcher and analysis script hashes are in [completion gate](completion_gate.json). Only technical validation and metadata handling were repaired.
- 108 rows: tool-only 101, response-only 7. One greedy answer per row; n=1, temperature=0, do_sample=false, top_k=-1, top_p=1, prompt/response caps 2048/1024.
- Actual configurations agree across models after accounting for frozen model path, corresponding Prometheus label, and isolated output locations.
- Runtime mapping: 108/108 per model, missing=0, duplicates=0. Frozen scorer independently recomputed for all 432 answers; finite values and persisted-field agreement.
- Validation-only return before training updates; epochs/steps=0, resume disabled, both KL flags off. No checkpoint/model files were written in output directories, and retained model/tokenizer hashes match their frozen manifests.

| model | outputs | raw total reward | accuracy reward | format reward |
|---|---:|---:|---:|---:|
| RL_INIT_V1 | 108 | 2.416575477 | 1.490649551 | 0.925925926 |
| GRPO-original35 | 108 | 2.945897568 | 2.001453124 | 0.944444444 |
| GDPO-current35 | 108 | 2.956086145 | 2.002382441 | 0.953703704 |
| GRPO-noKL35 | 108 | 2.945283676 | 2.000839231 | 0.944444444 |

## All predeclared paired comparisons

Right minus left. Primary: raw total reward over all N=108 rows. Percentile paired bootstrap: 10,000 common resamples, seed 42. No comparison is omitted.

| left → right | difference | 95% CI | improved / declined / tied |
|---|---:|---|---|
| GRPO-original35 → GRPO-noKL35 | -0.000613892 | [-0.082146079, +0.073898810] | 6 / 5 / 97 |
| GRPO-original35 → GDPO-current35 | +0.010188577 | [-0.121378968, +0.178048942] | 6 / 7 / 95 |
| GRPO-noKL35 → GDPO-current35 | +0.010802469 | [-0.096604938, +0.165432099] | 3 / 4 / 101 |
| RL_INIT_V1 → GRPO-original35 | +0.529322091 | [+0.303720686, +0.788203419] | 25 / 3 / 80 |
| RL_INIT_V1 → GDPO-current35 | +0.539510668 | [+0.302044554, +0.815225297] | 26 / 5 / 77 |
| RL_INIT_V1 → GRPO-noKL35 | +0.528708199 | [+0.303088082, +0.782163864] | 27 / 5 / 76 |

All accuracy/format component intervals, tool-only auxiliaries and response-only descriptions are retained in [machine-readable results](metrics_and_paired_comparisons.json). Response-only N=7 has no inferential CI.

## Bounded result

All three RL-minus-initialization primary intervals are above zero, supporting incremental improvement of these fixed trained models on this internal endpoint. All three RL-versus-RL primary intervals include zero; endpoint means remain close, without clear evidence of GDPO superiority or algorithm equivalence. These observations do not trigger new training or endpoint selection.

## Interpretation boundary

This is a post-hoc, fresh previously-unexposed internal corroborative holdout under the persisted-artifact audit, drawn from the optimizer-unused original train-split tail. It shares the tail source with formal validation. It is not the official ToolRL test, an external distribution benchmark, a multi-seed result, or an executed-Agent benchmark.

The bootstrap estimates sample variation for these fixed models, not training-seed variability. A CI crossing zero is not equivalence; a CI excluding zero is not general algorithm superiority. Original/GDPO retains the historical KL-treatment confound; noKL/GDPO compares complete KL-aligned advantage constructions, not per-dimension normalization alone.

## Preserved failure provenance and engineering repairs

Attempt 01 failed before model load with zero answers because an outer wrapper wrote metadata into the launcher output directory. Its directory remains intact. Retry 02 successfully generated the RL_INIT answers, but its old validator falsely failed on an 80-row smoke constant and a stale mapping hash. The separate resolved-config file was also missing; the actual configuration was recovered from its persisted runtime log.

The old technical failure report is preserved. Only validation counts/hash constants and metadata discovery were corrected. Existing RL_INIT JSONL remained SHA256 `0063a0c924fbaf3226de7e3ce5eb9e93c716b1237ba33b609301c2d7a753bb4a`. No generated answer was discarded, regenerated, or selected from multiple attempts.

## Archive and disposition

Raw JSONL, logs, actual configs, gate records and complete frozen-analysis output are retained in controlled storage; Git contains only reviewed compact summaries, hash evidence, reports and technical audit code. Shutdown follows successful publication and local archive verification. No 70/105-step run, second seed, step28, new reward, new endpoint or external benchmark was launched.
