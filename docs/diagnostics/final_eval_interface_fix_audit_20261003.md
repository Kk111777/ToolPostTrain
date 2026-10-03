# FINAL_EVAL_INTERFACE_FIX audit

Gate: **FINAL_EVAL_INTERFACE_FIX_PASS**

Date: 2026-10-03 (Asia/Shanghai). This was a CPU-only interface repair on a
branch based on frozen commit `fba66ff7e096801beeff6843462df823bf1bd24c`.
No GPU, model inference, FINAL_HOLDOUT_V1 scoring, training, resume, Ray or
vLLM process was started.

## 1. Original blockers and exact fixes

The frozen endpoint manifest hashes canonical source prompt JSON, while veRL
validation JSONL stores the decoded runtime prompt produced by the official
RLHFDataset/SingleTurnAgentLoop path. The analyzer now matches
`(runtime_input_sha256, ground_truth_sha256_exact)` and retains the canonical
prompt hash only as endpoint evidence. The runtime mapping is a supplement,
not a new endpoint, and contains no raw prompt, ground-truth or output text.

The launcher now checks hardcoded byte SHA256 values for both the endpoint
manifest and runtime mapping before parsing either file. It then checks the
manifest's parquet identity, ordered source IDs, row positions, and the
mapping's parent identity and unique pairs.

The analyzer now creates one shared `numpy.default_rng(42)` primary matrix for
all six comparisons and all three overall metrics, and one shared tool-only
matrix. Response-only N=7 effects are descriptive only. Formal mode refuses a
partial, duplicate, extra, or missing model matrix.

## 2. Frozen identities

| item | value |
|---|---|
| endpoint manifest SHA256 | `160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16` |
| endpoint parquet SHA256 | `e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22` |
| ordered source-ID SHA256 | `75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad` |
| endpoint rows | 108 (`tool_only=101`, `response_only=7`) |
| runtime mapping SHA256 | `ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd` |
| tokenizer | RL_INIT_V1 local tokenizer, `Qwen2Tokenizer` |
| chat-template SHA256 | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |
| official veRL revision | `1876b06d0a3e4e71e06230be10af14492ca8a75b` |

The runtime mapping has 108/108 unique pairs. Construction uses official
veRL `RLHFDataset`, the official text ContinuousToken builder, generation
prompt enabled, and the same `tokenizer.decode(..., skip_special_tokens=True)`
operation observed in validation. Construction is CPU-only and model-free.

## 3. Real trainer JSONL regression

An ephemeral mapping was built from `formal_validation_80.parquet` with the
same official runtime semantics. Persisted step-0 JSONL from all three Stage1
runs mapped by hashes with no positional fallback:

| run | matched | missing | duplicates | unique source IDs |
|---|---:|---:|---:|---:|
| GRPO-original35 | 80/80 | 0 | 0 | 80 |
| GDPO-current35 | 80/80 | 0 | 0 | 80 |
| GRPO-noKL35 | 80/80 | 0 | 0 | 80 |

Result: `REAL_TRAINER_JSONL_MAPPING_PASS`.

## 4. Bootstrap and matrix controls

The primary index matrix is shape `(10000,108)` with SHA256
`ccd589b57df0d4e12b06ce60e421f761610a2889994e8c1e67d88e279e9799ad`.
The tool-only index matrix is shape `(10000,101)` with SHA256
`dfc1313612a182b5c9e6bbc7095ea355f9c3b29ab0662283fb238f1e980c603c`.
Both use NumPy `default_rng` seed 42; NumPy version is recorded in each
analysis output. Response-only comparisons are explicitly descriptive-only.

Formal mode requires exactly:

`RL_INIT_V1`, `GRPO-original35`, `GDPO-current35`, `GRPO-noKL35`.

It requires 108 uniquely mapped rows per model and refuses incomplete or
duplicate matrices with `FINAL_ANALYSIS_REFUSED_INCOMPLETE_MATRIX`. Partial
fixtures are available only with explicit `--fixture-mode` and cannot produce
a formal report.

## 5. CPU regression results

All checks in `docs/diagnostics/final_eval_interface_fix_static_checks.json`
passed:

| check | result |
|---|---|
| A endpoint manifest byte identity | PASS |
| B endpoint parquet SHA | PASS |
| C ordered source-ID SHA | PASS |
| D runtime mapping 108/108 unique | PASS |
| E real trainer JSONL mapping | PASS |
| F synthetic analysis self-test | PASS |
| G formal validation mapping and analyzer regression | PASS |
| H common primary bootstrap | PASS |
| I common tool-only bootstrap | PASS |
| J missing model refusal | PASS |
| K duplicate model refusal | PASS |
| L endpoint manifest tamper refusal (analyzer and launcher) | PASS |
| M runtime mapping tamper refusal (analyzer and launcher) | PASS |
| N Python compile and bash syntax | PASS |
| O Hydra compose/resolve | PASS |
| P val-only source-flow audit | PASS |

The static tamper tests stopped before any model or vLLM launch. The current
server check found no `verl.trainer.main_ppo`, vLLM or Ray training process and
no final-test JSONL output. `nvidia-smi` is unavailable in the no-card mode;
no GPU process was started.

## 6. Boundary

This commit fixes only the evaluation interface and analysis safeguards. It
does not run the final holdout, choose a model, change the 108-row endpoint,
change generation settings, or alter the experiment design. The branch is
intended for independent review; it is not merged into `main`.
