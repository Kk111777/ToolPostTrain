# FINAL_HOLDOUT_V1 freeze audit

Date: 2026-10-03 (Asia/Shanghai)

Gate: **FINAL_HOLDOUT_V1_FROZEN_PASS**

This is a CPU/no-GPU endpoint-freeze audit. No model was loaded, no vLLM
generation was run, no reward was scored, and no model output was produced.

## Frozen identity

- branch: `freeze-final-holdout-v1-20261003`
- candidate-audit input commit: `a489a59e0117eb5d469297e15dda6ca4417ad80c`
- source: `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet`
- derived endpoint: `/root/autodl-tmp/ProjectB/env-modern/final_holdout_v1.parquet`
- row count: **108**
- target distribution: `tool_only=101`, `response_only=7`
- ordered source-ID SHA256: `75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad`
- derived parquet SHA256: `e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22`
- source train parquet SHA256: `f2e728ad4a379550887711870db5f97154a0ecc45efbe5d43db8a99dc3b9e1c8`
- prompt token lengths: min `402`, median `741`, p90 `1077`, p95 `1237.1`, max `1485`

The parquet was constructed by selecting the exact audited candidate source IDs
in original train-parquet order. The round-trip schema and every selected field
value/order were verified against the source table. The Git manifest contains
hashes and identities only; raw prompt and ground-truth bodies are not copied.

## Freeze re-audit

All checks below were recomputed from the final parquet, not inferred only from
the candidate JSON:

- exact candidate identity and ordered source IDs: PASS
- row count 108 and source schema/value/order: PASS
- zero optimizer-prefix exposure: PASS
- zero SFT v1 and SFT v2 exposure: PASS
- zero formal-validation exposure: PASS
- zero known diagnostic-generation exposure: PASS
- zero historical scoring exposure: PASS
- zero original `test.parquet` source-ID exposure: PASS
- zero exact/normalized prompt overlap against exposed corpora: PASS
- zero exact/normalized prompt+ground-truth overlap: PASS
- zero internal duplicate: PASS
- all prompt lengths `<=2048`: PASS

The historical ToolRL `test.parquet` is retained as observed legacy data and is
not the new endpoint. `FINAL_HOLDOUT_V1` is a fresh previously-unexposed
internal corroborative endpoint, not an external benchmark or proof of general
algorithm superiority.

## Static implementation checks

- `scripts/freeze_final_holdout_v1.py`: PASS; CPU-only row-preserving construction.
- `scripts/verify_final_holdout_v1.py`: PASS; full post-freeze re-audit.
- `scripts/run_final_test.sh` bash syntax: PASS.
- launcher default refusal and output collision guard: PASS.
- launcher default endpoint: `final_holdout_v1.parquet`; old `test.parquet` fallback: absent.
- launcher static manifest/hash/row-count/source-ID assertions: PASS.
- `scripts/analyze_final_test.py` Python compile and self-test: PASS.
- synthetic 108-row analysis fixture: PASS.
- formal-validation regression fixture: PASS (80 persisted rows, no test rows).
- Hydra compose/resolve with official veRL v0.9.1: PASS; resolved config is
  `configs/final_holdout_v1_hydra_resolved_20261003.yaml` with
  `val_only=true`, `val_before_train=true`, and the 108-row endpoint.
- val-only source audit: PASS; upstream validation returns before the training
  loop, and reference-policy construction remains disabled when both KL flags
  are false.
- model inference/reward scoring/background trainer/Ray/vLLM process: **NO**.

## Future evaluation boundary

Before any future GPU endpoint evaluation, run the separate infrastructure
smoke on `formal_validation_80.parquet` first. Only after that smoke passes may
the four frozen model identities be evaluated once each on FINAL_HOLDOUT_V1.
The current session did not perform either step.

## Artifact and shutdown status

- repository manifest: `manifests/final_holdout_v1_manifest.json`
- server-side manifest copy: `/root/autodl-tmp/ProjectB/final-test-preflight/final_holdout_v1_manifest.json`
- machine-readable verification: `docs/diagnostics/final_holdout_v1_freeze_verification.json`
- static gate record: `docs/diagnostics/final_holdout_v1_static_checks.json`
- server-side freeze report: `/root/autodl-tmp/ProjectB/final-test-preflight/FINAL_HOLDOUT_V1_FREEZE_REPORT.md`
- sync: required after final writes
- shutdown: **deferred by the current user instruction**; the no-GPU instance is
  intentionally kept running at negligible cost for follow-up CPU/static work.

An initial verifier attempt exposed only a path typo in the newly added audit
script; it was preserved as `FINAL_HOLDOUT_FREEZE_BLOCKED.initial-verifier-error`,
then the corrected verifier completed all checks above. The parquet was not
changed between attempts.
