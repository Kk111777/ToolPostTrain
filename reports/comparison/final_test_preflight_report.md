# ProjectB final-test CPU/no-GPU preflight report

Date: 2026-10-03 (Asia/Shanghai)
Branch: final-test-preflight-20261003
Status: GPU_FINAL_TEST_BLOCKED

## Current server snapshot

- /root/autodl-tmp: 200G total, 113G used, 88G available (57%).
- ProjectB: 113G.
- ProjectB/checkpoints: 3.5G.
- final Stage1 full checkpoints remain under each retained run directory; no
  checkpoint was deleted by this preflight.
- active verl/Ray/vLLM processes: none observed.
- final-test output directory: absent.

## Decision

The CPU/no-GPU preflight completed. No final-test model inference, GPU task,
training, resume, checkpoint write, or test-output read was performed.

The final GPU gate is blocked because persisted historical commands and logs
prove that test.parquet was loaded and scored before this preflight. The
evidence does not prove that those scores drove later budget, hyperparameter,
or checkpoint decisions, but absence of evidence is not proof of non-use.
Manual provenance review is required before treating the test as unseen.

## Passed CPU checks

- test rows: 80
- test SHA256: f7f300ebe755fd04985c83e18097d5a94a08619d72809fe06128ed3f00da7554
- formal validation path: /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet
- formal raw re-audit: complete coverage, finite reward fields, canonical score/format match
- archived paired bootstrap: 10000 resamples, seed 42, match
- model-only and RL_INIT file/header manifests: readable; no AutoModel load
- prompt audit: no prompt over 2048; no truncation
- official upstream scorer identity and source hashes: recorded
- overlap audit: exact and normalized prompt/hash overlaps are zero; source-id overlaps are recorded for manual review
- analysis script compile, self-test and formal-validation fixture: PASS
- official upstream Hydra compose/resolve: PASS
- val_only source-flow audit: PASS
- default launcher guard: PASS; no opt-in environment was supplied

## Explicit non-passes

- historical test provenance: BLOCKED
- final GPU test: NOT RUN
- unseen-test claim: NOT allowed

## Artifact locations

- CPU preflight evidence: /root/autodl-tmp/ProjectB/final-test-preflight/
- repository manifests: manifests/final_test_*
- launcher/config: scripts/run_final_test.sh and configs/final_test_effective_config.yaml
- analysis: scripts/analyze_final_test.py
- static audit: docs/diagnostics/final_test_launcher_static_audit.md
- analysis validation: docs/diagnostics/final_test_analysis_script_validation.md

## Safety boundary

No 70/105 training, retraining, resume, outcome-driven row deletion, or
additional experiment was initiated. The opt-in launcher remains disabled.
