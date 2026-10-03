# Documentation revision audit —2026-10-03

Revision base: main@557d899538f393a7d6a0617f47b443223f6dfef4. Scope: public-file CPU checks and P0/P1 documentation only. No server connection, GPU work, inference, formal-validation bootstrap rerun or training continuation.

## Independent public mechanism checks

Inputs: [shared JSON](../../reports/comparison/grpo_gdpo_shared_rollout_diagnostic.json) and [fresh JSON](../../reports/sft/fresh_holdout_reward_variation.json). Absolute tolerance 1e-8; sign_tau(x)=0 for |x|≤tau, otherwise the sign of x.

- Shared records: 32, groups: 8, within-group unordered pairs: 48.
- Exact sign mismatch: 16; all GRPO zero→GDPO small negative; positive↔negative reversals: 0.
- Exact within-group ranking mismatches: 11; all GRPO ties versus GDPO floating differences. Maximum difference: 3.725290298461914e-09.
- Tolerance-aware substantive within-group ranking mismatch: 0; strict rank reversal: 0.
- Fresh records: 128, groups: 32; center each reward separately in its prompt group. Opposite centered signs: 8 trajectories,6 groups; source rows 1024,2536,2062,1054,1471,1118.

Minimal reproduction of the corrected counts using Python standard-library operations on the public records:

```python
from itertools import combinations
from statistics import mean
TOL = 1e-8
def sign(x):
    return (x > 0) - (x < 0)
def sign_tau(x):
    return 1 if x > TOL else -1 if x < -TOL else 0
pairs = [(a, b) for a, b in combinations(shared["records"], 2)
         if a["prompt_group_id"] == b["prompt_group_id"]]
raw_rank = sum(sign(a["grpo_advantage"] - b["grpo_advantage"]) !=
               sign(a["gdpo_advantage"] - b["gdpo_advantage"])
               for a, b in pairs)
meaningful_rank = sum(sign_tau(a["grpo_advantage"] - b["grpo_advantage"]) !=
                      sign_tau(a["gdpo_advantage"] - b["gdpo_advantage"])
                      for a, b in pairs)
conflicts = []
for group in fresh["groups"]:
    fs, acs = group["format_reward_values"], group["accuracy_reward_values"]
    for i, (f, a) in enumerate(zip(fs, acs)):
        if sign_tau(f - mean(fs)) * sign_tau(a - mean(acs)) == -1:
            conflicts.append((group["source_row"], i))
# raw_rank=11; meaningful_rank=0;
# len(conflicts)=8; len({row for row, _ in conflicts})=6
```

## Numerical source boundary

The [three-run comparison JSON](../../reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.json) and three canonical metric files are the numerical sources. Compare mean/count/rate fields over both 80/79 subsets, six steps and all three runs. New result tables use the saved unrounded values, displayed to six decimals. Primary80, sensitivity79, bootstrap estimates and all existing JSON/YAML are preserved byte-for-byte.

The [bootstrap JSON](../../reports/comparison/paired_bootstrap_stage1_35.json) defines right_minus_left. The original/noKL and original/GDPO report keys are corrected. The noKL/GDPO table explicitly reverses the JSON direction, with interval endpoints negated/reversed; this is algebra, not independent resampling. Formal raw validation JSONL is unavailable in GitHub, so per-row pair identity, bootstrap sampling and generation parsing are not independently reverified here.

## Current boundaries

Sensitivity79 is post-hoc; it is not a preregistered clean split. The preserved row-disjoint audit has a content-duplicate correction. Original reports/prelaunch plans receive historical annotations rather than replacement of their recorded facts. Frozen training configurations remain untouched.

Test provenance is unresolved beyond formal Stage1's no-intermediate-test policy. The legacy smoke script's test configuration requires a future execution-log audit before an unseen-test claim. Full-checkpoint structural integrity, model-only loading and full training restore remain distinct. No new major optimizer/objective confound is established by this document-only review.

The [current budget decision](../experiment_design/stage1_closeout_decision.md) fixes 35; the [final-test protocol](../experiment_design/final_test_protocol.md) is not executed. Actual server-dependent hashes, data overlap and generation/analysis-program freeze remain pending.

## Repository checks completed

- 324 mean/count/rate values across three runs, six steps and 80/79 subsets match between the canonical metric files and three-run comparison JSON.
- All existing result-table rows in 16 touched report files are preserved; README endpoint and budget-delta values match the unrounded sources.
- All pre-existing JSON/JSONL/YAML/YML files match the revision base byte-for-byte; all changes are Markdown.
- The three pairwise bootstrap keys exist and their orientations match the accompanying text; archived estimates are unchanged.
- Full-repository consistency search found no remaining preregistration claim for sensitivity79 in current Markdown. Remaining old continuation/not-started statements are in explicitly labelled historical bodies or upstream material.
- Internal Markdown link targets in revised/new files resolve; no newly missing target was introduced. The full scan also encountered pre-existing vendored documentation routes, templates and missing example files, which are outside this ProjectB documentation revision.
- Git whitespace validation passes. No GPU or training test was needed for this documentation-only change.

## Complete file-purpose diff summary

33 Markdown files: 30 revised and 3 new. Seventeen historical documents receive annotation only; their original bodies remain unchanged. No source scripts or frozen machine-readable artifacts are edited.

| repository path | purpose |
|---|---|
| `README.md` | Research question, three-run status/results, bounded conclusions and final-test next step. |
| `configs/formal_experiment_config.md` | Historical annotation only: Pre-run design snapshot; NOT STARTED and the original continuation/retention rules describe that point in time. |
| `docs/diagnostics/documentation_revision_audit_20261003.md` | Verification evidence, reproduction snippet and complete file-purpose diff summary. |
| `docs/diagnostics/formal_validation_split_audit.md` | Preserve row-disjoint proof; add content-duplicate and test-provenance corrections. |
| `docs/diagnostics/gdpo_hidden_kl_use_final_audit.md` | Historical annotation only: Objective-path audit remains valid; future-tense noKL smoke instructions describe the prelaunch phase, not current run status. |
| `docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md` | Historical annotation only: Static prelaunch audit; its no-training-started/smoke-required statements describe that audit. Formal noKL35 later completed. |
| `docs/experiment_design/final_test_protocol.md` | Four-model evaluation, provenance gates, decoding, metrics, pairing and test-use firewall. |
| `docs/experiment_design/format_sft_plan.md` | Historical annotation only: Original SFT proposal; actual frozen initialization is Format SFT v2 / RL_INIT_V1. |
| `docs/experiment_design/format_sft_runbook.md` | Historical annotation only: Pre-training SFT runbook; its 未启动 statements are historical, not current status. |
| `docs/experiment_design/sft_validation_plan.md` | Historical annotation only: Original SFT validation plan, preserved as design history. |
| `docs/experiment_design/stage1_closeout_decision.md` | Accepted 35-step budget, continuation cost/ROI and frozen final models. |
| `docs/experiment_design/staged_formal_experiment_plan.md` | Historical annotation only: Original35→70→105 budget/retention plan; continuation and deletion rules are superseded for this project. |
| `docs/github_audit_20261002.md` | Historical annotation only: Original publication/server snapshot; its branch and progress values are historical. |
| `reports/README.md` | Current evidence index for three runs, KL, diagnostics, statistics and historical snapshots. |
| `reports/comparison/final_stage1_ablation_completion_report.md` | Append current budget decision and unexecuted final-test boundary to recorded closeout. |
| `reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md` | Post-hoc sensitivity, KL-aligned method interpretation and mechanism hypothesis. |
| `reports/comparison/grpo_gdpo_shared_rollout_diagnostic.md` | Raw versus tolerance-aware ranking/sign correction; valid-token statistic interpretation. |
| `reports/comparison/grpo_gdpo_stage1_35_comparison.md` | Historical annotation only: Original two-run comparison, superseded as the current result overview by the three-run report. A KL-aligned noKL run has since completed; the historical A/B confound remains. |
| `reports/comparison/grpo_nokl_vs_gdpo_stage1_35.md` | Post-hoc wording and explicit reversal of archived bootstrap orientation. |
| `reports/comparison/grpo_original_vs_gdpo_stage1_35.md` | Post-hoc wording and corrected bootstrap key/direction. |
| `reports/comparison/grpo_original_vs_nokl_stage1_35.md` | Post-hoc wording and corrected bootstrap key/direction. |
| `reports/comparison/kl_path_audit.md` | Historical annotation only: Initial two-run source/smoke audit; its not-yet-ablated statement predates completed noKL35. |
| `reports/comparison/paired_bootstrap_stage1_35.md` | Pair direction, uncertainty scope and no-independent-bootstrap-recompute boundary. |
| `reports/comparison/primary80_vs_sensitivity79_stage1_35.md` | Fixed post-hoc exclusion and one-row robustness boundary. |
| `reports/comparison/stage1_35x2_completion_report.md` | Historical annotation only: Two-run closeout snapshot before noKL completion. |
| `reports/gdpo/stage1_35/gdpo_stage1_report.md` | Historical annotation only: Original GDPO completion record. The no-third-experiment-started statement predates noKL completion. |
| `reports/grpo/stage1_35/grpo_nokl_stage1_report.md` | Historical annotation only: Historical compatibility copy of the noKL report; canonical report path is reports/grpo_nokl/stage1_35/grpo_nokl_stage1_report.md. |
| `reports/grpo/stage1_35/grpo_stage1_report.md` | Historical annotation only: Original run-completion record. NOT FINAL_BUDGET and resume-policy wording predate the current35-step closeout decision; its numerical results remain valid. |
| `reports/grpo_nokl/stage1_35/grpo_nokl_stage1_report.md` | Historical annotation only: Original noKL completion record. Its not-final-budget wording predates the current budget decision. |
| `reports/runtime_verification_20261002.md` | Historical annotation only: Early runtime snapshot, not the three-run final status. |
| `reports/sft/fresh_holdout_reward_variation.md` | Independent public-record reward-conflict count and its scope. |
| `reports/sft/sft_train_report.md` | Historical annotation only: SFT v1 run-completion snapshot; later SFT v2 and three RL runs completed. |
| `开始这里.md` | Current entry point and reading order; remove stale noKL-not-started status. |
