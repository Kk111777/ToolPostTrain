#!/usr/bin/env python3
"""Re-audit FINAL_HOLDOUT_V1 from the frozen parquet, CPU-only.

This is deliberately independent of the candidate parquet construction step:
it reads the derived parquet back, reconstructs source identities/signatures,
and blocks rather than repairing any mismatch.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

import freeze_final_holdout_v1 as freeze


ROOT = freeze.ROOT
REPO = freeze.REPO
TRAIN = freeze.TRAIN
CANDIDATE = freeze.CANDIDATE
FINAL = freeze.OUT
MANIFEST = freeze.MANIFEST
REPORT_JSON = REPO / "docs/diagnostics/final_holdout_v1_freeze_verification.json"
REPORT_MD = REPO / "docs/diagnostics/final_holdout_v1_freeze_verification.md"
BLOCKED = ROOT / "final-test-preflight/FINAL_HOLDOUT_FREEZE_BLOCKED"
FIELDS = (
    "prompt_sha256_exact",
    "prompt_sha256_normalized",
    "prompt_ground_truth_sha256_exact",
    "prompt_ground_truth_sha256_normalized",
)
EXPOSURE_CATEGORIES = (
    "training_optimizer",
    "sft_v1",
    "sft_v2",
    "formal_validation",
    "original_test",
    "fixed_diagnostic",
    "constrained_generation_diagnostic",
    "fresh_holdout_reward_variation",
    "shared_rollout_diagnostic",
    "capacity_smoke",
    "throughput_tuning",
    "one_step_smoke",
    "old_formal_grpo",
    "historical_scoring_generation",
    "unknown_persisted_evidence",
)


def source_id(row: dict[str, Any]) -> int:
    return int((row.get("extra_info") or {}).get("index"))


def signatures(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result = {field: set() for field in FIELDS}
    for row in rows:
        sid = source_id(row)
        sig = freeze.row_signature(row, sid)
        for field in FIELDS:
            result[field].add(sig[field])
    return result


def sft_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            assistant = next(
                (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "assistant"),
                None,
            )
            if assistant is None:
                continue
            rows.append(
                {
                    "prompt": messages[:assistant],
                    "reward_model": {"ground_truth": messages[assistant].get("content", "")},
                    "extra_info": {"index": item.get("source_row", -1)},
                }
            )
    return rows


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def audit() -> dict[str, Any]:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = pq.read_table(TRAIN)
    final = pq.read_table(FINAL)
    source_rows = source.to_pylist()
    final_rows = final.to_pylist()
    candidate_ids = [int(value) for value in candidate["candidate_source_ids_after_overlength"]]
    candidate_set = set(candidate_ids)
    source_ids = [source_id(row) for row in source_rows]
    final_ids = [source_id(row) for row in final_rows]
    source_by_id = {source_id(row): row for row in source_rows}
    expected_positions = [index for index, sid in enumerate(source_ids) if sid in candidate_set]
    expected = source.take(__import__("pyarrow").array(expected_positions, type=__import__("pyarrow").int64()))

    checks: list[dict[str, Any]] = []
    checks.append(check("row_count_108", len(final_rows) == 108, len(final_rows)))
    checks.append(check("candidate_identity_exact", set(final_ids) == candidate_set and len(set(final_ids)) == 108, {
        "candidate_count": len(candidate_ids),
        "final_count": len(final_ids),
        "ordered_source_ids_sha256": freeze.sha256_text("\n".join(map(str, final_ids))),
    }))
    checks.append(check("source_order_preserved", final.equals(expected, check_metadata=True), "final equals source-row take"))
    checks.append(check("schema_preserved", final.schema.equals(source.schema, check_metadata=True), str(final.schema)))
    checks.append(check("candidate_manifest_source_id_hash", manifest["candidate_inventory_source_ids_sha256"] == freeze.sha256_text("\n".join(map(str, candidate_ids))), manifest["candidate_inventory_source_ids_sha256"]))
    checks.append(check("derived_sha256_matches_manifest", freeze.sha256_file(FINAL) == manifest["derived_parquet_sha256"], freeze.sha256_file(FINAL)))
    checks.append(check("source_sha256_matches_manifest", freeze.sha256_file(TRAIN) == manifest["source_train_parquet_sha256"], freeze.sha256_file(TRAIN)))
    checks.append(check("manifest_order_hash_matches_final", manifest["ordered_source_ids_sha256"] == freeze.sha256_text("\n".join(map(str, final_ids))), manifest["ordered_source_ids_sha256"]))

    manifest_ids = [int(row["source_id"]) for row in manifest["rows"]]
    checks.append(check("manifest_rows_match_final", manifest_ids == final_ids and len(manifest_ids) == 108, {
        "manifest_count": len(manifest_ids),
        "final_count": len(final_ids),
    }))

    exposure = candidate.get("exposure_union_source_ids", [])
    exposure_overlap = sorted(candidate_set & {int(value) for value in exposure})
    checks.append(check("zero_union_exposure", not exposure_overlap, exposure_overlap))
    for category in EXPOSURE_CATEGORIES:
        values = candidate.get("exposure_categories", {}).get(category, {}).get("candidate_source_ids", [])
        overlap = sorted(candidate_set & {int(value) for value in values})
        checks.append(check(f"zero_{category}_exposure", not overlap, overlap))

    # Recompute tokenizer lengths on the final parquet, rather than trusting
    # the candidate JSON's earlier values.
    tokenizer, _ = freeze.tokenizer_identity()
    manifest_by_id = {int(row["source_id"]): row for row in manifest["rows"]}
    lengths = {}
    for sid, row in zip(final_ids, final_rows):
        length = freeze.prompt_length(tokenizer, row)
        lengths[sid] = length
        checks.append(check(f"prompt_length_{sid}", length <= 2048 and length == int(manifest_by_id[sid]["prompt_tokens"]), length))
    checks.append(check("all_prompt_lengths_le_2048", all(item["passed"] for item in checks if item["name"].startswith("prompt_length_")), {
        "min": min(lengths.values()),
        "max": max(lengths.values()),
    }))

    final_signatures = {sid: freeze.row_signature(row, sid) for sid, row in zip(final_ids, final_rows)}
    corpora = {
        "training_optimizer": signatures([source_by_id[sid] for sid in candidate["optimizer_exposed_source_ids"]]),
        "sft_v1": signatures(sft_rows(REPO / "dataset/format_sft_800.jsonl")),
        "sft_v2": signatures(sft_rows(REPO / "dataset/format_sft_800_v2.jsonl")),
        "formal_validation": signatures(pq.read_table(ROOT / "env-modern/formal_validation_80.parquet").to_pylist()),
        "original_test": signatures(pq.read_table(REPO / "verl-GDPO/dataset/rlla_4k/test.parquet").to_pylist()),
        "capacity_smoke": signatures(pq.read_table(ROOT / "env-modern/capacity_smoke_512.parquet").to_pylist()),
    }
    content_overlaps: list[dict[str, Any]] = []
    for sid, sig in final_signatures.items():
        for corpus_name, corpus in corpora.items():
            for field in FIELDS:
                if sig[field] in corpus[field]:
                    content_overlaps.append({"source_id": sid, "corpus": corpus_name, "field": field})
    checks.append(check("zero_exact_normalized_content_overlap", not content_overlaps, content_overlaps[:50]))

    internal_overlaps: list[dict[str, Any]] = []
    for field in FIELDS:
        seen: dict[str, list[int]] = defaultdict(list)
        for sid, sig in final_signatures.items():
            seen[sig[field]].append(sid)
        for digest, ids in seen.items():
            if len(ids) > 1:
                internal_overlaps.append({"field": field, "source_ids": ids})
    checks.append(check("zero_internal_duplicate", not internal_overlaps, internal_overlaps))
    candidate_duplicate_ids = set()
    for key in candidate.get("content_duplicate_details", {}):
        candidate_duplicate_ids.add(int(key))
    checks.append(check("candidate_content_duplicate_audit_empty", not (candidate_set & candidate_duplicate_ids), sorted(candidate_set & candidate_duplicate_ids)))
    checks.append(check("no_model_output_present", not any((ROOT / "final-test").rglob("*.jsonl")) if (ROOT / "final-test").exists() else True, "final-test JSONL absent"))

    category_counts = defaultdict(int)
    for row in manifest["rows"]:
        category_counts[row["target_category"]] += 1
    result = {
        "schema": "projectb_final_holdout_v1_freeze_verification",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "FINAL_HOLDOUT_V1_FROZEN_PASS" if all(item["passed"] for item in checks) else "FINAL_HOLDOUT_FREEZE_BLOCKED",
        "row_count": len(final_rows),
        "ordered_source_ids_sha256": freeze.sha256_text("\n".join(map(str, final_ids))),
        "derived_parquet_sha256": freeze.sha256_file(FINAL),
        "target_category_distribution": dict(sorted(category_counts.items())),
        "prompt_token_distribution": {
            "n": len(lengths),
            "min": min(lengths.values()),
            "max": max(lengths.values()),
        },
        "checks": checks,
        "inference_performed": False,
        "reward_scoring_performed": False,
    }
    return result


def write_blocked(message: str) -> None:
    BLOCKED.parent.mkdir(parents=True, exist_ok=True)
    BLOCKED.write_text(message + "\n", encoding="utf-8")


def main() -> None:
    try:
        result = audit()
    except Exception:
        message = "FINAL_HOLDOUT_FREEZE_BLOCKED\n" + traceback.format_exc()
        write_blocked(message)
        raise
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FINAL_HOLDOUT_V1 freeze verification",
        "",
        f"- gate: **{result['status']}**",
        f"- row count: **{result['row_count']}**",
        f"- ordered source ID SHA256: `{result['ordered_source_ids_sha256']}`",
        f"- derived parquet SHA256: `{result['derived_parquet_sha256']}`",
        f"- target distribution: `{json.dumps(result['target_category_distribution'], sort_keys=True)}`",
        f"- prompt length range: `{result['prompt_token_distribution']}`",
        "- model inference: **NO**",
        "- reward scoring: **NO**",
        "",
        "## Checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for item in result["checks"]:
        detail = json.dumps(item["detail"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{item['name']}` | {'PASS' if item['passed'] else 'FAIL'} | `{detail}` |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if result["status"] != "FINAL_HOLDOUT_V1_FROZEN_PASS":
        write_blocked(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
