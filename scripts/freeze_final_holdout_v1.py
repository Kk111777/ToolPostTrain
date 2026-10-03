#!/usr/bin/env python3
"""Freeze ProjectB FINAL_HOLDOUT_V1 from the audited candidate identities.

CPU-only and model-free.  The parquet is a row-preserving take from the
official training parquet; this script never generates, scores, or loads a
model.  The committed manifest contains hashes and identities, not prompt or
ground-truth text.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(os.environ.get("PROJECTB_ROOT", "/root/autodl-tmp/ProjectB"))
REPO = ROOT / "repo"
TRAIN = REPO / "verl-GDPO/dataset/rlla_4k/train.parquet"
CANDIDATE = REPO / "docs/diagnostics/final_holdout_candidate_inventory_20261003.json"
OUT = ROOT / "env-modern/final_holdout_v1.parquet"
MANIFEST = REPO / "manifests/final_holdout_v1_manifest.json"
SERVER_MANIFEST = ROOT / "final-test-preflight/final_holdout_v1_manifest.json"
TOKENIZER_PATH = ROOT / "checkpoints/rl_init_sft_v2_merged"
INPUT_COMMIT = "a489a59e0117eb5d469297e15dda6ca4417ad80c"
MAX_PROMPT = 2048
EXPECTED_N = 108


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).replace("\r\n", "\n").replace("\r", "\n")


def target_category(ground_truth: str) -> str:
    if "<tool_call>" in ground_truth and "<response>" in ground_truth:
        return "tool_and_response"
    if "<tool_call>" in ground_truth:
        return "tool_only"
    if "<response>" in ground_truth:
        return "response_only"
    return "other"


def row_signature(row: dict[str, Any], source_id: int) -> dict[str, str]:
    messages = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in (row.get("prompt") or [])
    ]
    ground_truth = str((row.get("reward_model") or {}).get("ground_truth", ""))
    exact = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    normalized_messages = [
        {"role": norm(item["role"]), "content": norm(item["content"])}
        for item in messages
    ]
    normalized = json.dumps(
        normalized_messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return {
        "source_id": str(source_id),
        "target_category": target_category(ground_truth),
        "prompt_sha256_exact": sha256_text(exact),
        "prompt_sha256_normalized": sha256_text(normalized),
        "prompt_ground_truth_sha256_exact": sha256_text(
            exact + "\n<GROUND_TRUTH>\n" + ground_truth
        ),
        "prompt_ground_truth_sha256_normalized": sha256_text(
            normalized + "\n<GROUND_TRUTH>\n" + norm(ground_truth)
        ),
        "ground_truth_sha256_exact": sha256_text(ground_truth),
        "ground_truth_sha256_normalized": sha256_text(norm(ground_truth)),
    }


def prompt_length(tokenizer: Any, row: dict[str, Any]) -> int:
    encoded = tokenizer.apply_chat_template(
        row.get("prompt") or [], tokenize=True, add_generation_prompt=True
    )
    try:
        encoded = encoded["input_ids"]
    except (KeyError, TypeError, IndexError):
        pass
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def tokenizer_identity() -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    return tokenizer, {
        "path": str(TOKENIZER_PATH),
        "name_or_path": str(getattr(tokenizer, "name_or_path", TOKENIZER_PATH)),
        "class": tokenizer.__class__.__name__,
        "chat_template_sha256": sha256_text(str(chat_template)) if chat_template is not None else None,
        "chat_template_present": chat_template is not None,
    }


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": int(min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": int(max(values)),
    }


def main() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    candidate_ids = [int(value) for value in candidate["candidate_source_ids_after_overlength"]]
    if len(candidate_ids) != EXPECTED_N or len(set(candidate_ids)) != EXPECTED_N:
        raise RuntimeError("candidate inventory is not exactly 108 unique source IDs")
    if candidate["counts"]["final_untouched_candidates"] != EXPECTED_N:
        raise RuntimeError("candidate inventory count disagrees with its source ID list")

    source = pq.read_table(TRAIN)
    source_rows = source.to_pylist()
    source_id_to_position: dict[int, int] = {}
    for position, row in enumerate(source_rows):
        source_id = int((row.get("extra_info") or {}).get("index"))
        if source_id in source_id_to_position:
            raise RuntimeError(f"duplicate source ID in train parquet: {source_id}")
        source_id_to_position[source_id] = position
    if not set(candidate_ids) <= set(source_id_to_position):
        raise RuntimeError("candidate inventory contains an ID absent from train parquet")

    # Selecting by original table position preserves source order and every
    # field value.  Candidate JSON ordering is retained separately as an
    # identity check, not used to reorder the derived parquet.
    selected_positions = [
        position
        for position, row in enumerate(source_rows)
        if int((row.get("extra_info") or {}).get("index")) in set(candidate_ids)
    ]
    ordered_source_ids = [
        int((source_rows[position].get("extra_info") or {}).get("index"))
        for position in selected_positions
    ]
    if len(ordered_source_ids) != EXPECTED_N or set(ordered_source_ids) != set(candidate_ids):
        raise RuntimeError("source-order selection does not preserve the exact candidate set")

    tokenizer, tokenizer_info = tokenizer_identity()
    candidate_rows = {
        int(item["source_id"]): item for item in candidate["final_candidate_rows"]
    }
    manifest_rows: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    for position, source_id in zip(selected_positions, ordered_source_ids):
        row = source_rows[position]
        sig = row_signature(row, source_id)
        tokens = prompt_length(tokenizer, row)
        expected = candidate_rows[source_id]
        for field in (
            "target_category",
            "prompt_sha256_exact",
            "prompt_sha256_normalized",
            "prompt_ground_truth_sha256_exact",
            "prompt_ground_truth_sha256_normalized",
        ):
            if sig[field] != str(expected[field]):
                raise RuntimeError(f"candidate identity mismatch for source {source_id}: {field}")
        if tokens != int(expected["prompt_tokens"]):
            raise RuntimeError(f"token length mismatch for source {source_id}")
        if tokens > MAX_PROMPT:
            raise RuntimeError(f"candidate source {source_id} is overlength")
        token_lengths.append(tokens)
        manifest_rows.append(
            {
                "row_position": int(position),
                "source_id": int(source_id),
                "candidate_tail_position": int(expected["tail_position"]),
                "target_category": sig["target_category"],
                "prompt_tokens": int(tokens),
                "prompt_sha256_exact": sig["prompt_sha256_exact"],
                "prompt_sha256_normalized": sig["prompt_sha256_normalized"],
                "prompt_ground_truth_sha256_exact": sig["prompt_ground_truth_sha256_exact"],
                "prompt_ground_truth_sha256_normalized": sig["prompt_ground_truth_sha256_normalized"],
                "ground_truth_sha256_exact": sig["ground_truth_sha256_exact"],
                "ground_truth_sha256_normalized": sig["ground_truth_sha256_normalized"],
                "data_source": str(row.get("data_source", "rlla")),
            }
        )

    selected = source.take(pa.array(selected_positions, type=pa.int64()))
    if not selected.schema.equals(source.schema, check_metadata=True):
        raise RuntimeError("selected parquet schema differs from source schema")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, OUT)
    roundtrip = pq.read_table(OUT)
    if len(roundtrip) != EXPECTED_N or not roundtrip.schema.equals(source.schema, check_metadata=True):
        raise RuntimeError("round-trip parquet row count or schema mismatch")
    if not roundtrip.equals(selected, check_metadata=True):
        raise RuntimeError("round-trip parquet field values/order mismatch")

    target_counts = dict(sorted(Counter(row["target_category"] for row in manifest_rows).items()))
    source_ids_sha256 = sha256_text("\n".join(map(str, ordered_source_ids)))
    manifest = {
        "schema": "projectb_final_holdout_v1_manifest",
        "manifest_version": "FINAL_HOLDOUT_V1",
        "status": "frozen_before_final_inference",
        "endpoint": {
            "name": "FINAL_HOLDOUT_V1",
            "kind": "fresh_previously_unexposed_internal_holdout",
            "not_official_toolrl_test": True,
            "not_external_benchmark": True,
            "not_proof_of_general_algorithm_superiority": True,
        },
        "row_count": EXPECTED_N,
        "source_train_parquet": str(TRAIN),
        "source_train_parquet_sha256": sha256_file(TRAIN),
        "derived_parquet": str(OUT),
        "derived_parquet_sha256": sha256_file(OUT),
        "ordered_source_ids": ordered_source_ids,
        "ordered_source_ids_sha256": source_ids_sha256,
        "candidate_inventory_source_ids": candidate_ids,
        "candidate_inventory_source_ids_sha256": sha256_text("\n".join(map(str, candidate_ids))),
        "target_category_distribution": target_counts,
        "prompt_token_distribution": distribution(token_lengths),
        "rows": manifest_rows,
        "exclusion_summary": {
            "raw_train_rows": candidate["counts"]["raw_train_rows"],
            "eligible_rows": candidate["counts"]["eligible_rows"],
            "optimizer_exposed_rows": candidate["counts"]["optimizer_exposed_rows"],
            "initial_tail_rows": candidate["counts"]["initial_tail_rows"],
            "union_exposure_exclusions": candidate["counts"]["union_exposure_exclusions"],
            "content_duplicate_exclusions": candidate["counts"]["content_duplicate_exclusions"],
            "internal_duplicate_exclusions": candidate["counts"]["internal_duplicate_exclusions"],
            "overlength_exclusions": candidate["counts"]["overlength_exclusions"],
            "optimizer_source_sequence_sha256": candidate["optimizer_source_sequence_sha256"],
            "exposure_categories": candidate["exposure_categories"],
        },
        "normalization": {
            "text": "Unicode NFC; CRLF/CR to LF",
            "serialization": "sorted role/content JSON with argument values preserved",
            "duplicate_policy": "exact and normalized prompt plus prompt-ground-truth hashes; no fuzzy or semantic filtering",
        },
        "eligibility": {
            "official_filter": "prompt tokens <= 2048 under RL_INIT_V1 tokenizer/chat template",
            "no_truncation": True,
            "source_order": "original train.parquet order",
        },
        "tokenizer": tokenizer_info,
        "construction": {
            "script": "scripts/freeze_final_holdout_v1.py",
            "input_audit_commit": INPUT_COMMIT,
            "constructed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "cpu_only": True,
            "model_loaded": False,
            "inference_performed": False,
            "reward_scoring_performed": False,
        },
        "historical_test_note": {
            "path": str(REPO / "verl-GDPO/dataset/rlla_4k/test.parquet"),
            "retained_as_historical_observed_data": True,
            "primary_endpoint": False,
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SERVER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MANIFEST, SERVER_MANIFEST)
    print(json.dumps({
        "status": "FINAL_HOLDOUT_V1_CONSTRUCTED",
        "row_count": EXPECTED_N,
        "ordered_source_ids_sha256": source_ids_sha256,
        "derived_parquet_sha256": manifest["derived_parquet_sha256"],
        "target_category_distribution": target_counts,
        "prompt_token_distribution": manifest["prompt_token_distribution"],
        "manifest": str(MANIFEST),
        "server_manifest": str(SERVER_MANIFEST),
        "inference_performed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
