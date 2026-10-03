#!/usr/bin/env python3
"""Audit the runtime mapping against persisted formal-validation JSONL only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_ENDPOINT_MANIFEST_SHA256 = (
    "160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16"
)
EXPECTED_RUNTIME_MAPPING_SHA256 = (
    "ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd"
)
EXPECTED_PARQUET_SHA256 = (
    "e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22"
)
EXPECTED_SOURCE_IDS_SHA256 = (
    "75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_jsonl(path: Path, mapping_rows: list[dict]) -> dict:
    by_key = {
        (row["runtime_input_sha256"], row["ground_truth_sha256_exact"]): row
        for row in mapping_rows
    }
    matched: list[dict] = []
    missing: list[int] = []
    duplicates: list[int] = []
    seen: set[tuple[str, str]] = set()
    for position, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        input_hash = hashlib.sha256(str(item.get("input", "")).encode("utf-8")).hexdigest()
        gt_hash = hashlib.sha256(str(item.get("gts", "")).encode("utf-8")).hexdigest()
        key = (input_hash, gt_hash)
        if key not in by_key:
            missing.append(position)
            continue
        if key in seen:
            duplicates.append(position)
            continue
        seen.add(key)
        matched.append(by_key[key])
    unique_source_ids = len({int(row["source_id"]) for row in matched})
    return {
        "path": str(path),
        "rows": len(matched),
        "expected_rows": 80,
        "matched_rows": len(matched),
        "missing_rows": len(missing),
        "duplicate_rows": len(duplicates),
        "unique_source_ids": unique_source_ids,
        "pass": len(matched) == 80 and not missing and not duplicates and unique_source_ids == 80,
    }


def build_formal_runtime_mapping(root: Path) -> list[dict]:
    """Build an ephemeral 80-row mapping with the same official runtime path."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoConfig, AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    tokenizer_path = root / "checkpoints/rl_init_sft_v2_merged"
    formal = root / "env-modern/formal_validation_80.parquet"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    hf_config = AutoConfig.from_pretrained(str(tokenizer_path), local_files_only=True)
    config = {
        "cache_dir": str(root / "tmp-runtime-mapping-cache"),
        "prompt_key": "prompt",
        "max_prompt_length": 2048,
        "return_raw_chat": True,
        "return_full_prompt": False,
        "truncation": "error",
        "filter_overlong_prompts": False,
        "filter_overlong_prompts_workers": 1,
        "apply_chat_template_kwargs": {},
        "mm_processor_kwargs": {},
        "tool_config_path": None,
        "function_tool_path": None,
        "need_tools_kwargs": False,
        "filter_prompts": True,
        "return_multi_modal_inputs": True,
        "shuffle": False,
        "seed": 42,
        "use_shm": False,
        "image_key": "images",
        "video_key": "videos",
        "audio_key": "audios",
    }
    dataset = RLHFDataset(
        data_files=[str(formal)], tokenizer=tokenizer, processor=None, config=config, max_samples=-1
    )
    builder = create_continuous_token_builder(
        tokenizer,
        hf_model_type=getattr(hf_config, "model_type", None),
        chat_template_kwargs={},
        processor=None,
    )
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        source_id = int((sample.get("extra_info") or {}).get("index", index))
        prompt_ids = builder.build_initial_tokens(sample["raw_prompt"])
        runtime = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ground_truth = str((sample.get("reward_model") or {}).get("ground_truth", ""))
        rows.append(
            {
                "source_id": source_id,
                "row_position": index,
                "runtime_input_sha256": hashlib.sha256(runtime.encode("utf-8")).hexdigest(),
                "ground_truth_sha256_exact": hashlib.sha256(ground_truth.encode("utf-8")).hexdigest(),
            }
        )
    if len(rows) != 80 or len({(r["runtime_input_sha256"], r["ground_truth_sha256_exact"]) for r in rows}) != 80:
        raise RuntimeError("formal runtime mapping is not 80/80 unique")
    return rows
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/ProjectB"))
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    root = args.root
    repo = root / "repo"
    endpoint_manifest_path = repo / "manifests/final_holdout_v1_manifest.json"
    mapping_path = repo / "manifests/final_holdout_v1_runtime_mapping.json"
    endpoint_manifest = json.loads(endpoint_manifest_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    endpoint_sha = sha256_file(endpoint_manifest_path)
    mapping_sha = sha256_file(mapping_path)
    rows = mapping["rows"]
    pairs = [(row["runtime_input_sha256"], row["ground_truth_sha256_exact"]) for row in rows]
    identity_pass = (
        endpoint_sha == EXPECTED_ENDPOINT_MANIFEST_SHA256
        and mapping_sha == EXPECTED_RUNTIME_MAPPING_SHA256
        and endpoint_manifest["derived_parquet_sha256"] == EXPECTED_PARQUET_SHA256
        and endpoint_manifest["ordered_source_ids_sha256"] == EXPECTED_SOURCE_IDS_SHA256
        and mapping["parent_endpoint_manifest_sha256"] == EXPECTED_ENDPOINT_MANIFEST_SHA256
        and mapping["endpoint_parquet_sha256"] == EXPECTED_PARQUET_SHA256
        and mapping["ordered_source_ids_sha256"] == EXPECTED_SOURCE_IDS_SHA256
        and len(rows) == 108
        and len(set(pairs)) == 108
    )
    checks = {}
    # The FINAL_HOLDOUT_V1 mapping is intentionally not used for formal rows:
    # this is an independent real-trainer regression using the same construction
    # semantics on formal_validation_80.parquet.
    formal_rows = build_formal_runtime_mapping(root)
    for run in ("grpo_stage1_35", "gdpo_stage1_35", "grpo_nokl_stage1_35"):
        path = root / "runs" / run / "eval_generations" / f"{args.step}.jsonl"
        checks[run] = audit_jsonl(path, formal_rows)
        checks[run]["mapping_source"] = "formal_validation_80.parquet (ephemeral; same official runtime semantics)"
    result = {
        "schema": "projectb_final_eval_runtime_mapping_audit_v1",
        "endpoint_manifest_sha256": endpoint_sha,
        "runtime_mapping_sha256": mapping_sha,
        "mapping_identity_pass": identity_pass,
        "checks": checks,
        "pass": identity_pass and all(item["pass"] for item in checks.values()),
        "no_model_inference": True,
        "no_final_holdout_scoring": True,
        "no_positional_fallback": True,
    }
    out_json = args.out_json or repo / "docs/diagnostics/final_eval_runtime_mapping_audit.json"
    out_md = args.out_md or repo / "docs/diagnostics/final_eval_runtime_mapping_audit.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "REAL_TRAINER_JSONL_MAPPING_PASS" if result["pass"] else "BLOCKED"
    lines = [
        "# FINAL_HOLDOUT_V1 runtime mapping audit",
        "",
        f"Gate: **{status}**",
        "",
        "This CPU-only audit maps persisted veRL formal-validation JSONL by `(runtime_input_sha256, ground_truth_sha256_exact)`.",
        "It does not use file position, fuzzy matching, model inference, or FINAL_HOLDOUT_V1 scoring.",
        "",
        f"- endpoint manifest SHA256: `{endpoint_sha}`",
        f"- runtime mapping SHA256: `{mapping_sha}`",
        f"- runtime mapping pairs: `{len(set(pairs))}/108` unique",
        f"- endpoint parquet SHA256: `{endpoint_manifest['derived_parquet_sha256']}`",
        f"- ordered source-ID SHA256: `{endpoint_manifest['ordered_source_ids_sha256']}`",
        "",
        "| persisted JSONL | rows matched | missing | duplicates | unique source IDs | result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for run, check in checks.items():
        lines.append(
            f"| `{run}` step {args.step} | {check['matched_rows']}/80 | {check['missing_rows']} | {check['duplicate_rows']} | {check['unique_source_ids']} | {'PASS' if check['pass'] else 'BLOCKED'} |"
        )
    lines += [
        "",
        "The mapping was constructed with official veRL v0.9.1 RLHFDataset and the text-only ContinuousToken builder using the frozen RL_INIT_V1 tokenizer; no model was loaded.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
