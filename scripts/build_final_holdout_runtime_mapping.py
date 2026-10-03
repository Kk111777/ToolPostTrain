#!/usr/bin/env python3
"""Build the frozen endpoint's veRL runtime-input mapping without inference.

The endpoint manifest intentionally hashes canonical source messages.  veRL's
validation JSONL instead records ``tokenizer.decode`` of the token IDs produced
by ``RLHFDataset`` plus the SingleTurnAgentLoop ContinuousToken builder.  This
script reproduces that CPU-only path and stores only hashes, never prompt or
ground-truth text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_ENDPOINT_MANIFEST_SHA256 = (
    "160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16"
)
EXPECTED_PARQUET_SHA256 = (
    "e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22"
)
EXPECTED_SOURCE_IDS_SHA256 = (
    "75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad"
)
EXPECTED_ROW_COUNT = 108
MAX_PROMPT_LENGTH = 2048


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


def file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def tokenizer_identity(tokenizer_path: Path, tokenizer: Any) -> dict[str, Any]:
    return {
        "path": str(tokenizer_path),
        "name_or_path": str(getattr(tokenizer, "name_or_path", tokenizer_path)),
        "class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_config_sha256": file_hash(tokenizer_path / "tokenizer_config.json"),
        "special_tokens_map_sha256": file_hash(tokenizer_path / "special_tokens_map.json"),
        "tokenizer_json_sha256": file_hash(tokenizer_path / "tokenizer.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PROJECTB_ROOT", "/root/autodl-tmp/ProjectB")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root
    repo = root / "repo"
    endpoint = root / "env-modern/final_holdout_v1.parquet"
    endpoint_manifest_path = repo / "manifests/final_holdout_v1_manifest.json"
    output = args.output or repo / "manifests/final_holdout_v1_runtime_mapping.json"
    tokenizer_path = root / "checkpoints/rl_init_sft_v2_merged"
    verl_path = root / "upstream/verl-v0.9.1"

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    endpoint_manifest_sha256 = sha256_file(endpoint_manifest_path)
    if endpoint_manifest_sha256 != EXPECTED_ENDPOINT_MANIFEST_SHA256:
        raise SystemExit("RUNTIME_MAPPING_REFUSED: endpoint manifest SHA256 changed")
    parquet_sha256 = sha256_file(endpoint)
    if parquet_sha256 != EXPECTED_PARQUET_SHA256:
        raise SystemExit("RUNTIME_MAPPING_REFUSED: endpoint parquet SHA256 changed")

    manifest = load_json(endpoint_manifest_path)
    if manifest.get("manifest_version") != "FINAL_HOLDOUT_V1":
        raise SystemExit("RUNTIME_MAPPING_REFUSED: wrong endpoint manifest version")
    rows = manifest.get("rows", [])
    if manifest.get("row_count") != EXPECTED_ROW_COUNT or len(rows) != EXPECTED_ROW_COUNT:
        raise SystemExit("RUNTIME_MAPPING_REFUSED: endpoint row count is not 108")
    ordered_ids = [int(row["source_id"]) for row in rows]
    ordered_ids_sha256 = sha256_text("\n".join(map(str, ordered_ids)))
    if ordered_ids_sha256 != EXPECTED_SOURCE_IDS_SHA256:
        raise SystemExit("RUNTIME_MAPPING_REFUSED: endpoint source-ID order changed")
    if manifest.get("derived_parquet_sha256") != EXPECTED_PARQUET_SHA256:
        raise SystemExit("RUNTIME_MAPPING_REFUSED: manifest parquet identity changed")

    from transformers import AutoConfig, AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    hf_config = AutoConfig.from_pretrained(str(tokenizer_path), local_files_only=True)
    data_config = {
        "cache_dir": str(root / "tmp-runtime-mapping-cache"),
        "prompt_key": "prompt",
        "max_prompt_length": MAX_PROMPT_LENGTH,
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
        data_files=[str(endpoint)],
        tokenizer=tokenizer,
        processor=None,
        config=data_config,
        max_samples=-1,
    )
    if len(dataset) != EXPECTED_ROW_COUNT:
        raise SystemExit(f"RUNTIME_MAPPING_REFUSED: RLHFDataset produced {len(dataset)} rows")

    builder = create_continuous_token_builder(
        tokenizer,
        hf_model_type=getattr(hf_config, "model_type", None),
        chat_template_kwargs=data_config["apply_chat_template_kwargs"],
        processor=None,
    )

    dataset_by_source: dict[int, dict[str, Any]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        source_id = int((sample.get("extra_info") or {}).get("index"))
        if source_id in dataset_by_source:
            raise SystemExit(f"RUNTIME_MAPPING_REFUSED: duplicate source ID {source_id}")
        messages = sample["raw_prompt"]
        prompt_ids = builder.build_initial_tokens(messages)
        if len(prompt_ids) > MAX_PROMPT_LENGTH:
            raise SystemExit(f"RUNTIME_MAPPING_REFUSED: source {source_id} exceeds prompt cap")
        runtime_input = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ground_truth = str((sample.get("reward_model") or {}).get("ground_truth", ""))
        dataset_by_source[source_id] = {
            "runtime_input_sha256": sha256_text(runtime_input),
            "ground_truth_sha256_exact": sha256_text(ground_truth),
            "prompt_tokens": len(prompt_ids),
        }

    mapping_rows: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str]] = set()
    for endpoint_row in rows:
        source_id = int(endpoint_row["source_id"])
        runtime = dataset_by_source.get(source_id)
        if runtime is None:
            raise SystemExit(f"RUNTIME_MAPPING_REFUSED: missing source ID {source_id}")
        if runtime["ground_truth_sha256_exact"] != endpoint_row["ground_truth_sha256_exact"]:
            raise SystemExit(f"RUNTIME_MAPPING_REFUSED: ground truth mismatch for {source_id}")
        pair = (runtime["runtime_input_sha256"], runtime["ground_truth_sha256_exact"])
        if pair in pair_keys:
            raise SystemExit(f"RUNTIME_MAPPING_REFUSED: duplicate runtime pair for {source_id}")
        pair_keys.add(pair)
        mapping_rows.append(
            {
                "source_id": source_id,
                "row_position": int(endpoint_row["row_position"]),
                "runtime_input_sha256": runtime["runtime_input_sha256"],
                "ground_truth_sha256_exact": runtime["ground_truth_sha256_exact"],
                "prompt_sha256_exact": endpoint_row["prompt_sha256_exact"],
                "target_category": endpoint_row["target_category"],
                "runtime_prompt_tokens": int(runtime["prompt_tokens"]),
            }
        )

    mapping = {
        "schema": "projectb_final_holdout_v1_runtime_mapping",
        "mapping_version": "FINAL_HOLDOUT_V1_RUNTIME_MAPPING_V1",
        "status": "frozen_before_final_inference",
        "row_count": EXPECTED_ROW_COUNT,
        "unique_runtime_ground_truth_pairs": len(pair_keys) == EXPECTED_ROW_COUNT,
        "parent_endpoint_manifest_sha256": endpoint_manifest_sha256,
        "endpoint_parquet_sha256": parquet_sha256,
        "ordered_source_ids_sha256": ordered_ids_sha256,
        "tokenizer": tokenizer_identity(tokenizer_path, tokenizer),
        "chat_template_sha256": sha256_text(str(getattr(tokenizer, "chat_template", None))),
        "official_verl_commit": git_revision(verl_path),
        "runtime_semantics": {
            "dataset_class": "verl.utils.dataset.rl_dataset.RLHFDataset",
            "agent_loop": "verl.experimental.agent_loop.single_turn_agent_loop.SingleTurnAgentLoop",
            "token_builder": builder.__class__.__module__ + "." + builder.__class__.__name__,
            "add_generation_prompt": True,
            "tools": None,
            "decode": "tokenizer.decode(ids, skip_special_tokens=True)",
            "prompt_length_cap": MAX_PROMPT_LENGTH,
            "shuffle": False,
        },
        "construction": {
            "script": "scripts/build_final_holdout_runtime_mapping.py",
            "cpu_only": True,
            "model_loaded": False,
            "model_inference": False,
            "reward_scoring_performed": False,
            "raw_prompt_or_ground_truth_stored": False,
        },
        "rows": mapping_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "row_count": EXPECTED_ROW_COUNT,
        "unique_runtime_ground_truth_pairs": len(pair_keys),
        "official_verl_commit": mapping["official_verl_commit"],
        "chat_template_sha256": mapping["chat_template_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
