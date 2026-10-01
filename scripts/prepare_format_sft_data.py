#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

DEFAULT_TRAIN = "/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet"
DEFAULT_MODEL = "/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_OUTPUT = "/root/autodl-tmp/ProjectB/repo/dataset/format_sft_800.jsonl"
DEFAULT_EXCLUDED_ROWS = "2619,456,102,3037,1126,1003,914,571,3016,419,2771,3033,3654,2233,356,2418"
STRICT_TOOL = re.compile(r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$", re.DOTALL)
STRICT_RESPONSE = re.compile(r"^<think>.*?</think>\n<response>.*?</response>$", re.DOTALL)
QUOTAS = {"response_only": 100, "single_tool": 400, "multi_tool_2": 200, "multi_tool_3plus": 100}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic format-only chat SFT subset."
    )
    parser.add_argument("--train-parquet", default=DEFAULT_TRAIN)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument(
        "--exclude-source-rows",
        default=DEFAULT_EXCLUDED_ROWS,
        help="Comma-separated source-row ids reserved for validation.",
    )
    return parser.parse_args()


def input_ids(encoded: Any) -> list[int]:
    try:
        values = encoded["input_ids"]
    except (KeyError, TypeError, IndexError):
        values = encoded
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def normalized_prompt(raw_prompt: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw_prompt, list) or len(raw_prompt) != 2:
        return None
    messages: list[dict[str, str]] = []
    for message in raw_prompt:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
    if [message["role"] for message in messages] != ["system", "user"]:
        return None
    return messages


def classify_target(target: Any) -> tuple[str, int] | None:
    if not isinstance(target, str) or not target:
        return None
    if chr(96) * 3 in target:
        return None
    if target.count("<think>") != 1 or target.count("</think>") != 1:
        return None

    has_tool = target.count("<tool_call>") == 1 and target.count("</tool_call>") == 1
    has_response = (
        target.count("<response>") == 1
        and target.count("</response>") == 1
    )

    if has_tool and not has_response:
        if STRICT_TOOL.fullmatch(target) is None:
            return None
        body = target.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        lines = [line for line in body.splitlines() if line.strip()]
        if not lines:
            return None
        for line in lines:
            try:
                tool = json.loads(line)
            except json.JSONDecodeError:
                return None
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("name"), str)
                or not isinstance(tool.get("parameters"), dict)
            ):
                return None
        if len(lines) == 1:
            return "single_tool", 1
        if len(lines) == 2:
            return "multi_tool_2", 2
        return "multi_tool_3plus", len(lines)

    if has_response and not has_tool:
        if STRICT_RESPONSE.fullmatch(target) is None:
            return None
        return "response_only", 0

    return None


def parse_excluded(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_parquet).resolve()
    if train_path.name != "train.parquet":
        raise ValueError(f"Expected a train.parquet input, got {train_path}")
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    excluded = parse_excluded(args.exclude_source_rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    rows = pq.read_table(train_path).to_pylist()
    buckets: dict[str, list[dict[str, Any]]] = {
        name: [] for name in QUOTAS
    }
    rejected = Counter()
    excluded_count = 0

    for fallback_row, row in enumerate(rows):
        extra_info = row.get("extra_info") or {}
        source_row = int(extra_info.get("index", fallback_row))
        if source_row in excluded:
            excluded_count += 1
            continue

        messages = normalized_prompt(row.get("prompt"))
        if messages is None:
            rejected["prompt_schema"] += 1
            continue
        target = (row.get("reward_model") or {}).get("ground_truth")
        classified = classify_target(target)
        if classified is None:
            rejected["target_contract"] += 1
            continue
        stratum, tool_count = classified

        full_messages = messages + [
            {"role": "assistant", "content": target}
        ]
        rendered = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        sequence_length = len(input_ids(rendered))
        if sequence_length > args.max_seq_length:
            rejected["too_long"] += 1
            continue

        buckets[stratum].append(
            {
                "id": f"rlla-train-{source_row}",
                "source_row": source_row,
                "messages": full_messages,
                "metadata": {
                    "source_file": str(train_path),
                    "source_split": "train",
                    "data_source": row.get("data_source"),
                    "ability": row.get("ability"),
                    "target_kind": (
                        "response_only"
                        if stratum == "response_only"
                        else "tool_only"
                    ),
                    "sampling_stratum": stratum,
                    "tool_count": tool_count,
                    "sequence_length": sequence_length,
                },
            }
        )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for stratum, quota in QUOTAS.items():
        available = buckets[stratum]
        if len(available) < quota:
            raise RuntimeError(
                f"Not enough legal rows for {stratum}: "
                f"need {quota}, found {len(available)}"
            )
        selected.extend(rng.sample(available, quota))
    rng.shuffle(selected)

    if len(selected) != 800:
        raise AssertionError(f"Expected 800 records, got {len(selected)}")
    if len({item["source_row"] for item in selected}) != 800:
        raise AssertionError("Duplicate source rows in selected dataset")
    if any(item["source_row"] in excluded for item in selected):
        raise AssertionError("A reserved validation row was selected")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"source_rows={len(rows)}")
    print(f"excluded_validation_rows={excluded_count}")
    print(f"rejected={dict(rejected)}")
    print(
        "eligible_by_stratum="
        f"{ {key: len(value) for key, value in buckets.items()} }"
    )
    print(
        "selected_by_stratum="
        f"{dict(Counter(item['metadata']['sampling_stratum'] for item in selected))}"
    )
    print(f"selected={len(selected)}")
    print(f"output={output_path}")
    print("FORMAT_SFT_DATA_PREP_PASS")


if __name__ == "__main__":
    main()
