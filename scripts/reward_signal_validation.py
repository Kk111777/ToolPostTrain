#!/usr/bin/env python3
"""Validate ToolRL reward and offline GRPO/GDPO advantage signals.

This is deliberately not a trainer.  It reads 16 parquet prompts, builds the
same single-turn continuous-token prompts used by the current smoke path,
rolls out with the local vLLM model, calls the official v0.9.1 GDPO reward
manager, and computes the official outcome-advantage functions on CPU.  No
actor model, backward pass, or optimizer is created.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.gdpo import GDPORewardManager
from verl.trainer.ppo.core_algos import (
    compute_gdpo_outcome_advantage,
    compute_grpo_outcome_advantage,
)
from verl.utils.reward_score.rlla import compute_score
from verl.utils.tokenizer import hf_tokenizer
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder


FORMAT_PATTERNS = {
    "response_only": re.compile(r"^<think>.*?</think>\n<response>.*?</response>$", re.DOTALL),
    "tool_only": re.compile(r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$", re.DOTALL),
    "tool_and_response": re.compile(
        r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$",
        re.DOTALL,
    ),
    "think_only": re.compile(r"^<think>.*?</think>$", re.DOTALL),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=1024)
    return parser.parse_args()


def as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric reward: {value!r}")
    return result


def expected_kind(ground_truth: str) -> str:
    has_tool = "<tool_call>" in ground_truth
    has_response = "<response>" in ground_truth
    if has_tool and has_response:
        return "tool_and_response"
    if has_tool:
        return "tool_only"
    if has_response:
        return "response_only"
    return "think_only"


def scorer_text(solution: str) -> str:
    # This mirrors the Qwen branch in rlla.py.  Special tokens are normally
    # removed by the reward manager decode, but keeping the split here makes
    # the diagnostic independent of that detail.
    return solution.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()


def format_match(solution: str, ground_truth: str) -> bool:
    text = scorer_text(solution)
    kind = expected_kind(ground_truth)
    if text.count("<response>") != (1 if kind in {"response_only", "tool_and_response"} else 0):
        return False
    if text.count("</response>") != (1 if kind in {"response_only", "tool_and_response"} else 0):
        return False
    if text.count("<tool_call>") != (1 if kind in {"tool_only", "tool_and_response"} else 0):
        return False
    if text.count("</tool_call>") != (1 if kind in {"tool_only", "tool_and_response"} else 0):
        return False
    return bool(FORMAT_PATTERNS[kind].search(text))


def tool_json_parseable(solution: str) -> bool:
    text = scorer_text(solution)
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return False
    try:
        lines = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip().split("\n")
        return bool(lines) and all(json.loads(line) for line in lines)
    except Exception:
        return False


def make_single_item(
    prompt_ids: list[int],
    response_ids: list[int],
    row: dict[str, Any],
) -> DataProto:
    if not response_ids:
        raise ValueError("vLLM returned an empty response; cannot validate reward manager token path")
    attention_mask = torch.ones(
        (1, len(prompt_ids) + len(response_ids)), dtype=torch.long, device="cpu"
    )
    tensors = {
        "responses": torch.tensor([response_ids], dtype=torch.long, device="cpu"),
        "attention_mask": attention_mask,
    }
    non_tensors = {
        "data_source": np.array([row["data_source"]], dtype=object),
        "reward_model": np.array([row["reward_model"]], dtype=object),
        "extra_info": np.array([dict(row.get("extra_info") or {})], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


class RewardRunner:
    def __init__(self, tokenizer):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        config = OmegaConf.create(
            {
                "trainer": {"experiment_name": "qwen2p5_1p5b_reward_signal"},
            }
        )
        self.manager = GDPORewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)

    def score(self, prompt_ids: list[int], response_ids: list[int], row: dict[str, Any]) -> dict[str, Any]:
        data = make_single_item(prompt_ids, response_ids, row)
        return self.loop.run_until_complete(self.manager.run_single(data))

    def close(self) -> None:
        self.loop.close()


def finite_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("reward statistics contain no values or non-finite values")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def compute_advantage_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot compute advantage diagnostics for empty records")

    batch_size = len(records)
    max_prompt = max(int(record["prompt_length"]) for record in records)
    max_response = max(int(record["response_length"]) for record in records)
    response_mask = torch.zeros((batch_size, max_response), dtype=torch.float32)
    token_rewards = torch.zeros((batch_size, max_response), dtype=torch.float32)
    prompt_batch = torch.zeros((batch_size, max_prompt), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_prompt + max_response), dtype=torch.long)

    for index, record in enumerate(records):
        prompt_len = int(record["prompt_length"])
        response_len = int(record["response_length"])
        response_mask[index, :response_len] = 1.0
        token_rewards[index, response_len - 1] = float(record["total_reward"])
        prompt_batch[index, :prompt_len] = 1
        attention_mask[index, :prompt_len] = 1
        attention_mask[index, max_prompt : max_prompt + response_len] = 1

    group_index = np.asarray([record["sample_id"] for record in records], dtype=object)
    grpo_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=group_index,
        norm_adv_by_std_in_grpo=True,
    )

    gdpo_non_tensor_batch = {
        "accuracy_reward": np.asarray([record["accuracy_reward"] for record in records], dtype=np.float32),
        "format_reward": np.asarray([record["format_reward"] for record in records], dtype=np.float32),
    }
    gdpo_config = OmegaConf.create(
        {
            "gdpo_reward_keys": ["accuracy_reward", "format_reward"],
            "norm_adv_by_std_in_grpo": True,
        }
    )
    gdpo_advantages, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=group_index,
        config=gdpo_config,
        non_tensor_batch=gdpo_non_tensor_batch,
        batch={"prompts": prompt_batch, "attention_mask": attention_mask},
    )

    for index, record in enumerate(records):
        valid = response_mask[index].bool()
        record["grpo_advantage_mean"] = float(grpo_advantages[index][valid].mean().item())
        record["gdpo_advantage_mean"] = float(gdpo_advantages[index][valid].mean().item())

    def summarize(values: torch.Tensor) -> dict[str, Any]:
        finite = bool(torch.isfinite(values).all().item())
        return {
            "finite": finite,
            "nonzero_token_count": int((values.abs() > 1e-12).sum().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }

    return {
        "group_count": len(set(group_index.tolist())),
        "rollouts_per_group": len(records) // len(set(group_index.tolist())),
        "grpo": summarize(grpo_advantages),
        "gdpo": summarize(gdpo_advantages),
    }


def build_analysis(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    format_nonzero = sum(abs(record["format_reward"]) > 1e-12 for record in records)
    accuracy_nonzero = sum(abs(record["accuracy_reward"]) > 1e-12 for record in records)
    manager_ok = sum(record["reward_manager_result_ok"] for record in records)
    format_matches = sum(record["format_shape_match"] for record in records)
    tool_expected = sum(record["expected_kind"] in {"tool_only", "tool_and_response"} for record in records)
    tool_outputs = sum(record["has_tool_call"] for record in records)
    tool_parseable = sum(record["tool_json_parseable"] for record in records)

    if summary["nonzero_reward_count"] > 0:
        status = "normal"
        reason = "At least one rollout has non-zero reward; reward signal is observable."
    else:
        status = "all_zero"
        reasons = []
        if format_matches == 0:
            reasons.append("rollout outputs did not match the strict rlla.py format pattern")
        if tool_expected > 0 and tool_outputs == 0:
            reasons.append("tool-call ground truths were present, but rollouts emitted no <tool_call> block")
        elif tool_expected > 0 and tool_parseable == 0:
            reasons.append("tool-call outputs were absent or not JSON-line parseable")
        if manager_ok == total:
            reasons.append("reward manager and evaluator returned structured results for every rollout")
        if not reasons:
            reasons.append("no single zero-reward cause was isolated by this sample")
        reason = "; ".join(reasons)

    return {
        "reward_signal_status": status,
        "reason": reason,
        "reward_manager_invocations_ok": manager_ok,
        "format_shape_match_count": format_matches,
        "format_shape_match_ratio": format_matches / total,
        "ground_truth_tool_call_count": tool_expected,
        "rollout_tool_call_count": tool_outputs,
        "rollout_tool_json_parseable_count": tool_parseable,
        "rollout_contains_response_tag_count": sum(record["has_response"] for record in records),
        "rollout_contains_think_pair_count": sum(record["has_think_pair"] for record in records),
    }


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["metadata"]
    summary = report["summary"]
    analysis = report["analysis"]
    lines = [
        "# ProjectB reward signal validation",
        "",
        f"- Status: **{analysis['reward_signal_status']}**",
        f"- Dataset samples: {meta['dataset_sample_count']} (seed={meta['dataset_sample_seed']})",
        f"- Rollout trajectories: {summary['trajectory_count']} ({meta['rollout_n']} per prompt)",
        f"- Model: `{meta['model_path']}`",
        f"- vLLM sampler switch: `VLLM_USE_FLASHINFER_SAMPLER={meta['vllm_use_flashinfer_sampler']}`",
        "- Optimizer: **not entered**",
        "",
        "## Aggregate reward",
        "",
        f"- mean: `{summary['reward_mean']:.6f}`",
        f"- std: `{summary['reward_std']:.6f}`",
        f"- min/max: `{summary['reward_min']:.6f}` / `{summary['reward_max']:.6f}`",
        f"- non-zero: `{summary['nonzero_reward_count']}/{summary['trajectory_count']}` ({summary['nonzero_reward_ratio']:.2%})",
        f"- format non-zero: `{summary['format_reward_nonzero_count']}`",
        f"- accuracy non-zero: `{summary['accuracy_reward_nonzero_count']}`",
        "",
        "## Advantage diagnostic",
        "",
        f"- GRPO finite: `{summary['advantage']['grpo']['finite']}`, non-zero tokens: `{summary['advantage']['grpo']['nonzero_token_count']}`",
        f"- GDPO finite: `{summary['advantage']['gdpo']['finite']}`, non-zero tokens: `{summary['advantage']['gdpo']['nonzero_token_count']}`",
        "- These are official v0.9.1 CPU outcome-advantage functions only; no policy update was run.",
        "",
        "## Zero-reward diagnosis",
        "",
        f"- {analysis['reason']}",
        f"- Reward-manager structured results: `{analysis['reward_manager_invocations_ok']}/{summary['trajectory_count']}`",
        f"- Strict format matches: `{analysis['format_shape_match_count']}/{summary['trajectory_count']}`",
        f"- Ground-truth tool-call trajectories: `{analysis['ground_truth_tool_call_count']}`",
        f"- Rollout tool-call blocks: `{analysis['rollout_tool_call_count']}`",
        f"- JSON-parseable tool-call blocks: `{analysis['rollout_tool_json_parseable_count']}`",
        "",
        "## Per-trajectory records",
        "",
        "| sample id | rollout | prompt | response | format | accuracy | total | non-zero | GRPO adv | GDPO adv |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in report["records"]:
        lines.append(
            "| {sample_id} | {rollout_index} | {prompt_length} | {response_length} | {format_reward:.3f} | "
            "{accuracy_reward:.3f} | {total_reward:.3f} | {nonzero_reward} | {grpo_advantage_mean:.6f} | "
            "{gdpo_advantage_mean:.6f} |".format(**record)
        )
    lines.extend(
        [
            "",
            "## Output-shape evidence",
            "",
            "| sample id | rollout | expected | think pair | tool block | response tag | tool JSON | response preview |",
            "|---:|---:|:---|:---:|:---:|:---:|:---:|:---|",
        ]
    )
    for record in report["records"]:
        preview = record["response_text"].replace("\n", " ").replace("|", "\\|")[:240]
        lines.append(
            f"| {record['sample_id']} | {record['rollout_index']} | {record['expected_kind']} | "
            f"{record['has_think_pair']} | {record['has_tool_call']} | {record['has_response']} | "
            f"{record['tool_json_parseable']} | {preview} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.num_samples != 16:
        raise SystemExit("this validation is intentionally fixed to 16 dataset samples")
    if args.rollout_n != 2:
        raise SystemExit("this validation preserves the current smoke rollout.n=2")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"

    data_path = Path(args.data)
    model_path = Path(args.model)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not data_path.is_file():
        raise SystemExit(f"missing parquet: {data_path}")
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"missing local model snapshot: {model_path}")

    table = pq.read_table(data_path)
    rows = table.to_pylist()
    if len(rows) < args.num_samples:
        raise SystemExit(f"dataset has {len(rows)} rows, need {args.num_samples}")
    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(len(rows)), args.num_samples)
    selected_rows = [rows[index] for index in selected_indices]

    tokenizer = hf_tokenizer(str(model_path), trust_remote_code=True, local_files_only=True)
    model_config = tokenizer.init_kwargs.get("_commit_hash") if hasattr(tokenizer, "init_kwargs") else None
    del model_config
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    builder = create_continuous_token_builder(tokenizer, hf_model_type=hf_config.model_type)

    prompt_rows: list[tuple[int, dict[str, Any], list[int]]] = []
    for source_row, row in zip(selected_indices, selected_rows, strict=True):
        messages = [dict(message) for message in row["prompt"]]
        prompt_ids = [int(token_id) for token_id in builder.build_initial_tokens(messages)]
        if len(prompt_ids) > args.max_prompt_length:
            raise RuntimeError(
                f"selected row {source_row} prompt length {len(prompt_ids)} exceeds current limit "
                f"{args.max_prompt_length}; no truncation was applied"
            )
        prompt_rows.append((source_row, row, prompt_ids))

    vllm_inputs = [{"prompt_token_ids": prompt_ids} for _, _, prompt_ids in prompt_rows]
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_prompt_length + args.max_response_length,
        gpu_memory_utilization=0.30,
        enforce_eager=True,
        max_num_batched_tokens=6144,
        max_num_seqs=4,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_response_length,
        n=args.rollout_n,
    )

    started = time.monotonic()
    outputs = llm.generate(vllm_inputs, sampling_params, use_tqdm=True)
    generation_seconds = time.monotonic() - started
    if len(outputs) != len(prompt_rows):
        raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(prompt_rows)} prompts")

    reward_runner = RewardRunner(tokenizer)
    records: list[dict[str, Any]] = []
    for (source_row, row, prompt_ids), request_output in zip(prompt_rows, outputs, strict=True):
        if len(request_output.outputs) != args.rollout_n:
            raise RuntimeError(
                f"row {source_row}: vLLM returned {len(request_output.outputs)} completions, "
                f"expected {args.rollout_n}"
            )
        sample_id = row.get("extra_info", {}).get("index", source_row)
        ground_truth = row["reward_model"]["ground_truth"]
        for rollout_index, completion in enumerate(request_output.outputs):
            response_ids = [int(token_id) for token_id in completion.token_ids]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            manager_result = reward_runner.score(prompt_ids, response_ids, row)
            if not isinstance(manager_result, dict):
                raise RuntimeError(f"row {source_row}: reward manager returned {type(manager_result)}")
            extra = manager_result.get("reward_extra_info", {})
            required = {"score", "format_reward", "accuracy_reward"}
            if not required.issubset(extra):
                raise RuntimeError(
                    f"row {source_row}: reward manager keys {sorted(extra)} do not contain {sorted(required)}"
                )
            total_reward = as_float(manager_result["reward_score"])
            format_reward = as_float(extra["format_reward"])
            accuracy_reward = as_float(extra["accuracy_reward"])
            record = {
                "source_row": int(source_row),
                "sample_id": int(sample_id) if isinstance(sample_id, (int, np.integer)) else str(sample_id),
                "rollout_index": int(rollout_index),
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "format_reward": format_reward,
                "accuracy_reward": accuracy_reward,
                "total_reward": total_reward,
                "nonzero_reward": bool(abs(total_reward) > 1e-12),
                "reward_manager_result_ok": True,
                "expected_kind": expected_kind(ground_truth),
                "format_shape_match": bool(format_match(response_text, ground_truth)),
                "has_think_pair": "<think>" in response_text and "</think>" in response_text,
                "has_tool_call": "<tool_call>" in response_text and "</tool_call>" in response_text,
                "has_response": "<response>" in response_text and "</response>" in response_text,
                "tool_json_parseable": bool(tool_json_parseable(response_text)),
                "response_text": response_text,
            }
            records.append(record)
    reward_runner.close()

    rewards = [record["total_reward"] for record in records]
    format_rewards = [record["format_reward"] for record in records]
    accuracy_rewards = [record["accuracy_reward"] for record in records]
    summary = {
        "trajectory_count": len(records),
        "prompt_count": args.num_samples,
        "reward_mean": finite_stats(rewards)["mean"],
        "reward_std": finite_stats(rewards)["std"],
        "reward_min": finite_stats(rewards)["min"],
        "reward_max": finite_stats(rewards)["max"],
        "format_reward_nonzero_count": int(sum(abs(value) > 1e-12 for value in format_rewards)),
        "accuracy_reward_nonzero_count": int(sum(abs(value) > 1e-12 for value in accuracy_rewards)),
        "nonzero_reward_count": int(sum(record["nonzero_reward"] for record in records)),
        "nonzero_reward_ratio": float(sum(record["nonzero_reward"] for record in records) / len(records)),
        "all_rewards_finite": bool(np.isfinite(np.asarray(rewards, dtype=np.float64)).all()),
        "generation_seconds": float(generation_seconds),
    }
    summary["advantage"] = compute_advantage_diagnostics(records)
    analysis = build_analysis(records, summary)

    report = {
        "metadata": {
            "dataset_path": str(data_path),
            "dataset_sample_count": args.num_samples,
            "dataset_sample_seed": args.seed,
            "selected_source_rows": selected_indices,
            "model_path": str(model_path),
            "rollout_backend": "vllm",
            "tensor_parallel_size": 1,
            "rollout_n": args.rollout_n,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_model_len": args.max_prompt_length + args.max_response_length,
            "gpu_memory_utilization": 0.30,
            "enforce_eager": True,
            "vllm_use_flashinfer_sampler": 0,
            "optimizer_entered": False,
            "package_changes": False,
            "model_downloaded": False,
        },
        "summary": summary,
        "analysis": analysis,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"reward_signal_status={analysis['reward_signal_status']}")
    print(f"reward_mean={summary['reward_mean']:.6f}")
    print(f"reward_std={summary['reward_std']:.6f}")
    print(f"nonzero_reward={summary['nonzero_reward_count']}/{summary['trajectory_count']}")
    print(f"format_nonzero={summary['format_reward_nonzero_count']}")
    print(f"accuracy_nonzero={summary['accuracy_reward_nonzero_count']}")
    print(f"grpo_advantage_nonzero_tokens={summary['advantage']['grpo']['nonzero_token_count']}")
    print(f"gdpo_advantage_nonzero_tokens={summary['advantage']['gdpo']['nonzero_token_count']}")
    print(f"report_json={output_json}")
    print(f"report_md={output_md}")
    print("REWARD_SIGNAL_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
