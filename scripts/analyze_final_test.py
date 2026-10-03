#!/usr/bin/env python3
"""CPU-only analysis for an explicitly authorized ProjectB final-test run."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

EXCLUDED_SOURCE_ID = 3814


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("projectb_final_test_rlla", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scorer_text(output: str) -> str:
    if "<|im_start|>assistant" in output:
        output = output.split("<|im_start|>assistant", 1)[-1]
        output = output.split("<|im_end|>", 1)[0].strip()
    return output


def strict_format(output: str, ground_truth: str) -> bool:
    output = scorer_text(output)
    if "<response>" in ground_truth and "<tool_call>" not in ground_truth:
        return bool(
            re.search(r"^<think>.*?</think>\n<response>.*?</response>$", output, re.DOTALL)
            and output.count("<response>") == output.count("</response>") == 1
        )
    if "<tool_call>" in ground_truth and "<response>" not in ground_truth:
        return bool(
            re.search(r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$", output, re.DOTALL)
            and output.count("<tool_call>") == output.count("</tool_call>") == 1
        )
    if "<tool_call>" in ground_truth and "<response>" in ground_truth:
        return bool(
            re.search(
                r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n"
                r"<response>.*?</response>$",
                output,
                re.DOTALL,
            )
            and output.count("<tool_call>") == output.count("</tool_call>") == 1
            and output.count("<response>") == output.count("</response>") == 1
        )
    return False


def tool_parse(output: str) -> bool:
    """JSON-line tool-call parse success over every output row."""
    output = scorer_text(output)
    try:
        block = output.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        return bool(lines) and all(isinstance(json.loads(line), dict) for line in lines)
    except Exception:
        return False


def response_wrapper(output: str) -> bool:
    output = scorer_text(output)
    return output.count("<response>") == output.count("</response>") == 1


def target_tool_parse(output: str, ground_truth: str) -> bool | None:
    if "<tool_call>" not in ground_truth:
        return None
    return tool_parse(output)


def response_only_wrapper(output: str, ground_truth: str) -> bool | None:
    if "<response>" not in ground_truth or "<tool_call>" in ground_truth:
        return None
    return response_wrapper(output)


def bool_summary(values: list[bool], drop_none: bool = False) -> dict[str, Any]:
    if drop_none:
        values = [value for value in values if value is not None]
    return {
        "count": int(sum(bool(x) for x in values)),
        "denominator": len(values),
        "rate": float(sum(bool(x) for x in values) / len(values)) if values else None,
    }


def mean_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else None,
        "finite": bool(np.isfinite(arr).all()) if len(arr) else False,
    }


def resolve_output(path: Path, step: int) -> Path:
    if path.is_dir():
        for candidate in (path / "eval_generations" / f"{step}.jsonl", path / f"{step}.jsonl"):
            if candidate.is_file():
                return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def output_records(
    output_path: Path,
    manifest_rows: list[dict[str, Any]],
    scorer: Any | None,
    experiment_name: str,
) -> list[dict[str, Any]]:
    by_key = {
        (row["prompt_sha256_exact"], row["ground_truth_sha256_exact"]): row
        for row in manifest_rows
    }
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for position, line in enumerate(output_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        prompt_hash = sha256_text(str(item.get("input", "")))
        gt_text = str(item.get("gts", ""))
        gt_hash = sha256_text(gt_text)
        key = (prompt_hash, gt_hash)
        row = by_key.get(key)
        if row is None:
            raise ValueError(f"output row {position} does not match the frozen manifest")
        if key in seen:
            raise ValueError(f"duplicate output row at position {position}")
        seen.add(key)
        output = str(item.get("output", ""))
        if scorer is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                scored = scorer.compute_score(
                    row.get("data_source", "rlla"),
                    output,
                    gt_text,
                    {"experiment_name": experiment_name},
                    step=0,
                )
            score = float(scored["score"])
            accuracy = float(scored["accuracy_reward"])
            fmt = float(scored["format_reward"])
        else:
            score = float(item["score"])
            accuracy = float(item["accuracy_reward"])
            fmt = float(item["format_reward"])
        records.append(
            {
                "row_position": int(row["row_position"]),
                "source_id": row["source_id"],
                "input_sha256": prompt_hash,
                "ground_truth_sha256": gt_hash,
                "output_sha256": sha256_text(output),
                "score": score,
                "accuracy_reward": accuracy,
                "format_reward": fmt,
                "strict_format": strict_format(output, gt_text),
                "tool_parse": target_tool_parse(output, gt_text),
                "response_wrapper": response_only_wrapper(output, gt_text),
                "tool_parse_auxiliary": tool_parse(output),
                "response_wrapper_auxiliary": response_wrapper(output),
            }
        )
    if len(records) != len(manifest_rows):
        raise ValueError(f"coverage mismatch: {len(records)} != {len(manifest_rows)}")
    return sorted(records, key=lambda row: row["row_position"])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "raw_total_validation_reward": mean_summary([row["score"] for row in records]),
        "accuracy_reward": mean_summary([row["accuracy_reward"] for row in records]),
        "format_reward": mean_summary([row["format_reward"] for row in records]),
        "strict_format": bool_summary([row["strict_format"] for row in records]),
        "tool_parse": bool_summary([row["tool_parse"] for row in records], drop_none=True),
        "response_wrapper": bool_summary([row["response_wrapper"] for row in records], drop_none=True),
        "tool_parse_auxiliary": bool_summary([row["tool_parse_auxiliary"] for row in records]),
        "response_wrapper_auxiliary": bool_summary([row["response_wrapper_auxiliary"] for row in records]),
    }


def bootstrap(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("paired bootstrap requires non-empty equal-length arrays")
    indices = rng.integers(0, len(left), size=(10000, len(left)))
    differences = right[indices].mean(axis=1) - left[indices].mean(axis=1)
    return {
        "n": int(len(left)),
        "mean_difference": float(right.mean() - left.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_median": float(np.quantile(differences, 0.5)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "approx_two_sided_tail": float(
            min(1.0, 2.0 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
        ),
    }


def parse_model_output(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model output must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model output must be NAME=PATH")
    return name, Path(path)


def self_test() -> None:
    assert tool_parse("<think>x</think>\n<tool_call>\n{}\n</tool_call>")
    assert not tool_parse("<think>x</think>\n<response>y</response>")
    assert response_wrapper("<think>x</think>\n<response>y</response>")
    assert strict_format(
        "<think>x</think>\n<response>y</response>",
        "<think>x</think>\n<response>z</response>",
    )
    rng = np.random.default_rng(42)
    assert bootstrap(np.array([1.0, 2.0]), np.array([2.0, 4.0]), rng)["n"] == 2
    print("ANALYSIS_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--experiment-name", default="qwen2p5_1p5b_grpo_one_step")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--model-output", action="append", type=parse_model_output, default=[])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.manifest is None or not args.model_output or args.out_json is None:
        parser.error("manifest, at least one --model-output, and --out-json are required")
    manifest = load_json(args.manifest)
    manifest_rows = manifest["rows"]
    scorer = load_scorer(args.scorer) if args.scorer else None
    runs: dict[str, Any] = {}
    for name, root in args.model_output:
        path = resolve_output(root, args.step)
        records = output_records(path, manifest_rows, scorer, args.experiment_name)
        runs[name] = {
            "output_path": str(path),
            "primary_80": summarize(records),
            "sensitivity_79": summarize(
                [row for row in records if int(row["source_id"]) != EXCLUDED_SOURCE_ID]
            ),
            "records": records,
        }
    names = list(runs)
    rng = np.random.default_rng(42)
    paired: dict[str, Any] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            key = f"{left_name}_vs_{right_name}"
            paired[key] = {}
            for scope, predicate in [
                ("primary_80", lambda row: True),
                ("sensitivity_79", lambda row: int(row["source_id"]) != EXCLUDED_SOURCE_ID),
            ]:
                left = [row for row in runs[left_name]["records"] if predicate(row)]
                right = [row for row in runs[right_name]["records"] if predicate(row)]
                paired[key][scope] = bootstrap(
                    np.asarray([row["score"] for row in left], dtype=np.float64),
                    np.asarray([row["score"] for row in right], dtype=np.float64),
                    rng,
                )
    result = {
        "schema": "projectb_final_test_analysis_v1",
        "source": "persisted final-test JSONL only; no model loading or inference",
        "step": args.step,
        "manifest": str(args.manifest),
        "excluded_source_id": EXCLUDED_SOURCE_ID,
        "bootstrap": {"replicates": 10000, "seed": 42},
        "runs": runs,
        "paired_bootstrap_right_minus_left": paired,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.out_md:
        lines = [
            "# ProjectB final-test analysis",
            "",
            "CPU-only analysis of persisted outputs; no model loading or inference.",
            "",
            f"- step: {args.step}",
            f"- models: {', '.join(names)}",
            "- primary: frozen 80-row test manifest",
            "- sensitivity: same rows excluding source id 3814",
            "- bootstrap: 10,000 paired resamples, seed 42",
        ]
        for name in names:
            metrics = runs[name]["primary_80"]
            lines.extend(
                [
                    "",
                    f"## {name}",
                    "",
                    f"- raw total reward mean: {metrics['raw_total_validation_reward']['mean']}",
                    f"- accuracy reward mean: {metrics['accuracy_reward']['mean']}",
                    f"- format reward mean: {metrics['format_reward']['mean']}",
                    f"- strict format: {metrics['strict_format']['count']}/{metrics['strict_format']['denominator']}",
                    f"- tool-call JSON parse on target rows: {metrics['tool_parse']['count']}/{metrics['tool_parse']['denominator']}",
                    f"- response-only wrapper on response-only targets: {metrics['response_wrapper']['count']}/{metrics['response_wrapper']['denominator']}",
                    f"- auxiliary tool-parse count over all rows: {metrics['tool_parse_auxiliary']['count']}/{metrics['tool_parse_auxiliary']['denominator']}",
                    f"- auxiliary response-wrapper count over all rows: {metrics['response_wrapper_auxiliary']['count']}/{metrics['response_wrapper_auxiliary']['denominator']}",
                ]
            )
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
