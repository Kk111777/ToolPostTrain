#!/usr/bin/env python3
"""CPU-only analysis for the frozen ProjectB final endpoint.

Formal mode requires the exact four-model matrix and verifies both frozen
endpoint identities before reading persisted JSONL. It never loads a model,
starts vLLM, or falls back to positional/fuzzy row matching.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np


TIE_TOLERANCE = 1e-8
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
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
EXPECTED_MODEL_NAMES = frozenset(
    {"RL_INIT_V1", "GRPO-original35", "GDPO-current35", "GRPO-noKL35"}
)
COMPARISONS = (
    ("GRPO-original35", "GRPO-noKL35", "GRPO-original35_vs_GRPO-noKL35", "noKL-original"),
    ("GRPO-original35", "GDPO-current35", "GRPO-original35_vs_GDPO-current35", "GDPO-original"),
    ("GRPO-noKL35", "GDPO-current35", "GRPO-noKL35_vs_GDPO-current35", "GDPO-noKL"),
    ("RL_INIT_V1", "GRPO-original35", "RL_INIT_V1_vs_GRPO-original35", "original-RL_INIT_V1"),
    ("RL_INIT_V1", "GDPO-current35", "RL_INIT_V1_vs_GDPO-current35", "GDPO-RL_INIT_V1"),
    ("RL_INIT_V1", "GRPO-noKL35", "RL_INIT_V1_vs_GRPO-noKL35", "noKL-RL_INIT_V1"),
)
METRICS = ("score", "accuracy_reward", "format_reward")


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
    return tool_parse(output) if "<tool_call>" in ground_truth else None


def response_only_wrapper(output: str, ground_truth: str) -> bool | None:
    return response_wrapper(output) if "<response>" in ground_truth and "<tool_call>" not in ground_truth else None


def bool_summary(values: list[bool | None], drop_none: bool = False) -> dict[str, Any]:
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


def verify_frozen_identity(
    endpoint_manifest_path: Path,
    runtime_mapping_path: Path,
    endpoint_parquet_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    endpoint_sha = sha256_file(endpoint_manifest_path)
    if endpoint_sha != EXPECTED_ENDPOINT_MANIFEST_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_ENDPOINT_MANIFEST_IDENTITY")
    runtime_sha = sha256_file(runtime_mapping_path)
    if runtime_sha != EXPECTED_RUNTIME_MAPPING_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_IDENTITY")
    endpoint = load_json(endpoint_manifest_path)
    mapping = load_json(runtime_mapping_path)
    if endpoint.get("manifest_version") != "FINAL_HOLDOUT_V1" or endpoint.get("row_count") != 108:
        raise ValueError("FINAL_ANALYSIS_REFUSED_ENDPOINT_MANIFEST_CONTENT")
    if endpoint_parquet_path.is_file() and sha256_file(endpoint_parquet_path) != EXPECTED_PARQUET_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_ENDPOINT_PARQUET_IDENTITY")
    if endpoint.get("derived_parquet_sha256") != EXPECTED_PARQUET_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_ENDPOINT_PARQUET_DECLARATION")
    if endpoint.get("ordered_source_ids_sha256") != EXPECTED_SOURCE_IDS_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_ENDPOINT_SOURCE_ORDER")
    rows = endpoint.get("rows", [])
    mapping_rows = mapping.get("rows", [])
    if len(rows) != 108 or len(mapping_rows) != 108:
        raise ValueError("FINAL_ANALYSIS_REFUSED_ENDPOINT_ROW_COUNT")
    if mapping.get("parent_endpoint_manifest_sha256") != endpoint_sha:
        raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_PARENT")
    if mapping.get("endpoint_parquet_sha256") != EXPECTED_PARQUET_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_PARQUET")
    if mapping.get("ordered_source_ids_sha256") != EXPECTED_SOURCE_IDS_SHA256:
        raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_SOURCE_ORDER")
    if [int(row["source_id"]) for row in mapping_rows] != [int(row["source_id"]) for row in rows]:
        raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_ROW_ORDER")
    endpoint_by_source = {int(row["source_id"]): row for row in rows}
    pairs: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for row in mapping_rows:
        source_id = int(row["source_id"])
        endpoint_row = endpoint_by_source.get(source_id)
        if endpoint_row is None:
            raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_SOURCE_ID")
        if int(row["row_position"]) != int(endpoint_row["row_position"]):
            raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_ROW_POSITION")
        pair = (row["runtime_input_sha256"], row["ground_truth_sha256_exact"])
        if pair in pairs:
            raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_DUPLICATE")
        pairs.add(pair)
        if row["ground_truth_sha256_exact"] != endpoint_row["ground_truth_sha256_exact"]:
            raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_GROUND_TRUTH")
        if row.get("prompt_sha256_exact") != endpoint_row.get("prompt_sha256_exact"):
            raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_ENDPOINT_ROW")
        merged.append({**row, "data_source": endpoint_row.get("data_source", "rlla")})
    if len(pairs) != 108:
        raise ValueError("FINAL_ANALYSIS_REFUSED_RUNTIME_MAPPING_COVERAGE")
    identity = {
        "endpoint_manifest_sha256": endpoint_sha,
        "runtime_mapping_sha256": runtime_sha,
        "endpoint_parquet_sha256": EXPECTED_PARQUET_SHA256,
        "ordered_source_ids_sha256": EXPECTED_SOURCE_IDS_SHA256,
        "runtime_pair_count": len(pairs),
    }
    return endpoint, merged, identity


def output_records(
    output_path: Path,
    mapping_rows: list[dict[str, Any]],
    scorer: Any | None,
    experiment_name: str,
    runtime_mapping: bool,
) -> list[dict[str, Any]]:
    if runtime_mapping:
        by_key = {
            (row["runtime_input_sha256"], row["ground_truth_sha256_exact"]): row for row in mapping_rows
        }
    else:
        by_key = {
            (row["prompt_sha256_exact"], row["ground_truth_sha256_exact"]): row for row in mapping_rows
        }
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for position, line in enumerate(output_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        input_hash = sha256_text(str(item.get("input", "")))
        gt_text = str(item.get("gts", ""))
        gt_hash = sha256_text(gt_text)
        key = (input_hash, gt_hash)
        row = by_key.get(key)
        if row is None:
            raise ValueError(f"FINAL_ANALYSIS_REFUSED_UNMAPPED_OUTPUT_ROW:{position}")
        if key in seen:
            raise ValueError(f"FINAL_ANALYSIS_REFUSED_DUPLICATE_OUTPUT_ROW:{position}")
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
                "source_id": int(row["source_id"]),
                "target_category": row.get("target_category", "unknown"),
                "input_sha256": input_hash,
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
    if len(records) != len(mapping_rows) or len(seen) != len(mapping_rows):
        raise ValueError(f"FINAL_ANALYSIS_REFUSED_INCOMPLETE_OUTPUT_COVERAGE:{len(records)}")
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


def bootstrap(left: np.ndarray, right: np.ndarray, indices: np.ndarray) -> dict[str, float | int]:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("paired bootstrap requires non-empty equal-length arrays")
    if indices.shape != (BOOTSTRAP_REPLICATES, len(left)):
        raise ValueError("bootstrap index shape does not match paired rows")
    differences = right[indices].mean(axis=1) - left[indices].mean(axis=1)
    return {
        "n": int(len(left)),
        "mean_difference": float(right.mean() - left.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_median": float(np.quantile(differences, 0.5)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "approx_two_sided_tail": float(min(1.0, 2.0 * min(np.mean(differences <= 0), np.mean(differences >= 0)))),
    }


def make_bootstrap_indices(n: int) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n), dtype=np.int64)
    return indices, sha256_bytes(indices.tobytes(order="C"))


def paired_effect(left: list[dict[str, Any]], right: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    left_by_id = {int(row["source_id"]): row for row in left}
    right_by_id = {int(row["source_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired comparison coverage mismatch")
    row_ids = sorted(left_by_id)
    if indices.shape != (BOOTSTRAP_REPLICATES, len(row_ids)):
        raise ValueError("paired index matrix does not match comparison scope")
    result: dict[str, Any] = {"n": len(row_ids), "tie_tolerance": TIE_TOLERANCE, "metrics": {}}
    for metric in METRICS:
        left_values = np.asarray([float(left_by_id[sid][metric]) for sid in row_ids], dtype=np.float64)
        right_values = np.asarray([float(right_by_id[sid][metric]) for sid in row_ids], dtype=np.float64)
        differences = right_values - left_values
        result["metrics"][metric] = {
            "bootstrap": bootstrap(left_values, right_values, indices),
            "improved": int(np.sum(differences > TIE_TOLERANCE)),
            "declined": int(np.sum(differences < -TIE_TOLERANCE)),
            "tied": int(np.sum(np.abs(differences) <= TIE_TOLERANCE)),
            "mean_difference": float(differences.mean()),
        }
    return result


def descriptive_effect(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {int(row["source_id"]): row for row in left}
    right_by_id = {int(row["source_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("descriptive comparison coverage mismatch")
    result: dict[str, Any] = {"descriptive_only": True, "n": len(left_by_id), "metrics": {}}
    row_ids = sorted(left_by_id)
    for metric in METRICS:
        differences = np.asarray(
            [float(right_by_id[sid][metric]) - float(left_by_id[sid][metric]) for sid in row_ids],
            dtype=np.float64,
        )
        result["metrics"][metric] = {
            "mean_difference": float(differences.mean()) if len(differences) else None,
            "improved": int(np.sum(differences > TIE_TOLERANCE)),
            "declined": int(np.sum(differences < -TIE_TOLERANCE)),
            "tied": int(np.sum(np.abs(differences) <= TIE_TOLERANCE)),
        }
    return result


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
    assert strict_format("<think>x</think>\n<response>y</response>", "<response>z</response>")
    indices, digest = make_bootstrap_indices(2)
    assert indices.shape == (BOOTSTRAP_REPLICATES, 2) and len(digest) == 64
    effect = paired_effect(
        [{"source_id": 1, "score": 1.0, "accuracy_reward": 1.0, "format_reward": 0.0}],
        [{"source_id": 1, "score": 2.0, "accuracy_reward": 1.0, "format_reward": 1.0}],
        np.zeros((BOOTSTRAP_REPLICATES, 1), dtype=np.int64),
    )
    assert effect["n"] == 1
    print("ANALYSIS_SELF_TEST_PASS")


def scope_records(records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [row for row in records if predicate(row)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", "--endpoint-manifest", dest="endpoint_manifest", type=Path)
    parser.add_argument("--runtime-mapping", type=Path)
    parser.add_argument("--endpoint-parquet", type=Path)
    parser.add_argument("--fixture-mode", action="store_true", help="explicitly permit non-formal fixtures")
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--experiment-name", default="qwen2p5_1p5b_grpo_one_step")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--model-output", action="append", type=parse_model_output, default=[])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--exclude-source-id", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model_output or args.out_json is None:
        parser.error("at least one --model-output and --out-json are required")
    model_names = [name for name, _ in args.model_output]
    if len(model_names) != len(set(model_names)):
        raise ValueError("FINAL_ANALYSIS_REFUSED_INCOMPLETE_MATRIX: duplicate model name")
    root = Path("/root/autodl-tmp/ProjectB")
    endpoint_manifest_path = args.endpoint_manifest or root / "repo/manifests/final_holdout_v1_manifest.json"
    runtime_mapping_path = args.runtime_mapping or root / "repo/manifests/final_holdout_v1_runtime_mapping.json"
    endpoint_parquet_path = args.endpoint_parquet or root / "env-modern/final_holdout_v1.parquet"
    if args.fixture_mode:
        endpoint = load_json(endpoint_manifest_path)
        mapping_rows = endpoint["rows"]
        identity = {"fixture_mode": True}
        runtime_mapping = False
        expected_n = int(endpoint.get("row_count", len(mapping_rows)))
    else:
        if set(model_names) != EXPECTED_MODEL_NAMES:
            missing = sorted(EXPECTED_MODEL_NAMES - set(model_names))
            extra = sorted(set(model_names) - EXPECTED_MODEL_NAMES)
            raise ValueError(
                f"FINAL_ANALYSIS_REFUSED_INCOMPLETE_MATRIX: missing={missing}; extra={extra}"
            )
        endpoint, mapping_rows, identity = verify_frozen_identity(
            endpoint_manifest_path, runtime_mapping_path, endpoint_parquet_path
        )
        runtime_mapping = True
        expected_n = 108
    if expected_n != len(mapping_rows):
        raise ValueError("FINAL_ANALYSIS_REFUSED_MANIFEST_ROW_COUNT")
    scorer = load_scorer(args.scorer) if args.scorer else None
    runs: dict[str, Any] = {}
    for name, root_path in args.model_output:
        path = resolve_output(root_path, args.step)
        records = output_records(path, mapping_rows, scorer, args.experiment_name, runtime_mapping)
        if len(records) != expected_n:
            raise ValueError(f"{name} coverage is not the expected N={expected_n}")
        runs[name] = {
            "output_path": str(path),
            "primary": summarize(records),
            "tool_only": summarize(scope_records(records, lambda row: row["target_category"] == "tool_only")),
            "response_only": summarize(scope_records(records, lambda row: row["target_category"] == "response_only")),
            "records": records,
        }
        if args.exclude_source_id is not None:
            sensitivity = scope_records(records, lambda row: int(row["source_id"]) != args.exclude_source_id)
            runs[name]["sensitivity_excluding_source_id"] = {
                "excluded_source_id": args.exclude_source_id,
                "metrics": summarize(sensitivity),
            }

    primary_indices, primary_digest = make_bootstrap_indices(expected_n)
    tool_records = scope_records(next(iter(runs.values()))["records"], lambda row: row["target_category"] == "tool_only")
    tool_indices, tool_digest = make_bootstrap_indices(len(tool_records))
    sensitivity_indices = None
    sensitivity_digest = None
    if args.exclude_source_id is not None:
        sensitivity_indices, sensitivity_digest = make_bootstrap_indices(expected_n - 1)

    stratification_counts = {
        "tool_only": sum(1 for row in mapping_rows if row.get("target_category") == "tool_only"),
        "response_only": sum(1 for row in mapping_rows if row.get("target_category") == "response_only"),
    }
    paired: dict[str, Any] = {}
    for left_name, right_name, key, direction in COMPARISONS:
        if left_name not in runs or right_name not in runs:
            if args.fixture_mode:
                continue
            raise ValueError("FINAL_ANALYSIS_REFUSED_INCOMPLETE_MATRIX")
        left_records = runs[left_name]["records"]
        right_records = runs[right_name]["records"]
        left_tool = scope_records(left_records, lambda row: row["target_category"] == "tool_only")
        right_tool = scope_records(right_records, lambda row: row["target_category"] == "tool_only")
        left_response = scope_records(left_records, lambda row: row["target_category"] == "response_only")
        right_response = scope_records(right_records, lambda row: row["target_category"] == "response_only")
        paired[key] = {
            "left": left_name,
            "right": right_name,
            "reported_direction": direction,
            "right_minus_left": {
                "primary": paired_effect(left_records, right_records, primary_indices),
                "tool_only": paired_effect(left_tool, right_tool, tool_indices),
                "response_only": descriptive_effect(left_response, right_response),
            },
        }
        if sensitivity_indices is not None:
            paired[key]["right_minus_left"]["sensitivity_excluding_source_id"] = paired_effect(
                scope_records(left_records, lambda row: int(row["source_id"]) != args.exclude_source_id),
                scope_records(right_records, lambda row: int(row["source_id"]) != args.exclude_source_id),
                sensitivity_indices,
            )

    result = {
        "schema": "projectb_final_endpoint_analysis_v3",
        "source": "persisted JSONL only; no model loading or inference",
        "endpoint": endpoint.get("manifest_version", "fixture"),
        "step": args.step,
        "endpoint_identity": identity,
        "row_count": expected_n,
        "formal_matrix": not args.fixture_mode,
        "model_names": model_names,
        "stratification": {
            **stratification_counts,
            "response_only_interpretation": "descriptive_only_small_sample",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.default_rng",
            "numpy_version": np.__version__,
            "primary_shape": list(primary_indices.shape),
            "primary_index_sha256": primary_digest,
            "tool_shape": list(tool_indices.shape),
            "tool_index_sha256": tool_digest,
            "sensitivity_shape": list(sensitivity_indices.shape) if sensitivity_indices is not None else None,
            "sensitivity_index_sha256": sensitivity_digest,
            "response_only_inferential_ci": False,
        },
        "tie_tolerance": TIE_TOLERANCE,
        "runs": runs,
        "paired_bootstrap_right_minus_left": paired,
        "kl_confound_note": "Final endpoint analysis retains the historical KL-treatment confound; comparisons are not pure advantage-estimator causal tests.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        lines = [
            "# ProjectB FINAL_HOLDOUT_V1 analysis",
            "",
            "CPU-only analysis of persisted outputs; no model loading or inference.",
            "",
            f"- formal matrix: `{not args.fixture_mode}`",
            f"- endpoint: `{result['endpoint']}`",
            f"- step: {args.step}",
            f"- models: {', '.join(model_names)}",
            f"- primary: all frozen N={expected_n} rows",
            f"- stratified auxiliary: tool-only N={stratification_counts['tool_only']}; response-only N={stratification_counts['response_only']} (descriptive only)",
            f"- bootstrap: 10,000 paired resamples, seed 42, NumPy {np.__version__}",
            f"- primary index SHA256: `{primary_digest}`",
            f"- tool-only index SHA256: `{tool_digest}`",
            "- paired direction: every reported difference is right minus left",
            "- interpretation: historical KL-treatment confound remains; no pure advantage-estimator causal claim",
        ]
        for name in model_names:
            metrics = runs[name]["primary"]
            lines.extend(
                [
                    "",
                    f"## {name}",
                    "",
                    f"- primary raw total reward mean: {metrics['raw_total_validation_reward']['mean']}",
                    f"- primary accuracy reward mean: {metrics['accuracy_reward']['mean']}",
                    f"- primary format reward mean: {metrics['format_reward']['mean']}",
                    f"- primary strict format: {metrics['strict_format']['count']}/{metrics['strict_format']['denominator']}",
                    f"- tool-only JSON parse: {runs[name]['tool_only']['tool_parse']['count']}/{runs[name]['tool_only']['tool_parse']['denominator']}",
                    f"- response-only wrapper: {runs[name]['response_only']['response_wrapper']['count']}/{runs[name]['response_only']['response_wrapper']['denominator']} (descriptive; N={stratification_counts['response_only']})",
                ]
            )
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
