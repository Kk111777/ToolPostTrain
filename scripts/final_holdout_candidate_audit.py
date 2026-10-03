#!/usr/bin/env python3
"""CPU-only audit of the untouched drop_last tail for ProjectB.

This script never loads a model, runs inference, computes rewards, writes a
parquet holdout, or starts a trainer. It records only source ids, hashes,
counts, and token-length summaries.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(os.environ.get("PROJECTB_ROOT", "/root/autodl-tmp/ProjectB"))
REPO = ROOT / "repo"
TRAIN = REPO / "verl-GDPO/dataset/rlla_4k/train.parquet"
TEST = REPO / "verl-GDPO/dataset/rlla_4k/test.parquet"
FORMAL = ROOT / "env-modern/formal_validation_80.parquet"
CAPACITY = ROOT / "env-modern/capacity_smoke_512.parquet"
MODEL = ROOT / "checkpoints/rl_init_sft_v2_merged"
RUNS = {
    "GRPO-original": ROOT / "runs/grpo_stage1_35",
    "GDPO-current": ROOT / "runs/gdpo_stage1_35",
    "GRPO-noKL": ROOT / "runs/grpo_nokl_stage1_35",
}
STEPS = (0, 7, 14, 21, 28, 35)
MAX_PROMPT = 2048
BATCH = 512
PREFIX_COUNT = 7 * BATCH

OUT_JSON = REPO / "docs/diagnostics/final_holdout_candidate_inventory_20261003.json"
OUT_MD = REPO / "docs/diagnostics/final_holdout_candidate_inventory_20261003.md"

CATEGORY_ORDER = [
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
]

ID_KEYS = {
    "source_row", "source_id", "source_index", "row_position", "source_row_id",
}
LIST_ID_KEYS = {
    "selected_source_rows", "source_rows", "excluded_source_rows", "source_ids",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).replace("\r\n", "\n").replace("\r", "\n")


def as_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        value = int(value)
        return value if 0 <= value < 3920 else None
    except (TypeError, ValueError):
        return None


def target_category(gt: str) -> str:
    if "<tool_call>" in gt and "<response>" in gt:
        return "tool_and_response"
    if "<tool_call>" in gt:
        return "tool_only"
    if "<response>" in gt:
        return "response_only"
    return "other"


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in (row.get("prompt") or [])
    ]


def row_sig(row: dict[str, Any], source_id: int | None = None) -> dict[str, Any]:
    messages = row_messages(row)
    gt = str((row.get("reward_model") or {}).get("ground_truth", ""))
    exact = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    normalized = json.dumps(
        [{"role": norm(x["role"]), "content": norm(x["content"])} for x in messages],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    sid = source_id
    if sid is None:
        sid = as_int((row.get("extra_info") or {}).get("index"))
    return {
        "source_id": sid,
        "target_category": target_category(gt),
        "prompt_sha256_exact": sha_text(exact),
        "prompt_sha256_normalized": sha_text(normalized),
        "prompt_ground_truth_sha256_exact": sha_text(exact + "\n<GROUND_TRUTH>\n" + gt),
        "prompt_ground_truth_sha256_normalized": sha_text(normalized + "\n<GROUND_TRUTH>\n" + norm(gt)),
        "ground_truth_sha256_exact": sha_text(gt),
        "ground_truth_sha256_normalized": sha_text(norm(gt)),
    }


def render_for_persisted_eval(row: dict[str, Any]) -> str:
    values: list[str] = []
    for message in row.get("prompt") or []:
        values.extend([str(message.get("role", "")), str(message.get("content", ""))])
    return "\n".join(values) + "\nassistant\n"


def sft_sig(item: dict[str, Any]) -> dict[str, Any] | None:
    messages = item.get("messages") or []
    assistant = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"),
        None,
    )
    if assistant is None:
        return None
    return row_sig(
        {
            "prompt": messages[:assistant],
            "reward_model": {"ground_truth": messages[assistant].get("content", "")},
        },
        source_id=as_int(item.get("source_row")),
    )


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def load_tokenizer():
    # Importing a tokenizer is CPU-only; AutoModel is intentionally never imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)


def prompt_len(tokenizer: Any, row: dict[str, Any]) -> int:
    encoded = tokenizer.apply_chat_template(row.get("prompt") or [], tokenize=True, add_generation_prompt=True)
    try:
        encoded = encoded["input_ids"]
    except (KeyError, TypeError, IndexError):
        pass
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def cfg_get(cfg: dict[str, Any], path: str) -> Any:
    value: Any = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def normalize_path_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_path_value(x) for x in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "$PROJECTB_ROOT")
    return value


def verify_run_configs() -> dict[str, Any]:
    required = {
        "data.train_path": str(TRAIN),
        "data.batch_size_prompts": BATCH,
        "data.max_prompt_length": MAX_PROMPT,
        "data.filter_overlong_prompts": True,
        "data.shuffle": False,
        "data.validation_shuffle": False,
        "algorithm_common.seed": 42,
        "data.drop_last": True,
        "budget.stage1.total_steps": 35,
    }
    reports: dict[str, Any] = {}
    fingerprints: list[dict[str, Any]] = []
    for name, path in RUNS.items():
        status_path = path / "run_status.json"
        cfg_path = path / "effective_config.yaml"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
        command = (path / "command.txt").read_text(errors="replace") if (path / "command.txt").is_file() else ""
        observed: dict[str, Any] = {}
        for key, expected in required.items():
            value = cfg_get(cfg, key)
            observed[key] = normalize_path_value(value)
        override_values: dict[str, str | None] = {}
        for key in [
            "data.train_max_samples", "data.train_batch_size", "data.shuffle", "data.validation_shuffle",
            "data.seed", "data.max_prompt_length", "data.filter_overlong_prompts",
        ]:
            match = re.search(rf"(?:^|\s|\")({re.escape(key)})=([^\s\"]+)", command)
            override_values[key] = match.group(2) if match else None
        observed["launcher_overrides"] = override_values
        observed["phase"] = status.get("phase")
        observed["validation_path"] = status.get("validation_path")
        observed["command_sha256"] = sha_file(path / "command.txt") if (path / "command.txt").is_file() else None
        observed["effective_config_sha256"] = sha_file(cfg_path) if cfg_path.is_file() else None
        reports[name] = {"path": str(path), "observed": observed}
        fingerprints.append({k: v for k, v in observed.items() if k not in {"command_sha256", "effective_config_sha256", "phase", "validation_path"}})
    reports["all_three_same_optimizer_inputs"] = len(fingerprints) == 3 and len({json.dumps(x, sort_keys=True, default=str) for x in fingerprints}) == 1
    reports["optimizer_semantics"] = {
        "shuffle": False,
        "drop_last": True,
        "batch_size": BATCH,
        "batches_per_epoch": 7,
        "optimizer_rows_per_epoch": PREFIX_COUNT,
        "source": "formal config + official RLHFDataset/DataLoader audit already frozen in full_train_overlong_filter_audit.md",
    }
    return reports


def classify_path(path: Path) -> str:
    s = str(path)
    if "format_sft_800_v2" in s:
        return "sft_v2"
    if "format_sft_800" in s:
        return "sft_v1"
    if "formal_validation" in s or "/runs/grpo_stage1_35/eval_generations/" in s or "/runs/gdpo_stage1_35/eval_generations/" in s or "/runs/grpo_nokl_stage1_35/eval_generations/" in s:
        return "formal_validation"
    if "test.parquet" in s or "/test/" in s:
        return "original_test"
    if "constrained_generation" in s:
        return "constrained_generation_diagnostic"
    if "fresh_holdout_reward_variation" in s:
        return "fresh_holdout_reward_variation"
    if "grpo_gdpo_shared_rollout" in s:
        return "shared_rollout_diagnostic"
    if "capacity_smoke" in s or "official-scale-capacity" in s or "eligible512" in s:
        return "capacity_smoke"
    if "throughput-tuning" in s:
        return "throughput_tuning"
    if "one_step_smoke" in s or "one-step" in s:
        return "one_step_smoke"
    if "/runs/grpo_formal_v1/" in s or "old_formal" in s:
        return "old_formal_grpo"
    if "rollout_format_debug" in s or "fixed_" in s or "format_debug" in s or "gpu-validation" in s:
        return "fixed_diagnostic"
    if "format_sft_target_audit" in s:
        return "fixed_diagnostic"
    if "/repo/manifests/" in s or "/repo/reports/" in s or "/repo/docs/" in s or "/env-modern/" in s or "/runs/" in s:
        return "historical_scoring_generation"
    return "unknown_persisted_evidence"


class Evidence:
    def __init__(self, train_by_id: dict[int, dict[str, Any]], render_lookup: dict[tuple[str, str], int]):
        self.train_by_id = train_by_id
        self.render_lookup = render_lookup
        self.ids: dict[str, set[int]] = {x: set() for x in CATEGORY_ORDER}
        self.files: dict[str, dict[str, set[int]]] = {x: defaultdict(set) for x in CATEGORY_ORDER}
        self.match_types: dict[str, dict[str, set[int]]] = {x: defaultdict(set) for x in CATEGORY_ORDER}

    def add_id(self, category: str, source_id: int | None, path: Path, match_type: str) -> None:
        if category not in self.ids or source_id is None or source_id not in self.train_by_id:
            return
        self.ids[category].add(source_id)
        self.files[category][str(path)].add(source_id)
        self.match_types[category][match_type].add(source_id)

    def add_sig_match(self, category: str, sig: tuple[str, str] | None, path: Path, match_type: str) -> None:
        if not sig:
            return
        sid = self.render_lookup.get(sig)
        self.add_id(category, sid, path, match_type)


def extract_ids(obj: Any, evidence: Evidence, category: str, path: Path, key_context: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in ID_KEYS:
                evidence.add_id(category, as_int(value), path, key_lower)
            elif key_lower in LIST_ID_KEYS and isinstance(value, list):
                for item in value:
                    evidence.add_id(category, as_int(item), path, key_lower)
            elif key_lower == "extra_info" and isinstance(value, dict):
                evidence.add_id(category, as_int(value.get("index")), path, "extra_info.index")
            elif key_lower in {"id", "sample_id"} and isinstance(value, str):
                match = re.search(r"rlla[-_]train[-_](\d+)", value, flags=re.I)
                if match:
                    evidence.add_id(category, as_int(match.group(1)), path, key_lower)
            extract_ids(value, evidence, category, path, key_lower)
    elif isinstance(obj, list):
        for value in obj:
            extract_ids(value, evidence, category, path, key_context)


def record_sig_from_obj(obj: dict[str, Any]) -> tuple[str, str] | None:
    if "input" in obj and "gts" in obj:
        return sha_text(str(obj.get("input", ""))), sha_text(str(obj.get("gts", "")))
    if "prompt" in obj and isinstance(obj.get("prompt"), list):
        row = {"prompt": obj.get("prompt"), "reward_model": {"ground_truth": obj.get("ground_truth", "")}}
        sig = row_sig(row)
        return sha_text(render_for_persisted_eval(row)), sig["ground_truth_sha256_exact"]
    return None


def scan_json_path(path: Path, evidence: Evidence) -> None:
    category = classify_path(path)
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        extract_ids(obj, evidence, category, path)
                        evidence.add_sig_match(category, record_sig_from_obj(obj), path, "content_render_gt")
        else:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            extract_ids(obj, evidence, category, path)
            if isinstance(obj, dict):
                evidence.add_sig_match(category, record_sig_from_obj(obj), path, "content_render_gt")
    except Exception:
        # A malformed or non-record report is not evidence by itself. The path
        # is retained in the scan inventory, while valid neighboring artifacts
        # continue to be audited.
        return


def scan_text_path(path: Path, evidence: Evidence) -> None:
    category = classify_path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    for match in re.finditer(r"rlla[-_ ]train[-_](\d+)", text, flags=re.I):
        evidence.add_id(category, as_int(match.group(1)), path, "rlla-train-id")
    patterns = [
        r"(?:source[_ -]?(?:row|id)|row[_ -]?position|sample[_ -]?id)\s*[:=]\s*(\d+)",
        r"(?:source\s+(?:row|id))\s+(\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            evidence.add_id(category, as_int(match.group(1)), path, "labelled-source-id")


def iter_evidence_files() -> Iterable[Path]:
    roots = [ROOT / "env-modern", ROOT / "runs", REPO / "manifests", REPO / "reports", REPO / "docs"]
    suffixes = {".json", ".jsonl", ".log", ".txt", ".md", ".yaml", ".yml", ".csv"}
    excluded_parts = {".git", "__pycache__", "hf_cache", "conda_pkgs", "checkpoints", "tmp", ".venv-modern"}
    metadata_only_names = {
        "eligible_train_source_manifest.json",
        "run_status.json",
        "checkpoint_manifest.json",
        "checkpoint_manifest_verified.json",
        "checkpoint_cleanup_manifest.json",
        "stage_completion_gate.json",
        "final_test_dataset_manifest.json",
        "final_test_models_manifest.json",
        "final_test_scorer_manifest.json",
        "final_test_overlap_audit.json",
        "formal_validation_raw_reaudit_20261003.json",
        "stage1_metrics_canonical.json",
        "validation_learning_curve.json",
        "final_holdout_candidate_inventory_20261003.json",
        "final_holdout_candidate_inventory_20261003.md",
        "grpo_gdpo_primary_sensitivity_analysis_plan.md",
    }
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            if path.name in metadata_only_names:
                continue
            try:
                if path.stat().st_size > 40 * 1024 * 1024:
                    continue
            except OSError:
                continue
            if path not in seen:
                seen.add(path)
                yield path


def build_corpus_signatures(rows: Iterable[dict[str, Any]], source_ids: Iterable[int] | None = None) -> dict[str, dict[str, set[int]]]:
    allowed = set(source_ids) if source_ids is not None else None
    out: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        sig = row_sig(row)
        sid = sig["source_id"]
        if sid is None or (allowed is not None and sid not in allowed):
            continue
        for field in ["prompt_sha256_exact", "prompt_sha256_normalized", "prompt_ground_truth_sha256_exact", "prompt_ground_truth_sha256_normalized"]:
            out[field][sig[field]].add(sid)
    return out


def sft_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            assistant = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"),
                None,
            )
            if assistant is None:
                continue
            rows.append({
                "prompt": messages[:assistant],
                "reward_model": {"ground_truth": messages[assistant].get("content", "")},
                "extra_info": {"index": as_int(item.get("source_row"))},
            })
    return rows


def union_sets(mapping: dict[str, set[int]]) -> set[int]:
    out: set[int] = set()
    for values in mapping.values():
        out.update(values)
    return out


def percentiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": float(statistics.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def decision_label(n: int) -> str:
    if n < 50:
        return "CANDIDATE_POOL_TOO_SMALL"
    if n < 80:
        return "CANDIDATE_POOL_WEAK"
    if n < 150:
        return "CANDIDATE_POOL_USABLE_SMALL"
    if n < 300:
        return "CANDIDATE_POOL_GOOD"
    return "CANDIDATE_POOL_STRONG"


def write_outputs(result: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    c = result["counts"]
    lines = [
        "# Final holdout candidate inventory (CPU/no-GPU)",
        "",
        "This is a candidate statistics audit only. No parquet holdout was constructed, no model was loaded, no inference or reward scoring was run, and no final endpoint was frozen.",
        "",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- branch: `{result['git']['branch']}`",
        f"- audited commit: `{result['git']['head']}`",
        f"- decision label: **{result['decision_label']}**",
        "",
        "## Core counts",
        "",
        f"- raw train rows: **{c['raw_train_rows']}**",
        f"- eligible rows after official overlong filter: **{c['eligible_rows']}**",
        f"- optimizer-exposed rows per epoch: **{c['optimizer_exposed_rows']}**",
        f"- initial drop_last tail: **{c['initial_tail_rows']}**",
        f"- final untouched candidates N: **{c['final_untouched_candidates']}**",
        "",
        "## Exposure exclusions",
        "",
        "Counts are unique candidate source IDs hit by each category; category counts may overlap. The union count is the deduplicated exclusion total.",
        "",
        "| category | raw hit count | source IDs |",
        "|---|---:|---|",
    ]
    for category in CATEGORY_ORDER:
        item = result["exposure_categories"][category]
        ids = item["candidate_source_ids"]
        shown = ", ".join(map(str, ids)) if ids else "—"
        lines.append(f"| `{category}` | {len(ids)} | {shown} |")
    lines += [
        "",
        f"- union exposure exclusions: **{c['union_exposure_exclusions']}**",
        f"- remaining after exposure: **{c['remaining_after_exposure']}**",
        "",
        "## Content duplicate and length exclusions",
        "",
        f"- content duplicate exclusions against known corpora: **{c['content_duplicate_exclusions']}**",
        f"- candidate-internal exact/normalized duplicate exclusions: **{c['internal_duplicate_exclusions']}**",
        f"- overlength exclusions after content audit: **{c['overlength_exclusions']}**",
        "",
        "Content comparison uses the frozen NFC/newline normalization and sorted role/content serialization; no semantic, embedding, fuzzy, or outcome-based filtering was used.",
        "",
        "## Final candidate profile",
        "",
        f"- target-category distribution: `{json.dumps(result['final_target_category_distribution'], ensure_ascii=False, sort_keys=True)}`",
        f"- prompt-token distribution: `{json.dumps(result['final_prompt_token_distribution'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Three-run optimizer exposure check",
        "",
        f"- all three runs completed: `{result['run_consistency']['all_three_completed']}`",
        f"- all three use the same optimizer-exposed row semantics: `{result['run_consistency']['all_three_same_optimizer_inputs']}`",
        f"- optimizer source-sequence SHA256: `{result['optimizer_source_sequence_sha256']}`",
        "",
        "## Interpretation",
        "",
        "The label is an engineering decision category, not a formal statistical power guarantee. Even a large candidate pool remains subject to one training seed and possible semantic overlap not detected by exact/normalized hashes.",
        "",
        "No next-stage holdout construction was performed automatically.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    train_rows = parquet_rows(TRAIN)
    train_by_id = {as_int((row.get("extra_info") or {}).get("index")): row for row in train_rows}
    train_by_id = {sid: row for sid, row in train_by_id.items() if sid is not None}
    tokenizer = load_tokenizer()
    lengths = [prompt_len(tokenizer, row) for row in train_rows]
    eligible_positions = [i for i, length in enumerate(lengths) if length <= MAX_PROMPT]
    eligible_ids = [as_int((train_rows[i].get("extra_info") or {}).get("index")) for i in eligible_positions]
    eligible_ids = [sid for sid in eligible_ids if sid is not None]
    prefix_ids = eligible_ids[:PREFIX_COUNT]
    tail_ids = eligible_ids[PREFIX_COUNT:]
    tail_position = {sid: i for i, sid in enumerate(tail_ids)}

    train_sigs = {sid: row_sig(train_by_id[sid], sid) for sid in eligible_ids}
    render_lookup: dict[tuple[str, str], int] = {}
    for sid in eligible_ids:
        sig = train_sigs[sid]
        render_lookup[(sha_text(render_for_persisted_eval(train_by_id[sid])), sig["ground_truth_sha256_exact"])] = sid
    evidence = Evidence(train_by_id, render_lookup)
    for sid in prefix_ids:
        evidence.add_id("training_optimizer", sid, TRAIN, "eligible_prefix")

    # SFT sources: source_row is an explicit source identity, and messages are
    # used for content matching without retaining message text.
    for path, category in [
        (REPO / "dataset/format_sft_800.jsonl", "sft_v1"),
        (REPO / "dataset/format_sft_800_v2.jsonl", "sft_v2"),
    ]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sig = sft_sig(item)
                if sig:
                    evidence.add_id(category, sig["source_id"], path, "source_row")
                    evidence.add_sig_match(category, (sha_text(render_for_persisted_eval({"prompt": item.get("messages", [])[:-1], "reward_model": {"ground_truth": item.get("messages", [])[-1].get("content", "") if item.get("messages") else ""}})), sig["ground_truth_sha256_exact"]), path, "sft_content")

    # Dataset-level exposure identities.
    for path, category in [(FORMAL, "formal_validation"), (TEST, "original_test"), (CAPACITY, "capacity_smoke")]:
        for row in parquet_rows(path):
            sig = row_sig(row)
            evidence.add_id(category, sig["source_id"], path, "parquet.extra_info.index")
            # A source id namespace collision is not enough; content overlap is
            # handled later against all four frozen hashes.

    # Persisted JSON/JSONL manifests, run captures and reports.
    scanned_files = []
    for path in iter_evidence_files():
        scanned_files.append(str(path))
        if path.suffix.lower() in {".json", ".jsonl"}:
            scan_json_path(path, evidence)
        else:
            scan_text_path(path, evidence)

    # All direct source ids are only exclusions if they are in the candidate
    # tail; non-tail hits remain recorded in evidence but do not affect N.
    exposure_sets = {category: set(ids) & set(tail_ids) for category, ids in evidence.ids.items()}
    union_exposed = union_sets(exposure_sets)
    remaining = set(tail_ids) - union_exposed

    # Build signatures for known corpora. Content duplicates are checked after
    # source-id exposure filtering and use no outcome/reward fields.
    corpora: dict[str, dict[str, dict[str, set[int]]]] = {}
    corpora["training_optimizer"] = build_corpus_signatures([train_by_id[sid] for sid in prefix_ids], prefix_ids)
    corpora["sft_v1"] = build_corpus_signatures(sft_rows(REPO / "dataset/format_sft_800.jsonl"))
    corpora["sft_v2"] = build_corpus_signatures(sft_rows(REPO / "dataset/format_sft_800_v2.jsonl"))
    for name, path in [("formal_validation", FORMAL), ("original_test", TEST), ("capacity_smoke", CAPACITY)]:
        corpora[name] = build_corpus_signatures(parquet_rows(path))
    # Known diagnostic datasets are represented by their persisted row IDs; if
    # no parquet is available, JSON records are content-matched by the scanner.
    for category in CATEGORY_ORDER:
        if category not in corpora:
            corpora[category] = defaultdict(lambda: defaultdict(set))
        for path_str, ids in evidence.files[category].items():
            path = Path(path_str)
            if category in {"formal_validation", "original_test", "capacity_smoke"}:
                continue
            for sid in ids:
                if sid in train_sigs:
                    sig = train_sigs[sid]
                    for field in ["prompt_sha256_exact", "prompt_sha256_normalized", "prompt_ground_truth_sha256_exact", "prompt_ground_truth_sha256_normalized"]:
                        corpora[category][field][sig[field]].add(sid)

    duplicate_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_target_ids: set[int] = set()
    fields = ["prompt_sha256_exact", "prompt_sha256_normalized", "prompt_ground_truth_sha256_exact", "prompt_ground_truth_sha256_normalized"]
    for sid in sorted(remaining):
        sig = train_sigs[sid]
        for corpus_name, corpus in corpora.items():
            for field in fields:
                matched = corpus.get(field, {}).get(sig[field], set())
                if matched and not (corpus_name == "training_optimizer" and matched == {sid}):
                    duplicate_target_ids.add(sid)
                    duplicate_details[str(sid)].append({"corpus": corpus_name, "field": field, "matched_source_ids": sorted(matched)[:50]})

    # Conservative internal duplicate policy: exclude every member of any
    # exact/normalized duplicate group so no ambiguous representative is chosen.
    internal_duplicate_ids: set[int] = set()
    internal_details: list[dict[str, Any]] = []
    for field in fields:
        groups: dict[str, list[int]] = defaultdict(list)
        for sid in sorted(remaining):
            groups[train_sigs[sid][field]].append(sid)
        for digest, ids in groups.items():
            if len(ids) > 1:
                internal_duplicate_ids.update(ids)
                internal_details.append({"field": field, "source_ids": ids})

    after_content = remaining - duplicate_target_ids - internal_duplicate_ids
    token_lengths = {sid: lengths[sid] for sid in []}
    for sid in after_content:
        # Source id equals original row position in this dataset; verify rather
        # than assume before using the compact list.
        row_pos = next(i for i, row in enumerate(train_rows) if as_int((row.get("extra_info") or {}).get("index")) == sid)
        token_lengths[sid] = lengths[row_pos]
    overlength_ids = {sid for sid, length in token_lengths.items() if length > MAX_PROMPT}
    final_ids = sorted(after_content - overlength_ids)

    target_distribution: dict[str, int] = defaultdict(int)
    for sid in final_ids:
        target_distribution[train_sigs[sid]["target_category"]] += 1

    git_branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    git_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sequence_sha = sha_text("\n".join(map(str, prefix_ids)))
    run_consistency = verify_run_configs()
    run_consistency["all_three_completed"] = all(item["observed"].get("phase") == "completed" for name, item in run_consistency.items() if name in RUNS)

    exposure_categories = {}
    for category in CATEGORY_ORDER:
        ids = sorted(exposure_sets[category])
        exposure_categories[category] = {
            "candidate_source_ids": ids,
            "raw_hit_count": len(ids),
            "evidence_files": {p: sorted(v & set(tail_ids)) for p, v in sorted(evidence.files[category].items()) if v & set(tail_ids)},
            "match_types": {k: sorted(v & set(tail_ids)) for k, v in sorted(evidence.match_types[category].items()) if v & set(tail_ids)},
        }

    final_rows = []
    for sid in final_ids:
        row_pos = next(i for i, row in enumerate(train_rows) if as_int((row.get("extra_info") or {}).get("index")) == sid)
        final_rows.append({
            "source_id": sid,
            "row_position": row_pos,
            "tail_position": tail_position[sid],
            "target_category": train_sigs[sid]["target_category"],
            "prompt_tokens": token_lengths[sid],
            "prompt_sha256_exact": train_sigs[sid]["prompt_sha256_exact"],
            "prompt_sha256_normalized": train_sigs[sid]["prompt_sha256_normalized"],
            "prompt_ground_truth_sha256_exact": train_sigs[sid]["prompt_ground_truth_sha256_exact"],
            "prompt_ground_truth_sha256_normalized": train_sigs[sid]["prompt_ground_truth_sha256_normalized"],
        })

    result = {
        "schema": "projectb_final_holdout_candidate_inventory_v1",
        "generated_at_utc": now(),
        "policy": {
            "cpu_only": True,
            "model_loaded": False,
            "inference_run": False,
            "reward_scoring_run": False,
            "parquet_holdout_constructed": False,
            "final_endpoint_frozen": False,
            "normalization": "Unicode NFC; CRLF/CR to LF; sorted role/content JSON; argument values preserved",
            "duplicate_policy": "exclude all members of any exact/normalized internal duplicate group; no fuzzy or semantic matching",
            "overlength_policy": "official <=2048 eligibility; no truncation",
        },
        "git": {"branch": git_branch, "head": git_head},
        "paths": {
            "train": str(TRAIN), "formal_validation": str(FORMAL), "test": str(TEST), "capacity": str(CAPACITY), "tokenizer": str(MODEL)
        },
        "run_consistency": run_consistency,
        "counts": {
            "raw_train_rows": len(train_rows),
            "eligible_rows": len(eligible_ids),
            "optimizer_exposed_rows": len(prefix_ids),
            "initial_tail_rows": len(tail_ids),
            "union_exposure_exclusions": len(union_exposed),
            "remaining_after_exposure": len(remaining),
            "content_duplicate_exclusions": len(duplicate_target_ids),
            "internal_duplicate_exclusions": len(internal_duplicate_ids),
            "overlength_exclusions": len(overlength_ids),
            "final_untouched_candidates": len(final_ids),
        },
        "optimizer_source_sequence_sha256": sequence_sha,
        "optimizer_exposed_source_ids": prefix_ids,
        "initial_tail_source_ids": tail_ids,
        "eligible_filter": {
            "max_prompt_length": MAX_PROMPT,
            "raw_prompt_lengths_min": min(lengths),
            "raw_prompt_lengths_max": max(lengths),
            "overlong_count": len(train_rows) - len(eligible_ids),
            "eligible_source_ids_sha256": sha_text("\n".join(map(str, eligible_ids))),
        },
        "exposure_categories": exposure_categories,
        "exposure_union_source_ids": sorted(union_exposed),
        "content_duplicate_details": dict(duplicate_details),
        "internal_duplicate_details": internal_details,
        "final_target_category_distribution": dict(sorted(target_distribution.items())),
        "final_prompt_token_distribution": percentiles([token_lengths[sid] for sid in final_ids]),
        "scanned_evidence_file_count": len(scanned_files),
        "candidate_source_ids_after_exposure": sorted(remaining),
        "candidate_source_ids_after_content_audit": sorted(after_content),
        "candidate_source_ids_after_overlength": final_ids,
        "final_candidate_rows": final_rows,
        "future_decision_note": "Use all clean N by default only after a separate user decision; do not construct a parquet automatically.",
    }
    result["decision_label"] = decision_label(len(final_ids))
    write_outputs(result)
    print(json.dumps({
        "decision_label": result["decision_label"],
        "counts": result["counts"],
        "target_distribution": result["final_target_category_distribution"],
        "token_distribution": result["final_prompt_token_distribution"],
        "branch": git_branch,
        "head": git_head,
        "output_json": str(OUT_JSON),
        "output_md": str(OUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
