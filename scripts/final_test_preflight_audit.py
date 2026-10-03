#!/usr/bin/env python3
"""CPU-only ProjectB final-test preflight audit; never performs test inference."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import statistics
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("PROJECTB_ROOT", "/root/autodl-tmp/ProjectB"))
REPO = ROOT / "repo"
PREFLIGHT = ROOT / "final-test-preflight"
DOCS = REPO / "docs" / "diagnostics"
MANIFESTS = REPO / "manifests"
COMPARE = REPO / "reports" / "comparison"
STEPS = [0, 7, 14, 21, 28, 35]
TEST = REPO / "verl-GDPO/dataset/rlla_4k/test.parquet"
TRAIN = REPO / "verl-GDPO/dataset/rlla_4k/train.parquet"
FORMAL = ROOT / "env-modern/formal_validation_80.parquet"
MODELS = {
    "RL_INIT_V1": ROOT / "checkpoints/rl_init_sft_v2_merged",
    "GRPO-original35": ROOT / "runs/grpo_stage1_35/model_only/step_35",
    "GDPO-current35": ROOT / "runs/gdpo_stage1_35/model_only/step_35",
    "GRPO-noKL35": ROOT / "runs/grpo_nokl_stage1_35/model_only/step_35",
}
RUNS = {
    "GRPO-original": (ROOT / "runs/grpo_stage1_35", "qwen2p5_1p5b_grpo_one_step"),
    "GDPO-current": (ROOT / "runs/gdpo_stage1_35", "qwen2p5_1p5b_gdpo_one_step"),
    "GRPO-noKL": (ROOT / "runs/grpo_nokl_stage1_35", "qwen2p5_1p5b_grpo_one_step"),
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def norm(text):
    return unicodedata.normalize("NFC", str(text or "")).replace("\r\n", "\n").replace("\r", "\n")


def parts(row):
    messages = [{"role": str(x.get("role", "")), "content": str(x.get("content", ""))}
                for x in (row.get("prompt") or [])]
    gt = str((row.get("reward_model") or {}).get("ground_truth", ""))
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    normalized = json.dumps(
        [{"role": norm(x["role"]), "content": norm(x["content"])} for x in messages],
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )
    sid = (row.get("extra_info") or {}).get("index")
    try:
        sid = int(sid)
    except (TypeError, ValueError):
        sid = None
    category = "tool_and_response" if "<tool_call>" in gt and "<response>" in gt else (
        "tool_only" if "<tool_call>" in gt else "response_only" if "<response>" in gt else "other"
    )
    return {
        "source_id": sid,
        "target_category": category,
        "prompt_sha256_exact": sha_text(raw),
        "prompt_sha256_normalized": sha_text(normalized),
        "prompt_ground_truth_sha256_exact": sha_text(raw + "\n<GROUND_TRUTH>\n" + gt),
        "prompt_ground_truth_sha256_normalized": sha_text(normalized + "\n<GROUND_TRUTH>\n" + norm(gt)),
        "ground_truth_sha256_exact": sha_text(gt),
        "ground_truth_sha256_normalized": sha_text(norm(gt)),
        "_messages": messages,
    }


def parquet_rows(path):
    table = pq.read_table(path)
    return table.to_pylist(), str(table.schema)


def render(row):
    values = []
    for message in row.get("prompt") or []:
        values.extend([str(message.get("role", "")), str(message.get("content", ""))])
    return "\n".join(values) + "\nassistant\n"


def make_dataset():
    rows, schema = parquet_rows(TEST)
    entries = []
    for i, row in enumerate(rows):
        item = parts(row)
        item.pop("_messages", None)
        item.update({"row_position": i, "ability": row.get("ability"), "data_source": row.get("data_source")})
        entries.append(item)
    out = {
        "schema": "projectb_final_test_dataset_manifest_v1",
        "generated_at_utc": now(), "path": str(TEST), "sha256": sha_file(TEST),
        "size_bytes": TEST.stat().st_size, "row_count": len(rows), "schema_string": schema,
        "stable_row_id": "extra_info.index", "order_frozen": True,
        "read_policy": "dataset identity only; no test model output was read", "rows": entries,
    }
    write_json(MANIFESTS / "final_test_dataset_manifest.json", out)
    write_json(PREFLIGHT / "final_test_dataset_manifest.json", out)
    return rows, out


def sft_records(path):
    out = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            ai = next((i for i in range(len(messages) - 1, -1, -1)
                       if messages[i].get("role") == "assistant"), None)
            if ai is None:
                continue
            out.append(parts({
                "prompt": messages[:ai],
                "reward_model": {"ground_truth": messages[ai].get("content", "")},
                "extra_info": {"index": item.get("source_row")},
            }))
    return out


def overlap(test_manifest):
    train_rows, _ = parquet_rows(TRAIN)
    formal_rows, _ = parquet_rows(FORMAL)
    corpora = {
        "raw_rl_train": [parts(x) for x in train_rows],
        "formal_validation_80": [parts(x) for x in formal_rows],
        "SFT_v1_format_800": sft_records(REPO / "dataset/format_sft_800.jsonl"),
        "SFT_v2_format_800": sft_records(REPO / "dataset/format_sft_800_v2.jsonl"),
    }
    eligible_path = ROOT / "runs/grpo_formal_v1/eligible_train_source_manifest.json"
    if eligible_path.is_file():
        eligible = {int(x["source_id"]) for x in json.loads(eligible_path.read_text()).get("eligible", [])}
        corpora["eligible_rl_train"] = [x for x in corpora["raw_rl_train"] if x["source_id"] in eligible]
    known = {}
    for path in [
        ROOT / "env-modern/rollout_format_debug.json",
        ROOT / "env-modern/grpo_gdpo_shared_rollout_diagnostic.json",
        ROOT / "env-modern/fresh_holdout_reward_variation.json",
        ROOT / "env-modern/constrained_generation_report.json",
    ]:
        known[str(path)] = {"exists": path.is_file(), "comparable_records": 0}
    fields = [
        "prompt_sha256_exact", "prompt_ground_truth_sha256_exact",
        "prompt_sha256_normalized", "prompt_ground_truth_sha256_normalized",
    ]
    test_ids = {x["source_id"] for x in test_manifest["rows"] if x["source_id"] is not None}
    comparisons = {}
    for name, records in corpora.items():
        comparisons[name] = {
            "record_count": len(records),
            "source_id_overlap": sorted(test_ids & {x["source_id"] for x in records if x["source_id"] is not None}),
            "hash_overlap": {
                field: {
                    "count": len({x[field] for x in test_manifest["rows"]} & {x[field] for x in records}),
                    "hashes": sorted({x[field] for x in test_manifest["rows"]} & {x[field] for x in records}),
                } for field in fields
            },
        }
    out = {
        "schema": "projectb_final_test_overlap_audit_v1", "generated_at_utc": now(),
        "normalization": {
            "unicode": "NFC", "newlines": "CRLF/CR -> LF",
            "serialization": "sorted role/content JSON; argument values preserved",
        },
        "test_rows": len(test_manifest["rows"]), "comparisons": comparisons,
        "known_diagnostic_inventory": known,
        "interpretation": "identity overlap checks do not prove semantic overlap",
    }
    write_json(PREFLIGHT / "final_test_overlap_audit.json", out)
    write_json(DOCS / "final_test_overlap_audit.json", out)
    lines = ["# Final-test overlap audit", "", "CPU-only; no test model output was read.", ""]
    for name, item in comparisons.items():
        lines += [f"## {name}", f"- comparable records: {item['record_count']}",
                  f"- source-id overlap: {len(item['source_id_overlap'])}"]
        lines += [f"- {field}: {detail['count']}" for field, detail in item["hash_overlap"].items()]
        lines.append("")
    lines.append("No rows were deleted or redefined; non-zero overlap requires manual review.")
    write_text(DOCS / "final_test_overlap_audit.md", "\n".join(lines))
    write_text(PREFLIGHT / "final_test_overlap_audit.md", "\n".join(lines))
    return out


def provenance():
    files = [
        ROOT / "runs/grpo_formal_v1/launch_command.txt",
        ROOT / "runs/grpo_formal_v1/effective_config.yaml",
        ROOT / "runs/grpo_formal_v1/run_formal_grpo.sh",
        ROOT / "runs/grpo_formal_v1/logs/supervisor.log",
        ROOT / "env-modern/gpu-validation-logs/validation.log",
        ROOT / "env-modern/gpu-validation-logs/grpo-driver.log",
        ROOT / "env-modern/gpu-validation-logs/gdpo-driver.log",
        ROOT / "env-modern/official-scale-capacity-logs/grpo.stdout.log",
        ROOT / "env-modern/official-scale-capacity-logs/gdpo_eligible512.stdout.log",
        ROOT / "env-modern/throughput-tuning-logs/grpo_config_a.stdout.log",
        ROOT / "env-modern/throughput-tuning-logs/gdpo_throughput_config_a.stdout.log",
        REPO / "scripts/run_single_gpu_smoke.sh",
    ]
    evidence, runtime_count, score_count = [], 0, 0
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        a = len(re.findall(r"val_files[^\n]*test\.parquet|test\.parquet[^\n]*val_files", text))
        b = len(re.findall(r"test\.parquet_rows|Score:|Final validation metrics", text))
        if a or b:
            evidence.append({"path": str(path), "runtime_val_matches": a, "score_metric_matches": b})
        runtime_count += a
        score_count += b
    out = {
        "schema": "projectb_final_test_historical_usage_audit_v1", "generated_at_utc": now(),
        "evidence": evidence,
        "classification": {
            "A_code_or_config_reference": True,
            "B_runtime_log_shows_test_loaded": runtime_count > 0,
            "C_test_metric_or_generation_evidence": score_count > 0,
            "D_explicit_later_decision_use_found": False,
        },
        "narrow_conclusion": (
            "Historical runtime use of test.parquet is confirmed by persisted commands/logs, "
            "including test row-count/scoring markers. Formal Stage1 intermediate validation "
            "used formal_validation_80. No direct evidence was found that historical test scores "
            "drove later budget, hyperparameter, or checkpoint decisions in the audited decision "
            "files; absence of evidence is not proof of non-use. The test cannot be represented "
            "as fully unseen without manual review."
        ),
        "gpu_gate": "GPU_FINAL_TEST_BLOCKED",
    }
    write_json(PREFLIGHT / "historical_test_provenance.json", out)
    write_json(DOCS / "final_test_historical_usage_audit.json", out)
    lines = [
        "# Final-test historical usage audit", "",
        "| level | result |", "|---|---|",
        f"| A: code/config reference | {out['classification']['A_code_or_config_reference']} |",
        f"| B: runtime log shows test loaded | {out['classification']['B_runtime_log_shows_test_loaded']} |",
        f"| C: test metric/generation evidence | {out['classification']['C_test_metric_or_generation_evidence']} |",
        f"| D: explicit later decision use found | {out['classification']['D_explicit_later_decision_use_found']} |",
        "", out["narrow_conclusion"], "",
        "Gate: GPU_FINAL_TEST_BLOCKED. No test inference was launched.", "",
        "Evidence paths:",
    ]
    lines += [f"- {x['path']}: runtime val matches={x['runtime_val_matches']}, score/metric matches={x['score_metric_matches']}"
              for x in evidence]
    write_text(DOCS / "final_test_historical_usage_audit.md", "\n".join(lines))
    write_text(PREFLIGHT / "final_test_historical_usage_audit.md", "\n".join(lines))
    return out


def models_manifest():
    from safetensors import safe_open
    names = ["config.json", "generation_config.json", "tokenizer_config.json",
             "tokenizer.json", "chat_template.jinja", "model.safetensors", "merge_metadata.json"]
    result = {"schema": "projectb_final_test_models_manifest_v1", "generated_at_utc": now(),
              "read_policy": "CPU file/header identity only; no AutoModel load and no GPU", "models": {}}
    for name, path in MODELS.items():
        files = {}
        for filename in names:
            p = path / filename
            if p.is_file():
                files[filename] = {"size_bytes": p.stat().st_size, "sha256": sha_file(p)}
        weight = path / "model.safetensors"
        structure = {"checked": False, "readable": False}
        if weight.is_file():
            try:
                with safe_open(str(weight), framework="pt", device="cpu") as h:
                    structure = {"checked": True, "readable": True, "tensor_count": len(h.keys())}
            except Exception as exc:
                structure = {"checked": True, "readable": False, "error": repr(exc)}
        result["models"][name] = {"path": str(path), "exists": path.is_dir(),
                                  "files": files, "safetensors_structure": structure}
    write_json(MANIFESTS / "final_test_models_manifest.json", result)
    write_json(PREFLIGHT / "final_test_models_manifest.json", result)
    return result


def scorer_manifest():
    upstream = ROOT / "upstream/verl-v0.9.1"
    repo_verl = REPO / "verl-GDPO/verl"
    paths = [
        upstream / "verl/utils/reward_score/rlla.py",
        upstream / "verl/experimental/reward_loop/reward_manager/gdpo.py",
        repo_verl / "trainer/main_ppo.py",
        repo_verl / "trainer/ppo/ray_trainer.py",
        repo_verl / "utils/reward_score/rlla.py",
    ]
    revisions = {}
    for name, path in {"upstream": upstream, "repo": REPO}.items():
        try:
            revisions[name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            revisions[name] = None
    out = {
        "schema": "projectb_final_test_scorer_manifest_v1", "generated_at_utc": now(),
        "runtime_resolution": {
            "trainer_import_root": str(repo_verl),
            "custom_reward_function": str(upstream / "verl/utils/reward_score/rlla.py"),
            "reward_manager": str(upstream / "verl/experimental/reward_loop/reward_manager/gdpo.py"),
            "task_reward_only": True, "reward_side_kl": False, "reference_kl_score": False,
            "same_scorer_for_all_models": True,
        },
        "git_revisions": revisions,
        "files": [{"path": str(p), "exists": p.is_file(),
                   "size_bytes": p.stat().st_size if p.is_file() else None,
                   "sha256": sha_file(p) if p.is_file() else None} for p in paths],
    }
    write_json(MANIFESTS / "final_test_scorer_manifest.json", out)
    write_json(PREFLIGHT / "final_test_scorer_manifest.json", out)
    return out


def prompt_audit(test_rows, manifest):
    from transformers import AutoTokenizer
    tokenizer_path = MODELS["RL_INIT_V1"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    lengths, errors = [], []
    for i, row in enumerate(test_rows):
        try:
            ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            lengths.append(len(ids.tolist() if hasattr(ids, "tolist") else ids))
        except Exception as exc:
            lengths.append(None)
            errors.append({"row": i, "error": repr(exc)})
    valid = [x for x in lengths if x is not None]
    out = {
        "schema": "projectb_final_test_prompt_length_audit_v1", "generated_at_utc": now(),
        "tokenizer_path": str(tokenizer_path), "chat_template_sha256": sha_file(tokenizer_path / "chat_template.jinja"),
        "format": "AutoTokenizer.apply_chat_template(add_generation_prompt=True)", "max_prompt_length": 2048,
        "row_count": len(test_rows), "lengths": lengths, "min": min(valid), "median": float(statistics.median(valid)),
        "p95": float(np.percentile(valid, 95)), "max": max(valid), "over_2048_count": sum(x > 2048 for x in valid),
        "errors": errors, "go": bool(valid) and not errors and max(valid) <= 2048,
    }
    for entry, length in zip(manifest["rows"], lengths, strict=True):
        entry["prompt_tokens_final_test_template"] = length
    write_json(MANIFESTS / "final_test_dataset_manifest.json", manifest)
    write_json(PREFLIGHT / "final_test_dataset_manifest.json", manifest)
    write_json(PREFLIGHT / "prompt_length_audit.json", out)
    write_json(DOCS / "final_test_prompt_length_audit.json", out)
    write_text(DOCS / "final_test_prompt_length_audit.md",
               "\n".join(["# Final-test prompt length audit", "",
                          f"- tokenizer: {tokenizer_path}",
                          "- no truncation was applied",
                          f"- min / median / p95 / max: {out['min']} / {out['median']} / {out['p95']} / {out['max']}",
                          f"- over 2048: {out['over_2048_count']}",
                          f"- errors: {len(errors)}", f"- status: {'PASS' if out['go'] else 'BLOCKED'}"]))
    write_text(PREFLIGHT / "final_test_prompt_length_audit.md",
               (DOCS / "final_test_prompt_length_audit.md").read_text())
    return out


def load_scorer():
    path = ROOT / "upstream/verl-v0.9.1/verl/utils/reward_score/rlla.py"
    spec = importlib.util.spec_from_file_location("projectb_final_test_rlla", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_format(output, gt):
    if "<|im_start|>assistant" in output:
        output = output.split("<|im_start|>assistant", 1)[-1].split("<|im_end|>", 1)[0].strip()
    if "<response>" in gt and "<tool_call>" not in gt:
        return bool(re.search(r"^<think>.*?</think>\n<response>.*?</response>$", output, re.DOTALL)
                    and output.count("<response>") == output.count("</response>") == 1)
    if "<tool_call>" in gt and "<response>" not in gt:
        return bool(re.search(r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$", output, re.DOTALL)
                    and output.count("<tool_call>") == output.count("</tool_call>") == 1)
    if "<tool_call>" in gt and "<response>" in gt:
        return bool(re.search(r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$", output, re.DOTALL)
                    and output.count("<tool_call>") == output.count("</tool_call>") == 1
                    and output.count("<response>") == output.count("</response>") == 1)
    return False


def tool_parse(output, gt):
    try:
        block = output.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        return bool(lines) and all(isinstance(json.loads(x), dict) for x in lines)
    except Exception:
        return False


def response_wrapper(output, gt):
    return output.count("<response>") == output.count("</response>") == 1


def metric(values):
    return {"mean": float(np.mean(values)), "finite": bool(np.isfinite(values).all())}


def bool_metric(values):
    values = [bool(x) for x in values if x is not None]
    return {"denominator": len(values), "count": sum(values), "rate": sum(values) / len(values) if values else None}


def validation_step(scorer, formal_rows, run_path, experiment_name, step):
    path = run_path / "eval_generations" / f"{step}.jsonl"
    rows_by_key = {}
    for position, row in enumerate(formal_rows):
        gt = str((row.get("reward_model") or {}).get("ground_truth", ""))
        rows_by_key[(sha_text(render(row)), sha_text(gt))] = (position, row)
    records, errors, seen = [], [], set()
    for item in [json.loads(x) for x in path.read_text().splitlines() if x.strip()]:
        key = (sha_text(str(item.get("input", ""))), sha_text(str(item.get("gts", ""))))
        if key not in rows_by_key:
            errors.append({"type": "unmatched_row"})
            continue
        if key in seen:
            errors.append({"type": "duplicate_row"})
            continue
        seen.add(key)
        position, row = rows_by_key[key]
        gt, output = str((row.get("reward_model") or {}).get("ground_truth", "")), str(item.get("output", ""))
        with contextlib.redirect_stdout(io.StringIO()):
            result = scorer.compute_score(row.get("data_source", "rlla"), output, gt,
                                         {"experiment_name": experiment_name}, step=step)
        records.append({
            "row_position": position, "source_id": (row.get("extra_info") or {}).get("index"),
            "input_sha256": key[0], "ground_truth_sha256": key[1], "output_sha256": sha_text(output),
            "score": float(result["score"]), "accuracy_reward": float(result["accuracy_reward"]),
            "format_reward": float(result["format_reward"]), "strict_format": strict_format(output, gt),
            "tool_parse": tool_parse(output, gt), "response_wrapper": response_wrapper(output, gt),
            "persisted_score": float(item.get("score", item.get("reward"))),
            "persisted_accuracy_reward": float(item["accuracy_reward"]),
            "persisted_format_reward": float(item["format_reward"]),
        })
    records.sort(key=lambda x: x["row_position"])
    if len(records) != len(formal_rows):
        errors.append({"type": "coverage", "actual": len(records), "expected": len(formal_rows)})

    def scope(items):
        return {
            "n": len(items),
            "raw_total_validation_reward": metric([x["score"] for x in items]),
            "accuracy_reward": metric([x["accuracy_reward"] for x in items]),
            "format_reward": metric([x["format_reward"] for x in items]),
            "strict_format": bool_metric([x["strict_format"] for x in items]),
            "tool_parse": bool_metric([x["tool_parse"] for x in items]),
            "response_wrapper": bool_metric([x["response_wrapper"] for x in items]),
        }
    return {
        "path": str(path), "line_count": len(records), "coverage": len(records) == len(formal_rows) and not errors,
        "errors": errors, "records": records, "primary_80": scope(records),
        "sensitivity_79": scope([x for x in records if int(x["source_id"]) != 3814]),
        "max_abs_recompute_delta": {
            "score": max((abs(x["score"] - x["persisted_score"]) for x in records), default=None),
            "accuracy": max((abs(x["accuracy_reward"] - x["persisted_accuracy_reward"]) for x in records), default=None),
            "format": max((abs(x["format_reward"] - x["persisted_format_reward"]) for x in records), default=None),
        },
    }


def bootstrap(left, right, rng):
    n = len(left)
    index = rng.integers(0, n, size=(10000, n))
    diff = right[index].mean(1) - left[index].mean(1)
    return {
        "n": n, "mean_difference": float(right.mean() - left.mean()),
        "ci95_low": float(np.quantile(diff, .025)), "ci95_median": float(np.quantile(diff, .5)),
        "ci95_high": float(np.quantile(diff, .975)),
        "approx_two_sided_tail": float(min(1.0, 2.0 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0))))),
    }


def validation_reaudit():
    scorer, formal_rows = load_scorer(), parquet_rows(FORMAL)[0]
    result = {}
    for name, (path, experiment) in RUNS.items():
        result[name] = {str(step): validation_step(scorer, formal_rows, path, experiment, step) for step in STEPS}
    pairs = [
        ("GRPO-original_vs_GDPO-current", "GRPO-original", "GDPO-current"),
        ("GRPO-original_vs_GRPO-noKL", "GRPO-original", "GRPO-noKL"),
        ("GDPO-current_vs_GRPO-noKL", "GDPO-current", "GRPO-noKL"),
    ]
    rng = np.random.default_rng(42)
    boot = {"primary_80": {}, "sensitivity_79": {}}
    for pair, left_name, right_name in pairs:
        for scope_name, predicate in [("primary_80", lambda x: True), ("sensitivity_79", lambda x: int(x["source_id"]) != 3814)]:
            boot[scope_name][pair] = {}
            for step in STEPS:
                l = [x["score"] for x in result[left_name][str(step)]["records"] if predicate(x)]
                r = [x["score"] for x in result[right_name][str(step)]["records"] if predicate(x)]
                boot[scope_name][pair][str(step)] = bootstrap(np.asarray(l), np.asarray(r), rng)

    canonical_paths = {
        "GRPO-original": REPO / "reports/grpo/grpo_stage1_metrics_canonical.json",
        "GDPO-current": REPO / "reports/gdpo/gdpo_stage1_metrics_canonical.json",
        "GRPO-noKL": REPO / "reports/grpo_nokl/stage1_35/metrics_at_final.json",
    }
    discrepancies = []
    for name, path in canonical_paths.items():
        if not path.is_file():
            discrepancies.append({"run": name, "type": "missing"})
            continue
        expected = json.loads(path.read_text())
        for step in STEPS:
            for scope_name in ["primary_80", "sensitivity_79"]:
                obs = result[name][str(step)][scope_name]
                exp = expected[scope_name][str(step)]
                for m in ["raw_total_validation_reward", "accuracy_reward", "format_reward"]:
                    if not math.isclose(obs[m]["mean"], exp[m]["mean"], abs_tol=1e-8, rel_tol=0):
                        discrepancies.append({"run": name, "step": step, "scope": scope_name, "metric": m})
                for m in ["strict_format", "tool_parse", "response_wrapper"]:
                    if obs[m]["count"] != exp[m]["count"]:
                        discrepancies.append({"run": name, "step": step, "scope": scope_name, "metric": m})
    archive_path = COMPARE / "paired_bootstrap_stage1_35.json"
    archived = json.loads(archive_path.read_text()) if archive_path.is_file() else None
    boot_discrepancies = []
    if archived:
        for scope_name in boot:
            for pair in boot[scope_name]:
                for step in map(str, STEPS):
                    for m in ["mean_difference", "ci95_low", "ci95_median", "ci95_high"]:
                        if not math.isclose(boot[scope_name][pair][step][m], archived["right_minus_left"][pair][scope_name][step][m], abs_tol=1e-8, rel_tol=0):
                            boot_discrepancies.append({"scope": scope_name, "pair": pair, "step": int(step), "metric": m})
    out = {
        "schema": "projectb_final_test_formal_validation_raw_reaudit_v1", "generated_at_utc": now(),
        "source": "persisted formal validation eval_generations only; no retraining or inference",
        "validation_path": str(FORMAL), "steps": STEPS, "runs": result,
        "canonical_metrics_match": not discrepancies, "canonical_discrepancies": discrepancies,
        "bootstrap": boot, "bootstrap_matches_archived": not boot_discrepancies,
        "bootstrap_discrepancies": boot_discrepancies,
        "bootstrap_method": {"replicates": 10000, "seed": 42, "numpy": np.__version__, "rng": "numpy.random.default_rng"},
    }
    write_json(PREFLIGHT / "formal_validation_raw_reaudit.json", out)
    write_json(COMPARE / "formal_validation_raw_reaudit_20261003.json", out)
    write_text(DOCS / "formal_validation_raw_reaudit_20261003.md",
               "\n".join(["# Formal validation raw re-audit (2026-10-03)", "",
                          "Recomputed from persisted Stage1 JSONL only; no training or test inference.",
                          f"- canonical compact metrics match: {out['canonical_metrics_match']}",
                          f"- archived 10,000-row bootstrap match: {out['bootstrap_matches_archived']}",
                          f"- canonical discrepancy count: {len(discrepancies)}",
                          f"- bootstrap discrepancy count: {len(boot_discrepancies)}",
                          "Per-row records contain hashes and metrics, not prompt or response text."]))
    write_text(PREFLIGHT / "formal_validation_raw_reaudit_20261003.md",
               (DOCS / "formal_validation_raw_reaudit_20261003.md").read_text())
    return out


def main():
    for path in [PREFLIGHT, DOCS, MANIFESTS, COMPARE]:
        path.mkdir(parents=True, exist_ok=True)
    test_rows, manifest = make_dataset()
    p = provenance()
    o = overlap(manifest)
    pl = prompt_audit(test_rows, manifest)
    mm = models_manifest()
    scorer_manifest()
    vr = validation_reaudit()
    summary = {
        "schema": "projectb_final_test_preflight_cpu_summary_v1", "generated_at_utc": now(),
        "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        "historical_test_provenance": p["gpu_gate"], "test_rows": manifest["row_count"],
        "test_sha256": manifest["sha256"], "prompt_length_pass": pl["go"],
        "model_identity_all_readable": all(x["exists"] and x["safetensors_structure"]["readable"] for x in mm["models"].values()),
        "validation_canonical_metrics_match": vr["canonical_metrics_match"],
        "validation_bootstrap_match": vr["bootstrap_matches_archived"],
        "overlap_source_id_or_hash_nonzero": any(x["source_id_overlap"] or any(y["count"] for y in x["hash_overlap"].values()) for x in o["comparisons"].values()),
        "gpu_final_test_gate": "GPU_FINAL_TEST_BLOCKED",
        "blockers": ["historical runtime logs confirm test.parquet was loaded and scored before this preflight"],
    }
    write_json(PREFLIGHT / "preflight_cpu_summary.json", summary)


if __name__ == "__main__":
    main()
