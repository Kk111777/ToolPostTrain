#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/root/autodl-tmp/ProjectB")
REPO = ROOT / "repo"
PYTHON = ROOT / ".venv-modern/bin/python"
PREFLIGHT = ROOT / "final-test-preflight"
RESULT = REPO / "docs/diagnostics/final_holdout_v1_static_checks.json"
PASS_MARKER = PREFLIGHT / "FINAL_HOLDOUT_V1_FROZEN_PASS"


def run(*args: str) -> tuple[int, str, str]:
    completed = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    return completed.returncode, completed.stdout, completed.stderr


def main() -> None:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    code, stdout, stderr = run(str(PYTHON), "-m", "py_compile", "scripts/freeze_final_holdout_v1.py", "scripts/verify_final_holdout_v1.py", "scripts/analyze_final_test.py", "scripts/resolve_final_holdout_v1.py")
    add("python_compile", code == 0, stderr[-500:])
    code, stdout, stderr = run("bash", "-n", "scripts/run_final_test.sh")
    add("bash_syntax", code == 0, stderr[-500:])
    code, stdout, stderr = run(str(PYTHON), "scripts/analyze_final_test.py", "--self-test")
    add("analysis_self_test", code == 0 and "ANALYSIS_SELF_TEST_PASS" in stdout, stdout.strip())
    code, stdout, stderr = run(str(PYTHON), "scripts/verify_final_holdout_v1.py")
    add("freeze_reaudit", code == 0 and "FINAL_HOLDOUT_V1_FROZEN_PASS" in stdout, stdout[-1000:] if stdout else stderr[-1000:])
    code, stdout, stderr = run("env", "-u", "PROJECTB_FINAL_TEST_ENABLE", "-u", "PROJECTB_FINAL_TEST_CONFIRM", "scripts/run_final_test.sh")
    add("launcher_default_refusal", code == 2 and "FINAL_TEST_REFUSED" in (stdout + stderr), (code, (stdout + stderr).strip()))

    launcher = (REPO / "scripts/run_final_test.sh").read_text(encoding="utf-8")
    protocol = (REPO / "docs/experiment_design/final_test_protocol.md").read_text(encoding="utf-8")
    analysis = (REPO / "scripts/analyze_final_test.py").read_text(encoding="utf-8")
    resolved = (REPO / "configs/final_holdout_v1_hydra_resolved_20261003.yaml").read_text(encoding="utf-8")
    add("launcher_endpoint", "final_holdout_v1.parquet" in launcher and "TEST_FILE=" not in launcher, "new endpoint only")
    add("launcher_hash_row_id_guard", all(token in launcher for token in ("derived_parquet_sha256", "ordered_source_ids_sha256", "row count")), "manifest/hash/count/order assertions")
    add("protocol_endpoint_revision", "FINAL_HOLDOUT_V1" in protocol and "historically observed ToolRL" in protocol, "fresh internal endpoint plus historical test disclosure")
    add("analysis_dynamic_n_and_strata", "expected_n" in analysis and "response_only" in analysis and "COMPARISONS" in analysis, "dynamic coverage, 101/7 strata, six fixed comparisons")
    add("hydra_resolved_static", "final_holdout_v1.parquet" in resolved and "val_only: true" in resolved and "val_max_samples: 108" in resolved, "official veRL v0.9.1 compose/resolve output")

    ray_source = (ROOT / "upstream/verl-v0.9.1/verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    ref_source = (ROOT / "upstream/verl-v0.9.1/verl/trainer/ppo/utils.py").read_text(encoding="utf-8")
    add("val_only_source_audit", "val_before_train" in ray_source and "val_only" in ray_source and "need_reference_policy" in ref_source, "val-only return and reference-policy gate present")

    manifest = json.loads((REPO / "manifests/final_holdout_v1_manifest.json").read_text(encoding="utf-8"))
    server_manifest = PREFLIGHT / "final_holdout_v1_manifest.json"
    add("manifest_copy_equal", server_manifest.read_bytes() == (REPO / "manifests/final_holdout_v1_manifest.json").read_bytes(), "repo/server manifest bytes")
    add("manifest_frozen_shape", manifest.get("manifest_version") == "FINAL_HOLDOUT_V1" and manifest.get("row_count") == 108 and len(manifest.get("rows", [])) == 108, {"row_count": manifest.get("row_count"), "rows": len(manifest.get("rows", []))})

    process_text = subprocess.run(["ps", "-eo", "args"], text=True, capture_output=True).stdout
    bad_processes = [line for line in process_text.splitlines() if re.search(r"verl\.trainer\.main_ppo|vllm|ray::", line, re.IGNORECASE)]
    add("no_training_or_inference_process", not bad_processes, bad_processes)
    add("no_model_output", not any((ROOT / "final-test").rglob("*.jsonl")) if (ROOT / "final-test").exists() else True, "final-test JSONL absent")

    result = {
        "schema": "projectb_final_holdout_v1_static_checks",
        "status": "PASS" if all(bool(item["passed"]) for item in checks) else "FAIL",
        "checks": checks,
        "inference_performed": False,
        "reward_scoring_performed": False,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    PASS_MARKER.write_text("FINAL_HOLDOUT_V1_FROZEN_PASS\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
