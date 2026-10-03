#!/usr/bin/env python3
"""Run the CPU/no-GPU FINAL_HOLDOUT_V1 interface regression gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(os.environ.get("PROJECTB_ROOT", "/root/autodl-tmp/ProjectB"))
REPO = ROOT / "repo"
PYTHON = ROOT / ".venv-modern/bin/python"
EXPECTED_ENDPOINT = "160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16"
EXPECTED_MAPPING = "ee7ab8143f57c6c9f60fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd"
EXPECTED_PARQUET = "e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22"
EXPECTED_SOURCE_IDS = "75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = f"{ROOT}/upstream/verl-v0.9.1:{merged.get('PYTHONPATH', '')}"
    if env:
        merged.update(env)
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True, env=merged)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    endpoint_path = REPO / "manifests/final_holdout_v1_manifest.json"
    mapping_path = REPO / "manifests/final_holdout_v1_runtime_mapping.json"
    parquet_path = ROOT / "env-modern/final_holdout_v1.parquet"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    add("A_endpoint_manifest_byte_identity", sha256_file(endpoint_path) == EXPECTED_ENDPOINT, sha256_file(endpoint_path))
    add("B_endpoint_parquet_sha", sha256_file(parquet_path) == EXPECTED_PARQUET, sha256_file(parquet_path))
    add("C_ordered_source_id_sha", endpoint.get("ordered_source_ids_sha256") == EXPECTED_SOURCE_IDS, endpoint.get("ordered_source_ids_sha256"))
    pairs = {(row["runtime_input_sha256"], row["ground_truth_sha256_exact"]) for row in mapping["rows"]}
    add("D_runtime_mapping_108_unique", len(mapping.get("rows", [])) == 108 and len(pairs) == 108, len(pairs))

    audit = REPO / "docs/diagnostics/final_eval_runtime_mapping_audit.json"
    audit_result = json.loads(audit.read_text(encoding="utf-8")) if audit.is_file() else {}
    add("E_real_trainer_jsonl_mapping", audit_result.get("pass") is True, audit_result.get("checks"))

    code, stdout, stderr = run(str(PYTHON), "scripts/analyze_final_test.py", "--self-test")
    add("F_analysis_synthetic_self_test", code == 0 and "ANALYSIS_SELF_TEST_PASS" in stdout, stdout.strip() or stderr.strip())
    add("G_formal_validation_mapping_regression", audit_result.get("pass") is True, "same real 80-row JSONL audit as E")
    with tempfile.TemporaryDirectory(prefix="projectb-formal-analyzer-") as temp:
        temp_path = Path(temp)
        audit_spec = importlib.util.spec_from_file_location(
            "projectb_mapping_audit", REPO / "scripts/audit_final_eval_runtime_mapping.py"
        )
        audit_module = importlib.util.module_from_spec(audit_spec)
        assert audit_spec and audit_spec.loader
        audit_spec.loader.exec_module(audit_module)
        fixture_rows = []
        formal_source_rows = pq.read_table(ROOT / "env-modern/formal_validation_80.parquet").to_pylist()
        for row, source_row in zip(audit_module.build_formal_runtime_mapping(ROOT), formal_source_rows):
            ground_truth = str((source_row.get("reward_model") or {}).get("ground_truth", ""))
            if "<tool_call>" in ground_truth and "<response>" in ground_truth:
                target_category = "tool_and_response"
            elif "<tool_call>" in ground_truth:
                target_category = "tool_only"
            elif "<response>" in ground_truth:
                target_category = "response_only"
            else:
                target_category = "other"
            fixture_rows.append(
                {
                    "row_position": row["row_position"],
                    "source_id": row["source_id"],
                    "target_category": target_category,
                    "prompt_sha256_exact": row["runtime_input_sha256"],
                    "ground_truth_sha256_exact": row["ground_truth_sha256_exact"],
                }
            )
        fixture_manifest = temp_path / "formal_runtime_fixture.json"
        fixture_manifest.write_text(
            json.dumps({"manifest_version": "FORMAL_RUNTIME_FIXTURE", "row_count": 80, "rows": fixture_rows}),
            encoding="utf-8",
        )
        fixture_output = temp_path / "formal_runtime_analysis.json"
        code, stdout, stderr = run(
            str(PYTHON),
            "scripts/analyze_final_test.py",
            "--fixture-mode",
            "--manifest",
            str(fixture_manifest),
            "--step",
            "0",
            "--out-json",
            str(fixture_output),
            "--model-output",
            f"GRPO-original35={ROOT}/runs/grpo_stage1_35/eval_generations/0.jsonl",
            "--model-output",
            f"GDPO-current35={ROOT}/runs/gdpo_stage1_35/eval_generations/0.jsonl",
            "--model-output",
            f"GRPO-noKL35={ROOT}/runs/grpo_nokl_stage1_35/eval_generations/0.jsonl",
        )
        fixture_result = json.loads(fixture_output.read_text(encoding="utf-8")) if fixture_output.is_file() else {}
        add(
            "G_formal_validation_analyzer_regression",
            code == 0 and fixture_result.get("row_count") == 80,
            stdout.strip() or stderr.strip(),
        )

    spec = importlib.util.spec_from_file_location("projectb_final_analysis", REPO / "scripts/analyze_final_test.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    primary, primary_sha = module.make_bootstrap_indices(108)
    tool, tool_sha = module.make_bootstrap_indices(101)
    add("H_common_primary_bootstrap", primary.shape == (10000, 108), {"shape": list(primary.shape), "sha256": primary_sha})
    add("I_common_tool_bootstrap", tool.shape == (10000, 101), {"shape": list(tool.shape), "sha256": tool_sha})

    with tempfile.TemporaryDirectory(prefix="projectb-final-eval-gate-") as temp:
        temp_path = Path(temp)
        out = temp_path / "out.json"
        code, stdout, stderr = run(
            str(PYTHON), "scripts/analyze_final_test.py", "--out-json", str(out),
            "--model-output", "RL_INIT_V1=/dev/null",
        )
        add("J_missing_model_formal_refused", code != 0 and "FINAL_ANALYSIS_REFUSED_INCOMPLETE_MATRIX" in stderr, stderr.strip())
        code, stdout, stderr = run(
            str(PYTHON), "scripts/analyze_final_test.py", "--out-json", str(out),
            "--model-output", "RL_INIT_V1=/dev/null", "--model-output", "RL_INIT_V1=/dev/null",
        )
        add("K_duplicate_model_formal_refused", code != 0 and "duplicate model name" in stderr, stderr.strip())

        endpoint_tampered = temp_path / "endpoint.json"
        endpoint_tampered.write_bytes(endpoint_path.read_bytes() + b"\n")
        full_names = ["RL_INIT_V1", "GRPO-original35", "GDPO-current35", "GRPO-noKL35"]
        full_args = []
        for name in full_names:
            full_args.extend(["--model-output", f"{name}=/dev/null"])
        code, stdout, stderr = run(
            str(PYTHON), "scripts/analyze_final_test.py", "--endpoint-manifest", str(endpoint_tampered),
            "--out-json", str(out), *full_args,
        )
        add("L_endpoint_manifest_tamper_refused", code != 0 and "ENDPOINT_MANIFEST_IDENTITY" in stderr, stderr.strip())

        mapping_tampered = temp_path / "mapping.json"
        mapping_tampered.write_bytes(mapping_path.read_bytes() + b"\n")
        code, stdout, stderr = run(
            str(PYTHON), "scripts/analyze_final_test.py", "--runtime-mapping", str(mapping_tampered),
            "--out-json", str(out), *full_args,
        )
        add("M_runtime_mapping_tamper_refused", code != 0 and "RUNTIME_MAPPING_IDENTITY" in stderr, stderr.strip())

        launcher_source = (REPO / "scripts/run_final_test.sh").read_text(encoding="utf-8")
        launch_root = ROOT / "final-test"
        launch_out = launch_root / ".interface-tamper-regression"
        if launch_out.exists():
            shutil.rmtree(launch_out)
        tampered_launcher = temp_path / "run_final_test_tampered.sh"
        tampered_endpoint = temp_path / "launcher_endpoint_manifest.json"
        tampered_endpoint.write_bytes(endpoint_path.read_bytes() + b"\n")
        tampered_launcher.write_text(
            launcher_source.replace(
                'HOLDOUT_MANIFEST="$ROOT/repo/manifests/final_holdout_v1_manifest.json"',
                f'HOLDOUT_MANIFEST="{tampered_endpoint}"',
            ),
            encoding="utf-8",
        )
        code, stdout, stderr = run(
            "bash", str(tampered_launcher), "RL_INIT_V1", str(ROOT / "checkpoints/rl_init_sft_v2_merged"), str(launch_out),
            env={"PROJECTB_FINAL_TEST_ENABLE": "1", "PROJECTB_FINAL_TEST_CONFIRM": "I_HAVE_REVIEWED_PROVENANCE"},
        )
        add("L_launcher_endpoint_manifest_tamper_refused", code != 0 and "endpoint manifest byte identity mismatch" in stderr, stderr.strip())
        if launch_out.exists():
            shutil.rmtree(launch_out)

        tampered_mapping = temp_path / "launcher_runtime_mapping.json"
        tampered_mapping.write_bytes(mapping_path.read_bytes() + b"\n")
        tampered_launcher.write_text(
            launcher_source.replace(
                'RUNTIME_MAPPING="$ROOT/repo/manifests/final_holdout_v1_runtime_mapping.json"',
                f'RUNTIME_MAPPING="{tampered_mapping}"',
            ),
            encoding="utf-8",
        )
        code, stdout, stderr = run(
            "bash", str(tampered_launcher), "RL_INIT_V1", str(ROOT / "checkpoints/rl_init_sft_v2_merged"), str(launch_out),
            env={"PROJECTB_FINAL_TEST_ENABLE": "1", "PROJECTB_FINAL_TEST_CONFIRM": "I_HAVE_REVIEWED_PROVENANCE"},
        )
        add("M_launcher_runtime_mapping_tamper_refused", code != 0 and "runtime mapping byte identity mismatch" in stderr, stderr.strip())
        if launch_out.exists():
            shutil.rmtree(launch_out)

    compile_targets = ["scripts/analyze_final_test.py", "scripts/build_final_holdout_runtime_mapping.py", "scripts/audit_final_eval_runtime_mapping.py", "scripts/final_eval_interface_static_gate.py"]
    code, stdout, stderr = run(str(PYTHON), "-m", "py_compile", *compile_targets)
    add("N_python_and_bash_compile", code == 0, stderr.strip())
    code, stdout, stderr = run("bash", "-n", "scripts/run_final_test.sh")
    checks[-1]["detail"] = {"python": checks[-1]["detail"], "bash": stderr.strip()}
    checks[-1]["passed"] = bool(checks[-1]["passed"] and code == 0)

    hydra_output = temp_hydra = REPO / "docs/diagnostics/.tmp_final_holdout_hydra_resolved.yaml"
    code, stdout, stderr = run(str(PYTHON), "scripts/resolve_final_holdout_v1.py", "--output", str(hydra_output))
    hydra_text = hydra_output.read_text(encoding="utf-8") if hydra_output.is_file() else ""
    if hydra_output.exists():
        hydra_output.unlink()
    add("O_hydra_compose_resolve", code == 0 and "FINAL_HOLDOUT_V1_HYDRA_COMPOSE_RESOLVE_PASS" in stdout and "val_only: true" in hydra_text, stdout.strip() or stderr.strip())

    ray_source = (ROOT / "upstream/verl-v0.9.1/verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    utils_source = (ROOT / "upstream/verl-v0.9.1/verl/trainer/ppo/utils.py").read_text(encoding="utf-8")
    add("P_val_only_source_flow", "val_only" in ray_source and "val_before_train" in ray_source and "need_reference_policy" in utils_source, "source audit")

    result = {
        "schema": "projectb_final_eval_interface_static_checks_v1",
        "status": "PASS" if all(item["passed"] for item in checks) else "BLOCKED",
        "checks": checks,
        "endpoint_manifest_sha256": sha256_file(endpoint_path),
        "runtime_mapping_sha256": sha256_file(mapping_path),
        "primary_bootstrap_index_sha256": primary_sha,
        "tool_bootstrap_index_sha256": tool_sha,
        "no_model_inference": True,
        "no_final_holdout_scoring": True,
    }
    output = REPO / "docs/diagnostics/final_eval_interface_fix_static_checks.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
