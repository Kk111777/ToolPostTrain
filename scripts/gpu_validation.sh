#!/usr/bin/env bash
set -euo pipefail

# The only GPU validation entry point for ProjectB modern verl.
# This script never installs, upgrades, downgrades, or downloads anything.
# It stops at the first failed stage and never starts formal training.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/ProjectB}"
REPO_DIR="${REPO_DIR:-$PROJECT_ROOT/repo}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_ROOT/upstream/verl-v0.9.1}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-modern}"
PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
MODEL_CACHE="${MODEL_CACHE:-$PROJECT_ROOT/hf_cache}"
MODEL_PATH="${MODEL_PATH:-}"
TOOLRL_DATA_DIR="${TOOLRL_DATA_DIR:-$REPO_DIR/verl-GDPO/dataset/rlla_4k}"
SMOKE_DRIVER="${SMOKE_DRIVER:-$REPO_DIR/verl-GDPO/modern_smoke_driver.py}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/env-modern/gpu-validation-logs}"
MIN_MEMORY_BYTES="${MIN_MEMORY_BYTES:-8589934592}"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/validation.log") 2>&1

fail() {
    echo "GPU_VALIDATION_FAIL: $*" >&2
    exit 1
}

test -x "$PYTHON" || fail "modern Python not found: $PYTHON"
test -d "$UPSTREAM_DIR" || fail "upstream checkout not found: $UPSTREAM_DIR"

echo "[1/5] torch CUDA / SM120 gate"
memory_max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)"
if [[ "$memory_max" != "max" && "$memory_max" -lt "$MIN_MEMORY_BYTES" ]]; then
    fail "container memory.max=$memory_max; need at least $MIN_MEMORY_BYTES before importing verl/vllm"
fi
"$PYTHON" - <<'PY'
import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.device_count() >= 1, "no CUDA device visible"
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"gpu_name={name}")
print(f"gpu_capability={capability[0]}.{capability[1]}")
assert capability == (12, 0), f"expected SM120, got {capability}"
print(f"torch_cuda_runtime={torch.version.cuda}")
PY

echo "[2/5] vLLM local Qwen generation"
test -n "$MODEL_PATH" || MODEL_PATH="$(find "$MODEL_CACHE/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"
test -n "$MODEL_PATH" && test -f "$MODEL_PATH/config.json" || fail "local Qwen snapshot not found"
export HF_HOME="$MODEL_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset VLLM_ATTENTION_BACKEND
# vLLM 0.24.0 selects FlashInfer by default, but this container's system
# nvcc is CUDA 12.8 while the GPU is SM120. Use the native sampler until a
# CUDA >= 12.9 toolkit is available; attention backend remains auto-selected.
export VLLM_USE_FLASHINFER_SAMPLER=0
"$PYTHON" - "$MODEL_PATH" <<'PY'
import sys
from vllm import LLM, SamplingParams

model_path = sys.argv[1]
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=256, gpu_memory_utilization=0.30)
outputs = llm.generate(["Say one short sentence about tools."], SamplingParams(temperature=0.0, max_tokens=32))
text = outputs[0].outputs[0].text.strip()
assert text, "vLLM returned empty text"
print(f"generation={text}")
PY

echo "[3/5] ToolRL dataset gate"
test -f "$TOOLRL_DATA_DIR/train.parquet" || fail "missing $TOOLRL_DATA_DIR/train.parquet"
test -f "$TOOLRL_DATA_DIR/test.parquet" || fail "missing $TOOLRL_DATA_DIR/test.parquet"
"$PYTHON" - "$TOOLRL_DATA_DIR" <<'PY'
import sys
import pyarrow.parquet as pq

data_dir = sys.argv[1]
for name in ("train.parquet", "test.parquet"):
    table = pq.read_table(f"{data_dir}/{name}")
    assert table.num_rows > 0, f"{name} is empty"
    print(f"{name}_rows={table.num_rows}")
PY

echo "[4/5] GRPO one optimizer step"
test -f "$SMOKE_DRIVER" || fail "modern smoke driver not found: $SMOKE_DRIVER"
"$PYTHON" "$SMOKE_DRIVER" \
    --algorithm grpo \
    --data-dir "$TOOLRL_DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --num-prompts 2 \
    --rollout-n 2 \
    --max-prompt-length 2048 \
    --max-response-length 1024 \
    --learning-rate 1e-6 \
    --max-steps 1 \
    --require-optimizer-step

echo "[5/5] GDPO one optimizer step"
"$PYTHON" "$SMOKE_DRIVER" \
    --algorithm gdpo \
    --data-dir "$TOOLRL_DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --num-prompts 2 \
    --rollout-n 2 \
    --max-prompt-length 2048 \
    --max-response-length 1024 \
    --learning-rate 1e-6 \
    --max-steps 1 \
    --require-optimizer-step

echo "GPU_VALIDATION_PASS: all five stages completed"
