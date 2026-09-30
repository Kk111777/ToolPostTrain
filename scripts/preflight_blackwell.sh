#!/usr/bin/env bash
set -u

# Read-only environment audit. This script never installs packages and never
# launches training. Run it inside the rented GPU instance.

echo "=== GDPO Blackwell preflight ==="
date -u
uname -a
python3 --version 2>&1 || true

echo "--- nvidia-smi ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 || true
    nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader 2>&1 || true
else
    echo "nvidia-smi: NOT FOUND"
fi

echo "--- nvcc ---"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>&1 || true
else
    echo "nvcc: NOT FOUND"
fi

echo "--- Python packages and CUDA probes ---"
python3 - <<'PY'
import importlib
import importlib.metadata as metadata
import platform
import sys

print(f"python.executable={sys.executable}")
print(f"python.version={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")

packages = ["torch", "vllm", "transformers", "ray", "flash-attn", "xformers"]
for package in packages:
    try:
        print(f"{package}.distribution={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}.distribution=NOT INSTALLED")

try:
    import torch
    print(f"torch.__version__={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"cuda[{index}].name={props.name}")
            print(f"cuda[{index}].capability={props.major}.{props.minor}")
            print(f"cuda[{index}].total_memory_bytes={props.total_memory}")
except Exception as exc:
    print(f"torch.probe_error={type(exc).__name__}: {exc}")

for module_name in ("vllm", "transformers", "ray", "flash_attn", "xformers"):
    try:
        module = importlib.import_module(module_name)
        print(f"{module_name}.import=OK")
        print(f"{module_name}.module_version={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{module_name}.import_error={type(exc).__name__}: {exc}")
PY

echo "=== preflight complete: no training was started ==="
