# ProjectB modern environment report

## Final status

`ENVIRONMENT_PREP_FAIL_RUNTIME_GATE`

The official frozen dependency installation completed, but the no-card runtime
import gate cannot pass in the current container. The container cgroup has a
2 GiB memory limit and both `import verl` and `import vllm` are killed with
status 137. This is a resource limit, not a missing Python dependency.

## Source and workflow

- Repository: `https://github.com/verl-project/verl.git`
- Tag: `v0.9.1`
- Commit: `1876b06d0a3e4e71e06230be10af14492ca8a75b`
- Checkout: `/root/autodl-tmp/ProjectB/upstream/verl-v0.9.1`
- Lock validation: `uv lock --check` PASS
- Environment: `/root/autodl-tmp/ProjectB/.venv-modern`
- Legacy environment `/root/autodl-tmp/ProjectB/.conda`: untouched

The official `vllm` extra was installed with `uv sync --frozen`. The PyPI
mirror was used only as a transport endpoint; the checked-in `uv.lock` remained
the version source.

## Resolved versions

| Component | Actual value |
|---|---:|
| Python | 3.12.14 |
| torch | 2.11.0+cu130 |
| CUDA runtime reported by torch | 13.0 |
| vLLM | 0.24.0 |
| transformers | 5.9.0 |
| ray | 2.55.1 |
| numpy | 2.3.5 |
| numba | 0.65.0 |
| xFormers | absent |

250 packages were installed by the frozen sync. `uv pip check` passes after
the explicit numpy correction described below. The official uv environment
does not contain the `pip` module, so `python -m pip check` is unavailable;
installing pip was intentionally avoided.

## Explicit upstream lock exceptions

1. The `fsdp` extra was omitted. The v0.9.1 lock points to
   `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl`, but the official
   `verl-wheelhouse` release returns HTTP 404. The modern verl source still
   contains its FSDP engine; optional fsdp acceleration packages are not
   installed by this environment.
2. The lock selects `numpy==2.4.6`, while installed
   `mistral-common==1.11.3` declares `numpy<2.4` for Python 3.12. Numpy was
   corrected to official `2.3.5` with uv and `--no-deps`; no other package was
   changed.

## Import gate

Passed in isolated processes:

- `import torch`
- `import transformers`
- `import ray`

Blocked by the container cgroup:

- `import verl` → SIGKILL 137
- `import vllm` → SIGKILL 137

Observed limits:

```text
memory.max     = 2147483648
memory.current ≈ 1.8 GiB before the import
```

No CUDA probe, vLLM generation, dataset read, optimizer step, or training was
run. The GPU validation script is generated but not executed.

## Preconditions still missing for GPU validation

- Increase the container CPU-memory limit beyond 2 GiB; the validation script
  refuses to start below 8 GiB.
- Provide the ToolRL `rlla_4k/train.parquet` and `test.parquet` files. They are
  not currently present under `/root/autodl-tmp/ProjectB`.
- Complete the modern verl reward-manager/one-step smoke driver before running
  GRPO/GDPO.

## Disk

At preparation time the ProjectB filesystem had approximately 73 GiB free.
The Qwen cache exists locally; no model was downloaded by this task.
