# ProjectB Format SFT v2 warmup report

This report covers exactly one v2 format-SFT epoch. No reward validation, GDPO, or GRPO run was started after training.

## A. Actual training configuration

- Base model: `/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Dataset: `/root/autodl-tmp/ProjectB/repo/dataset/format_sft_800_v2.jsonl`
- Output adapter: `/root/autodl-tmp/ProjectB/checkpoints/format_sft_v2_lora`
- Epochs: `1.0`
- Per-device batch size: `4`
- Gradient accumulation steps: `4`
- Effective batch size: `16`
- Learning rate: `0.0001`
- Max sequence length: `3072`
- BF16: `True`
- Gradient checkpointing: `True`
- LoRA rank / alpha / dropout: `16` / `32` / `0.05`
- LoRA target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Seed: `42`

## B. Steps and loss

- Final optimizer step: `50`
- Trainer aggregate train loss: `0.423457`
- Train runtime: `124.7679 seconds`
- Logged loss points: `11`
- NaN/Inf detected: **False**
- First logged loss: `1.035212` at step `1`
- Last logged loss: `0.267125` at step `50`

| step | epoch | loss | learning rate |
|---:|---:|---:|---:|
| 1 | 0.0200 | 1.035212 | 0 |
| 5 | 0.1000 | 0.880715 | 9.9888344e-05 |
| 10 | 0.2000 | 0.544354 | 9.6032494e-05 |
| 15 | 0.3000 | 0.361177 | 8.7082605e-05 |
| 20 | 0.4000 | 0.354226 | 7.4029088e-05 |
| 25 | 0.5000 | 0.424668 | 5.8316468e-05 |
| 30 | 0.6000 | 0.326810 | 4.1683532e-05 |
| 35 | 0.7000 | 0.419006 | 2.5970912e-05 |
| 40 | 0.8000 | 0.333106 | 1.2917395e-05 |
| 45 | 0.9000 | 0.292489 | 3.9675057e-06 |
| 50 | 1.0000 | 0.267125 | 1.1165607e-07 |

## C. GPU memory and trainable parameters

- GPU: `NVIDIA RTX 6000D`
- Peak sampled GPU memory: `80405 MiB`
- GPU memory capacity used for percentage: `85651 MiB`
- Approximate peak percentage: `93.88%`
- Trainable parameters: `18464768`
- Total parameters: `1562179072`
- Trainable percentage: `1.1820%`
- Memory was sampled with `nvidia-smi` during training; it is a device-level peak sample, not a model-internal allocator counter.

## D. Adapter artifacts

- `adapter_model.safetensors`: `True`
- `adapter_config.json`: `True`
- Output path: `/root/autodl-tmp/ProjectB/checkpoints/format_sft_v2_lora`
- Checkpoints: `['checkpoint-50']`
- Adapter save status: **success**

## Stop condition

Training stopped after the v2 SFT run. No reward validation or GDPO was executed.
