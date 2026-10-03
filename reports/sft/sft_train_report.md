# ProjectB Format SFT Warmup Training Report

> **Historical / superseded status or planning context — annotation added 2026-10-03.** SFT v1 run-completion snapshot; later SFT v2 and three RL runs completed. Original facts and values below are preserved. Current artifacts: [three-run completion](../comparison/final_stage1_ablation_completion_report.md); current budget: [35-step decision](../../docs/experiment_design/stage1_closeout_decision.md).

## A. 执行边界

- 执行位置：远程 AutoDL，RTX 6000D，SM120。
- base model：本地 Qwen2.5-1.5B-Instruct snapshot。
- SFT 数据：dataset/format_sft_800.jsonl，800 条。
- 本次使用 epoch=1，未执行原计划的 epoch=2。
- 未修改 GRPO/GDPO 代码、reward manager、源 parquet 或 Python 环境。
- 训练结束后未启动 reward validation、GRPO 或 GDPO。

训练前 checkpoint 输出目录不存在，因此本次运行没有覆盖已有 SFT 结果。

## B. 实际训练配置

| 配置项 | 实际值 |
|---|---|
| per-device train batch size | 4 |
| gradient accumulation steps | 4 |
| effective batch size | 16 |
| train samples | 800 |
| optimizer steps | 50 |
| learning rate | 1e-4 |
| scheduler | cosine |
| warmup ratio | 0.05 |
| epochs | 1 |
| max sequence length | 3072 |
| dtype | BF16 |
| gradient checkpointing | enabled |
| optimizer | adamw_torch_fused |
| seed / data seed | 42 / 42 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

trainable parameters：

    18,464,768 / 1,562,179,072
    trainable ratio = 1.1820%

## C. Loss 曲线

训练日志每 5 steps 记录一次，step 1 也记录：

| step | epoch | loss | learning rate |
|---:|---:|---:|---:|
| 1 | 0.02 | 1.0500 | 0 |
| 5 | 0.10 | 0.8555 | 9.989e-05 |
| 10 | 0.20 | 0.4800 | 9.603e-05 |
| 15 | 0.30 | 0.2400 | 8.708e-05 |
| 20 | 0.40 | 0.2902 | 7.403e-05 |
| 25 | 0.50 | 0.2377 | 5.832e-05 |
| 30 | 0.60 | 0.2638 | 4.168e-05 |
| 35 | 0.70 | 0.2413 | 2.597e-05 |
| 40 | 0.80 | 0.2457 | 1.292e-05 |
| 45 | 0.90 | 0.1961 | 3.968e-06 |
| 50 | 1.00 | 0.1439 | 1.117e-07 |

最终 Trainer 汇总：

    train_runtime = 114.7 seconds
    train_steps_per_second = 0.436
    train_samples_per_second = 6.978
    train_loss = 0.3233

loss 在本轮内总体下降，没有出现 NaN 或 Inf。

## D. GPU 显存

训练期间使用 nvidia-smi 约 1 秒间隔采样：

- peak memory used：73,351 MiB，约 71.6 GiB；
- GPU total：85,651 MiB，约 83.6 GiB；
- peak fraction：约 85.6%；
- 训练结束后：0 MiB。

这是系统级显存采样峰值，不是 torch.cuda.max_memory_allocated 的精确值。后续如果增加 sequence length、batch 或同时运行其他 GPU 服务，必须先重新做显存检查；当前配置已经不是低显存配置。

## E. Checkpoint 与 adapter 保存

主输出目录：

    /root/autodl-tmp/ProjectB/checkpoints/format_sft_lora

确认存在：

- adapter_config.json；
- adapter_model.safetensors，约 70.5 MiB；
- tokenizer.json 和 tokenizer_config.json；
- trainer_state.json；
- training_args.bin；
- checkpoint-50/；
- format_sft_run_config.json。

LoRA adapter 保存成功。训练进程退出状态为 0，并输出 FORMAT_SFT_TRAIN_PASS。

原始训练日志：

    /root/autodl-tmp/ProjectB/env-modern/sft_train_epoch1.log

## F. 当前停止点

本次只完成 format SFT warmup。当前没有执行：

- SFT 后 reward validation；
- vLLM adapter/merged-checkpoint generation；
- GRPO/GDPO 1-step；
- 正式多步训练。

下一步应先用固定的 16/32 validation rollout 对比 SFT 前后的 format_reward、tool-call parse rate 和 accuracy_reward，再决定是否进入 GRPO/GDPO。
