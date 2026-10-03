# ProjectB Format SFT Warmup Runbook

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Pre-training SFT runbook; its 未启动 statements are historical, not current status. Original facts and values below are preserved. Current artifacts: [three-run completion](../../reports/comparison/final_stage1_ablation_completion_report.md); current budget: [35-step decision](stage1_closeout_decision.md).

本 runbook 只描述后续执行方式。当前阶段已经完成数据准备和脚本静态检查，但没有启动 SFT、没有进入 optimizer、没有修改环境或 reward。

## 1. 已实现的文件

远程项目目录：

    /root/autodl-tmp/ProjectB/repo

脚本：

    scripts/prepare_format_sft_data.py
    scripts/run_format_sft_lora.py

输出：

    dataset/format_sft_800.jsonl

## 2. 数据来源与实际结果

数据准备脚本只读取：

    /root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet

它没有读取 test.parquet，也没有修改任何原始 parquet。

合法 target 检查包括：

- prompt 必须是原始 system、user 两条消息；
- assistant target 必须是 strict ToolRL wrapper；
- tool-only target 的 tool body 必须是每行一个 JSON object；
- 拒绝 Markdown code block、未闭合标签、重复标签和超出 max_seq_length 的样本；
- 排除之前固定的 16 条 validation rows；
- 使用本地 tokenizer 检查完整 chat 序列长度。

固定抽样 seed=42，max_seq_length=3072，800 条分层配额为：

| sampling stratum | 数量 |
|---|---:|
| response-only | 100 |
| single-tool | 400 |
| exactly two tools | 200 |
| three or more tools | 100 |
| total | 800 |

本次实际结果：

- train source rows：3920；
- 排除 validation rows：16；
- 因 Markdown code block 拒绝：3；
- 因长度超过 3072 拒绝：4；
- 输出：800 行；
- 输出文件大小约 3.2 MiB。

每行是 chat-template 所需的 JSON object：

    {
      "id": "rlla-train-...",
      "source_row": 123,
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>...</think> ..."}
      ],
      "metadata": {
        "target_kind": "tool_only",
        "sampling_stratum": "single_tool",
        "tool_count": 1,
        "sequence_length": 958
      }
    }

## 3. 重新生成数据的命令

这一步只读取 train parquet 并覆盖新生成的 SFT JSONL，不会触碰 test 或原始数据：

    ssh autodl 'set -euo pipefail
    export HF_HOME=/root/autodl-tmp/ProjectB/hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /root/autodl-tmp/ProjectB/repo
    /root/autodl-tmp/ProjectB/.venv-modern/bin/python \
      scripts/prepare_format_sft_data.py'

如果需要变更抽样 seed 或排除列表，应把它记录到实验日志，不能静默替换 800 条样本。

## 4. LoRA SFT 配置

scripts/run_format_sft_lora.py 使用 Transformers Trainer + PEFT，不依赖 TRL，不导入 GRPO/GDPO trainer，也不调用 reward manager。

| 参数 | 当前默认值 |
|---|---|
| base model | 本地 Qwen/Qwen2.5-1.5B-Instruct snapshot |
| dtype | BF16 |
| LoRA r / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| target modules | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| learning rate | 1e-4 |
| epochs | 2 |
| per-device batch | 4 |
| gradient accumulation | 4 |
| effective batch | 16 |
| max sequence length | 3072 |
| scheduler | cosine，warmup ratio 0.05 |
| gradient checkpointing | enabled |
| quantization | none |
| output | /root/autodl-tmp/ProjectB/checkpoints/format_sft_lora |

脚本只保存 PEFT adapter、tokenizer、Trainer state 和 run config；它不覆盖 base model。

## 5. 后续训练命令

用户确认可以训练、GPU 空闲且不需要修改环境后，才执行：

    ssh autodl 'set -euo pipefail
    export HF_HOME=/root/autodl-tmp/ProjectB/hf_cache
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd /root/autodl-tmp/ProjectB/repo
    /root/autodl-tmp/ProjectB/.venv-modern/bin/python \
      scripts/run_format_sft_lora.py \
      --model-path /root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
      --data-path /root/autodl-tmp/ProjectB/repo/dataset/format_sft_800.jsonl \
      --output-dir /root/autodl-tmp/ProjectB/checkpoints/format_sft_lora'

预计显存：RTX 6000D 86 GiB 上，BF16、LoRA、max_seq_length=3072、batch=4、gradient checkpointing 的容量估计明显低于 32 GiB；必须以实际 peak memory 为准。SFT 时不要同时启动 vLLM 或正式 RL。

## 6. SFT 后的 format reward 验证

SFT 完成后不能直接启动 GRPO/GDPO。先使用此前固定的 16 rows，排除它们不参与 SFT，保持同一 current-default sampling 和 rollout.n=2，得到 32 个 response。

Before/After 必须记录：

- strict format_reward positive rate；
- tool-call JSON parse rate；
- response-only wrapper rate；
- accuracy_reward positive rate；
- total reward mean/std；
- reward 和 advantage 是否 finite；
- Markdown fence、额外文本、未闭合 tag 数量。

建议 gate：

1. format_reward>0 至少 26/32，即 80%，目标 90%；
2. tool-only JSON parse rate 至少 90%；
3. response-only 样本完整通过；
4. 无 NaN/Inf；
5. accuracy_reward 不比 Before 明显下降；
6. rollout.n=4 的无 optimizer signal check 中，group reward 不能全部为相同常数，advantage 必须 finite。

只有通过这些 gate，才安排后续 GRPO/GDPO 1-step re-smoke。GRPO 和 GDPO 必须从同一个 SFT checkpoint 开始，保证 warmup 对两者公平。

## 7. 当前停止点

- format SFT 数据：已准备完成；
- SFT runner：已远程 py_compile，并已通过 --help；
- SFT 训练：未启动；
- GRPO/GDPO：未修改、未启动；
- package、model cache、reward manager：未修改。
