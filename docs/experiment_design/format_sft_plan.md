# ProjectB 最小 Format SFT Warmup 方案

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Original SFT proposal; actual frozen initialization is Format SFT v2 / RL_INIT_V1. Original facts and values below are preserved. Current artifacts: [three-run completion](../../reports/comparison/final_stage1_ablation_completion_report.md); current budget: [35-step decision](stage1_closeout_decision.md).

本方案只设计 format-following warmup，不执行训练，不改变现有 GRPO/GDPO pipeline，不修改 reward manager、parquet dataset 或 modern Python 环境。

## 1. 目标与边界

目标只有一个：让 Qwen/Qwen2.5-1.5B-Instruct 在 ToolRL prompt 下稳定生成当前 reward parser 能识别的结构：

    <think>
    ...
    </think>
    <tool_call>
    one JSON object per line
    </tool_call>

或：

    <think>
    ...
    </think>
    <response>
    ...
    </response>

不把这个 warmup 定义成 tool-use 能力训练、数学能力训练、reward 逻辑训练或正式 ToolRL 训练。

## 2. 推荐的最小数据集

推荐第一版使用 800 条，允许范围为 500--1000 条：

- 约 700 条 tool-only；
- 约 100 条 response-only；
- 在 tool-only 内保留 single-tool、two-tool 和 multi-tool；
- 从 train.parquet 选择，不使用 test.parquet 的 80 条；
- 从 SFT 候选池排除当前固定的 16 条 diagnostic rows，避免 Before/After 泄漏；
- 保持每条原始 prompt 的 system、user 内容不变；
- target 直接取 reward_model.ground_truth；
- 不把 target 改写成 Markdown code block，不插入额外解释，不增加新的标签协议。

800 不是算法超参数，而是第一轮 format gate 的工作集大小。它足以覆盖两种输出分支和多 tool-call 长度，同时仍然能快速验证 warmup 是否有效。

### SFT 样本的逻辑结构

每条样本应表达为：

    messages = row["prompt"]
    assistant_content = row["reward_model"]["ground_truth"]

训练序列由 Qwen chat template 负责添加 system/user/assistant 边界；ToolRL 标签保留在 assistant_content 内部。不能把完整 chat-template 渲染文本再次作为普通 user 文本喂入，否则会重复 ChatML 边界。

## 3. 三种数据来源方案比较

| 方案 | 做法 | 优点 | 风险 | 结论 |
|---|---|---|---|---|
| a. 原始 ToolRL chosen | 使用 reward_model.ground_truth；它与 extra_info.output 3920/3920 完全相同 | 标签和 tool schema 与现有 reward manager 一致，最接近真实实验 | 原始分布可能偏向 tool-only | 第一选择 |
| b. 自动过滤 | 用 strict wrapper、标签计数、tool JSONL 解析、长度和重复检查过滤，再分层抽样 | 可复现、没有人工主观标签、能防止坏 target 进入 warmup | 过滤规则必须与 rlla.py 一致 | 必须作为数据质量 gate |
| c. 人工/规则构造 | 手工补充 response-only、多 tool 或极少数边界样本 | 能补齐稀有形状 | 容易产生错误参数、虚构 tool schema 或另一套换行格式 | 初版不需要；只有覆盖缺口时少量补充 |

当前审计显示：3920 条 ground_truth 已全部通过 strict structure；3447 条 tool body 的每个 JSON line 均可解析。因此第一版不需要新增 synthetic target，使用 a+b 即可。

## 4. 自动过滤 contract

候选 target 至少满足：

1. 非空字符串；
2. 恰好一个 <think> 和一个 </think>；
3. 对 tool-only：恰好一个 <tool_call> 和一个 </tool_call>，wrapper 之间每行是 JSON object；
4. 对 response-only：恰好一个 <response> 和一个 </response>；
5. 与当前 rlla.py 的 anchored newline pattern 相容；
6. 没有 Markdown fence、前置/后置自然语言或未闭合 tag；
7. full chat sequence 在 SFT max sequence length 内，或被显式列为 outlier。

这些检查只用于未来构造 SFT 子集；本阶段没有执行过滤和写出数据文件。

## 5. token 长度与 max sequence length

使用当前本地 Qwen tokenizer 对 3920 条 prompt + chosen response 做了只读估计：

| 序列 | P50 | P95 | 最大值 |
|---|---:|---:|---:|
| prompt 加 generation boundary | 698 | 1354 | 4293 |
| assistant target | 73 | 185 | 706 |
| 完整 chat 序列 | 782 | 1504 | 4533 |

完整序列在 3072 tokens 内的比例为 3916/3920，即 99.9%。因此推荐 max_seq_length=3072，与当前 vLLM max_model_len 及原始 prompt/response 预算保持一致；4 个极长 outlier 不纳入第一版 800 条子集。若以后必须覆盖全部 3920 条，再单独评估 4096，而不是现在改变 RL 配置。

## 6. 推荐 LoRA SFT 配置

这是建议配置，不执行。

| 项目 | 建议值 |
|---|---|
| base model | Qwen/Qwen2.5-1.5B-Instruct；使用当前已下载的本地 snapshot |
| dtype | bfloat16 |
| quantization | 不使用；避免引入新的量化依赖 |
| LoRA rank | r=16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| bias | none |
| learning rate | 1e-4 |
| scheduler | cosine；warmup ratio 0.05 |
| per-device train batch | 4 |
| gradient accumulation | 4；effective batch size=16 |
| epochs | 2 |
| max sequence length | 3072 |
| packing | false；保持每条 ToolRL prompt/target 的边界 |
| gradient checkpointing | true |
| use_cache | false during training |
| seed | 42 |

### 为什么使用这一档配置

- r=16 足够表达标签和结构偏好变化，避免为了一个格式目标使用过大的 adapter。
- 同时覆盖 attention 和 MLP，使模型可以学习 wrapper、换行以及 JSON 序列的联合生成；不需要额外 reward shaping。
- 1e-4、2 epochs、effective batch 16 是对 500--1000 条 format-only 数据的保守起点，避免把模型推离原始 Instruct 能力。
- RTX 6000D 有 86 GiB 显存。以 3072、BF16、batch 4、gradient checkpointing 估计，峰值应明显低于 32 GiB；这是容量估计，不是已执行的 GPU 结果。
- 不需要 xFormers、量化或额外的 transitive dependency；本阶段不安装任何包。

## 7. 与现有 GRPO/GDPO 的关系

未来若执行 warmup，推荐将结果作为唯一共同初始化：

    Qwen/Qwen2.5-1.5B-Instruct
        ↓
    format-only LoRA SFT
        ↓
    fixed SFT checkpoint
        ├── GRPO
        └── GDPO
            ↓
        ToolRL evaluation

为保持现有 pipeline 不变，未来可以把 LoRA 合并成普通 Qwen checkpoint，再只改变 model checkpoint path；或者让现有 loader 直接读取 adapter，二者必须在 GRPO 与 GDPO 两条实验中保持一致。当前不执行合并，也不改变配置。

## 8. 公平比较与实验记录

Format SFT 会改变 policy initialization，所以它不是“没有变化的预处理”。README/简历中应明确写成：

“在相同的 Qwen2.5-1.5B-Instruct 初始化上，先使用 500--1000 条 ToolRL chosen responses 做 format-only LoRA warmup；GRPO 与 GDPO 从同一个 SFT checkpoint 开始，保持 reward manager、rollout、数据、训练步数和评估协议一致。”

应保留两个结果标签：

- cold-start：原始 Qwen 的 format-following 诊断，作为失败基线；
- format-warm-start：共同 SFT checkpoint 上的 GRPO/GDPO 对比。

不能把 warmup 后的结果直接描述成“原始零样本 GDPO 结果”，也不能把 format SFT 的收益归因给 GDPO estimator。

## 9. 当前阶段结论

推荐加入 format SFT warmup，但现在只保留本方案和验证门槛，不训练、不构造新 parquet、不修改现有 RL pipeline。
