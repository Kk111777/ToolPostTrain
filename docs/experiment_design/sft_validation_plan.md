# ProjectB Format SFT 验证方案

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Original SFT validation plan, preserved as design history. Original facts and values below are preserved. Current artifacts: [three-run completion](../../reports/comparison/final_stage1_ablation_completion_report.md); current budget: [35-step decision](stage1_closeout_decision.md).

本报告定义 SFT warmup 完成后的 validation gate。当前只设计，不训练、不启动 optimizer、不修改 GRPO/GDPO pipeline。

## 1. 固定验证集

为了和已经完成的 zero-shot diagnostic 可比，使用同一批 16 个 source rows，并在 SFT 抽样时排除它们：

    [2619, 456, 102, 3037, 1126, 1003, 914, 571,
     3016, 419, 2771, 3033, 3654, 2233, 356, 2418]

这批数据包含 14 条 tool-only 和 2 条 response-only。验证时保持：

- 同一 Qwen/Qwen2.5-1.5B-Instruct tokenizer 和 chat template；
- 同一 modern verl/vLLM 环境；
- 同一当前训练默认 sampling：temperature=1.0、top_p=1.0、top_k=-1；
- tensor_parallel_size=1；
- 相同 prompt、max response length 和本地模型路径；
- rollout.n=2，因此得到 32 个 response；
- Before 使用原始 base model，After 使用 format-SFT checkpoint；
- 不进入 optimizer，不运行 GRPO/GDPO。

对随机采样而言，应固定 seed；如果框架不能保证逐 token 完全复现，也要固定同一 seed、同一批 prompt 和同一 sampling config，并报告实际 seed。

## 2. 每个 response 记录

每条 response 至少记录：

- source row / sample id；
- expected kind：tool-only 或 response-only；
- prompt token length、response token length；
- generated response 原文；
- format_reward；
- accuracy_reward；
- total reward；
- <think>、<tool_call>、<response> 的开闭 tag；
- tool-call JSON parse success；
- 是否出现 Markdown fence、额外自然语言或未闭合 tag。

## 3. Before / After 对比表

主表使用同一 32 个 rollout 槽位：

| 指标 | Before base | After format SFT | 目的 |
|---|---:|---:|---|
| format_reward > 0 | 0/32 的历史基线应保留 | 待测 | 是否学会 strict wrapper |
| tool-only JSON parse rate | 待测 | 待测 | 是否能输出可解析 tool body |
| response-only wrapper rate | 待测 | 待测 | 是否覆盖另一条协议分支 |
| accuracy_reward > 0 | 待测 | 待测 | SFT 是否破坏/改善基础 tool correctness |
| total reward mean/std | 待测 | 待测 | reward 是否从稀疏/负值变得可用 |
| non-finite reward | 0 | 必须为 0 | 数值安全 |

历史 16 条单 response diagnostic 的 A setting 是 format_reward=0/16、tool JSON parse=2/16、accuracy_reward>0=1/16；之前 rollout.n=2 的 32 条报告也应作为辅助 baseline 保留，不能与 After 混用不同数据。

## 4. 推荐进入 RL 前的门槛

这些是 format warmup 的 go/no-go 建议，不是对论文结果的硬性定理。

### 必须满足

1. 32 个 response 中 strict format_reward>0 至少 26 个，即不低于 80%；理想目标是 90% 以上。
2. tool-only 子集的 tool-call JSON parse rate 至少 90%；不能只靠出现 <tool_call> 字符串而没有可解析 JSON。
3. response-only 子集必须输出完整 <think>...</think> 与 <response>...</response>；由于当前样本只有 4 个 response-only rollout，建议要求 4/4，而不是用总体比例掩盖小分母。
4. reward、format score、accuracy score 全部 finite；不得出现 NaN 或 Inf。
5. After 的 accuracy_reward 不得比 Before 明显恶化；format warmup 的目标是协议，不是牺牲工具参数正确性换标签。

### 进入多步 RL 前还应做一个无 optimizer 的 signal check

在通过上述 format gate 后，再对固定的少量 prompt 做 rollout.n=4，但仍然不训练，检查：

- 每个 prompt 的 rewards 能进入 advantage 计算；
- advantage 全部 finite；
- 全部 group reward 不是完全相同的常数；
- 至少一部分 prompt group 存在 reward variation，使 GRPO/GDPO 有可用的相对信号。

如果格式通过但所有 reward 都相同，不能直接宣布可以训练；这表示 format 学会了，但 outcome signal 仍需要单独诊断。

## 5. 失败时的判定

| 现象 | 判定 | 下一步 |
|---|---|---|
| format 仍低于 80% | warmup 没有解决协议问题 | 先检查 SFT input/assistant boundary 和 target filter，不改 reward |
| format 高但 JSON parse 低 | 学到了 tag 外壳，没学会 JSONL body | 增加 valid multi-tool target 覆盖，仍不改 parser |
| format 高但 accuracy 明显下降 | 过拟合格式或 SFT 配置过强 | 降低 warmup 强度或重新检查 target 分布 |
| format、parse 通过但 group reward 无变化 | 协议已学会，outcome signal 仍不足 | 只做 reward/advantage signal audit，不立即正式训练 |
| 出现 NaN/Inf | 环境或训练数值问题 | 停止，不进入 GRPO/GDPO |

## 6. 对 GRPO/GDPO 公平性的要求

Format SFT 改变了初始化，因此最终实验应明确分成：

1. cold-start base model：用于记录 zero-shot format failure；
2. common format-warm-start：同一个 SFT checkpoint 分别启动 GRPO 和 GDPO。

GRPO 与 GDPO 的比较必须共享：

- 相同的 SFT checkpoint；
- 相同的 dataset split；
- 相同的 rollout backend、rollout.n、prompt/response length；
- 相同的 reward manager；
- 相同的 optimizer、learning rate、step 数和 evaluation checkpoint；
- 相同或明确记录的 random seed。

这样改变的是共同 initialization，而不是 GRPO/GDPO 之间的算法变量。README/简历应把 format SFT 写成 warm-start 组成部分，并单独报告 cold-start 诊断；不能把 warm-start 结果包装成零样本结果。

## 7. 最终决策规则

- 未达到 format gate：不进入多步 GDPO，先修复 SFT 数据边界或 warmup 配置。
- 达到 format gate 且 advantage/signal check 通过：才允许安排后续 1-step re-smoke，再由用户确认是否进入正式训练。
- 当前阶段停在设计，不执行 SFT，也不改变已有 RL 代码和环境。
