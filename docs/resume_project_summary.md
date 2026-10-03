# ProjectB：简历与面试准备

项目定位：**LLM RL Post-Training Reproduction, Objective-Path Audit and Controlled Ablation**。

本文件供个人准备使用。事实以 [项目首页](../README.md)、[最终内部 holdout 报告](../reports/final_holdout_v1/completion_report.md)和[KL 路径审计](diagnostics/gdpo_hidden_kl_use_final_audit.md)为准。

## A. 一句话项目介绍

基于 veRL/vLLM 完成 Qwen2.5-1.5B 工具调用 GRPO/GDPO 后训练复现，通过目标路径审计发现 KL confound，补充 matched noKL 消融并完成冻结内部 holdout 配对评估。

## B. 两条简历 bullet

- 基于 veRL/vLLM 在单张 RTX 6000D 上复现 Qwen2.5-1.5B 工具调用 GRPO/GDPO 后训练；设计 800 样本、50 次 LoRA 更新的 Format SFT 统一初始化，完成三组各 35-step 的受控 RL 实验。
- 审计 reward → advantage → optimizer 路径，识别 GRPO/GDPO 的 reward-side KL 消费差异并设计 GRPO-noKL 对照；在 108 条冻结内部 holdout 上完成 432 条正式输出与 10,000 次 paired bootstrap，三组 RL 相对初始化 raw reward 增量约 +0.53，未观察到清晰的 GDPO 优势。

## C. 三条简历 bullet

- 适配 Blackwell/SM120 单卡 veRL/vLLM 后训练环境，使用 FSDP、SDPA 与动态 token batching；通过 800 样本 Format SFT 构建共享 RL_INIT_V1，完成 GRPO-original、GDPO 与 GRPO-noKL 三组 35-step 实验。
- 从 reward tensor、component extras、advantage dispatch 到 actor policy loss 追踪实际目标，发现相同 KL 开关对应不同奖励消费路径；补充 matched noKL control，并用 32 条 shared rollouts 与 128 条 sampled trajectories 诊断 advantage 行为和奖励冲突。
- 冻结 4 个模型与 108 条内部 holdout，完成 432 条 greedy 输出、108/108 唯一 mapping 和零误差 scorer 复核；paired bootstrap 中三组 RL−RL_INIT 区间均高于 0，GDPO−noKL 为 +0.010802、95% CI [−0.096605, +0.165432]，未外推算法优越性。

## D. 30 秒面试介绍

我做的是工具调用模型的 RL 后训练复现与受控消融。先用 Format SFT 建立共同初始化，再完成三组 35-step GRPO/GDPO 实验。重点是我追踪源码发现：相同 KL 配置并不意味着相同优化目标，所以增加了 GRPO-noKL 对照。最终在 108 条内部 holdout 上完成 432 条输出；三个 RL 模型都比初始化高约 0.53 reward，但 GDPO 相对 noKL 的配对区间跨 0，没有宣称算法更优。

## E. 两分钟项目介绍

这个项目研究：工具调用任务里，GDPO 的分奖励 advantage 构造，是否比 KL 对齐的 GRPO 参照有可测收益。算法来自上游，我主要负责现代训练环境适配、实验控制和目标路径审计。

开始时，Qwen2.5-1.5B-Instruct 不稳定满足 ToolRL 的严格输出契约。我先检查 prompt、chat template、tokenizer 与 scorer，确认格式指令没有在输入链路丢失；再用 800 条样本训练一轮 LoRA，共 50 次更新，合并为 RL_INIT_V1。三条正式 RL run 都从这个初始化重新开始，使用同一数据顺序、512 prompts/batch、4 rollouts、1e-6 学习率、seed 42 和 35-step 预算。

最关键的发现来自 reward 到 optimizer 的源码追踪。GRPO advantage 使用 KL-adjusted token_level_rewards；GDPO 在配置了 accuracy/format reward keys 后直接读 raw component rewards，分别归一化再做 masked whitening。两边虽然都开了 reward-side KL，实际目标却不一致。我因此补充 GRPO-noKL，作为 GDPO 的目标层面 KL 对齐参照，同时保留 original run。

诊断中，GDPO 确实改变 advantage 的幅度、中心化和样本权重；shared rollout 没有实质组内排名反转。另一个 sampled pilot 有奖励冲突，但不能代表正式训练冲突频率。最终冻结四个模型，在 108 条内部 holdout 上完成 432 条 greedy 输出和统一 paired bootstrap。三组 RL 相对初始化的 reward 增量约 +0.53，区间均高于 0；GDPO 相对 noKL 只有约 +0.011，区间跨 0。

我把结果理解为：真实后训练带来了这个内部 endpoint 上的增量，而经过 KL 控制后，当前规模没有给出 GDPO 的清晰优势。它是单 seed、35-step 的内部工具调用评分实验，没有证明等价、跨 seed 稳健性或真实 Agent 执行能力。

## F. 十个面试追问

1. 为什么先做 Format SFT？如何排除 prompt/chat-template 链路丢失格式指令？
2. runtime-LoRA 与 merged RL_INIT 的检查说明了什么？为什么不能声称逐位等价？
3. GRPO 和 GDPO 怎样构造 advantage？组内归一化与 masked batch whitening 分别改变什么？
4. `use_kl_in_reward` 和 `actor.use_kl_loss` 有什么区别？KL-adjusted reward 到底被谁消费？
5. 为什么 GRPO-original vs GDPO 存在 confound？noKL 对照控制了什么，仍没控制什么？
6. 512 prompts × 4 rollouts、PPO minibatch、physical microbatch 与动态 token budget 各表示什么？
7. 奖励维度冲突怎么定义？shared-rollout 的零碰撞与 fresh diagnostic 的 8/128 冲突为什么可以同时成立？
8. 为什么 advantage 数值明显不同，却没有实质组内排序反转或清晰终点收益？哪些解释只是 hypothesis？
9. paired bootstrap 如何保持 prompt 配对？CI 跨 0 为什么不等于算法等价，CI 高于 0 为什么不是一般算法优越性？
10. FINAL_HOLDOUT_V1 如何冻结？它与正式 validation、历史官方 test、真实 Agent benchmark 有什么区别？

## 表述边界

- reward 增量写 **+0.53 raw reward**；accuracy reward 不是准确率，不能转成“准确率提升 53%”。
- 将“区间高于 0”限定为当前固定模型、内部 endpoint 的 paired-bootstrap 结果，不包装成一般算法显著性检验。
- GDPO/noKL 的比较包含 per-dimension normalization 和 batch whitening；不能把收益或无收益归因于单个操作。
- 35-step 是预算终点，不是收敛结论；一个 training seed 不支持多 seed 稳健性。
- 算法、基础模型和任务数据归功于上游；个人贡献是环境适配、初始化、审计、对照、诊断和评估。
