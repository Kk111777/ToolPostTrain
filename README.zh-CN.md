# ToolPostTrain

[English](README.md) | **简体中文**

基于 **Qwen2.5-1.5B-Instruct** 的工具调用 GRPO/GDPO 后训练实验，包含优化目标路径审计、reward-side KL 对照和配对评估。

## 概述

项目基于 [NVlabs/GDPO](https://github.com/NVlabs/GDPO)、ToolRL 和 veRL，完成了 800 样本 Format SFT、三组 35-step RL 训练，以及四个固定模型在 108 条内部留出提示上的评估。

初始 GRPO/GDPO 比较存在 KL 混杂：配置启用了相同开关，构造 advantage 时却使用了不同的奖励。因此增加 **GRPO-noKL**，对齐 reward-side KL 处理。最终三个 RL checkpoint 在当前评估集上均优于统一初始化，GDPO 相对 noKL 没有清晰的额外收益。

**技术栈：** PyTorch · veRL · vLLM · LoRA · FSDP · 单张 RTX 6000D。

## 主要结果

四个冻结模型使用相同的 **108 条提示**，每条生成一次贪心答案，共 **432 条输出**。

| 模型 | 在对照中的作用 | Raw total reward | 相对 RL_INIT 的增量 |
|---|---|---:|---:|
| RL_INIT_V1 | 统一的 Format-SFT 初始化 | 2.417 | — |
| GRPO-original35 | 原始 GRPO 参照；训练中 reward-side KL 进入更新 | 2.946 | +0.529 |
| GDPO-current35 | 分奖励的 advantage 构造 | 2.956 | +0.540 |
| GRPO-noKL35 | 控制 reward-side KL 差异的参照 | 2.945 | +0.529 |

三组 RL 相对初始化的配对 95% 区间均高于 0。**GDPO − GRPO-noKL 为 +0.011**，95% 区间为 **[−0.097, +0.165]**。这一小幅均值差异不能证明 GDPO 更优，也不能证明算法等价。

数值保留三位小数。Raw total reward 为正确性奖励 + 格式奖励，不是准确率百分比。[精确结果与全部六组配对比较 →](reports/final_holdout_v1/completion_report.md)

## 实验设计

### 统一初始化

基础 Qwen checkpoint 经常无法满足 ToolRL 规定的输出结构，直接比较 RL 会混入格式遵循与任务质量的差异。核对 prompt/scorer 契约后，使用 800 条样本进行短程 Format SFT，建立统一的 **RL_INIT_V1**。LoRA checkpoint 在三组 RL 启动前完成合并。[SFT 前后诊断](reports/sft/sft_v2_after_reward_report.md) · [SFT 训练报告](reports/sft/sft_v2_train_report.md)

GRPO-original、GDPO 和 GRPO-noKL 使用相同的初始化、数据顺序、训练预算与评估日程。noKL 移除 GRPO 更新中的 reward-side KL，完整配置差异见[对照记录](docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md)。

### 相对上游设置的变化

| 环节 | 上游基础 | 当前仓库的设置 |
|---|---|---|
| 初始化 | Qwen 模型、ToolRL 数据与奖励 | Format SFT 和冻结的统一 RL_INIT_V1 |
| 目标审计 | GRPO/GDPO 实现与训练配置 | 追踪 reward → advantage → actor loss 的实际 KL 路径 |
| 对照 | GRPO/GDPO 训练方案 | 增加 GRPO-noKL，对齐 reward-side KL 处理 |
| 诊断 | Reward 与 advantage 张量 | 共享 rollout 的 advantage 比较、采样奖励冲突分析 |
| 评估 | Trainer 的 validation 支持 | 四个冻结模型、108 条内部留出提示与配对 bootstrap |

## 优化目标路径审计

初始两组均设置了 `use_kl_in_reward=true`，但这个开关不足以描述实际进入 actor 的优化目标：

```text
GRPO：KL 调整后的 token-level rewards → 组内归一化 → policy update
GDPO：原始 accuracy/format components → 分项归一化 → whitening → policy update
```

当前配置的 GDPO 分支直接从原始分项奖励构造 advantage。原始比较因此同时改变了 **advantage 构造和 KL 处理**。GRPO-noKL 提供 KL 处理对齐的参照，用于比较完整的 advantage 构造。

这一差异限定于已审计的实现与配置。GDPO 仍执行 reference-policy 计算，noKL 则随框架开关移除了这部分计算，因此两者的运行时间差不能直接衡量 estimator 效率。[源码路径与 loss 消费位置 →](docs/diagnostics/gdpo_hidden_kl_use_final_audit.md)

## 优化行为诊断

在 **8 条提示 × 4 条共享 rollouts** 上，GDPO 改变了 advantage 的尺度、中心化和样本权重，但在报告规定的容差下，没有实质性的组内排序反转。另一组采样诊断中，**8/128 条轨迹**的中心化正确性／格式奖励方向相反，涉及 **6/32 个组**。[共享 rollout 分析](reports/comparison/grpo_gdpo_shared_rollout_diagnostic.md) · [奖励冲突分析](reports/sft/fresh_holdout_reward_variation.md)

Advantage 的差异没有转化为清晰的最终分数收益。Greedy validation 的格式分数较高、诊断样本中观察到的冲突有限，可能是一个解释。这仍是 **plausible mechanism hypothesis（合理但未证明的机制假设）**，没有证明训练期格式饱和或因果机制。

## 复现中的实现问题

最终评估中有两个问题值得记录：

- **只做 validation 仍需要 actor 配置。** 首次基础设施 smoke 在模型加载前停止：launcher 缺少已完成训练所使用的 `ppo_micro_batch_size_per_gpu=1` 和 `use_dynamic_bsz=True`，未通过固定 veRL 版本的 actor schema 校验。补齐这两个字段后，配置验证通过；冻结的生成设置和零训练预算保持不变。[兼容性修复](docs/diagnostics/final_eval_actor_config_fix_audit_20261003.md)
- **旧 smoke 门禁误判了完整输出。** RL_INIT 已生成全部 108 条正式答案，validator 却仍检查 80 行和旧 mapping hash。保留已有 JSONL，修正技术检查后重新验证通过，没有重新生成答案。[失败记录](reports/final_holdout_v1/completion_report.md#preserved-failure-provenance-and-engineering-repairs)

现代单卡环境与上游归档设置不同。实际版本和容量检查保存在[运行环境记录](reports/final_holdout_v1/runtime_environment.json)及 [GRPO](reports/grpo/grpo_official_scale_capacity_report.md)／[GDPO](reports/gdpo/gdpo_official_scale_capacity_report.md) 容量报告中。

## 配置与本地检查

| 项目 | 设置 |
|---|---|
| 模型 / 初始化 | Qwen2.5-1.5B-Instruct / 合并后的 Format SFT，800 样本 |
| RL 变体 | GRPO-original / GDPO / GRPO-noKL |
| 预算 / rollouts | 每组 35 training steps；512 prompts/batch；每条 prompt 采样 4 条轨迹 |
| 学习率 / training seed | 1e-6 / 42 |
| 运行环境 | veRL + vLLM；单张 RTX 6000D |
| 最终评估 | 108 条提示 × 4 个固定模型；贪心 n=1 |

```bash
bash setup-local.sh
bash 运行本地验证.command
```

以上运行 CPU 检查。历史 GPU launchers 依赖已记录的服务器目录布局，以及单独保存的数据和模型。

`configs/` 和 `manifests/` 保存配置与身份；`scripts/` 保存运行和分析入口；`reports/` 保存结果；`docs/` 保存协议与实现审计。原始输出、数据集和权重保留在 Git 之外。

## 局限

- **一个 training seed：** 配对区间描述固定 checkpoints 的提示样本变异，不包含不同训练 seeds 的变异。
- **35-step 预算：** 这是选定的训练预算，不是收敛结论。
- **事后内部留出集：** 从依据暴露审计确认未进入 optimizer 的 train-split 尾部选取，与 validation 共享来源池。
- **任务范围：** 评估工具调用输出，没有真实工具执行或外部 benchmark。

## 报告与上游来源

- [三组训练比较](reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md)：validation 曲线与 35-step 对照。
- [最终评估报告](reports/final_holdout_v1/completion_report.md)：精确结果、行映射检查、scorer 重算与保留的执行失败。
- [结果与技术证据](reports/README.md)：配置、协议、身份、诊断和历史记录。

Qwen、ToolRL、GRPO/GDPO 实现及 veRL trainer 来自上游。当前仓库增加了上述初始化、目标审计、对照运行、诊断与评估。参见[上游 README](README.upstream.md)、[LICENSE](LICENSE) 和[第三方许可证](third_party_dependency.LICENSE)。
