# ToolPostTrain：工具调用后训练的 GRPO / GDPO 受控复现与消融

在单张 NVIDIA RTX 6000D 上，基于现代 veRL / vLLM，完成格式对齐初始化、KL 路径审计和三组独立的35步后训练实验，研究多奖励 advantage 构造差异是否转化为验证收益。

**2026-10-03（北京时间）文档状态：GRPO-original35 PASS · GDPO-current35 PASS · GRPO-noKL35 PASS。当前项目训练预算冻结为35步；FINAL_HOLDOUT_V1 已完成四模型 432/432 正式输出及冻结配对分析。**

[证据索引](reports/README.md) · [三组结果](reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md) · [预算收尾决定](docs/experiment_design/stage1_closeout_decision.md) · [final-test 协议](docs/experiment_design/final_test_protocol.md) · [实验完成记录](reports/comparison/final_stage1_ablation_completion_report.md)

## 研究问题

In format-aligned tool-calling post-training, does GDPO's multi-reward advantage construction produce a measurable benefit over a KL-aligned GRPO reference?

GRPO 对奖励求和后做组内归一化；GDPO 对各奖励维度分别归一化，再合并并做 batch masked whitening。项目比较的是这两套 advantage 构造，而不是单独识别其中一个归一化操作的因果效果。

## 为什么先做 Format SFT

基础 Qwen2.5-1.5B-Instruct 不稳定满足 ToolRL 自定义输出协议，使格式错误干扰奖励信号。Format SFT v2 使用800条训练来源样本、1 epoch、LoRA 50次更新，再合并得到共同初始化 **RL_INIT_V1**。三条 RL run 都独立从该初始化的 step0 开始。

[SFT 报告](reports/sft/sft_v2_train_report.md)与[合并检查](reports/comparison/rl_init_merge_validation.md)记录了这一步；16个检查样本的格式/解析判断一致，输出和奖励15/16一致，未宣称 adapter 与 merged 推理逐位等价。后续 sampled reward pilot 仍有组内变化：17/32组格式变化、27/32组正确性变化。

## 三组受控实验与 noKL 的作用

| run | advantage 输入与构造 | reward-side KL 进入 objective | actor KL loss | 完成状态 |
|---|---|---|---|---|
| GRPO-original | aggregate reward → group normalization | 是，配置开启 | false | 35/35 PASS |
| GDPO-current | raw accuracy / format → 分别 normalize → 合并 / whiten | 否；配置虽为 true，component path 不消费 KL 调整奖励 | false | 35/35 PASS |
| GRPO-noKL | aggregate reward → group normalization | 否，配置关闭 | false | 35/35 PASS |

[源码路径审计](docs/diagnostics/gdpo_hidden_kl_use_final_audit.md)发现，相同的 `use_kl_in_reward=true` 不保证两算法实际采用相同 KL 处理。因此补跑 GRPO-noKL，将它作为 GDPO 的目标层面 KL 对齐参照。[original/noKL 配置差异](docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md)记录了唯一主动实验语义变化：`algorithm.use_kl_in_reward: true → false`。

noKL 还按官方框架语义移除 reference-policy 计算。这是配置变化的运行后果，三组步时不能作为纯算法效率排名。原始 GRPO/GDPO 比较本身仍同时包含 KL 处理与 advantage 构造差异。

| 共同冻结项 | 设置 |
|---|---|
| 初始化 / seed | RL_INIT_V1 / 42 |
| 数据与顺序 | ToolRL rlla_4k：3920 raw → 3901 eligible；shuffle=false、drop_last=true，每 epoch 使用前3584条，317条尾部不进 optimizer |
| 预算 | 5 epochs × 7 batches = 35 optimizer steps；512 prompts/batch，rollout.n=4，PPO minibatch=128 |
| 学习率 / 长度 | lr=1e-6；prompt 2048 / response 1024；physical microbatch=1，dynamic token budget=6144 |
| vLLM Config A | TP=1；memory utilization=0.40；max_num_seqs=8；max_num_batched_tokens=6144 |
| 验证 | 固定 formal_validation_80；greedy n=1；steps 0/7/14/21/28/35 |

## Stage 1 验证结果

**以下是 validation endpoint，single training seed，n=80，尚非 final test。** 三组 step0 total 均为2.400171。accuracy reward 是工具调用正确性奖励分数，不是准确率百分比。

| run | step35 total reward | accuracy reward | format reward | total 相对 step0 |
|---|---:|---:|---:|---:|
| GRPO-original | 2.799721 | 1.849721 | 0.9500 | +0.399550 |
| GDPO-current | 2.820522 | 1.858022 | 0.9625 | +0.420351 |
| GRPO-noKL | 2.818737 | 1.856237 | 0.9625 | +0.418565 |

**Under the current 35-step single-seed setting, GDPO and the KL-aligned GRPO-noKL reference finish with nearly identical validation scores; the experiment therefore does not establish a clear downstream advantage for GDPO.**

noKL−original 的最终差值为+0.019016，归档95% paired-bootstrap 区间为[-0.125000, 0.165516]；noKL−GDPO 为−0.001786，区间为[-0.005357, 0]。这些区间描述当前固定模型在验证样本上的配对差异，不包含训练 seed 变异，也不证明算法等价或 KL 无影响。[完整曲线](reports/comparison/grpo_gdpo_nokl_stage1_35_comparison.md) · [归档 bootstrap](reports/comparison/paired_bootstrap_stage1_35.md)

验证集按 source row 与 RL/SFT 来源隔离，但 source3814 与训练/SFT source527 存在重复内容。保留原始 primary80；事后敏感性分析从同一批输出移除3814，固定规则应用于所有算法和步骤。79条最终 total 为2.809844 / 2.830909 / 2.829100，未改变当前解释；这不能证明其他内容泄漏不存在。[80/79 分析](reports/comparison/primary80_vs_sensitivity79_stage1_35.md)

## 机制诊断：数值变化与收益之间的距离

[shared-rollout](reports/comparison/grpo_gdpo_shared_rollout_diagnostic.md)在同一8 prompts × 4 rollouts 上比较 advantage，无 optimizer 更新。GDPO 改变了 magnitude、centering 和 sample weighting。原始16/32 sign disagreement 全部是零→约−0.0167，非正负反转；原始11对组内 ranking disagreement 来自约1e-9差异。在1e-8容差下，实质组内排名差异和严格排名反转均为0。原始 JSON 保留。

[fresh sampled diagnostic](reports/sft/fresh_holdout_reward_variation.md)存在 reward-dimension conflict：128条轨迹中8条、涉及6/32组，其格式与正确性奖励组内中心化后符号相反。正式训练中冲突的频率和强度尚未系统测量。

**Plausible mechanism hypothesis，未建立因果解释：** greedy validation 的格式指标接近其观测上限，验证分数剩余变化主要来自正确性奖励；冲突可能不足以让 GDPO 的机制在这个设置中形成清晰终点收益。sampled rollout 仍有格式变化，不能称训练期 format reward 已饱和。不同 advantage 与优化轨迹也可能在小型 greedy validation 上得到接近分数。

## 证据支持什么

- 三组35步真实训练完成，归档验证分数均高于共同初始化。
- KL 实际消费路径的审计促成了 noKL 消融；目标层面的 KL 对齐参照已完成。
- 公开诊断显示奖励变化、部分维度冲突及 advantage 数值/权重差异；已知重复行的79条敏感性未改变结论。

## 当前不能支持什么

- GDPO 优于 GRPO、KL 无影响、算法等价或已经完全收敛。
- 从这个 single-seed、n=80、35步设置推断更长预算或一般化效果；工具 JSON 解析也不能代替实际工具执行或完整 Agent 任务成功。
- 论文原配置的完整复现：这里采用现代 veRL、Format SFT 初始化和单卡预算。

本次文档修订核对了公开 canonical 数值和机制诊断记录；原始正式 validation JSONL 未公开，未独立重算其 bootstrap。checkpoint 清单的结构检查、model-only 加载与完整训练恢复检查是不同证据。

## 已完成：冻结35步后的内部 holdout 复核

[收尾决定](docs/experiment_design/stage1_closeout_decision.md)将35步作为当前项目训练终点；继续三组至70预计还需约30 GPU小时，且不能解决 single seed 和小验证集问题。这是预算决策，不是收敛声明。

[冻结协议](docs/experiment_design/final_test_protocol.md)已执行：RL_INIT_V1 与三条 step35 模型，各评估 FINAL_HOLDOUT_V1 的108条样本，每条1次 greedy generation，共432条。旧官方 test.parquet 已被历史加载并评分，不能称 unseen；本次使用依据可恢复暴露审计构造的 post-hoc internal holdout，不是官方 test 或外部分布 benchmark。

| 模型 | 内部 holdout raw total reward |
|---|---:|
| RL_INIT_V1 | 2.416575 |
| GRPO-original35 | 2.945898 |
| GDPO-current35 | 2.956086 |
| GRPO-noKL35 | 2.945284 |

三条 RL 模型相对 RL_INIT 的平均提升约 +0.529、+0.540、+0.529，其配对95%区间均高于0；三条 RL 模型之间的全部预定比较区间均包含0。这支持当前固定模型的 RL 增量改善在新内部样本上得到复核，未支持清晰的 GDPO 优势，也不证明算法等价或跨 seed 稳健性。

[完成报告](reports/final_holdout_v1/completion_report.md) · [完整预定比较与分层指标](reports/final_holdout_v1/metrics_and_paired_comparisons.json) · [432/432 gate 与输出 hashes](reports/final_holdout_v1/completion_gate.json)。原有 RL_INIT_retry_02 的108条答案完整保留，只修复错误技术门禁并复验；没有重生成或择优挑选。测试后不得用该 endpoint 调参、选择 checkpoint 或继续70。

## 复现与证据入口

[报告索引](reports/README.md)汇集 SFT、三组训练、KL、机制、统计和历史证据。[本次修订核验](docs/diagnostics/documentation_revision_audit_20261003.md)说明公开资料的核验范围。

Mac 上只运行 CPU 纯函数检查：

```bash
bash setup-local.sh
bash 运行本地验证.command
```

GPU 实验使用 veRL v0.9.1 commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` 与独立 `.venv-modern`。公开文件是小型报告、配置、指标与来源清单；原始正式 eval JSONL、parquet、权重和 optimizer state 保留在服务器。已有 launchers 依赖冻结服务器路径，不是新机器的一键入口，也不应用于重跑已完成目录。

## 来源与许可证

基于 [NVlabs/GDPO](https://github.com/NVlabs/GDPO)，原始固定 commit 为 `4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115`；GPU 实验另行固定官方 veRL v0.9.1。算法来自上游，个人工作集中在现代训练环境、奖励/目标路径审计、受控消融与证据整理。上游说明保存在 [README.upstream.md](README.upstream.md)；许可证见 [LICENSE](LICENSE) 与 [third_party_dependency.LICENSE](third_party_dependency.LICENSE)。
