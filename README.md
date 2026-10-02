# ToolPostTrain：工具调用的 GRPO / GDPO 复现实验

在单张 NVIDIA RTX 6000D 上，完成工具调用奖励诊断、Format SFT、GRPO/GDPO 单步训练和两算法 35-step Stage 1，记录配置、验证集、训练指标与可恢复 checkpoint。

**进度快照：2026-10-02（北京时间）。GRPO 与 GDPO Stage 1 均完成 35/35 并通过 completion gate；两算法最终汇总已归档。未启动 GRPO-noKL。**

[实验进度与证据](reports/README.md) · [GRPO Stage 1 报告](reports/grpo/stage1_35/grpo_stage1_report.md) · [GRPO/GDPO 35-step 比较](reports/comparison/grpo_gdpo_stage1_35_comparison.md) · [Stage 1 收尾报告](reports/comparison/stage1_35x2_completion_report.md) · [服务器文件审计](docs/github_audit_20261002.md) · [上游 GDPO 介绍](README.upstream.md)

## 我完成了什么

| 阶段 | 实际完成内容 | 状态与证据 |
|---|---|---|
| 官方实现审计 | 区分奖励求和后归一化与逐奖励归一化，验证 reward / advantage 纯函数和边界情况 | [CPU 检查协议](docs/reproduction_protocol.md)；历史检查 5 passed |
| Blackwell 环境 | 独立现代环境、SM120 GPU gate、vLLM 本地生成、真实 optimizer step | [环境与当前运行证据](reports/runtime_verification_20261002.md) |
| 奖励与输出契约 | 审计工具 JSON、标签格式、prompt 与 reward parser，保留失败输出 | [奖励诊断](docs/diagnostics/reward_signal_report.md) · [prompt 契约](docs/diagnostics/prompt_contract_report.md) |
| Format SFT v2 | 800 条格式样本、1 epoch、LoRA 50 optimizer steps；独立合并为共同 RL 初始化 | [SFT 报告](reports/sft/sft_v2_train_report.md) · [合并验证](reports/comparison/rl_init_merge_validation.md) |
| 训练信号诊断 | 32 个 fresh prompts × 4 rollouts；检查格式与正确性奖励的组内变化 | [holdout 诊断](reports/sft/fresh_holdout_reward_variation.md)；17/32 组格式变化、27/32 组正确性变化 |
| 两算法单步容量与吞吐 | 512 prompts × 4 rollouts，真实 rollout → reward → advantage → backward → optimizer step | [GRPO Config A](reports/grpo/grpo_throughput_config_a_report.md) · [GDPO Config A](reports/gdpo/gdpo_throughput_config_a_report.md) |
| GRPO Stage 1 | 完成 35 steps、6 次 formal validation、step35 full/model-only checkpoint 和 completion gate | **PASS**；[完成报告](reports/grpo/stage1_35/grpo_stage1_report.md) · [gate 快照](reports/grpo/stage1_35/STAGE_COMPLETE_OK) |
| GDPO Stage 1 | 从同一 RL_INIT_V1 独立 step0 启动；完成 35 steps、6 次 formal validation、step35 full/model-only 和 completion gate | **PASS**；[完成报告](reports/gdpo/stage1_35/gdpo_stage1_report.md) · [gate](reports/gdpo/stage1_35/STAGE_COMPLETE_OK) · [GRPO/GDPO 比较](reports/comparison/grpo_gdpo_stage1_35_comparison.md) |

我的工作集中在运行环境、奖励可观测性、数据隔离、实验冻结与证据整理。GRPO/GDPO 算法来自上游；本仓库保留原始源码与许可证。

## 已完成的 GRPO 验证曲线

同一 `formal_validation_80`，greedy `n=1`；accuracy reward 是工具调用正确性奖励分数，并非准确率百分比。

| step | total reward mean | format reward mean | accuracy reward mean |
|---:|---:|---:|---:|
| 0 | 2.400171 | 0.950000 | 1.450171 |
| 7 | 2.717582 | 0.950000 | 1.767582 |
| 14 | 2.799781 | 0.950000 | 1.849781 |
| 21 | 2.785379 | 0.950000 | 1.835379 |
| 28 | 2.831808 | 0.950000 | 1.881808 |
| 35 | 2.799721 | 0.950000 | 1.849721 |

step35 相比初始化的总奖励均值增加约 **0.400**，但低于 step28。这里只报告观测值，尚未据此确定延长预算，也不宣称 GDPO 优于 GRPO。[原生指标快照](reports/grpo/stage1_35/metrics_at_snapshot.json) 可核验表格；[checkpoint manifest](reports/grpo/stage1_35/checkpoint_manifest.json) 记录 full resume 与独立模型加载检查。

## 冻结的 Stage 1 设计

```text
Qwen2.5-1.5B-Instruct
  → Format SFT v2 → 合并模型 RL_INIT_V1
      ├─ GRPO：独立 step0 → 35 steps → PASS
      └─ GDPO：独立 step0 → 35 steps → PASS
```

| 项目 | 固定值 |
|---|---|
| 模型与初始化 | Qwen2.5-1.5B-Instruct；两算法使用同一 SFT v2 merged checkpoint |
| 数据 | ToolRL rlla_4k：raw 3920 → overlong filter 后 3901 |
| 训练顺序 | shuffle=false、drop_last=true；每 epoch 实际使用前 3584 条，317 条尾部不进 optimizer |
| Stage 1 预算 | batch 512，7 batches/epoch × 5 epochs = 35 steps |
| rollout / PPO | rollout.n=4；minibatch 128；physical microbatch 1；dynamic token budget 6144 |
| 序列 / 学习率 | prompt 2048 / response 1024；AdamW lr=1e-6 |
| vLLM Config A | TP=1；memory utilization=0.40；max_num_seqs=8；max_num_batched_tokens=6144 |
| 验证 | 固定 80 条尾部样本，排除 SFT 与已记录诊断来源；steps 0/7/14/21/28/35 |
| test | Stage 1 中间验证不使用 test.parquet；当前 35-step 收尾未启动最终 test 评估 |
| checkpoint | 两算法保留最终 step35 full resume 与 model-only；不自动继续 70/105 steps |

[冻结配置](configs/formal_experiment_config.yaml) · [validation 隔离审计](docs/diagnostics/formal_validation_split_audit.md) · [sampler 审计](docs/diagnostics/formal_sampler_determinism_audit.md) · [train_max_samples=-1 语义](docs/diagnostics/train_max_samples_semantics.md)

## 当前实验的边界

- 当前是 **现代 veRL + SFT 初始化 + 单卡 35-step 的分阶段复现实验**。上游原始工具调用配置使用更旧的依赖、8 GPU 和 15 epochs；不能称为论文原配置的完整复现。
- [KL 路径审计](reports/comparison/kl_path_audit.md)发现：当前 GRPO 使用 KL 调整后的总奖励，GDPO component path 使用原始分项奖励。该差异保留在实验中，尚未消融；不能将最终差异完全归因于归一化方式。
- [SFT 合并检查](reports/comparison/rl_init_merge_validation.md)中，16 个固定 prompts 的格式/解析判断一致，但只有 15/16 的输出与奖励一致；未宣称逐位或完整语义等价。正式两算法共同使用合并后的模型。
- 目前只有一组训练 seed；fresh holdout、shared rollout 和单步吞吐属于各自的诊断，不能替代同预算学习曲线与最终 test 评估。
- 早期失败环境报告、0/105 的旧启动报告和“NOT STARTED”配置说明保留为历史记录；当前进度以本页的时间戳和 Stage 1 快照为准。

## 如何审阅与复现

无需 GPU 即可阅读 `reports/`、`configs/`、`manifests/` 中的公开证据。原始模型权重、optimizer state、数据 parquet、环境和大日志保留在服务器，不上传 Git。

Mac 上只运行纯函数检查：

```bash
bash setup-local.sh
bash 运行本地验证.command
```

真实训练在 Linux/NVIDIA 服务器运行，使用 veRL v0.9.1 commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` 与独立 `.venv-modern`。[运行环境说明](reports/runtime_verification_20261002.md)记录依赖例外和早期失败。

`train_grpo_stage1.sh` / `train_gdpo_stage1.sh` 是针对现有服务器路径冻结的 launcher，依赖已准备的 RL_INIT_V1、ToolRL parquet 和 formal validation；它们不是新机器的一键重建入口，且会拒绝复用已有正常运行目录。审阅期间无需执行这些训练脚本。

```text
scripts/                 GPU 检查、SFT 数据与训练、Stage 1 launchers
configs/                 环境记录与冻结实验配置
manifests/               数据来源与 validation row manifest
reports/                 SFT、单步容量、吞吐、Stage 1 指标和报告
docs/diagnostics/        输出契约、奖励、KL、数据与 checkpoint 检查
docs/experiment_design/ SFT 与分阶段实验设计
local_checks/            Mac CPU 数值验证
verl-GDPO/               原始实现与现代 smoke driver
```

## 来源与许可证

基于 [NVlabs/GDPO](https://github.com/NVlabs/GDPO)，原始固定 commit 为 `4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115`；当前 GPU 实验使用另行固定的官方 veRL v0.9.1。上游项目说明、论文引用与图表保存在 [README.upstream.md](README.upstream.md)。原 NVIDIA Source Code License-NC 和第三方许可证见 [LICENSE](LICENSE) 与 [third_party_dependency.LICENSE](third_party_dependency.LICENSE)。
