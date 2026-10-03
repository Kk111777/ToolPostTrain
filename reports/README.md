# 实验结果与证据索引

文档更新：2026-10-03（北京时间）。GRPO-original35、GDPO-current35、GRPO-noKL35 均 PASS。当前项目训练预算冻结35步；final test 未执行。历史快照按各自采集时间解读。

## 当前状态与结果

| 审阅问题 | 主要证据 |
|---|---|
| 当前研究问题、结果与边界 | [首页](../README.md) |
| 三组35步完成及保留的产物 | [最终消融收尾报告](comparison/final_stage1_ablation_completion_report.md) |
| 三组共同验证曲线 | [三组比较](comparison/grpo_gdpo_nokl_stage1_35_comparison.md) · [JSON](comparison/grpo_gdpo_nokl_stage1_35_comparison.json) |
| 原始 GRPO 是否完成 | [报告](grpo/stage1_35/grpo_stage1_report.md) · [gate](grpo/stage1_35/STAGE_COMPLETE_OK) |
| GDPO 是否完成 | [报告](gdpo/stage1_35/gdpo_stage1_report.md) · [gate](gdpo/stage1_35/STAGE_COMPLETE_OK) |
| noKL 是否完成 | [报告](grpo_nokl/stage1_35/grpo_nokl_stage1_report.md) · [gate](grpo_nokl/stage1_35/STAGE_COMPLETE_OK) · [结构/加载清单](grpo_nokl/stage1_35/checkpoint_manifest_verified.json) |
| canonical validation 数值 | [original](grpo/grpo_stage1_metrics_canonical.json) · [GDPO](gdpo/gdpo_stage1_metrics_canonical.json) · [noKL](grpo/grpo_nokl_stage1_metrics_canonical.json) |
| 为什么停止35步、是否收敛 | [预算决定](../docs/experiment_design/stage1_closeout_decision.md)：停止追加训练不等于证明收敛 |
| final test 如何做 | [协议](../docs/experiment_design/final_test_protocol.md)：共同初始化+三条step35，当前未执行 |

## KL、机制与统计解释

| 审阅问题 | 主要证据 |
|---|---|
| GDPO 是否隐藏消费 KL 调整奖励 | [最终路径审计](../docs/diagnostics/gdpo_hidden_kl_use_final_audit.md) |
| noKL 主动变了哪些配置 | [配置差异](../docs/diagnostics/grpo_original_vs_grpo_nokl_effective_config_diff.md) |
| reward-side KL 的消融结果 | [original vs noKL](comparison/grpo_original_vs_nokl_stage1_35.md) |
| KL 对齐后 advantage 方案的结果 | [noKL vs GDPO](comparison/grpo_nokl_vs_gdpo_stage1_35.md) |
| 原始两算法比较的混杂边界 | [original vs GDPO](comparison/grpo_original_vs_gdpo_stage1_35.md) |
| 配对不确定性 | [bootstrap](comparison/paired_bootstrap_stage1_35.md) · [JSON](comparison/paired_bootstrap_stage1_35.json)，不覆盖训练seed变异 |
| 已知重复内容怎样处理 | [80/79](comparison/primary80_vs_sensitivity79_stage1_35.md) · [事后分析计划](comparison/grpo_gdpo_primary_sensitivity_analysis_plan.md) |
| advantage 数值变化是否代表排序反转 | [shared-rollout勘误](comparison/grpo_gdpo_shared_rollout_diagnostic.md)：1e-8容差下实质组内排名差异0 |
| SFT 后是否仍有奖励变化/冲突 | [fresh sampled diagnostic](sft/fresh_holdout_reward_variation.md)：8条冲突轨迹，涉及6/32组 |
| 本次独立复核到了哪一层 | [文档修订核验](../docs/diagnostics/documentation_revision_audit_20261003.md) |

正式 validation JSONL 未公开。归档 bootstrap 来自服务器持久化输出，本次文档修订未独立重算它。公开 shared-rollout / fresh-holdout JSON 含逐轨迹记录，可在 CPU 上复核本次机制勘误。正式 Stage 1 不用 test 做中间验证；整个项目历史的 test 使用需按 final-test 协议审计。

## 初始化、实现与历史证据

- [SFT v2](sft/sft_v2_train_report.md) · [merge 检查](comparison/rl_init_merge_validation.md) · [奖励诊断](../docs/diagnostics/reward_signal_report.md) · [prompt 契约](../docs/diagnostics/prompt_contract_report.md)。
- 真实容量与训练路径：[GRPO](grpo/grpo_official_scale_capacity_report.md) / [GDPO](gdpo/gdpo_official_scale_capacity_report.md)；Config A：[GRPO](grpo/grpo_throughput_config_a_report.md) / [GDPO](gdpo/gdpo_throughput_config_a_report.md)。
- [旧两组收尾](comparison/stage1_35x2_completion_report.md)、[初始 KL 审计](comparison/kl_path_audit.md)、[环境快照](runtime_verification_20261002.md)、[服务器来源清单](server_artifact_manifest.json)是带时间边界的历史记录，不是三组最终状态总表。
- [GRPO gate 失败与解决](grpo/stage1_35/STAGE_FAILED.resolution.md)、[旧0/105启动失败](grpo/historical_grpo_formal_run_report.md)、[初始环境失败](../configs/environment_report.md)保留，不回写历史事实。
- GRPO 原始报告的 `step_time=False` 是检查键与日志键不一致；原生35条 `perf/time_per_step` 有限。原始报告保留，当前数值见 canonical 与三组比较。
- full checkpoint 结构存在、model-only 独立加载、完整 optimizer/RNG/dataloader 恢复是不同检查；[旧 resume smoke](../docs/diagnostics/checkpoint_resume_smoke.md)不自动证明三条最终step35完成了完整恢复检查。
