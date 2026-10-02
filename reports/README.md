# 实验进度与证据索引

快照：2026-10-02（北京时间，Stage 1 收尾）。本目录保存已完成实验报告、机器可读指标和审计快照。

| 问题 | 证据 |
|---|---|
| 当前 Stage 1 状态是什么？ | [首页](../README.md)；GRPO 与 GDPO 均 35/35 PASS；[共同收尾报告](comparison/stage1_35x2_completion_report.md) |
| GDPO 35 steps 完成了吗？ | [完成报告](gdpo/stage1_35/gdpo_stage1_report.md)、[gate](gdpo/stage1_35/STAGE_COMPLETE_OK)、[最终指标](gdpo/stage1_35/metrics_at_final.json) |
| 两算法 35-step 如何比较？ | [比较报告](comparison/grpo_gdpo_stage1_35_comparison.md)、[机器可读比较](comparison/grpo_gdpo_stage1_35_comparison.json) |
| GRPO 35 steps 完成了吗？ | [完成报告](grpo/stage1_35/grpo_stage1_report.md)、[gate](grpo/stage1_35/STAGE_COMPLETE_OK)、[run status](grpo/stage1_35/run_status.json) |
| 验证曲线和训练数值是否可核验？ | [原生 GRPO 指标快照](grpo/stage1_35/metrics_at_snapshot.json)；含 step0 和 35 个训练记录 |
| checkpoint 是否保留且可加载？ | [manifest](grpo/stage1_35/checkpoint_manifest.json)、[resume smoke](../docs/diagnostics/checkpoint_resume_smoke.md) |
| gate 曾失败，为什么后来通过？ | [已解决失败记录](grpo/stage1_35/STAGE_FAILED.resolved)、[解决说明](grpo/stage1_35/STAGE_FAILED.resolution.md)；缺报告后补齐，无重训 |
| 使用哪个环境与初始化？ | [运行环境证据](runtime_verification_20261002.md)、[SFT v2](sft/sft_v2_train_report.md)、[合并验证](comparison/rl_init_merge_validation.md) |
| 官方规模单步能否真正更新？ | [GRPO capacity](grpo/grpo_official_scale_capacity_report.md)、[GDPO capacity](gdpo/gdpo_official_scale_capacity_report.md) |
| Config A 是否测过？ | [GRPO](grpo/grpo_throughput_config_a_report.md)、[GDPO](gdpo/gdpo_throughput_config_a_report.md) |
| 是否存在影响比较的实现差异？ | [KL audit](comparison/kl_path_audit.md)、[shared rollout diagnostic](comparison/grpo_gdpo_shared_rollout_diagnostic.md) |
| 服务器文件是否进入 Git？ | [审计说明](../docs/github_audit_20261002.md)、[来源与 SHA256 manifest](server_artifact_manifest.json) |

`metrics_at_snapshot.json` 只提取 stdout 中已记录的数值字段，不包含原始 prompt/response；缺失字段不补值。GRPO 六份原始 validation JSONL 各有 80 行，其来源、长度与 SHA256 保存在服务器 manifest 中。

GRPO 原始报告 E 节的 `step_time=False` 是报告检查字段与实际日志键不一致造成的缺陷。本次直接检查 35 条原生 `perf/time_per_step`，全部有限，范围 1073.70–1118.51 秒；保留原报告不改，避免改写历史证据。

历史记录：[0/105 的旧 GRPO 启动失败](grpo/historical_grpo_formal_run_report.md)；[初始环境准备失败](../configs/environment_report.md)。这些记录不代表当前 Stage 1 状态。
