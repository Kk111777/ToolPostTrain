# 运行环境与历史记录核对

> **Historical / superseded status or planning context — annotation added 2026-10-03.** Early runtime snapshot, not the three-run final status. Original facts and values below are preserved. Current artifacts: [three-run completion](comparison/final_stage1_ablation_completion_report.md); current budget: [35-step decision](../docs/experiment_design/stage1_closeout_decision.md).

核对时间：2026-10-02 08:21（北京时间）。通过 `ssh autodl` 只读检查服务器；没有安装包、加载新模型、启动训练或修改运行中的源码/配置。

- GPU：NVIDIA RTX 6000D，SM120；GPU-validation 日志记录 capability=12.0。
- 官方 veRL v0.9.1：`1876b06d0a3e4e71e06230be10af14492ca8a75b`；上游 checkout 为 clean detached tag。
- 环境路径：`/root/autodl-tmp/ProjectB/.venv-modern`。旧 `.conda` 不作为当前训练环境。
- 已记录依赖：Python 3.12.14、torch 2.11.0+cu130、CUDA runtime 13.0、vLLM 0.24.0、transformers 5.9.0、ray 2.55.1、NumPy 2.3.5；见 [resolved versions](../configs/resolved_versions.txt)。这些是准备时记录的版本，本次没有重新导入大型训练模块。
- 本次读取 cgroup memory.max=98784247808 bytes（92 GiB），初始报告中的 2 GiB 限制是历史状态。

## 已落盘的实际 GPU 验证

读取 `/root/autodl-tmp/ProjectB/env-modern/gpu-validation-logs/validation.log`：SM120 检查通过、vLLM 本地生成通过、train/test parquet 分别 3920/80 行；GRPO 与 GDPO smoke 均记录 `smoke_status=PASS optimizer_step_seen=True`，耗时分别 97.1/98.7 秒，最终 `GPU_VALIDATION_PASS`。

后续 512-prompts Config A 与 Stage 1 实际训练由独立报告和原生指标支持。本次没有重跑 GPU gate。

## 环境例外与兼容设置

冻结环境使用官方 vLLM extra；缺失 wheel 导致省略 fsdp optional extra。训练/参考模型使用 PyTorch SDPA；rollout 的 attention backend 由 vLLM 自动选择。NumPy 从 lock 中的 2.4.6 调整为 2.3.5 以满足 mistral-common；`VLLM_USE_FLASHINFER_SAMPLER=0` 避免早期 sampler 编译兼容问题。例外详见 [历史环境报告](../configs/environment_report.md) 和 [install script](../scripts/install_env.sh)。

原始 `configs/environment_report.md` / `resolved_versions.txt` 中的 `ENVIRONMENT_PREP_FAIL_RUNTIME_GATE`、`training=NOT_RUN` 和 dataset 未找到描述是准备时快照，保留其时间顺序；不能作为当前状态。

## 重建范围

本仓库记录实验源码、环境锁、launchers、来源 manifest、诊断与结果。完整从零重建仍需单独获得 ToolRL 数据、基础模型、SFT merged RL_INIT_V1 和服务器目录布局；权重、optimizer、大日志与 parquet 不在 Git 中。

Stage 1 launcher 不包含完整的 completion gate / full checkpoint 滚动清理 / 自动关机流程。这些在原对话的外部监控流程中处理；本次未在服务器上找到独立 orchestrator/watcher 文件，不能宣称 Git clone 已包含整套自动流水线。
