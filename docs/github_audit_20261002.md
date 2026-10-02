# GitHub / 本地 / 服务器文件审计

核对时间：2026-10-02 08:21（北京时间）。检查范围为 Kk111777/ToolPostTrain 和 Kk111777/ReliableToolAgent 两个个人仓库，没有修改上游仓库。

## 初始状态

| 仓库 | GitHub main | 工作副本 | 问题 |
|---|---|---|---|
| ToolPostTrain | 3fee1c5 | 服务器 main=3fee1c5，clean；Mac main=3b1c9b5，落后 4 commits | main 已有冻结实验文件；首页还是上游 README，最新 Stage 1 完成证据在仓库外 |
| ReliableToolAgent | 30bb116 | research/reliability-guards=7e0708e，clean | 个人成果已 push，但研究分支 5 commits 尚未进入 main |

本次先将 ToolPostTrain 本地 fast-forward 到已发布的服务器提交，然后在文档分支整理主页与新增证据。ReliableToolAgent 的既有研究成果作为同一次发布纳入 main；保留历史分支，不删除分支、不强制改写历史。

## 服务器文件核对

服务器项目：`/root/autodl-tmp/ProjectB`；repo 初始 SHA `3fee1c5`；实际训练使用另一个 clean veRL checkout `1876b06d0a3e4e71e06230be10af14492ca8a75b`。

[server_artifact_manifest.json](../reports/server_artifact_manifest.json) 将 env-modern 顶层全部 52 份小型文档/JSON/配置/锁/脚本逐一记录来源、大小、SHA256 与内容一致的仓库路径；具体数量以 manifest 为准。其中一份 constrained-generation Markdown 仅存在行尾空格差异，manifest 同时记录两份 SHA256 并标注 trailing_whitespace_only；保留已发布原文，其余已公开文件按内容匹配，不重复复制。

补入此前位于 repo 外部的 GRPO Stage 1 报告、checkpoint manifest、完成状态、resolved failure、diagnostics manifest 和实际 Hydra 配置快照；补入 GDPO resolved config、数据集 dry-run 和 `train_max_samples=-1` 语义报告。

原生 stdout 的紧凑数值快照包括 GRPO step0 + 35 个训练 step 和 GDPO step0 + 4 个训练 step。六个 GRPO validation JSONL 的来源、80 行计数、大小和 SHA256 已记录，原始大文件继续留在服务器。

## 有意不进入 Git 的文件

模型权重、LoRA/full/model-only checkpoint、optimizer/RNG/scheduler state、环境、HF cache、原始 parquet、大日志和实时 rollout 文件保留在服务器。Git 仓库完整并不意味着这些运行产物也应上传；公开 checkpoint manifest 只证明该快照检查到的保存/加载状态，不使权重成为可下载制品。

本次只读服务器文件并在 Mac 整理公开副本；没有 pull 到运行中的服务器，没有改变 reward、trainer、dataset、launch config、checkpoint、环境或监控任务。

## 尚未解决的复现边界

- 独立 orchestrator/watcher 未在服务器项目文件中发现，自动 gate/滚动删除/关机并非仓库内可独立重建的完整流程。
- frozen formal config 的状态与部分 checkpoint 清理文字是实验开始前的记录；当前 policy 保留两算法 step35 full + model-only，以 Stage 1 完成快照及本页为准。
- GDPO 尚未完成，未产生 step35 completion report/gate；不补造完成文件。
- 完整新机器重建所需的数据生成、合并权重与路径准备仍应按照报告进行；不是一键 clone 即可正式训练。

## 发布前验证

逐一验证新增副本 SHA256、JSON 可读性、Markdown 本地链接、数值快照有限性和 git diff 空白检查；检查新增范围不包含 secret、checkpoint、环境或大日志。GRPO 原报告 `step_time=False` 与原生字段不一致：本次直接验证 35 条 `perf/time_per_step` 全部有限，原文保留。

ReliableToolAgent 项目专用 `local_demo` 测试 50 passed。额外上游 memory/agents 测试在本机轻量环境下出现缺少 optional dependencies / fixtures 等问题；用未修改 origin/main 的临时副本对照，结果记录于 [ReliableToolAgent 发布检查](https://github.com/Kk111777/ReliableToolAgent/blob/main/reports/github_publication_20261002.md)。不把轻量验证称为完整上游测试通过。
