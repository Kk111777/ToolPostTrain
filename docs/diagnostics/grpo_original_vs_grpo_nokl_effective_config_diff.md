# GRPO-original vs GRPO-noKL：静态配置差异审计

状态：**仅完成静态准备，未启动 GRPO-noKL，未加载模型，未占用 GPU**。

## 目的与边界

Run A 是已完成的 GRPO-original35：

- 实际启动证据：`/root/autodl-tmp/ProjectB/runs/grpo_stage1_35/command.txt`
- Hydra snapshot：`/root/autodl-tmp/ProjectB/repo/outputs/2026-10-01/16-56-41/.hydra/`
- 初始化：`RL_INIT_V1`，从 `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged` 独立 step0 开始
- parser 合法名称：`qwen2p5_1p5b_grpo_one_step`

Run C 只作为后续 KL-aligned ablation 设计，不是已运行结果。它必须从同一 RL_INIT_V1 独立 step0 开始；run 目录和日志名可区分，但 `trainer.experiment_name` 继续使用上述合法 GRPO 名称。

## KL 字段实际语义

基于当前固定的 official veRL v0.9.1 源码：

- `verl/trainer/ppo/utils.py:78-79`：是否需要 reference policy 的条件是 `use_kl_in_reward OR actor.use_kl_loss`。
- `verl/trainer/ppo/ray_trainer.py:364`：`use_kl_in_reward=True` 时建立 in-reward KL controller。
- `verl/trainer/ppo/ray_trainer.py:1639`：该开关为真时调用 `apply_kl_penalty`，将 KL penalty 应用于 token-level reward；为假时直接使用原始 token-level scores。
- `verl/workers/utils/losses.py:132`：`actor.use_kl_loss=True` 才会把 KL loss 加入 actor policy loss。

因此本 ablation 的唯一 KL 变化是关闭 reward-side KL；actor KL loss 保持与当前 GDPO 一致为关闭。

## 实际 GRPO-original 与拟议 GRPO-noKL

| 字段 | GRPO-original（已运行） | GRPO-noKL（拟议，仅静态） | 是否变化 |
|---|---|---|---|
| `algorithm.adv_estimator` | `grpo` | `grpo` | 否 |
| `algorithm.use_kl_in_reward` | `True` | `False` | **唯一 KL 变化** |
| `algorithm.kl_penalty` | `kl` | `kl`（保留配置，开关关闭时不应用） | 否 |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | `0.001`（保留配置，开关关闭时不应用） | 否 |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | `False` | 否 |
| `trainer.experiment_name` | `qwen2p5_1p5b_grpo_one_step` | `qwen2p5_1p5b_grpo_one_step` | 否 |
| 初始化 | RL_INIT_V1 step0 | RL_INIT_V1 step0 | 否 |
| 数据、顺序、seed | 3901 eligible；shuffle=false；drop_last；seed=42 | 完全相同 | 否 |
| 预算 | 5 epochs / 35 steps | 5 epochs / 35 steps | 否 |
| validation | formal_validation_80，0/7/14/21/28/35，greedy n=1 | 完全相同 | 否 |
| rollout / batch | n=4；train batch=512；PPO minibatch=128；physical microbatch=1 | 完全相同 | 否 |
| 长度 / lr | 2048 / 1024；1e-6 | 完全相同 | 否 |
| vLLM Config A | TP=1；0.40；8；6144 | 完全相同 | 否 |
| reward parser | rlla.py / `compute_score` | 完全相同 | 否 |

拟议 Run C 的最小新增 override 是：

```text
algorithm.use_kl_in_reward=False
```

同时明确保持：

```text
actor_rollout_ref.actor.use_kl_loss=False
```

不应把 `gdpo_stage1`、`grpo_nokl` 或 run 名写入 `trainer.experiment_name`。

## 重要未决确认

由于两个 actor-side KL 开关都为 false，官方 `need_reference_policy` 判断会返回 false；这意味着 noKL 运行可能不创建 reference-policy 路径。这符合“关闭 reward-side KL、actor KL loss 也关闭”的字面语义，但在开 GPU 前仍应由用户确认是否就是预期的 no-reference ablation。

另外，已完成的 GDPO command snapshot 中仍记录 `algorithm.use_kl_in_reward=True`，同时使用 `gdpo_reward_keys=[accuracy_reward, format_reward]`。这说明现有 GRPO/GDPO 结果仍保留 KL-treatment / reward-dimension 的混杂，不能把两者差异写成纯 advantage-estimator 因果结论；本文件不修改历史结果。

## 结论

静态 diff 已准备完成；除 `algorithm.use_kl_in_reward=True -> False` 外没有计划配置漂移。GRPO-noKL 尚未 Hydra resolve、尚未启动、尚无结果。启动前仍需按该 diff 生成独立 launcher，完成 Hydra compose/resolve 和最终人工确认。
