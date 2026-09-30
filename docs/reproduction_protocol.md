# GDPO Tool-Calling Post-Training：Milestone 1 reproduction protocol

状态：协议冻结；租卡实测状态待补，2026-09-30

本文件只冻结复现协议、单 GPU 适配边界和 smoke-test 入口。Milestone 1 不启动 105-step 正式训练，不改变 reward、advantage、GRPO/GDPO 或 KL path。

## 1. 不可变的实验身份

| 项目 | 冻定值 |
| --- | --- |
| 官方 GDPO 仓库 | `https://github.com/NVlabs/GDPO` |
| GDPO repo commit | `4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115` |
| ToolRL repo commit | `8cee13ec0ca72f0461da372a93a6fd8140dbb840` |
| base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| train split | ToolRL processed `dataset/rlla_4k/train.parquet`，3920 条 |
| validation split | ToolRL processed `dataset/rlla_4k/test.parquet`，80 条 |
| 当前正式比较 | R1 = official GRPO；R2 = official GDPO |
| 主比较 checkpoint | `step 100` |
| 官方脚本推导出的总步数 | `105` |
| 本阶段允许的训练 | 仅 GRPO 1 step + GDPO 1 step smoke；禁止 105 steps |

ToolRL 的 `rlla_4k` 同时存在 raw 4,000 条 JSON 和 processed 3,920/80 parquet。正式复现只使用上述 processed parquet，不重新处理、不改字段、不改 split。

## 2. Official configuration（冻结）

以下值来自 `verl-GDPO/train_grpo.sh`、`verl-GDPO/train_gdpo.sh` 和 `verl-GDPO/verl/trainer/config/ppo_trainer.yaml`。没有写在脚本里的值，以该 YAML 的实际默认值为准。

| 参数 | Official 值 | 直接来源/说明 |
| --- | --- | --- |
| algorithm | GRPO 或 GDPO；其余代码路径相同 | 两个官方 shell script |
| train batch | `512 prompts` | shell script |
| validation batch | script 写 `128`；实际 validation DataLoader 用全部 `80` 条 | `ray_trainer.py` 的 `batch_size=len(val_dataset)` |
| rollout generations | `rollout.n=4` | shell script |
| PPO mini-batch | `128` | shell script |
| PPO micro-batch | `64` | YAML 默认值，script 未覆盖 |
| dynamic batch | `True` | shell script |
| learning rate | `1e-6` | shell script/YAML |
| optimizer | AdamW；betas `(0.9, 0.999)`；weight decay `1e-2`；constant warmup，warmup ratio `0` | FSDP worker optimizer construction |
| actor model dtype | 参数加载为 fp32；FSDP 默认 forward param dtype 为 bf16，reduce dtype 为 fp32 | `fsdp_workers.py` |
| rollout dtype | `bfloat16` | YAML |
| gradient checkpointing | enabled | shell script |
| prompt length | `2048` | shell script |
| response length | `1024` | shell script |
| model context used by rollout | `2048 + 1024 = 3072` | rollout wrapper |
| base model | `Qwen/Qwen2.5-1.5B-Instruct` | shell script |
| KL penalty | `algorithm.kl_penalty=kl` | shell/YAML |
| KL coefficient | fixed `algorithm.kl_ctrl.kl_coef=0.001` | shell/YAML |
| actor KL loss | `use_kl_loss=False`；因此 `actor.kl_loss_coef=0.001` 和 `low_var_kl` 不启用 | shell script |
| actual external KL path | `token_level_rewards = token_level_scores - 0.001 * kld` | `ray_trainer.py` |
| vLLM | `0.6.3`；`VLLM_ATTENTION_BACKEND=XFORMERS` | official requirements/script |
| vLLM memory fraction | `gpu_memory_utilization=0.6` | shell script 覆盖 YAML 的 `0.5` |
| vLLM execution | `enforce_eager=True`；`free_cache_engine=True`；`load_format=dummy_dtensor` | YAML |
| vLLM batching | `max_num_batched_tokens=8192`；`max_num_seqs=1024` | YAML |
| rollout tensor parallelism | `1` | official shell script 的 `ROLLOUT_TP_SIZE=1` |
| FSDP actor offload | parameter/gradient/optimizer 全部 `False` | shell script |
| FSDP reference offload | parameter `True` | shell script |
| FSDP nodes/GPUs | `nnodes=1`，`n_gpus_per_node=8` | shell script |
| total epochs | `15` | shell script |
| actual total steps | `floor(3920 / 512) * 15 = 7 * 15 = 105` | train DataLoader `drop_last=True`，trainer step injection |
| checkpoint interval | `save_freq=5` | shell script |
| validation interval | `test_freq=10`；训练结束后还会 unconditional final validation | shell script/trainer loop |
| logger | `console` + `wandb` | shell script |
| seed | training/DataLoader seed 未在 official PPO config 里指定；vLLM `LLM` 构造也未显式传 seed，旧 wrapper 的默认 vLLM seed 为 `0` | 不能把未指定的训练 seed 猜成 `42` |

### 2.1 两条官方 reward/KL 约定继续保持不变

本阶段不修正 Milestone 0 已确认的 confound：

```text
GRPO:
raw total score
→ external KL penalty
→ token_level_rewards
→ group normalization

GDPO:
format score       → group normalization
correctness score  → group normalization
→ sum
→ batch masked whitening
→ advantage
```

GRPO smoke 和 GDPO smoke 都必须忠实走官方路径。Controlled KL-path experiment 不属于本阶段。

## 3. Single-GPU adaptation（独立于 official configuration）

单卡适配不覆盖、不修改 `train_grpo.sh` 或 `train_gdpo.sh`。入口是本仓库的 `scripts/run_single_gpu_smoke.sh`，它只允许把 trainer 的 `total_training_steps` 覆盖为 `1`。

| 项目 | 单 GPU 适配值 | 是否改变算法 |
| --- | --- | --- |
| visible GPU | `CUDA_VISIBLE_DEVICES=0` | 否 |
| GPU count | `N_GPUS=1`；`trainer.n_gpus_per_node=1`；`trainer.nnodes=1` | 否 |
| rollout TP | `actor_rollout_ref.rollout.tensor_model_parallel_size=1` | 否；而且与官方 TP=1 一致 |
| train batch | 首轮仍为 `512` | 否；实测 OOM 前不缩 |
| rollout.n | 首轮仍为 `4` | 否；实测 OOM 前不缩 |
| prompt/response | `2048/1024` | 否；实测 OOM 前不缩 |
| mini/micro batch | `128/64` | 否；实测 OOM 前不缩 |
| vLLM memory fraction | `0.6` | 否 |
| actor/ref offload | 保持 official 值 | 否 |
| checkpoint/validation | smoke 时关闭周期 checkpoint/validation；trainer 最后的官方 final validation 仍可能执行 | 只减少 smoke 周期，不改变训练计算 |
| total steps | `1` | 只属于 smoke，不是正式 reproduction |

如果 1 GPU 在完整官方 batch/rollout/长度下 OOM，下一步只能单独产生一份“显存适配配置”，并记录具体先缩减的参数、OOM stack trace 和新配置；本文件的 official protocol 不变。

## 4. Environment plans

### Plan A — Legacy-faithful

目标环境：Linux x86_64，Python 3.10/3.11，PyTorch `2.4.0`（cu121），CUDA 12.1，vLLM `0.6.3`，official repo 的原始 `transformers<4.48` 约束，Ray 采用官方未锁定版本，FlashAttention 采用官方 requirements 的可构建版本。

判定：**在 RTX PRO 6000 96GB / RTX 6000D 84GB 这类 Blackwell 卡上 not viable，不硬装。**

原因是双重的：

1. NVIDIA 的 Blackwell compatibility guidance 将 CUDA 12.8 作为能够生成 Blackwell 原生 cubin 的工具链边界；12.7 及更早工具链不能生成 Blackwell 原生 cubin，旧二进制还必须包含可 JIT 的 PTX 才可能工作。
2. vLLM `0.6.3` 的 CMake 只列出 CUDA arch `7.0;7.5;8.0;8.6;8.9;9.0`，没有 SM 12.0；本地仓库的 vendored vLLM selector 也只接受 `0.3.1/0.4.2/0.5.4/0.6.3`。

因此 Plan A 只保留为“旧环境审计基线”，不作为 Blackwell 租卡安装方案。

### Plan B — Blackwell-compatible candidate

第一候选组合如下；这是上机 smoke 前的 candidate，不是已经在本地 GPU 验证过的事实。

| 组件 | 候选版本/约束 | 备注 |
| --- | --- | --- |
| OS/arch | Ubuntu Linux x86_64 | vLLM/FlashAttention 的实际支持路径 |
| Python | `3.10` 或 `3.11` | vLLM 0.9.0 支持 3.9–3.12；优先 3.10/3.11 减少旧项目依赖摩擦 |
| NVIDIA driver | 使用租卡镜像提供的、明确支持 CUDA 12.8 的 driver；preflight 必须记录准确版本 | 不在项目里擅自升级宿主 driver |
| CUDA toolkit | `12.8` | Blackwell 最低兼容基线；构建扩展时用该 `nvcc` |
| PyTorch | `2.7.0+cu128` | PyTorch 2.7 官方提供 Blackwell 支持和 cu128 wheel |
| vLLM | `0.9.0` | 官方 v0.9.0 期望 torch 2.7.0，并在 CUDA >=12.8 时包含 SM12.0 arch |
| Ray | `2.43.x`；避开 `2.44.*` | vLLM 0.9.0 的 CUDA requirements |
| xFormers | `0.0.30` | 与 vLLM 0.9.0/torch 2.7 对应；保留 `VLLM_ATTENTION_BACKEND=XFORMERS` 先测 |
| transformers | `4.51.1` 或与 vLLM 0.9.0 相容的更高 4.x | 这是相对官方 `<4.48` 的明确偏离，必须单独记录；Qwen2.5 仍是同一 base model |
| FlashAttention | `2.8.3` source build，显式 `FLASH_ATTN_CUDA_ARCHS=120`；smoke-gated | 2.8.3 setup.py 有 SM120 编译入口，但其 README 的正式 FA2 支持列表仍未包含 Blackwell；若 HF `flash_attention_2` forward/backward 失败，不能进入正式训练 |

Plan B 的关键不是重写 trainer，而是两个局部兼容点：

1. 当前仓库的 `verl/third_party/vllm/__init__.py` 会拒绝 vLLM 0.9.0，必须增加一个局部 vLLM 0.9.0 adapter/selector，并核对 `vLLMRollout` 的 `LLM`、`SamplingParams`、weight-sync 和 cache API。不能把版本号强行改成 0.6.3。
2. 当前 FSDP worker 把 Hugging Face attention backend 写死为 `flash_attention_2`。Plan B 先按上表 source-build 做原样 smoke；只有在确认 FA2 在目标卡不可用后，才可评估将 attention backend 切到 PyTorch SDPA 的最小非算法兼容改动。该 fallback 不是本阶段自动执行的修改。

截至本协议冻结日，没有指定云厂商、镜像 ID 或可访问的 Blackwell 实例，因此“租卡镜像实际提供的 driver/CUDA/Python/PyTorch”尚未被验证；不能把 Plan B candidate 写成已安装环境。

## 5. Preflight and smoke procedure

### 5.1 Preflight（不训练）

在目标实例执行：

```bash
bash scripts/preflight_blackwell.sh
```

该脚本只读输出以下信息：GPU 名称/compute capability/总显存、driver、`nvcc`/CUDA、Python、PyTorch、vLLM、Transformers、Ray、FlashAttention、torch CUDA availability。输出应保存到对应租卡目录，作为环境审计原始记录。

### 5.2 Smoke（每种 algorithm 只 1 step）

```bash
bash scripts/run_single_gpu_smoke.sh grpo
bash scripts/run_single_gpu_smoke.sh gdpo
```

wrapper 强制：`CUDA_VISIBLE_DEVICES=0`、`N_GPUS=1`、rollout TP=1、`trainer.total_training_steps=1`。它不调用官方 105-step shell script，也不改变 reward 或 advantage 计算。

wrapper 会把 preflight 保存为 `preflight.txt`，把完整训练输出保存为 `smoke.log`，并把 `nvidia-smi` 的设备级显存/利用率采样保存为 `nvidia-smi.csv`。PyTorch 的 `max_memory_allocated()` / `max_memory_reserved()` 仍需要后续 smoke-only observability hook；不能把 `nvidia-smi memory.used` 冒充为这两个 allocator peak。

每次 smoke 必须保存：

- GPU model、VRAM total、peak allocated/reserved VRAM；
- driver/CUDA/Python/PyTorch/vLLM/Transformers/Ray/FlashAttention 版本；
- rollout、reward、advantage、backward、optimizer.step 的 success/failure；
- step latency、OOM 状态、完整 stdout/stderr 和 stack trace；
- 2–3 个 rollout 的 format/correctness/total reward；
- GRPO 的 group reward/mean/std/advantage；
- GDPO 的 format/correctness score、各自 group-normalized advantage、combined advantage、batch-whitened final advantage。

上述 advantage/reward logging 必须是只读观测，不得插入额外归一化、detach、重新计算或替换 batch tensor。当前 upstream console metrics 主要是聚合统计；如果要保存逐样本张量，后续只允许加入 smoke-only observability hook，并在 diff 中证明训练张量未被改写。

### 5.3 Smoke 通过标准

GRPO 和 GDPO 各自都必须完成：

```text
model load
→ vLLM rollout（4 responses/prompt）
→ reward scorer
→ official advantage path
→ policy forward/backward
→ optimizer.step
```

且没有 OOM、CUDA kernel error、Ray actor error、vLLM API/weight-sync error 或 FlashAttention forward/backward error。只完成 rollout 或只完成 forward 不算通过。

## 6. 当前仍然禁止的操作

本协议期间禁止：105-step 正式训练、LoRA/QLoRA、Qwen3、reward 修改、GDPO 修改、TRL migration、量化训练、reward-scale experiment、KL-path correction、第二 benchmark，以及覆盖官方 shell script。

## 7. 需要在 GPU smoke 后补齐的结果

| 结果 | 当前状态 |
| --- | --- |
| 实际租卡 GPU/镜像版本 | 待提供云厂商/镜像并执行 preflight |
| GRPO 1-step | 未运行；本机为 macOS arm64，无 NVIDIA GPU |
| GDPO 1-step | 未运行；本机为 macOS arm64，无 NVIDIA GPU |
| peak VRAM/latency/reward/advantage samples | 待 smoke 原始日志 |
| 算法代码修改 | 当前为 0；Plan B 的 vLLM adapter 尚未写入 |
| 84GB vs 96GB 正式训练选择 | 必须以两卡 smoke/OOM 和单位价格为依据，当前不做决定 |

正式训练建议规则保持冻结：两卡都稳定通过且官方 batch/rollout/长度均不变时选单位训练成本更低者；84GB OOM 而 96GB 通过时选 96GB。

## 8. 冻结时的代码入口

这些是本阶段只需关注的文件，不需要通读整个 veRL：

1. `verl-GDPO/train_grpo.sh` / `train_gdpo.sh`：官方运行参数。
2. `verl-GDPO/verl/trainer/config/ppo_trainer.yaml`：mini/micro batch、precision、vLLM、FSDP 默认值。
3. `verl-GDPO/verl/trainer/ppo/ray_trainer.py`：数据 batch、105-step 推导、rollout 到 reward/advantage 的主路径。
4. `verl-GDPO/verl/trainer/ppo/core_algos.py`：group mean/std 和 masked whitening 的数学实现。
5. `verl-GDPO/verl/workers/fsdp_workers.py`：actor dtype、gradient checkpointing、HF attention backend、FSDP worker。
6. `verl-GDPO/verl/workers/rollout/vllm_rollout/vllm_rollout.py`：vLLM `LLM`/sampling/weight-sync 入口。
7. `verl-GDPO/verl/third_party/vllm/__init__.py`：当前 vLLM 版本白名单，Plan B adapter 的第一落点。
8. `verl-GDPO/verl/utils/reward_score/rlla.py`：format/correctness scorer 入口。
