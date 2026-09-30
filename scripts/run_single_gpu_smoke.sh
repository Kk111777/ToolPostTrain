#!/usr/bin/env bash
set -euo pipefail

# One-step smoke wrapper for Milestone 1.
# It intentionally does not call train_grpo.sh/train_gdpo.sh and cannot run
# the official 105-step protocol unless this file is explicitly rewritten.

if [[ $# -ne 1 || ( "$1" != "grpo" && "$1" != "gdpo" ) ]]; then
    echo "usage: bash scripts/run_single_gpu_smoke.sh grpo|gdpo" >&2
    exit 2
fi

ALGORITHM="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="0"
export N_GPUS="1"
export ROLLOUT_TP_SIZE="1"
export VLLM_ATTENTION_BACKEND="XFORMERS"
export RAY_USAGE_STATS_ENABLED="0"
export RAY_DISABLE_DOCKER_CPU_WARNING="1"
export WITHLENGTH="0"
export REFINEDREWARD="0"
export COARSEREWARD="0"
export STRICTMATCH="0"
export CORRECTMAX1="0"
export MAX1STEP30MAX3="0"
export SCHEDULEREWARD="0"
export SCHEDULELENGTH="0"

DATA_DIR="${DATA_DIR:-verl-GDPO/dataset/rlla_4k}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SMOKE_ROOT="${SMOKE_ROOT:-verl-GDPO/results/smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5B-${ALGORITHM}-tool-smoke}"
CKPT_DIR="${SMOKE_ROOT}/${ALGORITHM}"

mkdir -p "$CKPT_DIR"
bash "$PROJECT_DIR/scripts/preflight_blackwell.sh" \
    2>&1 | tee "$CKPT_DIR/preflight.txt"

LOG_PATH="$CKPT_DIR/smoke.log"
GPU_TELEMETRY_PATH="$CKPT_DIR/nvidia-smi.csv"
MONITOR_PID=""

cleanup() {
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu \
        --format=csv \
        --loop-ms=500 > "$GPU_TELEMETRY_PATH" 2>&1 &
    MONITOR_PID=$!
else
    echo "nvidia-smi: NOT FOUND; GPU telemetry was not started" > "$GPU_TELEMETRY_PATH"
fi

set +e
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ALGORITHM" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=GDPO_Milestone1 \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    2>&1 | tee "$LOG_PATH"
SMOKE_EXIT_CODE=${PIPESTATUS[0]}
set -e
exit "$SMOKE_EXIT_CODE"
