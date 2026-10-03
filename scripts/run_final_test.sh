#!/usr/bin/env bash
# Explicitly opt-in final-test launcher. Static-only until provenance review.
set -euo pipefail

if [[ "${PROJECTB_FINAL_TEST_ENABLE:-0}" != "1" ]]; then
  echo "FINAL_TEST_REFUSED: set PROJECTB_FINAL_TEST_ENABLE=1 after manual provenance review" >&2
  exit 2
fi
if [[ "${PROJECTB_FINAL_TEST_CONFIRM:-}" != "I_HAVE_REVIEWED_PROVENANCE" ]]; then
  echo "FINAL_TEST_REFUSED: set PROJECTB_FINAL_TEST_CONFIRM=I_HAVE_REVIEWED_PROVENANCE" >&2
  exit 2
fi
if (( $# != 3 )); then
  echo "usage: PROJECTB_FINAL_TEST_ENABLE=1 PROJECTB_FINAL_TEST_CONFIRM=I_HAVE_REVIEWED_PROVENANCE $0 MODEL_NAME MODEL_PATH OUTPUT_DIR" >&2
  exit 2
fi

MODEL_NAME="$1"
MODEL_PATH="$2"
OUT_DIR="$3"
ROOT=/root/autodl-tmp/ProjectB
UPSTREAM="$ROOT/upstream/verl-v0.9.1"
VENV="$ROOT/.venv-modern"
PYTHON="$VENV/bin/python"
TEST_FILE="$ROOT/repo/verl-GDPO/dataset/rlla_4k/test.parquet"
TEST_MANIFEST="$ROOT/final-test-preflight/final_test_dataset_manifest.json"
EXPERIMENT_NAME="qwen2p5_1p5b_grpo_one_step"

case "$MODEL_NAME" in
  RL_INIT_V1) EXPECTED="$ROOT/checkpoints/rl_init_sft_v2_merged" ;;
  GRPO-original35) EXPECTED="$ROOT/runs/grpo_stage1_35/model_only/step_35" ;;
  GDPO-current35) EXPECTED="$ROOT/runs/gdpo_stage1_35/model_only/step_35" ;;
  GRPO-noKL35) EXPECTED="$ROOT/runs/grpo_nokl_stage1_35/model_only/step_35" ;;
  *) echo "FINAL_TEST_REFUSED: unknown model identity $MODEL_NAME" >&2; exit 2 ;;
esac
[[ "$MODEL_PATH" == "$EXPECTED" ]] || { echo "FINAL_TEST_REFUSED: model path is not frozen for $MODEL_NAME" >&2; exit 2; }
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || { echo "missing model path" >&2; exit 2; }
[[ -f "$TEST_FILE" && -f "$TEST_MANIFEST" ]] || { echo "missing test dataset or manifest" >&2; exit 2; }
[[ "$OUT_DIR" == "$ROOT/final-test/"* ]] || { echo "FINAL_TEST_REFUSED: output must be under $ROOT/final-test/" >&2; exit 2; }
[[ "$OUT_DIR" != *"/runs/"* ]] || { echo "FINAL_TEST_REFUSED: output cannot be a Stage1 run directory" >&2; exit 2; }
if [[ -e "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "FINAL_TEST_REFUSED: output directory is not empty: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"

cd "$UPSTREAM"
exec "$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_penalty=kl \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="$ROOT/repo/verl-GDPO/dataset/rlla_4k/train.parquet" \
  data.val_files="$TEST_FILE" \
  data.train_max_samples=-1 \
  data.val_max_samples=80 \
  data.train_batch_size=512 \
  data.val_batch_size=80 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.shuffle=False \
  data.validation_shuffle=False \
  data.seed=42 \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  reward.custom_reward_function.path="$UPSTREAM/verl/utils/reward_score/rlla.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=gdpo \
  reward.num_workers=1 \
  trainer.validation_data_dir="$OUT_DIR" \
  trainer.rollout_data_dir="$OUT_DIR" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.resume_mode=disable \
  trainer.total_epochs=0 \
  trainer.total_training_steps=0 \
  trainer.project_name=ProjectB_final_test \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUT_DIR/checkpoints" \
  trainer.logger='[console]' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  ray_kwargs.ray_init.num_cpus=8
