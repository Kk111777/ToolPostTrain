#!/usr/bin/env bash
# Explicitly opt-in FINAL_HOLDOUT_V1 launcher. Static-only until provenance
# review and a separate infrastructure smoke have passed.
set -euo pipefail

if [[ "${PROJECTB_FINAL_TEST_ENABLE:-0}" != "1" ]]; then
  echo "FINAL_TEST_REFUSED: set PROJECTB_FINAL_TEST_ENABLE=1 after manual FINAL_HOLDOUT_V1 review" >&2
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
HOLDOUT_FILE="$ROOT/env-modern/final_holdout_v1.parquet"
HOLDOUT_MANIFEST="$ROOT/repo/manifests/final_holdout_v1_manifest.json"
RUNTIME_MAPPING="$ROOT/repo/manifests/final_holdout_v1_runtime_mapping.json"
FROZEN_ENDPOINT_MANIFEST_SHA256="160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16"
FROZEN_RUNTIME_MAPPING_SHA256="ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd"
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
[[ "$HOLDOUT_FILE" != *"test.parquet"* ]] || { echo "FINAL_TEST_REFUSED: old test.parquet is not an allowed endpoint" >&2; exit 2; }
[[ -f "$HOLDOUT_FILE" && -f "$HOLDOUT_MANIFEST" && -f "$RUNTIME_MAPPING" ]] || { echo "missing FINAL_HOLDOUT_V1 parquet, endpoint manifest, or runtime mapping" >&2; exit 2; }
[[ "$OUT_DIR" == "$ROOT/final-test/"* ]] || { echo "FINAL_TEST_REFUSED: output must be under $ROOT/final-test/" >&2; exit 2; }
[[ "$OUT_DIR" != *"/runs/"* ]] || { echo "FINAL_TEST_REFUSED: output cannot be a Stage1 run directory" >&2; exit 2; }
if [[ -e "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "FINAL_TEST_REFUSED: output directory is not empty: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

# Read-only endpoint guard. It checks the frozen identity before any model or
# vLLM process is started and has no fallback to test.parquet. The file hashes
# are hardcoded outside the files they protect, so a replacement manifest or
# mapping cannot redefine its own expected identity.
"$PYTHON" - "$HOLDOUT_FILE" "$HOLDOUT_MANIFEST" "$RUNTIME_MAPPING" "$FROZEN_ENDPOINT_MANIFEST_SHA256" "$FROZEN_RUNTIME_MAPPING_SHA256" <<'PY'
import hashlib
import json
import sys

import pyarrow.parquet as pq

holdout, manifest_path, mapping_path, expected_manifest_sha, expected_mapping_sha = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if sha256_file(manifest_path) != expected_manifest_sha:
    raise SystemExit("FINAL_TEST_REFUSED: endpoint manifest byte identity mismatch")
if sha256_file(mapping_path) != expected_mapping_sha:
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping byte identity mismatch")
manifest = json.load(open(manifest_path, encoding="utf-8"))
mapping = json.load(open(mapping_path, encoding="utf-8"))
if manifest.get("manifest_version") != "FINAL_HOLDOUT_V1":
    raise SystemExit("FINAL_TEST_REFUSED: manifest is not FINAL_HOLDOUT_V1")
if manifest.get("row_count") != 108:
    raise SystemExit("FINAL_TEST_REFUSED: frozen row count is not 108")
if "test.parquet" in holdout:
    raise SystemExit("FINAL_TEST_REFUSED: old test.parquet endpoint is forbidden")
digest = hashlib.sha256()
with open(holdout, "rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != manifest.get("derived_parquet_sha256"):
    raise SystemExit("FINAL_TEST_REFUSED: FINAL_HOLDOUT_V1 SHA256 mismatch")
table = pq.read_table(holdout)
if len(table) != 108:
    raise SystemExit("FINAL_TEST_REFUSED: FINAL_HOLDOUT_V1 row count mismatch")
ids = [int((row.get("extra_info") or {}).get("index")) for row in table.to_pylist()]
id_digest = hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()
if id_digest != manifest.get("ordered_source_ids_sha256"):
    raise SystemExit("FINAL_TEST_REFUSED: FINAL_HOLDOUT_V1 source-ID hash mismatch")
if ids != manifest.get("ordered_source_ids"):
    raise SystemExit("FINAL_TEST_REFUSED: FINAL_HOLDOUT_V1 source order mismatch")
mapping_rows = mapping.get("rows", [])
if mapping.get("parent_endpoint_manifest_sha256") != expected_manifest_sha:
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping parent identity mismatch")
if mapping.get("endpoint_parquet_sha256") != manifest.get("derived_parquet_sha256"):
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping parquet identity mismatch")
if mapping.get("ordered_source_ids_sha256") != manifest.get("ordered_source_ids_sha256"):
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping source-ID identity mismatch")
if len(mapping_rows) != 108:
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping row count mismatch")
if [int(row.get("source_id")) for row in mapping_rows] != [int(row.get("source_id")) for row in manifest.get("rows", [])]:
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping source-ID order mismatch")
if [int(row.get("row_position")) for row in mapping_rows] != [int(row.get("row_position")) for row in manifest.get("rows", [])]:
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping row-position mismatch")
pairs = {(row.get("runtime_input_sha256"), row.get("ground_truth_sha256_exact")) for row in mapping_rows}
if len(pairs) != 108:
    raise SystemExit("FINAL_TEST_REFUSED: runtime mapping pairs are not unique")
print("FINAL_HOLDOUT_V1_STATIC_GUARD_PASS")
PY

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
  data.val_files="$HOLDOUT_FILE" \
  data.train_max_samples=-1 \
  data.val_max_samples=108 \
  data.train_batch_size=512 \
  data.val_batch_size=108 \
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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
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
  trainer.project_name=ProjectB_final_holdout_v1 \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUT_DIR/checkpoints" \
  trainer.logger='[console]' \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  ray_kwargs.ray_init.num_cpus=8
