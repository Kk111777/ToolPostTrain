#!/usr/bin/env bash
# ProjectB formal GRPO-noKL Stage 1 launcher.
# This is the independent GRPO-noKL Stage-1 launcher; it does not modify train_grpo.sh or train_grpo_stage1.sh.
set -euo pipefail

ROOT=/root/autodl-tmp/ProjectB
UPSTREAM="$ROOT/upstream/verl-v0.9.1"
VENV="$ROOT/.venv-modern"
PYTHON="$VENV/bin/python"
FORMAL_CONFIG="$ROOT/env-modern/formal_experiment_config.yaml"
MODEL_PATH="$ROOT/checkpoints/rl_init_sft_v2_merged"
TRAIN_FILE="$ROOT/repo/verl-GDPO/dataset/rlla_4k/train.parquet"
VAL_FILE="$ROOT/env-modern/formal_validation_80.parquet"
TEST_FILE="$ROOT/repo/verl-GDPO/dataset/rlla_4k/test.parquet"
RUN_DIR="$ROOT/runs/grpo_nokl_stage1_35"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
EVAL_DIR="$RUN_DIR/eval_generations"
TRAIN_DUMP_DIR="$RUN_DIR/.rollout_capture"
DIAGNOSTIC_DIR="$RUN_DIR/diagnostics"
MODEL_ONLY_DIR="$RUN_DIR/model_only/step_35"
SESSION_NAME="projectb_grpo_nokl_stage1"
EXPERIMENT_NAME="qwen2p5_1p5b_grpo_one_step"
MIN_FREE_BYTES=$((45 * 1024 * 1024 * 1024))

MODE="${1:-run}"
STATUS_PHASE="preflight"
COMPACTOR_PID=""

fail() {
    echo "STAGE1_PREFLIGHT_FAIL: $*" >&2
    exit 1
}

preflight() {
    "$PYTHON" - "$FORMAL_CONFIG" "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE" <<'PYTHON_PREFLIGHT'
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
import yaml

formal_path, model_path, train_path, val_path, test_path = map(Path, sys.argv[1:])
cfg = yaml.safe_load(formal_path.read_text())
assert cfg["initialization"]["name"] == "RL_INIT_V1"
assert Path(cfg["initialization"]["path"]) == model_path
assert model_path.is_dir() and (model_path / "config.json").exists()
assert train_path.is_file() and test_path.is_file() and val_path.is_file()
assert Path(cfg["data"]["validation_path"]) == val_path
assert Path(cfg["data"]["test_path"]) == test_path
assert Path(cfg["evaluation"]["source_path"]) == val_path
assert Path(cfg["evaluation"]["manifest_path"]).is_file()
assert val_path != test_path
assert "test.parquet" not in str(cfg["data"]["validation_path"])
assert cfg["data"]["raw_train_rows"] == 3920
assert cfg["data"]["eligible_train_rows"] == 3901
assert cfg["data"]["filter_overlong_prompts"] is True
assert cfg["data"]["shuffle"] is False
assert cfg["data"]["drop_last"] is True
assert cfg["data"]["batch_size_prompts"] == 512
assert cfg["data"]["complete_batches_per_epoch"] == 7
assert cfg["budget"]["stage1"]["epochs"] == 5
assert cfg["budget"]["stage1"]["total_steps"] == 35
assert cfg["rollout"]["n"] == 4
assert cfg["rollout"]["tensor_parallel_size"] == 1
assert abs(cfg["rollout"]["gpu_memory_utilization"] - 0.40) < 1e-9
assert cfg["rollout"]["max_num_seqs"] == 8
assert cfg["rollout"]["max_num_batched_tokens"] == 6144
assert cfg["actor"]["ppo_mini_batch_prompts"] == 128
assert cfg["actor"]["ppo_micro_batch_per_gpu"] == 1
assert cfg["actor"]["dynamic_token_budget"] == 6144
assert abs(cfg["optimizer"]["learning_rate"] - 1e-6) < 1e-15
assert cfg["evaluation"]["steps"] == [0, 7, 14, 21, 28, 35]
assert cfg["evaluation"]["val_kwargs"]["n"] == 1
assert cfg["evaluation"]["val_kwargs"]["temperature"] == 0
assert cfg["evaluation"]["val_kwargs"]["do_sample"] is False
assert cfg["evaluation"]["test_policy"]["intermediate_eval"] == "forbidden"
assert cfg["algorithm_variants"]["grpo"]["experiment_name"] == "qwen2p5_1p5b_grpo_one_step"
assert cfg["algorithm_variants"]["gdpo"]["experiment_name"] == "qwen2p5_1p5b_gdpo_one_step"
train_rows = pq.read_table(train_path).num_rows
val_rows = pq.read_table(val_path).num_rows
test_rows = pq.read_table(test_path).num_rows
assert train_rows == 3920 and val_rows == 80 and test_rows == 80
manifest = json.loads(Path(cfg["evaluation"]["manifest_path"]).read_text())
assert manifest["selected_count"] == 80
val_records = pq.read_table(val_path).to_pylist()
val_ids = [int((row.get("extra_info") or {})["index"]) for row in val_records]
assert val_ids == manifest["selected_source_rows"]
print("STATIC_CONFIG_ASSERT_PASS")
print(json.dumps({
    "raw_train_rows": train_rows,
    "eligible_train_rows": cfg["data"]["eligible_train_rows"],
    "batches_per_epoch": cfg["data"]["complete_batches_per_epoch"],
    "stage1_epochs": cfg["budget"]["stage1"]["epochs"],
    "stage1_steps": cfg["budget"]["stage1"]["total_steps"],
    "validation_rows": val_rows,
    "validation_path": str(val_path),
    "test_path_not_used_for_validation": str(val_path) != str(test_path),
    "experiment_name": cfg["algorithm_variants"]["grpo"]["experiment_name"],
}, ensure_ascii=False))
PYTHON_PREFLIGHT

    if [[ -e "$RUN_DIR" ]]; then
        if [[ -f "$RUN_DIR/run_status.json" ]]; then
            "$PYTHON" - "$RUN_DIR/run_status.json" "$CHECKPOINT_DIR" <<'PYTHON_EXISTING_RUN'
import json
import sys
from pathlib import Path

status_path, checkpoint_dir = map(Path, sys.argv[1:])
status = json.loads(status_path.read_text())
assert status.get("phase") == "training_failed", (
    f"existing run is not a failed preflight/training run: {status.get('phase')!r}"
)
if Path(checkpoint_dir).exists():
    assert not any(Path(checkpoint_dir).iterdir()), (
        f"existing run has checkpoint state; refusing to reuse: {checkpoint_dir}"
    )
print("REUSING_FAILED_RUN_DIR", status.get("phase"))
PYTHON_EXISTING_RUN
            mkdir -p "$RUN_DIR/pre_hydra_failure"
            for old_file in command.txt effective_config.yaml stdout.log stderr.log run_status.json; do
                if [[ -e "$RUN_DIR/$old_file" && ! -e "$RUN_DIR/pre_hydra_failure/$old_file" ]]; then
                    cp -p "$RUN_DIR/$old_file" "$RUN_DIR/pre_hydra_failure/$old_file"
                fi
            done
        else
            [[ -f "$RUN_DIR/effective_config_prelaunch.yaml" ]] || fail "existing run has no run_status.json or prelaunch snapshot: $RUN_DIR"
            if [[ -d "$CHECKPOINT_DIR" && -n "$(find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                fail "static prelaunch run directory already contains checkpoint state: $CHECKPOINT_DIR"
            fi
            echo "REUSING_STATIC_PRELAUNCH_RUN_DIR $RUN_DIR"
        fi
    fi
    [[ -x "$PYTHON" ]] || fail "modern environment python missing: $PYTHON"

    local free_bytes
    free_bytes="$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')"
    [[ "$free_bytes" =~ ^[0-9]+$ ]] || fail "could not read free disk space"
    (( free_bytes >= MIN_FREE_BYTES )) || fail "free disk space ${free_bytes} is below required ${MIN_FREE_BYTES}"

    local gpu_mem="static-only"
    if [[ "$MODE" != "--resolve-only" ]]; then
        command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
        local gpu_apps
        gpu_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
        [[ -z "$(printf '%s' "$gpu_apps" | sed '/^[[:space:]]*$/d')" ]] || fail "GPU has active compute processes: $gpu_apps"
        local gpu_mem
        gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
        [[ "$gpu_mem" =~ ^[0-9]+$ ]] || fail "could not read GPU memory"
        (( gpu_mem <= 1024 )) || fail "GPU memory is not basically empty: ${gpu_mem} MiB"

    fi

    if ps -eo pid=,args= | awk '$0 !~ /awk/ && $0 !~ /grep/ && $0 ~ /verl\.trainer\.main_ppo|ray::TaskRunner/ {print; found=1} END {exit found ? 0 : 1}'; then
        fail "another verl/Ray trainer process is active"
    fi
    echo "PREFLIGHT_PASS free_bytes=$free_bytes gpu_memory_mib=$gpu_mem run_dir_absent=true"
}

write_effective_config() {
    "$PYTHON" - "$FORMAL_CONFIG" "$RUN_DIR/effective_config.yaml" "$RUN_DIR" <<'PYTHON_EFFECTIVE'
import sys
from pathlib import Path
import yaml
src, dst, run_dir = map(Path, sys.argv[1:])
cfg = yaml.safe_load(src.read_text())
cfg["algorithm_common"]["use_kl_in_reward"] = False
cfg["runtime_effective"] = {
    "launcher": "/root/autodl-tmp/ProjectB/repo/scripts/train_grpo_nokl_stage1.sh",
    "run_dir": str(run_dir),
    "initialization": "/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged",
    "trainer_experiment_name": "qwen2p5_1p5b_grpo_one_step",
    "algorithm": "grpo",
    "optimizer_resume": "disabled; start from RL_INIT_V1 step0",
    "validation_runtime_path": "/root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet",
    "validation_manifest": "/root/autodl-tmp/ProjectB/env-modern/formal_validation_80_manifest.json",
    "validation_steps": [0, 7, 14, 21, 28, 35],
    "validation_sampling": {"n": 1, "temperature": 0, "do_sample": False, "top_k": -1, "top_p": 1.0},
    "train_sequence": "first 3584 eligible prompts per epoch; native DataLoader drop_last=True",
    "checkpoint_policy": "save_freq=10, max_actor_ckpt_to_keep=1, final step35 retained",
    "model_only_export": str(run_dir / "model_only" / "step_35"),
    "diagnostic_policy": "native rollout capture is externally compacted to at most 64 rows at steps 1,20,35",
    "test_policy": "test.parquet is not used for intermediate validation",
}
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PYTHON_EFFECTIVE
}

write_status() {
    local phase="$1"
    local rc="${2:-0}"
    if [[ -d "$RUN_DIR" ]]; then
        "$PYTHON" - "$RUN_DIR/run_status.json" "$phase" "$rc" "$SESSION_NAME" <<'PYTHON_STATUS'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, phase, rc, session = sys.argv[1:]
obj = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "exit_code": int(rc),
    "tmux_session": session,
    "initialization": "/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged",
    "trainer_experiment_name": "qwen2p5_1p5b_grpo_one_step",
    "validation_path": "/root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet",
    "test_used_for_intermediate_validation": False,
}
Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
PYTHON_STATUS
    fi
}

write_compactor() {
    "$PYTHON" - "$RUN_DIR/compact_rollouts.py" <<'PYTHON_COMPACTOR_WRITER'
from pathlib import Path
import sys
script_path = Path(sys.argv[1])
script_path.write_text(r'''#!/usr/bin/env python3
from pathlib import Path
import json, sys, time

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
marker = source / "TRAINING_DUMP_DONE"
keep = {1, 20, 35}
dest.mkdir(parents=True, exist_ok=True)
stable = {}
errors = []
while True:
    files = sorted(source.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    for path in files:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        now = time.monotonic()
        previous = stable.get(path)
        if previous is None or previous[0] != size:
            stable[path] = (size, now)
            continue
        if now - previous[1] < 3.0:
            continue
        try:
            step = int(path.stem)
            with path.open(encoding="utf-8") as f:
                lines = []
                for _ in range(64):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
            if step in keep:
                out = dest / f"step_{step}.jsonl"
                out.write_text("".join(lines), encoding="utf-8")
                (dest / f"step_{step}.meta.json").write_text(json.dumps({
                    "source_file": str(path), "kept_rows": len(lines), "max_rows": 64,
                    "sampling": "first 64 completed native rollout dump rows"
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            path.unlink()
            stable.pop(path, None)
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
            stable.pop(path, None)
    if marker.exists() and not list(source.glob("*.jsonl")):
        break
    time.sleep(2)
if errors:
    (dest / "compactor_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
(dest / "manifest.json").write_text(json.dumps({
    "kept_steps": sorted(keep),
    "max_rows_per_step": 64,
    "errors": errors,
}, ensure_ascii=False, indent=2) + "\n")
''', encoding="utf-8")
script_path.chmod(0o755)
PYTHON_COMPACTOR_WRITER
}

preflight
if [[ "$MODE" == "--preflight-only" ]]; then
    exit 0
fi
[[ "$MODE" == "run" || "$MODE" == "--resolve-only" ]] || fail "unknown mode: $MODE"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$EVAL_DIR" "$TRAIN_DUMP_DIR" "$DIAGNOSTIC_DIR" "$RUN_DIR/model_only"
write_effective_config
write_compactor

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_ATTENTION_BACKEND || true
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_ROOT="$ROOT"

CMD=(
  "$PYTHON" -m verl.trainer.main_ppo
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=True"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_penalty=kl"
  "algorithm.kl_ctrl.kl_coef=0.001"
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$VAL_FILE"
  "data.train_max_samples=-1"
  "data.val_max_samples=80"
  "data.train_batch_size=512"
  "data.val_batch_size=80"
  "data.max_prompt_length=2048"
  "data.max_response_length=1024"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=1"
  "data.truncation=error"
  "data.shuffle=False"
  "data.validation_shuffle=False"
  "data.seed=42"
  "data.dataloader_num_workers=0"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.model.use_fused_kernels=False"
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
  "actor_rollout_ref.actor.strategy=fsdp"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=128"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144"
  "actor_rollout_ref.actor.use_torch_compile=False"
  "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.actor.fsdp_config.offload_policy=False"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.data_loader_seed=42"
  "actor_rollout_ref.actor.shuffle=False"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.ref.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.40"
  "actor_rollout_ref.rollout.n=4"
  "actor_rollout_ref.rollout.seed=42"
  "actor_rollout_ref.rollout.prompt_length=2048"
  "actor_rollout_ref.rollout.response_length=1024"
  "actor_rollout_ref.rollout.max_model_len=3072"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.max_num_batched_tokens=6144"
  "actor_rollout_ref.rollout.max_num_seqs=8"
  "actor_rollout_ref.rollout.agent.num_workers=4"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6144"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
  "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
  "reward.custom_reward_function.path=$UPSTREAM/verl/utils/reward_score/rlla.py"
  "reward.custom_reward_function.name=compute_score"
  "reward.reward_manager.name=gdpo"
  "reward.num_workers=1"
  "trainer.validation_data_dir=$EVAL_DIR"
  "trainer.rollout_data_dir=$TRAIN_DUMP_DIR"
  "trainer.val_before_train=True"
  "trainer.test_freq=7"
  "trainer.save_freq=10"
  "trainer.max_actor_ckpt_to_keep=1"
  "trainer.max_critic_ckpt_to_keep=1"
  "trainer.resume_mode=disable"
  "trainer.total_epochs=5"
  "trainer.total_training_steps=35"
  "trainer.project_name=ProjectB_formal_stage1"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.default_local_dir=$CHECKPOINT_DIR"
  "trainer.logger=[console]"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "trainer.use_v1=False"
  "trainer.val_only=False"
  "ray_kwargs.ray_init.num_cpus=8"
)
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
if [[ "$MODE" == "--resolve-only" ]]; then
    set +e
    "$PYTHON" -m verl.trainer.main_ppo --cfg job --resolve "${CMD[@]:3}" > "$RUN_DIR/effective_config_prelaunch.yaml" 2> "$RUN_DIR/hydra_resolve.stderr.log"
    RESOLVE_RC=$?
    set -e
    if (( RESOLVE_RC != 0 )); then
        fail "Hydra compose/resolve failed; see $RUN_DIR/hydra_resolve.stderr.log"
    fi
    echo "HYDRA_RESOLVE_PASS: $RUN_DIR/effective_config_prelaunch.yaml"
    exit 0
fi


STATUS_PHASE="launched"
write_status "$STATUS_PHASE" 0
write_compactor
"$PYTHON" "$RUN_DIR/compact_rollouts.py" "$TRAIN_DUMP_DIR" "$DIAGNOSTIC_DIR" > "$RUN_DIR/diagnostic_compactor.log" 2>&1 &
COMPACTOR_PID=$!

STATUS_PHASE="training"
set +e
"${CMD[@]}" > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
TRAIN_RC=$?
set -e
touch "$TRAIN_DUMP_DIR/TRAINING_DUMP_DONE"
if [[ -n "$COMPACTOR_PID" ]]; then
    wait "$COMPACTOR_PID" || true
fi
if (( TRAIN_RC != 0 )); then
    STATUS_PHASE="training_failed"
    write_status "$STATUS_PHASE" "$TRAIN_RC"
    exit "$TRAIN_RC"
fi

STATUS_PHASE="checkpoint_export"
write_status "$STATUS_PHASE" 0
FULL_ACTOR="$CHECKPOINT_DIR/global_step_35/actor"
[[ -d "$FULL_ACTOR" ]] || fail "step35 full actor checkpoint missing: $FULL_ACTOR"
mkdir -p "$RUN_DIR/model_only"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m verl.model_merger merge --backend fsdp --local_dir "$FULL_ACTOR" \
  --target_dir "$MODEL_ONLY_DIR" --use_cpu_initialization --trust-remote-code \
  > "$RUN_DIR/model_merger.stdout.log" 2> "$RUN_DIR/model_merger.stderr.log"

STATUS_PHASE="checkpoint_verify"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" - "$MODEL_ONLY_DIR" "$RUN_DIR/checkpoint_manifest.json" <<'PYTHON_VERIFY'
import json, os, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir=Path(sys.argv[1]); manifest_path=Path(sys.argv[2])
assert (model_dir / "config.json").exists()
model=AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
assert tok.chat_template
params=sum(p.numel() for p in model.parameters())
full=Path("/root/autodl-tmp/ProjectB/runs/grpo_stage1_35/checkpoints/global_step_35")
def bytes_under(p):
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
obj={
  "full_resume_checkpoint": str(full),
  "full_resume_exists": full.is_dir(),
  "full_resume_bytes": bytes_under(full),
  "model_only_checkpoint": str(model_dir),
  "model_only_exists": model_dir.is_dir(),
  "model_only_bytes": bytes_under(model_dir),
  "independent_from_pretrained_load": True,
  "tokenizer_load": True,
  "chat_template": True,
  "parameter_count": params,
  "optimizer_in_model_only": any("optim" in p.name.lower() for p in model_dir.rglob("*")),
}
manifest_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(obj,ensure_ascii=False))
PYTHON_VERIFY
STATUS_PHASE="completed"
write_status "$STATUS_PHASE" 0
