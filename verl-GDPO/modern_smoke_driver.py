#!/usr/bin/env python3
"""Run exactly one official verl v0.9.1 legacy-trainer smoke step.

This driver is intentionally narrow: it exercises the existing local model,
ToolRL parquet data, reward manager, advantage calculation, policy loss,
backward, and optimizer.step().  It refuses a run longer than one step so it
cannot accidentally become a training launcher.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


PROJECT_ROOT = pathlib.Path(os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/ProjectB"))
UPSTREAM_DIR = PROJECT_ROOT / "upstream" / "verl-v0.9.1"
LOG_DIR = PROJECT_ROOT / "env-modern" / "gpu-validation-logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("grpo", "gdpo"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--require-optimizer-step", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_steps != 1:
        raise SystemExit("refusing to run: modern_smoke_driver only permits --max-steps 1")
    if args.num_prompts < 1:
        raise SystemExit("refusing to run: --num-prompts must be positive")
    if args.rollout_n < 2:
        raise SystemExit("refusing to run: this smoke requires rollout.n >= 2")

    data_dir = pathlib.Path(args.data_dir)
    train_file = data_dir / "train.parquet"
    val_file = data_dir / "test.parquet"
    model_path = pathlib.Path(args.model_path)
    for required_path in (train_file, val_file, model_path / "config.json"):
        if not required_path.exists():
            raise SystemExit(f"required local path is missing: {required_path}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.algorithm}-driver.log"

    # v0.9.1's legacy RayPPOTrainer is selected deliberately here.  In this
    # tag it carries reward_extra_infos_dict into batch.non_tensor_batch before
    # compute_advantage, which is required for GDPO component rewards.
    overrides = [
        "data.train_files=" + str(train_file),
        "data.val_files=" + str(val_file),
        f"data.train_max_samples={args.num_prompts}",
        f"data.val_max_samples={args.num_prompts}",
        "data.train_batch_size=" + str(args.num_prompts),
        "data.val_batch_size=" + str(args.num_prompts),
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.filter_overlong_prompts_workers=1",
        "data.truncation=error",
        "data.shuffle=False",
        "data.dataloader_num_workers=0",
        "actor_rollout_ref.model.path=" + str(model_path),
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_fused_kernels=False",
        # The official lock intentionally does not install flash-attn.  Use
        # PyTorch SDPA for the FSDP training/ref model; vLLM keeps its own
        # automatic attention backend for rollout.
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.optim.lr=" + str(args.learning_rate),
        "actor_rollout_ref.actor.ppo_mini_batch_size=" + str(args.num_prompts),
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.30",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.prompt_length={args.max_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={args.max_response_length}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_response_length}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.max_num_batched_tokens=6144",
        # The legacy async agent-loop manager creates this many workers.  The
        # smoke has exactly num_prompts * rollout_n = 4 trajectories, and the
        # v0.9.1 DataProto chunker requires an even division.
        "actor_rollout_ref.rollout.max_num_seqs=4",
        "actor_rollout_ref.rollout.agent.num_workers=4",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6144",
        "algorithm.adv_estimator=" + args.algorithm,
        "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=True",
        "algorithm.kl_penalty=kl",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "reward.custom_reward_function.path="
        + str(UPSTREAM_DIR / "verl" / "utils" / "reward_score" / "rlla.py"),
        "reward.custom_reward_function.name=compute_score",
        "reward.num_workers=1",
        # The upstream ToolRL scorer selects its Qwen parsing branch from
        # extra_info["experiment_name"].  The official GDPO manager injects
        # that field and otherwise returns the same scalar `score`; for GRPO
        # the trainer still consumes only rm_scores, so this preserves the
        # scalar GRPO reward while avoiding a data-file rewrite.
        "reward.reward_manager.name=gdpo",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.use_v1=False",
        "trainer.resume_mode=disable",
        "trainer.val_before_train=False",
        "trainer.val_only=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.logger=[console]",
        "trainer.project_name=ProjectB_gpu_smoke",
        f"trainer.experiment_name=qwen2p5_1p5b_{args.algorithm}_one_step",
        "trainer.default_local_dir=" + str(PROJECT_ROOT / "env-modern" / "smoke-checkpoints" / args.algorithm),
        "ray_kwargs.ray_init.num_cpus=8",
    ]
    if args.algorithm == "gdpo":
        overrides.append('+algorithm.gdpo_reward_keys=["accuracy_reward","format_reward"]')

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "RAY_USAGE_STATS_ENABLED": "0",
            "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(UPSTREAM_DIR)
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    env.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
    env.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    print(f"smoke_algorithm={args.algorithm}", flush=True)
    print(f"smoke_command_log={log_path}", flush=True)
    print("smoke_trainer=official_verl_v0.9.1_main_ppo_v0", flush=True)
    print("smoke_max_steps=1", flush=True)
    started = time.monotonic()
    output_chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=UPSTREAM_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_chunks.append(line)
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
        return_code = process.wait()

    elapsed = time.monotonic() - started
    output = "".join(output_chunks)
    if return_code != 0:
        print(f"smoke_status=FAIL return_code={return_code} elapsed_seconds={elapsed:.1f}", flush=True)
        return return_code or 1

    # These are emitted by the official actor worker after backward and
    # optimizer.step(). Requiring one of them prevents a config-only PASS.
    optimizer_markers = ("actor/grad_norm", "actor/lr", "optimizer_step")
    optimizer_step_seen = any(marker in output for marker in optimizer_markers)
    if args.require_optimizer_step and not optimizer_step_seen:
        print(
            "smoke_status=FAIL reason=optimizer-step-marker-not-found "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        return 1

    for line in output.splitlines():
        if "actor/pg_loss" in line or "actor/grad_norm" in line or "data/" in line:
            print("smoke_metric=" + line, flush=True)
    print(
        f"smoke_status=PASS optimizer_step_seen={optimizer_step_seen} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
