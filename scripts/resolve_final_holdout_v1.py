#!/usr/bin/env python3
"""Compose the FINAL_HOLDOUT_V1 val-only Hydra config without running it."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path("/root/autodl-tmp/ProjectB")
UPSTREAM = ROOT / "upstream/verl-v0.9.1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_penalty=kl",
        "algorithm.kl_ctrl.kl_coef=0.001",
        f"data.train_files={ROOT}/repo/verl-GDPO/dataset/rlla_4k/train.parquet",
        f"data.val_files={ROOT}/env-modern/final_holdout_v1.parquet",
        "data.train_max_samples=-1",
        "data.val_max_samples=108",
        "data.train_batch_size=512",
        "data.val_batch_size=108",
        "data.max_prompt_length=2048",
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.shuffle=False",
        "data.validation_shuffle=False",
        "data.seed=42",
        f"actor_rollout_ref.model.path={ROOT}/checkpoints/rl_init_sft_v2_merged",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.seed=42",
        "actor_rollout_ref.rollout.prompt_length=2048",
        "actor_rollout_ref.rollout.response_length=1024",
        "actor_rollout_ref.rollout.max_model_len=3072",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.max_num_batched_tokens=6144",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        f"reward.custom_reward_function.path={UPSTREAM}/verl/utils/reward_score/rlla.py",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_manager.name=gdpo",
        "reward.num_workers=1",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.max_actor_ckpt_to_keep=0",
        "trainer.max_critic_ckpt_to_keep=0",
        "trainer.resume_mode=disable",
        "trainer.total_epochs=0",
        "trainer.total_training_steps=0",
        "trainer.project_name=ProjectB_final_holdout_v1",
        "trainer.experiment_name=qwen2p5_1p5b_grpo_one_step",
        f"trainer.default_local_dir={ROOT}/final-test/STATIC_ONLY",
        "trainer.logger=[console]",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.use_v1=False",
        "ray_kwargs.ray_init.num_cpus=8",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(UPSTREAM / "verl/trainer/config")):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    print("FINAL_HOLDOUT_V1_HYDRA_COMPOSE_RESOLVE_PASS")


if __name__ == "__main__":
    main()
