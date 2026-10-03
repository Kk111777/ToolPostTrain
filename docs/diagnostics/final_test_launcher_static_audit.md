# Final-test launcher static audit

Date: 2026-10-03 (Asia/Shanghai)

Status: STATIC_PREFLIGHT_PASS; GPU_FINAL_TEST_BLOCKED_PENDING_PROVENANCE_REVIEW

## Checks

- Bash syntax for scripts/run_final_test.sh: PASS.
- Python compile for scripts/analyze_final_test.py and scripts/final_test_preflight_audit.py: PASS.
- Analysis script CPU self-test: PASS.
- Hydra compose/resolve: PASS with official upstream veRL v0.9.1.
- Resolved configuration size: 30079 bytes.
- No model load, generation, training, resume, checkpoint write, or GPU task was run.
- Planned output directory is isolated under /root/autodl-tmp/ProjectB/final-test/ and is not a Stage1 run directory.
- Launcher has two explicit opt-in environment guards and refuses by default.

## Runtime source audit

The launcher executes from /root/autodl-tmp/ProjectB/upstream/verl-v0.9.1, not the legacy repository trainer.

In upstream/verl-v0.9.1/verl/trainer/ppo/ray_trainer.py:

- lines 1438-1442 run initial validation when trainer.val_before_train is true.
- lines 1443-1445 return immediately when trainer.val_only is true.
- the training loop begins at line 1451, after that return.
- line 364 gates the in-reward KL controller on algorithm.use_kl_in_reward.
- upstream/verl-v0.9.1/verl/trainer/ppo/utils.py lines 75-80 define
  need_reference_policy as the OR of algorithm.use_kl_in_reward and
  actor_rollout_ref.actor.use_kl_loss; both false therefore disable reference
  policy construction for the final-test configuration.

The resolved final-test settings are val_before_train=true, val_only=true,
total_epochs=0, total_training_steps=0, n=1, temperature=0,
do_sample=false, seed=42, max prompt 2048, max response 1024,
test.parquet as the sole validation file, and no checkpoint save frequency.

## Provenance gate

The CPU historical audit found persisted commands and logs that loaded and
scored test.parquet before this preflight. Therefore the final test cannot be
represented as fully unseen without manual review. No new test inference was
launched and no historical test result was used to change this preflight.

Gate: GPU_FINAL_TEST_BLOCKED.
