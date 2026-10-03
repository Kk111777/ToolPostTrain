# Final-test historical usage audit

| level | result |
|---|---|
| A: code/config reference | True |
| B: runtime log shows test loaded | True |
| C: test metric/generation evidence | True |
| D: explicit later decision use found | False |

Historical runtime use of test.parquet is confirmed by persisted commands/logs, including test row-count/scoring markers. Formal Stage1 intermediate validation used formal_validation_80. No direct evidence was found that historical test scores drove later budget, hyperparameter, or checkpoint decisions in the audited decision files; absence of evidence is not proof of non-use. The test cannot be represented as fully unseen without manual review.

Gate: GPU_FINAL_TEST_BLOCKED. No test inference was launched.

Evidence paths:
- /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/launch_command.txt: runtime val matches=1, score/metric matches=0
- /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/run_formal_grpo.sh: runtime val matches=1, score/metric matches=0
- /root/autodl-tmp/ProjectB/runs/grpo_formal_v1/logs/supervisor.log: runtime val matches=2, score/metric matches=21
- /root/autodl-tmp/ProjectB/env-modern/gpu-validation-logs/validation.log: runtime val matches=2, score/metric matches=3
- /root/autodl-tmp/ProjectB/env-modern/gpu-validation-logs/grpo-driver.log: runtime val matches=2, score/metric matches=2
- /root/autodl-tmp/ProjectB/env-modern/gpu-validation-logs/gdpo-driver.log: runtime val matches=2, score/metric matches=1
- /root/autodl-tmp/ProjectB/env-modern/official-scale-capacity-logs/grpo.stdout.log: runtime val matches=3, score/metric matches=0
- /root/autodl-tmp/ProjectB/env-modern/official-scale-capacity-logs/gdpo_eligible512.stdout.log: runtime val matches=2, score/metric matches=18
- /root/autodl-tmp/ProjectB/env-modern/throughput-tuning-logs/grpo_config_a.stdout.log: runtime val matches=2, score/metric matches=0
- /root/autodl-tmp/ProjectB/env-modern/throughput-tuning-logs/gdpo_throughput_config_a.stdout.log: runtime val matches=1, score/metric matches=24
- /root/autodl-tmp/ProjectB/repo/scripts/run_single_gpu_smoke.sh: runtime val matches=1, score/metric matches=0
