# GDPO toy example: why summed-reward normalization can collapse signals

This is a hand-computed example. It does not call a model and does not run a
training step. It uses the same sample-standard-deviation convention as the
official implementation (`torch.std` in
`verl-GDPO/verl/trainer/ppo/core_algos.py:146-148`) and the same final masked
whitening idea used by GDPO (`verl-GDPO/verl/utils/torch_functional.py:130-136`).

## Setup

Use one prompt and four rollouts. Treat the response-level reward as if each
response has one valid response token; in the actual code the scalar is placed
at the final valid response token and then broadcast over the response mask.

| rollout | format reward F | correctness reward C | summed reward F+C |
|---:|---:|---:|---:|
| 1 | 1 | 0 | 1 |
| 2 | 0 | 1 | 1 |
| 3 | 1 | 3 | 4 |
| 4 | 0 | -3 | -3 |

The values are within the official reward ranges: format is binary `[0, 1]`
and correctness is `[-3, 3]`. The first two rollouts are different reward
combinations but have the same sum.

## GRPO

GRPO first adds the rewards and then normalizes the group. The summed rewards
are:

```text
[1, 1, 4, -3]
```

Their group mean is `0.75` and their sample standard deviation is
`2.872281`. Therefore:

```text
GRPO advantage = (reward - 0.75) / (2.872281 + 1e-6)
               = [0.087039, 0.087039, 1.131504, -1.305582]
```

Rollouts 1 and 2 are different (`format-only` versus `correctness-only`), but
GRPO gives them exactly the same advantage because their summed reward is the
same. This is the collapse caused by aggregating heterogeneous rewards before
normalization.

## GDPO

GDPO normalizes each reward within the same four-sample group first.

### Format branch

```text
F = [1, 0, 1, 0]
mean(F) = 0.5
std(F)  = 0.577350
F_adv   = [ 0.866024, -0.866024,  0.866024, -0.866024 ]
```

### Correctness branch

```text
C = [0, 1, 3, -3]
mean(C) = 0.25
std(C)  = 2.5
C_adv   = [-0.100000,  0.300000,  1.100000, -1.300000]
```

With the official equal implicit weights, the pre-batch-normalization
advantage is the sum of the two normalized branches:

```text
F_adv + C_adv
= [ 0.766024, -0.566024, 1.966023, -2.166023 ]
```

The implementation then applies `masked_whiten` to this combined tensor. Its
mean is `0`, its sample standard deviation is approximately `1.776146`, so the
final GDPO advantage is:

```text
GDPO advantage = [0.431284, -0.318681, 1.106904, -1.219508]
```

Now rollouts 1 and 2 remain distinguishable: GDPO gives the format-only
rollout a positive signal and the correctness-only rollout a negative signal.
The absolute values differ from GRPO because GDPO deliberately changes the
normalization geometry; the important point is that the two reward
combinations are no longer identified solely by their raw sum.

## Code mapping

- GRPO: `verl-GDPO/verl/trainer/ppo/ray_trainer.py:148-159` calls
  `compute_grpo_outcome_advantage` on the already combined
  `token_level_rewards`.
- The GRPO group mean/std are computed in
  `verl-GDPO/verl/trainer/ppo/core_algos.py:130-153`.
- GDPO: `verl-GDPO/verl/trainer/ppo/ray_trainer.py:175-202` calls that same
  group-normalizer separately for correctness and format, adds the two
  normalized tensors, then applies `masked_whiten`.
- Both methods finally pass `data.batch['advantages']` into the PPO actor loss
  at `verl-GDPO/verl/workers/actor/dp_actor.py:239-252`.
