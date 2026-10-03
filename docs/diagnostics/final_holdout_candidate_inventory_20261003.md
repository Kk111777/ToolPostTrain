# Final holdout candidate inventory (CPU/no-GPU)

This is a candidate statistics audit only. No parquet holdout was constructed, no model was loaded, no inference or reward scoring was run, and no final endpoint was frozen.

- generated_at_utc: `2026-10-03T04:03:07.090924+00:00`
- branch: `final-holdout-candidate-audit-20261003`
- audited commit: `6b7221c9b9fb8524426f4043f698f3d480792778`
- decision label: **CANDIDATE_POOL_USABLE_SMALL**

## Core counts

- raw train rows: **3920**
- eligible rows after official overlong filter: **3901**
- optimizer-exposed rows per epoch: **3584**
- initial drop_last tail: **317**
- final untouched candidates N: **108**

## Exposure exclusions

Counts are unique candidate source IDs hit by each category; category counts may overlap. The union count is the deduplicated exclusion total.

| category | raw hit count | source IDs |
|---|---:|---|
| `training_optimizer` | 0 | — |
| `sft_v1` | 55 | 3602, 3606, 3611, 3615, 3624, 3629, 3632, 3639, 3641, 3648, 3661, 3683, 3685, 3690, 3697, 3698, 3699, 3704, 3716, 3717, 3719, 3736, 3742, 3754, 3761, 3769, 3770, 3772, 3779, 3782, 3783, 3792, 3794, 3796, 3799, 3802, 3808, 3814, 3817, 3818, 3823, 3828, 3831, 3838, 3848, 3854, 3860, 3874, 3876, 3885, 3888, 3890, 3894, 3899, 3907 |
| `sft_v2` | 61 | 3611, 3615, 3618, 3624, 3626, 3641, 3646, 3648, 3649, 3650, 3661, 3675, 3686, 3690, 3699, 3702, 3716, 3717, 3723, 3724, 3732, 3734, 3742, 3751, 3754, 3756, 3760, 3761, 3765, 3778, 3779, 3782, 3789, 3794, 3808, 3809, 3814, 3816, 3817, 3818, 3823, 3828, 3835, 3838, 3848, 3850, 3856, 3860, 3871, 3873, 3874, 3882, 3886, 3888, 3891, 3894, 3895, 3901, 3902, 3914, 3917 |
| `formal_validation` | 80 | 3616, 3621, 3622, 3623, 3625, 3627, 3631, 3644, 3645, 3647, 3657, 3660, 3670, 3672, 3679, 3680, 3687, 3688, 3691, 3692, 3693, 3696, 3700, 3705, 3707, 3711, 3713, 3714, 3721, 3722, 3728, 3730, 3735, 3738, 3740, 3743, 3745, 3746, 3747, 3748, 3750, 3752, 3762, 3774, 3777, 3780, 3787, 3790, 3791, 3797, 3803, 3805, 3806, 3807, 3810, 3811, 3814, 3815, 3819, 3830, 3832, 3837, 3839, 3847, 3862, 3863, 3864, 3865, 3866, 3880, 3881, 3884, 3887, 3889, 3896, 3909, 3910, 3915, 3918, 3919 |
| `original_test` | 0 | — |
| `fixed_diagnostic` | 7 | 3654, 3690, 3704, 3742, 3754, 3818, 3848 |
| `constrained_generation_diagnostic` | 1 | 3654 |
| `fresh_holdout_reward_variation` | 1 | 3731 |
| `shared_rollout_diagnostic` | 1 | 3654 |
| `capacity_smoke` | 52 | 3605, 3607, 3612, 3614, 3617, 3619, 3633, 3637, 3638, 3639, 3646, 3663, 3664, 3667, 3683, 3698, 3703, 3704, 3706, 3715, 3716, 3717, 3720, 3726, 3733, 3741, 3744, 3757, 3764, 3768, 3775, 3779, 3800, 3813, 3824, 3834, 3841, 3852, 3853, 3859, 3861, 3868, 3872, 3874, 3875, 3885, 3890, 3897, 3900, 3902, 3903, 3913 |
| `throughput_tuning` | 0 | — |
| `one_step_smoke` | 80 | 3616, 3621, 3622, 3623, 3625, 3627, 3631, 3644, 3645, 3647, 3657, 3660, 3670, 3672, 3679, 3680, 3687, 3688, 3691, 3692, 3693, 3696, 3700, 3705, 3707, 3711, 3713, 3714, 3721, 3722, 3728, 3730, 3735, 3738, 3740, 3743, 3745, 3746, 3747, 3748, 3750, 3752, 3762, 3774, 3777, 3780, 3787, 3790, 3791, 3797, 3803, 3805, 3806, 3807, 3810, 3811, 3814, 3815, 3819, 3830, 3832, 3837, 3839, 3847, 3862, 3863, 3864, 3865, 3866, 3880, 3881, 3884, 3887, 3889, 3896, 3909, 3910, 3915, 3918, 3919 |
| `old_formal_grpo` | 0 | — |
| `historical_scoring_generation` | 1 | 3654 |
| `unknown_persisted_evidence` | 0 | — |

- union exposure exclusions: **209**
- remaining after exposure: **108**

## Content duplicate and length exclusions

- content duplicate exclusions against known corpora: **0**
- candidate-internal exact/normalized duplicate exclusions: **0**
- overlength exclusions after content audit: **0**

Content comparison uses the frozen NFC/newline normalization and sorted role/content serialization; no semantic, embedding, fuzzy, or outcome-based filtering was used.

## Final candidate profile

- target-category distribution: `{"response_only": 7, "tool_only": 101}`
- prompt-token distribution: `{"max": 1485, "median": 741.0, "min": 402, "n": 108, "p90": 1077.0, "p95": 1237.0999999999997}`

## Three-run optimizer exposure check

- all three runs completed: `True`
- all three use the same optimizer-exposed row semantics: `True`
- optimizer source-sequence SHA256: `bf3061a57347c865819c065bbb0f3cf3d789be31e1c7cb02945b782349470628`

## Interpretation

The label is an engineering decision category, not a formal statistical power guarantee. Even a large candidate pool remains subject to one training seed and possible semantic overlap not detected by exact/normalized hashes.

No next-stage holdout construction was performed automatically.
