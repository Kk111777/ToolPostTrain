# FINAL_HOLDOUT_V1 freeze verification

- gate: **FINAL_HOLDOUT_V1_FROZEN_PASS**
- row count: **108**
- ordered source ID SHA256: `75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad`
- derived parquet SHA256: `e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22`
- target distribution: `{"response_only": 7, "tool_only": 101}`
- prompt length range: `{'n': 108, 'min': 402, 'max': 1485}`
- model inference: **NO**
- reward scoring: **NO**

## Checks

| check | status | detail |
|---|---|---|
| `row_count_108` | PASS | `108` |
| `candidate_identity_exact` | PASS | `{"candidate_count": 108, "final_count": 108, "ordered_source_ids_sha256": "75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad"}` |
| `source_order_preserved` | PASS | `"final equals source-row take"` |
| `schema_preserved` | PASS | `"data_source: string\nprompt: list<element: struct<content: string, role: string>>\n  child 0, element: struct<content: string, role: string>\n      child 0, content: string\n      child 1, role: string\nability: string\nreward_model: struct<ground_truth: string, style: string>\n  child 0, ground_truth: string\n  child 1, style: string\nextra_info: struct<index: int64, input: string, instruction: string, output: string, split: string>\n  child 0, index: int64\n  child 1, input: string\n  child 2, instruction: string\n  child 3, output: string\n  child 4, split: string\n-- schema metadata --\npandas: '{\"index_columns\": [{\"kind\": \"range\", \"name\": null, \"start\": 0, \"' + 857"` |
| `candidate_manifest_source_id_hash` | PASS | `"75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad"` |
| `derived_sha256_matches_manifest` | PASS | `"e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22"` |
| `source_sha256_matches_manifest` | PASS | `"f2e728ad4a379550887711870db5f97154a0ecc45efbe5d43db8a99dc3b9e1c8"` |
| `manifest_order_hash_matches_final` | PASS | `"75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad"` |
| `manifest_rows_match_final` | PASS | `{"final_count": 108, "manifest_count": 108}` |
| `zero_union_exposure` | PASS | `[]` |
| `zero_training_optimizer_exposure` | PASS | `[]` |
| `zero_sft_v1_exposure` | PASS | `[]` |
| `zero_sft_v2_exposure` | PASS | `[]` |
| `zero_formal_validation_exposure` | PASS | `[]` |
| `zero_original_test_exposure` | PASS | `[]` |
| `zero_fixed_diagnostic_exposure` | PASS | `[]` |
| `zero_constrained_generation_diagnostic_exposure` | PASS | `[]` |
| `zero_fresh_holdout_reward_variation_exposure` | PASS | `[]` |
| `zero_shared_rollout_diagnostic_exposure` | PASS | `[]` |
| `zero_capacity_smoke_exposure` | PASS | `[]` |
| `zero_throughput_tuning_exposure` | PASS | `[]` |
| `zero_one_step_smoke_exposure` | PASS | `[]` |
| `zero_old_formal_grpo_exposure` | PASS | `[]` |
| `zero_historical_scoring_generation_exposure` | PASS | `[]` |
| `zero_unknown_persisted_evidence_exposure` | PASS | `[]` |
| `prompt_length_3601` | PASS | `917` |
| `prompt_length_3603` | PASS | `684` |
| `prompt_length_3604` | PASS | `1077` |
| `prompt_length_3608` | PASS | `1047` |
| `prompt_length_3609` | PASS | `1249` |
| `prompt_length_3610` | PASS | `854` |
| `prompt_length_3613` | PASS | `970` |
| `prompt_length_3620` | PASS | `788` |
| `prompt_length_3628` | PASS | `801` |
| `prompt_length_3630` | PASS | `837` |
| `prompt_length_3634` | PASS | `1415` |
| `prompt_length_3635` | PASS | `569` |
| `prompt_length_3636` | PASS | `452` |
| `prompt_length_3640` | PASS | `751` |
| `prompt_length_3642` | PASS | `754` |
| `prompt_length_3643` | PASS | `617` |
| `prompt_length_3651` | PASS | `502` |
| `prompt_length_3652` | PASS | `697` |
| `prompt_length_3653` | PASS | `979` |
| `prompt_length_3655` | PASS | `815` |
| `prompt_length_3656` | PASS | `635` |
| `prompt_length_3658` | PASS | `488` |
| `prompt_length_3659` | PASS | `1292` |
| `prompt_length_3662` | PASS | `503` |
| `prompt_length_3665` | PASS | `469` |
| `prompt_length_3666` | PASS | `1215` |
| `prompt_length_3668` | PASS | `746` |
| `prompt_length_3669` | PASS | `606` |
| `prompt_length_3671` | PASS | `579` |
| `prompt_length_3673` | PASS | `640` |
| `prompt_length_3674` | PASS | `617` |
| `prompt_length_3676` | PASS | `439` |
| `prompt_length_3677` | PASS | `1077` |
| `prompt_length_3678` | PASS | `1203` |
| `prompt_length_3681` | PASS | `760` |
| `prompt_length_3682` | PASS | `616` |
| `prompt_length_3684` | PASS | `765` |
| `prompt_length_3689` | PASS | `753` |
| `prompt_length_3694` | PASS | `523` |
| `prompt_length_3695` | PASS | `737` |
| `prompt_length_3701` | PASS | `402` |
| `prompt_length_3708` | PASS | `420` |
| `prompt_length_3709` | PASS | `710` |
| `prompt_length_3710` | PASS | `770` |
| `prompt_length_3712` | PASS | `570` |
| `prompt_length_3718` | PASS | `451` |
| `prompt_length_3725` | PASS | `920` |
| `prompt_length_3727` | PASS | `1485` |
| `prompt_length_3729` | PASS | `1004` |
| `prompt_length_3737` | PASS | `599` |
| `prompt_length_3739` | PASS | `954` |
| `prompt_length_3749` | PASS | `999` |
| `prompt_length_3755` | PASS | `701` |
| `prompt_length_3758` | PASS | `1161` |
| `prompt_length_3759` | PASS | `648` |
| `prompt_length_3763` | PASS | `782` |
| `prompt_length_3766` | PASS | `1081` |
| `prompt_length_3767` | PASS | `1294` |
| `prompt_length_3771` | PASS | `803` |
| `prompt_length_3773` | PASS | `704` |
| `prompt_length_3776` | PASS | `769` |
| `prompt_length_3781` | PASS | `757` |
| `prompt_length_3784` | PASS | `940` |
| `prompt_length_3785` | PASS | `692` |
| `prompt_length_3786` | PASS | `482` |
| `prompt_length_3788` | PASS | `1046` |
| `prompt_length_3793` | PASS | `802` |
| `prompt_length_3795` | PASS | `519` |
| `prompt_length_3798` | PASS | `921` |
| `prompt_length_3801` | PASS | `603` |
| `prompt_length_3804` | PASS | `741` |
| `prompt_length_3812` | PASS | `480` |
| `prompt_length_3820` | PASS | `731` |
| `prompt_length_3821` | PASS | `583` |
| `prompt_length_3822` | PASS | `1065` |
| `prompt_length_3825` | PASS | `738` |
| `prompt_length_3826` | PASS | `1037` |
| `prompt_length_3827` | PASS | `605` |
| `prompt_length_3829` | PASS | `832` |
| `prompt_length_3833` | PASS | `927` |
| `prompt_length_3836` | PASS | `550` |
| `prompt_length_3842` | PASS | `430` |
| `prompt_length_3843` | PASS | `752` |
| `prompt_length_3844` | PASS | `479` |
| `prompt_length_3845` | PASS | `460` |
| `prompt_length_3846` | PASS | `712` |
| `prompt_length_3849` | PASS | `581` |
| `prompt_length_3851` | PASS | `562` |
| `prompt_length_3855` | PASS | `776` |
| `prompt_length_3857` | PASS | `705` |
| `prompt_length_3858` | PASS | `476` |
| `prompt_length_3867` | PASS | `790` |
| `prompt_length_3869` | PASS | `451` |
| `prompt_length_3870` | PASS | `881` |
| `prompt_length_3877` | PASS | `706` |
| `prompt_length_3878` | PASS | `1021` |
| `prompt_length_3879` | PASS | `741` |
| `prompt_length_3883` | PASS | `738` |
| `prompt_length_3892` | PASS | `775` |
| `prompt_length_3893` | PASS | `1387` |
| `prompt_length_3898` | PASS | `819` |
| `prompt_length_3904` | PASS | `1075` |
| `prompt_length_3905` | PASS | `675` |
| `prompt_length_3906` | PASS | `615` |
| `prompt_length_3908` | PASS | `510` |
| `prompt_length_3911` | PASS | `729` |
| `prompt_length_3912` | PASS | `638` |
| `prompt_length_3916` | PASS | `995` |
| `all_prompt_lengths_le_2048` | PASS | `{"max": 1485, "min": 402}` |
| `zero_exact_normalized_content_overlap` | PASS | `[]` |
| `zero_internal_duplicate` | PASS | `[]` |
| `candidate_content_duplicate_audit_empty` | PASS | `[]` |
| `no_model_output_present` | PASS | `"final-test JSONL absent"` |
