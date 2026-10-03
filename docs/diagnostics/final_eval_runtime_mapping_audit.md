# FINAL_HOLDOUT_V1 runtime mapping audit

Gate: **REAL_TRAINER_JSONL_MAPPING_PASS**

This CPU-only audit maps persisted veRL formal-validation JSONL by `(runtime_input_sha256, ground_truth_sha256_exact)`.
It does not use file position, fuzzy matching, model inference, or FINAL_HOLDOUT_V1 scoring.

- endpoint manifest SHA256: `160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16`
- runtime mapping SHA256: `ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd`
- runtime mapping pairs: `108/108` unique
- endpoint parquet SHA256: `e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22`
- ordered source-ID SHA256: `75373e7e89ed7004a2d5cb858b9626449d74f7e536d341fda7aec18eed5e29ad`

| persisted JSONL | rows matched | missing | duplicates | unique source IDs | result |
|---|---:|---:|---:|---:|---|
| `grpo_stage1_35` step 0 | 80/80 | 0 | 0 | 80 | PASS |
| `gdpo_stage1_35` step 0 | 80/80 | 0 | 0 | 80 | PASS |
| `grpo_nokl_stage1_35` step 0 | 80/80 | 0 | 0 | 80 | PASS |

The mapping was constructed with official veRL v0.9.1 RLHFDataset and the text-only ContinuousToken builder using the frozen RL_INIT_V1 tokenizer; no model was loaded.
