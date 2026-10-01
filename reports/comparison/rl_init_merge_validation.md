# ProjectB RL initialization merge validation

This report validates the standalone merged SFT v2 checkpoint against the original base model with the v2 LoRA adapter loaded at runtime. No optimizer, GRPO, or GDPO training was run.

## A. Merged checkpoint

- Path: `/root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged`
- Base: `/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- LoRA source: `/root/autodl-tmp/ProjectB/checkpoints/format_sft_v2_lora`

## B. Merge method

- PEFT method: `PeftModelForCausalLM.base_model.merge_and_unload(progressbar=False, safe_merge=True)`
- PEFT version: `0.19.1`
- Merge dtype: `bfloat16`
- Base parameter count: `1543714304`
- Merged parameter count: `1543714304`
- Reloaded parameter count: `1543714304`

## C. Independent load checks

- CPU `AutoModelForCausalLM.from_pretrained`: `True`
- Tokenizer reload: `True`
- Chat-template render: `True`
- Adapter/PEFT files in merged output: `[]`
- Saved files: `['chat_template.jinja', 'config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']`
- Merged checkpoint can be loaded without the PEFT adapter directory.

## D. Merge-before/after equivalence

- Fixed prompts: `16` previously audited prompts; rollout.n=1 for deterministic comparison
- Greedy generation: temperature `0`, top_p `1`, top_k `-1`, max_tokens `256`, seed `42`
- Exact response text match: `15/16`
- Exact response token-id match: `15/16`
- Strict-format result match: `16/16`
- Tool-parse result match: `16/16`
- Total reward match: `15/16`
- Maximum accuracy-reward absolute difference: `3.666666667`
- Maximum total-reward absolute difference: `3.666666667`
- Behavior drift detected: **True**

| source row | expected | text equal | tokens equal | format equal | parse equal | runtime total | merged total | runtime accuracy | merged accuracy |
|---:|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| 2619 | tool_only | True | True | True | True | 2.500000 | 2.500000 | 1.500000 | 1.500000 |
| 456 | tool_only | True | True | True | True | 4.000000 | 4.000000 | 3.000000 | 3.000000 |
| 102 | tool_only | True | True | True | True | 3.000000 | 3.000000 | 2.000000 | 2.000000 |
| 3037 | tool_only | True | True | True | True | 4.000000 | 4.000000 | 3.000000 | 3.000000 |
| 1126 | tool_only | True | True | True | True | 4.000000 | 4.000000 | 3.000000 | 3.000000 |
| 1003 | tool_only | True | True | True | True | 4.000000 | 4.000000 | 3.000000 | 3.000000 |
| 914 | tool_only | True | True | True | True | 4.000000 | 4.000000 | 3.000000 | 3.000000 |
| 571 | tool_only | True | True | True | True | 3.400000 | 3.400000 | 2.400000 | 2.400000 |
| 3016 | tool_only | True | True | True | True | 2.400000 | 2.400000 | 1.400000 | 1.400000 |
| 419 | tool_only | True | True | True | True | 4.000000 | 4.000000 | 3.000000 | 3.000000 |
| 2771 | tool_only | False | False | True | True | 4.000000 | 0.333333 | 3.000000 | -0.666667 |
| 3033 | response_only | True | True | True | True | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 3654 | tool_only | True | True | True | True | -1.200000 | -1.200000 | -2.200000 | -2.200000 |
| 2233 | response_only | True | True | True | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 356 | tool_only | True | True | True | True | 2.000000 | 2.000000 | 1.000000 | 1.000000 |
| 2418 | tool_only | True | True | True | True | 1.375000 | 1.375000 | 0.375000 | 0.375000 |

- Non-identical response rows: `[2771]`; all rows retained the same strict-format and tool-parse decisions.
- Reward-different rows: `[2771]`; this is a semantic top-1 difference, not only a serialized-token difference.
- A repeat run reproduced the row-2771 difference while a row-3033 text difference from the first run did not recur, so exact vLLM greedy text is not fully deterministic under this runtime.

## E. RL initialization decision

The PEFT merge is valid and the merged checkpoint is independently loadable. On this fixed sample, protocol behavior is equivalent (16/16 strict-format and tool-parse decisions), but strict semantic/output equivalence is conditional rather than exact because one stable row changes tool arguments and reward. The merged checkpoint may be retained as a candidate common GRPO/GDPO initialization only with this caveat recorded; this validation does not justify claiming bitwise or fully semantic identity with runtime-LoRA inference.
