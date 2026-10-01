# ProjectB format SFT v2 data report

v2 was generated from the original ToolRL train parquet. This step only prepares and audits data; it does not train, change the environment, reward manager, or GRPO/GDPO code.

## Source and sampling

- Source: `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet`
- Output: `/root/autodl-tmp/ProjectB/repo/dataset/format_sft_800_v2.jsonl`
- Seed: `42`
- Raw rows: `3920`
- Validation rows excluded: `16`
- Validation-row leakage in v2: `0`
- Maximum full chat-template sequence: `3072` tokens
- Rejected during eligibility scan: `{'too_long': 4, 'target_contract': 3}`
- Eligible rows by stratum: `{'response_only': 464, 'single_tool': 1782, 'multi_tool_2': 1091, 'multi_tool_3plus': 560}`

## 1. Class counts

| class | requested | actual | target kind |
|---|---:|---:|---|
| response_only | 200 | 200 | `response_only` |
| single_tool | 300 | 300 | `tool_only` |
| multi_tool_2 | 200 | 200 | `tool_only` |
| multi_tool_3plus | 100 | 100 | `tool_only` |
| total | 800 | 800 | `{'tool_only': 600, 'response_only': 200}` |

## 2. Token lengths by class

Lengths are measured after the Qwen chat template. Full length includes system/user/assistant template tokens; target length is the assistant loss span.

| class | full sequence min / mean / max | assistant target min / mean / max |
|---|---|---|
| response_only | `min=469, mean=1122.23, max=2925` | `min=30, mean=142.69, max=708` |
| single_tool | `min=460, mean=821.77, max=2635` | `min=39, mean=60.74, max=181` |
| multi_tool_2 | `min=495, mean=797.45, max=1762` | `min=60, mean=91.55, max=434` |
| multi_tool_3plus | `min=554, mean=955.06, max=1817` | `min=77, mean=137.49, max=504` |

## 3. Exact artifact audit

- Illegal strict-format targets: `0`
- Tag closure errors: `0`
- Markdown code blocks in assistant targets: `0`
- Target ending errors: `0`
- Chat-template prompt-prefix mismatches: `0`
- Full sequences over max length: `0`
- All target token spans non-empty: `True`

### Required target forms

- tool classes: `<think>...</think>\n<tool_call>\nJSONL\n</tool_call>`
- response-only: `<think>...</think>\n<response>...</response>`
- v2 contains no assistant prefix inside the assistant content and no fenced Markdown code block.

## 4. v1 comparison

| class | v1 | v2 | change |
|---|---:|---:|---|
| response-only | 100 | 200 | doubled |
| single-tool | 400 | 300 | reduced by 100 |
| multi-tool(2) | 200 | 200 | unchanged |
| multi-tool(3+) | 100 | 100 | unchanged |
| total | 800 | 800 | unchanged |

## Conclusion

v2 data preparation and audit passed. The only intended change from v1 is the class quota: response-only coverage is doubled while total size and multi-tool coverage remain unchanged. Training was not started.
