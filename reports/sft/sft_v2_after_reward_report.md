# ProjectB SFT v2 reward validation

Inference-only validation on the fixed ToolRL validation sample. No optimizer, GRPO, or GDPO training was run.

## Validation contract

- Fixed prompts: `16` prompts, source rows `[2619, 456, 102, 3037, 1126, 1003, 914, 571, 3016, 419, 2771, 3033, 3654, 2233, 356, 2418]`, seed `42`
- Rollout: vLLM, `tensor_parallel_size=1`, `n=2`, seed `42`
- Sampling: temperature `1.0`, top_p `1.0`, top_k `-1`
- Model: `/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Reward manager: official `GDPORewardManager` + `rlla.compute_score`
- Dataset, prompts, generation configuration, and reward manager were unchanged.

## Adapter

- Path: `/root/autodl-tmp/ProjectB/checkpoints/format_sft_v2_lora`
- Runtime load status: **loaded_runtime**
- Generation with v2 LoRA request succeeded: `True`
- No permanent merge was performed.

## Before / SFT v1 / SFT v2

| version | format > 0 | tool-only parse | response-only wrapper | accuracy > 0 | reward mean | reward std | output types |
|---|---:|---:|---:|---:|---:|---:|---|
| Before baseline | 0/32 | 1/28 (3.6%) | 0/4 (0.0%) | 28/32 | -2.468750 | 1.274372 | `{'tool_call': 1, 'response': 4, 'invalid': 27}` |
| SFT v1 | 24/32 | 22/28 (78.6%) | 0/4 (0.0%) | 27/32 | 1.103119 | 2.443001 | `{'tool_call': 26, 'response': 0, 'invalid': 6}` |
| SFT v2 | 27/32 | 23/28 (82.1%) | 2/4 (50.0%) | 28/32 | 1.539348 | 2.327668 | `{'tool_call': 25, 'response': 3, 'invalid': 4}` |

## Metric details

- v2 overall tool-call parse: `23/32`
- v2 accuracy reward mean/std: `0.695598` / `2.122811`
- v2 non-zero reward: `30/32` (93.75%)
- v2 all rewards finite: `True`
- v2 generation time: `22.125 seconds`

## Per-response v2 records

The JSON report contains complete response text and parsed tool-call results.

| sample | rollout | expected | output type | format | accuracy | total | tool JSON | response preview |
|---:|---:|---|---|---:|---:|---:|:---:|---|
| 2619 | 0 | tool_only | tool_call | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "maya.render_scene", "parameters": {"scene_id": "scene123", "render_settings": {"image_format": "PNG", "width": 1920, "height": 1080,  |
| 2619 | 1 | tool_only | tool_call | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "maya.render_scene", "parameters": {"scene_id": "scene123", "render_settings": {"output_format": "png", "resolution": {"width": "1920" |
| 456 | 0 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "10Xikol0h", "parameters": {"ZhOw8E": 10}} {"name": "10Xikol0h", "parameters": {"ZhOw8E": 7}} </tool_call> |
| 456 | 1 | tool_only | tool_call | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "10Xikol0h", "parameters": {"ZhOw8E": "10"}  <--- Include appropriate parameters } {"name": "10Xikol0h", "parameters": {"ZhOw8E": "7"} |
| 102 | 0 | tool_only | tool_call | 1.000 | 1.333 | 2.333 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "0Fhl44aiA5C", "parameters": {"zy61N.bU6f": "Mumbai"}} {"name": "3Ux4x5SjN", "parameters": {"N4DpteeZ5C": "Bangkok"}} {"name": "vNli_" |
| 102 | 1 | tool_only | tool_call | 1.000 | 1.222 | 2.222 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "0Fhl44aiA5C", "parameters": {"zy61N.bU6f": "Mumbai"}} {"name": "3Ux4x5SjN", "parameters": {"N4DpteeZ5C": "Bangkok", "f9.RRLq": "Bangl |
| 3037 | 0 | tool_only | tool_call | 1.000 | 0.375 | 1.375 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "recitations_by_juz_number", "parameters": {"recitation_id": 4, "juz_number": 30}} {"name": "getuserbyname", "parameters": {"username" |
| 3037 | 1 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "recitations_by_chapter_number", "parameters": {"chapter_number": 30, "recitation_id": 4}} {"name": "getuserbyname", "parameters": {"u |
| 1126 | 0 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "American Football Search API", "parameters": {"term": "Tom Brady"}} </tool_call> |
| 1126 | 1 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "American Football Search API", "parameters": {"term": "Tom Brady"}} </tool_call> |
| 1003 | 0 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "GetSupportedTypes", "parameters": {}} {"name": "Street-Level Crimes API", "parameters": {"lat": "51.5074", "lng": "-0.1278"}} </tool_ |
| 1003 | 1 | tool_only | tool_call | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get-supported-types", "parameters": {}} {"name": "Street-Level Crimes API", "parameters": {"lat": "51.5074", "lng": "-0.1278"}} </too |
| 914 | 0 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Story Details", "parameters": {"story_id": "115"}} {"name": "Get Story Details", "parameters": {"story_id": "204"}} {"name": "Get |
| 914 | 1 | tool_only | tool_call | 1.000 | 0.429 | 1.429 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Story Details", "parameters": {"story_id": 115}} {"name": "Get Story Details", "parameters": {"story_id": 204}} {"name": "Get Sto |
| 571 | 0 | tool_only | tool_call | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "revenue_estimate", "parameters": {"ticker": "AMZN"}} {"name": "gethistoricalscoresbyyear", "parameters": {"year": "2020"}} </tool_cal |
| 571 | 1 | tool_only | invalid | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "revenue_estimate", "parameters": {"ticker": "AMZN"}} {"name": "gethistoricalscoresbyyear", "parameters": {"year": "2020"}} <tool_call |
| 3016 | 0 | tool_only | tool_call | 1.000 | 1.400 | 2.400 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "lTV6R0o", "parameters": {"m8vNAVB6cP": 300, "1CZ5pIgV": "science,spirituality"}} </tool_call> |
| 3016 | 1 | tool_only | tool_call | 1.000 | 2.700 | 3.700 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "lTV6R0o", "parameters": {"m8vNAVB6cP": 300, "1CZ5pIgV": "science,spirituality", "pedMeJraE": 0, "H7zV6yV": 4}} </tool_call> |
| 419 | 0 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "ideas_get_view_count", "parameters": {"uuid": "abcde", "lang": "it"}} </tool_call> |
| 419 | 1 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "ideas_get_view_count", "parameters": {"uuid": "abcde", "lang": "it"}} </tool_call> |
| 2771 | 0 | tool_only | tool_call | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "main_endpoint", "parameters": {"asin": "B08Y6Z7890", "domain": "ca"}} {"name": "search_for_a_product", "parameters": {"q": "smart TV" |
| 2771 | 1 | tool_only | response | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> Relevant information was provided to respond to the user's need directly. Please search again according to the keywords and parameters directly if needed. </response> |
| 3033 | 0 | response_only | invalid | 0.000 | 0.000 | 0.000 | False | <think> I should directly respond to the user's need. </think> <response> The latest real-time candlestick data for the stock symbol 'TSLA' on a 15-minute frame is as follows: - Open: 185.23 - High: 187.45 - Low: 184.78 - Close: 186.89 - Volume: 1,324,567 unit |
| 3033 | 1 | response_only | invalid | 0.000 | 0.000 | 0.000 | False | <think> I should directly respond to the user's need. </think> <response> Here is the latest real-time candlestick data for the trading symbol TSLA on a 15-minute time frame: ``` {   "open": 185.23,   "high": 187.45,   "low": 184.78,   "close": 186.89,   "volu |
| 3654 | 0 | tool_only | tool_call | 1.000 | -2.500 | -1.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Rugby Manager Last Matches", "parameters": {"id": 2021}} </tool_call> |
| 3654 | 1 | tool_only | tool_call | 1.000 | -2.200 | -1.200 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Rugby Manager Last Matches", "parameters": {"id": 2021.0, "page": 1.0}} </tool_call> |
| 2233 | 0 | response_only | response | 1.000 | 0.000 | 1.000 | False | <think> I should directly respond to the user's need. </think> <response> I should directly respond to the user's need. </response> |
| 2233 | 1 | response_only | response | 1.000 | 0.000 | 1.000 | False | <think> I should directly respond to the user's need. </think> <response> Task planning is a function of the appropriate tool called, not directly responding to a user's need. </response> |
| 356 | 0 | tool_only | tool_call | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "find_majority_element", "parameters": {"nums": "[2, 2, 1, 1, 1, 2, 2]"}} </tool_call> |
| 356 | 1 | tool_only | tool_call | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "find_majority_element", "parameters": {"nums": "[2, 2, 1, 1, 1, 2, 2]"}} </tool_call> |
| 2418 | 0 | tool_only | invalid | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>  |
| 2418 | 1 | tool_only | tool_call | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "bP9F_z_", "parameters": {"VVNTFVcb": "Sunnyvale, CA", "3AfT": ["bread", "2", "cheese", "4", "milk", "2"]} </tool_call> |

## Decision gate

- This report only measures the reward signal; it does not authorize formal training.
- If the format gate passes, the next action is only a separately authorized no-optimizer GRPO/GDPO 1-step validation.
