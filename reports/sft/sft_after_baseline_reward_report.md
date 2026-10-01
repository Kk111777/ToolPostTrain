# ProjectB post-format-SFT reward validation

This is inference-only validation. No optimizer, backward pass, or further training was run.

## Validation contract

- Fixed validation prompts: `16` train rows, seed `42`
- Selected source rows: `[2619, 456, 102, 3037, 1126, 1003, 914, 571, 3016, 419, 2771, 3033, 3654, 2233, 356, 2418]`
- Rollout: vLLM, `tensor_parallel_size=1`, `n=2`, seed `42`
- Sampling: temperature `1.0`, top_p `1.0`, top_k `-1`
- Model: `/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Reward manager: official `GDPORewardManager` + `rlla.compute_score`
- Dataset, reward manager, prompts, and sampling configuration were unchanged.

## Adapter load

- Adapter: `/root/autodl-tmp/ProjectB/checkpoints/format_sft_lora`
- Adapter config found: `True`
- Runtime LoRA enabled: `True`
- LoRA request submitted: `True`
- Generation with adapter request succeeded: `True`
- Load status: **loaded_runtime**
- This validation used vLLM runtime LoRA loading; no permanent merge into the base model was performed.

## Before vs after

| metric | before SFT | after SFT | delta |
|---|---:|---:|---:|
| format_reward > 0 | 0 | 24 | +24 |
| tool_call parse success | 1 | 23 | +22 |
| accuracy_reward > 0 | 28 | 27 | -1 |
| reward mean | -2.468750 | 1.103119 | +3.571869 |
| reward std | 1.274372 | 2.443001 | +1.168629 |
| non-zero reward ratio | 87.50% | 84.38% | -3.12% |

## Output-degradation checks

- Empty responses after SFT: `0`
- All after-SFT rewards finite: `True`
- Reward manager succeeded for every after-SFT trajectory: `True`
- Accuracy non-zero delta: `-1`
- Obvious mechanical degradation detected: **False**
- Accuracy changes are shown as measurements; they are not silently reclassified as format failures.

## After-SFT trajectory records

| sample | rollout | prompt len | response len | format | accuracy | total | tool JSON | response preview |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---|
| 2619 | 0 | 596 | 85 | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "maya.render_scene", "parameters": {"scene_id": "scene123", "render_settings": {"image_format": "PNG", "width": 19 |
| 2619 | 1 | 596 | 83 | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "maya.render_scene", "parameters": {"scene_id": "scene123", "render_settings": "{\"output_format\": \"png\", \"res |
| 456 | 0 | 475 | 78 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "10Xikol0h", "parameters": {"ZhOw8E": 10}} {"name": "10Xikol0h", "parameters": {"ZhOw8E": 7}} </tool_call> |
| 456 | 1 | 475 | 94 | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "10Xikol0h", "parameters": {"ZhOw8E": "10"}  <--- For first draw {"name": "10Xikol0h", "parameters": {"ZhOw8E": "7 |
| 102 | 0 | 739 | 127 | 1.000 | 1.333 | 2.333 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "0Fhl44aiA5C", "parameters": {"zy61N.bU6f": "Mumbai"}} {"name": "3Ux4x5SjN", "parameters": {"N4DpteeZ5C": "Bangkok |
| 102 | 1 | 739 | 144 | 1.000 | 2.111 | 3.111 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "0Fhl44aiA5C", "parameters": {"zy61N.bU6f": "Mumbai"}} {"name": "3Ux4x5SjN", "parameters": {"N4DpteeZ5C": "Bangkok |
| 3037 | 0 | 732 | 75 | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "recitations_by_chapter_number", "parameters": {"chapter_number": "30", "recitation_id": "4"}} {"name": "getuserby |
| 3037 | 1 | 732 | 76 | 1.000 | -1.312 | -0.312 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "recitations_by_juz_number", "parameters": {"recitation_id": 4, "juz_number": 30}} {"name": "getuserbyname", "para |
| 1126 | 0 | 674 | 46 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "American Football Search API", "parameters": {"term": "Tom Brady"}} </tool_call> |
| 1126 | 1 | 674 | 46 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "American Football Search API", "parameters": {"term": "Tom Brady"}} </tool_call> |
| 1003 | 0 | 675 | 75 | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "GetSupportedTypes", "parameters": {}} {"name": "Street-Level Crimes API", "parameters": {"lat": "51.5074", "lng": |
| 1003 | 1 | 675 | 72 | 1.000 | 0.000 | 1.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "GetSupportedTypes", "parameters": {}} {"name": "Street-Level Crimes API", "parameters": {"lat": "51.5074, -0.1278 |
| 914 | 0 | 573 | 65 | 1.000 | -1.857 | -0.857 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Story Details", "parameters": {"story_id": "115", "story_id": "204", "story_id": "317"}} </tool_call> |
| 914 | 1 | 573 | 87 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Story Details", "parameters": {"story_id": "115"}} {"name": "Get Story Details", "parameters": {"story_id": " |
| 571 | 0 | 1073 | 68 | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "revenue_estimate", "parameters": {"ticker": "AMZN"}} {"name": "gethistoricalscoresbyyear", "parameters": {"year": |
| 571 | 1 | 1073 | 68 | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "revenue_estimate", "parameters": {"ticker": "AMZN"}} {"name": "gethistoricalscoresbyyear", "parameters": {"year": |
| 3016 | 0 | 645 | 94 | 1.000 | 2.700 | 3.700 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "lTV6R0o", "parameters": {"m8vNAVB6cP": 300, "1CZ5pIgV": "science,spirituality", "pedMeJraE": 0, "H7zV6yV": 4}} </ |
| 3016 | 1 | 645 | 72 | 1.000 | 1.400 | 2.400 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "lTV6R0o", "parameters": {"m8vNAVB6cP": 300, "1CZ5pIgV": "science,spirituality"}} </tool_call> |
| 419 | 0 | 479 | 52 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "ideas_get_view_count", "parameters": {"uuid": "abcde", "lang": "it"}} </tool_call> |
| 419 | 1 | 479 | 52 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "ideas_get_view_count", "parameters": {"uuid": "abcde", "lang": "it"}} </tool_call> |
| 2771 | 0 | 894 | 77 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "main_endpoint", "parameters": {"asin": "B08Y6Z7890", "domain": "ca"}} {"name": "search_for_a_product", "parameter |
| 2771 | 1 | 894 | 77 | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "main_endpoint", "parameters": {"asin": "B08Y6Z7890", "domain": "ca"}} {"name": "search_for_a_product", "parameter |
| 3033 | 0 | 792 | 242 | 0.000 | 0.000 | 0.000 | False | <think> I should directly respond to the user's need. </think> <response> I am able to acquire the latest candlestick data for the specified parameters. The data for symbol TSLA on a 15-minute time frame looks as follows:  - Open price: 185 |
| 3033 | 1 | 792 | 140 | 0.000 | 0.000 | 0.000 | False | <think> I should directly respond to the user's need. </think> <response> The latest candlestick data for the symbol TSLA on a 15-minute frame is as follows:  - Open Price: $185.23 - High Price: $187.45 - Low Price: $184.78 - Close Price: $ |
| 3654 | 0 | 709 | 56 | 1.000 | -1.000 | 0.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Rugby Manager Last Matches", "parameters": {"id": "2021", "page": "1"}} </tool_call> |
| 3654 | 1 | 709 | 60 | 1.000 | -2.200 | -1.200 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Rugby Manager Last Matches", "parameters": {"id": 2021.0, "page": 1.0}} </tool_call> |
| 2233 | 0 | 530 | 50 | 0.000 | 0.000 | 0.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "agile.sprint_planning", "parameters": {"sprint_duration": "4 weeks"}} </tool_call> |
| 2233 | 1 | 530 | 77 | 0.000 | 0.000 | 0.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "agile.sprint_planning", "parameters": {"sprint_duration": "2 weeks", "tasks": "[مثل: 实现基于REST协议的序列化与反序列化功能，实现登录功能 |
| 356 | 0 | 765 | 62 | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> {"name": "find_majority_element", "parameters": {"nums": "[2, 2, 1, 1, 1, 2, 2]"}} <tool_call> |
| 356 | 1 | 765 | 24 | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>  |
| 2418 | 0 | 981 | 110 | 1.000 | -0.250 | 0.750 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "bP9F_z_", "parameters": {"VVNTFVcb": "Sunnyvale", "3AfT": "[\"bread\"], [\"cheese\"], [\"milk\"]"}} {"name": "B93 |
| 2418 | 1 | 981 | 79 | 1.000 | -0.125 | 0.875 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "bP9F_z_", "parameters": {"VVNTFVcb": "Sunnyvale, CA"}} {"name": "B93bq", "parameters": {"0hxq3d0": "Charizard"}}  |

The JSON report contains the complete response text and parsed tool-call result for every trajectory.
