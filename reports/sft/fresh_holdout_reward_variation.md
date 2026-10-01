# ProjectB fresh holdout reward-variation pilot

Inference-only pilot using SFT v2. No optimizer, adapter merge, or GRPO/GDPO training was run.

## Holdout construction

- Source: `/root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet`
- Fresh prompts: `32`, selection seed `20261001`
- Rollout seed: `42`
- Selected source rows: `[1310, 1753, 1964, 2892, 3164, 3477, 2134, 681, 3282, 823, 1031, 426, 3731, 753, 200, 1024, 2536, 2062, 1184, 1856, 1054, 3087, 1471, 1398, 2128, 1118, 1711, 134, 1280, 3000, 1367, 1441]`
- SFT v1 source rows excluded: `800`
- SFT v2 source rows excluded: `800`
- Fixed validation rows excluded: `16`
- Constrained diagnostic rows excluded: `16`
- Overlap between selected rows and all exclusions: `0`
- Rollout: vLLM, `tensor_parallel_size=1`, `n=4`
- Sampling: temperature `1.0`, top_p `1.0`, top_k `-1`
- Model: `/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306` + `/root/autodl-tmp/ProjectB/checkpoints/format_sft_v2_lora`
- Reward manager: official `GDPORewardManager` + `rlla.compute_score`

## Overall reward signal

- Overall strict format success: `104/128` (81.25%)
- Overall format reward non-zero: `104/128` (81.25%)
- Tool-call JSON parse: `92/120` (76.67%)
- Response-only wrapper: `5/8` (62.50%)
- Accuracy reward non-zero: `117/128` (91.41%)
- Reward mean/std: `1.165857` / `2.542614`
- All rewards finite: `True`
- Output types: `{'tool_call': 102, 'response': 17, 'invalid': 9}`

## Group-level reward variation

- Groups: `32`; each has `4` rollouts
- Groups with format_reward variation: `17/32`
- Groups with accuracy_reward variation: `27/32`
- Groups with both dimensions varying: `16/32`
- Groups with total_reward variation: `28/32`
- Groups with all total rewards equal: `4/32`

| group | expected kind | format variance | accuracy variance | total variance | format-valid | tool parse | outputs |
|---:|---|---:|---:|---:|---:|---:|---|
| 1310 | tool_only | 0.250000 | 9.000000 | 12.250000 | 2/4 | 2/4 | `{'tool_call': 2, 'response': 2}` |
| 1753 | tool_only | 0.187500 | 6.000000 | 8.187500 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 1964 | tool_only | 0.187500 | 6.750000 | 9.187500 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 2892 | tool_only | 0.250000 | 9.000000 | 12.250000 | 2/4 | 2/4 | `{'tool_call': 2, 'response': 1, 'invalid': 1}` |
| 3164 | tool_only | 0.000000 | 1.273388 | 1.273388 | 4/4 | 2/4 | `{'tool_call': 4}` |
| 3477 | tool_only | 0.187500 | 4.320000 | 6.307500 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 2134 | tool_only | 0.187500 | 4.750000 | 6.687500 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 681 | tool_only | 0.000000 | 4.673469 | 4.673469 | 4/4 | 3/4 | `{'tool_call': 4}` |
| 3282 | tool_only | 0.000000 | 0.000000 | 0.000000 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 823 | tool_only | 0.000000 | 0.000000 | 0.000000 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 1031 | tool_only | 0.000000 | 0.421875 | 0.421875 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 426 | tool_only | 0.187500 | 6.015306 | 8.238520 | 3/4 | 3/4 | `{'tool_call': 3, 'invalid': 1}` |
| 3731 | tool_only | 0.000000 | 0.977431 | 0.977431 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 753 | tool_only | 0.000000 | 0.847969 | 0.847969 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 200 | tool_only | 0.000000 | 0.388800 | 0.388800 | 4/4 | 3/4 | `{'tool_call': 4}` |
| 1024 | tool_only | 0.187500 | 1.687500 | 2.650000 | 3/4 | 3/4 | `{'tool_call': 3, 'invalid': 1}` |
| 2536 | tool_only | 0.250000 | 2.687500 | 4.187500 | 2/4 | 2/4 | `{'tool_call': 2, 'invalid': 2}` |
| 2062 | tool_only | 0.187500 | 4.921875 | 6.421875 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 1184 | tool_only | 0.000000 | 0.083333 | 0.083333 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 1856 | response_only | 0.187500 | 0.000000 | 0.187500 | 1/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 1054 | tool_only | 0.187500 | 6.030000 | 8.167500 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |
| 3087 | tool_only | 0.250000 | 9.000000 | 12.250000 | 2/4 | 2/4 | `{'tool_call': 2, 'invalid': 2}` |
| 1471 | tool_only | 0.187500 | 0.607500 | 1.020000 | 3/4 | 1/4 | `{'tool_call': 3, 'response': 1}` |
| 1398 | tool_only | 0.000000 | 6.750000 | 6.750000 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 2128 | tool_only | 0.000000 | 1.239796 | 1.239796 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 1118 | tool_only | 0.187500 | 9.000000 | 10.687500 | 3/4 | 3/4 | `{'tool_call': 3, 'invalid': 1}` |
| 1711 | tool_only | 0.000000 | 0.046875 | 0.046875 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 134 | tool_only | 0.250000 | 1.764404 | 3.342529 | 2/4 | 2/4 | `{'response': 1, 'tool_call': 2, 'invalid': 1}` |
| 1280 | tool_only | 0.000000 | 0.000000 | 0.000000 | 4/4 | 4/4 | `{'tool_call': 4}` |
| 3000 | tool_only | 0.000000 | 2.854167 | 2.854167 | 4/4 | 3/4 | `{'tool_call': 4}` |
| 1367 | response_only | 0.000000 | 0.000000 | 0.000000 | 4/4 | 0/4 | `{'response': 4}` |
| 1441 | tool_only | 0.187500 | 3.796875 | 5.671875 | 3/4 | 3/4 | `{'tool_call': 3, 'response': 1}` |

## Per-rollout records

The JSON report contains the complete response text and parsed tool-call result for all 128 trajectories.

| group | rollout | expected | output | format valid | format reward | accuracy reward | total reward | tool JSON | response preview |
|---:|---:|---|---|:---:|---:|---:|---:|:---:|---|
| 1310 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getplatformlist", "parameters": {"page": 2}} {"name": "totd_tracks_by_date", "parameters": {" |
| 1310 | 1 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> AI's final response </response> |
| 1310 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getplatformlist", "parameters": {"page": 2}} {"name": "totd_tracks_by_date", "parameters": {" |
| 1310 | 3 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> I should directly respond to the user's need. </response> |
| 1753 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getFashionBlogDetails", "parameters": {"blog_id": "152"}} </tool_call> |
| 1753 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getFashionBlogDetails", "parameters": {"blog_id": "152"}} </tool_call> |
| 1753 | 2 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> To provide the desired information for the fashion blog based on Blog ID, the appropriate function would be the 'getFashionBlogDetails' with a 'bl |
| 1753 | 3 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getFashionBlogDetails", "parameters": {"blog_id": 152}} </tool_call> |
| 1964 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Bl5GQLe5L", "parameters": {"Qt_O": "vegan_chili_111"}} {"name": "Bl5GQLe5L", "parameters": {" |
| 1964 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Bl5GQLe5L", "parameters": {"Qt_O": "vegan_chili_111"}} {"name": "Bl5GQLe5L", "parameters": {" |
| 1964 | 2 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> If you look carefully, you can see that there is a discrepancy between the steps provided in the context and the actual end result. Can you provid |
| 1964 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Bl5GQLe5L", "parameters": {"Qt_O": "vegan_chili_111"}} {"name": "Bl5GQLe5L", "parameters": {" |
| 2892 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "vZjZMqA", "parameters": {"1S.o": "happy"}} {"name": "vZjZMqA", "parameters": {"1S.o": "sad"}} |
| 2892 | 1 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> I should directly respond to the user's need. </response> |
| 2892 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "vZjZMqA", "parameters": {"1S.o": "happy"}} {"name": "vZjZMqA", "parameters": {"1S.o": "sad"}} |
| 2892 | 3 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>  |
| 3164 | 0 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "zRqDSkvRo", "parameters": {"JvlYQh1.": [10, 10, 10, 11, 11, 11, 12, 12, 12]}} {"name": "uEHKF |
| 3164 | 1 | tool_only | tool_call | True | 1.000 | -0.673 | 0.327 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "zRqDSkvRo", "parameters": {"JvlYQh1.": "[10, 10, 10, 11, 11, 11, 12, 12, 12]"}} {"name": "uEH |
| 3164 | 2 | tool_only | tool_call | True | 1.000 | -0.818 | 0.182 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "zRqDSkvRo", "parameters": {"JvlYQh1.": [10, 10, 11, 11, 11, 12, 12, 12]}} {"name": "iDDGltAQE |
| 3164 | 3 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "zRqDSkvRo", "parameters": {"JvlYQh1.": "[10, 10, 10, 11, 11, 11, 12, 12, 12]"}} {"name": "uEH |
| 3477 | 0 | tool_only | tool_call | True | 1.000 | 1.800 | 2.800 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "gb9diPNfrm", "parameters": {"aUWa1": "osm-bright"}} {"name": "c0J36V", "parameters": {"JoFcpK |
| 3477 | 1 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> AI's final response </response> |
| 3477 | 2 | tool_only | tool_call | True | 1.000 | 1.800 | 2.800 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "gb9diPNfrm", "parameters": {"aUWa1": "osm-bright"}} {"name": "c0J36V", "parameters": {"JoFcpK |
| 3477 | 3 | tool_only | tool_call | True | 1.000 | 1.800 | 2.800 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "gb9diPNfrm", "parameters": {"aUWa1": "osm-bright"}} {"name": "c0J36V", "parameters": {"JoFcpK |
| 2134 | 0 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBlock", "parameters": {"block_number": 123456}} </tool_call> |
| 2134 | 1 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBlock", "parameters": {"block_number": 123456}} </tool_call> |
| 2134 | 2 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> The block with number 123456 does not exist in the official blockchain network, as there are only up to 520 blocks prior to its existence in the n |
| 2134 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBlock", "parameters": {"block_number": "123456"}} </tool_call> |
| 681 | 0 | tool_only | tool_call | True | 1.000 | -1.000 | 0.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "measure_distance", "parameters": {"lat1": "60", "lon1": "21", "lat2": "40.7128", "lon2": "-74 |
| 681 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "tz_lookup_by_location", "parameters": {"lat": 40.7128, "lng": -74.006, "timestamp": 160945920 |
| 681 | 2 | tool_only | tool_call | True | 1.000 | -0.143 | 0.857 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "measure_distance", "parameters": {"lon1": "36.6880794", "lon2": "39.2694784", "lat2": "-6.778 |
| 681 | 3 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "tz_lookup_by_location", "parameters": {"lat": 40.7128, "lng": -74.0060, "timestamp": 16094592 |
| 3282 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_restaurant_reviews", "parameters": {"restaurant_name": "Taste of Italy", "location": "dow |
| 3282 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_restaurant_reviews", "parameters": {"restaurant_name": "Taste of Italy", "location": "dow |
| 3282 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_restaurant_reviews", "parameters": {"restaurant_name": "Taste of Italy", "location": "dow |
| 3282 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_restaurant_reviews", "parameters": {"restaurant_name": "Taste of Italy", "location": "dow |
| 823 | 0 | tool_only | tool_call | True | 1.000 | 2.000 | 3.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getCourseContent", "parameters": {"course": "Data Science 101"}} {"name": "getCourseProgress" |
| 823 | 1 | tool_only | tool_call | True | 1.000 | 2.000 | 3.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getCourseContent", "parameters": {"course": "Data Science 101"}} {"name": "getCourseProgress" |
| 823 | 2 | tool_only | tool_call | True | 1.000 | 2.000 | 3.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getCourseContent", "parameters": {"course": "Data Science 101"}} {"name": "getCourseProgress" |
| 823 | 3 | tool_only | tool_call | True | 1.000 | 2.000 | 3.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getCourseContent", "parameters": {"course": "Data Science 101"}} {"name": "getCourseProgress" |
| 1031 | 0 | tool_only | tool_call | True | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBorderWaitTime", "parameters": {"border_crossing": "Peace Arch crossing", "direction": "en |
| 1031 | 1 | tool_only | tool_call | True | 1.000 | 0.000 | 1.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBorderWaitTime", "parameters": {"border_crossing": "Peace Arch crossing", "direction": "En |
| 1031 | 2 | tool_only | tool_call | True | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBorderWaitTime", "parameters": {"border_crossing": "Peace Arch", "direction": "enter"}} </ |
| 1031 | 3 | tool_only | tool_call | True | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "getBorderWaitTime", "parameters": {"border_crossing": "Peace Arch crossing", "direction": "en |
| 426 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "produce_mixy_words_list", "parameters": {"amount": 20, "difficulty": "easy"}} {"name": "produ |
| 426 | 1 | tool_only | tool_call | True | 1.000 | 1.286 | 2.286 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "produce_mixy_words_list", "parameters": {"amount": "20", "difficulty": "easy"}} {"name": "pro |
| 426 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "produce_mixy_words_list", "parameters": {"amount": 20, "difficulty": "easy"}} {"name": "produ |
| 426 | 3 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> {   "Thought": "'**Remarkably, we don't need to build an API action with multiple calls to achieve this task. I will directly respond based on the |
| 3731 | 0 | tool_only | tool_call | True | 1.000 | 1.667 | 2.667 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_movies_by_director", "parameters": {"movie_director": "David Fincher"}} {"name": "get_mov |
| 3731 | 1 | tool_only | tool_call | True | 1.000 | 0.000 | 1.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_movies_by_director", "parameters": {}} {"name": "get_movies_by_cast_name", "parameters":  |
| 3731 | 2 | tool_only | tool_call | True | 1.000 | 0.167 | 1.167 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_movies_by_director", "parameters": {"movie_director": "David Fincher"}} {"name": "get_mov |
| 3731 | 3 | tool_only | tool_call | True | 1.000 | 2.333 | 3.333 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_movies_by_director", "parameters": {"movie_director": "David Fincher"}} {"name": "get_mov |
| 753 | 0 | tool_only | tool_call | True | 1.000 | 2.500 | 3.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "local_weather_api", "parameters": {"q": "Berlin, Germany", "lang": "de", "num_of_days": 4, "a |
| 753 | 1 | tool_only | tool_call | True | 1.000 | 2.000 | 3.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "local_weather_api", "parameters": {"q": "Berlin", "lang": "de", "num_of_days": 4, "aqi": "yes |
| 753 | 2 | tool_only | tool_call | True | 1.000 | 1.100 | 2.100 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "local_weather_api", "parameters": {"q": "Berlin, Germany", "lang": "de", "num_of_days": 4, "f |
| 753 | 3 | tool_only | tool_call | True | 1.000 | 0.083 | 1.083 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "local_weather_api", "parameters": {"q": "Berlin", "lang": "de"}} {"name": "local_weather_api" |
| 200 | 0 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "vueStateSync", "parameters": {"componentState": "{\"items\": [{"id": "1", "name": "Apple", "q |
| 200 | 1 | tool_only | tool_call | True | 1.000 | -1.560 | -0.560 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "vueStateSync", "parameters": {"componentState": {"items": [{"id": "1", "name": "Apple", "quan |
| 200 | 2 | tool_only | tool_call | True | 1.000 | -1.560 | -0.560 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "vueStateSync", "parameters": {"componentState": {"items": [{"id": "1", "name": "Apple", "quan |
| 200 | 3 | tool_only | tool_call | True | 1.000 | -1.560 | -0.560 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "vueStateSync", "parameters": {"componentState": "{\"items\": [\"{\"id\": \"1\", \"name\": \"A |
| 1024 | 0 | tool_only | tool_call | True | 1.000 | 0.600 | 1.600 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_6_timeseries_endpoint", "parameters": {"start_date": "2017-01-01", "end_date": "2017-12-3 |
| 1024 | 1 | tool_only | tool_call | True | 1.000 | -1.800 | -0.800 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_4_date_endpoint", "parameters": {"base": "AUD", "symbols": "USD,NZD", "date": "2017-01-01 |
| 1024 | 2 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> The user wants to get the historical exchange rates between specific dates and provide information about those rates through the appropriate funct |
| 1024 | 3 | tool_only | tool_call | True | 1.000 | -1.600 | -0.600 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_4_date_endpoint", "parameters": {"base": "AUD", "symbols": "USD,NZD", "date": "2017-01-01 |
| 2536 | 0 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "InfluencerContentCreation.scheduleSponsoredPost", "parameters": {"influencer": "John Doe", "c |
| 2536 | 1 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> Here's a function call you could use for scheduling a sponsored post. Notice the `post`, `time`, and `badgeUrl` parameters. The badgeUrl is option |
| 2536 | 2 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "InfluencerContentCreation.scheduleSponsoredPost", "parameters": {"influencer": "John Doe", "c |
| 2536 | 3 | tool_only | tool_call | True | 1.000 | -2.000 | -1.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "InfluencerContentCreation.scheduleSponsoredPost", "parameters": {"influencer": "{\"name\": \" |
| 2062 | 0 | tool_only | tool_call | True | 1.000 | -1.500 | -0.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "retrieve_data", "parameters": {"rank": "top"}} {"name": "retrieve_data", "parameters": {"rank |
| 2062 | 1 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> I should choose a suitable function based on the user's need and return the appropriate response. </response> |
| 2062 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "retrieve_data", "parameters": {}} </tool_call> |
| 2062 | 3 | tool_only | tool_call | True | 1.000 | 0.000 | 1.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "retrieve_data", "parameters": {"rank": "top"}} </tool_call> |
| 1184 | 0 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Featured Channels", "parameters": {"id": "12345", "token": ""}} {"name": "Get Featured Ch |
| 1184 | 1 | tool_only | tool_call | True | 1.000 | 1.667 | 2.667 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Featured Channels", "parameters": {"id": "12345", "token": ""}} {"name": "Get Featured Ch |
| 1184 | 2 | tool_only | tool_call | True | 1.000 | 1.667 | 2.667 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Featured Channels", "parameters": {"id": "12345", "token": "some_token"}} {"name": "Get F |
| 1184 | 3 | tool_only | tool_call | True | 1.000 | 1.667 | 2.667 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Featured Channels", "parameters": {"id": "12345", "token": "f1aefdf1df12df45edc00c900909c |
| 1856 | 0 | response_only | tool_call | False | 0.000 | 0.000 | 0.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "StockDataFetcher.getHistoricalPrices", "parameters": {"symbol": "AAPL"}} </tool_call> |
| 1856 | 1 | response_only | tool_call | False | 0.000 | 0.000 | 0.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "StockDataFetcher.getHistoricalPrices", "parameters": {"symbol": "AAPL", "dateRange": {"Sdate" |
| 1856 | 2 | response_only | tool_call | False | 0.000 | 0.000 | 0.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "StockDataFetcher.getHistoricalPrices", "parameters": {"symbol": "AAPL"}} </tool_call> |
| 1856 | 3 | response_only | response | True | 1.000 | 0.000 | 1.000 | False | <think> I should directly respond to the user's need. </think> <response> Available.API 호출을 사용해 주세요. </response> |
| 1054 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "E.g_4gbheoONp.h", "parameters": {"TiAP": "ijkl9876"}} {"name": "x2Z1o6I21M8ik50", "parameters |
| 1054 | 1 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> You should either use the appropriate 'Name: E.g_4gbheoONp.h' tool with proper parameters or an appropriate 'x2Z1o6I21M8ik50' or 'w0VX6vjYVLAXW' t |
| 1054 | 2 | tool_only | tool_call | True | 1.000 | 0.600 | 1.600 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "E.g_4gbheoONp.h", "parameters": {"TiAP": "ijkl9876"}} {"name": "x2Z1o6I21M8ik50", "parameters |
| 1054 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "E.g_4gbheoONp.h", "parameters": {"TiAP": "ijkl9876"}} {"name": "x2Z1o6I21M8ik50", "parameters |
| 3087 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "WNBA Teams API", "parameters": {}} </tool_call> |
| 3087 | 1 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> The WNBA Teams API responds with the names, abbreviations, and locations of all WNBA teams by querying the provided SimpleCNL Aurora API. paramete |
| 3087 | 2 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> </tool_call> {"name": "WNBA Teams API", "parameters": {}} </tool_call> |
| 3087 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "WNBA Teams API", "parameters": {}} </tool_call> |
| 1471 | 0 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "podcast.analytics", "parameters": {"platform": "Podbean", "date_range": {"start_date": "2022- |
| 1471 | 1 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "podcast.analytics", "parameters": {"platform": "Podbean", "date_range": {"begin": "2023-01-01 |
| 1471 | 2 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> You should use the correct tool with proper parameters to respond to the user's need. </response> |
| 1471 | 3 | tool_only | tool_call | True | 1.000 | -1.200 | -0.200 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "podcast.analytics", "parameters": {"platform": "podbean", "date_range": {"start_date": "2023- |
| 1398 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "GetIpInfoLite", "parameters": {"ip": "192.168.1.100"}} </tool_call> |
| 1398 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "GetIpInfoLite", "parameters": {"ip": "192.168.1.100"}} </tool_call> |
| 1398 | 2 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Find By UDPRN", "parameters": {"udprn": "192.168.1.100"}} </tool_call> |
| 1398 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "GetIpInfoLite", "parameters": {"ip": "192.168.1.100"}} </tool_call> |
| 2128 | 0 | tool_only | tool_call | True | 1.000 | 0.429 | 1.429 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_by_trakt_tv_id", "parameters": {"t": "112641"}} {"name": "get_by_trakt_tv_id", "parameter |
| 2128 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_by_trakt_tv_id", "parameters": {"t": "112641", "m": "movie"}} {"name": "get_by_trakt_tv_i |
| 2128 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_by_trakt_tv_id", "parameters": {"t": "112641", "m": "movie"}} {"name": "get_by_trakt_tv_i |
| 2128 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "get_by_trakt_tv_id", "parameters": {"t": "112641", "m": "movie"}} {"name": "get_by_trakt_tv_i |
| 1118 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "NHlKijmSMxYu", "parameters": {"VbglkMxM5": "00:24:D7:6E:03:E8"}} {"name": "nvhNlwn", "paramet |
| 1118 | 1 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>  |
| 1118 | 2 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "nh1ND9ixRUo2QZb34sly2", "parameters": {"VbglkMxM5": "00:24:D7:6E:03:E8"}} {"name": "nh1NNa",  |
| 1118 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "NHlKijmSMxYu", "parameters": {"VbglkMxM5": "00:24:D7:6E:03:E8"}} {"name": "nvhNlwn", "paramet |
| 1711 | 0 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "music.theory.chord_progression_generator", "parameters": {"key": "G", "progression": "I-vi-IV |
| 1711 | 1 | tool_only | tool_call | True | 1.000 | 0.500 | 1.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "music.theory.chord_progression_generator", "parameters": {"key": "G", "progression": "ivviio" |
| 1711 | 2 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "music.theory.chord_progression_generator", "parameters": {"key": "G", "progression": "I-vi-IV |
| 1711 | 3 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "music.theory.scale_finder", "parameters": {"notes": ["G", "B", "D", "F#"], "include_modes": f |
| 134 | 0 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> If you want us to respond directly to the user's need, you need to clear your problem or ask a specific question about a hypothetical situation ou |
| 134 | 1 | tool_only | tool_call | True | 1.000 | -0.375 | 0.625 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Amazon Search Results", "parameters": {"searchQuery": "classic vinyl records"}} {"name": "Tar |
| 134 | 2 | tool_only | tool_call | True | 1.000 | -0.312 | 0.688 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "Get Amazon Search Results", "parameters": {"searchQuery": "classic vinyl records", "api_key": |
| 134 | 3 | tool_only | invalid | False | 0.000 | -3.000 | -3.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> </tool_call> {"name": "Get Amazon Search Results", "parameters": {"searchQuery": "classic vinyl records", "api_key" |
| 1280 | 0 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "drug_info", "parameters": {"drug": "Ibuprofen"}} </tool_call> |
| 1280 | 1 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "drug_info", "parameters": {"drug": "Ibuprofen"}} </tool_call> |
| 1280 | 2 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "drug_info", "parameters": {"drug": "Ibuprofen"}} </tool_call> |
| 1280 | 3 | tool_only | tool_call | True | 1.000 | 3.000 | 4.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "drug_info", "parameters": {"drug": "Ibuprofen"}} </tool_call> |
| 3000 | 0 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TradeAnalysis.getMarketTrends", "parameters": {"tradeData": "[{'date': '2023-01-01', 'volume' |
| 3000 | 1 | tool_only | tool_call | True | 1.000 | 1.000 | 2.000 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TradeAnalysis.getMarketTrends", "parameters": {"tradeData": "[{'date': '2023-01-01', 'volume' |
| 3000 | 2 | tool_only | tool_call | True | 1.000 | -1.333 | -0.333 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TradeAnalysis.getMarketTrends", "parameters": {"tradeData": [[75500, "$21.50", "$24.80"], [11 |
| 3000 | 3 | tool_only | tool_call | True | 1.000 | -3.000 | -2.000 | False | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TradeAnalysis.getMarketTrends", "parameters": {"tradeData": [[75500 {"priceRangeMin": 21.50,  |
| 1367 | 0 | response_only | response | True | 1.000 | 0.000 | 1.000 | False | <think> I should provide a direct response to the user's need. </think> <response> The net income of Apple Inc. is $57. 41 billion. </response> |
| 1367 | 1 | response_only | response | True | 1.000 | 0.000 | 1.000 | False | <think> I should directly respond to the user's need. </think> <response> The net income of Apple Inc. (symbol: 'AAPL') is 57,411,000,000 US dollars for the region US. </response> |
| 1367 | 2 | response_only | response | True | 1.000 | 0.000 | 1.000 | False | <think> I should directly respond to the user's need. </think> <response> The net income of Apple Inc. is 5,741,100,000 USD. </response> |
| 1367 | 3 | response_only | response | True | 1.000 | 0.000 | 1.000 | False | <think> I should directly respond to the user's need. </think> <response> The net income of Apple Inc is $57,411,000,000 in the US region. </response> |
| 1441 | 0 | tool_only | tool_call | True | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TKdj9CYTm.3sI", "parameters": {"dPWJl": "products", "LDAbFP": "[{'price': '>', 'cmp': '>', 'c |
| 1441 | 1 | tool_only | tool_call | True | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TKdj9CYTm.3sI", "parameters": {"dPWJl": "products", "LDAbFP": [{"price": {"$gt": 100}, "exist |
| 1441 | 2 | tool_only | tool_call | True | 1.000 | 1.500 | 2.500 | True | <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think> <tool_call> {"name": "TKdj9CYTm.3sI", "parameters": {"dPWJl": "products", "LDAbFP": [{"query": "price $100", "table |
| 1441 | 3 | tool_only | response | False | 0.000 | -3.000 | -3.000 | False | <think> I should directly respond to the user's need. </think> <response> Please provide the product's name (if any), the product's category, and the brand of the product. </response> |

## Scope boundary

This pilot checks fresh-holdout generalization and whether rollout groups contain usable outcome variation. It does not compute advantages, run backward, or authorize formal training.
