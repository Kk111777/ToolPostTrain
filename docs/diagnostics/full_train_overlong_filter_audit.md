# Full train overlong-filter audit

- source parquet: /root/autodl-tmp/ProjectB/repo/verl-GDPO/dataset/rlla_4k/train.parquet
- tokenizer/model: /root/autodl-tmp/ProjectB/checkpoints/rl_init_sft_v2_merged
- chat template: tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=True)
- max_prompt_length: 2048
- deterministic audit seed: 42

## Counts

- raw train samples: 3920
- filtered as overlong: 19
- eligible samples after filter: 3901
- eligible_count // 512: 7
- drop_last=True complete batches per epoch: 7
- total optimizer steps at 15 epochs: 105
- exact 105-step condition: PASS

## Filtered rows

| row position | source id | prompt tokens |
|---:|---|---:|
| 46 | 46 | 3606 |
| 172 | 172 | 2902 |
| 266 | 266 | 2077 |
| 354 | 354 | 2159 |
| 512 | 512 | 2573 |
| 558 | 558 | 2099 |
| 997 | 997 | 2310 |
| 1095 | 1095 | 4293 |
| 1176 | 1176 | 2127 |
| 1456 | 1456 | 2704 |
| 1459 | 1459 | 2172 |
| 1524 | 1524 | 2112 |
| 2162 | 2162 | 3861 |
| 2217 | 2217 | 2104 |
| 2264 | 2264 | 2105 |
| 3179 | 3179 | 2803 |
| 3520 | 3520 | 2230 |
| 3753 | 3753 | 2341 |
| 3840 | 3840 | 2102 |

## Filter implementation and verification

The audit uses the merged-checkpoint tokenizer and the same chat-template call as verl v0.9.1 RLHFDataset when no processor/tool schema is configured.
An eligible set was filtered before sampling. The temporary capacity batch contains exactly 512 rows, and reapplying the same filter kept all 512 rows.
Temporary batch: /root/autodl-tmp/ProjectB/env-modern/capacity_smoke_512.parquet
Manifest: /root/autodl-tmp/ProjectB/env-modern/capacity_smoke_512_manifest.json
