# GDPO Stage 1 dataset dry-run

Generated: 2026-10-01T23:13:27.169607+00:00

This was a CPU-only dataset/dataloader check while the GDPO trainer was running. No model weights were loaded and CUDA visibility was empty.

- Raw train rows: 3920
- Official overlong-filter eligible rows: 3901
- train_max_samples: -1
- train_batch_size: 512
- shuffle: false
- drop_last: true
- Full batches per epoch: 7
- Epochs: 5
- Expected optimizer steps: 35
- Max prompt length: 2048
- Intermediate validation: /root/autodl-tmp/ProjectB/env-modern/formal_validation_80.parquet
- test.parquet used for intermediate validation: false

## Semantic conclusion

In official verl v0.9.1, train_max_samples=-1 means no sample truncation. The full 3920-row parquet is loaded, then the official chat-template/token-length filter leaves 3901 eligible rows. With shuffle=false, drop_last=true, and batch size 512, the dataloader has 3901 // 512 = 7 full batches per epoch, so 5 epochs produce 35 optimizer steps. The one-step smoke value 512 was a deliberate capacity-smoke truncation and is not the formal Stage 1 dataset setting.
