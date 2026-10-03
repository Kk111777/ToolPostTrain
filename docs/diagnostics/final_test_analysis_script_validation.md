# Final-test analysis script validation

Date: 2026-10-03 (Asia/Shanghai)

The CPU-only analysis script was compiled, self-tested, and run against the
persisted formal_validation_80 JSONL outputs for GRPO-original35,
GDPO-current35, and GRPO-noKL35 at step 35. No test.parquet row or test-model
output was read.

Checks:

- complete unique coverage: 80 rows for each run: PASS
- fixed source3814 sensitivity subset: 79 rows for each run: PASS
- scorer recomputation with pinned upstream rlla.py: PASS
- raw total, accuracy, format and strict-format values finite: PASS
- target-aware tool-call JSON parse denominator: 73 target rows: PASS
- response-only wrapper denominator: 7 response-only rows: PASS
- auxiliary all-row parser/wrapper counts retained separately: PASS
- paired bootstrap: 10000 resamples, NumPy default_rng seed 42: PASS

Formal fixture primary step35 values:

| run | raw total mean | accuracy mean | format mean | strict | tool JSON parse | response-only wrapper |
|---|---:|---:|---:|---:|---:|---:|
| GRPO-original35 | 2.799720982142857 | 1.8497209821428569 | 0.95 | 76/80 | 71/73 | 4/7 |
| GDPO-current35 | 2.8205223214285713 | 1.8580223214285716 | 0.9625 | 77/80 | 72/73 | 4/7 |
| GRPO-noKL35 | 2.8187366071428572 | 1.856236607142857 | 0.9625 | 77/80 | 72/73 | 4/7 |

Existing canonical files are preserved. Their tool_parse and
response_wrapper fields correspond to auxiliary all-row counts; the new
analysis script additionally emits the protocol-required target-aware
denominators and labels both semantics explicitly.
