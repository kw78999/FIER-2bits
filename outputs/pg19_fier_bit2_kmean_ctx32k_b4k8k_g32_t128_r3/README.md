# PG19: FIER and 2-bit K-mean comparison at 32K

This run compares FIER Triton with the existing 2-bit CUDA scorer and the two
K-mean scoring prototypes. It used one NVIDIA RTX A6000, Llama-3.1-8B in
FP16, SDPA, a 32,768-token context, group size 32, and budgets 4,096 and 8,192.
The PPL estimate contains three PG19 blocks and 128 paired decode tokens per
block (384 tokens per variant). The first two transformer layers use full
attention. Exact top-k recall was disabled to keep this long run bounded.

## Aggregate results

| Method | Budget | PPL | Search ms/token | Decode ms/token |
| --- | ---: | ---: | ---: | ---: |
| Existing 2-bit CUDA POPC | 4K | 11.7082 | 31.48 | 622.52 |
| FIER Triton | 4K | 10.2385 | 62.17 | 647.37 |
| 2-bit signed 4-mean | 4K | 10.2098 | 529.55 | 1115.55 |
| 2-bit absolute 2-mean | 4K | 10.2063 | 296.51 | 883.23 |
| Existing 2-bit CUDA POPC | 8K | 10.5983 | 31.62 | 657.90 |
| FIER Triton | 8K | 10.1978 | 62.10 | 681.67 |
| 2-bit signed 4-mean | 8K | 10.1982 | 529.53 | 1151.86 |
| 2-bit absolute 2-mean | 8K | 10.1986 | 296.36 | 916.91 |

At 4K, the 2-mean prototype has the lowest PPL, 0.32% below FIER, while the
existing POPC scorer is 14.35% above FIER. At 8K, FIER and both mean variants
are effectively tied (within 0.008% PPL). Increasing the budget from 4K to 8K
improves PPL by 9.48% for existing POPC, but only 0.40% for FIER, 0.11% for
4-mean, and 0.07% for 2-mean.

The 2-mean prototype is the better mean-based candidate: its quality matches
4-mean, while its search time is about 44% lower. It is not yet a speed
optimization. Mean tensors are reconstructed from full K at every decode step
instead of being stored and incrementally updated in the cache. Consequently,
its measured search time is 4.77x FIER and total decode is 34–36% slower. These
latencies are a prototype upper bound; a fused cached-mean CUDA/Triton kernel
is required for a fair optimized-speed comparison.

See `metadata.json` for the exact configuration, `summary.csv` for aggregate
values, and `samples.jsonl` for per-block measurements.
