# PG19 five-way fused 2-mean experiment

One RTX A6000, `meta-llama/Llama-3.1-8B`, FP16/SDPA, 32,768-token
context, group size 32, three PG19 blocks, and 128 paired decode tokens per
block (384 tokens per variant). Layers 0 and 1 use full attention. Exact top-k
recall was disabled.

| Method | Budget | PPL | Search ms/token | Attention ms/token | Decode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Attention | — | 10.2090 | 5.04 | 260.75 | **365.80** |
| Quest | 4K | 11.0825 | 581.26 | 160.12 | 902.55 |
| Quest | 8K | 10.4719 | 603.87 | 160.72 | 923.60 |
| FIER Triton | 4K | 10.2385 | 61.78 | 157.42 | 648.92 |
| FIER Triton | 8K | **10.1978** | 61.80 | 158.98 | 681.96 |
| Existing 2-bit CUDA | 4K | 11.7082 | 31.30 | 157.10 | 620.79 |
| Existing 2-bit CUDA | 8K | 10.5983 | 31.31 | 159.39 | 654.17 |
| Fused weighted 2-mean CUDA | 4K | 20.5754 | **6.56** | 159.43 | 643.17 |
| Fused weighted 2-mean CUDA | 8K | 17.4031 | **6.55** | 160.25 | 680.25 |

The fused 2-mean scorer is 4.77–4.78x faster than existing 2-bit candidate
search and about 9.4x faster than FIER search. CUDA scoring was selected over
Triton after a 32K microbenchmark (0.119 ms versus 0.145 ms mean raw scoring;
about 0.192 ms versus 0.197 ms with top-k).

Quality does not support replacing the baseline with this exact weighted-bit
formula. Relative to existing 2-bit, 2-mean PPL is 75.7% higher at 4K and
64.2% higher at 8K. End-to-end decode is also 3.6–4.0% slower despite faster
search, so framework/cache overhead outside the measured candidate event
dominates the saved search time in this evaluator.

This result differs from the earlier `group_mean2` prototype: that prototype
kept the RoPE'd Q in full precision and reconstructed channel-wise K values.
The new fused formula quantizes Q to sign and magnitude and uses one low/high
mean per 32-channel word, which discards substantially more ranking signal.

`metadata.json` records the configuration, `summary.csv` contains aggregates,
and `samples.jsonl` contains all per-token measurements.
