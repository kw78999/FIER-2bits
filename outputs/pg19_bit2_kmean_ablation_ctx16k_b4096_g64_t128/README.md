# PG19 2-bit K-mean weighting ablation

## Configuration

- Model: `meta-llama/Llama-3.1-8B`
- Context: 16,384 tokens
- Sparse budget: 4,096 tokens
- Group size: 64
- Blocks: 4 shared PG19 blocks
- Evaluated next tokens: 128 per block, 512 per method
- First two transformer layers: full attention
- Dtype: FP16
- Exact-QK top-k recall diagnostics: disabled

## Methods

- `cuda_popc`: existing sign/magnitude 3-POPC score.
- `group_mean4`: signed means for positive-low, positive-high,
  negative-low, and negative-high; full-precision Q scores reconstructed K.
- `group_mean2`: pooled low/high absolute means plus the K sign;
  full-precision Q scores reconstructed K.

## Results

| Method | PPL | vs. existing | Search ms/token | Search ratio | Selected attention ms/token | Cache update ms/token | Decode ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing 2-bit | 8.1535 | baseline | 28.6 | 1.00x | 144.9 | 7.45 | 571.7 |
| Signed 4-mean | 7.9436 | -2.57% | 285.4 | 9.96x | 144.3 | 7.49 | 825.3 |
| Absolute 2-mean | 7.9412 | -2.60% | 169.4 | 5.91x | 144.1 | 7.49 | 707.9 |

Across the four paired blocks, the block-bootstrap PPL-change intervals were
-4.21% to -0.07% for 4-mean and -4.28% to -0.10% for 2-mean. These intervals
are pilot diagnostics rather than strong confidence bounds because only four
blocks were evaluated.

The 2-mean and 4-mean quality results are nearly identical. The 2-mean method
is preferable at this stage because it needs half as many mean values and its
prototype search is substantially faster.

## Timing caveat

This quality prototype recomputes category means from the full K tensor on
every decode. It does not yet persist means in an incremental KV-side cache.
Consequently, its search latency measures reconstruction plus mean calculation
and should be treated as an upper bound for a fused cached-mean CUDA kernel.
