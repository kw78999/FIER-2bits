# PG19 2-bit group-size sweep

## Configuration

- Model: `meta-llama/Llama-3.1-8B`
- Context: 32,768 tokens
- Sparse budget: 4,096 tokens
- Group sizes: 32, 64, 128, 256
- Blocks: 4 shared PG19 blocks
- Evaluated next tokens: 128 per block, 512 per group size
- First two transformer layers: full attention
- Backend: CUDA POPC
- Dtype: FP16
- Exact-QK top-k recall diagnostics: enabled

## Results

| Group | PPL | vs. g32 | Search ms/token | Search vs. g32 | Selected attention ms/token | Cache update ms/token | Decode ms/token | Top-k recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 12.5289 | baseline | 31.4 | 1.00x | 155.5 | 14.6 | 1130.5 | 0.3732 |
| 64 | 12.3792 | -1.19% | 53.2 | 1.69x | 155.0 | 14.6 | 1151.7 | 0.3765 |
| 128 | 12.2491 | -2.23% | 83.0 | 2.64x | 155.1 | 14.6 | 1181.5 | 0.3783 |
| 256 | 11.9444 | -4.67% | 143.8 | 4.57x | 155.2 | 14.6 | 1242.3 | 0.3797 |

Larger groups improved aggregate PPL and exact-QK top-k recall in this pilot,
but increased candidate-search latency substantially. Group 64 provides the
smallest end-to-end slowdown (+1.87% versus group 32) while retaining a modest
PPL improvement. Group 256 gives the best PPL but is primarily a
quality-oriented setting.

The reported decode latency includes exact-QK top-k recall diagnostics and is
therefore higher than a production decode run with diagnostics disabled.
