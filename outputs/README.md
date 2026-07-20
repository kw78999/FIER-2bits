# Experiment outputs

This directory contains the retained, non-smoke results produced on one NVIDIA
RTX A6000. Each PG19 directory includes:

- `metadata.json`: model and experiment configuration;
- `summary.json` and `summary.csv`: aggregated PPL and timing metrics;
- `samples.jsonl` and `samples.csv`: per-token measurements.

The LongBench pilot contains its sample manifest, append-only raw generations,
metadata, and aggregate summary.

## Included runs

| Directory | Description |
| --- | --- |
| `longbench_sparse_pilot_a6000_50m` | 50-minute LongBench pilot on Qasper, MultiFieldQA-en, and TriviaQA with a 2K base budget. |
| `pg19_fourway_budget_sweep_ctx4k_64k` | Full Attention, Quest, FIER Triton, and 2-bit CUDA across 4K–64K contexts and 2K/4K/6K/8K budgets. |
| `pg19_longdecode_ctx32k_64k_t128` | 32K/64K PG19 run with 128 evaluated decode tokens and 4K/8K budgets. |
| `pg19_longdecode_ctx32k_64k_t256` | 32K/64K PG19 run with 256 evaluated decode tokens and 4K/8K budgets. |
| `pg19_sixway_b4096_ctx4k_32k_r3` | Reference/optimized six-way comparison at a 4K budget, with three blocks per context. |
| `pg19_bit2_threshold_ablation_b4096_ctx4k_64k_t32` | Historical max/2 versus signed-median threshold ablation. The signed-median experimental backend is not part of the current source tree. |
| `pg19_bit2_groupsize_g32_256_ctx32k_b4096_t128` | 2-bit CUDA group-size sweep (32/64/128/256) at 32K context and 4K budget, using four blocks and 128 evaluated tokens per block. |
| `pg19_bit2_kmean_ablation_ctx16k_b4096_g64_t128` | Existing 2-bit POPC versus signed 4-mean and absolute 2-mean K reconstruction at 16K context, 4K budget, and group size 64. |
| `pg19_fier_bit2_kmean_ctx32k_b4k8k_g32_t128_r3` | FIER Triton and three 2-bit scoring variants at 32K context, 4K/8K budgets, group size 32, and three blocks of 128 evaluated tokens. |
| `pg19_fiveway_2mean_fused_ctx32k_b4k8k_g32_t128_r3` | Full Attention, Quest, FIER, baseline 2-bit, and fused weighted 2-mean at 32K context and 4K/8K budgets. |

These results measure the repository's experimental Python/PyTorch evaluator,
including framework and instrumentation overhead. Consult each run's metadata
before comparing absolute latency values.
