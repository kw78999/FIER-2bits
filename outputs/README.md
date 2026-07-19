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

These results measure the repository's experimental Python/PyTorch evaluator,
including framework and instrumentation overhead. Consult each run's metadata
before comparing absolute latency values.
