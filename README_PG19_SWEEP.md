# PG19 FIER / 2-bit CUDA Context Sweep

This experiment compares four decode modes on PG19 long-context PPL.

FIER uses the authors-inspired Triton group-wise 1-bit pack kernel and fused dequantization/GEMV scorer; token recall uses CUDA Top-k.

- `full_attention`
- `quest_p16_b4096`
- `fier_triton_g32_b4096`
- `bit2_cuda_popc_g32_b4096`

Default settings are chosen to follow the FIER-style evaluation convention while keeping a single RTX A6000 run near a 3-hour budget, excluding first-time model download.

## Default Plan

- Model: `meta-llama/Llama-3.1-8B`
- Dataset: PG19 test split via the partial loader
- Context plan: `4096:4,8192:4,16384:2,32768:2`
- Decode tokens per block: `16`
- Sparse token budget: `4096`
- Group size: `32`
- Dense layers for sparse methods: `0,1`
- Attention implementation for dense prefix: `sdpa`
- Dtype: `float16`
- Metrics: PPL and decode/search/attention/update timing

## Authentication

Use an environment variable:

```bash
export HF_TOKEN=hf_...
```

Or create a local ignored `.env` file:

```bash
echo "HF_TOKEN=hf_..." > .env
```

Or authenticate once with Hugging Face CLI:

```bash
huggingface-cli login
```

Do not commit tokens.

## Run

```bash
python3 run_pg19_fier_bit2_sweep.py
```

Useful shorter validation run:

```bash
python3 run_pg19_fier_bit2_sweep.py \
  --context-plan 4096:1 \
  --generate-tokens 4 \
  --budget 512
```

Outputs are written to `outputs/pg19_fier_bit2_sweep_<UTC timestamp>/`:

- `metadata.json`
- `summary.json` / `summary.csv`
- `samples.jsonl` / `samples.csv`

The `outputs/` directory is ignored by Git except for `.gitkeep`.
