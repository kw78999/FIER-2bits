# FIER-2bits

Experimental long-context sparse-attention evaluation code for comparing:

- full attention;
- Quest page retrieval;
- FIER 1-bit retrieval with a PyTorch reference or Triton scorer;
- 2-bit sign/magnitude retrieval with a PyTorch reference or CUDA POPC scorer.

Selected KV entries always use the original FP16/BF16 key and value tensors for
the final attention computation. The first two transformer layers are dense by
default.

## Requirements

- Linux
- Python 3.10+
- CUDA-capable PyTorch
- NVIDIA GPU (the long-context experiments were developed on an RTX A6000)
- access to the requested gated Llama model

Install PyTorch for the CUDA version available on your system, then install the
remaining packages:

```bash
python3 -m pip install -r requirements-ppl.txt
export HF_TOKEN=hf_...
```

The FIER Triton and 2-bit CUDA extensions compile on first use. Build artifacts,
downloaded datasets, credentials, and model weights are ignored by Git.

## Quick validation

```bash
python3 -m unittest test_sparse_ppl.py
python3 run_pg19_fier_bit2_sweep.py --dry-run
```

## PG19 context sweep

```bash
python3 run_pg19_fier_bit2_sweep.py \
  --context-plan 4096:4,8192:4,16384:2,32768:2 \
  --generate-tokens 16 \
  --budget-sweep 2048,4096 \
  --group-size 32
```

This runner evaluates Full Attention, Quest, FIER Triton, and 2-bit CUDA POPC
on exactly the same PG19 token blocks. See
[README_PG19_SWEEP.md](README_PG19_SWEEP.md) for experiment semantics and
authentication options.

## Time-bounded LongBench pilot

```bash
python3 run_longbench_sparse_pilot.py \
  --time-limit-minutes 50 \
  --max-context-length 8192 \
  --budget 2048
```

The pilot evaluates Qasper, MultiFieldQA-en, and TriviaQA. It appends every
completed generation to JSONL, stops starting new work at the time limit, and
keeps the tokenized prompt identical across methods for each sample.

## Other runners

- `run_wikitext2_ppl.py`: sampled decode-style WikiText-2 PPL and retrieval
  diagnostics.
- `run_prepacked_long_context.py`: continuous-decode PG19/LongBench benchmark
  with configurable reference and optimized backends.

Detailed WikiText-2 semantics and options are in
[README_PPL.md](README_PPL.md).

## Implementation

- `fier_triton.py`: group-wise 1-bit packing and fused Triton QK scoring.
- `bit2_cuda.py`: Python wrapper for 2-bit packing, CUDA POPC scoring, and
  histogram top-k.
- `csrc/bit2_popc_ext.cu`: CUDA kernels for the 2-bit backend.
- `sparse_ppl.py`: shared quantization, retrieval, exact selected attention,
  and Llama decoding logic.

The optimized FIER layout is adapted from the
[authors' public FIER implementation](https://github.com/SimWangArizona/FIER)
at commit `e0b34153591dd7a55171f09f30abee35b0f08356`. This repository adds
partial-group handling, GQA/MQA mapping, correctness tests, and benchmark
integration.

## Outputs and scope

Experiment outputs are written below `outputs/`. Curated experiment results are
versioned with the repository; see [outputs/README.md](outputs/README.md) for an
index. Dataset fragments are cached below `data/` and are excluded.

These runners measure an experimental Python/PyTorch decoding stack. Reported
end-to-end latency includes framework and instrumentation overhead and should
not be interpreted as a production serving benchmark.
