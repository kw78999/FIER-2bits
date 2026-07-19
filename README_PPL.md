# WikiText-2 sparse decode PPL comparison

This runner compares one dense baseline and five KV-retrieval methods on
WikiText-2:

- `full`: exact attention over the full prefix KV cache.
- `pqsift`: PCA sign/quantile bucket intersection from `PQ_SIFT.ipynb`.
- `loki`: low-rank PCA score followed by token-level top-k retrieval.
- `quest`: page min/max upper-bound scoring and top-page retrieval.
- `fier`: group-wise 1-bit RTN key scoring and token-level top-k retrieval. The default `triton` backend packs 32 token bits per int32 and fuses dequantization with QK GEMV.
- `bit2_qk`: packed 2-bit sign/magnitude QK scoring followed by token-level
  top-k retrieval. Q uses min/2 and max/2 across the current query's head
  dimensions; K uses min/2 and max/2 per token group and channel. Selected
  tokens use the original BF16/FP16 K/V for exact attention.

`bit2_qk` stores sign and magnitude as two packed bitplanes (two physical bits
per scalar). Its packed score uses sign XNOR plus magnitude OR/AND:
`2*popcnt(S)-D + 2*popcnt(S&O)-popcnt(O) + 2*popcnt(S&A)-popcnt(A)`, where
`S=sign XNOR`, `O=q_mag OR k_mag`, and `A=q_mag AND k_mag`. This gives per-channel
weights 1/2/3 for neither/either/both large, with the sign flipped when signs
differ. The `reference` backend uses PyTorch uint8 bitplanes; `cuda_popc` and
`cuda_popc_histogram` use packed int32 words and compiled CUDA kernels.

The supplied PCA cache is used by PQ-SIFT and Loki. Its internal `model_id`
must exactly match the requested model.

Run the quality comparison with optional exact-QK Top-K recall diagnostics:

```bash
python3 run_wikitext2_ppl.py \
  --methods full,fier,bit2_qk \
  --context-length 4096 \
  --num-blocks 8 \
  --query-positions 511,1023,1535,2047,2559,3071,3583,4095 \
  --budget-sweep 512,1024,2048 \
  --group-size-sweep 32,64 \
  --measure-topk-recall
```

## Optimized FIER backend

`--fier-backend triton` (the default) uses `fier_triton.py`. It supports FP16, BF16, FP32, partial final groups, and GQA/MQA head mapping. Use `--fier-backend reference` for the original PyTorch baseline. The kernel layout is adapted from the [official FIER implementation](https://github.com/SimWangArizona/FIER) at commit `e0b34153591dd7a55171f09f30abee35b0f08356`; the local implementation fixes partial-group handling and validates every score against `fier_dequantize_1bit`.

## Evaluation semantics

The experiment reports **sampled decode-style PPL**. For each selected position,
the prefix is processed once by dense SDPA to build a shared KV cache. The
current token is then independently decoded by every method, and its logits are
scored against the next WikiText-2 token.

Quest and the `reference` FIER/bit2 paths are quality implementations. The default FIER `triton` path follows the authors’ public group-wise quantize-pack and fused dequantization/GEMV design; CUDA `torch.topk` performs token recall. The exact selected attention remains in PyTorch, so end-to-end timing is an evaluator measurement rather than a reproduction of the authors’ FlashInfer/Quest stack. PQ-SIFT uses a variable candidate count determined
by its keep ratio; Loki/FIER use a token budget and Quest rounds the budget to
whole pages.

## Setup

```bash
python3 -m pip install -r requirements-ppl.txt
export HF_TOKEN="your Hugging Face token"
```

The default model is gated `meta-llama/Llama-3.1-8B`. The token must have access
to that exact base model. Do not substitute the Instruct model: the supplied PCA
basis metadata is for the base model.

Validate arguments and the PCA cache without downloading anything:

```bash
python3 run_wikitext2_ppl.py --dry-run
```

Small functional run:

```bash
python3 run_wikitext2_ppl.py \
  --context-length 4096 \
  --num-blocks 1 \
  --query-positions 4095 \
  --methods full,pqsift,loki,quest,fier
```

PQ-SIFT axis sweep at a fixed per-axis keep ratio:

```bash
python3 run_wikitext2_ppl.py \
  --methods full,pqsift \
  --context-length 4096 \
  --num-blocks 1 \
  --query-positions 4095 \
  --pqsift-axes-sweep 2,3,4,5,6 \
  --pqsift-r 0.75
```

PQ-SIFT axis/ratio grid with comparison baselines:

```bash
python3 run_wikitext2_ppl.py \
  --methods full,pqsift,loki,quest,fier \
  --context-length 4096 \
  --num-blocks 8 \
  --query-positions 1023,1279,1535,1791,2047,2303,2559,2815,3071,3327,3583,3839,4095 \
  --pqsift-axes-sweep 2,3,4,5,6,7 \
  --pqsift-r-sweep 0.7,0.75,0.8 \
  --budget 512
```

Axis and ratio sweep values form a Cartesian product. All variants for one
sample share the same dense prefix KV cache.

Fixed-4K comparison with two baseline budgets:

```bash
python3 run_wikitext2_ppl.py \
  --methods full,pqsift,loki,quest,fier \
  --context-length 4096 \
  --num-blocks 32 \
  --query-positions 4095 \
  --pqsift-axes-sweep 2,3,4,5,6,7 \
  --pqsift-r-sweep 0.7,0.75,0.8 \
  --budget-sweep 512,1024
```

Larger sampled run:

```bash
python3 run_wikitext2_ppl.py \
  --context-length 4096 \
  --num-blocks 4 \
  --query-positions 1023,2047,3071,4095 \
  --budget 512 \
  --pqsift-axes 4 \
  --pqsift-r 0.75 \
  --loki-rank 64 \
  --quest-page-size 16 \
  --fier-group-size 32
```

Outputs are written under `outputs/ppl_<timestamp>/`:

- `metadata.json`
- `samples.jsonl`
- `samples.csv` (all per-token decode/search/attention timings)
- `summary.json`
- `summary.csv`

Each sample and summary includes separate component timings:

- `candidate_search_ms`: candidate selection over all layers and heads.
- `candidate_search_ops_proxy`: rough candidate-search operation proxy covering
  min/max or distribution analysis, quantization/dequantization or bit operations,
  score accumulation, and final Top-K selection.
- `selected_attention_ms`: exact attention over the selected KV tokens.
- `decode_ms`: the complete one-token decode, including projections, MLPs,
  component timing instrumentation, and Python overhead.

CUDA component timings use CUDA events. They are diagnostic timings for this
PyTorch reference implementation, not production-kernel latency.

The first two transformer layers remain dense for every sparse method, matching
the Quest/FIER evaluation convention. Override this with `--full-layers`.
