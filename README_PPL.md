# WikiText-2 sparse decode PPL comparison

This runner compares one dense baseline and five KV-retrieval methods on
WikiText-2:

- `full`: exact attention over the full prefix KV cache.
- `pqsift`: PCA sign/quantile bucket intersection from `PQ_SIFT.ipynb`.
- `loki`: low-rank PCA score followed by token-level top-k retrieval.
- `quest`: page min/max upper-bound scoring and top-page retrieval.
- `fier`: group-32, 1-bit RTN key scoring and token-level top-k retrieval.
- `bit2_qk`: packed 2-bit sign/magnitude QK scoring followed by token-level
  top-k retrieval. Q uses min/2 and max/2 across the current query's head
  dimensions; K uses min/2 and max/2 per token group and channel. Selected
  tokens use the original BF16/FP16 K/V for exact attention.

`bit2_qk` stores sign and magnitude as two packed uint8 bitplanes (two physical
bits per scalar). Its packed score uses sign XNOR plus magnitude OR/AND:
`2*popcnt(S)-D + 2*popcnt(S&O)-popcnt(O) + 2*popcnt(S&A)-popcnt(A)`, where
`S=sign XNOR`, `O=q_mag OR k_mag`, and `A=q_mag AND k_mag`. This gives per-channel
weights 1/2/3 for neither/either/both large, with the sign flipped when signs differ.
This is a correctness reference; it is not a fused CUDA/Triton popcount kernel.

The supplied PCA cache is used by PQ-SIFT and Loki. Its internal `model_id`
must exactly match the requested model.

Run the quality comparison with optional exact-QK Top-K recall diagnostics:

```bash
python /workspace/run_wikitext2_ppl.py \
  --methods full,fier,bit2_qk \
  --context-length 4096 \
  --num-blocks 8 \
  --query-positions 511,1023,1535,2047,2559,3071,3583,4095 \
  --budget-sweep 512,1024,2048 \
  --group-size-sweep 32,64 \
  --measure-topk-recall
```

## Evaluation semantics

The experiment reports **sampled decode-style PPL**. For each selected position,
the prefix is processed once by dense SDPA to build a shared KV cache. The
current token is then independently decoded by every method, and its logits are
scored against the next WikiText-2 token.

This is a quality/reference implementation. Quest, FIER, and bit2_qk follow their
retrieval rules in PyTorch, but do not use custom fused CUDA/Triton kernels.
Consequently, timing is diagnostic and must not be reported as a
kernel-speed reproduction. PQ-SIFT uses a variable candidate count determined
by its keep ratio; Loki/FIER use a token budget and Quest rounds the budget to
whole pages.

## Setup

```bash
pip install -r /workspace/requirements-ppl.txt
export HF_TOKEN="your Hugging Face token"
```

The default model is gated `meta-llama/Llama-3.1-8B`. The token must have access
to that exact base model. Do not substitute the Instruct model: the supplied PCA
basis metadata is for the base model.

Validate arguments and the PCA cache without downloading anything:

```bash
python /workspace/run_wikitext2_ppl.py --dry-run
```

Small functional run:

```bash
python /workspace/run_wikitext2_ppl.py \
  --context-length 4096 \
  --num-blocks 1 \
  --query-positions 4095 \
  --methods full,pqsift,loki,quest,fier
```

PQ-SIFT axis sweep at a fixed per-axis keep ratio:

```bash
python /workspace/run_wikitext2_ppl.py \
  --methods full,pqsift \
  --context-length 4096 \
  --num-blocks 1 \
  --query-positions 4095 \
  --pqsift-axes-sweep 2,3,4,5,6 \
  --pqsift-r 0.75
```

PQ-SIFT axis/ratio grid with comparison baselines:

```bash
python /workspace/run_wikitext2_ppl.py \
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
python /workspace/run_wikitext2_ppl.py \
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
python /workspace/run_wikitext2_ppl.py \
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

Outputs are written under `/workspace/outputs/ppl_<timestamp>/`:

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
