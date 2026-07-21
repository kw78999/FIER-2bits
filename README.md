# Pop-KV

> Anonymous artifact for a COLM Workshop submission. Author names, affiliations, and identifying links are intentionally omitted for double-blind review.

Pop-KV is a training-free KV-cache retrieval method for long-context Llama decoding. It encodes query and key channels with sign and magnitude indicators, retains two representative magnitudes per 32-channel word, and ranks cached tokens with a fused CUDA bit-popcount score. Only selected full-precision K/V entries are gathered for exact scaled dot-product attention.

This repository contains the final method used in the submission and exactly three comparison paths:

- `full`: dense FullAttention;
- `quest`: page-level min/max upper-bound retrieval with persistent page metadata;
- `fier`: FIER-style sequence-grouped one-bit RTN retrieval with a Triton scorer;
- `popkv`: the proposed representative-weighted bit-popcount retrieval.

Historical prototypes and ablations are deliberately excluded.

## Method

For channel `i` in 32-channel word `w`, Pop-KV reconstructs a query or key as

```text
x_hat[i] = sign[i] * (low[w] + magnitude[i] * delta[w])
```

Here, `sign` is ±1, `magnitude` is binary, `low` is the mean absolute value of low-magnitude channels, and `delta = high - low`. Magnitude thresholds are `max/2` for positive values and `min/2` for negative values. Key thresholds are computed per 32-token group and channel; query thresholds are computed per query head.

The approximate score is evaluated without materializing reconstructed vectors:

```text
score = sum_w (q_low*k_low*Csign
             + q_delta*k_low*Cq
             + q_low*k_delta*Ck
             + q_delta*k_delta*Cqk) / sqrt(head_dim)
```

`Csign`, `Cq`, `Ck`, and `Cqk` are signed agreement counts produced by 32-bit popcount operations. Retrieval is performed independently for every query head. The newest token always reserves one budget slot.

## Repository layout

```text
popkv.py                  Final Pop-KV, FullAttention, Quest, and FIER paths
popkv_cuda.py             JIT interface for the Pop-KV CUDA extension
csrc/popkv_ext.{cpp,cu}   Final packing and fused scoring kernels
fier_triton.py            FIER comparison scorer
run_pg19.py               PG19 sampled-decode PPL and latency experiment
run_longbench.py          LongBench quality experiment
tests/test_popkv.py       CPU and CUDA correctness tests
results/                  Compact paper-result tables
```

## Installation

The released kernels require Linux, an NVIDIA GPU, a CUDA toolkit with `nvcc`, and a CUDA-enabled PyTorch installation. The reported run used an RTX A6000, PyTorch 2.4.1, and CUDA 12.4.

```bash
git clone <anonymous-repository-url>
cd PopKV
python -m venv .venv
source .venv/bin/activate

# Install a CUDA-enabled PyTorch build appropriate for the local system first.
pip install torch
pip install -r requirements.txt
```

The CUDA extension is compiled just in time on the first Pop-KV call. Build products are stored in PyTorch's extension cache, not in this repository.

## Hugging Face authentication

Both reported models are gated Llama checkpoints. Request access on Hugging Face and authenticate once:

```bash
hf auth login
```

Alternatively:

```bash
export HF_TOKEN=hf_your_read_token
```

Tokens are never written to result files. `.env`, model caches, datasets, generated outputs, and logs are excluded by `.gitignore`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests verify the public four-method surface, fixed release configuration, Quest metadata-cache equivalence across page boundaries, and Pop-KV CUDA scores against a readable reconstruction reference.

Configuration-only checks do not download a model:

```bash
python run_pg19.py --dry-run
python run_longbench.py --dry-run
```

## PG19 reproduction

A short smoke test is recommended first:

```bash
python run_pg19.py \
  --context-length 4096 \
  --generate-tokens 8 \
  --num-blocks 1 \
  --budget 1024 \
  --output-dir outputs/pg19_smoke
```

The main 32K experiment reported below is reproduced with:

```bash
python run_pg19.py \
  --model meta-llama/Llama-3.1-8B \
  --methods full,quest,fier,popkv \
  --context-length 32768 \
  --budget 4096 \
  --group-size 32 \
  --generate-tokens 128 \
  --num-blocks 32 \
  --output-dir outputs/pg19_32k_b4096_t128_blocks32
```

The loader downloads the official PG19 test-file manifest from Hugging Face and caches the referenced public-domain books under `data/`. Use `--text-file PATH` to run on a local text corpus instead.

### PG19 results

The table contains 32 non-overlapping blocks and 4,096 evaluated next tokens per method. Latencies are milliseconds per decoded token; PPL is `exp(mean NLL)` over all evaluated tokens.

| Method | PPL | Score | TopK | Search | Gather | Attention | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullAttention | 8.884 | 0.00 | 0.00 | 0.00 | 19.10 | 122.00 | 167.69 |
| Quest | 10.038 | 17.96 | 5.07 | 23.03 | 18.41 | 13.83 | 131.90 |
| FIER | 8.930 | 9.06 | 2.67 | 11.73 | 18.52 | 10.74 | 129.60 |
| **Pop-KV** | **9.052** | **3.41** | **2.42** | **5.84** | 18.61 | 10.74 | **124.79** |

Relative to FullAttention, Pop-KV reduces measured decode latency by 25.6% while increasing PPL by 1.9%. Relative to FIER, its scoring stage is 2.65x faster and end-to-end decode is 3.7% faster, with a 1.36% PPL increase. These timings are hardware- and software-dependent; use the provided runner for local comparisons.

Machine-readable results: [`results/pg19_32k_b4096_t128_blocks32.csv`](results/pg19_32k_b4096_t128_blocks32.csv).

## LongBench reproduction

The quality pilot uses Llama-3.1-8B-Instruct, six LongBench tasks, 12 examples per task, greedy decoding, and middle truncation at 32K tokens:

```bash
python run_longbench.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --methods full,quest,fier,popkv \
  --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa,gov_report,triviaqa \
  --budgets 512,1024,2048,4096 \
  --samples-per-dataset 12 \
  --max-context 32768 \
  --group-size 32 \
  --hard-limit-minutes 240 \
  --stop-reserve-minutes 10 \
  --output-dir outputs/longbench_n12
```

The runner is resumable and writes every completed prediction before advancing. QA tasks use token-level F1 and `gov_report` uses ROUGE-L, following the included evaluator.

### LongBench results

Scores are unweighted averages over six tasks. “Retention” averages each task's percentage relative to FullAttention.

| Method | Budget | Task average | Mean retention | Completed |
|---|---:|---:|---:|---:|
| FullAttention | full | 55.601 | 100.00% | 72 |
| Quest | 2048 | 52.049 | 95.34% | 72 |
| FIER | 2048 | 54.282 | 98.45% | 72 |
| **Pop-KV** | **2048** | **56.024** | **101.82%** | **72** |
| Quest | 4096 | 54.383 | 98.77% | 72 |
| FIER | 4096 | 53.997 | 97.15% | 72 |
| **Pop-KV** | **4096** | **53.910** | **96.94%** | **72** |

The pilot is small and exhibits non-monotonic budget effects. A score above FullAttention at budget 2048 should be interpreted as competitive quality within this sample, not evidence that sparse retrieval improves the underlying model. Full per-task and all-budget results are available in [`results/longbench_per_task.csv`](results/longbench_per_task.csv) and [`results/longbench_summary.csv`](results/longbench_summary.csv).

## Why set recall can look low while quality remains close

An exploratory diagnostic at budget 4096 found that Pop-KV's exact top-4096 overlap was 0.484 at 32K and 0.422 at 64K, while it retained 0.927 and 0.913 of exact softmax attention mass. Attention outputs computed on the selected set had cosine similarity 0.982 and 0.985 to full attention. This suggests that many set disagreements occur among low-mass tokens near the selection boundary. Diagnostic data: [`results/recall_quality_diagnostic.csv`](results/recall_quality_diagnostic.csv).

## Scope and limitations

- This is a research prototype for Llama-style grouped-query attention, not a drop-in replacement for every Transformers architecture.
- Prefix prefill remains dense. The experiments target autoregressive decode after a long prefix.
- The first two transformer layers use FullAttention in sparse configurations.
- Selected K/V values remain in the model's original precision; Pop-KV compresses retrieval metadata and approximate scoring, not the values used by final attention.
- PG19 “PPL” is sampled next-token decode PPL from fixed long-context blocks, not conventional full-corpus teacher-forced PPL.
- LongBench uses only 12 examples per task and is an exploratory workshop-scale evaluation.
- CUDA JIT compilation and hardware differences affect absolute latency. Compare methods in the same process and environment.
- The repository intentionally contains no author, affiliation, machine hostname, private path, access token, or raw prediction text.

## Anonymous citation

Citation metadata will be added after the double-blind review period. During review, please refer to the method and artifact as **Pop-KV**.
