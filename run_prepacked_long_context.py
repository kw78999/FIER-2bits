from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sparse_ppl import (
    EvaluationVariant,
    LlamaSparseDecoder,
    PPLResult,
    RetrievalConfig,
    _safe_perplexity,
    parse_int_tuple,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepacked FIER/bit2 continuous-decode benchmark on PG19 or LongBench text."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--benchmark", choices=("pg19_ppl", "longbench_ppl"), default="pg19_ppl")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument(
        "--token-prefixes", default=None,
        help="Comma-separated decode prefixes summarized from one continuous run.",
    )
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--methods", default="fier,bit2_qk")
    parser.add_argument("--budget-sweep", default="1024,2048")
    parser.add_argument("--group-size-sweep", default="32,64")
    parser.add_argument(
        "--fier-backend", choices=("reference", "triton"), default="triton"
    )
    parser.add_argument(
        "--bit2-backend",
        choices=(
            "reference",
            "cuda_popc",
            "cuda_popc_direct",
            "cuda_popc_histogram",
            "group_mean4",
            "group_mean2",
            "cuda_2mean",
            "triton_2mean",
            "2mean",
            "qk_2mean",
            "cuda_qk_2mean",
            "triton_qk_2mean",
        ),
        default="reference",
    )
    parser.add_argument(
        "--bit2-backend-sweep",
        default=None,
        help="Comma-separated 2-bit backends evaluated on the same blocks.",
    )
    parser.add_argument("--full-layers", default="0,1")
    parser.add_argument(
        "--measure-topk-recall",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--measure-score-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Measure Q/K 2mean MAE, normalized MAE, and Spearman diagnostics. "
            "Independent of top-k recall; adds substantial latency."
        ),
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dataset-trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-name", default="deepmind/pg19")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--pg19-loader", choices=("partial", "hf"), default="partial")
    parser.add_argument("--pg19-cache-dir", default="data/pg19_partial")
    parser.add_argument("--pg19-max-books", type=int, default=64)
    parser.add_argument("--text-file", default=None, help="Local plain text file used instead of datasets.load_dataset.")
    parser.add_argument("--longbench-subset", default="qasper")
    parser.add_argument("--max-longbench-samples", type=int, default=1)
    return parser.parse_args()


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [m for m in methods if m not in {"full", "quest", "fier", "bit2_qk"}]
    if unknown:
        raise ValueError(f"This runner supports full/Quest/FIER/bit2, got {unknown}")
    return methods


def load_model_and_tokenizer(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("Install transformers/accelerate first") from exc

    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=token, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def _row_text(row: dict[str, Any]) -> str:
    for key in ("text", "context", "input", "document", "prompt"):
        if key in row and str(row[key]).strip():
            return str(row[key])
    pieces = []
    for key, value in row.items():
        if isinstance(value, str):
            pieces.append(value)
    return "\n".join(pieces)


def build_pg19_partial_blocks(tokenizer: Any, args: argparse.Namespace) -> list[torch.Tensor]:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required for PG19 partial loading") from exc

    if args.dataset_split not in {"train", "validation", "test"}:
        raise ValueError("PG19 partial loader expects train/validation/test split")
    split_list = hf_hub_download(
        repo_id="deepmind/pg19",
        repo_type="dataset",
        filename=f"data/{args.dataset_split}_files.txt",
    )
    file_names = [line.strip() for line in Path(split_list).read_text().splitlines() if line.strip()]
    required = args.num_blocks * (args.context_length + args.generate_tokens)
    cache_dir = Path(args.pg19_cache_dir).expanduser().resolve() / args.dataset_split
    cache_dir.mkdir(parents=True, exist_ok=True)
    stream: list[int] = []
    eos = tokenizer.eos_token_id
    root = "https://storage.googleapis.com/deepmind-gutenberg/"
    used_books = 0
    for name in sorted(file_names):
        if used_books >= args.pg19_max_books:
            break
        local = cache_dir / Path(name).name
        if not local.exists():
            urllib.request.urlretrieve(root + name, local)
        text = local.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            stream.extend(int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"])
            if eos is not None:
                stream.append(int(eos))
        used_books += 1
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError(
            f"PG19 partial loader produced {len(stream)} tokens from {used_books} books; "
            f"need {required}. Increase --pg19-max-books."
        )
    block_size = args.context_length + args.generate_tokens
    print(
        f"PG19 partial loader: used_books={used_books} tokens={len(stream)} "
        f"required={required} cache_dir={cache_dir}",
        flush=True,
    )
    return [
        torch.tensor(stream[i * block_size : (i + 1) * block_size], dtype=torch.long)
        for i in range(args.num_blocks)
    ]


def build_text_file_blocks(tokenizer: Any, args: argparse.Namespace) -> list[torch.Tensor]:
    path = Path(args.text_file).expanduser().resolve()
    text = path.read_text(errors="ignore")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        ids.append(int(eos))
    required = args.num_blocks * (args.context_length + args.generate_tokens)
    if len(ids) < required:
        repeats = math.ceil(required / max(1, len(ids)))
        ids = (ids * repeats)[:required]
    block_size = args.context_length + args.generate_tokens
    return [
        torch.tensor(ids[i * block_size : (i + 1) * block_size], dtype=torch.long)
        for i in range(args.num_blocks)
    ]


def build_pg19_blocks(tokenizer: Any, args: argparse.Namespace) -> list[torch.Tensor]:
    from datasets import load_dataset

    kwargs = {"split": args.dataset_split, "trust_remote_code": args.dataset_trust_remote_code}
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        dataset = load_dataset(args.dataset_name, **kwargs)
    required = args.num_blocks * (args.context_length + args.generate_tokens)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id")
    stream: list[int] = []
    for row in dataset:
        text = _row_text(dict(row))
        if not text.strip():
            continue
        stream.extend(int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"])
        stream.append(int(eos))
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError(f"Dataset produced {len(stream)} tokens; need {required}")
    block_size = args.context_length + args.generate_tokens
    return [torch.tensor(stream[i * block_size : (i + 1) * block_size], dtype=torch.long) for i in range(args.num_blocks)]


def build_longbench_blocks(tokenizer: Any, args: argparse.Namespace) -> list[torch.Tensor]:
    from datasets import load_dataset

    dataset = load_dataset(
        "THUDM/LongBench",
        args.longbench_subset,
        split="test",
        trust_remote_code=args.dataset_trust_remote_code,
    )
    blocks = []
    needed = args.context_length + args.generate_tokens
    eos = tokenizer.eos_token_id
    for row in dataset.select(range(min(args.max_longbench_samples, len(dataset)))):
        text = _row_text(dict(row))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < needed:
            ids = ids + [int(eos)] * (needed - len(ids))
        blocks.append(torch.tensor(ids[:needed], dtype=torch.long))
    if not blocks:
        raise RuntimeError("No LongBench blocks built")
    return blocks


def make_variants(
    methods: tuple[str, ...],
    retrieval: RetrievalConfig,
    budgets: tuple[int, ...],
    groups: tuple[int, ...],
    bit2_backends: tuple[str, ...],
) -> list[EvaluationVariant]:
    variants = []
    for method in methods:
        if method == "full":
            variants.append(EvaluationVariant("full", "full", retrieval))
            continue
        if method == "quest":
            for budget in budgets:
                variants.append(
                    EvaluationVariant(
                        "quest", f"quest_b{budget}", replace(retrieval, budget=budget)
                    )
                )
            continue
        for group in groups:
            for budget in budgets:
                backends = (
                    bit2_backends
                    if method == "bit2_qk"
                    else (retrieval.fier_backend,)
                )
                for selected_backend in backends:
                    backend_update = (
                        {"bit2_backend": selected_backend}
                        if method == "bit2_qk"
                        else {}
                    )
                    variant_retrieval = replace(
                        retrieval,
                        budget=budget,
                        **(
                            {"fier_group_size": group}
                            if method == "fier"
                            else {"bit2_group_size": group}
                        ),
                        **backend_update,
                    )
                    backend = (
                        variant_retrieval.fier_backend
                        if method == "fier"
                        else variant_retrieval.bit2_backend
                    )
                    variants.append(
                        EvaluationVariant(
                            method,
                            (
                                f"bit2_qk_2mean_g{group}_b{budget}"
                                if method == "bit2_qk" and backend == "qk_2mean"
                                else f"{method}_{backend}_g{group}_b{budget}"
                            ),
                            variant_retrieval,
                        )
                    )
    return variants


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def continuous_ppl(decoder: LlamaSparseDecoder, blocks: list[torch.Tensor], variants: list[EvaluationVariant], *, context_length: int, generate_tokens: int) -> PPLResult:
    result = PPLResult()
    for block_idx, block in enumerate(blocks):
        if block.numel() < context_length + generate_tokens:
            raise ValueError("block too short")
        prefix_ids = block[: context_length - 1]
        current_ids = block[context_length - 1 : context_length - 1 + generate_tokens]
        target_ids = block[context_length : context_length + generate_tokens]

        prefix_started = time.perf_counter()
        base_past = decoder.build_prefix_cache(prefix_ids)
        _sync(decoder.device)
        prefix_ms = (time.perf_counter() - prefix_started) * 1000.0

        for variant in variants:
            decoder.retrieval = variant.retrieval
            method = variant.method
            if method == "fier":
                group_size = variant.retrieval.fier_group_size
            elif method == "bit2_qk":
                group_size = variant.retrieval.bit2_group_size
            else:
                group_size = 0

            use_packed_cache = (
                (method == "fier")
                or (method == "bit2_qk" and variant.retrieval.bit2_backend == "reference")
                or (
                    method == "bit2_qk"
                    and variant.retrieval.bit2_backend == "cuda_popc"
                )
                or (
                    method == "bit2_qk"
                    and variant.retrieval.bit2_backend in {
                        "2mean", "cuda_2mean", "triton_2mean",
                        "qk_2mean", "cuda_qk_2mean", "triton_qk_2mean",
                    }
                )
            )
            if not use_packed_cache:
                packed = None
                prepack_ms = 0.0
            else:
                pack_started = time.perf_counter()
                packed = decoder.build_packed_retrieval_caches(base_past, method=method, group_size=group_size)
                _sync(decoder.device)
                prepack_ms = (time.perf_counter() - pack_started) * 1000.0

            past = decoder.build_static_kv_cache(
                base_past, reserve_tokens=generate_tokens
            )
            _sync(decoder.device)
            # One unmeasured decode removes Triton/CUDA JIT and allocator warm-up
            # from the reported search, attention, and end-to-end decode latency.
            decoder.decode(
                past,
                current_ids[0],
                method=method,
                retrieval_caches=packed,
                return_new_kv=False,
            )
            _sync(decoder.device)

            total_nll = 0.0
            total_decode_ms = 0.0
            total_search_ms = 0.0
            total_score_ms = 0.0
            total_topk_ms = 0.0
            total_gather_ms = 0.0
            total_update_ms = 0.0
            total_attention_ms = 0.0
            total_ops = 0.0
            total_ratio = 0.0
            recall_sum = 0.0
            recall_calls = 0

            for step, (current_id, target_id) in enumerate(zip(current_ids, target_ids)):
                _sync(decoder.device)
                started = time.perf_counter()
                logits, diagnostics, new_keys, new_values = decoder.decode(
                    past,
                    current_id,
                    method=method,
                    retrieval_caches=packed,
                    return_new_kv=True,
                )
                _sync(decoder.device)
                decode_ms = (time.perf_counter() - started) * 1000.0

                update_started = time.perf_counter()
                past = decoder.append_to_past_key_values(past, new_keys, new_values)
                if packed is not None:
                    decoder.append_to_packed_retrieval_caches(packed, new_keys)
                _sync(decoder.device)
                update_ms = (time.perf_counter() - update_started) * 1000.0
                diagnostics.cache_update_ms = update_ms

                target = target_id.to(decoder.device).view(1)
                nll = F.cross_entropy(logits.view(1, -1), target).item()
                total_nll += nll
                total_decode_ms += decode_ms
                total_search_ms += diagnostics.candidate_search_ms
                total_score_ms += diagnostics.candidate_score_ms
                total_topk_ms += diagnostics.candidate_topk_ms
                total_gather_ms += diagnostics.selected_gather_ms
                total_update_ms += update_ms
                total_attention_ms += diagnostics.selected_attention_ms
                total_ops += diagnostics.candidate_search_ops
                total_ratio += diagnostics.candidate_ratio
                if diagnostics.topk_recall is not None:
                    recall_sum += diagnostics.topk_recall
                    recall_calls += 1

                is_mean_backend = method == "bit2_qk" and "2mean" in variant.retrieval.bit2_backend
                words_per_token = math.ceil(decoder.head_dim / 32)

                result.samples.append({
                    "block_idx": block_idx,
                    "step": step,
                    "context_tokens": context_length + step,
                    "method": variant.label,
                    "nll": float(nll),
                    "candidate_ratio": diagnostics.candidate_ratio,
                    "prefix_ms_shared": prefix_ms,
                    "prepack_ms": prepack_ms,
                    "decode_ms": decode_ms,
                    "candidate_search_ms": diagnostics.candidate_search_ms,
                    "candidate_score_ms": diagnostics.candidate_score_ms,
                    "candidate_topk_ms": diagnostics.candidate_topk_ms,
                    "selected_gather_ms": diagnostics.selected_gather_ms,
                    "cache_update_ms": update_ms,
                    "selected_attention_ms": diagnostics.selected_attention_ms,
                    "candidate_search_ops_proxy": diagnostics.candidate_search_ops,
                    "topk_recall": diagnostics.topk_recall,
                    "topk_recall_1024": diagnostics.topk_recall_at(1024),
                    "topk_recall_2048": diagnostics.topk_recall_at(2048),
                    "topk_recall_3072": diagnostics.topk_recall_at(3072),
                    "topk_recall_4096": diagnostics.topk_recall_at(4096),
                    "attention_mass_recall": diagnostics.attention_mass_recall,
                    "score_mae": diagnostics.score_mae,
                    "score_normalized_mae": diagnostics.score_normalized_mae,
                    "spearman_correlation": diagnostics.spearman,
                    "metadata_bytes_per_token": 4 * words_per_token if is_mean_backend else 0,
                    "packed_k_bytes_per_token": (12 * words_per_token if is_mean_backend else (8 * words_per_token if method == "bit2_qk" else None)),
                    "budget": variant.retrieval.budget,
                    "group_size": group_size,
                })

            mean_nll = total_nll / generate_tokens
            print(
                f"block={block_idx} method={variant.label} mean_nll={mean_nll:.5f} "
                f"ppl={_safe_perplexity(mean_nll):.3f} prepack_ms={prepack_ms:.1f} "
                f"search_ms/tok={total_search_ms/generate_tokens:.1f} "
                f"score_ms/tok={total_score_ms/generate_tokens:.1f} "
                f"topk_ms/tok={total_topk_ms/generate_tokens:.1f} "
                f"gather_ms/tok={total_gather_ms/generate_tokens:.1f} "
                f"update_ms/tok={total_update_ms/generate_tokens:.3f} "
                f"decode_ms/tok={total_decode_ms/generate_tokens:.1f} "
                f"topk_recall={(recall_sum/recall_calls if recall_calls else float('nan')):.4f}",
                flush=True,
            )
            del past, packed
            if decoder.device.type == "cuda":
                torch.cuda.empty_cache()
    return result


def summarize_continuous(result: PPLResult) -> list[dict[str, Any]]:
    rows = []
    methods = sorted({str(r["method"]) for r in result.samples})
    for method in methods:
        rs = [r for r in result.samples if r["method"] == method]
        mean_nll = sum(float(r["nll"]) for r in rs) / len(rs)
        recall_values = [float(r["topk_recall"]) for r in rs if r.get("topk_recall") is not None]
        recall_by_k_values = {
            k: [float(r[f"topk_recall_{k}"]) for r in rs if r.get(f"topk_recall_{k}") is not None]
            for k in (1024, 2048, 3072, 4096)
        }
        mass_values = [float(r["attention_mass_recall"]) for r in rs if r.get("attention_mass_recall") is not None]
        mae_values = [float(r["score_mae"]) for r in rs if r.get("score_mae") is not None]
        normalized_mae_values = [float(r["score_normalized_mae"]) for r in rs if r.get("score_normalized_mae") is not None]
        spearman_values = [float(r["spearman_correlation"]) for r in rs if r.get("spearman_correlation") is not None]
        rows.append({
            "method": method,
            "num_tokens": len(rs),
            "mean_nll": mean_nll,
            "perplexity": _safe_perplexity(mean_nll),
            "mean_candidate_ratio": sum(float(r["candidate_ratio"]) for r in rs) / len(rs),
            "mean_prepack_ms": sum(float(r["prepack_ms"]) for r in rs) / len(rs),
            "mean_decode_ms_per_token": sum(float(r["decode_ms"]) for r in rs) / len(rs),
            "mean_candidate_search_ms_per_token": sum(float(r["candidate_search_ms"]) for r in rs) / len(rs),
            "mean_candidate_score_ms_per_token": sum(float(r["candidate_score_ms"]) for r in rs) / len(rs),
            "mean_candidate_topk_ms_per_token": sum(float(r["candidate_topk_ms"]) for r in rs) / len(rs),
            "mean_selected_gather_ms_per_token": sum(float(r["selected_gather_ms"]) for r in rs) / len(rs),
            "mean_cache_update_ms_per_token": sum(float(r["cache_update_ms"]) for r in rs) / len(rs),
            "mean_selected_attention_ms_per_token": sum(float(r["selected_attention_ms"]) for r in rs) / len(rs),
            "mean_candidate_search_ops_proxy": sum(float(r["candidate_search_ops_proxy"]) for r in rs) / len(rs),
            "mean_topk_recall": None if not recall_values else sum(recall_values) / len(recall_values),
            "mean_topk_recall_1024": None if not recall_by_k_values[1024] else sum(recall_by_k_values[1024]) / len(recall_by_k_values[1024]),
            "mean_topk_recall_2048": None if not recall_by_k_values[2048] else sum(recall_by_k_values[2048]) / len(recall_by_k_values[2048]),
            "mean_topk_recall_3072": None if not recall_by_k_values[3072] else sum(recall_by_k_values[3072]) / len(recall_by_k_values[3072]),
            "mean_topk_recall_4096": None if not recall_by_k_values[4096] else sum(recall_by_k_values[4096]) / len(recall_by_k_values[4096]),
            "mean_attention_mass_recall": None if not mass_values else sum(mass_values) / len(mass_values),
            "mean_score_mae": None if not mae_values else sum(mae_values) / len(mae_values),
            "mean_score_normalized_mae": None if not normalized_mae_values else sum(normalized_mae_values) / len(normalized_mae_values),
            "mean_spearman_correlation": None if not spearman_values else sum(spearman_values) / len(spearman_values),
            "metadata_bytes_per_token": rs[0].get("metadata_bytes_per_token"),
            "packed_k_bytes_per_token": rs[0].get("packed_k_bytes_per_token"),
        })
    return rows


def write_outputs(output_dir: Path, result: PPLResult, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_continuous(result)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    with (output_dir / "samples.jsonl").open("w") as h:
        for row in result.samples:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    if result.samples:
        fields = list(dict.fromkeys(k for row in result.samples for k in row.keys()))
        with (output_dir / "samples.csv").open("w", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=fields)
            writer.writeheader(); writer.writerows(result.samples)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if summary:
        with (output_dir / "summary.csv").open("w", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(summary[0].keys()))
            writer.writeheader(); writer.writerows(summary)
    for prefix in metadata.get("token_prefixes") or ():
        prefix_result = PPLResult(samples=[
            row for row in result.samples if int(row["step"]) < int(prefix)
        ])
        prefix_summary = summarize_continuous(prefix_result)
        (output_dir / f"summary_{prefix}.json").write_text(
            json.dumps(prefix_summary, indent=2, ensure_ascii=False) + "\n"
        )
        if prefix_summary:
            with (output_dir / f"summary_{prefix}.csv").open("w", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=list(prefix_summary[0].keys()))
                writer.writeheader(); writer.writerows(prefix_summary)


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    budgets = parse_int_tuple(args.budget_sweep)
    groups = parse_int_tuple(args.group_size_sweep)
    bit2_backends = (
        (args.bit2_backend,)
        if args.bit2_backend_sweep is None
        else tuple(
            item.strip()
            for item in args.bit2_backend_sweep.split(",")
            if item.strip()
        )
    )
    valid_bit2_backends = {
        "reference",
        "cuda_popc",
        "cuda_popc_direct",
        "cuda_popc_histogram",
        "group_mean4",
        "group_mean2",
        "cuda_2mean",
        "triton_2mean",
        "2mean",
        "qk_2mean",
        "cuda_qk_2mean",
        "triton_qk_2mean",
    }
    if not bit2_backends or any(
        backend not in valid_bit2_backends for backend in bit2_backends
    ):
        raise SystemExit(f"Invalid --bit2-backend-sweep: {bit2_backends}")
    full_layers = parse_int_tuple(args.full_layers)
    token_prefixes = (
        () if args.token_prefixes is None else parse_int_tuple(args.token_prefixes)
    )
    if args.context_length < 2 or args.generate_tokens <= 0:
        raise SystemExit("context length and generate tokens must be positive")
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    retrieval = RetrievalConfig(
        budget=budgets[0],
        fier_group_size=groups[0],
        fier_backend=args.fier_backend,
        bit2_group_size=groups[0],
        bit2_backend=bit2_backends[0],
        full_layers=full_layers,
        measure_topk_recall=args.measure_topk_recall,
        measure_score_diagnostics=args.measure_score_diagnostics,
    )
    variants = make_variants(methods, retrieval, budgets, groups, bit2_backends)
    if args.dry_run:
        print(json.dumps({"benchmark": args.benchmark, "variants": [v.label for v in variants]}, indent=2))
        return

    model, tokenizer = load_model_and_tokenizer(args)
    decoder = LlamaSparseDecoder(model, pca_cache=None, retrieval=retrieval)
    if args.text_file:
        blocks = build_text_file_blocks(tokenizer, args)
    elif args.benchmark == "pg19_ppl" and args.pg19_loader == "partial":
        blocks = build_pg19_partial_blocks(tokenizer, args)
    else:
        blocks = build_pg19_blocks(tokenizer, args) if args.benchmark == "pg19_ppl" else build_longbench_blocks(tokenizer, args)
    result = continuous_ppl(decoder, blocks, variants, context_length=args.context_length, generate_tokens=args.generate_tokens)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"outputs/prepacked_{args.benchmark}_{timestamp}")
    metadata = {
        "created_at_utc": timestamp,
        "benchmark": args.benchmark,
        "model_id": args.model,
        "context_length": args.context_length,
        "generate_tokens": args.generate_tokens,
        "token_prefixes": token_prefixes,
        "num_blocks": args.num_blocks,
        "methods": methods,
        "budget_sweep": budgets,
        "group_size_sweep": groups,
        "full_layers": full_layers,
        "measure_topk_recall": args.measure_topk_recall,
        "measure_score_diagnostics": args.measure_score_diagnostics,
        "cache_policy": "persistent_2mean_cache_or_backend_specific",
        "full_baseline": "all_layers_full_attention_no_retrieval",
        "sparse_full_layers": full_layers,
        "fier_cache": "1bit_key_plus_group_minmax",
        "fier_backend": args.fier_backend,
        "bit2_backend": bit2_backends[0],
        "bit2_backend_sweep": bit2_backends,
        "group_mean_prototype": (
            "means_recomputed_from_full_k_each_decode"
            if any(backend.startswith("group_mean") for backend in bit2_backends)
            else None
        ),
        "bit2_cache": "sign_plus_magnitude_uint32_and_optional_fp16_low_delta",
        "text_file": args.text_file,
        "pg19_loader": args.pg19_loader,
        "pg19_cache_dir": args.pg19_cache_dir,
        "pg19_max_books": args.pg19_max_books,
        "dataset_name": args.dataset_name if args.benchmark == "pg19_ppl" else "THUDM/LongBench",
        "dataset_config": args.dataset_config if args.benchmark == "pg19_ppl" else args.longbench_subset,
        "dataset_split": args.dataset_split if args.benchmark == "pg19_ppl" else "test",
        "dtype": args.dtype,
        "attention": args.attention,
        "dataset_trust_remote_code": args.dataset_trust_remote_code,
        "torch_version": torch.__version__,
    }
    write_outputs(output_dir, result, metadata)
    print("\nSummary")
    for row in summarize_continuous(result):
        recall = row["mean_topk_recall"]
        recall_text = "n/a" if recall is None else f"{recall:.4f}"
        print(
            f"{row['method']:20s} tokens={row['num_tokens']:4d} "
            f"PPL={row['perplexity']:.4f} NLL={row['mean_nll']:.6f} "
            f"prepack_ms={row['mean_prepack_ms']:.1f} "
            f"search_ms/tok={row['mean_candidate_search_ms_per_token']:.1f} "
            f"score_ms/tok={row['mean_candidate_score_ms_per_token']:.1f} "
            f"topk_ms/tok={row['mean_candidate_topk_ms_per_token']:.1f} "
            f"gather_ms/tok={row['mean_selected_gather_ms_per_token']:.1f} "
            f"attention_ms/tok={row['mean_selected_attention_ms_per_token']:.1f} "
            f"update_ms/tok={row['mean_cache_update_ms_per_token']:.3f} "
            f"decode_ms/tok={row['mean_decode_ms_per_token']:.1f} "
            f"ops={row['mean_candidate_search_ops_proxy']/1e6:.2f}M "
            f"topk_recall={recall_text}"
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
