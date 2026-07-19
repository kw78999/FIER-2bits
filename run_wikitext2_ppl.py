from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from sparse_ppl import (
    EvaluationVariant,
    LlamaSparseDecoder,
    PCABasisCache,
    RetrievalConfig,
    build_wikitext2_blocks,
    evaluate_sampled_decode_ppl,
    parse_int_tuple,
    parse_methods,
)


DEFAULT_PCA_BASIS = "pca_basis_meta-llama_Llama-3.1-8B_wikitext-103.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full attention, PQ-SIFT, Loki, Quest, FIER and bit2_qk using "
            "sampled decode-style perplexity on WikiText-2."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--pca-basis", default=DEFAULT_PCA_BASIS)
    parser.add_argument("--methods", default="full,pqsift,loki,quest,fier")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument(
        "--query-positions",
        default="1023,2047,3071,4095",
        help="Current-token positions; each predicts the token at position+1.",
    )
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument(
        "--budget-sweep",
        default=None,
        help=(
            "Comma-separated token budgets for Loki, Quest, FIER and bit2_qk. "
            "Full and PQ-SIFT are evaluated once per own configuration."
        ),
    )
    parser.add_argument("--loki-rank", type=int, default=64)
    parser.add_argument("--pqsift-axes", type=int, default=4)
    parser.add_argument(
        "--pqsift-axes-sweep",
        default=None,
        help=(
            "Comma-separated PQ-SIFT axis counts. When set, evaluate one PQ-SIFT "
            "variant per axis count (for example: 2,3,4,5,6)."
        ),
    )
    parser.add_argument("--pqsift-r", type=float, default=0.75)
    parser.add_argument(
        "--pqsift-r-sweep",
        default=None,
        help=(
            "Comma-separated PQ-SIFT per-axis keep ratios. Combined with "
            "--pqsift-axes-sweep as a Cartesian product."
        ),
    )
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--fier-group-size", type=int, default=32)
    parser.add_argument(
        "--fier-backend", choices=("reference", "triton"), default="triton"
    )
    parser.add_argument("--bit2-group-size", type=int, default=64)
    parser.add_argument(
        "--bit2-backend",
        choices=("reference", "cuda_popc", "cuda_popc_histogram"),
        default="reference",
    )
    parser.add_argument(
        "--group-size-sweep",
        default=None,
        help="Shared comma-separated token group sizes for FIER and bit2_qk.",
    )
    parser.add_argument(
        "--measure-topk-recall",
        action="store_true",
        help="Measure exact-QK Top-K recall for FIER and bit2_qk (diagnostic overhead).",
    )
    parser.add_argument(
        "--full-layers",
        default="0,1",
        help="Layers that remain dense for every sparse method.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/ppl_<UTC timestamp>.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to Hugging Face model/tokenizer loading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and PCA cache without loading model/data.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers/accelerate are required. Install with: "
            "python3 -m pip install -r requirements-ppl.txt"
        ) from exc

    token = os.environ.get("HF_TOKEN") or None
    if token is None:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=token,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
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


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    query_positions = parse_int_tuple(args.query_positions)
    full_layers = parse_int_tuple(args.full_layers)
    pqsift_axes_sweep = (
        () if args.pqsift_axes_sweep is None else parse_int_tuple(args.pqsift_axes_sweep)
    )
    pqsift_r_sweep = (
        ()
        if args.pqsift_r_sweep is None
        else tuple(
            float(item.strip())
            for item in args.pqsift_r_sweep.split(",")
            if item.strip()
        )
    )
    budget_sweep = (
        () if args.budget_sweep is None else parse_int_tuple(args.budget_sweep)
    )
    group_size_sweep = (
        () if args.group_size_sweep is None else parse_int_tuple(args.group_size_sweep)
    )
    if args.context_length < 2:
        raise SystemExit("--context-length must be at least 2")
    if not query_positions:
        raise SystemExit("--query-positions cannot be empty")
    if max(query_positions) >= args.context_length:
        raise SystemExit("Every query position must be smaller than --context-length")
    if args.pqsift_axes_sweep is not None and not pqsift_axes_sweep:
        raise SystemExit("--pqsift-axes-sweep cannot be empty")
    if any(axes <= 0 for axes in pqsift_axes_sweep):
        raise SystemExit("Every --pqsift-axes-sweep value must be positive")
    if args.pqsift_r_sweep is not None and not pqsift_r_sweep:
        raise SystemExit("--pqsift-r-sweep cannot be empty")
    if any(not 0.0 < ratio <= 1.0 for ratio in pqsift_r_sweep):
        raise SystemExit("Every --pqsift-r-sweep value must be in (0, 1]")
    if args.budget_sweep is not None and not budget_sweep:
        raise SystemExit("--budget-sweep cannot be empty")
    if any(budget <= 0 for budget in budget_sweep):
        raise SystemExit("Every --budget-sweep value must be positive")
    if args.group_size_sweep is not None and not group_size_sweep:
        raise SystemExit("--group-size-sweep cannot be empty")
    if any(group_size <= 0 for group_size in group_size_sweep):
        raise SystemExit("Every --group-size-sweep value must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    retrieval = RetrievalConfig(
        budget=args.budget,
        loki_rank=args.loki_rank,
        pqsift_axes=args.pqsift_axes,
        pqsift_keep_ratio=args.pqsift_r,
        quest_page_size=args.quest_page_size,
        fier_group_size=args.fier_group_size,
        fier_backend=args.fier_backend,
        bit2_group_size=args.bit2_group_size,
        bit2_backend=args.bit2_backend,
        full_layers=full_layers,
        measure_topk_recall=args.measure_topk_recall,
    )
    retrieval.validate()

    needs_pca = any(method in {"pqsift", "loki"} for method in methods)
    required_pca_axes = max(
        args.loki_rank,
        args.pqsift_axes,
        max(pqsift_axes_sweep, default=0),
    )
    pca_cache = None
    if needs_pca:
        pca_cache = PCABasisCache.load(
            args.pca_basis,
            expected_model_id=args.model,
            min_axes=required_pca_axes,
        )
        print(
            "PCA cache:",
            pca_cache.path,
            f"pairs={len(pca_cache.bases)}",
            f"calibration={pca_cache.metadata.get('pca_calib_dataset_config')}",
            f"blocks={pca_cache.metadata.get('actual_calibration_blocks')}",
        )

    print(
        json.dumps(
            {
                "model": args.model,
                "methods": methods,
                "context_length": args.context_length,
                "num_blocks": args.num_blocks,
                "query_positions": query_positions,
                "retrieval": retrieval.__dict__,
                "pqsift_axes_sweep": pqsift_axes_sweep or None,
                "pqsift_r_sweep": pqsift_r_sweep or None,
                "budget_sweep": budget_sweep or None,
                "group_size_sweep": group_size_sweep or None,
                "attention": args.attention,
            },
            indent=2,
            default=str,
        )
    )
    if args.dry_run:
        print("dry-run ok")
        return

    model, tokenizer = load_model_and_tokenizer(args)
    pca_cache = (
        None
        if not needs_pca
        else PCABasisCache.load(
            args.pca_basis,
            expected_model_id=args.model,
            min_axes=required_pca_axes,
            expected_layers=len(model.model.layers),
            expected_heads=int(model.config.num_attention_heads),
        )
    )
    blocks = build_wikitext2_blocks(
        tokenizer,
        context_length=args.context_length,
        num_blocks=args.num_blocks,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
    )
    decoder = LlamaSparseDecoder(
        model,
        pca_cache=pca_cache,
        retrieval=retrieval,
    )
    axes_values = pqsift_axes_sweep or (args.pqsift_axes,)
    ratio_values = pqsift_r_sweep or (args.pqsift_r,)
    budget_values = budget_sweep or (args.budget,)
    variants = []
    for method in methods:
        if method == "full":
            variants.append(
                EvaluationVariant(
                    method=method,
                    label=method,
                    retrieval=retrieval,
                )
            )
            continue
        if method in {"loki", "quest"}:
            for budget in budget_values:
                variant_retrieval = replace(retrieval, budget=budget)
                label = method if not budget_sweep else f"{method}_b{budget}"
                variants.append(
                    EvaluationVariant(
                        method=method,
                        label=label,
                        retrieval=variant_retrieval,
                    )
                )
            continue
        if method in {"fier", "bit2_qk"}:
            default_group = (
                retrieval.fier_group_size
                if method == "fier"
                else retrieval.bit2_group_size
            )
            for group_size in group_size_sweep or (default_group,):
                for budget in budget_values:
                    variant_retrieval = replace(
                        retrieval,
                        budget=budget,
                        **(
                            {"fier_group_size": group_size}
                            if method == "fier"
                            else {"bit2_group_size": group_size}
                        ),
                    )
                    if group_size_sweep:
                        label = f"{method}_g{group_size}_b{budget}"
                    else:
                        label = method if not budget_sweep else f"{method}_b{budget}"
                    variants.append(
                        EvaluationVariant(method=method, label=label, retrieval=variant_retrieval)
                    )
            continue
        for axes in axes_values:
            for ratio in ratio_values:
                variant_retrieval = replace(
                    retrieval,
                    pqsift_axes=axes,
                    pqsift_keep_ratio=ratio,
                )
                variants.append(
                    EvaluationVariant(
                        method="pqsift",
                        label=f"pqsift_A{axes}_r{ratio:g}",
                        retrieval=variant_retrieval,
                    )
                )
    result = evaluate_sampled_decode_ppl(
        decoder,
        blocks,
        methods=methods,
        query_positions=query_positions,
        method_variants=variants,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"outputs/ppl_{timestamp}")
    metadata = {
        "created_at_utc": timestamp,
        "evaluation_type": "sampled_decode_ppl",
        "model_id": args.model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "context_length": args.context_length,
        "num_blocks": args.num_blocks,
        "query_positions": query_positions,
        "methods": methods,
        "retrieval": retrieval.__dict__,
        "pqsift_axes_sweep": list(pqsift_axes_sweep) or None,
        "pqsift_r_sweep": list(pqsift_r_sweep) or None,
        "budget_sweep": list(budget_sweep) or None,
        "group_size_sweep": list(group_size_sweep) or None,
        "evaluated_variants": [variant.label for variant in variants],
        "pca_basis": None if pca_cache is None else str(pca_cache.path),
        "pca_metadata": None if pca_cache is None else pca_cache.metadata,
        "bit2_range_mode": "query_minmax_half__keys_group_channel_minmax_half",
        "bit2_score_mode": "sign_xnor_weight_1_plus_mag_or_plus_mag_and",
        "fier_backend": args.fier_backend,
        "bit2_backend": args.bit2_backend,
        "bit2_storage": "two_uint8_packed_bitplanes_2_bits_per_scalar",
        "bit2_popcount": "uint8_lookup_table_reference",
        "dtype": args.dtype,
        "attention_implementation_for_dense_prefix": args.attention,
        "component_timing": {
            "candidate_search_ms": "select_candidates across all layers and heads",
            "candidate_search_ops_proxy": "rough threshold/quantize/bitops-or-FIER-score/topk operation proxy",
            "selected_attention_ms": "exact_attention across all layers and heads",
            "cuda_timer": "torch.cuda.Event",
        },
        "torch_version": torch.__version__,
    }
    result.write(output_dir, metadata)
    print("\nSummary")
    for row in result.summary():
        recall = row["mean_topk_recall"]
        recall_text = "n/a" if recall is None else f"{recall:.4f}"
        print(
            f"{row['method']:8s} samples={row['num_samples']:3d} "
            f"NLL={row['mean_nll']:.6f} PPL={row['perplexity']:.4f} "
            f"candidate_ratio={row['mean_candidate_ratio']:.4f} "
            f"search_ms={row['mean_candidate_search_ms']:.1f} "
            f"search_ops={row['mean_candidate_search_ops_proxy'] / 1e6:.2f}M "
            f"attention_ms={row['mean_selected_attention_ms']:.1f} "
            f"decode_ms={row['mean_decode_ms']:.1f} "
            f"total_decode_s={row['total_decode_ms'] / 1000.0:.2f} "
            f"topk_recall={recall_text}"
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
