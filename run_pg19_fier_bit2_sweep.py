from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from run_prepacked_long_context import (
    build_pg19_blocks,
    build_pg19_partial_blocks,
    continuous_ppl,
    load_model_and_tokenizer,
)
from sparse_ppl import EvaluationVariant, LlamaSparseDecoder, RetrievalConfig, _safe_perplexity, parse_int_tuple



def load_local_env_token(path: Path = Path('.env')) -> None:
    if os.environ.get('HF_TOKEN') or not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, value = stripped.split('=', maxsplit=1)
        if key.strip() == 'HF_TOKEN':
            os.environ['HF_TOKEN'] = value.strip().strip('"').strip("'")
            return

def parse_context_plan(value: str) -> list[tuple[int, int]]:
    plan: list[tuple[int, int]] = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            ctx, blocks = item.split(':', maxsplit=1)
            plan.append((int(ctx), int(blocks)))
        else:
            plan.append((int(item), 1))
    if not plan:
        raise ValueError('context plan cannot be empty')
    for context_length, num_blocks in plan:
        if context_length < 2 or num_blocks <= 0:
            raise ValueError(f'invalid context plan entry: {context_length}:{num_blocks}')
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PG19 PPL budget sweep for FullAttention, Quest, FIER Triton, and 2-bit CUDA POPC.'
    )
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    parser.add_argument(
        '--context-plan',
        default='4096:4,8192:4,16384:2,32768:2',
        help='Comma list of context[:num_blocks]. Default balances confidence and <3h runtime.',
    )
    parser.add_argument('--generate-tokens', type=int, default=16)
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument(
        '--budget-sweep', default=None,
        help='Comma-separated sparse token budgets; FullAttention runs once.',
    )
    parser.add_argument('--group-size', type=int, default=32)
    parser.add_argument('--quest-page-size', type=int, default=16)
    parser.add_argument('--full-layers', default='0,1')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='float16')
    parser.add_argument('--device-map', default='cuda:0')
    parser.add_argument('--attention', choices=('sdpa', 'eager', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--dataset-trust-remote-code', action='store_true')
    parser.add_argument('--pg19-loader', choices=('partial', 'hf'), default='partial')
    parser.add_argument('--dataset-name', default='deepmind/pg19')
    parser.add_argument('--dataset-config', default=None)
    parser.add_argument('--dataset-split', default='test')
    parser.add_argument('--pg19-cache-dir', default='data/pg19_partial')
    parser.add_argument('--pg19-max-books', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def make_variants(
    base: RetrievalConfig, budgets: tuple[int, ...]
) -> list[EvaluationVariant]:
    variants = [EvaluationVariant("full", "full_attention", base)]
    for budget in budgets:
        retrieval = replace(base, budget=budget)
        variants.extend(
            [
                EvaluationVariant(
                    "quest",
                    f"quest_p{retrieval.quest_page_size}_b{budget}",
                    retrieval,
                ),
                EvaluationVariant(
                    "fier",
                    f"fier_triton_g{retrieval.fier_group_size}_b{budget}",
                    replace(retrieval, fier_backend="triton"),
                ),
                EvaluationVariant(
                    "bit2_qk",
                    f"bit2_cuda_popc_g{retrieval.bit2_group_size}_b{budget}",
                    replace(retrieval, bit2_backend="cuda_popc"),
                ),
            ]
        )
    return variants


def build_blocks_for_context(tokenizer: Any, args: argparse.Namespace, context_length: int, num_blocks: int) -> list[torch.Tensor]:
    block_args = SimpleNamespace(**vars(args))
    block_args.context_length = context_length
    block_args.num_blocks = num_blocks
    block_args.benchmark = 'pg19_ppl'
    block_args.text_file = None
    block_args.longbench_subset = 'qasper'
    block_args.max_longbench_samples = 1
    if args.pg19_loader == 'partial':
        return build_pg19_partial_blocks(tokenizer, block_args)
    return build_pg19_blocks(tokenizer, block_args)


def summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(int(row['context_length']), str(row['method'])) for row in samples})
    rows: list[dict[str, Any]] = []
    for context_length, method in keys:
        group = [row for row in samples if int(row['context_length']) == context_length and str(row['method']) == method]
        mean_nll = sum(float(row['nll']) for row in group) / len(group)
        recall_values = [float(row['topk_recall']) for row in group if row.get('topk_recall') is not None]
        rows.append({
            'context_length': context_length,
            'method': method,
            'num_samples': len(group),
            'mean_nll': mean_nll,
            'perplexity': _safe_perplexity(mean_nll),
            'mean_candidate_ratio': sum(float(row['candidate_ratio']) for row in group) / len(group),
            'mean_topk_recall_vs_full': 1.0 if method == 'full_attention' else (None if not recall_values else sum(recall_values) / len(recall_values)),
            'mean_decode_ms_per_token': sum(float(row['decode_ms']) for row in group) / len(group),
            'mean_candidate_search_ms_per_token': sum(float(row['candidate_search_ms']) for row in group) / len(group),
            'mean_selected_attention_ms_per_token': sum(float(row['selected_attention_ms']) for row in group) / len(group),
            'mean_cache_update_ms_per_token': sum(float(row['cache_update_ms']) for row in group) / len(group),
            'mean_prepack_ms': sum(float(row['prepack_ms']) for row in group) / len(group),
            'mean_prefix_ms_shared': sum(float(row['prefix_ms_shared']) for row in group) / len(group),
            'mean_candidate_search_ops_proxy': sum(float(row['candidate_search_ops_proxy']) for row in group) / len(group),
        })
    return rows


def write_outputs(output_dir: Path, samples: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(samples)
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (output_dir / 'samples.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in samples))
    if samples:
        fields = list(dict.fromkeys(key for row in samples for key in row.keys()))
        with (output_dir / 'samples.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(samples)
    if summary:
        with (output_dir / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader(); writer.writerows(summary)


def main() -> None:
    args = parse_args()
    load_local_env_token()
    context_plan = parse_context_plan(args.context_plan)
    full_layers = parse_int_tuple(args.full_layers)
    budgets = (
        (args.budget,)
        if args.budget_sweep is None
        else parse_int_tuple(args.budget_sweep)
    )
    if not budgets or any(budget <= 0 for budget in budgets):
        raise SystemExit("Every sparse budget must be positive")
    hf_auth_available = bool(os.environ.get('HF_TOKEN'))
    if not hf_auth_available:
        try:
            from huggingface_hub import get_token

            hf_auth_available = bool(get_token())
        except Exception:
            hf_auth_available = False
    if not hf_auth_available:
        raise SystemExit(
            'No Hugging Face auth token is visible. Export HF_TOKEN or run huggingface-cli login before the PG19 sweep.'
        )
    if args.generate_tokens <= 0:
        raise SystemExit('--generate-tokens must be positive')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base = RetrievalConfig(
        budget=args.budget,
        quest_page_size=args.quest_page_size,
        fier_group_size=args.group_size,
        bit2_group_size=args.group_size,
        bit2_backend='reference',
        full_layers=full_layers,
        measure_topk_recall=False,
    )
    variants = make_variants(base, budgets)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_dir = Path(args.output_dir or f'outputs/pg19_fier_bit2_sweep_{timestamp}')
    metadata = {
        'created_at_utc': timestamp,
        'benchmark': 'pg19_ppl_context_sweep',
        'model_id': args.model,
        'context_plan': context_plan,
        'generate_tokens': args.generate_tokens,
        'methods': [variant.label for variant in variants],
        'budget_sweep': budgets,
        'group_size': args.group_size,
        'quest_page_size': args.quest_page_size,
        'fier_backend': 'triton',
        'full_layers': full_layers,
        'measure_topk_recall_vs_exact_qk_topk': False,
        'dataset_name': args.dataset_name,
        'dataset_split': args.dataset_split,
        'pg19_loader': args.pg19_loader,
        'pg19_max_books': args.pg19_max_books,
        'dtype': args.dtype,
        'attention': args.attention,
        'device_map': args.device_map,
        'expected_runtime_note': 'Default plan is intended to finish within roughly 3 hours on one RTX A6000, excluding first-time model download.',
    }
    print(json.dumps(metadata, indent=2, default=str), flush=True)
    if args.dry_run:
        print('dry-run ok')
        return

    model, tokenizer = load_model_and_tokenizer(args)
    decoder = LlamaSparseDecoder(model, pca_cache=None, retrieval=base)
    all_samples: list[dict[str, Any]] = []
    write_outputs(output_dir, all_samples, metadata)

    for context_length, num_blocks in context_plan:
        print(f'\n=== context={context_length} num_blocks={num_blocks} generate_tokens={args.generate_tokens} ===', flush=True)
        blocks = build_blocks_for_context(tokenizer, args, context_length, num_blocks)
        result = continuous_ppl(
            decoder,
            blocks,
            variants,
            context_length=context_length,
            generate_tokens=args.generate_tokens,
        )
        for row in result.samples:
            row['context_length'] = context_length
            row['planned_num_blocks'] = num_blocks
            row['model_id'] = args.model
            all_samples.append(row)
        write_outputs(output_dir, all_samples, metadata)
        for row in summarize(all_samples):
            if int(row['context_length']) == context_length:
                recall = row['mean_topk_recall_vs_full']
                recall_text = 'n/a' if recall is None else f'{recall:.4f}'
                print(
                    f"ctx={row['context_length']:5d} {row['method']:34s} "
                    f"samples={row['num_samples']:4d} PPL={row['perplexity']:.4f} "
                    f"recall={recall_text} decode={row['mean_decode_ms_per_token']:.1f}ms "
                    f"search={row['mean_candidate_search_ms_per_token']:.1f}ms "
                    f"attention={row['mean_selected_attention_ms_per_token']:.1f}ms",
                    flush=True,
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'Wrote results to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
