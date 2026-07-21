from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from popkv import LlamaSparseDecoder, METHODS, RetrievalConfig, load_hf_model


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PG19 PPL and decode-latency comparison")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--generate-tokens", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--quest-page-size", type=int, default=16)
    p.add_argument("--full-layers", default="0,1")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--pg19-cache-dir", default="data/pg19")
    p.add_argument("--pg19-max-books", type=int, default=64)
    p.add_argument("--text-file", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)


def parse_methods(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(x.strip().lower() for x in value.split(",") if x.strip()))
    unknown = set(result) - set(METHODS)
    if not result or unknown: raise ValueError(f"methods must be selected from {METHODS}; got {unknown}")
    return result


def load_blocks(tokenizer: Any, args: argparse.Namespace) -> list[torch.Tensor]:
    required = args.num_blocks * (args.context_length + args.generate_tokens)
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="ignore")
        stream = [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]
        if len(stream) < required: raise RuntimeError(f"text file has {len(stream)} tokens; need {required}")
    else:
        from huggingface_hub import hf_hub_download
        listing = hf_hub_download(
            repo_id="deepmind/pg19", repo_type="dataset", filename="data/test_files.txt",
        )
        names = [x.strip() for x in Path(listing).read_text().splitlines() if x.strip()]
        cache = Path(args.pg19_cache_dir) / "test"; cache.mkdir(parents=True, exist_ok=True)
        stream, eos = [], tokenizer.eos_token_id
        for number, name in enumerate(sorted(names)):
            if number >= args.pg19_max_books: break
            local = cache / Path(name).name
            if not local.exists():
                urllib.request.urlretrieve("https://storage.googleapis.com/deepmind-gutenberg/" + name, local)
            text = local.read_text(encoding="utf-8", errors="ignore")
            stream.extend(int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"])
            if eos is not None: stream.append(int(eos))
            if len(stream) >= required: break
        if len(stream) < required: raise RuntimeError(f"PG19 produced {len(stream)} tokens; need {required}")
    size = args.context_length + args.generate_tokens
    return [torch.tensor(stream[i * size:(i + 1) * size], dtype=torch.long) for i in range(args.num_blocks)]


def write_results(output: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if rows:
        with (output / "samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with (output / "samples.jsonl").open("w") as handle:
            for row in rows: handle.write(json.dumps(row) + "\n")
    summary = []
    for method in metadata["methods"]:
        selected = [x for x in rows if x["method"] == method]
        if not selected: continue
        mean_nll = sum(x["nll"] for x in selected) / len(selected)
        item = {"method": method, "tokens": len(selected), "perplexity": math.exp(mean_nll), "mean_nll": mean_nll}
        for key in ("prepack_ms", "decode_ms", "search_ms", "score_ms", "topk_ms", "gather_ms", "attention_ms", "update_ms"):
            item[key + "_per_token" if key != "prepack_ms" else key] = sum(x[key] for x in selected) / len(selected)
        summary.append(item)
    if summary:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = arguments(); methods = parse_methods(args.methods)
    config = RetrievalConfig(
        budget=args.budget, group_size=args.group_size, quest_page_size=args.quest_page_size,
        full_layers=tuple(int(x) for x in args.full_layers.split(",") if x.strip()),
    ); config.validate()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or f"outputs/pg19_{stamp}")
    metadata = vars(args) | {"methods": methods, "created_at_utc": stamp, "retrieval": asdict(config)}
    if args.dry_run:
        print(json.dumps(metadata, indent=2)); return
    torch.manual_seed(args.seed)
    model, tokenizer = load_hf_model(
        args.model, dtype=args.dtype, device_map=args.device_map,
        attention=args.attention, trust_remote_code=args.trust_remote_code,
    )
    decoder = LlamaSparseDecoder(model, config); blocks = load_blocks(tokenizer, args); rows = []
    for block_idx, block in enumerate(blocks):
        prefix_ids = block[:args.context_length - 1]
        current_ids = block[args.context_length - 1:args.context_length - 1 + args.generate_tokens]
        targets = block[args.context_length:args.context_length + args.generate_tokens]
        prefix = decoder.build_prefix_cache(prefix_ids); sync(decoder.device)
        for method in methods:
            pack_started = time.perf_counter()
            retrieval = None if method == "full" else decoder.build_retrieval_caches(
                prefix, method=method, reserve_tokens=args.generate_tokens + 1,
            )
            sync(decoder.device); prepack_ms = (time.perf_counter() - pack_started) * 1000
            kv = decoder.build_static_kv_cache(prefix, reserve_tokens=args.generate_tokens + 1)
            decoder.decode(kv, current_ids[0], method=method, retrieval_caches=retrieval); sync(decoder.device)
            for step, (current, target) in enumerate(zip(current_ids, targets)):
                sync(decoder.device); started = time.perf_counter()
                logits, diag, new_k, new_v = decoder.decode(
                    kv, current, method=method, retrieval_caches=retrieval, return_new_kv=True,
                )
                sync(decoder.device); decode_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter(); decoder.append_kv(kv, new_k, new_v)
                if retrieval is not None: decoder.append_retrieval_caches(retrieval, new_k)
                sync(decoder.device); update_ms = (time.perf_counter() - started) * 1000
                rows.append({
                    "block": block_idx, "step": step, "method": method,
                    "nll": float(F.cross_entropy(logits[None], target.to(decoder.device)[None]).item()),
                    "prepack_ms": prepack_ms, "decode_ms": decode_ms,
                    "search_ms": diag.candidate_search_ms, "score_ms": diag.candidate_score_ms,
                    "topk_ms": diag.candidate_topk_ms, "gather_ms": diag.selected_gather_ms,
                    "attention_ms": diag.selected_attention_ms, "update_ms": update_ms,
                    "candidate_ratio": diag.candidate_ratio,
                })
            write_results(output, rows, metadata)
            done = [x for x in rows if x["block"] == block_idx and x["method"] == method]
            nll = sum(x["nll"] for x in done) / len(done)
            print(f"block={block_idx} method={method} ppl={math.exp(nll):.4f} decode_ms={sum(x['decode_ms'] for x in done)/len(done):.1f}", flush=True)
    print(f"results={output}")


if __name__ == "__main__": main()
