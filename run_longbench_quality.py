from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import string
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from run_prepacked_long_context import load_model_and_tokenizer
from sparse_ppl import LlamaSparseDecoder, RetrievalConfig


PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
}

MAX_GEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "gov_report": 512,
    "triviaqa": 32,
}
QA_DATASETS = frozenset(PROMPTS) - {"gov_report"}
NO_CHAT_DATASETS = frozenset({"triviaqa"})


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    source_index: int
    input_ids: list[int]
    input_sha256: str
    prompt_tokens_before_truncation: int
    prompt_tokens: int
    truncated: bool
    answers: list[str]
    max_new_tokens: int


@dataclass(frozen=True)
class MethodSpec:
    method: str
    label: str
    budget: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-bounded FIER LongBench quality reproduction")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--methods", default="full,quest,fier,2mean")
    p.add_argument("--datasets", default=",".join(PROMPTS))
    p.add_argument("--budgets", default="512,1024,2048,4096")
    p.add_argument("--samples-per-dataset", type=int, default=40)
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--quest-page-size", type=int, default=16)
    p.add_argument("--full-layers", default="0,1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-dir", default="results/longbench")
    p.add_argument("--cache-dir", default="data/longbench_v1")
    p.add_argument("--hard-limit-minutes", type=float, default=465.0)
    p.add_argument("--stop-reserve-minutes", type=float, default=10.0)
    p.add_argument("--max-gen-scale", type=float, default=1.0, help="Smoke-only generation-length multiplier")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def comma_tuple(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))


def int_tuple(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(x.strip()) for x in value.split(",") if x.strip()))


def token_hash(ids: Iterable[int]) -> str:
    return hashlib.sha256(b"".join(int(x).to_bytes(4, "little", signed=False) for x in ids)).hexdigest()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(truth).split()
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, truth: str) -> float:
    pred, gold = prediction.split(), truth.split()
    if not pred or not gold:
        return 0.0
    lcs = _lcs_length(pred, gold)
    precision, recall = lcs / len(pred), lcs / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score_prediction(dataset: str, prediction: str, answers: Iterable[str]) -> float:
    if dataset == "triviaqa":
        prediction = prediction.lstrip("\n").split("\n", 1)[0]
    metric = rouge_l if dataset == "gov_report" else qa_f1
    return max((metric(prediction, str(answer)) for answer in answers), default=0.0)


def load_token() -> None:
    if os.environ.get("HF_TOKEN") or not Path(".env").exists():
        return
    for line in Path(".env").read_text().splitlines():
        if line.strip().startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")
            return


def dataset_path(dataset: str, cache_dir: Path) -> Path:
    target = cache_dir / f"{dataset}.jsonl"
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download
    archive = Path(hf_hub_download("THUDM/LongBench", repo_type="dataset", filename="data.zip"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        matches = [name for name in bundle.namelist() if name == f"{dataset}.jsonl" or name.endswith(f"/{dataset}.jsonl")]
        if not matches:
            raise FileNotFoundError(f"{dataset}.jsonl not in {archive}")
        with bundle.open(matches[0]) as source, target.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
    return target


def official_input_ids(tokenizer: Any, dataset: str, row: dict[str, Any], max_context: int) -> tuple[list[int], int, bool]:
    prompt = PROMPTS[dataset].format(**row)
    raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    original = int(raw_ids.numel())
    truncated = original > max_context
    if truncated:
        half = max_context // 2
        raw_ids = torch.cat([raw_ids[:half], raw_ids[-(max_context - half):]])
        prompt = tokenizer.decode(raw_ids, skip_special_tokens=True)
    if dataset not in NO_CHAT_DATASETS and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    ids = [int(x) for x in tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]]
    return ids, original, truncated


def load_samples(tokenizer: Any, args: argparse.Namespace, datasets: tuple[str, ...]) -> list[Sample]:
    selected: list[Sample] = []
    cache_dir = Path(args.cache_dir)
    for dataset in datasets:
        rows = [json.loads(line) for line in dataset_path(dataset, cache_dir).read_text().splitlines() if line.strip()]
        count = 0
        for source_index, row in enumerate(rows):
            answers = [str(x) for x in row.get("answers", []) if str(x).strip()]
            if not answers:
                continue
            ids, original, truncated = official_input_ids(tokenizer, dataset, row, args.max_context)
            if len(ids) < 2:
                continue
            max_new = max(1, int(math.ceil(MAX_GEN[dataset] * args.max_gen_scale)))
            selected.append(Sample(
                dataset=dataset,
                sample_id=str(row.get("_id") or f"{dataset}:{source_index}"),
                source_index=source_index,
                input_ids=ids,
                input_sha256=token_hash(ids),
                prompt_tokens_before_truncation=original,
                prompt_tokens=len(ids),
                truncated=truncated,
                answers=answers,
                max_new_tokens=max_new,
            ))
            count += 1
            if count >= args.samples_per_dataset:
                break
        if count < args.samples_per_dataset:
            raise RuntimeError(f"Only {count} valid rows in {dataset}; requested {args.samples_per_dataset}")
        print(f"selected dataset={dataset} samples={count} token_range="
              f"{min(x.prompt_tokens for x in selected if x.dataset == dataset)}.."
              f"{max(x.prompt_tokens for x in selected if x.dataset == dataset)}", flush=True)
    return selected


def method_specs(methods: tuple[str, ...], budgets: tuple[int, ...], group: int, page: int) -> list[MethodSpec]:
    result: list[MethodSpec] = []
    for method in methods:
        if method == "full":
            result.append(MethodSpec("full", "full", None))
        else:
            internal = "bit2_qk" if method in {"2mean", "popkv", "bit2_qk"} else method
            for budget in budgets:
                label = f"2mean_qk_g{group}_b{budget}" if internal == "bit2_qk" else (
                    f"quest_p{page}_b{budget}" if internal == "quest" else f"fier_g{group}_b{budget}"
                )
                result.append(MethodSpec(internal, label, budget))
    return result


def retrieval_for(args: argparse.Namespace, spec: MethodSpec) -> RetrievalConfig:
    return RetrievalConfig(
        budget=spec.budget or max(int_tuple(args.budgets)),
        quest_page_size=args.quest_page_size,
        fier_group_size=args.group_size,
        fier_backend="triton",
        bit2_group_size=args.group_size,
        bit2_backend="qk_2mean",
        full_layers=int_tuple(args.full_layers),
        measure_topk_recall=False,
        measure_score_diagnostics=False,
    )


def eos_ids(model: Any, tokenizer: Any) -> set[int]:
    result: set[int] = set()
    for value in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if value is None:
            continue
        result.update(int(x) for x in value) if isinstance(value, (list, tuple, set)) else result.add(int(value))
    return result


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_one(decoder: LlamaSparseDecoder, tokenizer: Any, sample: Sample, spec: MethodSpec,
                 base_past: Any, prefill_seconds: float) -> dict[str, Any]:
    if token_hash(sample.input_ids) != sample.input_sha256:
        raise RuntimeError("prompt token hash changed")
    decoder.retrieval = retrieval = replace(decoder.retrieval, budget=spec.budget or decoder.retrieval.budget)
    packed = None
    prepack_started = time.perf_counter()
    if spec.method in {"fier", "bit2_qk"}:
        group = retrieval.fier_group_size if spec.method == "fier" else retrieval.bit2_group_size
        packed = decoder.build_packed_retrieval_caches(base_past, method=spec.method, group_size=group)
    sync(decoder.device)
    prepack_seconds = time.perf_counter() - prepack_started
    past = decoder.build_static_kv_cache(base_past, reserve_tokens=sample.max_new_tokens + 1)
    current_id = int(sample.input_ids[-1])
    decoder.decode(past, current_id, method=spec.method, retrieval_caches=packed, return_new_kv=False)
    sync(decoder.device)
    if decoder.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(decoder.device)
    totals: dict[str, float] = defaultdict(float)
    generated: list[int] = []
    stops = eos_ids(decoder.model, tokenizer)
    stopped_on_eos = False
    generation_started = time.perf_counter()
    for _ in range(sample.max_new_tokens):
        sync(decoder.device)
        step_started = time.perf_counter()
        logits, diag, new_keys, new_values = decoder.decode(
            past, current_id, method=spec.method, retrieval_caches=packed, return_new_kv=True
        )
        sync(decoder.device)
        totals["decode_ms"] += (time.perf_counter() - step_started) * 1000
        update_started = time.perf_counter()
        decoder.append_to_past_key_values(past, new_keys, new_values)
        if packed is not None:
            decoder.append_to_packed_retrieval_caches(packed, new_keys)
        sync(decoder.device)
        totals["update_ms"] += (time.perf_counter() - update_started) * 1000
        totals["search_ms"] += diag.candidate_search_ms
        totals["score_ms"] += diag.candidate_score_ms
        totals["topk_ms"] += diag.candidate_topk_ms
        totals["gather_ms"] += diag.selected_gather_ms
        totals["attention_ms"] += diag.selected_attention_ms
        totals["candidate_ratio"] += diag.candidate_ratio
        next_id = int(torch.argmax(logits).item())
        if next_id in stops:
            stopped_on_eos = True
            break
        generated.append(next_id)
        current_id = next_id
    sync(decoder.device)
    generation_seconds = time.perf_counter() - generation_started
    steps = max(1, len(generated) + int(stopped_on_eos))
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
    peak = torch.cuda.max_memory_allocated(decoder.device) if decoder.device.type == "cuda" else 0
    row = {
        "prediction": prediction,
        "output_ids": generated,
        "score": score_prediction(sample.dataset, prediction, sample.answers),
        "generated_tokens": len(generated),
        "decode_steps": steps,
        "stopped_on_eos": stopped_on_eos,
        "prefill_seconds_shared": prefill_seconds,
        "prepack_seconds": prepack_seconds,
        "generation_seconds": generation_seconds,
        "wall_decode_ms_per_token": generation_seconds * 1000 / steps,
        "peak_gpu_memory_bytes": peak,
    }
    for name in ("decode_ms", "update_ms", "search_ms", "score_ms", "topk_ms", "gather_ms", "attention_ms", "candidate_ratio"):
        row[f"{name}_per_token"] = totals[name] / steps
    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["dataset"]), str(row["sample_id"]), str(row["method_label"])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output_dir: Path, model_id: str) -> None:
    rows = [r for r in read_jsonl(output_dir / "raw_results.jsonl") if r.get("status") == "ok"]
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"], row.get("budget"))].append(row)
    full_scores = {dataset: 100 * sum(float(r["score"]) for r in group) / len(group)
                   for (method, dataset, _), group in groups.items() if method == "full"}
    raw_scores, runtime = [], []
    for (method, dataset, budget), group in sorted(groups.items(), key=lambda x: str(x[0])):
        score = 100 * sum(float(r["score"]) for r in group) / len(group)
        full = full_scores.get(dataset)
        failed = sum(1 for r in read_jsonl(output_dir / "errors.jsonl")
                     if r.get("method") == method and r.get("dataset") == dataset and r.get("budget") == budget)
        raw_scores.append({
            "model": model_id, "method": method, "dataset": dataset, "budget": budget,
            "metric": "ROUGE-L" if dataset == "gov_report" else "F1", "score": score,
            "full_score": full, "retention_percent": (score / full * 100 if full else None),
            "num_samples": len(group) + failed, "num_success": len(group), "num_failed": failed,
            "avg_input_tokens": sum(r["prompt_tokens"] for r in group) / len(group),
            "avg_output_tokens": sum(r["generated_tokens"] for r in group) / len(group),
        })
        runtime.append({
            "method": method, "budget": budget, "dataset": dataset, "num_success": len(group),
            "total_generation_seconds": sum(r["generation_seconds"] for r in group),
            "avg_wall_decode_ms_per_token": sum(r["wall_decode_ms_per_token"] for r in group) / len(group),
            "avg_search_ms_per_token": sum(r["search_ms_per_token"] for r in group) / len(group),
            "avg_score_ms_per_token": sum(r["score_ms_per_token"] for r in group) / len(group),
            "avg_topk_ms_per_token": sum(r["topk_ms_per_token"] for r in group) / len(group),
            "avg_gather_ms_per_token": sum(r["gather_ms_per_token"] for r in group) / len(group),
            "avg_attention_ms_per_token": sum(r["attention_ms_per_token"] for r in group) / len(group),
            "peak_gpu_memory_bytes": max(r["peak_gpu_memory_bytes"] for r in group), "failed_samples": failed,
        })
    score_cols = ["model", "method", "dataset", "budget", "metric", "score", "full_score", "retention_percent",
                  "num_samples", "num_success", "num_failed", "avg_input_tokens", "avg_output_tokens"]
    runtime_cols = list(runtime[0]) if runtime else ["method", "budget", "dataset"]
    write_csv(output_dir / "scores_raw.csv", raw_scores, score_cols)
    write_csv(output_dir / "runtime_summary.csv", runtime, runtime_cols)
    summaries = []
    by_config: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_scores:
        by_config[(row["method"], row["budget"])].append(row)
    for (method, budget), group in sorted(by_config.items(), key=lambda x: str(x[0])):
        retentions = [r["retention_percent"] for r in group if r["retention_percent"] is not None]
        summaries.append({
            "method": method, "budget": budget, "task_average": sum(r["score"] for r in group) / len(group),
            "qa_average": sum(r["score"] for r in group if r["dataset"] in QA_DATASETS) /
                          max(1, sum(r["dataset"] in QA_DATASETS for r in group)),
            "gov_report_rouge_l": next((r["score"] for r in group if r["dataset"] == "gov_report"), None),
            "mean_retention_percent": sum(retentions) / len(retentions) if retentions else None,
            "minimum_task_retention_percent": min(retentions) if retentions else None,
            "total_success": sum(r["num_success"] for r in group),
        })
    summary_cols = list(summaries[0]) if summaries else ["method", "budget"]
    write_csv(output_dir / "scores_summary.csv", summaries, summary_cols)
    write_csv(output_dir / "retention_vs_full.csv",
              [r for r in raw_scores if r["method"] != "full"], score_cols)
    (output_dir / "summary.json").write_text(json.dumps({"scores_raw": raw_scores, "scores_summary": summaries,
                                                          "runtime_summary": runtime}, indent=2, ensure_ascii=False) + "\n")
    write_readme(output_dir, raw_scores, summaries, runtime)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x)) for x in row) + " |")
    return "\n".join(out)


def write_readme(output_dir: Path, scores: list[dict[str, Any]], summaries: list[dict[str, Any]], runtime: list[dict[str, Any]]) -> None:
    task_order = list(PROMPTS)
    task_rows = []
    for summary in summaries:
        group = [r for r in scores if r["method"] == summary["method"] and r["budget"] == summary["budget"]]
        mapping = {r["dataset"]: r["score"] for r in group}
        task_rows.append([summary["method"], summary["budget"], *[mapping.get(x) for x in task_order], summary["task_average"]])
    retention_rows = [[r["method"], r["budget"], r["task_average"], r["mean_retention_percent"],
                       r["minimum_task_retention_percent"]] for r in summaries]
    runtime_rows = [[r["method"], r["budget"], r["dataset"], r["avg_wall_decode_ms_per_token"],
                     r["avg_search_ms_per_token"], r["avg_attention_ms_per_token"], r["failed_samples"]] for r in runtime]
    text = "# LongBench FIER/2mean reproduction\n\n"
    text += "Scores are percentages. Only actually completed samples are included. Original Llama-3 has an 8K configured context window; the paper-requested 32K setting extrapolates RoPE beyond that window.\n\n"
    text += "## Task scores\n\n" + markdown_table(["Method", "Budget", *task_order, "Average"], task_rows) + "\n\n"
    text += "## Retention vs Full\n\n" + markdown_table(["Method", "Budget", "Average", "Retention %", "Min task %"], retention_rows) + "\n\n"
    text += "## Runtime\n\n" + markdown_table(["Method", "Budget", "Dataset", "Decode ms/tok", "Search", "Attention", "Failed"], runtime_rows) + "\n"
    (output_dir / "README.md").write_text(text)


def environment_text() -> str:
    commands = (["git", "rev-parse", "HEAD"], ["git", "status", "--short"],
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    lines = [f"python={sys.version}", f"platform={platform.platform()}", f"torch={torch.__version__}"]
    try:
        import transformers
        lines.append(f"transformers={transformers.__version__}")
    except Exception as exc:
        lines.append(f"transformers_error={exc!r}")
    for command in commands:
        try:
            lines.append("$ " + " ".join(command) + "\n" + subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip())
        except Exception as exc:
            lines.append("$ " + " ".join(command) + f"\nERROR {exc!r}")
    return "\n".join(lines) + "\n"


def validate_short(decoder: LlamaSparseDecoder, tokenizer: Any, sample: Sample, args: argparse.Namespace) -> None:
    ids = sample.input_ids[-512:]
    short = replace(sample, input_ids=ids, input_sha256=token_hash(ids), prompt_tokens=len(ids), max_new_tokens=8)
    tensor = torch.tensor(ids, device=decoder.device).view(1, -1)
    hf = decoder.model.generate(tensor, max_new_tokens=8, do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id)
    hf_ids = [int(x) for x in hf[0, len(ids):]]
    base = decoder.build_prefix_cache(torch.tensor(ids[:-1]))
    decoder.retrieval = retrieval_for(args, MethodSpec("full", "full", None))
    manual = generate_one(decoder, tokenizer, short, MethodSpec("full", "full", None), base, 0.0)["output_ids"]
    if manual != hf_ids[:len(manual)]:
        raise AssertionError(f"HF/manual Full mismatch: hf={hf_ids} manual={manual}")
    for method in ("quest", "fier", "bit2_qk"):
        spec = MethodSpec(method, f"validate_{method}", len(ids) + 8)
        decoder.retrieval = retrieval_for(args, spec)
        result = generate_one(decoder, tokenizer, short, spec, base, 0.0)
        if not all(math.isfinite(float(result[x])) for x in ("score", "wall_decode_ms_per_token")):
            raise AssertionError(f"non-finite validation result for {method}")
    print("validation_ok full_matches_hf=true sparse_budget_ge_context_finite=true", flush=True)


def main() -> None:
    process_started = time.monotonic()
    args = parse_args()
    if args.hard_limit_minutes <= 0 or args.hard_limit_minutes > 480:
        raise SystemExit("--hard-limit-minutes must be in (0, 480]")
    if args.stop_reserve_minutes < 1 or args.stop_reserve_minutes >= args.hard_limit_minutes:
        raise SystemExit("invalid --stop-reserve-minutes")
    methods, datasets, budgets = comma_tuple(args.methods), comma_tuple(args.datasets), int_tuple(args.budgets)
    unknown_methods = set(methods) - {"full", "quest", "fier", "2mean", "popkv", "bit2_qk"}
    unknown_datasets = set(datasets) - set(PROMPTS)
    if unknown_methods or unknown_datasets or not budgets:
        raise SystemExit(f"invalid methods={unknown_methods} datasets={unknown_datasets} budgets={budgets}")
    specs = method_specs(methods, budgets, args.group_size, args.quest_page_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    config = vars(args) | {
        "resolved_methods": [asdict(x) for x in specs], "resolved_datasets": datasets,
        "paper_model_context_mismatch": "Meta-Llama-3-8B-Instruct config is 8192; paper states max input 32768",
        "gqa_policy": "per query-head Top-k; query heads sharing a KV head may select different token positions; exact FP K/V gathered by head_to_kv",
        "fier": "sequence-grouped g32 1-bit RTN, token Top-k, current token reserves one slot",
        "2mean": "Q/K sign+magnitude two-bit encoding with max/2,min/2 thresholds; qk_2mean fused CUDA scoring; per-query-head Top-k; current token reserves one slot",
        "sink_recent_forced": False,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "environment.txt").write_text(environment_text())
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    load_token()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    base_retrieval = retrieval_for(args, specs[0])
    decoder = LlamaSparseDecoder(model, pca_cache=None, retrieval=base_retrieval)
    samples = load_samples(tokenizer, args, datasets)
    manifest = [{k: v for k, v in asdict(s).items() if k != "input_ids"} for s in samples]
    (output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if args.validate:
        validate_short(decoder, tokenizer, samples[0], args)
    raw_path = output_dir / "raw_results.jsonl"
    if raw_path.exists() and not args.resume:
        raise SystemExit(f"{raw_path} exists and --no-resume was used")
    completed = {result_key(r) for r in read_jsonl(raw_path) if r.get("status") == "ok"}
    durations: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=12))
    deadline = process_started + args.hard_limit_minutes * 60
    soft_deadline = deadline - args.stop_reserve_minutes * 60
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"signal_received={signum}; stopping after current generation", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    total_target = len(samples) * len(specs)
    for sample_no in range(args.samples_per_dataset):
        for dataset in datasets:
            sample = next(s for s in samples if s.dataset == dataset and sum(
                1 for x in samples if x.dataset == dataset and x.source_index <= s.source_index
            ) == sample_no + 1)
            pending = [spec for spec in specs if (sample.dataset, sample.sample_id, spec.label) not in completed]
            if not pending:
                continue
            remaining = soft_deadline - time.monotonic()
            defaults = {"full": .34, "quest": .82, "fier": .52, "bit2_qk": .50}
            estimate = 20.0 + sum((sum(durations[x.label]) / len(durations[x.label]) if durations[x.label]
                                   else 3 + defaults[x.method] * sample.max_new_tokens) for x in pending)
            if stop_requested or remaining <= estimate:
                print(f"deadline_stop remaining={remaining:.1f}s next_sample_estimate={estimate:.1f}s", flush=True)
                aggregate(output_dir, args.model)
                return
            sync(decoder.device)
            prefill_started = time.perf_counter()
            try:
                base_past = decoder.build_prefix_cache(torch.tensor(sample.input_ids[:-1], dtype=torch.long))
                sync(decoder.device)
                prefill_seconds = time.perf_counter() - prefill_started
            except Exception as exc:
                error = {"status": "error", "dataset": sample.dataset, "sample_id": sample.sample_id,
                         "method": "prefill", "budget": None, "error": repr(exc), "time_utc": datetime.now(timezone.utc).isoformat()}
                append_jsonl(output_dir / "errors.jsonl", error)
                print(f"prefill_error dataset={sample.dataset} sample={sample_no} error={exc!r}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            for spec in pending:
                decoder.retrieval = retrieval_for(args, spec)
                common = {
                    "status": "ok", "model": args.model, "dataset": sample.dataset,
                    "sample_id": sample.sample_id, "source_index": sample.source_index,
                    "method": "2mean" if spec.method == "bit2_qk" else spec.method,
                    "method_internal": spec.method, "method_label": spec.label, "budget": spec.budget,
                    "prompt_tokens": sample.prompt_tokens, "prompt_tokens_before_truncation": sample.prompt_tokens_before_truncation,
                    "truncated": sample.truncated, "input_sha256": sample.input_sha256,
                    "max_new_tokens": sample.max_new_tokens,
                }
                started = time.monotonic()
                try:
                    result = generate_one(decoder, tokenizer, sample, spec, base_past, prefill_seconds)
                    row = common | result | {"elapsed_seconds": time.monotonic() - process_started}
                    append_jsonl(raw_path, row)
                    pred_name = f"{args.model.split('/')[-1]}_{common['method']}_{sample.dataset}_budget{spec.budget if spec.budget is not None else 'full'}.jsonl"
                    append_jsonl(output_dir / "predictions" / pred_name,
                                 {"sample_id": sample.sample_id, "prediction": result["prediction"],
                                  "answers": sample.answers, "score": result["score"], "input_sha256": sample.input_sha256})
                    completed.add(result_key(row))
                    durations[spec.label].append(time.monotonic() - started)
                except Exception as exc:
                    error = common | {"status": "error", "error": repr(exc), "elapsed_seconds": time.monotonic() - process_started}
                    append_jsonl(output_dir / "errors.jsonl", error)
                    print(f"generation_error dataset={sample.dataset} method={spec.label} sample={sample_no} error={exc!r}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                done = len(completed)
                recent = durations[spec.label][-1] if durations[spec.label] else 0
                print(f"progress={done}/{total_target} dataset={sample.dataset} sample={sample_no+1}/{args.samples_per_dataset} "
                      f"method={spec.label} duration={recent:.1f}s elapsed={(time.monotonic()-process_started)/60:.1f}m "
                      f"remaining={(deadline-time.monotonic())/60:.1f}m", flush=True)
                if stop_requested:
                    break
            del base_past
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            aggregate(output_dir, args.model)
            if stop_requested:
                return
    aggregate(output_dir, args.model)
    print(f"finished completed={len(completed)}/{total_target} elapsed={(time.monotonic()-process_started)/60:.2f}m", flush=True)


if __name__ == "__main__":
    main()
