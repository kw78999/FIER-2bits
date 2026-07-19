from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
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


# LongBench v1 official prompts, limited to this pilot's three QA datasets.
PROMPTS = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", '
        'or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n'
        "Answer the question based on the above article as concisely as you can, using "
        "a single phrase or sentence if possible. If the question cannot be answered "
        'based on the information in the article, write "unanswerable". If the question '
        'is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide '
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the "
        "following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do "
        "not output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
}

DATASET_CONFIGS = {
    "qasper": {"base": 3, "maximum": 5, "max_new_tokens": 64},
    "multifieldqa_en": {"base": 3, "maximum": 5, "max_new_tokens": 64},
    "triviaqa": {"base": 6, "maximum": 10, "max_new_tokens": 32},
}


def load_local_env_token(path: Path = Path(".env")) -> None:
    if os.environ.get("HF_TOKEN") or not path.is_file():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        if key.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip(chr(34)).strip(chr(39))
            return


@dataclass(frozen=True)
class PilotSample:
    dataset: str
    sample_id: str
    source_index: int
    input_ids: list[int]
    input_sha256: str
    prompt_tokens_before_truncation: int
    prompt_tokens: int
    truncated: bool
    answers: list[str]
    all_classes: list[str]
    max_new_tokens: int
    tier: str


@dataclass(frozen=True)
class MethodSpec:
    method: str
    label: str
    budget: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-bounded LongBench sparse-attention pilot")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--time-limit-minutes", type=float, default=50.0)
    parser.add_argument("--expansion-cutoff-minutes", type=float, default=45.0)
    parser.add_argument("--max-context-length", type=int, default=8192)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--optional-budget", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--full-layers", default="0,1")
    parser.add_argument("--min-prompt-tokens", type=int, default=2048)
    parser.add_argument("--dataset-scan-limit", type=int, default=200)
    parser.add_argument("--longbench-cache-dir", default="data/longbench_v1")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="float16")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attention", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-trust-remote-code", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-quest", action="store_true")
    parser.add_argument("--disable-expansion", action="store_true")
    parser.add_argument("--disable-optional-budget", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def qa_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def longbench_score(dataset: str, prediction: str, answers: Iterable[str]) -> float:
    if dataset == "triviaqa":
        prediction = prediction.lstrip("\n").split("\n", maxsplit=1)[0]
    return max((qa_f1(prediction, str(answer)) for answer in answers), default=0.0)


def _chat_wrap(tokenizer: Any, dataset: str, prompt: str) -> str:
    # LongBench's official runner does not add a chat wrapper to TriviaQA.
    if dataset == "triviaqa" or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_prompt(
    tokenizer: Any,
    dataset: str,
    row: dict[str, Any],
    *,
    max_context_length: int,
) -> tuple[list[int], int, bool]:
    prompt = PROMPTS[dataset].format(**row)
    wrapped = _chat_wrap(tokenizer, dataset, prompt)
    ids = [int(token) for token in tokenizer(wrapped, add_special_tokens=True)["input_ids"]]
    original = len(ids)
    if original <= max_context_length:
        return ids, original, False
    # Official LongBench truncates in the middle. The head retains task instruction and
    # document opening; the tail retains the document ending, question, and answer cue.
    head = max_context_length // 2
    tail = max_context_length - head
    return ids[:head] + ids[-tail:], original, True


def select_samples(tokenizer: Any, args: argparse.Namespace) -> list[PilotSample]:
    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(repo_id="THUDM/LongBench", repo_type="dataset", filename="data.zip"))
    cache_dir = Path(args.longbench_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    def dataset_rows(dataset: str) -> list[dict[str, Any]]:
        target = cache_dir / f"{dataset}.jsonl"
        if not target.exists():
            with zipfile.ZipFile(archive) as bundle:
                matches = [name for name in bundle.namelist() if name.rstrip("/").endswith(f"/{dataset}.jsonl") or name == f"{dataset}.jsonl"]
                if not matches:
                    raise FileNotFoundError(f"{dataset}.jsonl not found in {archive}")
                with bundle.open(matches[0]) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
        return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]

    selected: list[PilotSample] = []
    for dataset, config in DATASET_CONFIGS.items():
        rows = dataset_rows(dataset)
        valid: list[PilotSample] = []
        for source_index in range(min(len(rows), args.dataset_scan_limit)):
            row = dict(rows[source_index])
            if not str(row.get("context", "")).strip() or not str(row.get("input", "")).strip():
                continue
            answers = [str(answer) for answer in (row.get("answers") or []) if str(answer).strip()]
            if not answers:
                continue
            try:
                ids, original, truncated = tokenize_prompt(
                    tokenizer,
                    dataset,
                    row,
                    max_context_length=args.max_context_length,
                )
                if len(ids) < args.min_prompt_tokens or len(ids) < 2:
                    continue
                # Validate the metric path before admitting the sample.
                longbench_score(dataset, answers[0], answers)
            except Exception as exc:
                print(f"sample_skip dataset={dataset} source_index={source_index} error={exc!r}", flush=True)
                continue
            sample_id = str(row.get("_id") or f"{dataset}:{source_index}")
            digest = hashlib.sha256(
                b"".join(int(token).to_bytes(4, "little", signed=False) for token in ids)
            ).hexdigest()
            tier = "base" if len(valid) < int(config["base"]) else "expansion"
            valid.append(
                PilotSample(
                    dataset=dataset,
                    sample_id=sample_id,
                    source_index=source_index,
                    input_ids=ids,
                    input_sha256=digest,
                    prompt_tokens_before_truncation=original,
                    prompt_tokens=len(ids),
                    truncated=truncated,
                    answers=answers,
                    all_classes=[str(value) for value in (row.get("all_classes") or [])],
                    max_new_tokens=int(config["max_new_tokens"]),
                    tier=tier,
                )
            )
            if len(valid) >= int(config["maximum"]):
                break
        if len(valid) < int(config["base"]):
            raise RuntimeError(
                f"Only {len(valid)} valid {dataset} samples found; need {config['base']}"
            )
        selected.extend(valid)
        print(
            f"selected dataset={dataset} base={config['base']} maximum={len(valid)} "
            f"source_indices={[sample.source_index for sample in valid]} "
            f"prompt_tokens={[sample.prompt_tokens for sample in valid]}",
            flush=True,
        )
    return selected


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class ExperimentClock:
    def __init__(self, limit_minutes: float, cutoff_minutes: float) -> None:
        self.started = time.monotonic()
        self.limit_seconds = limit_minutes * 60.0
        self.cutoff_seconds = cutoff_minutes * 60.0
        self.durations: deque[float] = deque(maxlen=8)
        self.seconds_per_token: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=8))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_seconds - self.elapsed)

    def estimate(self, spec: MethodSpec, max_new_tokens: int) -> float:
        values = self.seconds_per_token.get(spec.label)
        if values:
            return 2.0 + (sum(values) / len(values)) * max_new_tokens
        defaults = {"full": 0.34, "fier": 0.52, "bit2_qk": 0.50, "quest": 0.82}
        return 3.0 + defaults[spec.method] * max_new_tokens

    def can_start(self, spec: MethodSpec, max_new_tokens: int) -> tuple[bool, str | None]:
        if self.elapsed >= self.limit_seconds:
            return False, "time_limit_reached"
        estimate = self.estimate(spec, max_new_tokens)
        if self.elapsed >= self.cutoff_seconds and estimate > self.remaining:
            return False, f"after_cutoff_estimate_{estimate:.1f}s_gt_remaining_{self.remaining:.1f}s"
        return True, None

    def phase_fits_before_cutoff(
        self, combinations: Iterable[tuple[PilotSample, MethodSpec]]
    ) -> tuple[bool, float]:
        estimate = sum(self.estimate(spec, sample.max_new_tokens) for sample, spec in combinations)
        return self.elapsed + estimate <= self.cutoff_seconds, estimate

    def record(self, label: str, duration: float, generated_tokens: int) -> None:
        self.durations.append(duration)
        if generated_tokens > 0:
            self.seconds_per_token[label].append(duration / generated_tokens)

    def recent_average(self) -> float:
        return 0.0 if not self.durations else sum(self.durations) / len(self.durations)


def method_config(base: RetrievalConfig, spec: MethodSpec) -> RetrievalConfig:
    return base if spec.budget is None else replace(base, budget=spec.budget)


def eos_ids(model: Any, tokenizer: Any) -> set[int]:
    values: set[int] = set()
    for raw in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            values.update(int(value) for value in raw)
        else:
            values.add(int(raw))
    return values


@torch.inference_mode()
def generate_one(
    decoder: LlamaSparseDecoder,
    tokenizer: Any,
    sample: PilotSample,
    spec: MethodSpec,
    retrieval: RetrievalConfig,
    *,
    warmed_methods: set[str],
) -> dict[str, Any]:
    decoder.retrieval = retrieval
    ids = sample.input_ids
    check = hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little", signed=False) for token in ids)
    ).hexdigest()
    if check != sample.input_sha256:
        raise RuntimeError(f"Tokenized input changed for sample {sample.sample_id}")

    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)
    prefill_started = time.perf_counter()
    base_past = decoder.build_prefix_cache(torch.tensor(ids[:-1], dtype=torch.long))
    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)
    prefill_seconds = time.perf_counter() - prefill_started

    current_id = int(ids[-1])
    if spec.label not in warmed_methods:
        decoder.decode(base_past, current_id, method=spec.method, return_new_kv=False)
        if decoder.device.type == "cuda":
            torch.cuda.synchronize(decoder.device)
        warmed_methods.add(spec.label)

    past = base_past
    generated: list[int] = []
    total_model_decode_ms = 0.0
    total_search_ms = 0.0
    total_attention_ms = 0.0
    total_update_ms = 0.0
    total_candidate_ratio = 0.0
    stopped_on_eos = False
    stops = eos_ids(decoder.model, tokenizer)
    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)
    generation_started = time.perf_counter()
    for _ in range(sample.max_new_tokens):
        if decoder.device.type == "cuda":
            torch.cuda.synchronize(decoder.device)
        decode_started = time.perf_counter()
        logits, diagnostics, new_keys, new_values = decoder.decode(
            past,
            current_id,
            method=spec.method,
            return_new_kv=True,
        )
        if decoder.device.type == "cuda":
            torch.cuda.synchronize(decoder.device)
        total_model_decode_ms += (time.perf_counter() - decode_started) * 1000.0

        update_started = time.perf_counter()
        past = decoder.append_to_past_key_values(past, new_keys, new_values)
        if decoder.device.type == "cuda":
            torch.cuda.synchronize(decoder.device)
        update_ms = (time.perf_counter() - update_started) * 1000.0

        next_id = int(torch.argmax(logits, dim=-1).item())
        total_search_ms += diagnostics.candidate_search_ms
        total_attention_ms += diagnostics.selected_attention_ms
        total_update_ms += update_ms
        total_candidate_ratio += diagnostics.candidate_ratio
        if next_id in stops:
            stopped_on_eos = True
            break
        generated.append(next_id)
        current_id = next_id
    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)
    generation_seconds = time.perf_counter() - generation_started
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
    metric = longbench_score(sample.dataset, prediction, sample.answers)
    denominator = max(1, len(generated) + int(stopped_on_eos))
    return {
        "prediction": prediction,
        "score": metric,
        "generated_tokens": len(generated),
        "decode_steps": denominator,
        "stopped_on_eos": stopped_on_eos,
        "prefill_seconds": prefill_seconds,
        "generation_seconds": generation_seconds,
        "wall_decode_ms_per_token": generation_seconds * 1000.0 / denominator,
        "model_decode_ms_per_token": total_model_decode_ms / denominator,
        "candidate_search_ms_per_token": total_search_ms / denominator,
        "selected_attention_ms_per_token": total_attention_ms / denominator,
        "cache_update_ms_per_token": total_update_ms / denominator,
        "candidate_ratio": total_candidate_ratio / denominator,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status") == "ok"]
    keys = sorted({(row["dataset"], row["method_label"]) for row in valid})
    summary: list[dict[str, Any]] = []
    for dataset, label in keys:
        group = [row for row in valid if row["dataset"] == dataset and row["method_label"] == label]
        summary.append(
            {
                "dataset": dataset,
                "method": label,
                "num_generations": len(group),
                "num_generated_tokens": sum(int(row["generated_tokens"]) for row in group),
                "longbench_qa_f1": 100.0 * sum(float(row["score"]) for row in group) / len(group),
                "mean_prompt_tokens": sum(int(row["prompt_tokens"]) for row in group) / len(group),
                "mean_generation_seconds": sum(float(row["generation_seconds"]) for row in group) / len(group),
                "mean_wall_decode_ms_per_token": sum(float(row["wall_decode_ms_per_token"]) for row in group) / len(group),
                "mean_candidate_search_ms_per_token": sum(float(row["candidate_search_ms_per_token"]) for row in group) / len(group),
                "mean_selected_attention_ms_per_token": sum(float(row["selected_attention_ms_per_token"]) for row in group) / len(group),
                "mean_cache_update_ms_per_token": sum(float(row["cache_update_ms_per_token"]) for row in group) / len(group),
            }
        )
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_aggregate(output_dir: Path, metadata: dict[str, Any], omitted: list[dict[str, Any]]) -> None:
    rows = read_jsonl(output_dir / "raw_results.jsonl")
    summary = summarize(rows)
    metadata = dict(metadata)
    metadata["completed_generations"] = sum(row.get("status") == "ok" for row in rows)
    metadata["error_generations"] = sum(row.get("status") == "error" for row in rows)
    metadata["omitted_combinations"] = omitted
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def run_phase(
    name: str,
    combinations: list[tuple[PilotSample, MethodSpec]],
    *,
    decoder: LlamaSparseDecoder,
    tokenizer: Any,
    base_retrieval: RetrievalConfig,
    clock: ExperimentClock,
    raw_path: Path,
    warmed_methods: set[str],
    core_done: dict[str, set[str]],
    omitted: list[dict[str, Any]],
    core_labels: set[str],
) -> None:
    print(f"phase_start name={name} combinations={len(combinations)}", flush=True)
    for index, (sample, spec) in enumerate(combinations):
        allowed, reason = clock.can_start(spec, sample.max_new_tokens)
        if not allowed:
            for skipped_sample, skipped_spec in combinations[index:]:
                omitted.append(
                    {"phase": name, "sample_id": skipped_sample.sample_id, "method": skipped_spec.label, "reason": reason}
                )
            print(f"phase_stop name={name} reason={reason} omitted={len(combinations)-index}", flush=True)
            return
        generation_started = time.monotonic()
        common = {
            "phase": name,
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "source_index": sample.source_index,
            "tier": sample.tier,
            "method": spec.method,
            "method_label": spec.label,
            "budget": spec.budget,
            "prompt_tokens_before_truncation": sample.prompt_tokens_before_truncation,
            "prompt_tokens": sample.prompt_tokens,
            "truncated": sample.truncated,
            "input_sha256": sample.input_sha256,
            "max_new_tokens": sample.max_new_tokens,
            "answers": sample.answers,
        }
        try:
            generated = generate_one(
                decoder,
                tokenizer,
                sample,
                spec,
                method_config(base_retrieval, spec),
                warmed_methods=warmed_methods,
            )
            duration = time.monotonic() - generation_started
            row = {**common, "status": "ok", **generated, "experiment_elapsed_seconds": clock.elapsed}
            append_jsonl(raw_path, row)
            clock.record(spec.label, duration, int(generated["decode_steps"]))
            if spec.label in core_labels:
                core_done[sample.sample_id].add(spec.label)
        except Exception as exc:
            duration = time.monotonic() - generation_started
            row = {
                **common,
                "status": "error",
                "error": repr(exc),
                "generation_seconds": duration,
                "experiment_elapsed_seconds": clock.elapsed,
            }
            append_jsonl(raw_path, row)
            clock.record(spec.label, duration, 0)
            print(f"generation_error sample={sample.sample_id} method={spec.label} error={exc!r}", flush=True)
        completed_rows = read_jsonl(raw_path)
        completed_generations = sum(row.get("status") == "ok" for row in completed_rows)
        completed_samples = sum(core_labels.issubset(labels) for labels in core_done.values())
        remaining_combinations = combinations[index + 1 :]
        eta = sum(clock.estimate(next_spec, next_sample.max_new_tokens) for next_sample, next_spec in remaining_combinations)
        print(
            f"progress elapsed={clock.elapsed/60.0:.2f}m completed_generations={completed_generations} "
            f"completed_core_samples={completed_samples} recent_mean_generation={clock.recent_average():.1f}s "
            f"phase_eta={eta/60.0:.2f}m remaining={clock.remaining/60.0:.2f}m "
            f"next_omitted_if_stopped={len(remaining_combinations)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    load_local_env_token()
    if args.time_limit_minutes <= 0 or args.expansion_cutoff_minutes <= 0:
        raise SystemExit("Time limits must be positive")
    if args.expansion_cutoff_minutes >= args.time_limit_minutes:
        raise SystemExit("Expansion cutoff must be below the hard time limit")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    full_layers = parse_int_tuple(args.full_layers)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"outputs/longbench_sparse_pilot_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    if raw_path.exists():
        raise SystemExit(f"Refusing to overwrite existing raw results: {raw_path}")

    if args.dry_run:
        print(json.dumps({"datasets": DATASET_CONFIGS, "time_limit_minutes": args.time_limit_minutes}, indent=2))
        return

    load_started = time.monotonic()
    model, tokenizer = load_model_and_tokenizer(args)
    model_load_seconds = time.monotonic() - load_started
    # Requirement: the experiment wall clock begins immediately after model loading.
    clock = ExperimentClock(args.time_limit_minutes, args.expansion_cutoff_minutes)
    samples = select_samples(tokenizer, args)
    (output_dir / "sample_manifest.json").write_text(
        json.dumps([asdict(sample) for sample in samples], indent=2, ensure_ascii=False) + "\n"
    )

    base_retrieval = RetrievalConfig(
        budget=args.budget,
        quest_page_size=args.quest_page_size,
        fier_group_size=args.group_size,
        fier_backend="triton",
        bit2_group_size=args.group_size,
        bit2_backend="cuda_popc",
        full_layers=full_layers,
        measure_topk_recall=False,
    )
    decoder = LlamaSparseDecoder(model, pca_cache=None, retrieval=base_retrieval)
    full = MethodSpec("full", "full_attention", None)
    fier = MethodSpec("fier", f"fier_triton_g{args.group_size}_b{args.budget}", args.budget)
    bit2 = MethodSpec("bit2_qk", f"bit2_cuda_popc_g{args.group_size}_b{args.budget}", args.budget)
    quest = MethodSpec("quest", f"quest_p{args.quest_page_size}_b{args.budget}", args.budget)
    core_specs = [full, fier, bit2]
    core_labels = {spec.label for spec in core_specs}
    base_samples = [sample for sample in samples if sample.tier == "base"]
    expansion_samples = [sample for sample in samples if sample.tier == "expansion"]
    core_done: dict[str, set[str]] = defaultdict(set)
    warmed_methods: set[str] = set()
    omitted: list[dict[str, Any]] = []

    metadata = {
        "created_at_utc": timestamp,
        "benchmark": "longbench_sparse_pilot",
        "model_id": args.model,
        "model_load_seconds": model_load_seconds,
        "experiment_timer_starts": "immediately_after_model_loading",
        "time_limit_minutes": args.time_limit_minutes,
        "expansion_cutoff_minutes": args.expansion_cutoff_minutes,
        "max_context_length": args.max_context_length,
        "base_budget": args.budget,
        "optional_budget": args.optional_budget,
        "group_size": args.group_size,
        "quest_page_size": args.quest_page_size,
        "full_layers": full_layers,
        "dtype": args.dtype,
        "attention": args.attention,
        "seed": args.seed,
        "sample_counts": {key: value for key, value in DATASET_CONFIGS.items()},
        "prompt_source": "THUDM/LongBench v1 official dataset2prompt.json",
        "metric": "THUDM/LongBench v1 qa_f1_score",
        "truncation": "tokenized_prompt_middle_removed; equal head/tail retained",
    }
    write_aggregate(output_dir, metadata, omitted)

    core_base = [(sample, spec) for sample in base_samples for spec in core_specs]
    run_phase(
        "core_base", core_base, decoder=decoder, tokenizer=tokenizer,
        base_retrieval=base_retrieval, clock=clock, raw_path=raw_path,
        warmed_methods=warmed_methods, core_done=core_done, omitted=omitted,
        core_labels=core_labels,
    )
    write_aggregate(output_dir, metadata, omitted)

    if not args.disable_quest and all(core_labels.issubset(core_done[sample.sample_id]) for sample in base_samples):
        quest_base = [(sample, quest) for sample in base_samples]
        fits, estimate = clock.phase_fits_before_cutoff(quest_base)
        if fits:
            run_phase(
                "quest_base", quest_base, decoder=decoder, tokenizer=tokenizer,
                base_retrieval=base_retrieval, clock=clock, raw_path=raw_path,
                warmed_methods=warmed_methods, core_done=core_done, omitted=omitted,
                core_labels=core_labels,
            )
        else:
            omitted.extend({"phase": "quest_base", "sample_id": s.sample_id, "method": m.label, "reason": f"phase_estimate_{estimate:.1f}s_exceeds_45m_cutoff"} for s, m in quest_base)
            print(f"phase_skip name=quest_base estimate={estimate/60.0:.2f}m", flush=True)
    write_aggregate(output_dir, metadata, omitted)

    if not args.disable_expansion and expansion_samples:
        expansion_specs = core_specs + ([] if args.disable_quest else [quest])
        expansion = [(sample, spec) for sample in expansion_samples for spec in expansion_specs]
        fits, estimate = clock.phase_fits_before_cutoff(expansion)
        if fits:
            run_phase(
                "auto_expansion", expansion, decoder=decoder, tokenizer=tokenizer,
                base_retrieval=base_retrieval, clock=clock, raw_path=raw_path,
                warmed_methods=warmed_methods, core_done=core_done, omitted=omitted,
                core_labels=core_labels,
            )
        else:
            omitted.extend({"phase": "auto_expansion", "sample_id": s.sample_id, "method": m.label, "reason": f"phase_estimate_{estimate:.1f}s_exceeds_45m_cutoff"} for s, m in expansion)
            print(f"phase_skip name=auto_expansion estimate={estimate/60.0:.2f}m", flush=True)
    write_aggregate(output_dir, metadata, omitted)

    if not args.disable_optional_budget:
        optional_specs = [
            MethodSpec("fier", f"fier_triton_g{args.group_size}_b{args.optional_budget}", args.optional_budget),
            MethodSpec("bit2_qk", f"bit2_cuda_popc_g{args.group_size}_b{args.optional_budget}", args.optional_budget),
        ]
        if not args.disable_quest:
            optional_specs.append(MethodSpec("quest", f"quest_p{args.quest_page_size}_b{args.optional_budget}", args.optional_budget))
        optional = [(sample, spec) for sample in base_samples for spec in optional_specs]
        fits, estimate = clock.phase_fits_before_cutoff(optional)
        if fits:
            run_phase(
                "optional_budget_512", optional, decoder=decoder, tokenizer=tokenizer,
                base_retrieval=base_retrieval, clock=clock, raw_path=raw_path,
                warmed_methods=warmed_methods, core_done=core_done, omitted=omitted,
                core_labels=core_labels,
            )
        else:
            omitted.extend({"phase": "optional_budget_512", "sample_id": s.sample_id, "method": m.label, "reason": f"phase_estimate_{estimate:.1f}s_exceeds_45m_cutoff"} for s, m in optional)
            print(f"phase_skip name=optional_budget_512 estimate={estimate/60.0:.2f}m", flush=True)

    metadata["experiment_elapsed_seconds"] = clock.elapsed
    metadata["finished_before_hard_limit"] = clock.elapsed <= clock.limit_seconds
    write_aggregate(output_dir, metadata, omitted)
    print(
        f"finished elapsed={clock.elapsed/60.0:.2f}m hard_limit={args.time_limit_minutes:.2f}m "
        f"results={raw_path} omitted={len(omitted)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
