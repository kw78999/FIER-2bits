"""Pop-KV and three comparison attention paths used in the COLM workshop experiments.

The public API intentionally contains only the final Pop-KV method and the three
baselines used in the paper: FullAttention, Quest, and FIER.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

import torch
import torch.nn.functional as F

METHODS = ("full", "quest", "fier", "popkv")
T = TypeVar("T")


@dataclass(frozen=True)
class RetrievalConfig:
    budget: int = 4096
    group_size: int = 32
    quest_page_size: int = 16
    full_layers: tuple[int, ...] = (0, 1)

    def validate(self) -> None:
        if self.budget <= 1:
            raise ValueError("budget must reserve at least one retrieved and one current token")
        if self.group_size != 32:
            raise ValueError("the released Pop-KV/FIER kernels use group_size=32")
        if self.quest_page_size <= 0:
            raise ValueError("quest_page_size must be positive")


@dataclass
class QuestCache:
    page_size: int
    token_count: int
    sealed_pages: int
    page_min: torch.Tensor
    page_max: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class FIERCache:
    group_size: int
    token_count: int
    sealed_tokens: int
    packed: torch.Tensor
    group_min: torch.Tensor
    group_max: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class PopKVCache:
    group_size: int
    token_count: int
    sealed_tokens: int
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    active_keys: torch.Tensor


@dataclass
class RetrievalCaches:
    method: str
    layers: dict[int, QuestCache | FIERCache | PopKVCache]


@dataclass
class StaticKVCache:
    key_storage: list[torch.Tensor]
    value_storage: list[torch.Tensor]
    length: int
    capacity: int


@dataclass
class DecodeDiagnostics:
    selected_tokens: int = 0
    available_tokens: int = 0
    sparse_head_calls: int = 0
    candidate_search_ms: float = 0.0
    candidate_score_ms: float = 0.0
    candidate_topk_ms: float = 0.0
    selected_gather_ms: float = 0.0
    selected_attention_ms: float = 0.0

    @property
    def candidate_ratio(self) -> float:
        return 1.0 if self.available_tokens == 0 else self.selected_tokens / self.available_tokens


def load_hf_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attention: str = "sdpa",
    token: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a gated Hugging Face Llama model after `hf auth login` or HF_TOKEN."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except Exception:
                token = None
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, use_fast=True, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attention,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def _append_current(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    current = torch.full(
        (*indices.shape[:-1], 1), num_tokens - 1, device=indices.device, dtype=torch.long,
    )
    return torch.cat([indices.long(), current], dim=-1)


# ---- Quest -----------------------------------------------------------------

def build_quest_cache(keys: torch.Tensor, *, page_size: int, reserve_tokens: int) -> QuestCache:
    kv_heads, tokens, dim = (int(x) for x in keys.shape)
    sealed_pages = tokens // page_size
    sealed_tokens = sealed_pages * page_size
    capacity_pages = math.ceil((tokens + reserve_tokens + 1) / page_size)
    page_min = torch.empty((kv_heads, capacity_pages, dim), device=keys.device, dtype=torch.float32)
    page_max = torch.empty_like(page_min)
    if sealed_pages:
        pages = keys[:, :sealed_tokens].float().view(kv_heads, sealed_pages, page_size, dim)
        page_min[:, :sealed_pages].copy_(pages.amin(dim=2))
        page_max[:, :sealed_pages].copy_(pages.amax(dim=2))
    return QuestCache(
        page_size, tokens, sealed_pages, page_min, page_max,
        keys[:, sealed_tokens:].detach().clone().contiguous(),
    )


def quest_scores(
    queries: torch.Tensor, cache: QuestCache, current_keys: torch.Tensor,
    head_to_kv: torch.Tensor,
) -> torch.Tensor:
    staged = torch.cat([cache.active_keys, current_keys[:, None].detach()], dim=1).float()
    page = cache.sealed_pages
    cache.page_min[:, page].copy_(staged.amin(dim=1))
    cache.page_max[:, page].copy_(staged.amax(dim=1))
    mins = cache.page_min[:, : page + 1].index_select(0, head_to_kv)
    maxs = cache.page_max[:, : page + 1].index_select(0, head_to_kv)
    q = queries.float()[:, None]
    return torch.maximum(q * mins, q * maxs).sum(dim=-1)


def quest_indices(
    scores: torch.Tensor, *, num_tokens: int, page_size: int, budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_budget = min(int(scores.shape[-1]), max(1, math.ceil(budget / page_size)))
    pages = torch.topk(scores, k=page_budget, dim=-1, sorted=False).indices
    offsets = torch.arange(page_size, device=scores.device)
    indices = (pages[:, :, None] * page_size + offsets).flatten(1)
    valid = indices.lt(num_tokens)
    current = num_tokens - 1
    has_current = (indices.eq(current) & valid).any(dim=-1)
    indices[:, -1] = torch.where(has_current, indices[:, -1], indices.new_tensor(current))
    valid[:, -1] = torch.where(has_current, valid[:, -1], valid.new_tensor(True))
    return indices, valid


def append_quest(cache: QuestCache, keys: torch.Tensor) -> None:
    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    cache.token_count += int(keys.shape[1])
    if int(cache.active_keys.shape[1]) == cache.page_size:
        page = cache.sealed_pages
        active = cache.active_keys.float()
        cache.page_min[:, page].copy_(active.amin(dim=1))
        cache.page_max[:, page].copy_(active.amax(dim=1))
        cache.sealed_pages += 1
        cache.active_keys = cache.active_keys[:, :0].contiguous()


# ---- FIER ------------------------------------------------------------------

def build_fier_cache(keys: torch.Tensor, *, group_size: int, reserve_tokens: int) -> FIERCache:
    from fier_triton import pack_keys

    packed, mins, maxs = pack_keys(keys.contiguous(), group_size=group_size)
    kv_heads, tokens, dim = (int(x) for x in keys.shape)
    capacity = math.ceil((tokens + reserve_tokens) / group_size) * group_size
    packed_storage = torch.empty(
        (kv_heads, dim, math.ceil(capacity / 32)), device=keys.device, dtype=packed.dtype,
    )
    min_storage = torch.empty(
        (kv_heads, math.ceil(capacity / group_size), dim), device=keys.device, dtype=mins.dtype,
    )
    max_storage = torch.empty_like(min_storage)
    packed_storage[:, :, : packed.shape[2]].copy_(packed)
    min_storage[:, : mins.shape[1]].copy_(mins)
    max_storage[:, : maxs.shape[1]].copy_(maxs)
    sealed = (tokens // group_size) * group_size
    return FIERCache(
        group_size, tokens, sealed, packed_storage, min_storage, max_storage,
        keys[:, sealed:].detach().clone().contiguous(),
    )


def stage_fier(cache: FIERCache, current_keys: torch.Tensor) -> None:
    from fier_triton import pack_keys

    staged = torch.cat([cache.active_keys, current_keys.detach()], dim=1).contiguous()
    packed, mins, maxs = pack_keys(staged, group_size=cache.group_size)
    word = cache.sealed_tokens // 32
    group = cache.sealed_tokens // cache.group_size
    cache.packed[:, :, word : word + packed.shape[2]].copy_(packed)
    cache.group_min[:, group : group + mins.shape[1]].copy_(mins)
    cache.group_max[:, group : group + maxs.shape[1]].copy_(maxs)


def append_fier(cache: FIERCache, keys: torch.Tensor) -> None:
    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    cache.token_count += int(keys.shape[1])
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()


# ---- Pop-KV ----------------------------------------------------------------

def build_popkv_cache(keys: torch.Tensor, *, group_size: int, reserve_tokens: int) -> PopKVCache:
    from popkv_cuda import pack_keys

    if group_size != 32:
        raise ValueError("Pop-KV release kernel requires group_size=32")
    tokens = int(keys.shape[1])
    capacity = math.ceil((tokens + reserve_tokens) / group_size) * group_size
    packed = pack_keys(keys.contiguous(), token_capacity=capacity)
    sealed = (tokens // group_size) * group_size
    return PopKVCache(
        group_size, tokens, sealed, packed,
        keys[:, sealed:].detach().clone().contiguous(),
    )


def append_popkv(cache: PopKVCache, keys: torch.Tensor) -> None:
    from popkv_cuda import pack_keys_into

    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    if cache.sealed_tokens + int(cache.active_keys.shape[1]) > int(cache.packed[0].shape[1]):
        raise RuntimeError("Pop-KV cache reserve exhausted")
    pack_keys_into(cache.active_keys, cache.packed, token_offset=cache.sealed_tokens)
    cache.token_count += 1
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()


def popkv_reference_scores(
    queries: torch.Tensor, keys: torch.Tensor, *, head_to_kv: torch.Tensor, group_size: int = 32,
) -> torch.Tensor:
    """Readable reference for the final Pop-KV score used by correctness tests."""
    def representatives(x: torch.Tensor, magnitude: torch.Tensor):
        words = math.ceil(x.shape[-1] / 32)
        pad = words * 32 - x.shape[-1]
        absolute = x.abs()
        valid = torch.ones_like(magnitude)
        if pad:
            shape = (*x.shape[:-1], pad)
            magnitude = torch.cat([magnitude, torch.zeros(shape, dtype=torch.bool, device=x.device)], -1)
            valid = torch.cat([valid, torch.zeros(shape, dtype=torch.bool, device=x.device)], -1)
            absolute = torch.cat([absolute, torch.zeros(shape, device=x.device)], -1)
        shape = (*magnitude.shape[:-1], words, 32)
        mag, valid, absolute = magnitude.view(shape), valid.view(shape), absolute.view(shape)
        low_mask, high_mask = (~mag) & valid, mag & valid
        low_count, high_count = low_mask.sum(-1), high_mask.sum(-1)
        low = (absolute * low_mask).sum(-1) / low_count.clamp_min(1)
        high = (absolute * high_mask).sum(-1) / high_count.clamp_min(1)
        low = torch.where(low_count > 0, low, high)
        high = torch.where(high_count > 0, high, low)
        return low, high

    q = queries.float()
    qmin, qmax = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    qsign = q >= 0
    qmag = torch.where(qsign, q > qmax * .5, q < qmin * .5)
    qlow, qhigh = representatives(q, qmag)

    k = keys.float()
    kvh, tokens, dim = k.shape
    groups = math.ceil(tokens / group_size)
    padded = groups * group_size - tokens
    extrema = F.pad(k, (0, 0, 0, padded), value=float("nan")) if padded else k
    grouped = extrema.view(kvh, groups, group_size, dim)
    minimum = torch.nan_to_num(grouped, nan=float("inf")).amin(2)
    maximum = torch.nan_to_num(grouped, nan=-float("inf")).amax(2)
    group_ids = torch.arange(tokens, device=k.device) // group_size
    minimum, maximum = minimum[:, group_ids], maximum[:, group_ids]
    ksign = k >= 0
    kmag = torch.where(ksign, k > maximum * .5, k < minimum * .5)
    klow, khigh = representatives(k, kmag)

    words = qlow.shape[-1]
    ql = qlow.repeat_interleave(32, -1)[..., :dim]
    qh = qhigh.repeat_interleave(32, -1)[..., :dim]
    kl = klow.index_select(0, head_to_kv).repeat_interleave(32, -1)[..., :dim]
    kh = khigh.index_select(0, head_to_kv).repeat_interleave(32, -1)[..., :dim]
    qhat = torch.where(qsign, 1.0, -1.0) * torch.where(qmag, qh, ql)
    mapped_sign = ksign.index_select(0, head_to_kv)
    mapped_mag = kmag.index_select(0, head_to_kv)
    khat = torch.where(mapped_sign, 1.0, -1.0) * torch.where(mapped_mag, kh, kl)
    return torch.einsum("bhd,htd->bht", qhat, khat) / math.sqrt(dim)


# ---- Shared attention and model adapter ------------------------------------

def gather_selected_kv(
    prefix_keys: torch.Tensor, prefix_values: torch.Tensor,
    current_keys: torch.Tensor, current_values: torch.Tensor,
    indices: torch.Tensor, head_to_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_tokens = int(prefix_keys.shape[1])
    safe = indices.clamp_max(prefix_tokens - 1)
    kv_heads = head_to_kv[:, None].expand_as(safe)
    selected_keys = prefix_keys[kv_heads, safe]
    selected_values = prefix_values[kv_heads, safe]
    current_mask = indices.eq(prefix_tokens).unsqueeze(-1)
    current_keys_q = current_keys.index_select(0, head_to_kv)[:, None]
    current_values_q = current_values.index_select(0, head_to_kv)[:, None]
    return (
        torch.where(current_mask, current_keys_q, selected_keys),
        torch.where(current_mask, current_values_q, selected_values),
    )


def selected_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        queries[None, :, None], keys[None], values[None],
        attn_mask=None if valid_mask is None else valid_mask[None, :, None],
        dropout_p=0.0, is_causal=False,
    )[0, :, 0]


def _timed(operation: Callable[[], T], device: torch.device, events: list) -> tuple[T, float]:
    if device.type == "cuda":
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); result = operation(); end.record()
        events.append((start, end, device))
        return result, 0.0
    started = time.perf_counter(); result = operation()
    return result, (time.perf_counter() - started) * 1000


def _finish(events: Sequence) -> float:
    if not events:
        return 0.0
    for device in {x[2] for x in events}:
        torch.cuda.synchronize(device)
    return sum(float(start.elapsed_time(end)) for start, end, _ in events)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _layer_cache(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, StaticKVCache):
        return cache.key_storage[layer][..., :cache.length, :], cache.value_storage[layer][..., :cache.length, :]
    try:
        return cache[layer][:2]
    except (TypeError, IndexError, KeyError):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return cache.key_cache[layer], cache.value_cache[layer]
        raise TypeError(f"unsupported Transformers cache: {type(cache).__name__}")


class LlamaSparseDecoder:
    """One-token Llama decoder exposing FullAttention, Quest, FIER, and Pop-KV."""

    def __init__(self, model: Any, retrieval: RetrievalConfig = RetrievalConfig()) -> None:
        retrieval.validate()
        self.model = model
        self.backbone = getattr(model, "model", None)
        if self.backbone is None:
            raise TypeError("expected a LlamaForCausalLM-compatible model")
        self.layers = self.backbone.layers
        self.retrieval = retrieval
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(getattr(model.config, "num_key_value_heads", self.num_heads))
        self.kv_group_size = self.num_heads // self.num_kv_heads
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // self.num_heads))
        self.head_to_kv = torch.arange(self.num_heads, device=self.device, dtype=torch.long) // self.kv_group_size

    @property
    def device(self) -> torch.device:
        return self.backbone.embed_tokens.weight.device

    @torch.inference_mode()
    def build_prefix_cache(self, prefix_ids: torch.Tensor):
        ids = prefix_ids.to(self.device).view(1, -1)
        return self.backbone(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
            output_hidden_states=False, return_dict=True,
        ).past_key_values

    @torch.inference_mode()
    def build_retrieval_caches(
        self, prefix_cache: Any, *, method: str, reserve_tokens: int = 2048,
    ) -> RetrievalCaches:
        if method not in {"quest", "fier", "popkv"}:
            raise ValueError("retrieval caches are only used by sparse methods")
        layers: dict[int, QuestCache | FIERCache | PopKVCache] = {}
        for layer in range(len(self.layers)):
            keys, _ = _layer_cache(prefix_cache, layer)
            keys = keys[0].to(self.device)
            if method == "quest":
                layers[layer] = build_quest_cache(
                    keys, page_size=self.retrieval.quest_page_size, reserve_tokens=reserve_tokens,
                )
            elif method == "fier":
                layers[layer] = build_fier_cache(
                    keys, group_size=self.retrieval.group_size, reserve_tokens=reserve_tokens,
                )
            else:
                layers[layer] = build_popkv_cache(
                    keys, group_size=self.retrieval.group_size, reserve_tokens=reserve_tokens,
                )
        return RetrievalCaches(method, layers)

    @torch.inference_mode()
    def append_retrieval_caches(self, caches: RetrievalCaches, new_keys: Sequence[torch.Tensor]) -> None:
        for layer, keys in enumerate(new_keys):
            cache = caches.layers[layer]
            if isinstance(cache, QuestCache): append_quest(cache, keys[0])
            elif isinstance(cache, FIERCache): append_fier(cache, keys[0])
            elif isinstance(cache, PopKVCache): append_popkv(cache, keys[0])
            else: raise TypeError("unknown retrieval cache")

    @torch.inference_mode()
    def build_static_kv_cache(self, prefix_cache: Any, *, reserve_tokens: int) -> StaticKVCache:
        first, _ = _layer_cache(prefix_cache, 0)
        length, capacity = int(first.shape[-2]), int(first.shape[-2]) + reserve_tokens
        keys_out, values_out = [], []
        for layer in range(len(self.layers)):
            keys, values = _layer_cache(prefix_cache, layer)
            key_store = torch.empty((*keys.shape[:-2], capacity, keys.shape[-1]), device=keys.device, dtype=keys.dtype)
            value_store = torch.empty_like(key_store)
            key_store[..., :length].copy_(keys); value_store[..., :length].copy_(values)
            keys_out.append(key_store); values_out.append(value_store)
        return StaticKVCache(keys_out, values_out, length, capacity)

    @torch.inference_mode()
    def append_kv(self, cache: StaticKVCache, new_keys: Sequence[torch.Tensor], new_values: Sequence[torch.Tensor]) -> None:
        if cache.length >= cache.capacity:
            raise RuntimeError("static KV cache capacity exhausted")
        position = cache.length
        for layer, (keys, values) in enumerate(zip(new_keys, new_values)):
            cache.key_storage[layer][..., position:position + 1, :].copy_(keys)
            cache.value_storage[layer][..., position:position + 1, :].copy_(values)
        cache.length += 1

    def _position_embeddings(self, hidden: torch.Tensor, positions: torch.Tensor, values: torch.Tensor):
        rotary = getattr(self.backbone, "rotary_emb", None)
        if rotary is not None:
            try: return rotary(hidden, positions)
            except TypeError: pass
        rotary = getattr(self.layers[0].self_attn, "rotary_emb", None)
        if rotary is None: raise RuntimeError("cannot find Llama rotary embedding")
        try: return rotary(values, positions)
        except TypeError:
            cos, sin = rotary(values, seq_len=int(positions.max()) + 1)
            position = positions.item()
            return cos[:, position:position + 1], sin[:, position:position + 1]

    @torch.inference_mode()
    def decode(
        self, cache: StaticKVCache, token_id: int | torch.Tensor, *, method: str,
        retrieval_caches: RetrievalCaches | None = None, return_new_kv: bool = False,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if method != "full" and retrieval_caches is None:
            raise ValueError(f"{method} requires build_retrieval_caches()")
        prefix_length = cache.length
        token = torch.as_tensor(token_id, device=self.device, dtype=torch.long).view(1, 1)
        hidden = self.backbone.embed_tokens(token)
        positions = torch.tensor([[prefix_length]], device=self.device)
        first_attn = self.layers[0].self_attn
        first_value = first_attn.v_proj(hidden).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._position_embeddings(hidden, positions, first_value)
        diag = DecodeDiagnostics()
        score_events, topk_events, gather_events, attention_events = [], [], [], []
        new_keys, new_values = [], []

        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            q = attention.q_proj(normalized).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = attention.k_proj(normalized).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attention.v_proj(normalized).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, cos.to(q.device), sin.to(q.device))
            prefix_k, prefix_v = _layer_cache(cache, layer_idx)
            if return_new_kv:
                new_keys.append(k.detach()); new_values.append(v.detach())

            force_full = method == "full" or layer_idx in self.retrieval.full_layers
            if force_full:
                def gather_full(): return torch.cat([prefix_k, k], 2), torch.cat([prefix_v, v], 2)
                (all_k, all_v), ms = _timed(gather_full, q.device, gather_events); diag.selected_gather_ms += ms
                def attend_full():
                    grouped = q[0].view(self.num_kv_heads, self.kv_group_size, 1, self.head_dim)
                    return F.scaled_dot_product_attention(
                        grouped, all_k[0, :, None], all_v[0, :, None], dropout_p=0.0, is_causal=False,
                    ).reshape(self.num_heads, self.head_dim)
                output, ms = _timed(attend_full, q.device, attention_events); diag.selected_attention_ms += ms
            else:
                num_tokens = prefix_length + 1
                sparse_cache = retrieval_caches.layers[layer_idx]  # type: ignore[union-attr]
                valid = None
                if method == "quest":
                    assert isinstance(sparse_cache, QuestCache)
                    scores, ms = _timed(
                        lambda: quest_scores(q[0, :, 0], sparse_cache, k[0, :, 0], self.head_to_kv),
                        q.device, score_events,
                    ); diag.candidate_score_ms += ms
                    (indices, valid), ms = _timed(
                        lambda: quest_indices(scores, num_tokens=num_tokens,
                            page_size=self.retrieval.quest_page_size, budget=self.retrieval.budget),
                        q.device, topk_events,
                    ); diag.candidate_topk_ms += ms
                elif method == "fier":
                    assert isinstance(sparse_cache, FIERCache)
                    from fier_triton import score_packed_batched
                    def score_fier():
                        stage_fier(sparse_cache, k[0])
                        return score_packed_batched(
                            q[0, :, 0].unsqueeze(0), sparse_cache.packed,
                            sparse_cache.group_min, sparse_cache.group_max,
                            self.head_to_kv, tokens=num_tokens, group_size=sparse_cache.group_size,
                        )
                    scores, ms = _timed(score_fier, q.device, score_events); diag.candidate_score_ms += ms
                    def topk_fier():
                        selected = torch.topk(
                            scores[..., :prefix_length], k=min(self.retrieval.budget - 1, prefix_length),
                            dim=-1, sorted=False,
                        ).indices[0]
                        return _append_current(selected, num_tokens)
                    indices, ms = _timed(topk_fier, q.device, topk_events); diag.candidate_topk_ms += ms
                else:
                    assert isinstance(sparse_cache, PopKVCache)
                    from popkv_cuda import score as score_popkv
                    scores, ms = _timed(
                        lambda: score_popkv(
                            q[0, :, 0].unsqueeze(0).contiguous(), sparse_cache.packed,
                            head_to_kv=self.head_to_kv, tokens=sparse_cache.token_count,
                        ), q.device, score_events,
                    ); diag.candidate_score_ms += ms
                    def topk_popkv():
                        selected = torch.topk(
                            scores, k=min(self.retrieval.budget - 1, prefix_length),
                            dim=-1, sorted=False,
                        ).indices[0]
                        return _append_current(selected, num_tokens)
                    indices, ms = _timed(topk_popkv, q.device, topk_events); diag.candidate_topk_ms += ms

                diag.selected_tokens += int(indices.numel())
                diag.available_tokens += num_tokens * self.num_heads
                diag.sparse_head_calls += self.num_heads
                def gather_sparse():
                    return gather_selected_kv(prefix_k[0], prefix_v[0], k[0, :, 0], v[0, :, 0], indices, self.head_to_kv)
                (selected_k, selected_v), ms = _timed(gather_sparse, q.device, gather_events); diag.selected_gather_ms += ms
                output, ms = _timed(
                    lambda: selected_attention(q[0, :, 0], selected_k, selected_v, valid),
                    q.device, attention_events,
                ); diag.selected_attention_ms += ms

            output = output.reshape(1, 1, -1).to(normalized.dtype)
            hidden = residual + attention.o_proj(output)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

        hidden = self.backbone.norm(hidden)
        logits = self.model.lm_head(hidden).float()[0, -1]
        diag.candidate_score_ms += _finish(score_events)
        diag.candidate_topk_ms += _finish(topk_events)
        diag.candidate_search_ms = diag.candidate_score_ms + diag.candidate_topk_ms
        diag.selected_gather_ms += _finish(gather_events)
        diag.selected_attention_ms += _finish(attention_events)
        if return_new_kv:
            return logits, diag, new_keys, new_values
        return logits, diag
