from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_popc backend requires CUDA")
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    return load(
        name="bit2_popc_ext",
        sources=[
            str(_ROOT / "csrc" / "bit2_popc_ext.cpp"),
            str(_ROOT / "csrc" / "bit2_popc_ext.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=bool(os.environ.get("BIT2_CUDA_VERBOSE")),
    )


def pack_query(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_extension().quantize_query_pack(query.contiguous()))


def pack_keys(
    keys: torch.Tensor, *, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_extension().quantize_key_pack(keys.contiguous(), int(group_size)))


def pack_keys_2mean(
    keys: torch.Tensor, *, group_size: int, token_capacity: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack K sign/magnitude words plus FP16 low/delta means."""

    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    capacity = int(keys.shape[1]) if token_capacity is None else int(token_capacity)
    return tuple(
        _extension().quantize_key_pack_2mean(keys, int(group_size), capacity)
    )


def pack_keys_cached(
    keys: torch.Tensor, *, group_size: int, token_capacity: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    capacity = int(keys.shape[1]) if token_capacity is None else int(token_capacity)
    return tuple(_extension().quantize_key_pack_cached(keys, int(group_size), capacity))


def pack_keys_cached_into(
    keys: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor],
    *, group_size: int, token_offset: int,
) -> None:
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    _extension().quantize_key_pack_cached_into(
        keys, *packed, int(group_size), int(token_offset)
    )


def score_popc_cached_cuda_packed(
    queries: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor],
    *, head_to_kv: torch.Tensor, tokens: int,
) -> torch.Tensor:
    if not queries.is_contiguous():
        raise ValueError("queries must be contiguous")
    return _extension().score_popc_cached_fused(
        queries, *packed, head_to_kv, int(queries.shape[-1]), int(tokens)
    )


def pack_keys_2mean_into(
    keys: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    group_size: int,
    token_offset: int,
) -> None:
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    _extension().quantize_key_pack_2mean_into(
        keys, *packed, int(group_size), int(token_offset)
    )


def score_2mean_cuda_packed(
    queries: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    head_to_kv: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    if not queries.is_contiguous():
        raise ValueError("queries must be contiguous")
    k_sign, k_mag, low_mean, delta_mean = packed
    return _extension().score_2mean_fused(
        queries,
        k_sign,
        k_mag,
        low_mean,
        delta_mean,
        head_to_kv,
        int(queries.shape[-1]),
        int(tokens),
    )


def score_qk_2mean_cuda_packed(
    queries: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    head_to_kv: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    """Fused Q/K representative-magnitude weighted POPC scoring."""
    if not queries.is_contiguous():
        raise ValueError("queries must be contiguous")
    return _extension().score_qk_2mean_fused(
        queries, *packed, head_to_kv, int(queries.shape[-1]), int(tokens)
    )


def reference_q_2mean_components(
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 Q sign/magnitude and per-query/per-32D low/delta metadata."""
    if queries.ndim != 3:
        raise ValueError("queries must be [B, QH, D]")
    q = queries.float()
    q_min, q_max = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    sign = q >= 0
    magnitude = torch.where(sign, q > q_max * 0.5, q < q_min * 0.5)
    batch, heads, head_dim = q.shape
    words = (head_dim + 31) // 32
    pad = words * 32 - head_dim
    abs_q = q.abs()
    valid = torch.ones_like(sign)
    if pad:
        shape = (batch, heads, pad)
        zeros_bool = torch.zeros(shape, dtype=torch.bool, device=q.device)
        zeros_float = torch.zeros(shape, dtype=q.dtype, device=q.device)
        magnitude_padded = torch.cat([magnitude, zeros_bool], -1)
        valid = torch.cat([valid, zeros_bool], -1)
        abs_q = torch.cat([abs_q, zeros_float], -1)
    else:
        magnitude_padded = magnitude
    shape4 = (batch, heads, words, 32)
    mag4, valid4, abs4 = (
        magnitude_padded.view(shape4), valid.view(shape4), abs_q.view(shape4)
    )
    low_mask, high_mask = (~mag4) & valid4, mag4 & valid4
    low_count, high_count = low_mask.sum(-1), high_mask.sum(-1)
    low = (abs4 * low_mask).sum(-1) / low_count.clamp_min(1)
    high = (abs4 * high_mask).sum(-1) / high_count.clamp_min(1)
    low = torch.where(low_count > 0, low, high)
    high = torch.where(high_count > 0, high, low)
    return sign, magnitude, low, high - low


def reference_qk_2mean_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Direct q_hat*k_hat reconstruction reference, accumulated in FP32."""
    q_sign, q_mag, q_low, q_delta = reference_q_2mean_components(queries)
    k_sign, k_mag, k_low, k_delta = reference_2mean_components(
        keys, group_size=group_size
    )
    head_dim = int(queries.shape[-1])
    ql = q_low.repeat_interleave(32, -1)[..., :head_dim]
    qd = q_delta.repeat_interleave(32, -1)[..., :head_dim]
    kl = k_low.index_select(0, head_to_kv).repeat_interleave(32, -1)[..., :head_dim]
    kd = k_delta.index_select(0, head_to_kv).repeat_interleave(32, -1)[..., :head_dim]
    q_hat = torch.where(q_sign, 1.0, -1.0) * torch.where(q_mag, ql + qd, ql)
    mapped_k_sign = k_sign.index_select(0, head_to_kv)
    mapped_k_mag = k_mag.index_select(0, head_to_kv)
    k_hat = torch.where(mapped_k_sign, 1.0, -1.0) * torch.where(
        mapped_k_mag, kl + kd, kl
    )
    return torch.einsum("bhd,htd->bht", q_hat, k_hat) / (head_dim ** 0.5)


def reference_qk_2mean_count_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Independent reference for the four signed-count weighted terms."""
    q_sign, q_mag, q_low, q_delta = reference_q_2mean_components(queries)
    k_sign, k_mag, k_low, k_delta = reference_2mean_components(keys, group_size=group_size)
    ks = k_sign.index_select(0, head_to_kv).unsqueeze(0)
    km = k_mag.index_select(0, head_to_kv).unsqueeze(0)
    batch, q_heads, head_dim = q_sign.shape
    tokens, words = int(ks.shape[2]), int(q_low.shape[-1])
    qs = q_sign.unsqueeze(2).expand(batch, q_heads, tokens, head_dim)
    qm = q_mag.unsqueeze(2).expand_as(qs)
    match = qs == ks.expand(batch, -1, -1, -1)
    valid = torch.ones_like(match)
    pad = words * 32 - head_dim
    if pad:
        zeros = torch.zeros((*match.shape[:-1], pad), device=match.device, dtype=torch.bool)
        match, qm, km, valid = (
            torch.cat([x, zeros], -1) for x in (match, qm, km.expand(batch, -1, -1, -1), valid)
        )
    else:
        km = km.expand(batch, -1, -1, -1)
    shape = (batch, q_heads, tokens, words, 32)
    match, qm, km, valid = (x.reshape(shape) for x in (match, qm, km, valid))
    def signed(mask: torch.Tensor) -> torch.Tensor:
        return 2 * (match & mask).sum(-1) - mask.sum(-1)
    cs, cq, ck, cqk = signed(valid), signed(qm), signed(km), signed(qm & km)
    ql, dq = q_low.unsqueeze(2), q_delta.unsqueeze(2)
    kl = k_low.index_select(0, head_to_kv).unsqueeze(0)
    dk = k_delta.index_select(0, head_to_kv).unsqueeze(0)
    result = ql * kl * cs + dq * kl * cq + ql * dk * ck + dq * dk * cqk
    return result.sum(-1) / (head_dim ** 0.5)


def reference_2mean_components(
    keys: torch.Tensor, *, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 reference K bits and per-token/per-32D low/delta metadata."""

    if keys.ndim != 3:
        raise ValueError("keys must be [KVH, T, D]")
    values = keys.float()
    kv_heads, tokens, head_dim = values.shape
    token_groups = (tokens + group_size - 1) // group_size
    padded_tokens = token_groups * group_size
    if padded_tokens != tokens:
        values_for_extrema = torch.cat(
            [values, values.new_full((kv_heads, padded_tokens - tokens, head_dim), float("nan"))],
            dim=1,
        )
    else:
        values_for_extrema = values
    grouped = values_for_extrema.view(kv_heads, token_groups, group_size, head_dim)
    minimum = torch.nan_to_num(grouped, nan=float("inf")).amin(dim=2)
    maximum = torch.nan_to_num(grouped, nan=-float("inf")).amax(dim=2)
    group_ids = torch.arange(tokens, device=keys.device) // group_size
    minimum = minimum[:, group_ids]
    maximum = maximum[:, group_ids]
    sign = values >= 0
    magnitude = torch.where(
        sign, values > maximum * 0.5, values < minimum * 0.5
    )
    words = (head_dim + 31) // 32
    padded_dim = words * 32
    pad = padded_dim - head_dim
    abs_values = values.abs()
    valid = torch.ones_like(sign)
    if pad:
        shape = (kv_heads, tokens, pad)
        sign = torch.cat([sign, torch.zeros(shape, dtype=torch.bool, device=keys.device)], dim=-1)
        magnitude = torch.cat([magnitude, torch.zeros(shape, dtype=torch.bool, device=keys.device)], dim=-1)
        valid = torch.cat([valid, torch.zeros(shape, dtype=torch.bool, device=keys.device)], dim=-1)
        abs_values = torch.cat([abs_values, torch.zeros(shape, dtype=values.dtype, device=keys.device)], dim=-1)
    shape4 = (kv_heads, tokens, words, 32)
    mag4 = magnitude.view(shape4)
    valid4 = valid.view(shape4)
    abs4 = abs_values.view(shape4)
    low_mask = (~mag4) & valid4
    high_mask = mag4 & valid4
    low_count = low_mask.sum(-1)
    high_count = high_mask.sum(-1)
    low = (abs4 * low_mask).sum(-1) / low_count.clamp_min(1)
    high = (abs4 * high_mask).sum(-1) / high_count.clamp_min(1)
    low = torch.where(low_count > 0, low, high)
    high = torch.where(high_count > 0, high, low)
    return sign[..., :head_dim], magnitude[..., :head_dim], low, high - low


def reference_2mean_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Direct FP32 implementation of sign_product * weighted magnitude."""

    k_sign, k_mag, low, delta = reference_2mean_components(
        keys, group_size=group_size
    )
    q = queries.float()
    q_min = q.amin(dim=-1, keepdim=True)
    q_max = q.amax(dim=-1, keepdim=True)
    q_sign = q >= 0
    q_mag = torch.where(q_sign, q > q_max * 0.5, q < q_min * 0.5)
    mapped_sign = k_sign.index_select(0, head_to_kv).unsqueeze(0)
    mapped_mag = k_mag.index_select(0, head_to_kv).unsqueeze(0)
    words = low.shape[-1]
    expanded_low = low.index_select(0, head_to_kv).repeat_interleave(32, dim=-1)[..., : q.shape[-1]].unsqueeze(0)
    expanded_delta = delta.index_select(0, head_to_kv).repeat_interleave(32, dim=-1)[..., : q.shape[-1]].unsqueeze(0)
    weight = (1 + q_mag.to(torch.float32)).unsqueeze(2) * (
        expanded_low + expanded_delta * mapped_mag.to(torch.float32)
    )
    sign_product = torch.where(
        q_sign.unsqueeze(2) == mapped_sign, 1.0, -1.0
    )
    return (sign_product * weight).sum(dim=-1)


def reference_2mean_count_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """FP32 reference for L(Cs+Cq) + (H-L)(Ck+Cqk)."""
    k_sign, k_mag, low, delta = reference_2mean_components(keys, group_size=group_size)
    q = queries.float()
    q_min, q_max = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    q_sign = q >= 0
    q_mag = torch.where(q_sign, q > q_max * 0.5, q < q_min * 0.5)
    ks = k_sign.index_select(0, head_to_kv).unsqueeze(0)
    km = k_mag.index_select(0, head_to_kv).unsqueeze(0)
    batch, q_heads, tokens, head_dim = ks.shape
    words = low.shape[-1]
    pad = words * 32 - head_dim
    qs = q_sign.unsqueeze(2).expand(batch, q_heads, tokens, head_dim)
    qm = q_mag.unsqueeze(2).expand_as(qs)
    match = qs == ks
    valid = torch.ones_like(match)
    if pad:
        zeros = torch.zeros((*match.shape[:-1], pad), device=match.device, dtype=torch.bool)
        match = torch.cat([match, zeros], -1)
        qm = torch.cat([qm, zeros], -1)
        km = torch.cat([km, zeros], -1)
        valid = torch.cat([valid, zeros], -1)
    shape = (batch, q_heads, tokens, words, 32)
    match, qm, km, valid = (x.view(shape) for x in (match, qm, km, valid))
    def signed(mask: torch.Tensor) -> torch.Tensor:
        return 2 * (match & mask).sum(-1) - mask.sum(-1)
    c_sign = signed(valid)
    c_q = signed(qm)
    c_k = signed(km)
    c_both = signed(qm & km)
    l = low.index_select(0, head_to_kv).unsqueeze(0)
    d = delta.index_select(0, head_to_kv).unsqueeze(0)
    return (l * (c_sign + c_q) + d * (c_k + c_both)).sum(-1)


def reconstruct_keys_from_group_means(
    keys: torch.Tensor, *, group_size: int, mean_mode: str
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Reconstruct K from per-group/channel 2-bit category means.

    `signed4` stores signed means for +low/+high/-low/-high. `abs2`
    stores pooled absolute low/high means and reuses the sign bit.
    """

    if keys.ndim != 3:
        raise ValueError("keys must be [KVH, T, D]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if mean_mode not in {"signed4", "abs2"}:
        raise ValueError("mean_mode must be signed4 or abs2")

    kv_heads, tokens, head_dim = (int(value) for value in keys.shape)
    num_groups = (tokens + group_size - 1) // group_size
    pad = num_groups * group_size - tokens
    values = keys.float()
    if pad:
        values = torch.cat(
            [values, values.new_zeros(kv_heads, pad, head_dim)], dim=1
        )
    grouped = values.view(kv_heads, num_groups, group_size, head_dim)
    valid = (
        torch.arange(num_groups * group_size, device=keys.device)
        .view(1, num_groups, group_size, 1)
        .lt(tokens)
    )
    positive_inf = torch.full((), torch.inf, device=keys.device)
    negative_inf = torch.full((), -torch.inf, device=keys.device)
    minimum = torch.where(valid, grouped, positive_inf).amin(dim=2)
    maximum = torch.where(valid, grouped, negative_inf).amax(dim=2)
    sign = grouped >= 0
    high = torch.where(
        sign,
        grouped > maximum.unsqueeze(2) * 0.5,
        grouped < minimum.unsqueeze(2) * 0.5,
    )

    def category_mean(mask: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask & valid
        count = mask.sum(dim=2)
        total = torch.where(mask, source, 0.0).sum(dim=2)
        return total / count.clamp_min(1), count

    abs_values = grouped.abs()
    low_abs, low_count = category_mean(~high, abs_values)
    high_abs, high_count = category_mean(high, abs_values)
    low_abs = torch.where(low_count > 0, low_abs, torch.zeros_like(low_abs))
    high_abs = torch.where(high_count > 0, high_abs, low_abs)

    if mean_mode == "abs2":
        magnitude = torch.where(
            high, high_abs.unsqueeze(2), low_abs.unsqueeze(2)
        )
        reconstructed = torch.where(sign, magnitude, -magnitude)
        means = (low_abs, high_abs)
    else:
        positive_low, positive_low_count = category_mean(sign & ~high, grouped)
        positive_high, positive_high_count = category_mean(sign & high, grouped)
        negative_low, negative_low_count = category_mean(~sign & ~high, grouped)
        negative_high, negative_high_count = category_mean(~sign & high, grouped)
        positive_low = torch.where(
            positive_low_count > 0, positive_low, low_abs
        )
        positive_high = torch.where(
            positive_high_count > 0, positive_high, high_abs
        )
        negative_low = torch.where(
            negative_low_count > 0, negative_low, -low_abs
        )
        negative_high = torch.where(
            negative_high_count > 0, negative_high, -high_abs
        )
        reconstructed = torch.where(
            sign,
            torch.where(
                high, positive_high.unsqueeze(2), positive_low.unsqueeze(2)
            ),
            torch.where(
                high, negative_high.unsqueeze(2), negative_low.unsqueeze(2)
            ),
        )
        means = (
            positive_low,
            positive_high,
            negative_low,
            negative_high,
        )

    reconstructed = reconstructed.view(
        kv_heads, num_groups * group_size, head_dim
    )[:, :tokens]
    return reconstructed, means


def score_group_mean_tensors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    valid_tokens: torch.Tensor | None = None,
    group_size: int,
    mean_mode: str,
) -> torch.Tensor:
    """Score full-precision Q against K reconstructed from group means."""

    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("queries must be [B, QH, D] and keys must be [KVH, T, D]")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query/key head_dim mismatch")
    reconstructed, _ = reconstruct_keys_from_group_means(
        keys, group_size=group_size, mean_mode=mean_mode
    )
    batch, query_heads, _ = queries.shape
    tokens = int(keys.shape[1])
    scores = torch.empty(
        batch, query_heads, tokens, device=queries.device, dtype=torch.float32
    )
    for kv_head in range(int(keys.shape[0])):
        query_indices = torch.nonzero(
            head_to_kv == kv_head, as_tuple=False
        ).flatten()
        if query_indices.numel() == 0:
            continue
        scores[:, query_indices] = torch.einsum(
            "bhd,td->bht",
            queries[:, query_indices].float(),
            reconstructed[kv_head],
        )
    if valid_tokens is not None:
        valid_tokens = valid_tokens.to(device=scores.device, dtype=torch.long)
        if valid_tokens.numel() == 1:
            valid_tokens = valid_tokens.expand(batch)
        positions = torch.arange(tokens, device=scores.device)
        scores.masked_fill_(
            positions.view(1, 1, -1) >= valid_tokens.view(-1, 1, 1),
            -torch.inf,
        )
    return scores


def score_packed_batched(
    q_sign: torch.Tensor,
    q_mag: torch.Tensor,
    q_mag_count: torch.Tensor,
    k_sign: torch.Tensor,
    k_mag: torch.Tensor,
    k_mag_count: torch.Tensor,
    head_to_kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    return _extension().score_packed_batched(
        q_sign.contiguous(),
        q_mag.contiguous(),
        q_mag_count.contiguous(),
        k_sign.contiguous(),
        k_mag.contiguous(),
        k_mag_count.contiguous(),
        head_to_kv.to(device=q_sign.device, dtype=torch.long).contiguous(),
        valid_tokens.to(device=q_sign.device, dtype=torch.long).contiguous(),
        int(head_dim),
    )


def histogram_topk_from_scores(
    scores: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    budget: int,
    head_dim: int,
) -> torch.Tensor:
    return _extension().histogram_topk_from_scores(
        scores.contiguous(),
        valid_tokens.to(device=scores.device, dtype=torch.long).contiguous(),
        int(budget),
        int(head_dim),
    )


def score_tensors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    valid_tokens: torch.Tensor | None = None,
    group_size: int,
) -> torch.Tensor:
    """Quantize Q/K and score all Q heads in one CUDA POPC launch.

    queries: [B, QH, D], keys: [KVH, T, D].
    Returns int32 scores [B, QH, T].
    """

    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("queries must be [B, QH, D] and keys must be [KVH, T, D]")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query/key head_dim mismatch")
    if valid_tokens is None:
        valid_tokens = torch.tensor([keys.shape[1]], device=queries.device, dtype=torch.long)
    q_sign, q_mag, q_mag_count = pack_query(queries)
    k_sign, k_mag, k_mag_count = pack_keys(keys, group_size=group_size)
    return score_packed_batched(
        q_sign,
        q_mag,
        q_mag_count,
        k_sign,
        k_mag,
        k_mag_count,
        head_to_kv,
        valid_tokens,
        int(queries.shape[-1]),
    )
