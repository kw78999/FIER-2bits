from __future__ import annotations

"""Triton kernels for FIER's group-wise 1-bit key retrieval.

The layout follows the FIER authors' public implementation: keys are grouped
along the token axis, quantized with per-channel min/max metadata, and packed
32-to-1 before a fused dequantization/GEMV kernel.  This implementation adds
partial-group handling and GQA head mapping needed by this evaluator.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _group_minmax_kernel(
    keys,
    mins,
    maxs,
    tokens,
    head_dim: tl.constexpr,
    num_groups,
    group_size: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    d = row % head_dim
    group = (row // head_dim) % num_groups
    kvh = row // (head_dim * num_groups)
    offsets = tl.arange(0, BLOCK_GROUP)
    token = group * group_size + offsets
    mask = (offsets < group_size) & (token < tokens)
    values = tl.load(
        keys + (kvh * tokens + token) * head_dim + d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    lo = tl.min(tl.where(mask, values, float("inf")), axis=0)
    hi = tl.max(tl.where(mask, values, -float("inf")), axis=0)
    metadata_offset = (kvh * num_groups + group) * head_dim + d
    tl.store(mins + metadata_offset, lo)
    tl.store(maxs + metadata_offset, hi)


@triton.jit
def _pack_kernel(
    keys,
    mins,
    maxs,
    packed,
    tokens,
    head_dim: tl.constexpr,
    num_groups,
    words,
    group_size: tl.constexpr,
):
    row = tl.program_id(0)
    word = row % words
    d = (row // words) % head_dim
    kvh = row // (words * head_dim)
    lane = tl.arange(0, 32)
    token = word * 32 + lane
    group = token // group_size
    metadata_offset = (kvh * num_groups + group) * head_dim + d
    lo = tl.load(mins + metadata_offset, mask=token < tokens, other=0.0)
    hi = tl.load(maxs + metadata_offset, mask=token < tokens, other=0.0)
    value = tl.load(
        keys + (kvh * tokens + token) * head_dim + d,
        mask=token < tokens,
        other=0.0,
    ).to(tl.float32)
    bit = (value >= (lo + hi) * 0.5) & (token < tokens)
    # Accumulate in int64 so lane 31 does not overflow during the reduction;
    # storing to int32 preserves the desired bit pattern.
    word_value = tl.sum(bit.to(tl.int64) << lane, axis=0)
    tl.store(packed + (kvh * head_dim + d) * words + word, word_value)


@triton.jit
def _score_kernel(
    queries,
    packed,
    mins,
    maxs,
    head_to_kv,
    scores,
    tokens,
    q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups,
    words,
    group_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_block = tl.program_id(0)
    row = tl.program_id(1)
    qh = row % q_heads
    kvh = tl.load(head_to_kv + qh)
    token = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    token_mask = token < tokens
    dim_mask = dims < head_dim

    query = tl.load(
        queries + row * head_dim + dims,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    words_value = tl.load(
        packed + (kvh * head_dim + dims[None, :]) * words + token[:, None] // 32,
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0,
    )
    bit = (words_value >> (token[:, None] % 32)) & 1
    group = token // group_size
    metadata_offset = (
        (kvh * num_groups + group[:, None]) * head_dim + dims[None, :]
    )
    lo = tl.load(
        mins + metadata_offset,
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    hi = tl.load(
        maxs + metadata_offset,
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dequantized = tl.where(bit != 0, hi, lo)
    result = tl.sum(dequantized * query[None, :], axis=1)
    tl.store(scores + row * tokens + token, result, mask=token_mask)


def _validate_keys(keys: torch.Tensor, group_size: int) -> None:
    if keys.ndim != 3:
        raise ValueError("keys must be [KVH, T, D]")
    if not keys.is_cuda:
        raise RuntimeError("FIER Triton backend requires CUDA tensors")
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    if keys.shape[1] <= 0 or keys.shape[2] <= 0:
        raise ValueError("keys must have non-empty token and head dimensions")
    if group_size <= 0 or group_size > 1024:
        raise ValueError("group_size must be in [1, 1024]")


def pack_keys(
    keys: torch.Tensor, *, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack [KVH, T, D] keys and return (bits, group_min, group_max)."""

    keys = keys.contiguous()
    _validate_keys(keys, int(group_size))
    kv_heads, tokens, head_dim = (int(v) for v in keys.shape)
    num_groups = math.ceil(tokens / group_size)
    words = math.ceil(tokens / 32)
    metadata_shape = (kv_heads, num_groups, head_dim)
    # Match the authors’ cache layout: min/max metadata uses the source key
    # dtype. Group extrema are source values, so this does not add quantization error.
    mins = torch.empty(metadata_shape, device=keys.device, dtype=keys.dtype)
    maxs = torch.empty_like(mins)
    packed = torch.empty(
        (kv_heads, head_dim, words), device=keys.device, dtype=torch.int32
    )
    block_group = triton.next_power_of_2(group_size)
    with torch.cuda.device(keys.device):
        _group_minmax_kernel[(kv_heads * num_groups * head_dim,)](
            keys,
            mins,
            maxs,
            tokens=tokens,
            head_dim=head_dim,
            num_groups=num_groups,
            group_size=group_size,
            BLOCK_GROUP=block_group,
            num_warps=min(8, max(1, block_group // 32)),
        )
        _pack_kernel[(kv_heads * head_dim * words,)](
            keys,
            mins,
            maxs,
            packed,
            tokens=tokens,
            head_dim=head_dim,
            num_groups=num_groups,
            words=words,
            group_size=group_size,
            num_warps=1,
        )
    return packed, mins, maxs


def score_packed_batched(
    queries: torch.Tensor,
    packed: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
    head_to_kv: torch.Tensor,
    *,
    tokens: int,
    group_size: int,
) -> torch.Tensor:
    """Fused dequantization/GEMV for queries [B, QH, D]."""

    if queries.ndim != 3 or packed.ndim != 3:
        raise ValueError("queries and packed keys must be rank 3")
    if not queries.is_cuda or queries.device != packed.device:
        raise RuntimeError("all FIER Triton tensors must share a CUDA device")
    batch, q_heads, head_dim = (int(v) for v in queries.shape)
    if int(packed.shape[1]) != head_dim:
        raise ValueError("query/key head_dim mismatch")
    if mins.shape != maxs.shape or mins.ndim != 3:
        raise ValueError("group min/max tensors must have matching rank-3 shapes")
    if int(mins.shape[0]) != int(packed.shape[0]):
        raise ValueError("packed key and metadata KV-head counts differ")
    if tokens <= 0 or tokens > int(packed.shape[2]) * 32:
        raise ValueError("tokens is inconsistent with packed key storage")
    expected_groups = math.ceil(tokens / group_size)
    if int(mins.shape[1]) < expected_groups or int(mins.shape[2]) != head_dim:
        raise ValueError("metadata capacity is inconsistent with tokens/group_size")
    storage_groups = int(mins.shape[1])
    storage_words = int(packed.shape[2])
    if head_to_kv.numel() != q_heads:
        raise ValueError("head_to_kv must contain one entry per query head")

    queries = queries.contiguous()
    head_to_kv = head_to_kv.to(
        device=queries.device, dtype=torch.int64
    ).contiguous()
    scores = torch.empty(
        (batch, q_heads, tokens), device=queries.device, dtype=torch.float32
    )
    block_d = triton.next_power_of_2(head_dim)
    block_t = 32
    with torch.cuda.device(queries.device):
        _score_kernel[(triton.cdiv(tokens, block_t), batch * q_heads)](
            queries,
            packed,
            mins,
            maxs,
            head_to_kv,
            scores,
            tokens=tokens,
            q_heads=q_heads,
            head_dim=head_dim,
            num_groups=storage_groups,
            words=storage_words,
            group_size=group_size,
            BLOCK_D=block_d,
            BLOCK_T=block_t,
            num_warps=4,
        )
    return scores


def score_tensors(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    head_to_kv: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    packed, mins, maxs = pack_keys(keys, group_size=group_size)
    return score_packed_batched(
        queries,
        packed,
        mins,
        maxs,
        head_to_kv,
        tokens=int(keys.shape[1]),
        group_size=group_size,
    )
