from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _popcount32(x):
    x = x.to(tl.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _pack_q_2mean_kernel(
    queries, q_sign, q_mag, q_low, q_delta,
    head_dim: tl.constexpr, words: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    valid = dims < head_dim
    values = tl.load(queries + row * head_dim + dims, mask=valid, other=0.0).to(tl.float32)
    minimum = tl.min(tl.where(valid, values, float("inf")), axis=0)
    maximum = tl.max(tl.where(valid, values, -float("inf")), axis=0)
    sign = (values >= 0.0) & valid
    mag = tl.where(sign, values > maximum * 0.5, values < minimum * 0.5) & valid
    word = dims // 32
    lane = dims % 32
    for w in range(words):
        in_word = valid & (word == w)
        sw = tl.sum(tl.where(sign & in_word, 1, 0).to(tl.int64) << lane, axis=0)
        mw = tl.sum(tl.where(mag & in_word, 1, 0).to(tl.int64) << lane, axis=0)
        low_mask = in_word & ~mag
        high_mask = in_word & mag
        low_count = tl.sum(low_mask.to(tl.float32), axis=0)
        high_count = tl.sum(high_mask.to(tl.float32), axis=0)
        low_sum = tl.sum(tl.where(low_mask, tl.abs(values), 0.0), axis=0)
        high_sum = tl.sum(tl.where(high_mask, tl.abs(values), 0.0), axis=0)
        low_raw = low_sum / tl.maximum(low_count, 1.0)
        high_raw = high_sum / tl.maximum(high_count, 1.0)
        low = tl.where(low_count > 0, low_raw, high_raw)
        high = tl.where(high_count > 0, high_raw, low)
        tl.store(q_sign + row * words + w, sw)
        tl.store(q_mag + row * words + w, mw)
        tl.store(q_low + row * words + w, low)
        tl.store(q_delta + row * words + w, high - low)


@triton.jit
def _score_2mean_kernel(
    q_sign, q_mag, k_sign, k_mag, low_mean, delta_mean,
    head_to_kv, scores, tokens: tl.constexpr, capacity: tl.constexpr,
    q_heads: tl.constexpr, words: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_W: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row = tl.program_id(1)
    qh = row % q_heads
    kvh = tl.load(head_to_kv + qh)
    word = tl.arange(0, BLOCK_W)
    token_mask = token < tokens
    word_mask = word < words
    qbase = row * words + word
    kbase = (kvh * capacity + token[:, None]) * words + word[None, :]
    qs = tl.load(q_sign + qbase, mask=word_mask, other=0).to(tl.uint32)
    qm = tl.load(q_mag + qbase, mask=word_mask, other=0).to(tl.uint32)
    ks = tl.load(k_sign + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0).to(tl.uint32)
    km = tl.load(k_mag + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0).to(tl.uint32)
    valid_bits = head_dim - word * 32
    valid_bits = tl.minimum(tl.maximum(valid_bits, 0), 32)
    valid_mask = tl.where(valid_bits == 32, 0xFFFFFFFF, (1 << valid_bits) - 1).to(tl.uint32)
    qs = qs & valid_mask
    qm = qm & valid_mask
    ks = ks & valid_mask[None, :]
    km = km & valid_mask[None, :]
    match = ~(qs[None, :] ^ ks) & valid_mask[None, :]
    both = qm[None, :] & km
    c_sign = 2 * _popcount32(match) - valid_bits[None, :]
    c_q = 2 * _popcount32(match & qm[None, :]) - _popcount32(qm)[None, :]
    c_k = 2 * _popcount32(match & km) - _popcount32(km)
    c_both = 2 * _popcount32(match & both) - _popcount32(both)
    low = tl.load(low_mean + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0.0).to(tl.float32)
    delta = tl.load(delta_mean + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0.0).to(tl.float32)
    result = tl.sum(low * (c_sign + c_q) + delta * (c_k + c_both), axis=1)
    tl.store(scores + row * tokens + token, result, mask=token_mask)


@triton.jit
def _score_qk_2mean_kernel(
    q_sign, q_mag, q_low, q_delta, k_sign, k_mag, k_low, k_delta,
    head_to_kv, scores, tokens: tl.constexpr, capacity: tl.constexpr,
    q_heads: tl.constexpr, words: tl.constexpr, head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_W: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row = tl.program_id(1)
    qh = row % q_heads
    kvh = tl.load(head_to_kv + qh)
    word = tl.arange(0, BLOCK_W)
    token_mask, word_mask = token < tokens, word < words
    qbase = row * words + word
    kbase = (kvh * capacity + token[:, None]) * words + word[None, :]
    qs = tl.load(q_sign + qbase, mask=word_mask, other=0).to(tl.uint32)
    qm = tl.load(q_mag + qbase, mask=word_mask, other=0).to(tl.uint32)
    ql = tl.load(q_low + qbase, mask=word_mask, other=0.0).to(tl.float32)
    dq = tl.load(q_delta + qbase, mask=word_mask, other=0.0).to(tl.float32)
    ks = tl.load(k_sign + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0).to(tl.uint32)
    km = tl.load(k_mag + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0).to(tl.uint32)
    kl = tl.load(k_low + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0.0).to(tl.float32)
    dk = tl.load(k_delta + kbase, mask=token_mask[:, None] & word_mask[None, :], other=0.0).to(tl.float32)
    valid_bits = tl.minimum(tl.maximum(head_dim - word * 32, 0), 32)
    valid_mask = tl.where(valid_bits == 32, 0xFFFFFFFF, (1 << valid_bits) - 1).to(tl.uint32)
    qs, qm = qs & valid_mask, qm & valid_mask
    ks, km = ks & valid_mask[None, :], km & valid_mask[None, :]
    match = ~(qs[None, :] ^ ks) & valid_mask[None, :]
    both = qm[None, :] & km
    cs = 2 * _popcount32(match) - valid_bits[None, :]
    cq = 2 * _popcount32(match & qm[None, :]) - _popcount32(qm)[None, :]
    ck = 2 * _popcount32(match & km) - _popcount32(km)
    cqk = 2 * _popcount32(match & both) - _popcount32(both)
    group_score = ql[None, :] * kl * cs + dq[None, :] * kl * cq + ql[None, :] * dk * ck + dq[None, :] * dk * cqk
    result = tl.sum(group_score, axis=1) * scale
    tl.store(scores + row * tokens + token, result, mask=token_mask)


def score_2mean_triton_packed(
    queries: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    head_to_kv: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    """Triton weighted-popcount scorer; K remains packed throughout."""
    from bit2_cuda import pack_query

    if queries.ndim != 3 or not queries.is_cuda:
        raise ValueError("queries must be CUDA [B, QH, D]")
    k_sign, k_mag, low_mean, delta_mean = packed
    q_sign, q_mag, _ = pack_query(queries.contiguous())
    batch, q_heads, head_dim = map(int, queries.shape)
    words = int(k_sign.shape[-1])
    capacity = int(k_sign.shape[1])
    if tokens <= 0 or tokens > capacity:
        raise ValueError("invalid token count")
    mapping = head_to_kv.to(device=queries.device, dtype=torch.int64).contiguous()
    scores = torch.empty((batch, q_heads, tokens), device=queries.device, dtype=torch.float32)
    block_t = 64
    block_w = triton.next_power_of_2(words)
    with torch.cuda.device(queries.device):
        _score_2mean_kernel[(triton.cdiv(tokens, block_t), batch * q_heads)](
            q_sign, q_mag, k_sign, k_mag, low_mean, delta_mean,
            mapping, scores, tokens=tokens, capacity=capacity,
            q_heads=q_heads, words=words, head_dim=head_dim,
            BLOCK_T=block_t, BLOCK_W=block_w, num_warps=4,
        )
    return scores


def score_qk_2mean_triton_packed(
    queries: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *, head_to_kv: torch.Tensor, tokens: int,
) -> torch.Tensor:
    """Triton Q pack/mean plus fused Q/K weighted-popcount scoring."""
    if queries.ndim != 3 or not queries.is_cuda:
        raise ValueError("queries must be CUDA [B, QH, D]")
    queries = queries.contiguous()
    k_sign, k_mag, k_low, k_delta = packed
    batch, q_heads, head_dim = map(int, queries.shape)
    words, capacity = int(k_sign.shape[-1]), int(k_sign.shape[1])
    if tokens <= 0 or tokens > capacity:
        raise ValueError("invalid token count")
    mapping = head_to_kv.to(device=queries.device, dtype=torch.int64).contiguous()
    packed_shape = (batch, q_heads, words)
    q_sign = torch.empty(packed_shape, device=queries.device, dtype=torch.int32)
    q_mag = torch.empty_like(q_sign)
    q_low = torch.empty(packed_shape, device=queries.device, dtype=torch.float32)
    q_delta = torch.empty_like(q_low)
    scores = torch.empty((batch, q_heads, tokens), device=queries.device, dtype=torch.float32)
    block_d, block_w, block_t = triton.next_power_of_2(head_dim), triton.next_power_of_2(words), 64
    with torch.cuda.device(queries.device):
        _pack_q_2mean_kernel[(batch * q_heads,)](
            queries, q_sign, q_mag, q_low, q_delta,
            head_dim=head_dim, words=words, BLOCK_D=block_d, num_warps=1,
        )
        _score_qk_2mean_kernel[(triton.cdiv(tokens, block_t), batch * q_heads)](
            q_sign, q_mag, q_low, q_delta, k_sign, k_mag, k_low, k_delta,
            mapping, scores, tokens=tokens, capacity=capacity,
            q_heads=q_heads, words=words, head_dim=head_dim,
            scale=head_dim ** -0.5,
            BLOCK_T=block_t, BLOCK_W=block_w, num_warps=4,
        )
    return scores
