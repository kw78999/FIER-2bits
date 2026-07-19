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
