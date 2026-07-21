from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("Pop-KV CUDA scoring requires an NVIDIA GPU")
    return load(
        name="popkv_ext",
        sources=[str(_ROOT / "csrc" / "popkv_ext.cpp"), str(_ROOT / "csrc" / "popkv_ext.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def pack_keys(keys: torch.Tensor, *, token_capacity: int | None = None):
    """Pack K sign/magnitude plus FP16 low/delta representatives (group size 32)."""
    if keys.ndim != 3 or not keys.is_cuda or not keys.is_contiguous():
        raise ValueError("keys must be contiguous CUDA [KV heads, tokens, head dim]")
    capacity = int(keys.shape[1]) if token_capacity is None else int(token_capacity)
    return tuple(_extension().pack(keys, capacity))


def pack_keys_into(keys: torch.Tensor, packed, *, token_offset: int) -> None:
    if keys.ndim != 3 or not keys.is_cuda or not keys.is_contiguous():
        raise ValueError("keys must be contiguous CUDA [KV heads, tokens, head dim]")
    _extension().pack_into(keys, *packed, int(token_offset))


def score(queries: torch.Tensor, packed, *, head_to_kv: torch.Tensor, tokens: int) -> torch.Tensor:
    """Return Pop-KV approximate QK scores for every query head and token."""
    if queries.ndim != 3 or not queries.is_cuda or not queries.is_contiguous():
        raise ValueError("queries must be contiguous CUDA [batch, query heads, head dim]")
    return _extension().score(queries, *packed, head_to_kv.contiguous(), int(tokens))
