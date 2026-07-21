from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import torch
import torch.nn.functional as F


SUPPORTED_METHODS = ("full", "pqsift", "loki", "quest", "fier", "bit2_qk")
T = TypeVar("T")
_POPCOUNT_LUT_BY_DEVICE: dict[str, torch.Tensor] = {}


@dataclass(frozen=True)
class PCABasis:
    mean: torch.Tensor
    components: torch.Tensor
    eigvals: torch.Tensor | None = None
    total_variance: float | None = None


class PCABasisCache:
    """Loader for the metadata-rich cache produced by PQ_SIFT.ipynb."""

    def __init__(
        self,
        path: Path,
        metadata: dict[str, Any],
        bases: dict[tuple[int, int], PCABasis],
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.bases = bases
        self._device_cache: dict[tuple[int, int, str, torch.dtype], PCABasis] = {}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model_id: str | None = None,
        min_axes: int = 1,
        expected_layers: int | None = None,
        expected_heads: int | None = None,
    ) -> "PCABasisCache":
        cache_path = Path(path).expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"PCA basis cache not found: {cache_path}")

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")

        if not isinstance(payload, dict) or not isinstance(payload.get("basis_by_pair"), dict):
            raise ValueError("Expected a pca_basis_cache_v2 payload containing basis_by_pair")

        metadata = dict(payload.get("metadata", {}))
        cache_model_id = metadata.get("model_id")
        if expected_model_id and cache_model_id != expected_model_id:
            raise ValueError(
                f"PCA cache model mismatch: cache={cache_model_id!r}, requested={expected_model_id!r}"
            )
        cache_version = metadata.get("cache_version")
        if cache_version not in {None, "pca_basis_cache_v2"}:
            raise ValueError(f"Unsupported PCA cache version: {cache_version!r}")

        bases: dict[tuple[int, int], PCABasis] = {}
        for raw_pair, item in payload["basis_by_pair"].items():
            if not isinstance(raw_pair, str) or ":" not in raw_pair:
                raise ValueError(f"Invalid layer/head key in PCA cache: {raw_pair!r}")
            layer_idx, head_idx = (int(part) for part in raw_pair.split(":", maxsplit=1))
            mean = item["mean"].detach().cpu().float().contiguous()
            components = item["components"].detach().cpu().float().contiguous()
            if mean.ndim != 1 or components.ndim != 2:
                raise ValueError(f"Invalid basis shape for pair {raw_pair}")
            if components.shape[0] != mean.shape[0]:
                raise ValueError(f"Mean/component dimension mismatch for pair {raw_pair}")
            if components.shape[1] < min_axes:
                raise ValueError(
                    f"Pair {raw_pair} has {components.shape[1]} PCA axes; {min_axes} required"
                )
            eigvals = item.get("eigvals")
            bases[(layer_idx, head_idx)] = PCABasis(
                mean=mean,
                components=components,
                eigvals=None if eigvals is None else eigvals.detach().cpu().float().contiguous(),
                total_variance=(
                    None if item.get("total_variance") is None else float(item["total_variance"])
                ),
            )

        if expected_layers is not None and expected_heads is not None:
            expected = {
                (layer_idx, head_idx)
                for layer_idx in range(expected_layers)
                for head_idx in range(expected_heads)
            }
            missing = sorted(expected.difference(bases))
            if missing:
                raise ValueError(
                    f"PCA cache is missing {len(missing)} layer/head pairs; first={missing[:4]}"
                )

        return cls(cache_path, metadata, bases)

    def get(
        self,
        layer_idx: int,
        head_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> PCABasis:
        source = self.bases.get((int(layer_idx), int(head_idx)))
        if source is None:
            raise KeyError(f"No PCA basis for layer={layer_idx}, head={head_idx}")
        key = (int(layer_idx), int(head_idx), str(device), dtype)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = PCABasis(
                mean=source.mean.to(device=device, dtype=dtype),
                components=source.components.to(device=device, dtype=dtype),
                eigvals=(
                    None
                    if source.eigvals is None
                    else source.eigvals.to(device=device, dtype=dtype)
                ),
                total_variance=source.total_variance,
            )
            self._device_cache[key] = cached
        return cached


@dataclass(frozen=True)
class Bit2Thresholds:
    neg: torch.Tensor
    pos: torch.Tensor


@dataclass
class PackedFIERHeadCache:
    group_size: int
    sealed_bits: torch.Tensor
    group_min: torch.Tensor
    group_max: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class PackedBit2HeadCache:
    group_size: int
    dimension: int
    sealed_sign: torch.Tensor
    sealed_magnitude: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class PackedBit2MeanLayerCache:
    group_size: int
    token_count: int
    sealed_tokens: int
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    active_keys: torch.Tensor


@dataclass
class PackedBit2CudaLayerCache:
    group_size: int
    token_count: int
    sealed_tokens: int
    packed: tuple[torch.Tensor, torch.Tensor]
    active_keys: torch.Tensor


@dataclass
class PackedFIERTritonLayerCache:
    group_size: int
    token_count: int
    sealed_tokens: int
    packed: torch.Tensor
    group_min: torch.Tensor
    group_max: torch.Tensor
    active_keys: torch.Tensor


@dataclass
class StaticKVCache:
    key_storage: list[torch.Tensor]
    value_storage: list[torch.Tensor]
    length: int
    capacity: int


@dataclass
class PackedRetrievalCaches:
    method: str
    caches: dict[
        tuple[int, int],
        PackedFIERHeadCache | PackedFIERTritonLayerCache | PackedBit2HeadCache | PackedBit2MeanLayerCache | PackedBit2CudaLayerCache,
    ]


@dataclass(frozen=True)
class RetrievalConfig:
    budget: int = 512
    loki_rank: int = 64
    pqsift_axes: int = 4
    pqsift_keep_ratio: float = 0.75
    quest_page_size: int = 16
    fier_group_size: int = 32
    fier_backend: str = "reference"
    bit2_group_size: int = 64
    bit2_backend: str = "reference"
    full_layers: tuple[int, ...] = (0, 1)
    measure_topk_recall: bool = False
    measure_score_diagnostics: bool = False

    def validate(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.loki_rank <= 0:
            raise ValueError("loki_rank must be positive")
        if self.pqsift_axes <= 0:
            raise ValueError("pqsift_axes must be positive")
        if not 0.0 < self.pqsift_keep_ratio <= 1.0:
            raise ValueError("pqsift_keep_ratio must be in (0, 1]")
        if self.quest_page_size <= 0:
            raise ValueError("quest_page_size must be positive")
        if self.fier_group_size <= 0:
            raise ValueError("fier_group_size must be positive")
        if self.fier_backend not in {"reference", "triton"}:
            raise ValueError("fier_backend must be reference or triton")
        if self.bit2_group_size <= 0:
            raise ValueError("bit2_group_size must be positive")
        if self.bit2_backend not in {
            "reference",
            "cuda_popc",
            "cuda_popc_direct",
            "cuda_popc_histogram",
            "group_mean4",
            "group_mean2",
            "cuda_2mean",
            "triton_2mean",
            "2mean",
            "qk_2mean",
            "cuda_qk_2mean",
            "triton_qk_2mean",
        }:
            raise ValueError(
                "bit2_backend must be reference, cuda_popc, cuda_popc_direct, cuda_popc_histogram, "
                "group_mean4, group_mean2, 2mean, cuda_2mean, triton_2mean, "
                "qk_2mean, cuda_qk_2mean, or triton_qk_2mean"
            )


@dataclass(frozen=True)
class EvaluationVariant:
    method: str
    label: str
    retrieval: RetrievalConfig


def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(max(1, int(k)), int(scores.numel()))
    return torch.topk(scores, k=k, largest=True, sorted=False).indices


def _include_current(indices: torch.Tensor, num_tokens: int, max_tokens: int | None) -> torch.Tensor:
    current = num_tokens - 1
    indices = torch.unique(indices.to(dtype=torch.long), sorted=False)
    if not bool(torch.any(indices == current)):
        if max_tokens is not None and indices.numel() >= max_tokens:
            indices = indices[: max(0, max_tokens - 1)]
        indices = torch.cat([indices, indices.new_tensor([current])])
    if max_tokens is not None and indices.numel() > max_tokens:
        keep = indices[:max_tokens].clone()
        if not bool(torch.any(keep == current)):
            keep[-1] = current
        indices = keep
    return torch.sort(torch.unique(indices)).values


def _append_current_reserved(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Append current token when Top-k already reserved exactly one slot."""
    current = torch.full(
        (*indices.shape[:-1], 1), num_tokens - 1,
        device=indices.device, dtype=torch.long,
    )
    return torch.cat([indices.to(dtype=torch.long), current], dim=-1)


def select_full(keys: torch.Tensor) -> torch.Tensor:
    return torch.arange(keys.shape[0], device=keys.device, dtype=torch.long)


def project_pca(
    query: torch.Tensor,
    keys: torch.Tensor,
    basis: PCABasis,
    axes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = min(int(axes), int(basis.components.shape[1]))
    components = basis.components[:, :rank]
    keys32 = keys.float()
    query32 = query.float()
    projected_keys = (keys32 - basis.mean) @ components
    projected_query = (query32 - basis.mean) @ components
    return projected_query, projected_keys


def select_loki(
    query: torch.Tensor,
    keys: torch.Tensor,
    basis: PCABasis,
    *,
    rank: int,
    budget: int,
) -> torch.Tensor:
    projected_query, projected_keys = project_pca(query, keys, basis, rank)
    indices = _topk_indices(projected_keys @ projected_query, budget)
    return _include_current(indices, keys.shape[0], min(budget, keys.shape[0]))


def select_pqsift(
    query: torch.Tensor,
    keys: torch.Tensor,
    basis: PCABasis,
    *,
    axes: int,
    keep_ratio: float,
) -> torch.Tensor:
    """PQ-SIFT sign/quantile bucket retrieval from the supplied notebook.

    Every PCA axis keeps its top-r fraction for a positive projected query
    coordinate or its bottom-r fraction for a negative coordinate. The final
    bucket is the intersection across axes.
    """

    projected_query, projected_keys = project_pca(query, keys, basis, axes)
    num_tokens = int(keys.shape[0])
    keep_count = min(num_tokens, max(1, int(math.ceil(keep_ratio * num_tokens))))
    mask = torch.ones(num_tokens, dtype=torch.bool, device=keys.device)
    for axis in range(projected_keys.shape[1]):
        largest = bool(projected_query[axis] >= 0)
        axis_ids = torch.topk(
            projected_keys[:, axis],
            k=keep_count,
            largest=largest,
            sorted=False,
        ).indices
        axis_mask = torch.zeros_like(mask)
        axis_mask[axis_ids] = True
        mask &= axis_mask
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    return _include_current(indices, num_tokens, None)


def quest_page_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Quest's per-page upper-bound score: sum_i max(q_i*kmin_i, q_i*kmax_i)."""

    num_tokens, dim = keys.shape
    num_pages = math.ceil(num_tokens / page_size)
    pad = num_pages * page_size - num_tokens
    keys32 = keys.float()
    if pad:
        keys32 = torch.cat([keys32, keys32[-1:].expand(pad, dim)], dim=0)
    pages = keys32.view(num_pages, page_size, dim)
    page_min = pages.amin(dim=1)
    page_max = pages.amax(dim=1)
    query32 = query.float().unsqueeze(0)
    return torch.maximum(query32 * page_min, query32 * page_max).sum(dim=-1)


def quest_page_scores_batched(
    queries: torch.Tensor, prefix_keys: torch.Tensor, current_keys: torch.Tensor,
    *, head_to_kv: torch.Tensor, page_size: int,
) -> torch.Tensor:
    """Quest page bounds for every Q head, computing page metadata once per KV head."""
    keys32 = torch.cat([prefix_keys, current_keys[:, None, :]], dim=1).float()
    kv_heads, num_tokens, dim = keys32.shape
    num_pages = math.ceil(num_tokens / page_size)
    pad = num_pages * page_size - num_tokens
    if pad:
        keys32 = torch.cat([keys32, keys32[:, -1:, :].expand(kv_heads, pad, dim)], dim=1)
    pages = keys32.view(kv_heads, num_pages, page_size, dim)
    page_min = pages.amin(dim=2).index_select(0, head_to_kv)
    page_max = pages.amax(dim=2).index_select(0, head_to_kv)
    query32 = queries.float()[:, None, :]
    return torch.maximum(query32 * page_min, query32 * page_max).sum(dim=-1)


def quest_topk_indices_batched(
    scores: torch.Tensor, *, num_tokens: int, page_size: int, budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_budget = min(int(scores.shape[-1]), max(1, math.ceil(budget / page_size)))
    pages = torch.topk(scores, k=page_budget, dim=-1, largest=True, sorted=False).indices
    offsets = torch.arange(page_size, device=scores.device)
    indices = (pages[:, :, None] * page_size + offsets).flatten(1)
    valid = indices.lt(num_tokens)
    current = num_tokens - 1
    has_current = (indices.eq(current) & valid).any(dim=-1)
    indices[:, -1] = torch.where(has_current, indices[:, -1], indices.new_tensor(current))
    valid[:, -1] = torch.where(has_current, valid[:, -1], valid.new_tensor(True))
    return indices, valid


def select_quest(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    page_size: int,
    budget: int,
) -> torch.Tensor:
    num_tokens = int(keys.shape[0])
    scores = quest_page_scores(query, keys, page_size=page_size)
    page_budget = min(scores.numel(), max(1, math.ceil(budget / page_size)))
    selected_pages = _topk_indices(scores, page_budget)
    offsets = torch.arange(page_size, device=keys.device)
    indices = (selected_pages[:, None] * page_size + offsets[None, :]).flatten()
    indices = indices[indices < num_tokens]
    # Quest is page-granular, so the actual count may be up to page_size above budget.
    return _include_current(indices, num_tokens, None)


def fier_dequantize_1bit(keys: torch.Tensor, *, group_size: int) -> torch.Tensor:
    """Reference 1-bit RTN quantizer used for FIER token ranking.

    Groups are formed along the token dimension independently per key channel.
    At one bit, min/max linear RTN reconstructs each value as its group minimum
    or maximum according to the midpoint threshold.
    """

    num_tokens, dim = keys.shape
    num_groups = math.ceil(num_tokens / group_size)
    pad = num_groups * group_size - num_tokens
    keys32 = keys.float()
    if pad:
        keys32 = torch.cat([keys32, keys32[-1:].expand(pad, dim)], dim=0)
    groups = keys32.view(num_groups, group_size, dim)
    group_min = groups.amin(dim=1, keepdim=True)
    group_max = groups.amax(dim=1, keepdim=True)
    midpoint = (group_min + group_max) * 0.5
    dequantized = torch.where(groups >= midpoint, group_max, group_min)
    return dequantized.view(num_groups * group_size, dim)[:num_tokens]


def select_fier(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    group_size: int,
    budget: int,
    backend: str = "reference",
) -> torch.Tensor:
    if backend == "reference":
        scores = fier_dequantize_1bit(keys, group_size=group_size) @ query.float()
    elif backend == "triton":
        if not query.is_cuda or not keys.is_cuda:
            raise RuntimeError("FIER Triton backend requires CUDA query and keys")
        from fier_triton import score_tensors

        scores = score_tensors(
            query.view(1, 1, -1),
            keys.view(1, keys.shape[0], keys.shape[1]).contiguous(),
            head_to_kv=torch.zeros(1, device=query.device, dtype=torch.long),
            group_size=group_size,
        )[0, 0]
    else:
        raise ValueError(f"Unknown FIER backend: {backend}")
    indices = _topk_indices(scores, budget)
    return _include_current(indices, keys.shape[0], min(budget, keys.shape[0]))


def minmax_half_thresholds(values: torch.Tensor, *, dim: int) -> Bit2Thresholds:
    """Use min/2 and max/2 as the negative/positive magnitude boundaries."""
    values32 = values.float()
    return Bit2Thresholds(
        neg=values32.amin(dim=dim) * 0.5,
        pos=values32.amax(dim=dim) * 0.5,
    )


def quantize_bit2(values: torch.Tensor, thresholds: Bit2Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unpacked sign and magnitude bits."""
    sign = values >= 0
    magnitude = torch.where(sign, values > thresholds.pos, values < thresholds.neg)
    return sign, magnitude


def quantize_query_bit2(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one query using min/2 and max/2 across head dimensions."""
    if query.ndim != 1:
        raise ValueError("Expected a single [head_dim] query")
    return quantize_bit2(query.float(), minmax_half_thresholds(query, dim=0))


def quantize_grouped_keys_bit2(
    keys: torch.Tensor, *, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K with per-group, per-channel min/2 and max/2 thresholds."""
    if keys.ndim != 2:
        raise ValueError("Expected keys shaped [tokens, head_dim]")
    num_tokens, head_dim = keys.shape
    num_groups = math.ceil(num_tokens / group_size)
    pad = num_groups * group_size - num_tokens
    keys32 = keys.float()
    if pad:
        keys32 = torch.cat([keys32, keys32[-1:].expand(pad, head_dim)], dim=0)
    groups = keys32.view(num_groups, group_size, head_dim)
    thresholds = minmax_half_thresholds(groups, dim=1)
    sign = groups >= 0
    magnitude = torch.where(
        sign,
        groups > thresholds.pos.unsqueeze(1),
        groups < thresholds.neg.unsqueeze(1),
    )
    return (
        sign.view(num_groups * group_size, head_dim)[:num_tokens],
        magnitude.view(num_groups * group_size, head_dim)[:num_tokens],
    )


def bit2_interaction_scores_from_bits(
    query_sign: torch.Tensor,
    query_magnitude: torch.Tensor,
    key_sign: torch.Tensor,
    key_magnitude: torch.Tensor,
) -> torch.Tensor:
    """Exact 3-POPC score, expressed on unpacked bool tensors.

    x = q_sign XOR k_sign; score = D + popc(q_mag) + popc(k_mag)
    - 2 * (popc(x) + popc(x & q_mag) + popc(x & k_mag)).
    This is algebraically identical to the older XNOR + OR/AND expression.
    """
    if query_sign.ndim != 1 or query_magnitude.shape != query_sign.shape:
        raise ValueError("Query sign/magnitude bits must be matching 1-D tensors")
    if key_sign.ndim != 2 or key_magnitude.shape != key_sign.shape:
        raise ValueError("Key sign/magnitude bits must be matching 2-D tensors")
    if key_sign.shape[1] != query_sign.numel():
        raise ValueError("Query/key bit dimensions do not match")
    x = torch.logical_xor(key_sign, query_sign.unsqueeze(0))
    q = query_magnitude.unsqueeze(0)
    k = key_magnitude
    dimension = query_sign.numel()
    return (
        dimension
        + query_magnitude.sum(dtype=torch.int32)
        + key_magnitude.sum(dim=-1, dtype=torch.int32)
        - 2
        * (
            x.sum(dim=-1, dtype=torch.int32)
            + (x & q).sum(dim=-1, dtype=torch.int32)
            + (x & k).sum(dim=-1, dtype=torch.int32)
        )
    )


def pack_bool_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack the final bool dimension eight-to-one into uint8 words."""
    if bits.dtype != torch.bool:
        raise ValueError("pack_bool_bits expects torch.bool input")
    dimension = bits.shape[-1]
    pad = (-dimension) % 8
    if pad:
        bits = F.pad(bits, (0, pad), value=False)
    weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128],
        dtype=torch.uint8,
        device=bits.device,
    )
    reshaped = bits.reshape(*bits.shape[:-1], -1, 8).to(torch.uint8)
    return (reshaped * weights).sum(dim=-1, dtype=torch.int16).to(torch.uint8)


def _popcount_uint8(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.uint8:
        raise ValueError("popcount input must be uint8")
    device_key = str(values.device)
    lut = _POPCOUNT_LUT_BY_DEVICE.get(device_key)
    if lut is None:
        lut = torch.tensor(
            [value.bit_count() for value in range(256)],
            dtype=torch.uint8,
            device=values.device,
        )
        _POPCOUNT_LUT_BY_DEVICE[device_key] = lut
    return lut[values.long()]


def bit2_interaction_scores_packed(
    query_sign: torch.Tensor,
    query_magnitude: torch.Tensor,
    key_sign: torch.Tensor,
    key_magnitude: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    """Exact 3-POPC score using packed uint8 bitplanes."""
    if query_sign.dtype != torch.uint8 or query_magnitude.dtype != torch.uint8:
        raise ValueError("Packed query bitplanes must be uint8")
    if key_sign.dtype != torch.uint8 or key_magnitude.dtype != torch.uint8:
        raise ValueError("Packed key bitplanes must be uint8")
    if query_sign.ndim != 1 or query_magnitude.shape != query_sign.shape:
        raise ValueError("Packed query bitplanes must be matching 1-D tensors")
    if key_sign.ndim != 2 or key_magnitude.shape != key_sign.shape:
        raise ValueError("Packed key bitplanes must be matching 2-D tensors")
    if key_sign.shape[1] != query_sign.numel():
        raise ValueError("Packed query/key word counts do not match")

    x = torch.bitwise_xor(key_sign, query_sign.unsqueeze(0))
    q = query_magnitude.unsqueeze(0)
    k = key_magnitude
    valid_bits = dimension % 8
    if valid_bits:
        mask = (1 << valid_bits) - 1
        x = x.clone()
        q = q.clone()
        k = k.clone()
        x[:, -1] = torch.bitwise_and(x[:, -1], mask)
        q[:, -1] = torch.bitwise_and(q[:, -1], mask)
        k[:, -1] = torch.bitwise_and(k[:, -1], mask)
    q_count = _popcount_uint8(q).sum(dtype=torch.int32)
    k_count = _popcount_uint8(k).sum(dim=-1, dtype=torch.int32)
    return (
        int(dimension)
        + q_count
        + k_count
        - 2
        * (
            _popcount_uint8(x).sum(dim=-1, dtype=torch.int32)
            + _popcount_uint8(torch.bitwise_and(x, q)).sum(dim=-1, dtype=torch.int32)
            + _popcount_uint8(torch.bitwise_and(x, k)).sum(dim=-1, dtype=torch.int32)
        )
    )


def bit2_approximate_scores(
    query: torch.Tensor, keys: torch.Tensor, *, group_size: int
) -> torch.Tensor:
    query_sign, query_magnitude = quantize_query_bit2(query)
    key_sign, key_magnitude = quantize_grouped_keys_bit2(keys, group_size=group_size)
    return bit2_interaction_scores_packed(
        pack_bool_bits(query_sign),
        pack_bool_bits(query_magnitude),
        pack_bool_bits(key_sign),
        pack_bool_bits(key_magnitude),
        dimension=query.numel(),
    )


def select_bit2_qk(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    group_size: int,
    budget: int,
    backend: str = "reference",
) -> torch.Tensor:
    if backend == "reference":
        indices = _topk_indices(
            bit2_approximate_scores(query, keys, group_size=group_size), budget
        )
        return _include_current(indices, keys.shape[0], min(budget, keys.shape[0]))
    if backend in {
        "cuda_popc",
        "cuda_popc_histogram",
        "group_mean4",
        "group_mean2",
    }:
        if not query.is_cuda or not keys.is_cuda:
            raise RuntimeError(f"{backend} requires CUDA query and keys")
        from bit2_cuda import (
            histogram_topk_from_scores,
            score_group_mean_tensors,
            score_tensors,
        )

        score_args = {
            "head_to_kv": torch.zeros(1, device=query.device, dtype=torch.long),
            "valid_tokens": torch.tensor(
                [keys.shape[0]], device=query.device, dtype=torch.long
            ),
            "group_size": group_size,
        }
        if backend in {"group_mean4", "group_mean2"}:
            scores = score_group_mean_tensors(
                query.view(1, 1, -1),
                keys.view(1, keys.shape[0], keys.shape[1]),
                mean_mode="signed4" if backend == "group_mean4" else "abs2",
                **score_args,
            )[0, 0]
        else:
            scores = score_tensors(
                query.view(1, 1, -1),
                keys.view(1, keys.shape[0], keys.shape[1]),
                **score_args,
            )[0, 0]
        if backend == "cuda_popc_histogram":
            indices = histogram_topk_from_scores(
                scores.view(1, 1, -1),
                torch.tensor([keys.shape[0]], device=query.device, dtype=torch.long),
                budget=budget,
                head_dim=int(query.numel()),
            )[0, 0]
        else:
            indices = torch.topk(scores, k=min(budget, int(scores.numel())), largest=True, sorted=False).indices
        return _include_current(indices, keys.shape[0], min(budget, keys.shape[0]))
    raise ValueError(f"Unknown bit2 backend: {backend}")


def build_packed_fier_head_cache(keys: torch.Tensor, *, group_size: int) -> PackedFIERHeadCache:
    if keys.ndim != 2:
        raise ValueError("Expected keys shaped [tokens, head_dim]")
    sealed_tokens = (int(keys.shape[0]) // group_size) * group_size
    sealed = keys[:sealed_tokens].float()
    active = keys[sealed_tokens:].detach().clone()
    if sealed_tokens:
        groups = sealed.view(-1, group_size, keys.shape[1])
        group_min = groups.amin(dim=1)
        group_max = groups.amax(dim=1)
        midpoint = (group_min[:, None, :] + group_max[:, None, :]) * 0.5
        sealed_bits = groups >= midpoint
        sealed_bits = sealed_bits.reshape(sealed_tokens, keys.shape[1])
    else:
        group_min = keys.new_empty((0, keys.shape[1]), dtype=torch.float32)
        group_max = keys.new_empty((0, keys.shape[1]), dtype=torch.float32)
        sealed_bits = torch.empty((0, keys.shape[1]), dtype=torch.bool, device=keys.device)
    return PackedFIERHeadCache(group_size, sealed_bits, group_min, group_max, active)


def _seal_fier_active(cache: PackedFIERHeadCache) -> None:
    if int(cache.active_keys.shape[0]) < cache.group_size:
        return
    seal = cache.active_keys[: cache.group_size].float()
    group_min = seal.amin(dim=0, keepdim=True)
    group_max = seal.amax(dim=0, keepdim=True)
    midpoint = (group_min + group_max) * 0.5
    bits = seal >= midpoint
    cache.group_min = torch.cat([cache.group_min, group_min], dim=0)
    cache.group_max = torch.cat([cache.group_max, group_max], dim=0)
    cache.sealed_bits = torch.cat([cache.sealed_bits, bits], dim=0)
    cache.active_keys = cache.active_keys[cache.group_size :].detach().clone()


def append_packed_fier_head_cache(cache: PackedFIERHeadCache, key: torch.Tensor) -> None:
    cache.active_keys = torch.cat([cache.active_keys, key.view(1, -1).detach()], dim=0)
    _seal_fier_active(cache)


def build_packed_fier_triton_layer_cache(
    keys: torch.Tensor, *, group_size: int, reserve_tokens: int = 2048,
) -> PackedFIERTritonLayerCache:
    """Build persistent FIER storage; appends repack only one active group."""
    if group_size != 32:
        raise ValueError("Persistent FIER Triton cache currently requires group_size=32")
    from fier_triton import pack_keys
    keys = keys.contiguous()
    packed, mins, maxs = pack_keys(keys, group_size=group_size)
    kv_heads, tokens, head_dim = (int(v) for v in keys.shape)
    capacity = ((tokens + reserve_tokens + group_size - 1) // group_size) * group_size
    capacity = int(capacity)
    packed_storage = torch.empty(
        (kv_heads, head_dim, math.ceil(capacity / 32)),
        device=keys.device, dtype=packed.dtype,
    )
    min_storage = torch.empty(
        (kv_heads, math.ceil(capacity / group_size), head_dim),
        device=keys.device, dtype=mins.dtype,
    )
    max_storage = torch.empty_like(min_storage)
    packed_storage[:, :, : packed.shape[2]].copy_(packed)
    min_storage[:, : mins.shape[1]].copy_(mins)
    max_storage[:, : maxs.shape[1]].copy_(maxs)
    sealed = (tokens // group_size) * group_size
    return PackedFIERTritonLayerCache(
        group_size, tokens, sealed, packed_storage, min_storage, max_storage,
        keys[:, sealed:].detach().clone().contiguous(),
    )


def stage_packed_fier_current(
    cache: PackedFIERTritonLayerCache, keys: torch.Tensor,
) -> None:
    """Write active-group metadata including current K without advancing cache."""
    from fier_triton import pack_keys
    staged = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    if cache.sealed_tokens + int(staged.shape[1]) > int(cache.packed.shape[2]) * 32:
        raise RuntimeError("persistent FIER cache reserve exhausted")
    packed, mins, maxs = pack_keys(staged, group_size=cache.group_size)
    word_offset = cache.sealed_tokens // 32
    group_offset = cache.sealed_tokens // cache.group_size
    cache.packed[:, :, word_offset : word_offset + packed.shape[2]].copy_(packed)
    cache.group_min[:, group_offset : group_offset + mins.shape[1]].copy_(mins)
    cache.group_max[:, group_offset : group_offset + maxs.shape[1]].copy_(maxs)


def append_packed_fier_triton_layer_cache(
    cache: PackedFIERTritonLayerCache, keys: torch.Tensor,
) -> None:
    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    cache.token_count += int(keys.shape[1])
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()


def select_fier_prepacked(
    query: torch.Tensor,
    cache: PackedFIERHeadCache,
    *,
    budget: int,
    extra_key: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = []
    if cache.sealed_bits.numel():
        group_ids = torch.arange(
            cache.group_min.shape[0], device=query.device
        ).repeat_interleave(cache.group_size)
        mins = cache.group_min.index_select(0, group_ids)
        maxs = cache.group_max.index_select(0, group_ids)
        dequant = torch.where(cache.sealed_bits.to(query.device), maxs, mins)
        scores.append(dequant @ query.float())
    active = cache.active_keys
    if extra_key is not None:
        active = torch.cat([active, extra_key.view(1, -1)], dim=0)
    if active.numel():
        scores.append(fier_dequantize_1bit(active, group_size=cache.group_size) @ query.float())
    all_scores = torch.cat(scores, dim=0) if scores else query.new_empty((0,), dtype=torch.float32)
    indices = _topk_indices(all_scores, budget)
    return _include_current(indices, int(all_scores.numel()), min(budget, int(all_scores.numel())))


def build_packed_bit2_head_cache(keys: torch.Tensor, *, group_size: int) -> PackedBit2HeadCache:
    if keys.ndim != 2:
        raise ValueError("Expected keys shaped [tokens, head_dim]")
    sealed_tokens = (int(keys.shape[0]) // group_size) * group_size
    sealed = keys[:sealed_tokens]
    active = keys[sealed_tokens:].detach().clone()
    if sealed_tokens:
        sign, magnitude = quantize_grouped_keys_bit2(sealed, group_size=group_size)
        sealed_sign = pack_bool_bits(sign)
        sealed_magnitude = pack_bool_bits(magnitude)
    else:
        words = math.ceil(int(keys.shape[1]) / 8)
        sealed_sign = torch.empty((0, words), dtype=torch.uint8, device=keys.device)
        sealed_magnitude = torch.empty((0, words), dtype=torch.uint8, device=keys.device)
    return PackedBit2HeadCache(
        group_size, int(keys.shape[1]), sealed_sign, sealed_magnitude, active
    )


def _seal_bit2_active(cache: PackedBit2HeadCache) -> None:
    if int(cache.active_keys.shape[0]) < cache.group_size:
        return
    seal = cache.active_keys[: cache.group_size]
    sign, magnitude = quantize_grouped_keys_bit2(seal, group_size=cache.group_size)
    cache.sealed_sign = torch.cat([cache.sealed_sign, pack_bool_bits(sign)], dim=0)
    cache.sealed_magnitude = torch.cat(
        [cache.sealed_magnitude, pack_bool_bits(magnitude)], dim=0
    )
    cache.active_keys = cache.active_keys[cache.group_size :].detach().clone()


def append_packed_bit2_head_cache(cache: PackedBit2HeadCache, key: torch.Tensor) -> None:
    cache.active_keys = torch.cat([cache.active_keys, key.view(1, -1).detach()], dim=0)
    _seal_bit2_active(cache)


def build_packed_bit2mean_layer_cache(
    keys: torch.Tensor, *, group_size: int, reserve_tokens: int = 2048
) -> PackedBit2MeanLayerCache:
    """Build persistent [KVH,T,D] 2-mean storage with append capacity."""
    from bit2_cuda import pack_keys_2mean

    if keys.ndim != 3 or not keys.is_cuda:
        raise ValueError("Expected CUDA keys shaped [KVH, tokens, head_dim]")
    tokens = int(keys.shape[1])
    capacity = ((tokens + reserve_tokens + group_size - 1) // group_size) * group_size
    packed = pack_keys_2mean(
        keys.contiguous(), group_size=group_size, token_capacity=capacity
    )
    sealed = (tokens // group_size) * group_size
    return PackedBit2MeanLayerCache(
        group_size=group_size,
        token_count=tokens,
        sealed_tokens=sealed,
        packed=packed,
        active_keys=keys[:, sealed:].detach().clone().contiguous(),
    )


def append_packed_bit2mean_layer_cache(
    cache: PackedBit2MeanLayerCache, keys: torch.Tensor
) -> None:
    """Repack only the active token-axis group directly into cache storage."""
    from bit2_cuda import pack_keys_2mean_into

    if keys.ndim != 3 or int(keys.shape[1]) != 1:
        raise ValueError("new keys must be [KVH, 1, D]")
    cache.active_keys = torch.cat(
        [cache.active_keys, keys.detach()], dim=1
    ).contiguous()
    if cache.sealed_tokens + int(cache.active_keys.shape[1]) > int(cache.packed[0].shape[1]):
        raise RuntimeError("2-mean cache reserve exhausted")
    pack_keys_2mean_into(
        cache.active_keys,
        cache.packed,
        group_size=cache.group_size,
        token_offset=cache.sealed_tokens,
    )
    cache.token_count += 1
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()


def build_packed_bit2cuda_layer_cache(
    keys: torch.Tensor, *, group_size: int, reserve_tokens: int = 2048
) -> PackedBit2CudaLayerCache:
    from bit2_cuda import pack_keys_cached
    tokens = int(keys.shape[1])
    capacity = ((tokens + reserve_tokens + group_size - 1) // group_size) * group_size
    packed = pack_keys_cached(keys.contiguous(), group_size=group_size, token_capacity=capacity)
    sealed = (tokens // group_size) * group_size
    return PackedBit2CudaLayerCache(
        group_size, tokens, sealed, packed,
        keys[:, sealed:].detach().clone().contiguous(),
    )


def append_packed_bit2cuda_layer_cache(
    cache: PackedBit2CudaLayerCache, keys: torch.Tensor
) -> None:
    from bit2_cuda import pack_keys_cached_into
    cache.active_keys = torch.cat([cache.active_keys, keys.detach()], dim=1).contiguous()
    if cache.sealed_tokens + int(cache.active_keys.shape[1]) > int(cache.packed[0].shape[1]):
        raise RuntimeError("persistent cuda_popc cache reserve exhausted")
    pack_keys_cached_into(
        cache.active_keys, cache.packed,
        group_size=cache.group_size, token_offset=cache.sealed_tokens,
    )
    cache.token_count += 1
    if int(cache.active_keys.shape[1]) == cache.group_size:
        cache.sealed_tokens += cache.group_size
        cache.active_keys = cache.active_keys[:, :0].contiguous()


def select_bit2_prepacked(
    query: torch.Tensor,
    cache: PackedBit2HeadCache,
    *,
    budget: int,
    extra_key: torch.Tensor | None = None,
) -> torch.Tensor:
    query_sign, query_magnitude = quantize_query_bit2(query)
    scores = []
    if cache.sealed_sign.numel():
        scores.append(
            bit2_interaction_scores_packed(
                pack_bool_bits(query_sign),
                pack_bool_bits(query_magnitude),
                cache.sealed_sign,
                cache.sealed_magnitude,
                dimension=cache.dimension,
            ).to(torch.float32)
        )
    active = cache.active_keys
    if extra_key is not None:
        active = torch.cat([active, extra_key.view(1, -1)], dim=0)
    if active.numel():
        scores.append(bit2_approximate_scores(query, active, group_size=cache.group_size).to(torch.float32))
    all_scores = torch.cat(scores, dim=0) if scores else query.new_empty((0,), dtype=torch.float32)
    indices = _topk_indices(all_scores, budget)
    return _include_current(indices, int(all_scores.numel()), min(budget, int(all_scores.numel())))


def select_candidates_prepacked(
    method: str,
    query: torch.Tensor,
    cache: PackedFIERHeadCache | PackedBit2HeadCache,
    *,
    budget: int,
    extra_key: torch.Tensor | None = None,
) -> torch.Tensor:
    if method == "fier" and isinstance(cache, PackedFIERHeadCache):
        return select_fier_prepacked(query, cache, budget=budget, extra_key=extra_key)
    if method == "bit2_qk" and isinstance(cache, PackedBit2HeadCache):
        return select_bit2_prepacked(query, cache, budget=budget, extra_key=extra_key)
    raise TypeError(f"Packed cache mismatch for method {method!r}")


def select_candidates(
    method: str,
    query: torch.Tensor,
    keys: torch.Tensor,
    config: RetrievalConfig,
    basis: PCABasis | None = None,
) -> torch.Tensor:
    if method == "full":
        return select_full(keys)
    if method == "pqsift":
        if basis is None:
            raise ValueError("PQ-SIFT requires a PCA basis")
        return select_pqsift(
            query,
            keys,
            basis,
            axes=config.pqsift_axes,
            keep_ratio=config.pqsift_keep_ratio,
        )
    if method == "loki":
        if basis is None:
            raise ValueError("Loki requires a PCA basis")
        return select_loki(
            query,
            keys,
            basis,
            rank=config.loki_rank,
            budget=config.budget,
        )
    if method == "quest":
        return select_quest(
            query,
            keys,
            page_size=config.quest_page_size,
            budget=config.budget,
        )
    if method == "fier":
        return select_fier(
            query,
            keys,
            group_size=config.fier_group_size,
            budget=config.budget,
            backend=config.fier_backend,
        )
    if method == "bit2_qk":
        return select_bit2_qk(
            query,
            keys,
            group_size=config.bit2_group_size,
            budget=config.budget,
            backend=config.bit2_backend,
        )
    raise ValueError(f"Unknown method {method!r}; expected one of {SUPPORTED_METHODS}")


def gather_selected_kv(
    keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        keys.index_select(0, indices).float(),
        values.index_select(0, indices).float(),
    )


def exact_attention_selected(
    query: torch.Tensor, selected_keys: torch.Tensor, selected_values: torch.Tensor
) -> torch.Tensor:
    scores = (selected_keys @ query.float()) / math.sqrt(query.numel())
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ selected_values


def gather_selected_kv_batched(
    prefix_keys: torch.Tensor, prefix_values: torch.Tensor,
    current_keys: torch.Tensor, current_values: torch.Tensor,
    indices: torch.Tensor, head_to_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather [QH,K,D] directly without repeating full GQA K/V."""
    prefix_tokens = int(prefix_keys.shape[1])
    prefix_indices = indices.clamp_max(prefix_tokens - 1)
    kv_heads = head_to_kv[:, None].expand_as(prefix_indices)
    selected_keys = prefix_keys[kv_heads, prefix_indices]
    selected_values = prefix_values[kv_heads, prefix_indices]
    current_mask = indices.eq(prefix_tokens).unsqueeze(-1)
    current_keys_q = current_keys.index_select(0, head_to_kv)[:, None, :]
    current_values_q = current_values.index_select(0, head_to_kv)[:, None, :]
    return (
        torch.where(current_mask, current_keys_q, selected_keys),
        torch.where(current_mask, current_values_q, selected_values),
    )


def exact_attention_selected_batched(
    queries: torch.Tensor, selected_keys: torch.Tensor, selected_values: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused SDPA over all query heads while retaining low-precision K/V."""
    return F.scaled_dot_product_attention(
        queries.unsqueeze(0).unsqueeze(2), selected_keys.unsqueeze(0),
        selected_values.unsqueeze(0),
        attn_mask=None if valid_mask is None else valid_mask[None, :, None, :],
        dropout_p=0.0, is_causal=False,
    )[0, :, 0]


def exact_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    selected_keys, selected_values = gather_selected_kv(keys, values, indices)
    return exact_attention_selected(query, selected_keys, selected_values)


def estimate_candidate_search_ops(
    method: str,
    *,
    num_tokens: int,
    head_dim: int,
    budget: int,
) -> float:
    """Rough candidate-search operation proxy for FIER/bit2 comparisons."""

    if method not in {"fier", "bit2_qk"}:
        return 0.0
    tokens = float(num_tokens)
    dim = float(head_dim)
    topk_ops = tokens * math.log2(max(2, min(int(budget), int(num_tokens))))
    if method == "fier":
        minmax_ops = 2.0 * tokens * dim
        dequant_ops = 2.0 * tokens * dim
        weighted_qk_ops = 2.0 * tokens * dim
        return minmax_ops + dequant_ops + weighted_qk_ops + topk_ops

    packed_words = math.ceil(head_dim / 8)
    query_threshold_ops = 2.0 * dim
    query_quantize_ops = 2.0 * dim
    key_threshold_ops = 2.0 * tokens * dim
    key_quantize_ops = 2.0 * tokens * dim
    pack_ops = 2.0 * (tokens + 1.0) * dim
    bitwise_ops = 7.0 * tokens * packed_words
    popcount_ops = 5.0 * tokens * packed_words
    integer_accum_ops = 9.0 * tokens
    return (
        query_threshold_ops
        + query_quantize_ops
        + key_threshold_ops
        + key_quantize_ops
        + pack_ops
        + bitwise_ops
        + popcount_ops
        + integer_accum_ops
        + topk_ops
    )


def estimate_prepacked_candidate_search_ops(
    method: str,
    *,
    num_tokens: int,
    head_dim: int,
    budget: int,
) -> float:
    if method not in {"fier", "bit2_qk"}:
        return 0.0
    tokens = float(num_tokens)
    dim = float(head_dim)
    topk_ops = tokens * math.log2(max(2, min(int(budget), int(num_tokens))))
    if method == "fier":
        weighted_qk_ops = 2.0 * tokens * dim
        return weighted_qk_ops + topk_ops

    packed_words = math.ceil(head_dim / 8)
    query_threshold_ops = 2.0 * dim
    query_quantize_ops = 2.0 * dim
    query_pack_ops = 2.0 * dim
    bitwise_ops = 7.0 * tokens * packed_words
    popcount_ops = 5.0 * tokens * packed_words
    integer_accum_ops = 9.0 * tokens
    return (
        query_threshold_ops
        + query_quantize_ops
        + query_pack_ops
        + bitwise_ops
        + popcount_ops
        + integer_accum_ops
        + topk_ops
    )


@dataclass
class DecodeDiagnostics:
    selected_tokens: int = 0
    available_tokens: int = 0
    sparse_head_calls: int = 0
    candidate_search_ms: float = 0.0
    candidate_score_ms: float = 0.0
    candidate_topk_ms: float = 0.0
    candidate_search_ops: float = 0.0
    cache_update_ms: float = 0.0
    selected_gather_ms: float = 0.0
    selected_attention_ms: float = 0.0
    topk_recall_sum: float = 0.0
    topk_recall_calls: int = 0
    topk_recall_by_k_sum: dict[int, float] = field(default_factory=dict)
    topk_recall_by_k_calls: dict[int, int] = field(default_factory=dict)
    attention_mass_recall_sum: float = 0.0
    score_mae_sum: float = 0.0
    score_normalized_mae_sum: float = 0.0
    spearman_sum: float = 0.0
    score_diagnostic_calls: int = 0

    @property
    def candidate_ratio(self) -> float:
        if self.available_tokens == 0:
            return 1.0
        return self.selected_tokens / self.available_tokens

    @property
    def topk_recall(self) -> float | None:
        if self.topk_recall_calls == 0:
            return None
        return self.topk_recall_sum / self.topk_recall_calls

    def topk_recall_at(self, k: int) -> float | None:
        calls = self.topk_recall_by_k_calls.get(int(k), 0)
        return None if calls == 0 else self.topk_recall_by_k_sum[int(k)] / calls

    @property
    def attention_mass_recall(self) -> float | None:
        return None if not self.topk_recall_calls else self.attention_mass_recall_sum / self.topk_recall_calls

    @property
    def score_mae(self) -> float | None:
        return None if not self.score_diagnostic_calls else self.score_mae_sum / self.score_diagnostic_calls

    @property
    def score_normalized_mae(self) -> float | None:
        return None if not self.score_diagnostic_calls else self.score_normalized_mae_sum / self.score_diagnostic_calls

    @property
    def spearman(self) -> float | None:
        return None if not self.score_diagnostic_calls else self.spearman_sum / self.score_diagnostic_calls


def _timed_component(
    operation: Callable[[], T],
    *,
    device: torch.device,
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]],
) -> tuple[T, float]:
    """Run one component and return CPU time or defer CUDA timing to events."""

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            start.record()
            result = operation()
            end.record()
        cuda_events.append((start, end, device))
        return result, 0.0

    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000.0


def _finish_cuda_component_timing(
    events: Sequence[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]],
) -> float:
    if not events:
        return 0.0
    devices = {device for _, _, device in events}
    for device in devices:
        torch.cuda.synchronize(device)
    return sum(float(start.elapsed_time(end)) for start, end, _ in events)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # LlamaModel returns [batch, seq, head_dim]. Heads are dimension 1 here.
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query_states * cos + _rotate_half(query_states) * sin,
        key_states * cos + _rotate_half(key_states) * sin,
    )


def _layer_cache(past_key_values: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(past_key_values, StaticKVCache):
        return (
            past_key_values.key_storage[layer_idx][..., : past_key_values.length, :],
            past_key_values.value_storage[layer_idx][..., : past_key_values.length, :],
        )
    try:
        key, value = past_key_values[layer_idx][:2]
        return key, value
    except (TypeError, IndexError, KeyError):
        pass
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    raise TypeError(f"Unsupported transformers cache type: {type(past_key_values).__name__}")


class LlamaSparseDecoder:
    """One-token Llama decoder with pluggable KV retrieval.

    Prefix states/KV are produced by the unmodified dense model. Only the
    current token is decoded with the selected retrieval method. This measures
    sampled decode-style PPL, not teacher-forced full-corpus PPL.
    """

    def __init__(
        self,
        model: Any,
        *,
        pca_cache: PCABasisCache | None,
        retrieval: RetrievalConfig,
    ) -> None:
        retrieval.validate()
        self.model = model
        self.backbone = getattr(model, "model", None)
        if self.backbone is None:
            raise TypeError("Expected a LlamaForCausalLM-like model with a .model backbone")
        self.layers = self.backbone.layers
        self.config = model.config
        self.pca_cache = pca_cache
        self.retrieval = retrieval
        self.num_heads = int(self.config.num_attention_heads)
        self.num_kv_heads = int(
            getattr(self.config, "num_key_value_heads", self.num_heads)
        )
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.kv_group_size = self.num_heads // self.num_kv_heads
        self.head_dim = int(
            getattr(
                self.config,
                "head_dim",
                int(self.config.hidden_size) // self.num_heads,
            )
        )
        self.head_to_kv = (
            torch.arange(self.num_heads, device=self.device, dtype=torch.long)
            // self.kv_group_size
        )
        self.diagnostic_layer = next(
            (i for i in range(len(self.layers)) if i not in retrieval.full_layers), 0
        )

    @property
    def device(self) -> torch.device:
        return self.backbone.embed_tokens.weight.device

    @torch.inference_mode()
    def build_prefix_cache(self, prefix_ids: torch.Tensor) -> Any:
        prefix_ids = prefix_ids.to(self.device).view(1, -1)
        attention_mask = torch.ones_like(prefix_ids)
        outputs = self.backbone(
            input_ids=prefix_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.past_key_values

    @torch.inference_mode()
    def build_packed_retrieval_caches(
        self, past_key_values: Any, *, method: str, group_size: int
    ) -> PackedRetrievalCaches:
        if method not in {"fier", "bit2_qk"}:
            raise ValueError("Packed retrieval caches are only implemented for FIER/bit2")
        caches: dict[
            tuple[int, int],
            PackedFIERHeadCache | PackedFIERTritonLayerCache | PackedBit2HeadCache | PackedBit2MeanLayerCache | PackedBit2CudaLayerCache,
        ] = {}
        for layer_idx in range(len(self.layers)):
            keys, _ = _layer_cache(past_key_values, layer_idx)
            keys = keys.to(self.device)
            if method == "fier" and self.retrieval.fier_backend == "triton":
                caches[(layer_idx, -1)] = build_packed_fier_triton_layer_cache(
                    keys[0], group_size=group_size
                )
                continue
            if method == "bit2_qk" and self.retrieval.bit2_backend == "cuda_popc":
                caches[(layer_idx, -1)] = build_packed_bit2cuda_layer_cache(
                    keys[0], group_size=group_size
                )
                continue
            if method == "bit2_qk" and self.retrieval.bit2_backend in {
                "2mean", "cuda_2mean", "triton_2mean",
                "qk_2mean", "cuda_qk_2mean", "triton_qk_2mean",
            }:
                caches[(layer_idx, -1)] = build_packed_bit2mean_layer_cache(
                    keys[0], group_size=group_size
                )
                continue
            for kv_head_idx in range(self.num_kv_heads):
                head_keys = keys[0, kv_head_idx]
                if method == "fier":
                    cache = build_packed_fier_head_cache(
                        head_keys, group_size=group_size
                    )
                else:
                    cache = build_packed_bit2_head_cache(
                        head_keys, group_size=group_size
                    )
                caches[(layer_idx, kv_head_idx)] = cache
        return PackedRetrievalCaches(method=method, caches=caches)

    @torch.inference_mode()
    def append_to_packed_retrieval_caches(
        self,
        packed: PackedRetrievalCaches,
        new_keys_by_layer: Sequence[torch.Tensor],
    ) -> None:
        for layer_idx, key_states in enumerate(new_keys_by_layer):
            layer_cache = packed.caches.get((layer_idx, -1))
            if isinstance(layer_cache, PackedFIERTritonLayerCache):
                append_packed_fier_triton_layer_cache(layer_cache, key_states[0])
                continue
            if isinstance(layer_cache, PackedBit2CudaLayerCache):
                append_packed_bit2cuda_layer_cache(layer_cache, key_states[0])
                continue
            if isinstance(layer_cache, PackedBit2MeanLayerCache):
                append_packed_bit2mean_layer_cache(layer_cache, key_states[0])
                continue
            for kv_head_idx in range(self.num_kv_heads):
                cache = packed.caches[(layer_idx, kv_head_idx)]
                key = key_states[0, kv_head_idx, 0]
                if packed.method == "fier" and isinstance(cache, PackedFIERHeadCache):
                    append_packed_fier_head_cache(cache, key)
                elif packed.method == "bit2_qk" and isinstance(cache, PackedBit2HeadCache):
                    append_packed_bit2_head_cache(cache, key)
                else:
                    raise TypeError("Packed retrieval cache type mismatch")

    @torch.inference_mode()
    def build_static_kv_cache(
        self, past_key_values: Any, *, reserve_tokens: int,
    ) -> StaticKVCache:
        first_key, _ = _layer_cache(past_key_values, 0)
        length = int(first_key.shape[-2])
        capacity = length + int(reserve_tokens)
        key_storage: list[torch.Tensor] = []
        value_storage: list[torch.Tensor] = []
        for layer_idx in range(len(self.layers)):
            key, value = _layer_cache(past_key_values, layer_idx)
            key_store = torch.empty(
                (*key.shape[:-2], capacity, key.shape[-1]),
                device=key.device, dtype=key.dtype,
            )
            value_store = torch.empty_like(key_store)
            key_store[..., :length, :].copy_(key)
            value_store[..., :length, :].copy_(value)
            key_storage.append(key_store)
            value_storage.append(value_store)
        return StaticKVCache(key_storage, value_storage, length, capacity)

    @torch.inference_mode()
    def append_to_past_key_values(
        self,
        past_key_values: Any,
        new_keys_by_layer: Sequence[torch.Tensor],
        new_values_by_layer: Sequence[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(past_key_values, StaticKVCache):
            if past_key_values.length >= past_key_values.capacity:
                raise RuntimeError("static KV cache capacity exhausted")
            position = past_key_values.length
            for layer_idx, (new_key, new_value) in enumerate(
                zip(new_keys_by_layer, new_values_by_layer)
            ):
                past_key_values.key_storage[layer_idx][..., position : position + 1, :].copy_(new_key)
                past_key_values.value_storage[layer_idx][..., position : position + 1, :].copy_(new_value)
            past_key_values.length += 1
            return past_key_values

        updated = []
        for layer_idx, (new_key, new_value) in enumerate(
            zip(new_keys_by_layer, new_values_by_layer)
        ):
            old_key, old_value = _layer_cache(past_key_values, layer_idx)
            old_key = old_key.to(new_key.device)
            old_value = old_value.to(new_value.device)
            updated.append(
                (
                    torch.cat([old_key, new_key.detach()], dim=-2),
                    torch.cat([old_value, new_value.detach()], dim=-2),
                )
            )
        return updated

    def _position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        first_value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary = getattr(self.backbone, "rotary_emb", None)
        if rotary is not None:
            try:
                return rotary(hidden_states, position_ids)
            except TypeError:
                pass
        rotary = getattr(self.layers[0].self_attn, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("Cannot find Llama rotary embedding module")
        try:
            return rotary(first_value_states, position_ids)
        except TypeError:
            cos, sin = rotary(first_value_states, seq_len=int(position_ids.max()) + 1)
            return cos[:, position_ids.item() : position_ids.item() + 1], sin[
                :, position_ids.item() : position_ids.item() + 1
            ]

    @torch.inference_mode()
    def decode(
        self,
        past_key_values: Any,
        current_token_id: int | torch.Tensor,
        *,
        method: str,
        retrieval_caches: PackedRetrievalCaches | None = None,
        return_new_kv: bool = False,
    ) -> tuple[torch.Tensor, DecodeDiagnostics] | tuple[torch.Tensor, DecodeDiagnostics, list[torch.Tensor], list[torch.Tensor]]:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}")
        first_key, _ = _layer_cache(past_key_values, 0)
        prefix_length = int(first_key.shape[-2])

        token = torch.as_tensor(current_token_id, device=self.device, dtype=torch.long).view(1, 1)
        hidden_states = self.backbone.embed_tokens(token)
        position_ids = torch.tensor([[prefix_length]], device=self.device, dtype=torch.long)

        first_attn = self.layers[0].self_attn
        first_value = first_attn.v_proj(hidden_states).view(
            1, 1, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        cos, sin = self._position_embeddings(
            hidden_states,
            position_ids,
            first_value,
        )
        diagnostics = DecodeDiagnostics()
        candidate_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]] = []
        score_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]] = []
        topk_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]] = []
        gather_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]] = []
        attention_events: list[tuple[torch.cuda.Event, torch.cuda.Event, torch.device]] = []
        new_keys_by_layer: list[torch.Tensor] = []
        new_values_by_layer: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)

            query_states = attention.q_proj(normalized).view(
                1, 1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = attention.k_proj(normalized).view(
                1, 1, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            value_states = attention.v_proj(normalized).view(
                1, 1, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            query_states, key_states = _apply_rope(
                query_states,
                key_states,
                cos.to(query_states.device),
                sin.to(query_states.device),
            )

            prefix_keys, prefix_values = _layer_cache(past_key_values, layer_idx)
            prefix_keys = prefix_keys.to(query_states.device)
            prefix_values = prefix_values.to(query_states.device)
            if return_new_kv:
                new_keys_by_layer.append(key_states.detach())
                new_values_by_layer.append(value_states.detach())

            head_outputs = []
            force_full = method == "full" or layer_idx in self.retrieval.full_layers
            if force_full:
                def gather_full_layer() -> tuple[torch.Tensor, torch.Tensor]:
                    return (
                        torch.cat([prefix_keys, key_states], dim=2),
                        torch.cat([prefix_values, value_states], dim=2),
                    )

                (full_keys, full_values), gather_cpu_ms = _timed_component(
                    gather_full_layer, device=query_states.device, cuda_events=gather_events,
                )
                diagnostics.selected_gather_ms += gather_cpu_ms

                def attend_full_layer() -> torch.Tensor:
                    grouped_queries = query_states[0].view(
                        self.num_kv_heads, self.kv_group_size, 1, self.head_dim
                    )
                    return F.scaled_dot_product_attention(
                        grouped_queries, full_keys[0].unsqueeze(1),
                        full_values[0].unsqueeze(1), dropout_p=0.0, is_causal=False,
                    ).reshape(self.num_heads, self.head_dim)

                attention_output, attention_cpu_ms = _timed_component(
                    attend_full_layer, device=query_states.device, cuda_events=attention_events,
                )
                diagnostics.selected_attention_ms += attention_cpu_ms
                attention_output = attention_output.reshape(1, 1, -1).to(normalized.dtype)
                hidden_states = residual + attention.o_proj(attention_output)
                residual = hidden_states
                hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
                continue

            batched_fier_indices = None
            if (
                method == "fier"
                and self.retrieval.fier_backend == "triton"
                and not force_full
            ):
                if retrieval_caches is None:
                    raise RuntimeError("FIER Triton optimization requires persistent cache")
                fier_cache = retrieval_caches.caches[(layer_idx, -1)]
                if not isinstance(fier_cache, PackedFIERTritonLayerCache):
                    raise TypeError("persistent FIER layer cache mismatch")
                def score_all_fier_heads() -> torch.Tensor:
                    from fier_triton import score_packed_batched
                    stage_packed_fier_current(fier_cache, key_states[0])
                    tokens = fier_cache.token_count + 1
                    return score_packed_batched(
                        query_states[0, :, 0, :].unsqueeze(0),
                        fier_cache.packed, fier_cache.group_min, fier_cache.group_max,
                        self.head_to_kv, tokens=tokens,
                        group_size=fier_cache.group_size,
                    )

                fier_scores, score_cpu_ms = _timed_component(
                    score_all_fier_heads,
                    device=query_states.device,
                    cuda_events=score_events,
                )
                diagnostics.candidate_score_ms += score_cpu_ms

                def topk_all_fier_heads() -> torch.Tensor:
                    prefix_tokens = int(fier_scores.shape[-1]) - 1
                    topk_budget = min(self.retrieval.budget - 1, prefix_tokens)
                    prefix_indices = torch.topk(
                        fier_scores[..., :prefix_tokens], k=topk_budget, dim=-1,
                        largest=True, sorted=False,
                    ).indices[0]
                    return _append_current_reserved(prefix_indices, prefix_tokens + 1)

                batched_fier_indices, topk_cpu_ms = _timed_component(
                    topk_all_fier_heads,
                    device=query_states.device,
                    cuda_events=topk_events,
                )
                diagnostics.candidate_topk_ms += topk_cpu_ms

            batched_bit2_indices = None
            batched_bit2_scores = None
            if (
                method == "bit2_qk"
                and self.retrieval.bit2_backend
                in {
                    "cuda_popc",
                    "cuda_popc_direct",
                    "cuda_popc_histogram",
                    "group_mean4",
                    "group_mean2",
                    "cuda_2mean",
                    "triton_2mean",
                    "2mean",
                    "qk_2mean",
                    "cuda_qk_2mean",
                    "triton_qk_2mean",
                }
                and not force_full
            ):
                is_cached_2mean = (
                    retrieval_caches is not None
                    and self.retrieval.bit2_backend in {
                        "2mean", "cuda_2mean", "triton_2mean",
                        "qk_2mean", "cuda_qk_2mean", "triton_qk_2mean",
                    }
                )
                is_cached_cuda_popc = (
                    retrieval_caches is not None
                    and self.retrieval.bit2_backend == "cuda_popc"
                )
                if retrieval_caches is not None and not (is_cached_2mean or is_cached_cuda_popc):
                    raise RuntimeError(
                        "cuda_popc backends currently use the direct packed CUDA path, "
                        "not prepacked retrieval_caches"
                    )

                if (
                    is_cached_2mean
                    and self.retrieval.bit2_backend in {"qk_2mean", "cuda_qk_2mean"}
                ):
                    cache = retrieval_caches.caches[(layer_idx, -1)]
                    if not isinstance(cache, PackedBit2MeanLayerCache):
                        raise TypeError("2-mean layer cache mismatch")
                    from bit2_cuda import score_qk_2mean_cuda_packed

                    def score_all_qk_2mean_heads() -> torch.Tensor:
                        return score_qk_2mean_cuda_packed(
                            query_states[0, :, 0, :].unsqueeze(0).contiguous(),
                            cache.packed,
                            head_to_kv=self.head_to_kv,
                            tokens=cache.token_count,
                        )

                    qk_scores, score_cpu_ms = _timed_component(
                        score_all_qk_2mean_heads,
                        device=query_states.device,
                        cuda_events=score_events,
                    )
                    diagnostics.candidate_score_ms += score_cpu_ms

                    def topk_all_qk_2mean_heads() -> torch.Tensor:
                        topk_budget = min(
                            self.retrieval.budget - 1, cache.token_count
                        )
                        prefix_indices = torch.topk(
                            qk_scores, k=topk_budget, dim=-1,
                            largest=True, sorted=False,
                        ).indices[0]
                        return _append_current_reserved(
                            prefix_indices, cache.token_count + 1
                        )

                    batched_bit2_indices, topk_cpu_ms = _timed_component(
                        topk_all_qk_2mean_heads,
                        device=query_states.device,
                        cuda_events=topk_events,
                    )
                    diagnostics.candidate_topk_ms += topk_cpu_ms
                    batched_bit2_scores = qk_scores[0]

                def select_all_bit2_heads() -> torch.Tensor:
                    from bit2_cuda import (
                        histogram_topk_from_scores,
                        score_group_mean_tensors,
                        score_tensors,
                    )

                    head_to_kv = self.head_to_kv
                    if is_cached_cuda_popc:
                        cache = retrieval_caches.caches[(layer_idx, -1)]
                        if not isinstance(cache, PackedBit2CudaLayerCache):
                            raise TypeError("persistent cuda_popc layer cache mismatch")
                        from bit2_cuda import score_popc_cached_cuda_packed
                        scores = score_popc_cached_cuda_packed(
                            query_states[0, :, 0, :].unsqueeze(0).contiguous(),
                            cache.packed, head_to_kv=head_to_kv,
                            tokens=cache.token_count,
                        )
                        # Current token is appended by _include_current below, so
                        # reserve one budget slot instead of dropping an arbitrary
                        # unsorted Top-k entry afterward.
                        topk_budget = min(self.retrieval.budget - 1, cache.token_count)
                        indices = torch.topk(
                            scores, k=topk_budget, dim=-1, largest=True, sorted=False
                        ).indices[0]
                        return indices, scores[0]
                    if is_cached_2mean:
                        cache = retrieval_caches.caches[(layer_idx, -1)]
                        if not isinstance(cache, PackedBit2MeanLayerCache):
                            raise TypeError("2-mean layer cache mismatch")
                        if self.retrieval.bit2_backend in {"2mean", "cuda_2mean"}:
                            from bit2_cuda import score_2mean_cuda_packed
                            scores = score_2mean_cuda_packed(
                                query_states[0, :, 0, :].unsqueeze(0).contiguous(),
                                cache.packed,
                                head_to_kv=head_to_kv,
                                tokens=cache.token_count,
                            )
                        elif self.retrieval.bit2_backend == "triton_2mean":
                            from bit2_2mean_triton import score_2mean_triton_packed
                            scores = score_2mean_triton_packed(
                                query_states[0, :, 0, :].unsqueeze(0),
                                cache.packed,
                                head_to_kv=head_to_kv,
                                tokens=cache.token_count,
                            )
                        elif self.retrieval.bit2_backend in {"qk_2mean", "cuda_qk_2mean"}:
                            from bit2_cuda import score_qk_2mean_cuda_packed
                            scores = score_qk_2mean_cuda_packed(
                                query_states[0, :, 0, :].unsqueeze(0).contiguous(),
                                cache.packed, head_to_kv=head_to_kv,
                                tokens=cache.token_count,
                            )
                        else:  # triton_qk_2mean
                            from bit2_2mean_triton import score_qk_2mean_triton_packed
                            scores = score_qk_2mean_triton_packed(
                                query_states[0, :, 0, :].unsqueeze(0),
                                cache.packed, head_to_kv=head_to_kv,
                                tokens=cache.token_count,
                            )
                        topk_budget = min(self.retrieval.budget - 1, cache.token_count)
                        indices = torch.topk(
                            scores, k=topk_budget, dim=-1, largest=True, sorted=False
                        ).indices[0]
                        return indices, scores[0]
                    layer_keys = torch.cat([prefix_keys[0], key_states[0]], dim=1)
                    valid_tokens = torch.tensor(
                        [layer_keys.shape[1]], device=query_states.device, dtype=torch.long
                    )
                    score_args = {
                        "head_to_kv": head_to_kv,
                        "valid_tokens": valid_tokens,
                        "group_size": self.retrieval.bit2_group_size,
                    }
                    if self.retrieval.bit2_backend in {
                        "group_mean4",
                        "group_mean2",
                    }:
                        scores = score_group_mean_tensors(
                            query_states[0, :, 0, :].unsqueeze(0),
                            layer_keys,
                            mean_mode=(
                                "signed4"
                                if self.retrieval.bit2_backend == "group_mean4"
                                else "abs2"
                            ),
                            **score_args,
                        )
                    else:
                        scores = score_tensors(
                            query_states[0, :, 0, :].unsqueeze(0),
                            layer_keys,
                            **score_args,
                        )
                    topk_budget = min(self.retrieval.budget, int(layer_keys.shape[1]))
                    if self.retrieval.bit2_backend == "cuda_popc_histogram":
                        indices = histogram_topk_from_scores(
                            scores,
                            valid_tokens,
                            budget=topk_budget,
                            head_dim=self.head_dim,
                        )[0]
                        return indices, scores[0]
                    indices = torch.topk(
                        scores, k=topk_budget, dim=-1, largest=True, sorted=False
                    ).indices[0]
                    return indices, scores[0]

                if batched_bit2_indices is None:
                    (
                        (batched_bit2_indices, batched_bit2_scores),
                        candidate_cpu_ms,
                    ) = _timed_component(
                        select_all_bit2_heads,
                        device=query_states.device,
                        cuda_events=candidate_events,
                    )
                    diagnostics.candidate_search_ms += candidate_cpu_ms

            batched_quest_indices = None
            batched_quest_valid = None
            if method == "quest":
                quest_scores, score_cpu_ms = _timed_component(
                    lambda: quest_page_scores_batched(
                        query_states[0, :, 0], prefix_keys[0], key_states[0, :, 0],
                        head_to_kv=self.head_to_kv,
                        page_size=self.retrieval.quest_page_size,
                    ),
                    device=query_states.device, cuda_events=score_events,
                )
                diagnostics.candidate_score_ms += score_cpu_ms
                (batched_quest_indices, batched_quest_valid), topk_cpu_ms = _timed_component(
                    lambda: quest_topk_indices_batched(
                        quest_scores, num_tokens=int(prefix_keys.shape[2]) + 1,
                        page_size=self.retrieval.quest_page_size, budget=self.retrieval.budget,
                    ),
                    device=query_states.device, cuda_events=topk_events,
                )
                diagnostics.candidate_topk_ms += topk_cpu_ms

            use_batched_sparse_attention = (
                batched_fier_indices is not None or batched_bit2_indices is not None
                or batched_quest_indices is not None
            )
            batched_attention_indices: list[torch.Tensor] = []
            batched_attention_valid: list[torch.Tensor] = []
            num_tokens = int(prefix_keys.shape[2]) + 1
            for head_idx in range(self.num_heads):
                kv_head_idx = head_idx // self.kv_group_size
                query = query_states[0, head_idx, 0]
                if use_batched_sparse_attention:
                    keys = None
                    values = None
                else:
                    keys = torch.cat(
                        [prefix_keys[0, kv_head_idx], key_states[0, kv_head_idx]], dim=0,
                    )
                    values = torch.cat(
                        [prefix_values[0, kv_head_idx], value_states[0, kv_head_idx]], dim=0,
                    )
                active_method = method
                basis = None
                if active_method in {"pqsift", "loki"}:
                    if self.pca_cache is None:
                        raise ValueError(f"{active_method} requires --pca-basis")
                    basis = self.pca_cache.get(
                        layer_idx,
                        head_idx,
                        device=query.device,
                    )
                if batched_fier_indices is not None and active_method == "fier":
                    indices = batched_fier_indices[head_idx]
                    candidate_cpu_ms = 0.0
                elif batched_bit2_indices is not None and active_method == "bit2_qk":
                    indices = batched_bit2_indices[head_idx]
                    candidate_cpu_ms = 0.0
                elif batched_quest_indices is not None and active_method == "quest":
                    indices = batched_quest_indices[head_idx]
                    index_valid = batched_quest_valid[head_idx]
                    candidate_cpu_ms = 0.0
                elif active_method == "quest":
                    quest_scores, score_cpu_ms = _timed_component(
                        lambda: quest_page_scores(
                            query, keys, page_size=self.retrieval.quest_page_size
                        ),
                        device=query.device,
                        cuda_events=score_events,
                    )
                    diagnostics.candidate_score_ms += score_cpu_ms

                    def topk_quest_pages() -> torch.Tensor:
                        page_size = self.retrieval.quest_page_size
                        page_budget = min(
                            int(quest_scores.numel()),
                            max(1, math.ceil(self.retrieval.budget / page_size)),
                        )
                        selected_pages = _topk_indices(quest_scores, page_budget)
                        offsets = torch.arange(page_size, device=keys.device)
                        selected = (
                            selected_pages[:, None] * page_size + offsets[None, :]
                        ).flatten()
                        selected = selected[selected < keys.shape[0]]
                        return _include_current(
                            selected, keys.shape[0],
                            min(self.retrieval.budget, keys.shape[0]),
                        )

                    indices, topk_cpu_ms = _timed_component(
                        topk_quest_pages,
                        device=query.device,
                        cuda_events=topk_events,
                    )
                    diagnostics.candidate_topk_ms += topk_cpu_ms
                    candidate_cpu_ms = 0.0
                elif (
                    retrieval_caches is not None
                    and not force_full
                    and active_method in {"quest", "fier", "bit2_qk"}
                ):
                    packed_head_cache = retrieval_caches.caches[(layer_idx, kv_head_idx)]
                    indices, candidate_cpu_ms = _timed_component(
                        lambda: select_candidates_prepacked(
                            active_method,
                            query,
                            packed_head_cache,
                            budget=self.retrieval.budget,
                            extra_key=key_states[0, kv_head_idx, 0],
                        ),
                        device=query.device,
                        cuda_events=candidate_events,
                    )
                else:
                    indices, candidate_cpu_ms = _timed_component(
                        lambda: select_candidates(
                            active_method,
                            query,
                            keys,
                            self.retrieval,
                            basis,
                        ),
                        device=query.device,
                        cuda_events=candidate_events,
                    )
                diagnostics.candidate_search_ms += candidate_cpu_ms
                index_valid = (
                    index_valid if batched_quest_indices is not None
                    else torch.ones_like(indices, dtype=torch.bool)
                )
                metric_indices = indices[index_valid]
                ops_estimator = (
                    estimate_prepacked_candidate_search_ops
                    if retrieval_caches is not None and active_method in {"quest", "fier", "bit2_qk"}
                    else estimate_candidate_search_ops
                )
                diagnostics.candidate_search_ops += ops_estimator(
                    active_method,
                    num_tokens=num_tokens,
                    head_dim=int(query.numel()),
                    budget=self.retrieval.budget,
                )
                if not force_full:
                    diagnostics.selected_tokens += int(metric_indices.numel())
                    diagnostics.available_tokens += num_tokens
                    diagnostics.sparse_head_calls += 1
                    if (
                        self.retrieval.measure_topk_recall
                        and active_method in {"quest", "fier", "bit2_qk"}
                        and layer_idx == self.diagnostic_layer
                    ):
                        recall_k = min(self.retrieval.budget, num_tokens)
                        recall_ks = tuple(
                            k for k in (1024, 2048, 3072, 4096)
                            if k <= recall_k
                        )
                        if recall_k not in recall_ks:
                            recall_ks = (*recall_ks, recall_k)
                        if keys is None:
                            keys_for_recall = torch.cat(
                                [prefix_keys[0, kv_head_idx], key_states[0, kv_head_idx]], dim=0,
                            )
                        else:
                            keys_for_recall = keys
                        exact_scores = (keys_for_recall.float() @ query.float()) / math.sqrt(self.head_dim)
                        recall_by_k: dict[int, float] = {}
                        for diagnostic_k in recall_ks:
                            if active_method == "fier" and fier_scores is not None:
                                prefix_k = min(diagnostic_k - 1, num_tokens - 1)
                                approx_prefix = torch.topk(
                                    fier_scores[0, head_idx, : num_tokens - 1],
                                    k=prefix_k, largest=True, sorted=False,
                                ).indices
                                approx_indices = _append_current_reserved(
                                    approx_prefix, num_tokens
                                )
                            elif active_method == "bit2_qk" and batched_bit2_scores is not None:
                                prefix_k = min(diagnostic_k - 1, num_tokens - 1)
                                approx_prefix = torch.topk(
                                    batched_bit2_scores[head_idx],
                                    k=prefix_k, largest=True, sorted=False,
                                ).indices
                                approx_indices = _append_current_reserved(
                                    approx_prefix, num_tokens
                                )
                            elif active_method == "quest" and quest_scores is not None:
                                quest_indices, quest_valid = quest_topk_indices_batched(
                                    quest_scores[head_idx : head_idx + 1],
                                    num_tokens=num_tokens,
                                    page_size=self.retrieval.quest_page_size,
                                    budget=diagnostic_k,
                                )
                                approx_indices = quest_indices[0][quest_valid[0]]
                            else:
                                approx_indices = metric_indices
                            exact_topk_k = _topk_indices(exact_scores, diagnostic_k)
                            recall_value = float(
                                torch.isin(approx_indices, exact_topk_k).sum().item()
                                / diagnostic_k
                            )
                            recall_by_k[diagnostic_k] = recall_value
                            diagnostics.topk_recall_by_k_sum[diagnostic_k] = (
                                diagnostics.topk_recall_by_k_sum.get(diagnostic_k, 0.0)
                                + recall_value
                            )
                            diagnostics.topk_recall_by_k_calls[diagnostic_k] = (
                                diagnostics.topk_recall_by_k_calls.get(diagnostic_k, 0) + 1
                            )
                        diagnostics.topk_recall_sum += recall_by_k[recall_k]
                        diagnostics.attention_mass_recall_sum += float(
                            torch.softmax(exact_scores, dim=0).index_select(0, metric_indices).sum().item()
                        )
                        diagnostics.topk_recall_calls += 1
                        if (
                            self.retrieval.measure_score_diagnostics
                            and active_method == "bit2_qk"
                            and batched_bit2_scores is not None
                        ):
                            approx = batched_bit2_scores[head_idx].float()
                            exact_for_score = exact_scores[: approx.numel()]
                            raw_mae = (approx - exact_for_score).abs().mean()
                            approx_norm = (approx - approx.mean()) / approx.std().clamp_min(1e-6)
                            exact_norm = (exact_for_score - exact_for_score.mean()) / exact_for_score.std().clamp_min(1e-6)
                            norm_mae = (approx_norm - exact_norm).abs().mean()
                            approx_rank = torch.argsort(torch.argsort(approx)).float()
                            exact_rank = torch.argsort(torch.argsort(exact_for_score)).float()
                            approx_rank = approx_rank - approx_rank.mean()
                            exact_rank = exact_rank - exact_rank.mean()
                            spearman = (approx_rank * exact_rank).sum() / (
                                approx_rank.square().sum().sqrt() * exact_rank.square().sum().sqrt()
                            ).clamp_min(1e-12)
                            diagnostics.score_mae_sum += float(raw_mae.item())
                            diagnostics.score_normalized_mae_sum += float(norm_mae.item())
                            diagnostics.spearman_sum += float(spearman.item())
                            diagnostics.score_diagnostic_calls += 1
                if use_batched_sparse_attention:
                    batched_attention_indices.append(indices)
                    batched_attention_valid.append(index_valid)
                    continue
                assert keys is not None and values is not None
                (selected_keys, selected_values), gather_cpu_ms = _timed_component(
                    lambda: gather_selected_kv(keys, values, indices),
                    device=query.device,
                    cuda_events=gather_events,
                )
                diagnostics.selected_gather_ms += gather_cpu_ms
                head_output, attention_cpu_ms = _timed_component(
                    lambda: exact_attention_selected(
                        query, selected_keys, selected_values
                    ),
                    device=query.device,
                    cuda_events=attention_events,
                )
                diagnostics.selected_attention_ms += attention_cpu_ms
                head_outputs.append(head_output)

            if use_batched_sparse_attention:
                all_indices = torch.stack(batched_attention_indices, dim=0)
                all_valid = torch.stack(batched_attention_valid, dim=0)
                def gather_sparse_layer() -> tuple[torch.Tensor, torch.Tensor]:
                    return gather_selected_kv_batched(
                        prefix_keys[0], prefix_values[0], key_states[0, :, 0],
                        value_states[0, :, 0], all_indices, self.head_to_kv,
                    )
                (selected_keys, selected_values), gather_cpu_ms = _timed_component(
                    gather_sparse_layer, device=query_states.device, cuda_events=gather_events,
                )
                diagnostics.selected_gather_ms += gather_cpu_ms
                attention_mask = all_valid if batched_quest_indices is not None else None
                attention_output, attention_cpu_ms = _timed_component(
                    lambda: exact_attention_selected_batched(
                        query_states[0, :, 0], selected_keys, selected_values, attention_mask
                    ),
                    device=query_states.device, cuda_events=attention_events,
                )
                diagnostics.selected_attention_ms += attention_cpu_ms
            else:
                attention_output = torch.stack(head_outputs, dim=0)
            attention_output = attention_output.reshape(1, 1, -1).to(normalized.dtype)
            hidden_states = residual + attention.o_proj(attention_output)
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.backbone.norm(hidden_states)
        logits = self.model.lm_head(hidden_states).float()[0, -1]
        diagnostics.candidate_score_ms += _finish_cuda_component_timing(score_events)
        diagnostics.candidate_topk_ms += _finish_cuda_component_timing(topk_events)
        diagnostics.candidate_search_ms += (
            diagnostics.candidate_score_ms
            + diagnostics.candidate_topk_ms
            + _finish_cuda_component_timing(candidate_events)
        )
        diagnostics.selected_gather_ms += _finish_cuda_component_timing(gather_events)
        diagnostics.selected_attention_ms += _finish_cuda_component_timing(attention_events)
        if return_new_kv:
            return logits, diagnostics, new_keys_by_layer, new_values_by_layer
        return logits, diagnostics


def parse_methods(value: str | Sequence[str]) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    methods = tuple(dict.fromkeys(str(item).strip().lower() for item in items if str(item).strip()))
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected {SUPPORTED_METHODS}")
    if not methods:
        raise ValueError("At least one method is required")
    return methods


def parse_int_tuple(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        if not value.strip():
            return ()
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def build_wikitext2_blocks(
    tokenizer: Any,
    *,
    context_length: int,
    num_blocks: int,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> list[torch.Tensor]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets is required: pip install datasets") from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    required = num_blocks * (context_length + 1)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    stream: list[int] = []
    for row in dataset:
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        stream.extend(int(token_id) for token_id in token_ids)
        stream.append(int(eos_id))
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError(f"Dataset produced {len(stream)} tokens; {required} required")

    block_size = context_length + 1
    return [
        torch.tensor(stream[i * block_size : (i + 1) * block_size], dtype=torch.long)
        for i in range(num_blocks)
    ]


def _safe_perplexity(mean_nll: float) -> float:
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return math.inf


@dataclass
class PPLResult:
    samples: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> list[dict[str, Any]]:
        methods = sorted({str(row["method"]) for row in self.samples})
        result = []
        for method in methods:
            rows = [row for row in self.samples if row["method"] == method]
            mean_nll = sum(float(row["nll"]) for row in rows) / len(rows)
            recall_values = [
                float(row["topk_recall"])
                for row in rows
                if row.get("topk_recall") is not None
            ]
            result.append(
                {
                    "method": method,
                    "num_samples": len(rows),
                    "mean_nll": mean_nll,
                    "perplexity": _safe_perplexity(mean_nll),
                    "mean_candidate_ratio": (
                        sum(float(row["candidate_ratio"]) for row in rows) / len(rows)
                    ),
                    "mean_decode_ms": (
                        sum(float(row["decode_ms"]) for row in rows) / len(rows)
                    ),
                    "total_decode_ms": sum(float(row["decode_ms"]) for row in rows),
                    "mean_candidate_search_ms": (
                        sum(float(row["candidate_search_ms"]) for row in rows) / len(rows)
                    ),
                    "total_candidate_search_ms": sum(
                        float(row["candidate_search_ms"]) for row in rows
                    ),
                    "mean_candidate_search_ops_proxy": (
                        sum(float(row.get("candidate_search_ops_proxy", 0.0)) for row in rows)
                        / len(rows)
                    ),
                    "total_candidate_search_ops_proxy": sum(
                        float(row.get("candidate_search_ops_proxy", 0.0)) for row in rows
                    ),
                    "mean_selected_attention_ms": (
                        sum(float(row["selected_attention_ms"]) for row in rows) / len(rows)
                    ),
                    "total_selected_attention_ms": sum(
                        float(row["selected_attention_ms"]) for row in rows
                    ),
                    "mean_topk_recall": (
                        None if not recall_values else sum(recall_values) / len(recall_values)
                    ),
                }
            )
        return result

    def write(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        with (output_path / "samples.jsonl").open("w") as handle:
            for row in self.samples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.samples:
            sample_fields = list(
                dict.fromkeys(key for row in self.samples for key in row.keys())
            )
            with (output_path / "samples.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sample_fields)
                writer.writeheader()
                writer.writerows(self.samples)
        with (output_path / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        if summary:
            with (output_path / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)


@torch.inference_mode()
def evaluate_sampled_decode_ppl(
    decoder: LlamaSparseDecoder,
    blocks: Sequence[torch.Tensor],
    *,
    methods: Sequence[str],
    query_positions: Sequence[int],
    method_labels: Mapping[str, str] | None = None,
    method_variants: Sequence[EvaluationVariant] | None = None,
) -> PPLResult:
    method_labels = {} if method_labels is None else dict(method_labels)
    if method_variants is None:
        methods = parse_methods(methods)
        variants = tuple(
            EvaluationVariant(
                method=method,
                label=method_labels.get(method, method),
                retrieval=decoder.retrieval,
            )
            for method in methods
        )
    else:
        variants = tuple(method_variants)
        if not variants:
            raise ValueError("method_variants cannot be empty")
        unknown = [variant.method for variant in variants if variant.method not in SUPPORTED_METHODS]
        if unknown:
            raise ValueError(f"Unknown variant methods: {unknown}")
    result = PPLResult()

    for block_idx, block in enumerate(blocks):
        for query_position in query_positions:
            qpos = int(query_position)
            if qpos < 1 or qpos + 1 >= block.numel():
                raise ValueError(
                    f"query position {qpos} invalid for block containing {block.numel()} tokens"
                )
            prefix_ids = block[:qpos]
            current_id = block[qpos]
            target_id = block[qpos + 1].to(decoder.device)

            prefix_started = time.perf_counter()
            past_key_values = decoder.build_prefix_cache(prefix_ids)
            if decoder.device.type == "cuda":
                torch.cuda.synchronize(decoder.device)
            prefix_ms = (time.perf_counter() - prefix_started) * 1000.0

            for variant in variants:
                method = variant.method
                method_label = variant.label
                decoder.retrieval = variant.retrieval
                if decoder.device.type == "cuda":
                    torch.cuda.synchronize(decoder.device)
                started = time.perf_counter()
                logits, diagnostics = decoder.decode(
                    past_key_values,
                    current_id,
                    method=method,
                )
                if decoder.device.type == "cuda":
                    torch.cuda.synchronize(decoder.device)
                decode_ms = (time.perf_counter() - started) * 1000.0
                nll = F.cross_entropy(logits.view(1, -1), target_id.view(1)).item()
                row = {
                    "block_idx": block_idx,
                    "query_position": qpos,
                    "context_tokens": qpos + 1,
                    "target_token_id": int(target_id),
                    "method": method_label,
                    "nll": float(nll),
                    "candidate_ratio": diagnostics.candidate_ratio,
                    "sparse_head_calls": diagnostics.sparse_head_calls,
                    "prefix_ms_shared": prefix_ms,
                    "decode_ms": decode_ms,
                    "candidate_search_ms": diagnostics.candidate_search_ms,
                    "candidate_search_ops_proxy": diagnostics.candidate_search_ops,
                    "selected_attention_ms": diagnostics.selected_attention_ms,
                    "single_token_decode_ms": decode_ms,
                    "single_token_search_ms": diagnostics.candidate_search_ms,
                    "single_token_attention_ms": diagnostics.selected_attention_ms,
                    "topk_recall": diagnostics.topk_recall,
                }
                if method == "pqsift":
                    row["pqsift_axes"] = variant.retrieval.pqsift_axes
                    row["pqsift_r"] = variant.retrieval.pqsift_keep_ratio
                if method in {"loki", "quest", "fier", "bit2_qk"}:
                    row["budget"] = variant.retrieval.budget
                if method == "fier":
                    row["group_size"] = variant.retrieval.fier_group_size
                if method == "bit2_qk":
                    row["group_size"] = variant.retrieval.bit2_group_size
                result.samples.append(row)
                print(
                    f"block={block_idx} qpos={qpos} method={method_label} "
                    f"nll={nll:.5f} ppl={_safe_perplexity(nll):.3f} "
                    f"candidates={diagnostics.candidate_ratio:.4f} "
                    f"search_ms={diagnostics.candidate_search_ms:.1f} "
                    f"attention_ms={diagnostics.selected_attention_ms:.1f} "
                    f"decode_ms={decode_ms:.1f}",
                    flush=True,
                )

            del past_key_values
            if decoder.device.type == "cuda":
                torch.cuda.empty_cache()
    return result
