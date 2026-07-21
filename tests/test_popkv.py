from __future__ import annotations

import math
import unittest

import torch

from popkv import (
    METHODS, RetrievalConfig, append_quest, build_quest_cache,
    popkv_reference_scores, quest_indices, quest_scores,
)


class PopKVTests(unittest.TestCase):
    def test_public_methods_are_only_paper_methods(self) -> None:
        self.assertEqual(METHODS, ("full", "quest", "fier", "popkv"))

    def test_release_configuration_is_explicit(self) -> None:
        RetrievalConfig().validate()
        with self.assertRaises(ValueError):
            RetrievalConfig(group_size=64).validate()

    def test_quest_cache_matches_direct_page_bounds_across_boundary(self) -> None:
        torch.manual_seed(3)
        keys = torch.randn(8, 31, 16, dtype=torch.bfloat16)
        mapping = torch.arange(32) // 4
        cache = build_quest_cache(keys, page_size=8, reserve_tokens=8)
        for _ in range(4):
            queries = torch.randn(32, 16, dtype=torch.bfloat16)
            current = torch.randn(8, 16, dtype=torch.bfloat16)
            cached = quest_scores(queries, cache, current, mapping)
            all_keys = torch.cat([keys, current[:, None]], dim=1).float()
            pad = (-all_keys.shape[1]) % 8
            if pad:
                all_keys = torch.cat([all_keys, all_keys[:, -1:].expand(8, pad, 16)], dim=1)
            pages = all_keys.view(8, math.ceil(all_keys.shape[1] / 8), 8, 16)
            mins, maxs = pages.amin(2).index_select(0, mapping), pages.amax(2).index_select(0, mapping)
            direct = torch.maximum(queries.float()[:, None] * mins, queries.float()[:, None] * maxs).sum(-1)
            self.assertTrue(torch.equal(cached, direct))
            left = quest_indices(cached, num_tokens=keys.shape[1] + 1, page_size=8, budget=16)
            right = quest_indices(direct, num_tokens=keys.shape[1] + 1, page_size=8, budget=16)
            self.assertTrue(torch.equal(left[0], right[0]) and torch.equal(left[1], right[1]))
            append_quest(cache, current[:, None]); keys = torch.cat([keys, current[:, None]], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_popkv_cuda_matches_readable_reference(self) -> None:
        from popkv_cuda import pack_keys, score
        torch.manual_seed(11)
        keys = torch.randn(2, 37, 64, device="cuda", dtype=torch.bfloat16)
        queries = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        mapping = torch.arange(8, device="cuda") // 4
        expected = popkv_reference_scores(queries, keys, head_to_kv=mapping)
        actual = score(queries.contiguous(), pack_keys(keys.contiguous()), head_to_kv=mapping, tokens=37)
        torch.testing.assert_close(actual, expected, atol=0.12, rtol=0.02)


if __name__ == "__main__": unittest.main()
