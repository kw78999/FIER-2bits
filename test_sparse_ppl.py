from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sparse_ppl import (
    Bit2Thresholds,
    PCABasis,
    PCABasisCache,
    RetrievalConfig,
    bit2_approximate_scores,
    bit2_interaction_scores_from_bits,
    bit2_interaction_scores_packed,
    exact_attention,
    fier_dequantize_1bit,
    parse_methods,
    pack_bool_bits,
    quantize_bit2,
    quantize_grouped_keys_bit2,
    quantize_query_bit2,
    select_candidates,
    select_fier,
)


def _old_bit2_score(query_sign, query_mag, key_sign, key_mag):
    same_sign = key_sign == query_sign.unsqueeze(0)
    both_large = key_mag & query_mag.unsqueeze(0)
    either_large = key_mag | query_mag.unsqueeze(0)
    dimension = query_sign.numel()
    return (
        2 * same_sign.sum(dim=-1, dtype=torch.int32) - dimension
        + 2 * (same_sign & either_large).sum(dim=-1, dtype=torch.int32)
        - either_large.sum(dim=-1, dtype=torch.int32)
        + 2 * (same_sign & both_large).sum(dim=-1, dtype=torch.int32)
        - both_large.sum(dim=-1, dtype=torch.int32)
    )


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.keys = torch.randn(65, 8)
        self.query = torch.randn(8)
        self.values = torch.randn(65, 8)
        self.basis = PCABasis(
            mean=self.keys.mean(0),
            components=torch.linalg.svd(self.keys - self.keys.mean(0), full_matrices=False).Vh.T,
        )
        self.config = RetrievalConfig(
            budget=16,
            loki_rank=4,
            pqsift_axes=3,
            pqsift_keep_ratio=0.75,
            quest_page_size=8,
            fier_group_size=8,
        )

    def test_all_selectors_include_current_and_are_valid(self) -> None:
        for method in ("full", "pqsift", "loki", "quest", "fier", "bit2_qk"):
            basis = self.basis if method in {"pqsift", "loki"} else None
            indices = select_candidates(method, self.query, self.keys, self.config, basis)
            self.assertTrue(bool(torch.any(indices == self.keys.shape[0] - 1)), method)
            self.assertGreater(indices.numel(), 0, method)
            self.assertGreaterEqual(int(indices.min()), 0, method)
            self.assertLess(int(indices.max()), self.keys.shape[0], method)
            output = exact_attention(self.query, self.keys, self.values, indices)
            self.assertEqual(tuple(output.shape), (8,))
            self.assertTrue(torch.isfinite(output).all(), method)

    def test_budgeted_token_selectors_respect_budget(self) -> None:
        for method in ("loki", "fier", "bit2_qk"):
            basis = self.basis if method == "loki" else None
            indices = select_candidates(method, self.query, self.keys, self.config, basis)
            self.assertLessEqual(indices.numel(), self.config.budget)

    def test_fier_dequantization_has_two_values_per_group_channel(self) -> None:
        dequantized = fier_dequantize_1bit(self.keys, group_size=8)
        self.assertEqual(dequantized.shape, self.keys.shape)
        for start in range(0, self.keys.shape[0], 8):
            group = dequantized[start : start + 8]
            for channel in range(group.shape[1]):
                self.assertLessEqual(torch.unique(group[:, channel]).numel(), 2)

    def test_parse_methods(self) -> None:
        self.assertEqual(parse_methods("full,bit2_qk,full"), ("full", "bit2_qk"))
        with self.assertRaises(ValueError):
            parse_methods("unknown")

    def test_bit2_truth_tables(self) -> None:
        for query_sign in (False, True):
            for key_sign in (False, True):
                self.assertEqual(query_sign == key_sign, not (query_sign ^ key_sign))
        for query_mag, key_mag, expected_and, expected_or in (
            (False, False, False, False),
            (False, True, False, True),
            (True, False, False, True),
            (True, True, True, True),
        ):
            self.assertEqual(query_mag and key_mag, expected_and)
            self.assertEqual(query_mag or key_mag, expected_or)

    def test_bit2_count_formula_matches_direct_channel_contributions(self) -> None:
        torch.manual_seed(9)
        query_sign = torch.randint(0, 2, (13,), dtype=torch.bool)
        query_mag = torch.randint(0, 2, (13,), dtype=torch.bool)
        key_sign = torch.randint(0, 2, (7, 13), dtype=torch.bool)
        key_mag = torch.randint(0, 2, (7, 13), dtype=torch.bool)
        formula = bit2_interaction_scores_from_bits(
            query_sign, query_mag, key_sign, key_mag
        )
        direct = []
        for token in range(key_sign.shape[0]):
            score = 0
            for channel in range(key_sign.shape[1]):
                same = bool(query_sign[channel] == key_sign[token, channel])
                q_large = bool(query_mag[channel])
                k_large = bool(key_mag[token, channel])
                weight = 1 + int(q_large or k_large) + int(q_large and k_large)
                score += (1 if same else -1) * weight
            direct.append(score)
        self.assertTrue(torch.equal(formula.cpu(), torch.tensor(direct, dtype=torch.int32)))

    def test_packed_popcount_matches_bool_reference_with_padding(self) -> None:
        torch.manual_seed(17)
        query_sign = torch.randint(0, 2, (13,), dtype=torch.bool)
        query_mag = torch.randint(0, 2, (13,), dtype=torch.bool)
        key_sign = torch.randint(0, 2, (9, 13), dtype=torch.bool)
        key_mag = torch.randint(0, 2, (9, 13), dtype=torch.bool)
        reference = bit2_interaction_scores_from_bits(
            query_sign, query_mag, key_sign, key_mag
        )
        packed = bit2_interaction_scores_packed(
            pack_bool_bits(query_sign),
            pack_bool_bits(query_mag),
            pack_bool_bits(key_sign),
            pack_bool_bits(key_mag),
            dimension=13,
        )
        self.assertTrue(torch.equal(reference, packed))


    def test_three_popc_matches_old_formula_exhaustive_single_bit(self) -> None:
        for qs in (False, True):
            for qm in (False, True):
                for ks in (False, True):
                    for km in (False, True):
                        query_sign = torch.tensor([qs], dtype=torch.bool)
                        query_mag = torch.tensor([qm], dtype=torch.bool)
                        key_sign = torch.tensor([[ks]], dtype=torch.bool)
                        key_mag = torch.tensor([[km]], dtype=torch.bool)
                        new_score = bit2_interaction_scores_from_bits(
                            query_sign, query_mag, key_sign, key_mag
                        )
                        old_score = _old_bit2_score(query_sign, query_mag, key_sign, key_mag)
                        self.assertTrue(torch.equal(new_score, old_score))

    def test_three_popc_matches_old_formula_random_dimensions(self) -> None:
        for dim in (31, 32, 64, 95, 96, 127, 128):
            torch.manual_seed(dim)
            query_sign = torch.randint(0, 2, (dim,), dtype=torch.bool)
            query_mag = torch.randint(0, 2, (dim,), dtype=torch.bool)
            key_sign = torch.randint(0, 2, (11, dim), dtype=torch.bool)
            key_mag = torch.randint(0, 2, (11, dim), dtype=torch.bool)
            new_score = bit2_interaction_scores_from_bits(
                query_sign, query_mag, key_sign, key_mag
            )
            old_score = _old_bit2_score(query_sign, query_mag, key_sign, key_mag)
            self.assertTrue(torch.equal(new_score, old_score), dim)

    def test_three_popc_packed_matches_unpacked_random_dimensions(self) -> None:
        for dim in (31, 32, 64, 95, 96, 127, 128):
            torch.manual_seed(1000 + dim)
            query_sign = torch.randint(0, 2, (dim,), dtype=torch.bool)
            query_mag = torch.randint(0, 2, (dim,), dtype=torch.bool)
            key_sign = torch.randint(0, 2, (13, dim), dtype=torch.bool)
            key_mag = torch.randint(0, 2, (13, dim), dtype=torch.bool)
            reference = bit2_interaction_scores_from_bits(
                query_sign, query_mag, key_sign, key_mag
            )
            packed = bit2_interaction_scores_packed(
                pack_bool_bits(query_sign),
                pack_bool_bits(query_mag),
                pack_bool_bits(key_sign),
                pack_bool_bits(key_mag),
                dimension=dim,
            )
            self.assertTrue(torch.equal(reference, packed), dim)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_popc_matches_python_packed_score(self) -> None:
        from bit2_cuda import histogram_topk_from_scores, score_tensors

        for dtype in (torch.float16, torch.bfloat16):
            for batch in (1, 2, 4):
                for dim in (31, 32, 64, 96, 128):
                    torch.manual_seed(2000 + batch + dim)
                    q_heads = 6
                    kv_heads = 2
                    tokens = 37
                    group_size = 8
                    queries = torch.randn(batch, q_heads, dim, device="cuda", dtype=dtype)
                    keys = torch.randn(kv_heads, tokens, dim, device="cuda", dtype=dtype)
                    head_to_kv = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda")
                    scores = score_tensors(
                        queries,
                        keys,
                        head_to_kv=head_to_kv,
                        valid_tokens=torch.tensor([tokens], device="cuda"),
                        group_size=group_size,
                    ).cpu()
                    expected = torch.empty_like(scores)
                    for b in range(batch):
                        for qh in range(q_heads):
                            kvh = int(head_to_kv[qh])
                            expected[b, qh] = bit2_approximate_scores(
                                queries[b, qh].float().cpu(),
                                keys[kvh].float().cpu(),
                                group_size=group_size,
                            )
                    self.assertTrue(torch.equal(scores, expected), (dtype, batch, dim))
                    hist = histogram_topk_from_scores(
                        scores.cuda(),
                        torch.tensor([tokens], device="cuda"),
                        budget=9,
                        head_dim=dim,
                    ).cpu()
                    for b in range(batch):
                        for qh in range(q_heads):
                            row = scores[b, qh]
                            deterministic = sorted(range(tokens), key=lambda i: (-int(row[i]), i))[:9]
                            self.assertEqual(hist[b, qh].tolist(), deterministic)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fier_triton_matches_reference_scores_and_topk(self) -> None:
        from fier_triton import score_tensors

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tokens in (1, 31, 32, 37, 65):
                for group_size in (8, 32, 64):
                    torch.manual_seed(3000 + tokens + group_size)
                    queries = torch.randn(2, 6, 13, device="cuda", dtype=dtype)
                    keys = torch.randn(2, tokens, 13, device="cuda", dtype=dtype)
                    head_to_kv = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda")
                    scores = score_tensors(
                        queries,
                        keys,
                        head_to_kv=head_to_kv,
                        group_size=group_size,
                    )
                    expected = torch.stack(
                        [
                            torch.stack(
                                [
                                    fier_dequantize_1bit(
                                        keys[int(head_to_kv[head])],
                                        group_size=group_size,
                                    )
                                    @ queries[batch, head].float()
                                    for head in range(6)
                                ]
                            )
                            for batch in range(2)
                        ]
                    )
                    self.assertTrue(
                        torch.allclose(scores, expected, rtol=0.0, atol=2e-5),
                        (dtype, tokens, group_size),
                    )

        query = torch.randn(13, device="cuda", dtype=torch.float16)
        keys = torch.randn(37, 13, device="cuda", dtype=torch.float16)
        reference = select_fier(
            query, keys, group_size=8, budget=9, backend="reference"
        )
        optimized = select_fier(
            query, keys, group_size=8, budget=9, backend="triton"
        )
        self.assertTrue(torch.equal(reference, optimized))

    def test_two_bitplanes_use_two_physical_bits_per_scalar(self) -> None:
        bits = torch.zeros((7, 128), dtype=torch.bool)
        sign = pack_bool_bits(bits)
        magnitude = pack_bool_bits(bits)
        self.assertEqual(sign.shape, (7, 16))
        packed_bytes = (
            sign.numel() * sign.element_size()
            + magnitude.numel() * magnitude.element_size()
        )
        self.assertEqual(packed_bytes, 7 * 128 * 2 // 8)

    def test_bit2_quantization_boundaries(self) -> None:
        thresholds = Bit2Thresholds(neg=torch.tensor([-0.5]), pos=torch.tensor([0.5]))
        values = torch.tensor([[-1.0], [-0.25], [0.0], [0.25], [1.0]])
        sign, magnitude = quantize_bit2(values, thresholds)
        self.assertEqual(sign.flatten().tolist(), [False, False, True, True, True])
        self.assertEqual(magnitude.flatten().tolist(), [True, False, False, False, True])

    def test_query_thresholds_come_from_current_head_dimensions(self) -> None:
        query = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        sign, magnitude = quantize_query_bit2(query)
        self.assertEqual(sign.tolist(), [False, False, False, True, True, True, True])
        self.assertEqual(magnitude.tolist(), [True, False, False, False, False, False, True])

    def test_key_thresholds_are_independent_per_token_group_and_channel(self) -> None:
        keys = torch.tensor(
            [
                [-5.0, -50.0], [-3.0, -30.0], [-1.0, -10.0], [2.0, 20.0], [4.0, 40.0],
                [-50.0, -5.0], [-30.0, -3.0], [-10.0, -1.0], [20.0, 2.0], [40.0, 4.0],
            ]
        )
        _, magnitude = quantize_grouped_keys_bit2(keys, group_size=5)
        expected = torch.tensor(
            [[True, True], [True, True], [False, False], [False, False], [True, True]] * 2
        )
        self.assertTrue(torch.equal(magnitude, expected))


class CacheTests(unittest.TestCase):
    def test_cache_validation(self) -> None:
        payload = {
            "metadata": {
                "cache_version": "pca_basis_cache_v2",
                "model_id": "test/model",
            },
            "basis_by_pair": {
                "0:0": {
                    "mean": torch.zeros(4),
                    "components": torch.eye(4),
                    "eigvals": torch.ones(4),
                    "total_variance": 4.0,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.pt"
            torch.save(payload, path)
            cache = PCABasisCache.load(
                path,
                expected_model_id="test/model",
                min_axes=4,
                expected_layers=1,
                expected_heads=1,
            )
            self.assertEqual(len(cache.bases), 1)
            with self.assertRaises(ValueError):
                PCABasisCache.load(path, expected_model_id="wrong/model")


if __name__ == "__main__":
    unittest.main()
