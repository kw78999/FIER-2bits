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
