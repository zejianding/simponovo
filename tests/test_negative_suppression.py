"""Unit tests for the tensor-only negative-suppression objective."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pi-PrimeNovo"))

from PrimeNovo.denovo.preference_config import (  # noqa: E402
    normalize_negative_suppression_config,
    validate_negative_suppression_for_training,
)
from PrimeNovo.denovo.preference_losses import (  # noqa: E402
    compute_preference_loss_terms,
    compute_negative_suppression,
)


class NegativeSuppressionTensorTest(unittest.TestCase):
    def test_soft_threshold_is_monotonic_for_any_k(self) -> None:
        positive = torch.full((2,), 1.0)
        for width in (1, 2, 4):
            negative = torch.tensor([[0.001, 0.05, 0.15, 0.50]] * 2)[:, :width]
            loss, gate, _ = compute_negative_suppression(
                positive, negative, threshold=0.15, temperature=0.05, gate="violation"
            )
            self.assertTrue(torch.all(gate == 1))
            self.assertTrue(torch.isfinite(loss))
            per_value = 0.05 * F.softplus((0.15 - negative[0]) / 0.05)
            self.assertTrue(torch.all(per_value[:-1] > per_value[1:]))

    def test_gradient_is_negative_and_has_mean_reduction(self) -> None:
        positive = torch.full((2,), 0.7, requires_grad=True)
        negative = torch.full((2, 3), 0.001, requires_grad=True)
        loss, gate, _ = compute_negative_suppression(
            positive, negative, threshold=0.15, temperature=0.01, gate="violation"
        )
        positive_grad, negative_grad = torch.autograd.grad(
            loss, (positive, negative), allow_unused=True
        )
        self.assertTrue(torch.all(gate == 1))
        self.assertTrue(torch.all(negative_grad < 0))
        self.assertIsNone(positive_grad)
        self.assertTrue(torch.allclose(negative_grad, torch.full_like(negative, -1 / 6), atol=1e-5))

    def test_violation_gate_and_soft_threshold_semantics(self) -> None:
        positive = torch.tensor([0.1, 0.7])
        negative = torch.tensor([[0.2, 0.3], [0.001, 0.5]])
        loss, gate, strong_mask = compute_negative_suppression(
            positive, negative, threshold=0.15, temperature=0.05, gate="violation"
        )
        self.assertEqual(gate.tolist(), [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(strong_mask.tolist(), [[False, False], [True, False]])
        self.assertGreater(float(loss), 0.0)

    def test_none_gate_ignores_ranking(self) -> None:
        positive = torch.tensor([0.1])
        negative = torch.tensor([[0.2, 0.3]])
        loss, gate, _ = compute_negative_suppression(
            positive, negative, threshold=0.15, temperature=0.05, gate="none"
        )
        self.assertTrue(torch.all(gate == 1))
        self.assertGreater(float(loss), 0.0)


class PreferenceLossIntegrationTest(unittest.TestCase):
    def test_disabled_and_zero_weight_match_original_formula(self) -> None:
        normalized_nll = torch.tensor([[0.7, 0.001, 0.9], [0.2, 0.4, 0.1]])
        expected_margins = -normalized_nll[:, :1] - (-normalized_nll[:, 1:])
        expected = -F.logsigmoid(2.0 * (expected_margins - 0.3)).mean()
        expected = expected + 0.5 * normalized_nll[:, 0].mean()
        for enabled, weight, threshold in ((False, 0.5, None), (True, 0.0, None)):
            losses = compute_preference_loss_terms(
                -normalized_nll,
                normalized_nll,
                beta=2.0,
                target_margin=0.3,
                positive_ctc_weight=0.5,
                negative_suppression_enabled=enabled,
                negative_suppression_weight=weight,
                negative_suppression_threshold=threshold,
                negative_suppression_temperature=0.01,
                negative_suppression_gate="violation",
            )
            self.assertTrue(torch.equal(losses["total_loss"], expected))
            self.assertEqual(float(losses["negative_suppression_loss"]), 0.0)

    def test_enabled_loss_keeps_simpo_formula_and_k_dimension(self) -> None:
        normalized_nll = torch.tensor([[0.7, 0.001, 0.9]], requires_grad=True)
        losses = compute_preference_loss_terms(
            -normalized_nll,
            normalized_nll,
            beta=2.0,
            target_margin=0.3,
            positive_ctc_weight=0.5,
            negative_suppression_enabled=True,
            negative_suppression_weight=0.5,
            negative_suppression_threshold=0.15,
            negative_suppression_temperature=0.01,
            negative_suppression_gate="violation",
        )
        expected_simpo = -F.logsigmoid(2.0 * ((normalized_nll[:, 1:] - normalized_nll[:, :1]) - 0.3)).mean()
        self.assertTrue(torch.equal(losses["simpo_loss"], expected_simpo))
        self.assertEqual(tuple(losses["negative_nll"].shape), (1, 2))
        self.assertTrue(math.isfinite(float(losses["total_loss"].detach())))


class NegativeSuppressionConfigTest(unittest.TestCase):
    @staticmethod
    def _config(value=None) -> dict:
        preference = {"num_negatives": 2, "beta": 1.0, "target_margin": 0.0}
        if value is not None:
            preference["negative_suppression"] = value
        return {"preference": preference}

    def test_legacy_config_defaults_to_disabled(self) -> None:
        resolved = normalize_negative_suppression_config(self._config())
        self.assertFalse(resolved["enabled"])
        self.assertIsNone(resolved["threshold"])

    def test_active_training_requires_yaml_threshold(self) -> None:
        config = self._config({"enabled": True, "weight": 0.5})
        with self.assertRaisesRegex(ValueError, "threshold is unset"):
            validate_negative_suppression_for_training(config)

    def test_invalid_config_is_rejected(self) -> None:
        for value in (
            {"weight": -0.1},
            {"temperature": 0.0},
            {"gate": "unknown"},
            {"threshold": float("nan")},
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_negative_suppression_config(self._config(value))


if __name__ == "__main__":
    unittest.main()
