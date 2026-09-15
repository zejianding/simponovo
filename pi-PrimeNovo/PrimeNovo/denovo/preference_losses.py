"""Tensor-only loss terms shared by SimPO preference training and unit tests."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_negative_suppression(
    positive_nll: torch.Tensor,
    negative_nll: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
    gate: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean suppression loss, pair gate, and strong-suppression mask."""
    if positive_nll.ndim != 1 or negative_nll.ndim != 2:
        raise ValueError("positive_nll must be [batch] and negative_nll must be [batch, K]")
    if negative_nll.shape[0] != positive_nll.shape[0]:
        raise ValueError("positive_nll and negative_nll batch sizes must match")
    if temperature <= 0:
        raise ValueError("negative suppression temperature must be positive")
    if gate == "violation":
        pair_gate = (negative_nll.detach() < positive_nll.detach()[:, None]).to(
            dtype=negative_nll.dtype
        )
    elif gate == "none":
        pair_gate = torch.ones_like(negative_nll)
    else:
        raise ValueError(f"Unsupported negative suppression gate: {gate}")

    per_pair_loss = pair_gate * temperature * F.softplus((threshold - negative_nll) / temperature)
    strong_mask = pair_gate.bool() & (negative_nll.detach() < threshold)
    return per_pair_loss.mean(), pair_gate, strong_mask


def compute_preference_loss_terms(
    rewards: torch.Tensor,
    normalized_nll: torch.Tensor,
    *,
    beta: float,
    target_margin: float,
    positive_ctc_weight: float,
    negative_suppression_enabled: bool,
    negative_suppression_weight: float,
    negative_suppression_threshold: float | None,
    negative_suppression_temperature: float,
    negative_suppression_gate: str,
) -> Dict[str, torch.Tensor]:
    """Compute SimPO, positive CTC, and optional negative-suppression terms."""
    if rewards.shape != normalized_nll.shape or normalized_nll.ndim != 2:
        raise ValueError("rewards and normalized_nll must share shape [batch, 1 + K]")
    if normalized_nll.shape[1] < 2:
        raise ValueError("preference loss requires at least one negative")

    positive_rewards = rewards[:, 0]
    negative_rewards = rewards[:, 1:]
    margins = positive_rewards[:, None] - negative_rewards
    # Keep the existing SimPO parameterization exactly: target_margin is inside beta.
    simpo_loss = -F.logsigmoid(beta * (margins - target_margin)).mean()
    positive_nll = normalized_nll[:, 0]
    negative_nll = normalized_nll[:, 1:]
    positive_ctc_loss = positive_nll.mean()
    baseline_total_loss = simpo_loss + positive_ctc_weight * positive_ctc_loss
    violation_mask = negative_nll.detach() < positive_nll.detach()[:, None]
    zero = baseline_total_loss.new_zeros(())

    if negative_suppression_enabled and negative_suppression_weight > 0:
        if negative_suppression_threshold is None:
            raise RuntimeError("Negative suppression threshold was not resolved before training")
        negative_suppression_loss, negative_suppression_gate, strong_suppression_mask = (
            compute_negative_suppression(
                positive_nll,
                negative_nll,
                threshold=negative_suppression_threshold,
                temperature=negative_suppression_temperature,
                gate=negative_suppression_gate,
            )
        )
        weighted_negative_suppression_loss = (
            negative_suppression_weight * negative_suppression_loss
        )
        total_loss = baseline_total_loss + weighted_negative_suppression_loss
        high_confidence_mask = negative_nll.detach() < negative_suppression_threshold
    else:
        # Keep the old objective on an exact code path when disabled or weight is zero.
        negative_suppression_loss = zero
        weighted_negative_suppression_loss = zero
        negative_suppression_gate = torch.zeros_like(negative_nll)
        strong_suppression_mask = torch.zeros_like(negative_nll, dtype=torch.bool)
        high_confidence_mask = torch.zeros_like(negative_nll, dtype=torch.bool)
        total_loss = baseline_total_loss

    return {
        "total_loss": total_loss,
        "simpo_loss": simpo_loss,
        "positive_ctc_loss": positive_ctc_loss,
        "negative_suppression_loss": negative_suppression_loss,
        "weighted_negative_suppression_loss": weighted_negative_suppression_loss,
        "positive_rewards": positive_rewards,
        "negative_rewards": negative_rewards,
        "positive_nll": positive_nll,
        "negative_nll": negative_nll,
        "margins": margins,
        "violation_mask": violation_mask,
        "negative_suppression_gate": negative_suppression_gate,
        "high_confidence_mask": high_confidence_mask,
        "strong_suppression_mask": strong_suppression_mask,
    }
