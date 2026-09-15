"""Validation for optional preference-training objective configuration."""

from __future__ import annotations

import math


def normalize_negative_suppression_config(config: dict) -> dict:
    """Install and validate backwards-compatible negative-suppression defaults."""
    preference = config["preference"]
    provided = preference.get("negative_suppression") or {}
    if not isinstance(provided, dict):
        raise ValueError("preference.negative_suppression must be a mapping")
    unknown = set(provided) - {"enabled", "weight", "threshold", "temperature", "gate"}
    if unknown:
        raise ValueError(f"Unknown negative suppression settings: {sorted(unknown)}")
    resolved = {
        "enabled": False,
        "weight": 0.5,
        "threshold": None,
        "temperature": 0.01,
        "gate": "violation",
    }
    resolved.update(provided)
    resolved["enabled"] = bool(resolved["enabled"])
    resolved["weight"] = float(resolved["weight"])
    resolved["temperature"] = float(resolved["temperature"])
    if resolved["threshold"] is not None:
        resolved["threshold"] = float(resolved["threshold"])
    if resolved["weight"] < 0:
        raise ValueError("preference.negative_suppression.weight must be non-negative")
    if not math.isfinite(resolved["temperature"]) or resolved["temperature"] <= 0:
        raise ValueError("preference.negative_suppression.temperature must be positive and finite")
    if resolved["threshold"] is not None and not math.isfinite(resolved["threshold"]):
        raise ValueError("preference.negative_suppression.threshold must be finite when provided")
    if resolved["gate"] not in {"violation", "none"}:
        raise ValueError("preference.negative_suppression.gate must be 'violation' or 'none'")
    preference["negative_suppression"] = resolved
    return resolved


def validate_negative_suppression_for_training(config: dict) -> None:
    """Require an explicit YAML threshold for an active suppression objective."""
    suppression = normalize_negative_suppression_config(config)
    if not suppression["enabled"] or suppression["weight"] == 0:
        return
    if suppression["threshold"] is None:
        raise ValueError(
            "negative suppression is enabled but threshold is unset; set "
            "preference.negative_suppression.threshold in the YAML configuration"
        )
