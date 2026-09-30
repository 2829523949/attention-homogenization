"""Explicit processor-specific score normalization."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .evidence import EvidenceScore


def _kwargs(diagnostics: Mapping[str, Any] | None, view_id: str | None, claim_fingerprint: str | None) -> dict[str, Any]:
    return {
        "diagnostics": dict(diagnostics or {}),
        "view_id": view_id,
        "claim_fingerprint": claim_fingerprint,
    }


def normalize_probability(
    processor: str,
    value: Any,
    *,
    claimed_value: Any,
    diagnostics: Mapping[str, Any] | None = None,
    view_id: str | None = None,
    claim_fingerprint: str | None = None,
) -> EvidenceScore:
    raw = float(value)
    return EvidenceScore(
        processor,
        "probability",
        raw,
        max(0.0, min(1.0, raw)),
        None,
        claimed_value,
        "SUPPORTED",
        "probability_identity_v1",
        **_kwargs(diagnostics, view_id, claim_fingerprint),
    )


def normalize_confidence(
    processor: str,
    value: Any,
    *,
    claimed_value: Any,
    diagnostics: Mapping[str, Any] | None = None,
    view_id: str | None = None,
    claim_fingerprint: str | None = None,
) -> EvidenceScore:
    raw = float(value)
    return EvidenceScore(
        processor,
        "processor_confidence",
        raw,
        max(0.0, min(1.0, raw)),
        None,
        claimed_value,
        "SUPPORTED",
        f"{processor}:confidence_identity_v1",
        **_kwargs(diagnostics, view_id, claim_fingerprint),
    )


def normalize_sigmoid_margin(
    processor: str,
    *,
    margin: float,
    raw_logit: float,
    claimed_value: Any,
    diagnostics: Mapping[str, Any] | None = None,
    view_id: str | None = None,
    claim_fingerprint: str | None = None,
) -> EvidenceScore:
    """Normalize a real target-vs-best-alternative logit margin.

    The normalized score is ``sigmoid(margin)``.  This is a monotone,
    documented normalization of a model-derived contrastive margin, not a
    fabricated confidence.
    """

    value = float(margin)
    normalized = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
    return EvidenceScore(
        processor,
        "fixed_candidate_logit_margin",
        float(raw_logit),
        normalized,
        value,
        claimed_value,
        "SUPPORTED",
        f"{processor}:sigmoid_logit_margin_v1",
        **_kwargs(diagnostics, view_id, claim_fingerprint),
    )
