"""Processor-local evidence scores and normalization metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceScore:
    processor: str
    score_kind: str
    # Raw scores may be arbitrary finite model logits.  Only normalized_score
    # is required to live in [0, 1] and may be consumed by a gate.
    raw_score: float | None
    normalized_score: float | None
    margin: float | None = None
    claimed_value: Any = None
    support: str = "SUPPORTED"
    normalizer_revision: str = "unspecified"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    view_id: str | None = None
    claim_fingerprint: str | None = None
    # Evidence-layer semantics.  A monotone normalization is not a calibrated
    # probability unless a separate calibration artifact exists.
    normalization_semantics: str = "unspecified"
    calibrated: bool = False
    evidence_kind: str = "MODEL_DERIVED_EVIDENCE"

    def __post_init__(self) -> None:
        for name in ("raw_score", "margin"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present")
        if self.normalized_score is not None:
            value = float(self.normalized_score)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("normalized_score must be finite and in [0,1] when present")
        if self.support not in {"SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"}:
            raise ValueError(f"unknown evidence support: {self.support}")
        if self.support != "SUPPORTED" and self.normalized_score is not None:
            raise ValueError("unsupported/ambiguous evidence must not expose a normalized certification score")

    @classmethod
    def unsupported(
        cls,
        processor: str,
        *,
        reason: str,
        claimed_value: Any = None,
        diagnostics: dict[str, Any] | None = None,
        view_id: str | None = None,
        claim_fingerprint: str | None = None,
        evidence_kind: str = "UNSUPPORTED_EVIDENCE",
    ) -> "EvidenceScore":
        merged = {"reason": reason, **dict(diagnostics or {})}
        return cls(
            processor,
            "unsupported",
            None,
            None,
            None,
            claimed_value,
            "UNSUPPORTED",
            "none",
            merged,
            view_id,
            claim_fingerprint,
            "not_applicable",
            False,
            evidence_kind,
        )

    def to_dict(self) -> dict[str, Any]:
        claimed = self.claimed_value
        if hasattr(claimed, "bbox_xyxy_pixels"):
            claimed = {
                "bbox_xyxy_pixels": list(claimed.bbox_xyxy_pixels),
                "bbox_xyxy_normalized": list(claimed.bbox_xyxy_normalized),
                "score": None if claimed.score is None else float(claimed.score),
                "score_available": bool(claimed.score_available),
                "source": str(claimed.source),
            }
        return {
            "processor": self.processor,
            "score_kind": self.score_kind,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "margin": self.margin,
            "claimed_value": claimed,
            "support": self.support,
            "normalizer_revision": self.normalizer_revision,
            "diagnostics": self.diagnostics,
            "view_id": self.view_id,
            "claim_fingerprint": self.claim_fingerprint,
            "normalization_semantics": self.normalization_semantics,
            "calibrated": self.calibrated,
            "evidence_kind": self.evidence_kind,
        }
