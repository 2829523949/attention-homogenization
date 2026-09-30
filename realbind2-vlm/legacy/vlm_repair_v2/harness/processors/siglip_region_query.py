"""Separate fixed-region / fixed-query semantic evidence scorer.

This is NOT a detector, NOT a Florence confidence, and NOT a full logical
identity certificate.  For a fixed persistent region R and a fixed legal
referent query q it measures:

    semantic support of q inside the visual content of R
    under the current counterfactual view.

The same region, crop policy, query string, template, model, and
normalization are used for RAW / FOCUS / REMOVE; only the view image changes.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from harness.certification.evidence import EvidenceScore
from harness.processors.base import RegionPrediction
from harness.processors.crop_policy import CROP_POLICY_ID
from harness.referents.region_query import RegionQuery


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def region_identity(entity_id: str, region: RegionPrediction) -> str:
    digest = hashlib.sha256(
        "|".join(f"{float(value):.6f}" for value in region.bbox_xyxy_pixels).encode("utf-8")
    ).hexdigest()[:16]
    return f"region_{entity_id}_{digest}"


class Siglip2RegionQueryScorer:
    name = "siglip2_region_query"
    version = "google_siglip2_base_patch16_224_region_query_v1"
    can_acquire = False
    supports_same_claim_rescoring = True
    supports_certification_score = True
    comparable_across_counterfactual_views = True
    requires_fixed_region = True
    requires_fixed_query = True
    score_semantics = "fixed_region_query_compatibility"
    score_normalization = "siglip2_query_logit_sigmoid_v1"
    evidence_kind = "FIXED_REGION_QUERY_COMPATIBILITY"

    def __init__(self, backend, *, model_revision: str | None = None):
        self.backend = backend
        self.model_path = getattr(backend, "model_path", None)
        self.model_revision = model_revision or getattr(backend, "version", "unknown")
        self.model_fingerprint = getattr(
            backend, "model_fingerprint", getattr(backend, "model_sha256", "unknown")
        )

    def score_fixed_region_query(
        self,
        image_view: Path,
        entity_id: str,
        region: RegionPrediction,
        query: RegionQuery,
        *,
        view_id: str,
    ) -> EvidenceScore:
        if not isinstance(region, RegionPrediction):
            raise TypeError("fixed-region scorer requires a persistent RegionPrediction")
        if not isinstance(query, RegionQuery):
            raise TypeError("fixed-region scorer requires a serialized RegionQuery")
        if view_id not in {"RAW", "FOCUS", "REMOVE"}:
            raise ValueError(f"unsupported counterfactual view id: {view_id!r}")

        region_id = region_identity(entity_id, region)
        logits = self.backend.logits_for_crop(image_view, region.bbox_xyxy_pixels, [query.text])
        if len(logits) != 1:
            raise RuntimeError(f"expected one fixed-query logit, got {len(logits)}")
        raw_logit = float(logits[0])
        normalized = _sigmoid(raw_logit)
        claim = {
            "entity_id": entity_id,
            "region_id": region_id,
            "bbox_xyxy_pixels": [float(value) for value in region.bbox_xyxy_pixels],
            "query_hash": query.query_hash,
            "query_template_id": query.template_id,
            "crop_policy_id": CROP_POLICY_ID,
        }
        diagnostics = {
            "view_id": view_id,
            "entity_id": entity_id,
            "region_id": region_id,
            "region_bbox_xyxy_pixels": [float(value) for value in region.bbox_xyxy_pixels],
            "query_text": query.text,
            "query_template_id": query.template_id,
            "query_hash": query.query_hash,
            "query_provenance": list(query.provenance),
            "query_components": dict(query.components),
            "crop_policy_id": CROP_POLICY_ID,
            "image_view": str(image_view),
            "raw_model_score_kind": "siglip_image_text_logit_for_fixed_query",
            "raw_logit": raw_logit,
            "margin_score": None,
            "margin_status": "NO_FROZEN_NEGATIVE_SET; raw fixed-query score only",
            "normalization_id": self.score_normalization,
            "normalization_semantics": "monotonic_sigmoid_transform_not_calibrated_probability",
            "score_semantics": self.score_semantics,
            "evidence_kind": self.evidence_kind,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "identity_layer": "SEMANTIC_REGION_SUPPORT",
            "full_logical_identity": False,
            "instance_uniqueness_claimed": False,
        }
        return EvidenceScore(
            processor=self.name,
            score_kind="fixed_region_query_logit",
            raw_score=raw_logit,
            normalized_score=normalized,
            margin=None,
            claimed_value=claim,
            support="SUPPORTED",
            normalizer_revision=self.score_normalization,
            diagnostics=diagnostics,
            view_id=view_id,
            normalization_semantics="monotonic_sigmoid_transform_not_calibrated_probability",
            calibrated=False,
            evidence_kind=self.evidence_kind,
        )

    # Name used by generic documentation; explicit alias keeps the contract
    # obvious at call sites.
    score_claim = score_fixed_region_query
