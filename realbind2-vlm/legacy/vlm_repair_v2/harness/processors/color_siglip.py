"""SigLIP-2 fixed-vocabulary color reader and fixed-candidate scorer.

The reader returns the arg-max color and its real SigLIP probability for
acquisition.  ``score_claim`` evaluates a *fixed candidate value* on any
supplied counterfactual view; it does not first take an independent arg-max on
that view and reduce the score to winner agreement.

Score semantics used for certification:

    raw_score       = real SigLIP image-text logit for ``a {candidate} object``
    margin          = candidate_logit - max(logit of the other ten colors)
    normalized_score = sigmoid(margin)

The normalization is a documented monotone function of a real model-derived
contrastive margin (``siglip2_color_margin_sigmoid_v1``).  It is not a
calibrated probability and the frozen smoke remains ``UNCALIBRATED_DIAGNOSTIC``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .base import ColorPrediction, RegionPrediction
from .color_pixel import REALBIND_COLORS


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    top = max(float(value) for value in values)
    exps = [math.exp(float(value) - top) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


class SiglipColorReader:
    name = "siglip2_color"
    version = "google_siglip2_base_patch16_224_realbind_vocab_v2"
    can_acquire = True
    supports_same_claim_rescoring = True
    supports_certification_score = True
    comparable_across_counterfactual_views = True
    score_semantics = (
        "fixed-candidate SigLIP-2 color contrastive logit margin; "
        "raw=candidate logit, margin=candidate logit minus best alternative color logit"
    )
    score_normalization = "siglip2_color_margin_sigmoid_v1"

    def __init__(self, model_path: Path | None = None, vocabulary: Sequence[str] = REALBIND_COLORS, device: str | None = None, *, backend=None):
        if tuple(vocabulary) != REALBIND_COLORS:
            raise ValueError("SigLIP must use frozen RealBind vocabulary")
        if backend is None:
            from .siglip_backend import Siglip2Backend

            if model_path is None:
                raise ValueError("model_path is required when no shared Siglip2Backend is supplied")
            backend = Siglip2Backend(model_path, device=device)
        self.backend = backend
        self.model_path = getattr(backend, "model_path", None)
        self.model_revision = getattr(backend, "version", "unknown")
        self.model_fingerprint = getattr(backend, "model_fingerprint", getattr(backend, "model_sha256", "unknown"))
        self.vocabulary = tuple(vocabulary)
        self.templates = tuple(f"a {color} object" for color in self.vocabulary)

    def _logits(self, image_path: Path, region: RegionPrediction) -> list[float]:
        return self.backend.logits_for_crop(image_path, region.bbox_xyxy_pixels, self.templates)

    def read(self, image_path: Path, region: RegionPrediction) -> ColorPrediction:
        logits = self._logits(image_path, region)
        probabilities = _softmax(logits)
        index = max(range(len(logits)), key=lambda item: (logits[item], -item))
        diagnostics = {
            "templates": list(self.templates),
            "logits": [float(value) for value in logits],
            "probabilities": [float(value) for value in probabilities],
            "score_semantics": self.score_semantics,
            "score_normalization": self.score_normalization,
            "certification_capable": True,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
        }
        return ColorPrediction(self.vocabulary[index], float(probabilities[index]), self.name, diagnostics)

    def score_claim(self, image_path: Path, region: RegionPrediction, claimed_value: str, *, view_id: str) -> "EvidenceScore":
        from harness.certification.evidence import EvidenceScore

        claimed = str(claimed_value)
        if claimed not in self.vocabulary:
            return EvidenceScore.unsupported(
                self.name,
                reason=f"candidate color {claimed!r} is not in the frozen RealBind vocabulary",
                claimed_value=claimed,
                diagnostics={"view_id": view_id},
                view_id=view_id,
            )
        logits = self._logits(image_path, region)
        index = self.vocabulary.index(claimed)
        candidate_logit = float(logits[index])
        alternative_indices = [position for position in range(len(logits)) if position != index]
        best_alternative_index = max(alternative_indices, key=lambda position: float(logits[position]))
        best_alternative = float(logits[best_alternative_index])
        margin = candidate_logit - best_alternative
        normalized = _sigmoid(margin)
        probabilities = _softmax(logits)
        diagnostics = {
            "view_id": view_id,
            "template": self.templates[index],
            "templates": list(self.templates),
            "candidate": claimed,
            "candidate_logit": candidate_logit,
            "best_alternative_color": self.vocabulary[best_alternative_index],
            "best_alternative_logit": best_alternative,
            "margin": margin,
            "candidate_softmax_probability": float(probabilities[index]),
            "logits": [float(value) for value in logits],
            "probabilities": [float(value) for value in probabilities],
            "region_bbox_xyxy_pixels": list(region.bbox_xyxy_pixels),
            "region_source": region.source,
            "score_definition": "sigmoid(candidate_logit - best_alternative_logit)",
            "score_semantics": self.score_semantics,
            "score_normalization": self.score_normalization,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
        }
        return EvidenceScore(
            self.name,
            "fixed_candidate_logit_margin",
            candidate_logit,
            normalized,
            margin,
            claimed,
            "SUPPORTED",
            self.score_normalization,
            diagnostics,
            view_id,
        )
