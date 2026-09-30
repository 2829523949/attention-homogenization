"""Backend-neutral candidate acquisition contracts.

Candidate generation ends at region proposals.  These objects deliberately do
not contain evaluator IoU, correctness, selected-answer, or certification
fields.  Evaluators must join ground truth in a separate namespace after
inference has completed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from harness.evaluators.leakage_checks import assert_inference_safe
from harness.processors.base import RegionPrediction
from harness.referents.referent_spec import GroundingQuery, ReferentSpec


GROUNDING_QUERY_SERIALIZER_VERSION = "legal_semantic_referent_v1"


@dataclass(frozen=True)
class SerializedGroundingQuery:
    logical_ref: str
    text: str
    query_hash: str
    provenance: tuple[str, ...]
    serializer_version: str
    source_grounding_query: GroundingQuery

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "logical_ref": self.logical_ref,
            "text": self.text,
            "query_hash": self.query_hash,
            "provenance": list(self.provenance),
            "serializer_version": self.serializer_version,
            "source_grounding_query": self.source_grounding_query.to_dict(),
        }
        assert_inference_safe(payload, context="SerializedGroundingQuery")
        return payload


class GroundingDinoQuerySerializer:
    """Serialize only the legal semantic text already present in a ReferentSpec."""

    version = GROUNDING_QUERY_SERIALIZER_VERSION

    def serialize(self, spec: ReferentSpec) -> SerializedGroundingQuery:
        if not isinstance(spec, ReferentSpec):
            raise TypeError("query serializer requires ReferentSpec")
        spec.assert_legal()
        query = GroundingQuery.from_spec(spec)
        # No prompt enrichment is allowed here.  In particular, constraints
        # are retained for provenance but are not appended to the detector text.
        text = query.semantic_text.strip()
        provenance = tuple(
            sorted({item.value for values in query.token_provenance.values() for item in values})
        )
        digest_payload = json.dumps(
            {"serializer_version": self.version, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return SerializedGroundingQuery(
            logical_ref=spec.logical_ref,
            text=text,
            query_hash=hashlib.sha256(digest_payload).hexdigest(),
            provenance=provenance,
            serializer_version=self.version,
            source_grounding_query=query,
        )


@dataclass(frozen=True)
class CandidateRegion:
    candidate_id: str
    bbox_xyxy_pixels: tuple[float, float, float, float]
    bbox_xyxy_normalized: tuple[float, float, float, float]
    native_grounding_score: float | None
    native_label_text: str | None
    source_backend: str
    query_hash: str
    provenance: tuple[str, ...]
    raw_candidate_index: int
    postprocessing_metadata: dict[str, Any] = field(default_factory=dict)
    score_semantics: str | None = None

    def __post_init__(self) -> None:
        if self.native_grounding_score is not None:
            score = float(self.native_grounding_score)
            if not (0.0 <= score <= 1.0):
                raise ValueError("model-native Grounding DINO score must be in [0, 1]")
        if self.raw_candidate_index < 0:
            raise ValueError("raw_candidate_index must be non-negative")
        assert_inference_safe(self.to_dict(), context="CandidateRegion")

    @property
    def score(self) -> float | None:
        """Compatibility alias for existing deterministic acquisition ranking."""
        return self.native_grounding_score

    @property
    def label(self) -> str:
        return self.native_label_text or ""

    @property
    def source(self) -> str:
        return self.source_backend

    @property
    def mask_path(self) -> None:
        return None

    def as_region_prediction(self) -> RegionPrediction:
        return RegionPrediction(
            self.bbox_xyxy_pixels,
            self.bbox_xyxy_normalized,
            self.native_grounding_score,
            self.native_label_text or "",
            self.source_backend,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xyxy_pixels"] = list(self.bbox_xyxy_pixels)
        payload["bbox_xyxy_normalized"] = list(self.bbox_xyxy_normalized)
        payload["provenance"] = list(self.provenance)
        assert_inference_safe(payload, context="CandidateRegion")
        return payload


@dataclass(frozen=True)
class CandidateSet:
    request_id: str
    logical_ref: str
    original_referent_spec: dict[str, Any]
    legal_query: str
    query_hash: str
    query_provenance: tuple[str, ...]
    serializer_version: str
    acquisition_backend: str
    model_revision: str
    candidates: tuple[CandidateRegion, ...]
    acquisition_trace: dict[str, Any]

    def __post_init__(self) -> None:
        if any(candidate.query_hash != self.query_hash for candidate in self.candidates):
            raise ValueError("candidate query hash differs from CandidateSet query hash")
        assert_inference_safe(self.to_dict(), context="CandidateSet")

    def regions(self) -> tuple[RegionPrediction, ...]:
        return tuple(candidate.as_region_prediction() for candidate in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "logical_ref": self.logical_ref,
            "original_referent_spec": self.original_referent_spec,
            "legal_query": self.legal_query,
            "query_hash": self.query_hash,
            "query_provenance": list(self.query_provenance),
            "serializer_version": self.serializer_version,
            "acquisition_backend": self.acquisition_backend,
            "model_revision": self.model_revision,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "acquisition_trace": self.acquisition_trace,
        }
        assert_inference_safe(payload, context="CandidateSet")
        return payload


def candidate_request_id(image_path: Path, serialized: SerializedGroundingQuery, backend: str) -> str:
    payload = json.dumps(
        {
            "image_path": str(Path(image_path).resolve()),
            "logical_ref": serialized.logical_ref,
            "query_hash": serialized.query_hash,
            "backend": backend,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "ACQ_" + hashlib.sha256(payload).hexdigest()[:24]


@runtime_checkable
class CandidateGenerator(Protocol):
    name: str
    version: str

    def acquire(self, image_path: Path, referent: ReferentSpec) -> CandidateSet:
        ...


class FlorenceCandidateGenerator:
    """CandidateSet adapter around the unchanged Florence baseline grounder."""

    name = "florence2_query_grounding"
    can_acquire = True
    supports_certification_score = False
    supports_same_claim_rescoring = False
    comparable_across_counterfactual_views = False
    score_normalization = "none"

    def __init__(self, grounder: Any, serializer: GroundingDinoQuerySerializer | None = None):
        self.grounder = grounder
        self.serializer = serializer or GroundingDinoQuerySerializer()
        self.version = str(getattr(grounder, "version", "unknown"))
        self.score_semantics = str(getattr(grounder, "score_semantics", "native score unavailable"))
        self.last_candidate_trace: dict[str, Any] | None = None

    def acquire(self, image_path: Path, referent: ReferentSpec) -> CandidateSet:
        serialized = self.serializer.serialize(referent)
        regions = list(self.grounder.ground(image_path, serialized.source_grounding_query))
        backend_trace = getattr(self.grounder, "last_candidate_trace", None) or {}
        candidates = tuple(
            CandidateRegion(
                candidate_id=f"{serialized.logical_ref}:florence2:{index}",
                bbox_xyxy_pixels=tuple(region.bbox_xyxy_pixels),
                bbox_xyxy_normalized=tuple(region.bbox_xyxy_normalized),
                native_grounding_score=region.score,
                native_label_text=region.label,
                source_backend=region.source,
                query_hash=serialized.query_hash,
                provenance=serialized.provenance + ("TOOL_OBSERVATION",),
                raw_candidate_index=index,
                postprocessing_metadata={"adapter": "unchanged_florence_output", "accepted_index": index},
                score_semantics=None if region.score is None else "model_native_candidate_score",
            )
            for index, region in enumerate(regions)
        )
        trace = {
            **backend_trace,
            "backend": "florence2",
            "final_candidate_set_count": len(candidates),
            "serializer_version": serialized.serializer_version,
            "query_hash": serialized.query_hash,
        }
        self.last_candidate_trace = trace
        return CandidateSet(
            request_id=candidate_request_id(image_path, serialized, "florence2"),
            logical_ref=referent.logical_ref,
            original_referent_spec=referent.to_dict(),
            legal_query=serialized.text,
            query_hash=serialized.query_hash,
            query_provenance=serialized.provenance,
            serializer_version=serialized.serializer_version,
            acquisition_backend="florence2",
            model_revision=self.version,
            candidates=candidates,
            acquisition_trace=trace,
        )

    def ground(self, image_path: Path, query: GroundingQuery) -> list[RegionPrediction]:
        """Compatibility route retained for old callers; Florence behavior is unchanged."""
        return list(self.grounder.ground(image_path, query))

