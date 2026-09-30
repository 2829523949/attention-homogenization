"""Capability registry enforcing specialist boundaries.

Capabilities separate *acquisition* from *score-based certification*.  A
processor that can propose a region or attribute need not expose a valid
comparable score; the certifier refuses such a claim rather than inventing a
value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from harness.acquisition.observations import Observation

from harness.certification.evidence import EvidenceScore
from harness.evaluators.leakage_checks import assert_d_drive_path
from harness.referents.referent_spec import GroundingQuery


@dataclass(frozen=True)
class ProcessorCapability:
    primitive: str
    legal_input_types: tuple[str, ...]
    output_schema: str
    score_semantics: str
    model_revision: str
    model_path: str | None
    deterministic: bool
    # Backwards-compatible name retained for older callers.  New callers
    # should use supports_certification_score.
    supports_evidence_scoring: bool = True
    notes: str = ""
    can_acquire: bool = True
    supported_predicates: tuple[str, ...] = ()
    supports_same_claim_rescoring: bool = False
    supports_certification_score: bool = False
    score_normalization: str = "none"
    comparable_across_counterfactual_views: bool = False
    requires_fixed_region: bool = False
    requires_fixed_query: bool = False
    supported_primitives: tuple[str, ...] = ()
    returns_multiple_candidates: bool = False
    supports_native_candidate_score: bool = False
    supports_fixed_region_certification_score: bool = False

    def __post_init__(self) -> None:
        if any(token in self.output_schema.casefold() for token in ("final_answer", "compositional_answer", "answer_field")):
            raise AssertionError("specialist capability cannot declare a final-answer output")
        if self.model_path and self.model_path.startswith("/"):
            assert_d_drive_path(Path(self.model_path), artifact_kind="model")
        if self.supports_certification_score:
            if not self.can_acquire and not self.supports_same_claim_rescoring:
                raise ValueError("score-only capabilities still need an explicit score contract")
            if not self.score_semantics.strip():
                raise ValueError("certification-capable processors must document score_semantics")
            if not self.score_normalization.strip() or self.score_normalization == "none":
                raise ValueError("certification-capable processors must declare score_normalization")
            if not self.supports_same_claim_rescoring:
                raise ValueError("certification-capable processors must support same-claim rescoring")
            if not self.comparable_across_counterfactual_views:
                raise ValueError("certification-capable processors must be comparable across counterfactual views")
        if self.supports_same_claim_rescoring and not self.supports_certification_score:
            raise ValueError("same-claim rescoring without a certification score is not usable for certification")
        if self.supports_certification_score and self.requires_fixed_region and not self.supports_same_claim_rescoring:
            raise ValueError("fixed-region certification scores require same-claim rescoring")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "legal_input_types": list(self.legal_input_types),
            "output_schema": self.output_schema,
            "score_semantics": self.score_semantics,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "deterministic": self.deterministic,
            "supports_evidence_scoring": self.supports_evidence_scoring,
            "notes": self.notes,
            "can_acquire": self.can_acquire,
            "supported_predicates": list(self.supported_predicates),
            "supports_same_claim_rescoring": self.supports_same_claim_rescoring,
            "supports_certification_score": self.supports_certification_score,
            "score_normalization": self.score_normalization,
            "comparable_across_counterfactual_views": self.comparable_across_counterfactual_views,
            "requires_fixed_region": self.requires_fixed_region,
            "requires_fixed_query": self.requires_fixed_query,
            "supported_primitives": list(self.supported_primitives),
            "returns_multiple_candidates": self.returns_multiple_candidates,
            "supports_native_candidate_score": self.supports_native_candidate_score,
            "supports_fixed_region_certification_score": self.supports_fixed_region_certification_score,
        }


@dataclass(frozen=True)
class RegisteredProcessor:
    name: str
    capability: ProcessorCapability
    implementation: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capability": self.capability.to_dict(),
            "implementation_class": self.implementation.__class__.__name__,
            "implementation_version": str(getattr(self.implementation, "version", "unknown")),
        }


class ProcessorRegistry:
    def __init__(self) -> None:
        self._processors: dict[str, RegisteredProcessor] = {}

    def register(self, name: str, capability: ProcessorCapability, implementation: Any) -> None:
        if name in self._processors:
            raise ValueError(f"duplicate processor registration: {name}")
        self._processors[name] = RegisteredProcessor(name, capability, implementation)

    def get(self, name: str) -> RegisteredProcessor:
        try:
            return self._processors[name]
        except KeyError as exc:
            raise KeyError(f"unregistered processor: {name}") from exc

    def require(self, name: str, primitive: str) -> RegisteredProcessor:
        item = self.get(name)
        if item.capability.primitive != primitive:
            raise TypeError(f"processor {name} provides {item.capability.primitive}, required {primitive}")
        return item

    def require_certification_score(self, name: str, primitive: str) -> RegisteredProcessor:
        item = self.require(name, primitive)
        if not item.capability.supports_certification_score:
            raise ValueError(f"processor {name} is not certification-score capable for {primitive}")
        return item

    def describe(self) -> dict[str, Any]:
        return {name: item.to_dict() for name, item in sorted(self._processors.items())}


class FlorenceQueryGroundingAdapter:
    """Legal semantic-query acquisition around the existing Florence checkpoint.

    The bundled Florence-2 open-vocabulary postprocessor returns only boxes and
    labels; it exposes no numeric referent-region confidence.  The adapter is
    therefore acquisition-only.  It must never turn detection recurrence or
    IoU into a synthetic model confidence.
    """

    name = "florence2_query_grounding"
    version = "florence2_open_vocabulary_query_acquisition_v2"
    can_acquire = True
    supports_certification_score = False
    supports_same_claim_rescoring = False
    comparable_across_counterfactual_views = False
    score_semantics = (
        "CERTIFICATION_SCORE_UNSUPPORTED: Florence-2 open-vocabulary postprocessing "
        "exposes bboxes/labels but no referent-region compatibility score."
    )
    score_normalization = "none"

    def __init__(self, processor: Any):
        self.processor = processor

    def ground(self, image_path: Path, query: GroundingQuery) -> list[Any]:
        if not isinstance(query, GroundingQuery):
            raise TypeError("grounding requires GroundingQuery, not a logical string")
        return list(self.processor.predict_query(image_path, query.semantic_text))

    def score(self, image_path: Path, query: GroundingQuery, region: Any, *, view_id: str = "RAW") -> EvidenceScore:
        if not isinstance(query, GroundingQuery):
            raise TypeError("grounding requires GroundingQuery")
        candidates = list(self.processor.predict_query(image_path, query.semantic_text))
        return EvidenceScore.unsupported(
            self.name,
            reason=(
                "CERTIFICATION_SCORE_UNSUPPORTED: Florence-2 postprocessing output has no "
                "referent-region confidence; box recurrence is retained only as a diagnostic."
            ),
            claimed_value=region,
            diagnostics={
                "candidate_count": len(candidates),
                "certification_capability": "CERTIFICATION_SCORE_UNSUPPORTED",
                "region_recurrence_diagnostic_only": True,
            },
            view_id=view_id,
        )


class ColorReaderAdapter:
    """Adapter exposing primitive color observations and fixed-candidate scores.

    ``observe`` returns an acquisition observation.  ``score`` evaluates the
    *fixed original candidate* on any supplied view.  It never substitutes
    winner agreement for a fixed-candidate score.
    """

    def __init__(self, reader: Any, *, name: str | None = None):
        self.reader = reader
        self.name = name or getattr(reader, "name", reader.__class__.__name__)
        self.version = getattr(reader, "version", "unknown")
        self.can_acquire = bool(getattr(reader, "can_acquire", True))
        self.supports_certification_score = bool(getattr(reader, "supports_certification_score", False))
        self.supports_same_claim_rescoring = bool(getattr(reader, "supports_same_claim_rescoring", False))
        self.comparable_across_counterfactual_views = bool(getattr(reader, "comparable_across_counterfactual_views", False))
        self.score_semantics = str(getattr(reader, "score_semantics", "CERTIFICATION_SCORE_UNSUPPORTED: no fixed-candidate score"))
        self.score_normalization = str(getattr(reader, "score_normalization", "none"))

    def observe(self, image_path: Path, entity_id: str, region: Any, *, view_id: str = "RAW") -> "Observation":
        from harness.acquisition.observations import Observation

        result = self.reader.read(image_path, region)
        diagnostics = dict(result.diagnostics)
        diagnostics["acquisition_only"] = not self.supports_certification_score
        if self.supports_certification_score and result.color is not None:
            score = self.score(image_path, entity_id, region, result.color, view_id=view_id)
        else:
            score = EvidenceScore.unsupported(
                self.name,
                reason=f"CERTIFICATION_SCORE_UNSUPPORTED: {self.score_semantics}",
                claimed_value=result.color,
                diagnostics=diagnostics,
                view_id=view_id,
            )
        return Observation(entity_id, "COLOR", result.color, score, self.name, ("TOOL_OBSERVATION",), view_id, diagnostics)

    def score(self, image_path: Path, entity_id: str, region: Any, claimed_value: Any, *, view_id: str) -> EvidenceScore:
        if not self.supports_certification_score:
            return EvidenceScore.unsupported(
                self.name,
                reason=f"CERTIFICATION_SCORE_UNSUPPORTED: {self.score_semantics}",
                claimed_value=claimed_value,
                diagnostics={"certification_capability": "CERTIFICATION_SCORE_UNSUPPORTED"},
                view_id=view_id,
            )
        scorer = getattr(self.reader, "score_claim", None)
        if scorer is None:
            return EvidenceScore.unsupported(
                self.name,
                reason="CERTIFICATION_SCORE_UNSUPPORTED: reader does not implement score_claim",
                claimed_value=claimed_value,
                view_id=view_id,
            )
        score = scorer(image_path, region, claimed_value, view_id=view_id)
        if not isinstance(score, EvidenceScore):
            raise TypeError("score_claim must return EvidenceScore")
        if score.processor != self.name:
            raise ValueError(f"score_claim processor mismatch: {score.processor!r} vs {self.name!r}")
        return score
