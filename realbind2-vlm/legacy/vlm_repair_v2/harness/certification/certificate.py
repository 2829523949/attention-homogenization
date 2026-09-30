"""Binding certificates produced only after counterfactual evidence checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .evidence import EvidenceScore


class CertificateStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"
    UNCALIBRATED = "UNCALIBRATED"


@dataclass(frozen=True)
class BindingCertificate:
    certificate_id: str
    entity_id: str
    predicate: str
    proposed_value: Any
    binding_kind: str
    raw_score: EvidenceScore
    focus_score: EvidenceScore
    remove_score: EvidenceScore
    focus_delta: float | None
    necessity_delta: float | None
    status: CertificateStatus
    calibrated: bool
    gate_config: dict[str, float]
    provenance: tuple[str, ...] = ()
    rationale: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    dag_node_id: str | None = None
    dependency_ids: tuple[str, ...] = ()
    # Distinguishes a full logical entity binding from a narrower semantic
    # region-content evidence record.
    evidence_kind: str = "ENTITY_BINDING_EVIDENCE"
    identity_layer: str = "FULL_LOGICAL_IDENTITY"
    calibration_status: str = "UNCALIBRATED_DIAGNOSTIC"

    def to_dict(self) -> dict[str, Any]:
        proposed = self.proposed_value
        if hasattr(proposed, "bbox_xyxy_pixels"):
            proposed = {
                "bbox_xyxy_pixels": list(proposed.bbox_xyxy_pixels),
                "bbox_xyxy_normalized": list(proposed.bbox_xyxy_normalized),
                "score": None if proposed.score is None else float(proposed.score),
                "score_available": bool(proposed.score_available),
                "source": str(proposed.source),
            }
        return {
            "certificate_id": self.certificate_id,
            "entity_id": self.entity_id,
            "predicate": self.predicate,
            "proposed_value": proposed,
            "binding_kind": self.binding_kind,
            "raw_score": self.raw_score.to_dict(),
            "focus_score": self.focus_score.to_dict(),
            "remove_score": self.remove_score.to_dict(),
            "focus_delta": self.focus_delta,
            "necessity_delta": self.necessity_delta,
            "status": self.status.value,
            "calibrated": self.calibrated,
            "gate_config": dict(self.gate_config),
            "provenance": list(self.provenance),
            "rationale": self.rationale,
            "diagnostics": self.diagnostics,
            "dag_node_id": self.dag_node_id,
            "dependency_ids": list(self.dependency_ids),
            "evidence_kind": self.evidence_kind,
            "identity_layer": self.identity_layer,
            "calibration_status": self.calibration_status,
        }
