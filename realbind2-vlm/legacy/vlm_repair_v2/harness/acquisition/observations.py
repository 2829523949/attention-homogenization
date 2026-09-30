"""Primitive observations and auditable acquisition traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.certification.evidence import EvidenceScore
from harness.evaluators.leakage_checks import assert_inference_safe, assert_no_final_answer
from harness.referents.referent_spec import Provenance


@dataclass(frozen=True)
class Observation:
    entity_id: str
    predicate: str
    proposed_value: Any
    score: EvidenceScore
    processor: str
    provenance: tuple[Provenance | str, ...]
    view_id: str = "RAW"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.predicate in {"COMPOSE", "OUTPUT", "FINAL_ANSWER"}:
            raise AssertionError("specialist processors may return primitive observations only")
        if not self.entity_id.startswith("E_"):
            raise ValueError(f"observation must be entity-indexed: {self.entity_id}")
        if not self.provenance:
            raise ValueError("observation provenance is required")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "entity_id": self.entity_id,
            "predicate": self.predicate,
            "proposed_value": self.proposed_value,
            "score": self.score.to_dict(),
            "processor": self.processor,
            "provenance": [item.value if isinstance(item, Provenance) else str(item) for item in self.provenance],
            "view_id": self.view_id,
            "diagnostics": self.diagnostics,
        }
        assert_inference_safe(payload, context="Observation")
        assert_no_final_answer(payload, context="Observation")
        return payload


@dataclass(frozen=True)
class AcquisitionTrace:
    step: int
    operation: str
    node_id: str
    entity_id: str | None
    processor: str | None
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step": self.step,
            "operation": self.operation,
            "node_id": self.node_id,
            "entity_id": self.entity_id,
            "processor": self.processor,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "provenance": list(self.provenance),
        }
        assert_inference_safe(payload, context="AcquisitionTrace")
        return payload

