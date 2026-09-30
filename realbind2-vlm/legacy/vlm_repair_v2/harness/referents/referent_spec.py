"""Legality-preserving representations for logical references and queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness.evaluators.leakage_checks import assert_inference_safe


class Provenance(str, Enum):
    QUESTION_TEXT = "QUESTION_TEXT"
    PREVIOUS_CERTIFIED_FACT = "PREVIOUS_CERTIFIED_FACT"
    PREVIOUS_ENTITY_STATE = "PREVIOUS_ENTITY_STATE"
    TOOL_OBSERVATION = "TOOL_OBSERVATION"
    PROTOCOL_INTERFACE = "PROTOCOL_INTERFACE"
    EVALUATOR_ONLY = "EVALUATOR_ONLY"


@dataclass(frozen=True)
class Provenanced:
    value: Any
    provenance: Provenance
    source_detail: str

    def __post_init__(self) -> None:
        if not self.source_detail.strip():
            raise ValueError("every provenance-bearing field needs source_detail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "source_detail": self.source_detail,
        }


def _field_values(spec: "ReferentSpec") -> list[Provenanced]:
    values: list[Provenanced] = []
    for item in (
        spec.text_description,
        spec.category_constraint,
        spec.existing_entity_id,
        spec.existing_region,
    ):
        if item is not None:
            values.append(item)
    values.extend(spec.attribute_constraints)
    values.extend(spec.relation_constraints)
    return values


@dataclass(frozen=True)
class ReferentSpec:
    logical_ref: str
    text_description: Provenanced | None = None
    category_constraint: Provenanced | None = None
    attribute_constraints: tuple[Provenanced, ...] = ()
    relation_constraints: tuple[Provenanced, ...] = ()
    existing_entity_id: Provenanced | None = None
    existing_region: Provenanced | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.logical_ref):
            raise ValueError(f"invalid logical reference: {self.logical_ref!r}")
        for item in _field_values(self):
            if item.provenance is Provenance.EVALUATOR_ONLY:
                raise AssertionError(f"evaluator-only provenance cannot construct automatic ReferentSpec: {self.logical_ref}")
        if self.category_constraint is None and self.text_description is None and self.existing_entity_id is None:
            raise ValueError(f"ReferentSpec {self.logical_ref} has no legal identifying input")

    def assert_legal(self) -> None:
        for item in _field_values(self):
            if item.provenance is Provenance.EVALUATOR_ONLY:
                raise AssertionError(f"illegal evaluator-only ReferentSpec field: {self.logical_ref}")
        assert_inference_safe(self.to_dict(), context="ReferentSpec")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "logical_ref": self.logical_ref,
            "text_description": self.text_description.to_dict() if self.text_description else None,
            "category_constraint": self.category_constraint.to_dict() if self.category_constraint else None,
            "attribute_constraints": [item.to_dict() for item in self.attribute_constraints],
            "relation_constraints": [item.to_dict() for item in self.relation_constraints],
            "existing_entity_id": self.existing_entity_id.to_dict() if self.existing_entity_id else None,
            "existing_region": self.existing_region.to_dict() if self.existing_region else None,
        }
        assert_inference_safe(payload, context="ReferentSpec")
        return payload


@dataclass(frozen=True)
class GroundingQuery:
    logical_ref: str
    semantic_text: str
    constraints: tuple[Provenanced, ...]
    tokens: tuple[str, ...]
    token_provenance: dict[str, tuple[Provenance, ...]]
    source_spec: str

    def __post_init__(self) -> None:
        if not self.semantic_text.strip():
            raise ValueError("grounding query must contain semantic text")
        if self.semantic_text.strip().casefold() in {"a", "b"} and not self.constraints:
            raise AssertionError("logical symbol cannot be used as a detector query")
        if any(item.provenance is Provenance.EVALUATOR_ONLY for item in self.constraints):
            raise AssertionError("evaluator-only constraint cannot enter GroundingQuery")
        if not self.tokens:
            raise ValueError("grounding query must expose token provenance")
        missing = [token for token in self.tokens if token not in self.token_provenance]
        if missing:
            raise ValueError(f"missing token provenance: {missing}")

    @classmethod
    def from_spec(cls, spec: ReferentSpec) -> "GroundingQuery":
        spec.assert_legal()
        source = spec.category_constraint or spec.text_description
        if source is None:
            raise ValueError(f"{spec.logical_ref} has no semantic query source")
        semantic_text = str(source.value).strip()
        if semantic_text.casefold() in {"a", "b"}:
            raise AssertionError("logical symbol is not a legal semantic query")
        tokens = tuple(dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", semantic_text)))
        if not tokens:
            raise ValueError(f"no query tokens in {semantic_text!r}")
        constraints = tuple(item for item in _field_values(spec) if item is not source)
        token_provenance = {token: (source.provenance,) for token in tokens}
        return cls(spec.logical_ref, semantic_text, constraints, tokens, token_provenance, f"ReferentSpec:{spec.logical_ref}")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "logical_ref": self.logical_ref,
            "semantic_text": self.semantic_text,
            "constraints": [item.to_dict() for item in self.constraints],
            "tokens": list(self.tokens),
            "token_provenance": {key: [item.value for item in value] for key, value in sorted(self.token_provenance.items())},
            "source_spec": self.source_spec,
        }
        assert_inference_safe(payload, context="GroundingQuery")
        return payload
