"""Deterministic legal query serialization for fixed-region evidence scoring.

The query is built only from a ``GroundingQuery`` that was itself constructed
from legal visible nouns/constraints.  Evaluator-only provenance is rejected
by ``GroundingQuery`` before this module is reached.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .referent_spec import GroundingQuery, Provenance


TEMPLATE_ID = "siglip2_region_query_photo_of_legal_referent_v1"

_LEGAL_ATTRIBUTE_PROVENANCE = {
    Provenance.QUESTION_TEXT,
    Provenance.PREVIOUS_ENTITY_STATE,
    Provenance.PREVIOUS_CERTIFIED_FACT,
}

_RELATION_WORDS = {"left", "right"}


@dataclass(frozen=True)
class RegionQuery:
    text: str
    template_id: str
    query_hash: str
    provenance: tuple[str, ...]
    components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.text,
            "query_template_id": self.template_id,
            "query_hash": self.query_hash,
            "query_provenance": list(self.provenance),
            "query_components": dict(self.components),
        }


def build_region_query(query: GroundingQuery) -> RegionQuery:
    """Serialize one fixed legal query for all counterfactual views."""

    if not isinstance(query, GroundingQuery):
        raise TypeError("region query construction requires a GroundingQuery")
    category = str(query.semantic_text).strip()
    if not category:
        raise ValueError("legal referent query has no semantic category text")

    attributes: list[tuple[str, str, str]] = []
    for constraint in query.constraints:
        if constraint.provenance is Provenance.EVALUATOR_ONLY:
            raise AssertionError("evaluator-only constraint cannot enter region query construction")
        if constraint.provenance not in _LEGAL_ATTRIBUTE_PROVENANCE:
            continue
        value = str(constraint.value).strip()
        if not value or value.casefold() == category.casefold():
            continue
        if value.casefold() in _RELATION_WORDS or value.casefold() in {"a", "b", "target"}:
            continue
        attributes.append((value, constraint.provenance.value, constraint.source_detail))
    deduped: dict[str, tuple[str, str, str]] = {}
    for value, provenance, detail in attributes:
        deduped.setdefault(value.casefold(), (value, provenance, detail))
    ordered_attributes = [deduped[key] for key in sorted(deduped)]

    noun_phrase = " ".join([item[0] for item in ordered_attributes] + [category]).strip()
    text = f"a photo of a {noun_phrase}"
    query_hash = hashlib.sha256(f"{TEMPLATE_ID}|{text}".encode("utf-8")).hexdigest()[:32]

    provenance: set[str] = set()
    for provenances in query.token_provenance.values():
        provenance.update(item.value for item in provenances)
    provenance.update(item[1] for item in ordered_attributes)
    components = {
        "category": category,
        "attributes": [
            {"value": value, "provenance": provenance_value, "source_detail": detail}
            for value, provenance_value, detail in ordered_attributes
        ],
        "category_provenance": sorted(
            {item.value for provenances in query.token_provenance.values() for item in provenances}
        ),
    }
    return RegionQuery(text, TEMPLATE_ID, query_hash, tuple(sorted(provenance)), components)
