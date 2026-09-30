from __future__ import annotations

import json
from typing import Any

from harness.processors.base import VisualState


def compile_facts(state: VisualState, reference_names: dict[str, str]) -> dict[str, Any]:
    entities = []
    for entity in sorted(state.entities, key=lambda item: item.reference_id):
        entities.append({"reference_id": entity.reference_id, "identity": reference_names.get(entity.reference_id, entity.category), "color": entity.color, "position_x": entity.center_x, "position_y": entity.center_y, "confidence": {"region": entity.region_confidence, "color": entity.color_confidence}})
    relations = [{"source": row.source_entity, "target": row.target_entity, "relation": row.relation, "confidence": row.confidence} for row in sorted(state.relations, key=lambda item: (item.source_entity, item.target_entity))]
    return {"schema_version": "realbind_atomic_visual_facts_v1", "case_id": state.case_id, "entities": entities, "relations": relations, "contains_final_answer": False}


def serialize_facts(facts: dict[str, Any]) -> str:
    lines = ["Observed atomic visual facts:"]
    for entity in facts["entities"]:
        position = "unknown" if entity["position_x"] is None else f"x={entity['position_x']:.6f}"
        lines.append(f"Entity {entity['reference_id']} ({entity['identity']}): color={entity['color'] or 'unknown'}; position={position}.")
    for relation in facts["relations"]:
        lines.append(f"Relation: {relation['source']} is {relation['relation'] or 'unknown'} of {relation['target']}.")
    lines.append("Use these atomic facts to answer the unchanged question. The facts do not state the final answer.")
    return "\n".join(lines)


def prompt_with_facts(frozen_question: str, facts: dict[str, Any], handoff: str | None = None) -> str:
    prefix = serialize_facts(facts)
    if handoff is not None: prefix += f"\nSelected target reference: Entity {handoff}."
    # Preserve the frozen question byte-for-byte as the final block.
    return prefix + "\n\nFrozen question:\n" + frozen_question
