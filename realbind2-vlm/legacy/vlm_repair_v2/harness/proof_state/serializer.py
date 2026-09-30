"""Deterministic, human-readable, image-free proof-state serialization."""

from __future__ import annotations

from typing import Any

from .compiler import CertifiedFact, DerivedFact, MinimalProofState


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def serialize_proof_state(state: MinimalProofState) -> str:
    lines = [
        "MINIMAL CERTIFIED PROOF STATE (IMAGE-FREE HANDOFF)",
        f"proof_complete={state.proof_complete} certified_proof_complete={state.certified_proof_complete} diagnostic={state.diagnostic}",
    ]
    for entity in sorted(state.entities, key=lambda item: item.entity_id):
        refs = ",".join(entity.logical_refs)
        nodes = ",".join(entity.dag_node_ids)
        lines.append(
            f"ENTITY {entity.entity_id} logical_refs=[{refs}] region_binding={entity.region_status} "
            f"certificate={entity.region_certificate_id or 'none'} dag_nodes=[{nodes}]"
        )
    for fact in sorted(state.facts, key=lambda item: item.fact_id):
        dependencies = ",".join(getattr(fact, "dependencies", ()) or getattr(fact, "dependency_ids", ()))
        if isinstance(fact, CertifiedFact):
            lines.append(
                f"CERTIFIED_FACT {fact.predicate}({fact.entity_id})={_format_value(fact.value)} "
                f"status={fact.certificate_status} certificate={fact.certificate_id} "
                f"dag_node={fact.dag_node_id or 'none'} dependencies=[{dependencies}]"
            )
        elif isinstance(fact, DerivedFact):
            entities = ",".join(fact.source_entity_ids)
            lines.append(
                f"DERIVED_FACT {fact.predicate}({entities})={_format_value(fact.value)} "
                f"rule={fact.deterministic_rule} dag_node={fact.dag_node_id or 'none'} dependencies=[{dependencies}]"
            )
    lines.append("Only dependency-required entity facts are present; no raw image, proposal dump, evaluator annotation, or language output is included.")
    return "\n".join(lines)
