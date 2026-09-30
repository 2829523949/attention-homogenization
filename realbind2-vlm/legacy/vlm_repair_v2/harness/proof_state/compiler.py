"""Dependency-aware minimal certified state compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from harness.certification.certificate import BindingCertificate, CertificateStatus
from harness.entities.entity_manager import EntityManager
from harness.evaluators.leakage_checks import assert_inference_safe, assert_no_final_answer, assert_no_image_payload
from harness.planner.predicate_dag import PredicateDAG

from .obligations import ObligationReport, ProofObligation, evaluate_obligations


def _compact_value(value: Any) -> Any:
    if hasattr(value, "bbox_xyxy_pixels"):
        return {
            "bbox_xyxy_pixels": [float(item) for item in value.bbox_xyxy_pixels],
            "bbox_xyxy_normalized": [float(item) for item in value.bbox_xyxy_normalized],
            "score": None if value.score is None else float(value.score),
            "source": str(value.source),
        }
    if isinstance(value, dict):
        return {str(key): _compact_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_compact_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ProofEntity:
    entity_id: str
    logical_refs: tuple[str, ...]
    region: Any
    region_certificate_id: str | None
    region_status: str
    provenance: tuple[str, ...]
    dag_node_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "logical_refs": list(self.logical_refs),
            "region": _compact_value(self.region),
            "region_certificate_id": self.region_certificate_id,
            "region_status": self.region_status,
            "provenance": list(self.provenance),
            "dag_node_ids": list(self.dag_node_ids),
        }


@dataclass(frozen=True)
class CertifiedFact:
    fact_id: str
    entity_id: str
    predicate: str
    value: Any
    certificate_id: str
    certificate_status: str
    provenance: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    dag_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_kind": "CERTIFIED_PRIMITIVE",
            "entity_id": self.entity_id,
            "predicate": self.predicate,
            "value": _compact_value(self.value),
            "certificate_id": self.certificate_id,
            "certificate_status": self.certificate_status,
            "provenance": list(self.provenance),
            "dependencies": list(self.dependencies),
            "dag_node_id": self.dag_node_id,
        }


@dataclass(frozen=True)
class DerivedFact:
    fact_id: str
    predicate: str
    value: Any
    source_entity_ids: tuple[str, ...]
    source_certificate_ids: tuple[str, ...]
    deterministic_rule: str
    provenance: tuple[str, ...]
    dag_node_id: str | None = None
    dependency_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_kind": "DETERMINISTIC_DERIVED",
            "predicate": self.predicate,
            "value": _compact_value(self.value),
            "source_entity_ids": list(self.source_entity_ids),
            "source_certificate_ids": list(self.source_certificate_ids),
            "deterministic_rule": self.deterministic_rule,
            "provenance": list(self.provenance),
            "dag_node_id": self.dag_node_id,
            "dependencies": list(self.dependency_ids),
            "derivation_status": "SATISFIED_DEPENDENCIES",
        }


@dataclass(frozen=True)
class ProofState:
    schema_version: str
    task_type: str
    dag_id: str
    entities: tuple[ProofEntity, ...]
    facts: tuple[CertifiedFact | DerivedFact, ...]
    dependencies: dict[str, tuple[str, ...]]
    compilation_notes: tuple[str, ...] = ()
    obligations: tuple[ProofObligation, ...] = ()
    required_node_ids: tuple[str, ...] = ()
    proof_complete: bool = False
    certified_proof_complete: bool = False
    diagnostic: bool = False
    dependency_closure_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "dag_id": self.dag_id,
            "entities": [entity.to_dict() for entity in self.entities],
            "facts": [fact.to_dict() for fact in self.facts],
            "dependencies": {key: list(value) for key, value in sorted(self.dependencies.items())},
            "compilation_notes": list(self.compilation_notes),
            "obligations": [item.to_dict() for item in self.obligations],
            "required_node_ids": list(self.required_node_ids),
            "proof_complete": self.proof_complete,
            "certified_proof_complete": self.certified_proof_complete,
            "diagnostic": self.diagnostic,
            "dependency_closure_valid": self.dependency_closure_valid,
        }
        assert_inference_safe(payload, context="MinimalProofState")
        assert_no_final_answer(payload, context="MinimalProofState")
        assert_no_image_payload(payload, context="MinimalProofState")
        return payload


MinimalProofState = ProofState


def compile_minimal_proof_state(
    dag: PredicateDAG,
    entity_manager: EntityManager,
    certificates: Iterable[BindingCertificate],
    derived_facts: Iterable[DerivedFact],
    *,
    include_uncalibrated: bool = False,
) -> ProofState:
    """Compile only required DAG nodes with fully satisfied dependencies.

    ``include_uncalibrated=True`` builds a clearly labelled diagnostic state
    for smoke inspection.  It is not eligible for the main hard handoff.
    """

    certificates = tuple(certificates)
    derived_facts = tuple(derived_facts)
    report: ObligationReport = evaluate_obligations(
        dag,
        entity_manager,
        certificates,
        derived_facts,
        allow_uncalibrated=include_uncalibrated,
    )

    required_refs = set(dag.required_logical_refs())
    allowed_statuses = {CertificateStatus.CERTIFIED}
    if include_uncalibrated:
        allowed_statuses.add(CertificateStatus.UNCALIBRATED)

    region_cert_by_entity: dict[str, BindingCertificate] = {}
    for certificate in certificates:
        if certificate.predicate == "REFERENT_TO_REGION":
            region_cert_by_entity[certificate.entity_id] = certificate

    proof_entities: list[ProofEntity] = []
    for entity in entity_manager.all():
        if not set(entity.logical_refs).intersection(required_refs):
            continue
        region_cert = region_cert_by_entity.get(entity.entity_id)
        included = region_cert is not None and region_cert.certificate_id in report.included_certificate_ids
        region_for_state = entity.region if included and region_cert.status in allowed_statuses else None
        proof_entities.append(
            ProofEntity(
                entity.entity_id,
                entity.logical_refs,
                region_for_state,
                region_cert.certificate_id if region_cert else None,
                region_cert.status.value if region_cert else "UNRESOLVED",
                entity.provenance,
                (region_cert.dag_node_id,) if region_cert and region_cert.dag_node_id else (),
            )
        )

    facts: list[CertifiedFact | DerivedFact] = []
    dependencies: dict[str, tuple[str, ...]] = {}
    for certificate in sorted(certificates, key=lambda item: item.certificate_id):
        if certificate.certificate_id not in report.included_certificate_ids:
            continue
        if certificate.predicate == "REFERENT_TO_REGION":
            continue
        fact_id = f"fact_{certificate.certificate_id}"
        dependency_ids = tuple(dep for dep in certificate.dependency_ids if not dep.startswith("localize_"))
        fact = CertifiedFact(
            fact_id,
            certificate.entity_id,
            certificate.predicate,
            certificate.proposed_value,
            certificate.certificate_id,
            certificate.status.value,
            certificate.provenance,
            dependency_ids,
            certificate.dag_node_id,
        )
        facts.append(fact)
        dependencies[fact_id] = dependency_ids

    for derived in sorted(derived_facts, key=lambda item: item.fact_id):
        if getattr(derived, "fact_id", "") not in report.included_derived_ids:
            continue
        facts.append(derived)
        dependencies[derived.fact_id] = tuple(getattr(derived, "dependency_ids", ()) or getattr(derived, "source_certificate_ids", ()))

    included_ids = set(report.included_certificate_ids) | set(report.included_derived_ids)
    closure_valid = all(set(deps).issubset(included_ids) for deps in dependencies.values())
    state = ProofState(
        "realbind_minimal_certified_proof_state_v2",
        dag.task_type,
        "realbind_corrected_predicate_dag_v1",
        tuple(sorted(proof_entities, key=lambda item: item.entity_id)),
        tuple(sorted(facts, key=lambda item: item.fact_id)),
        dependencies,
        (
            "compiled_by_predicate_dag",
            f"proof_complete={report.proof_complete}",
            f"certified_proof_complete={report.certified_proof_complete}",
            f"diagnostic={include_uncalibrated}",
            "unused_observations_excluded",
            "raw_image_exits_before_reasoning",
        ),
        report.obligations,
        tuple(dag.required_node_ids()),
        report.proof_complete,
        report.certified_proof_complete,
        include_uncalibrated,
        bool(report.dependency_closure_valid and closure_valid),
    )
    state.to_dict()
    return state
