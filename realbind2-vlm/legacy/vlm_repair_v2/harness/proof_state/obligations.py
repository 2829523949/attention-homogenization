"""Predicate-DAG obligation evaluation and dependency closure.

A proof state may contain only facts whose source DAG node is required to
answer the query and whose own dependencies are already satisfied by
certified evidence or deterministic derivations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from harness.certification.certificate import BindingCertificate, CertificateStatus
from harness.entities.entity_manager import EntityManager
from harness.planner.predicate_dag import Operation, PredicateDAG

SATISFIED = "SATISFIED"
UNCALIBRATED = "UNCALIBRATED"
REJECTED = "REJECTED"
UNSUPPORTED = "UNSUPPORTED"
MISSING = "MISSING"
UNRESOLVED = "UNRESOLVED"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class ProofObligation:
    node_id: str
    operation: str
    predicate: str | None
    status: str
    detail: str
    evidence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "predicate": self.predicate,
            "status": self.status,
            "detail": self.detail,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class ObligationReport:
    obligations: tuple[ProofObligation, ...]
    proof_complete: bool
    certified_proof_complete: bool
    dependency_closure_valid: bool
    included_certificate_ids: frozenset[str]
    included_derived_ids: frozenset[str]
    status_by_node: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligations": [item.to_dict() for item in self.obligations],
            "proof_complete": self.proof_complete,
            "certified_proof_complete": self.certified_proof_complete,
            "dependency_closure_valid": self.dependency_closure_valid,
            "included_certificate_ids": sorted(self.included_certificate_ids),
            "included_derived_ids": sorted(self.included_derived_ids),
            "status_by_node": dict(sorted(self.status_by_node.items())),
        }


def _logical_ref(value: str) -> str | None:
    if value in {"A", "B", "TARGET"}:
        return value
    if isinstance(value, str) and value.startswith("E_") and value[2:] in {"A", "B", "TARGET"}:
        return value[2:]
    return None


def _cert_status(certificate: BindingCertificate | None) -> str:
    if certificate is None:
        return MISSING
    if certificate.status is CertificateStatus.CERTIFIED:
        return SATISFIED
    if certificate.status is CertificateStatus.UNCALIBRATED:
        return UNCALIBRATED
    if certificate.status is CertificateStatus.REJECTED:
        return REJECTED
    failure = str(certificate.diagnostics.get("failure_code") or "")
    if failure == "CERTIFICATION_SCORE_UNSUPPORTED":
        return UNSUPPORTED
    return UNRESOLVED


def _cert_node_id(certificate: BindingCertificate, manager: EntityManager, dag: PredicateDAG) -> str | None:
    if certificate.dag_node_id:
        return certificate.dag_node_id
    node_ids = {node.node_id for node in dag.nodes}
    if certificate.predicate == "REFERENT_TO_REGION":
        for entity in manager.all():
            if entity.entity_id != certificate.entity_id:
                continue
            for ref in entity.logical_refs:
                candidate = f"certify_region_{ref}"
                if candidate in node_ids:
                    return candidate
    if certificate.predicate == "COLOR":
        for candidate in ("certify_selected_color", "certify_color"):
            if candidate in node_ids:
                return candidate
    return None


def evaluate_obligations(
    dag: PredicateDAG,
    entity_manager: EntityManager,
    certificates: Iterable[BindingCertificate],
    derived_facts: Iterable[Any],
    *,
    allow_uncalibrated: bool,
) -> ObligationReport:
    certificates = tuple(certificates)
    derived_facts = tuple(derived_facts)
    accepted_statuses = {SATISFIED}
    if allow_uncalibrated:
        accepted_statuses.add(UNCALIBRATED)

    # 1. Region/entity certificates establish structural facts.
    cert_node_ids: dict[str, str] = {}
    cert_statuses: dict[str, str] = {}
    effective_cert_status: dict[str, str] = {}
    for certificate in certificates:
        node_id = _cert_node_id(certificate, entity_manager, dag) or certificate.certificate_id
        cert_node_ids[certificate.certificate_id] = node_id
        status = _cert_status(certificate)
        cert_statuses[certificate.certificate_id] = status
        effective_cert_status[certificate.certificate_id] = status

    accepted_cert_ids: set[str] = {
        certificate.certificate_id
        for certificate in certificates
        if cert_statuses.get(certificate.certificate_id) in accepted_statuses
    }

    # 2. Deterministic derived facts are accepted only when every dependency
    #    is already accepted. Pipeline order is geometry then selection.
    derived_status_by_node: dict[str, str] = {}
    derived_node_by_id: dict[str, str | None] = {}
    accepted_derived_ids: set[str] = set()
    dependency_violations: list[str] = []
    for derived in derived_facts:
        node_id = getattr(derived, "dag_node_id", None) or (
            "compare_horizontal" if getattr(derived, "predicate", None) == "HORIZONTAL_RELATION"
            else "select_target" if getattr(derived, "predicate", None) == "SELECT_RELATION"
            else None
        )
        derived_node_by_id[getattr(derived, "fact_id", "")] = node_id
        deps = tuple(getattr(derived, "dependency_ids", ()) or ())
        if not deps:
            deps = tuple(getattr(derived, "source_certificate_ids", ()) or ())
        satisfied = all(dep in accepted_cert_ids or dep in accepted_derived_ids or dep.startswith("localize_") for dep in deps)
        if not satisfied:
            dependency_violations.append(getattr(derived, "fact_id", "derived"))
            if node_id:
                derived_status_by_node[node_id] = MISSING
            continue
        if node_id:
            derived_status_by_node[node_id] = SATISFIED
        accepted_derived_ids.add(getattr(derived, "fact_id", ""))

    accepted_evidence_ids = set(accepted_cert_ids) | set(accepted_derived_ids)

    # 3. Colour/attribute certificates depend on region certificates and on
    #    deterministic selection facts; propagate unsatisfied dependencies.
    for certificate in certificates:
        deps = tuple(certificate.dependency_ids or ())
        if not deps:
            continue
        satisfied = all(dep in accepted_evidence_ids or dep.startswith("localize_") for dep in deps)
        if not satisfied:
            effective_cert_status[certificate.certificate_id] = UNRESOLVED
            dependency_violations.append(certificate.certificate_id)
            accepted_cert_ids.discard(certificate.certificate_id)

    # 4. Walk required DAG nodes in topological order.
    obligations: list[ProofObligation] = []
    status_by_node: dict[str, str] = {}
    by_node_cert: dict[str, BindingCertificate] = {}
    for certificate in certificates:
        node_id = cert_node_ids[certificate.certificate_id]
        by_node_cert[node_id] = certificate

    def status_for_certificate(node_id: str) -> tuple[str, str | None, str]:
        cert = by_node_cert.get(node_id)
        if cert is None:
            return MISSING, None, "no certificate for required DAG node"
        status = effective_cert_status.get(cert.certificate_id, UNRESOLVED)
        return status, cert.certificate_id, cert.rationale

    def region_node_for_ref(ref: str) -> str:
        return f"certify_region_{ref}"

    def color_cert_node_for(read_node_id: str) -> str:
        if read_node_id == "read_selected_color":
            return "certify_selected_color"
        if read_node_id == "read_color":
            return "certify_color"
        return "certify_" + read_node_id.removeprefix("read_")

    for node in dag.required_proof_nodes():
        node_id = node.node_id
        operation = node.operation
        evidence_id: str | None = None
        detail = ""
        if operation is Operation.LOCALIZE:
            ref = _logical_ref(node.required_inputs[0]) or "TARGET"
            status, evidence_id, detail = status_for_certificate(region_node_for_ref(ref))
        elif operation is Operation.CERTIFY_BINDING:
            status, evidence_id, detail = status_for_certificate(node_id)
        elif operation is Operation.GET_POSITION:
            ref = _logical_ref(node.required_inputs[0]) or "TARGET"
            status, evidence_id, detail = status_for_certificate(region_node_for_ref(ref))
        elif operation is Operation.COMPARE_HORIZONTAL:
            status = derived_status_by_node.get(node_id, MISSING)
            evidence_id = next((getattr(item, "fact_id", None) for item in derived_facts if (getattr(item, "dag_node_id", None) or ("compare_horizontal" if getattr(item, "predicate", None) == "HORIZONTAL_RELATION" else None)) == node_id), None)
            detail = "deterministic geometry dependency" if status == SATISFIED else "geometry dependency not satisfied"
        elif operation is Operation.SELECT_RELATION:
            status = derived_status_by_node.get(node_id, MISSING)
            evidence_id = next((getattr(item, "fact_id", None) for item in derived_facts if (getattr(item, "dag_node_id", None) or ("select_target" if getattr(item, "predicate", None) == "SELECT_RELATION" else None)) == node_id), None)
            detail = "deterministic selection dependency" if status == SATISFIED else "selection dependency not satisfied"
        elif operation is Operation.READ_COLOR:
            status, evidence_id, detail = status_for_certificate(color_cert_node_for(node_id))
        elif operation in {Operation.READ_CATEGORY, Operation.READ_STATE}:
            status, evidence_id, detail = status_for_certificate(color_cert_node_for(node_id))
        else:
            status, evidence_id, detail = MISSING, None, "unsupported required proof operation"
        obligations.append(ProofObligation(node_id, operation.value, node.predicate, status, detail, evidence_id))
        status_by_node[node_id] = status

    required_proof_node_ids = {node.node_id for node in dag.required_proof_nodes()}
    included_cert_ids = {
        certificate.certificate_id
        for certificate in certificates
        if certificate.certificate_id in accepted_cert_ids and cert_node_ids.get(certificate.certificate_id) in required_proof_node_ids
    }
    included_derived_ids = {
        fact_id
        for fact_id, node_id in derived_node_by_id.items()
        if fact_id in accepted_derived_ids and node_id in required_proof_node_ids
    }
    accepted_nodes = [item for item in obligations if item.status in accepted_statuses]
    proof_complete = len(accepted_nodes) == len(obligations)
    certified_proof_complete = all(item.status == SATISFIED for item in obligations)
    closure_valid = not dependency_violations

    return ObligationReport(
        tuple(obligations),
        proof_complete,
        certified_proof_complete,
        closure_valid,
        frozenset(included_cert_ids),
        frozenset(included_derived_ids),
        status_by_node,
    )
