"""Controller-level proof-completeness and abstention gate.

The final language reasoner is never asked whether the evidence is
sufficient.  This deterministic controller decides, before any model call,
whether every required Predicate-DAG obligation is backed by certified
evidence with a satisfied dependency closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from harness.certification.certificate import BindingCertificate
from harness.entities.entity_manager import EntityManager
from harness.planner.predicate_dag import Operation, PredicateDAG

from .obligations import (
    MISSING,
    REJECTED,
    SATISFIED,
    UNCALIBRATED,
    UNRESOLVED,
    UNSUPPORTED,
    ObligationReport,
    evaluate_obligations,
)

PROOF_COMPLETE = "PROOF_COMPLETE"
ABSTAIN_UNCERTIFIED_BINDING = "ABSTAIN_UNCERTIFIED_BINDING"
ABSTAIN_UNCALIBRATED_BINDING = "ABSTAIN_UNCALIBRATED_BINDING"
ABSTAIN_MISSING_ATTRIBUTE = "ABSTAIN_MISSING_ATTRIBUTE"
ABSTAIN_INCOMPLETE_PROOF_STATE = "ABSTAIN_INCOMPLETE_PROOF_STATE"
ABSTAIN_UNRESOLVED_REFERENT = "ABSTAIN_UNRESOLVED_REFERENT"


@dataclass(frozen=True)
class ControllerDecision:
    status: str
    proof_complete: bool
    allow_reasoner: bool
    reason_code: str
    rationale: str
    obligations: tuple[Any, ...] = ()
    rejected_nodes: tuple[str, ...] = ()
    unsupported_nodes: tuple[str, ...] = ()
    unresolved_nodes: tuple[str, ...] = ()
    missing_nodes: tuple[str, ...] = ()
    uncalibrated_nodes: tuple[str, ...] = ()
    dependency_closure_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "proof_complete": self.proof_complete,
            "allow_reasoner": self.allow_reasoner,
            "reason_code": self.reason_code,
            "rationale": self.rationale,
            "obligations": [item.to_dict() if hasattr(item, "to_dict") else item for item in self.obligations],
            "rejected_nodes": list(self.rejected_nodes),
            "unsupported_nodes": list(self.unsupported_nodes),
            "unresolved_nodes": list(self.unresolved_nodes),
            "missing_nodes": list(self.missing_nodes),
            "uncalibrated_nodes": list(self.uncalibrated_nodes),
            "dependency_closure_valid": self.dependency_closure_valid,
        }


def _nodes(report: ObligationReport, status: str) -> tuple[str, ...]:
    return tuple(item.node_id for item in report.obligations if item.status == status)


def decide_controller(
    dag: PredicateDAG,
    entity_manager: EntityManager,
    certificates: Iterable[BindingCertificate],
    derived_facts: Iterable[Any],
    *,
    calibration_mode: str,
    calibration_artifact_id: str | None = None,
) -> ControllerDecision:
    certificates = tuple(certificates)
    derived_facts = tuple(derived_facts)
    strict = evaluate_obligations(dag, entity_manager, certificates, derived_facts, allow_uncalibrated=False)
    diagnostic = evaluate_obligations(dag, entity_manager, certificates, derived_facts, allow_uncalibrated=True)
    uncalibrated = set(_nodes(diagnostic, UNCALIBRATED))
    if strict.proof_complete and strict.dependency_closure_valid and not uncalibrated:
        return ControllerDecision(
            PROOF_COMPLETE,
            True,
            True,
            "ALL_REQUIRED_OBLIGATIONS_CERTIFIED",
            "all required Predicate-DAG obligations are CERTIFIED with satisfied dependencies",
            diagnostic.obligations,
            dependency_closure_valid=strict.dependency_closure_valid,
        )

    rejected = _nodes(diagnostic, REJECTED)
    unsupported = _nodes(diagnostic, UNSUPPORTED)
    unresolved = _nodes(diagnostic, UNRESOLVED)
    missing = _nodes(diagnostic, MISSING)

    if diagnostic.proof_complete and uncalibrated:
        return ControllerDecision(
            ABSTAIN_UNCALIBRATED_BINDING,
            False,
            False,
            "UNCALIBRATED_DIAGNOSTIC_BINDING",
            (
                "all required evidence patterns are present but the only passing "
                "certificates are UNCALIBRATED diagnostics; no final model handoff is permitted"
            ),
            diagnostic.obligations,
            uncalibrated_nodes=tuple(sorted(uncalibrated)),
            dependency_closure_valid=diagnostic.dependency_closure_valid,
        )

    if unsupported:
        return ControllerDecision(
            ABSTAIN_UNCERTIFIED_BINDING,
            False,
            False,
            "CERTIFICATION_SCORE_UNSUPPORTED",
            "a required binding has no valid certification score; refusing to certify or reason",
            diagnostic.obligations,
            unsupported_nodes=unsupported,
            rejected_nodes=rejected,
            unresolved_nodes=unresolved,
            missing_nodes=missing,
            uncalibrated_nodes=tuple(sorted(uncalibrated)),
            dependency_closure_valid=diagnostic.dependency_closure_valid,
        )

    if rejected:
        return ControllerDecision(
            ABSTAIN_UNCERTIFIED_BINDING,
            False,
            False,
            "REQUIRED_BINDING_REJECTED",
            "a required binding failed the counterfactual necessity/base gate",
            diagnostic.obligations,
            rejected_nodes=rejected,
            unresolved_nodes=unresolved,
            missing_nodes=missing,
            uncalibrated_nodes=tuple(sorted(uncalibrated)),
            dependency_closure_valid=diagnostic.dependency_closure_valid,
        )

    attribute_ops = {Operation.READ_COLOR.value, Operation.READ_CATEGORY.value, Operation.READ_STATE.value}
    missing_attribute = tuple(
        item.node_id for item in diagnostic.obligations if item.operation in attribute_ops and item.status in {MISSING, UNRESOLVED}
    )
    if missing_attribute:
        return ControllerDecision(
            ABSTAIN_MISSING_ATTRIBUTE,
            False,
            False,
            "REQUIRED_ATTRIBUTE_MISSING",
            "a required attribute observation/certificate is missing",
            diagnostic.obligations,
            missing_nodes=missing_attribute + tuple(missing),
            unresolved_nodes=unresolved,
            uncalibrated_nodes=tuple(sorted(uncalibrated)),
            dependency_closure_valid=diagnostic.dependency_closure_valid,
        )

    return ControllerDecision(
        ABSTAIN_INCOMPLETE_PROOF_STATE,
        False,
        False,
        "INCOMPLETE_PROOF_STATE",
        "required proof obligations remain unresolved or missing; final reasoning is blocked",
        diagnostic.obligations,
        unresolved_nodes=unresolved,
        missing_nodes=missing,
        uncalibrated_nodes=tuple(sorted(uncalibrated)),
        dependency_closure_valid=diagnostic.dependency_closure_valid,
    )
