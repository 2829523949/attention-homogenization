"""Executable corrected DECOMPOSE -> ... -> HARD HANDOFF pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.acquisition.router import AcquisitionRouter
from harness.certification.certificate import BindingCertificate
from harness.certification.certifier import CertificationConfig, EvidenceCertifier
from harness.certification.evidence import EvidenceScore
from harness.entities.entity_manager import EntityManager
from harness.evaluators.leakage_checks import assert_d_drive_path, assert_inference_safe, assert_no_image_payload
from harness.planner.predicate_dag import Operation, build_predicate_dag
from harness.processors.geometry import derive_horizontal_fact
from harness.processors.defocus import DEFOCUS_CONFIG
from harness.proof_state.compiler import DerivedFact, compile_minimal_proof_state
from harness.proof_state.controller import decide_controller
from harness.proof_state.serializer import serialize_proof_state
from harness.reasoners.hard_handoff_qwen import Qwen35HardHandoffReasoner
from harness.referents.referent_resolver import ReferentInterfaceStatus, resolve_question
from harness.referents.region_query import build_region_query
from harness.views.focus import focus_view
from harness.views.raw import ViewArtifact, raw_view
from harness.views.remove import remove_view


@dataclass(frozen=True)
class RefactoredCaseInput:
    case_id: str
    task_type: str
    image_path: Path
    question: str

    def to_dict(self) -> dict[str, Any]:
        payload = {"case_id": self.case_id, "task_type": self.task_type, "image_path": str(self.image_path), "question": self.question}
        assert_inference_safe(payload, context="main case input")
        return payload


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        payload = dict(value.__dict__)
        for key in ("bbox_xyxy_pixels", "bbox_xyxy_normalized"):
            if isinstance(payload.get(key), tuple):
                payload[key] = list(payload[key])
        return {str(key): _json_safe(item) for key, item in payload.items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _capability_contract(registered: Any) -> dict[str, Any]:
    capability = registered.capability
    return {
        "processor": registered.name,
        "supports_certification_score": bool(capability.supports_certification_score),
        "supports_same_claim_rescoring": bool(capability.supports_same_claim_rescoring),
        "comparable_across_counterfactual_views": bool(capability.comparable_across_counterfactual_views),
        "score_semantics": capability.score_semantics,
        "score_normalization": capability.score_normalization,
        "requires_fixed_region": bool(getattr(capability, "requires_fixed_region", False)),
        "requires_fixed_query": bool(getattr(capability, "requires_fixed_query", False)),
        "evidence_kind": getattr(capability, "evidence_kind", "MODEL_DERIVED_EVIDENCE"),
    }


def _unsupported_region_scores(processor: str, region: Any, reason: str) -> dict[str, EvidenceScore]:
    return {
        view_id: EvidenceScore.unsupported(processor, reason=reason, claimed_value=region, view_id=view_id)
        for view_id in ("RAW", "FOCUS", "REMOVE")
    }


def _supports_predicate(registered: Any, predicate: str) -> bool:
    return registered is not None and predicate in set(getattr(registered.capability, "supported_predicates", ()))


def _is_full_region_scorer(registered: Any) -> bool:
    return _supports_predicate(registered, "REFERENT_TO_REGION") and bool(
        getattr(registered.capability, "supports_certification_score", False)
    )


def _is_semantic_region_scorer(registered: Any) -> bool:
    return _supports_predicate(registered, "SEMANTIC_REGION_SUPPORT") and hasattr(
        registered.implementation, "score_fixed_region_query"
    )


def _candidate_count(router: AcquisitionRouter, entity_id: str) -> int:
    for step in router.trace:
        if step.operation == "LOCALIZE" and step.entity_id == entity_id:
            return int(step.outputs.get("candidate_count", 0))
    return 0


def run_refactored_case(
    case: RefactoredCaseInput,
    *,
    registry: Any,
    reasoner: Any,
    output_root: Path,
    certification_config: CertificationConfig | None = None,
    acquisition_processor: str = "florence2_query_grounding",
    grounding_score_processor: str | None = None,
    color_processor: str = "siglip2_color",
) -> dict[str, Any]:
    """Run one case using only corrected main-method interfaces.

    Acquisition and score-based certification are separated explicitly. A
    processor that cannot expose a valid fixed-claim score yields an
    UNCERTAIN/UNSUPPORTED certificate and the controller abstains; it never
    receives a fabricated confidence.
    """

    if not case.image_path.is_file():
        raise FileNotFoundError(case.image_path)
    assert_d_drive_path(output_root, artifact_kind="refactor output")
    resolved = resolve_question(case.question, case.task_type)
    if resolved.status is ReferentInterfaceStatus.REFERENT_INTERFACE_UNDERSPECIFIED:
        raise RuntimeError("REFERENT_INTERFACE_UNDERSPECIFIED: " + "; ".join(resolved.notes))
    dag = build_predicate_dag(case.question, case.task_type, resolved)
    manager = EntityManager()
    router = AcquisitionRouter(registry, manager)
    config = certification_config or CertificationConfig()
    certifier = EvidenceCertifier(config)
    trace: dict[str, Any] = {
        "case_id": case.case_id,
        "task_type": case.task_type,
        "image_path": str(case.image_path),
        "question": case.question,
        "predicate_dag": dag.to_dict(),
        "referents": resolved.to_dict(),
        "grounding_queries": [query.to_dict() for query in resolved.queries()],
        "processor_contracts": registry.describe() if hasattr(registry, "describe") else {},
        "acquisition_trace": [],
        "entity_creation_and_reuse": [],
        "raw_specialist_observations": [],
        "views": [],
        "binding_certificates": [],
        "derived_geometry_facts": [],
        "derived_selection_facts": [],
        "selection": None,
        "region_certification": [],
        "region_query_evidence": [],
        "diagnostic_color_probe": None,
    }

    # ------------------------------------------------------------------
    # ACQUIRE persistent entity regions.
    # ------------------------------------------------------------------
    acquisition_registered = registry.require(acquisition_processor, "grounding")
    observations = router.acquire_regions(case.image_path, dag, resolved.specs, grounding_processor=acquisition_processor)
    observation_by_entity = {observation.entity_id: observation for observation in observations}
    trace["acquisition_trace"] = [item.to_dict() for item in router.trace]
    trace["entity_creation_and_reuse"] = [event.to_dict() for event in manager.events]
    trace["raw_specialist_observations"].extend([observation.to_dict() for observation in observations])

    # ------------------------------------------------------------------
    # COUNTERFACTUAL VIEWS and REGION CERTIFICATION.
    # ------------------------------------------------------------------
    view_root = output_root / f"{case.case_id}__views"
    view_root.mkdir(parents=True, exist_ok=True)
    view_config = dict(DEFOCUS_CONFIG)
    views_by_entity: dict[str, dict[str, ViewArtifact]] = {}
    region_certificates: list[BindingCertificate] = []
    region_cert_by_ref: dict[str, BindingCertificate] = {}

    score_registered = None
    if grounding_score_processor is not None:
        score_registered = registry.require(grounding_score_processor, "region_evidence")
    elif acquisition_registered.capability.supports_certification_score:
        score_registered = acquisition_registered
    full_score_registered = score_registered if _is_full_region_scorer(score_registered) else None
    semantic_score_registered = score_registered if _is_semantic_region_scorer(score_registered) else None

    for spec in resolved.specs:
        entity = manager.get(spec.logical_ref)
        if entity.region is None:
            continue
        raw = raw_view(case.image_path, source_entity=entity.entity_id, predicted_region=entity.region)
        focus = focus_view(case.image_path, entity.region, view_root / f"{entity.entity_id}__focus.png", entity_id=entity.entity_id, config=view_config)
        remove = remove_view(case.image_path, entity.region, view_root / f"{entity.entity_id}__remove.png", entity_id=entity.entity_id, config=view_config)
        views = {artifact.view_id: artifact for artifact in (raw, focus, remove)}
        views_by_entity[entity.entity_id] = views
        trace["views"].extend([artifact.to_dict() for artifact in views.values()])

        query = entity.grounding_query
        if query is None:
            raise RuntimeError(f"missing persistent grounding query for {entity.entity_id}")
        if full_score_registered is None:
            reason = (
                f"CERTIFICATION_SCORE_UNSUPPORTED: acquisition processor {acquisition_registered.name} "
                "does not expose a valid fixed-claim region score; no substitute proxy is used."
            )
            scores = _unsupported_region_scores(acquisition_registered.name, entity.region, reason)
            contract = _capability_contract(acquisition_registered)
            scoring_processor_name = acquisition_registered.name
        else:
            score_fn = full_score_registered.implementation.score
            scores = {
                "RAW": score_fn(Path(raw.path), query, entity.region, view_id="RAW"),
                "FOCUS": score_fn(Path(focus.path), query, entity.region, view_id="FOCUS"),
                "REMOVE": score_fn(Path(remove.path), query, entity.region, view_id="REMOVE"),
            }
            contract = _capability_contract(full_score_registered)
            scoring_processor_name = full_score_registered.name

        certificate = certifier.certify(
            entity_id=entity.entity_id,
            predicate="REFERENT_TO_REGION",
            proposed_value=entity.region,
            binding_kind="entity_binding",
            raw=scores["RAW"],
            focus=scores["FOCUS"],
            remove=scores["REMOVE"],
            provenance=("QUESTION_TEXT", "TOOL_OBSERVATION", "PREDICTED_REGION"),
            dag_node_id=f"certify_region_{spec.logical_ref}",
            dependency_ids=(f"localize_{spec.logical_ref}",),
            score_contract=contract,
            acquisition_processor=acquisition_registered.name,
            view_ids=("RAW", "FOCUS", "REMOVE"),
        )
        region_certificates.append(certificate)
        region_cert_by_ref[spec.logical_ref] = certificate
        manager.attach_certificate(spec.logical_ref, certificate.certificate_id, certificate.status.value)
        trace["binding_certificates"].append(certificate.to_dict())
        trace["region_certification"].append(
            {
                "entity_id": entity.entity_id,
                "logical_ref": spec.logical_ref,
                "s_raw": certificate.raw_score.normalized_score,
                "s_focus": certificate.focus_score.normalized_score,
                "s_remove": certificate.remove_score.normalized_score,
                "focus_delta": certificate.focus_delta,
                "binding_score": certificate.necessity_delta,
                "status": certificate.status.value,
                "failure_code": certificate.diagnostics.get("failure_code"),
                "score_processor": scoring_processor_name,
                "acquisition_processor": acquisition_registered.name,
                "score_semantics": certificate.diagnostics.get("score_semantics"),
                "score_normalization": certificate.diagnostics.get("score_normalization"),
                "raw_observation_score_support": observation_by_entity.get(entity.entity_id).score.support if entity.entity_id in observation_by_entity else None,
            }
        )

        # Separate semantic evidence layer: SigLIP2 scores the fixed
        # persistent region against the fixed legal query.  This DOES NOT
        # satisfy the full REFERENT_TO_REGION identity obligation.
        if semantic_score_registered is not None:
            region_query = build_region_query(query)
            semantic_scores = {
                "RAW": semantic_score_registered.implementation.score_fixed_region_query(
                    Path(raw.path), entity.entity_id, entity.region, region_query, view_id="RAW"
                ),
                "FOCUS": semantic_score_registered.implementation.score_fixed_region_query(
                    Path(focus.path), entity.entity_id, entity.region, region_query, view_id="FOCUS"
                ),
                "REMOVE": semantic_score_registered.implementation.score_fixed_region_query(
                    Path(remove.path), entity.entity_id, entity.region, region_query, view_id="REMOVE"
                ),
            }
            fixed_region_ids = {score.diagnostics.get("region_id") for score in semantic_scores.values()}
            fixed_query_hashes = {score.diagnostics.get("query_hash") for score in semantic_scores.values()}
            fixed_template_ids = {score.diagnostics.get("query_template_id") for score in semantic_scores.values()}
            fixed_crop_policies = {score.diagnostics.get("crop_policy_id") for score in semantic_scores.values()}
            if not (len(fixed_region_ids) == len(fixed_query_hashes) == len(fixed_template_ids) == len(fixed_crop_policies) == 1):
                raise RuntimeError("fixed-region/fixed-query contract violated before certification")
            semantic_certificate = certifier.certify(
                entity_id=entity.entity_id,
                predicate="SEMANTIC_REGION_SUPPORT",
                proposed_value=semantic_scores["RAW"].claimed_value,
                binding_kind="region_content_evidence",
                raw=semantic_scores["RAW"],
                focus=semantic_scores["FOCUS"],
                remove=semantic_scores["REMOVE"],
                provenance=("QUESTION_TEXT", "TOOL_OBSERVATION", "PREDICTED_REGION", "COUNTERFACTUAL_VIEWS"),
                dag_node_id=f"semantic_region_support_{spec.logical_ref}",
                dependency_ids=(f"localize_{spec.logical_ref}",),
                score_contract=_capability_contract(semantic_score_registered),
                acquisition_processor=acquisition_registered.name,
                view_ids=("RAW", "FOCUS", "REMOVE"),
                evidence_kind="FIXED_REGION_QUERY_COMPATIBILITY",
                identity_layer="SEMANTIC_REGION_SUPPORT",
                calibration_status=config.calibration_mode,
                gate_policy="NON_CALIBRATED_INFORMATIONAL",
            )
            candidate_count = _candidate_count(router, entity.entity_id)
            if candidate_count > 1:
                uniqueness_status = "MULTIPLE_COMPATIBLE_INSTANCES"
            elif candidate_count == 1:
                uniqueness_status = "UNKNOWN"
            else:
                uniqueness_status = "NOT_CHECKED"
            acquisition_observation = observation_by_entity.get(entity.entity_id)
            trace["region_query_evidence"].append(
                {
                    "case_id": case.case_id,
                    "logical_ref": spec.logical_ref,
                    "entity_id": entity.entity_id,
                    "referent_spec": spec.to_dict(),
                    "referent_provenance": list(region_query.provenance),
                    "acquisition_processor": acquisition_registered.name,
                    "acquisition_output": acquisition_observation.to_dict() if acquisition_observation else None,
                    "persistent_bbox_xyxy_pixels": [float(value) for value in entity.region.bbox_xyxy_pixels],
                    "region_id": semantic_certificate.raw_score.diagnostics.get("region_id"),
                    "instance_uniqueness_status": uniqueness_status,
                    "candidate_count": candidate_count,
                    "query_text": region_query.text,
                    "query_template_id": region_query.template_id,
                    "query_hash": region_query.query_hash,
                    "query_provenance": list(region_query.provenance),
                    "crop_policy_id": semantic_certificate.raw_score.diagnostics.get("crop_policy_id"),
                    "score_processor": semantic_score_registered.name,
                    "model_revision": semantic_certificate.raw_score.diagnostics.get("model_revision"),
                    "model_fingerprint": semantic_certificate.raw_score.diagnostics.get("model_fingerprint"),
                    "raw_crop_path": None,
                    "focus_crop_path": None,
                    "remove_crop_path": None,
                    "s_raw_raw": semantic_certificate.raw_score.raw_score,
                    "s_focus_raw": semantic_certificate.focus_score.raw_score,
                    "s_remove_raw": semantic_certificate.remove_score.raw_score,
                    "s_raw": semantic_certificate.raw_score.normalized_score,
                    "s_focus": semantic_certificate.focus_score.normalized_score,
                    "s_remove": semantic_certificate.remove_score.normalized_score,
                    "focus_delta": semantic_certificate.focus_delta,
                    "binding_evidence": semantic_certificate.necessity_delta,
                    "evidence_kind": semantic_certificate.evidence_kind,
                    "identity_layer": semantic_certificate.identity_layer,
                    "calibration_status": semantic_certificate.calibration_status,
                    "evidence_status": semantic_certificate.status.value,
                    "failure_code": semantic_certificate.diagnostics.get("failure_code"),
                    "diagnostic_pattern_pass": semantic_certificate.diagnostics.get("diagnostic_pattern_pass"),
                    "diagnostic_pattern_note": semantic_certificate.diagnostics.get("diagnostic_pattern_note"),
                    "full_logical_identity_status": certificate.status.value,
                    "normalization_semantics": semantic_certificate.raw_score.normalization_semantics,
                    "certificate": semantic_certificate.to_dict(),
                    "controller_decision": None,
                    "reasoner_called": False,
                    "qwen_called": False,
                }
            )
            trace["region_query_evidence"][-1] = _json_safe(trace["region_query_evidence"][-1])

    # ------------------------------------------------------------------
    # Deterministic geometry and selection derived facts.
    # ------------------------------------------------------------------
    derived_facts: list[DerivedFact] = []
    geometry_fact: DerivedFact | None = None
    selection_fact: DerivedFact | None = None
    accepted_region_statuses = {"CERTIFIED", "UNCALIBRATED"}
    requires_selection = any(node.operation is Operation.SELECT_RELATION for node in dag.nodes)
    requires_color = any(node.operation is Operation.READ_COLOR for node in dag.nodes)
    if {"A", "B"}.issubset(region_cert_by_ref):
        cert_a, cert_b = region_cert_by_ref["A"], region_cert_by_ref["B"]
        if cert_a.status.value in accepted_region_statuses and cert_b.status.value in accepted_region_statuses:
            entity_a, entity_b = manager.get("A"), manager.get("B")
            geometry_fact = derive_horizontal_fact(
                entity_a.entity_id,
                entity_a.region,
                entity_b.entity_id,
                entity_b.region,
                source_certificate_ids=(cert_a.certificate_id, cert_b.certificate_id),
                dag_node_id="compare_horizontal",
                dependency_ids=(cert_a.certificate_id, cert_b.certificate_id),
            )
            derived_facts.append(geometry_fact)
            trace["derived_geometry_facts"].append({**geometry_fact.to_dict(), "state_inclusion": "ELIGIBLE_DEPENDING_ON_STATUS"})
        else:
            trace["derived_geometry_facts"].append(
                {
                    "fact_id": "derived_horizontal_A_B",
                    "status": "BLOCKED_UNCERTIFIED_REGION",
                    "source_certificate_ids": [cert_a.certificate_id, cert_b.certificate_id],
                    "source_entity_ids": [manager.get("A").entity_id, manager.get("B").entity_id],
                }
            )

    if requires_selection and geometry_fact is not None and geometry_fact.value != "overlapping_center":
        desired = "left" if str(case.task_type).endswith("LEFT") else "right"
        selected_ref = "A" if geometry_fact.value == desired else "B"
        selected_entity = manager.get(selected_ref)
        selection_fact = DerivedFact(
            f"derived_select_{selected_ref}",
            "SELECT_RELATION",
            {"logical_ref": selected_ref, "entity_id": selected_entity.entity_id, "relation": geometry_fact.value, "requested_side": desired},
            (manager.get("A").entity_id, manager.get("B").entity_id),
            tuple(geometry_fact.source_certificate_ids),
            "deterministic_center_side_selection_v1",
            ("PREVIOUS_CERTIFIED_FACT", "DETERMINISTIC_GEOMETRY"),
            "select_target",
            (geometry_fact.fact_id,),
        )
        derived_facts.append(selection_fact)
        trace["derived_selection_facts"].append({**selection_fact.to_dict(), "state_inclusion": "ELIGIBLE_DEPENDING_ON_STATUS"})
        trace["selection"] = {
            "status": "SELECTED",
            "requested_side": desired,
            "relation": geometry_fact.value,
            "logical_ref": selected_ref,
            "entity_id": selected_entity.entity_id,
            "dependency_fact_id": geometry_fact.fact_id,
        }
    elif geometry_fact is not None and requires_selection:
        trace["selection"] = {"status": "ABSTAIN_OVERLAPPING_CENTER", "relation": geometry_fact.value}
    elif not requires_selection:
        trace["selection"] = {"status": "NOT_REQUIRED_BY_DAG"}
    else:
        trace["selection"] = {"status": "BLOCKED_UNCERTIFIED_ENTITY_BINDING"}

    # ------------------------------------------------------------------
    # Attribute acquisition/certification for the selected entity.
    # ------------------------------------------------------------------
    color_certificates: list[BindingCertificate] = []
    color_certificate: BindingCertificate | None = None
    color_registered = None
    if case.task_type in {"I2C_A", "I2C_B"}:
        target_ref = "A" if case.task_type.endswith("A") else "B"
        dependency_ids = (region_cert_by_ref[target_ref].certificate_id,) if target_ref in region_cert_by_ref else ()
    else:
        target_ref = trace["selection"].get("logical_ref") if trace["selection"].get("status") == "SELECTED" else None
        dependency_ids = ()
        if target_ref is not None and target_ref in region_cert_by_ref and selection_fact is not None:
            dependency_ids = (region_cert_by_ref[target_ref].certificate_id, selection_fact.fact_id)

    target_region_status = region_cert_by_ref.get(target_ref).status.value if target_ref in region_cert_by_ref else None
    if requires_color and target_ref is not None and target_region_status in accepted_region_statuses:
        color_registered = registry.require(color_processor, "color")
        color_observation = router.acquire_color(case.image_path, target_ref, color_processor=color_processor)
        trace["acquisition_trace"] = [item.to_dict() for item in router.trace]
        trace["raw_specialist_observations"].append(color_observation.to_dict())
        target_entity = manager.get(target_ref)
        views = views_by_entity[target_entity.entity_id]
        if color_registered.capability.supports_certification_score:
            score_fn = color_registered.implementation.score
            raw_score = score_fn(Path(views["RAW"].path), target_entity.entity_id, target_entity.region, color_observation.proposed_value, view_id="RAW")
            focus_score = score_fn(Path(views["FOCUS"].path), target_entity.entity_id, target_entity.region, color_observation.proposed_value, view_id="FOCUS")
            remove_score = score_fn(Path(views["REMOVE"].path), target_entity.entity_id, target_entity.region, color_observation.proposed_value, view_id="REMOVE")
        else:
            reason = f"CERTIFICATION_SCORE_UNSUPPORTED: color processor {color_registered.name} has no fixed-candidate score"
            raw_score, focus_score, remove_score = (
                EvidenceScore.unsupported(color_registered.name, reason=reason, claimed_value=color_observation.proposed_value, view_id=view_id)
                for view_id in ("RAW", "FOCUS", "REMOVE")
            )
        node_id = "certify_color" if case.task_type in {"I2C_A", "I2C_B"} else "certify_selected_color"
        color_certificate = certifier.certify(
            entity_id=target_entity.entity_id,
            predicate="COLOR",
            proposed_value=color_observation.proposed_value,
            binding_kind="entity_fact",
            raw=raw_score,
            focus=focus_score,
            remove=remove_score,
            provenance=("PREVIOUS_ENTITY_STATE", "TOOL_OBSERVATION", "COUNTERFACTUAL_VIEWS"),
            dag_node_id=node_id,
            dependency_ids=dependency_ids,
            score_contract=_capability_contract(color_registered),
            acquisition_processor=color_registered.name,
        )
        color_certificates.append(color_certificate)
        manager.attach_certificate(target_ref, color_certificate.certificate_id, color_certificate.status.value)
        trace["binding_certificates"].append(color_certificate.to_dict())
    else:
        if requires_color:
            trace["selection"] = trace.get("selection") or {}
            trace["selection"]["color_acquisition"] = "BLOCKED_REQUIRED_REGION_CERTIFICATE"
        elif not requires_selection:
            trace["selection"] = {"status": "NOT_REQUIRED_BY_DAG"}

    # ------------------------------------------------------------------
    # MINIMAL CERTIFIED PROOF STATE and controller abstention gate.
    # ------------------------------------------------------------------
    strict_state = compile_minimal_proof_state(dag, manager, region_certificates + color_certificates, derived_facts, include_uncalibrated=False)
    diagnostic_state = compile_minimal_proof_state(dag, manager, region_certificates + color_certificates, derived_facts, include_uncalibrated=True)
    decision = decide_controller(
        dag,
        manager,
        region_certificates + color_certificates,
        derived_facts,
        calibration_mode=config.calibration_mode,
        calibration_artifact_id=config.calibration_artifact_id,
    )
    assert_no_image_payload(strict_state.to_dict(), context="main final reasoner input")

    trace["minimal_certified_proof_state"] = strict_state.to_dict()
    trace["minimal_certified_proof_state_text"] = serialize_proof_state(strict_state)
    trace["diagnostic_proof_state"] = diagnostic_state.to_dict()
    trace["diagnostic_proof_state_text"] = serialize_proof_state(diagnostic_state)
    trace["proof_state_dependency_satisfaction"] = {
        "strict": {
            "proof_complete": strict_state.proof_complete,
            "certified_proof_complete": strict_state.certified_proof_complete,
            "dependency_closure_valid": strict_state.dependency_closure_valid,
            "obligations": [item.to_dict() for item in strict_state.obligations],
        },
        "diagnostic": {
            "proof_complete": diagnostic_state.proof_complete,
            "dependency_closure_valid": diagnostic_state.dependency_closure_valid,
            "obligations": [item.to_dict() for item in diagnostic_state.obligations],
        },
    }
    trace["controller_decision"] = decision.to_dict()
    trace["reasoner_called"] = False
    trace["qwen_called"] = False
    trace["hard_handoff_reasoner_input"] = None
    trace["final_answer"] = None
    trace["reasoner_output"] = None

    if decision.allow_reasoner:
        reasoning = reasoner.generate(case.question, strict_state)
        trace["reasoner_called"] = True
        trace["qwen_called"] = isinstance(reasoner, Qwen35HardHandoffReasoner)
        trace["hard_handoff_reasoner_input"] = {
            "question": case.question,
            "proof_state": strict_state.to_dict(),
            "image_received": False,
            "api": "HardHandoffReasoner.generate(question, proof_state)",
        }
        trace["final_answer"] = reasoning.get("answer")
        trace["reasoner_output"] = reasoning

    for evidence_record in trace["region_query_evidence"]:
        evidence_record["controller_decision"] = decision.status
        evidence_record["reasoner_called"] = trace["reasoner_called"]
        evidence_record["qwen_called"] = trace["qwen_called"]
    trace["entity_creation_and_reuse"] = [event.to_dict() for event in manager.events]
    trace["acquisition_trace"] = [item.to_dict() for item in router.trace]
    trace["architecture_status"] = {
        "raw_image_exited_before_reasoning": True,
        "main_method": True,
        "calibration_mode": config.calibration_mode,
        "calibration_artifact_id": config.calibration_artifact_id,
        "scientific_gain_claim": False,
        "remove_operator": "target_region_neutral_fill_ablation_v2",
    }
    return _json_safe(trace)
