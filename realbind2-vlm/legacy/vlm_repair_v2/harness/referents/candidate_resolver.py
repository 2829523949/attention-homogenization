"""Joint hard-constraint candidate resolution; evaluator fields never enter this module."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping
from harness.evaluators.leakage_checks import assert_inference_safe
from harness.planner.predicate_dag import PredicateDAG, Operation
from harness.referents.referent_spec import ReferentSpec

VERSION = "constraint_only_joint_v1"

def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def center_x(candidate: Mapping[str, Any]) -> float:
    box = candidate["bbox_xyxy_pixels"]
    return (float(box[0]) + float(box[2])) / 2

def same_region(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    # IDs are scoped by reference; exact pixel box equality defines collision.
    return tuple(a["bbox_xyxy_pixels"]) == tuple(b["bbox_xyxy_pixels"])

def explicit_relation(dag: PredicateDAG, specs: Mapping[str, ReferentSpec]) -> str | None:
    """Only asserted relations count. COMPARE/SELECT nodes request computation."""
    values = []
    for spec in specs.values():
        for item in spec.relation_constraints:
            if item.value in {"LEFT_OF(A,B)", "RIGHT_OF(A,B)"}: values.append(item.value)
    for node in dag.nodes:
        if node.predicate in {"LEFT_OF(A,B)", "RIGHT_OF(A,B)"} and node.operation not in {Operation.COMPARE_HORIZONTAL, Operation.SELECT_RELATION}:
            values.append(node.predicate)
    if len(set(values)) > 1: raise ValueError("contradictory asserted relations")
    return values[0] if values else None

def distinct_required(dag: PredicateDAG) -> bool:
    return bool(re.fullmatch(r"Between the .+ and the .+,\n(?:what color is the one on the (?:left|right)|which one is on the (?:left|right))\?\nAnswer with only (?:the color|the object name)\.", dag.question, re.I))

@dataclass(frozen=True)
class ResolutionResult:
    family_id: str
    status: str
    initial_count: int
    valid_count: int
    candidate_set_hashes: dict[str, str]
    predicate_dag_hash: str
    constraints: tuple[str, ...]
    hypotheses: tuple[dict[str, Any], ...]
    selected: dict[str, str] | None
    persistent_entities: dict[str, dict[str, Any]]
    def to_dict(self) -> dict[str, Any]:
        value = vars(self).copy()
        value["constraints"] = list(self.constraints)
        value["hypotheses"] = list(self.hypotheses)
        assert_inference_safe(value, context="ResolutionResult")
        return value

class CandidateResolver:
    def resolve(self, predicate_dag: PredicateDAG, referent_specs: tuple[ReferentSpec, ...], candidate_sets: Mapping[str, list[dict[str, Any]]], persistent_state: Mapping[str, Any], *, family_id: str, run_id: str) -> ResolutionResult:
        specs = {s.logical_ref: s for s in referent_specs}
        if set(specs) != {"A", "B"} or set(candidate_sets) != {"A", "B"}: raise ValueError("requires exactly A and B")
        for spec in specs.values(): spec.assert_legal()
        allowed_candidate_keys = {"bbox_xyxy_pixels", "bbox_xyxy_normalized", "candidate_id", "candidate_index", "native_grounding_score", "native_label_text", "native_score_semantics", "postprocessing_metadata", "provenance", "query_hash", "raw_candidate_index", "siglip_diagnostic", "source_backend"}
        required_candidate_keys = {"bbox_xyxy_pixels", "candidate_id", "source_backend", "native_grounding_score", "siglip_diagnostic"}
        for ref, candidates in candidate_sets.items():
            ids = [candidate["candidate_id"] for candidate in candidates]
            if len(ids) != len(set(ids)): raise ValueError(f"duplicate candidate IDs for {ref}")
            for candidate in candidates:
                if not required_candidate_keys <= candidate.keys() or not candidate.keys() <= allowed_candidate_keys:
                    raise ValueError(f"invalid candidate schema for {ref}")
        assert_inference_safe(candidate_sets, context="candidate sets")
        assert_inference_safe(persistent_state, context="persistent state")
        relation = explicit_relation(predicate_dag, specs)
        distinct = distinct_required(predicate_dag)
        constraints = (["DISTINCT_EXACT_REGION"] if distinct else []) + ([relation] if relation else [])
        for ref in ("A", "B"):
            if persistent_state.get(ref, {}).get("region") is not None: constraints.append(f"PERSISTENT_IDENTITY_{ref}")
        ca, cb = candidate_sets["A"], candidate_sets["B"]
        hypotheses = []
        for a in ca:
            for b in cb:
                checks: dict[str, bool] = {}
                if distinct: checks["DISTINCT_EXACT_REGION"] = not same_region(a, b)
                ax, bx = center_x(a), center_x(b)
                if relation: checks[relation] = ax < bx if relation == "LEFT_OF(A,B)" else ax > bx
                for ref, cand in (("A", a), ("B", b)):
                    state = persistent_state.get(ref, {})
                    if state.get("region") is not None:
                        region = state["region"]
                        box = region.get("bbox_xyxy_pixels") if isinstance(region, dict) else region.bbox_xyxy_pixels
                        checks[f"PERSISTENT_IDENTITY_{ref}"] = tuple(cand["bbox_xyxy_pixels"]) == tuple(box)
                hypotheses.append({"candidate_ids": {"A": a["candidate_id"], "B": b["candidate_id"]},
                    "candidates": {ref: {"bbox_xyxy_pixels": c["bbox_xyxy_pixels"], "source_backend": c["source_backend"], "native_grounding_score": c["native_grounding_score"], "siglip_diagnostic_score": c["siglip_diagnostic"]["z_raw"]} for ref,c in (("A",a),("B",b))},
                    "geometry": {"center_x_A": ax, "center_x_B": bx, "computed_relation": "LEFT_OF(A,B)" if ax < bx else "RIGHT_OF(A,B)" if ax > bx else "TIE"},
                    "constraint_checks": checks, "valid": all(checks.values())})
        valid = [h for h in hypotheses if h["valid"]]
        status = "RESOLVED_UNIQUE" if len(valid)==1 else "ABSTAIN_NO_VALID_ASSIGNMENT" if not valid else "ABSTAIN_AMBIGUOUS"
        selected = valid[0]["candidate_ids"] if len(valid)==1 else None
        entities = {}
        if selected:
            for ref, candidates in (("A",ca),("B",cb)):
                cand = next(c for c in candidates if c["candidate_id"] == selected[ref])
                entities[ref] = {"logical_ref": ref, "entity_id": persistent_state.get(ref, {}).get("entity_id", f"E_{ref}"), "candidate_id": cand["candidate_id"], "region": cand["bbox_xyxy_pixels"], "source_backend": cand["source_backend"], "evidence_status": "RESOLVED_BUT_UNCERTIFIED", "resolver_version": VERSION, "constraints_used": constraints, "run_id": run_id}
        return ResolutionResult(family_id,status,len(hypotheses),len(valid),{r:digest(candidate_sets[r]) for r in ("A","B")},digest(predicate_dag.to_dict()),tuple(constraints),tuple(hypotheses),selected,entities)
