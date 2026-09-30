"""Requirement-driven routing from PredicateDAG obligations to primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.entities.entity_manager import EntityManager
from harness.planner.predicate_dag import Operation, PredicateDAG
from harness.referents.referent_spec import GroundingQuery, ReferentSpec

from .observations import AcquisitionTrace, Observation


class AcquisitionRouter:
    def __init__(self, registry: Any, entity_manager: EntityManager):
        self.registry = registry
        self.entity_manager = entity_manager
        self.trace: list[AcquisitionTrace] = []

    def acquire_regions(self, image_path: Path, dag: PredicateDAG, specs: tuple[ReferentSpec, ...], *, grounding_processor: str = "grounding") -> tuple[Observation, ...]:
        registered = self.registry.require(grounding_processor, "grounding")
        observations: list[Observation] = []
        spec_by_ref = {spec.logical_ref: spec for spec in specs}
        for node in dag.topological_order():
            if node.operation is not Operation.LOCALIZE:
                continue
            logical_ref = node.required_inputs[0]
            spec = spec_by_ref[logical_ref]
            query = GroundingQuery.from_spec(spec)
            entity = self.entity_manager.ensure_entity(logical_ref, query)
            candidate_set = None
            if hasattr(registered.implementation, "acquire"):
                candidate_set = registered.implementation.acquire(image_path, spec)
                if candidate_set.logical_ref != logical_ref:
                    raise RuntimeError("candidate generator changed logical reference identity")
                candidates = list(candidate_set.regions())
            else:
                # Compatibility for older specialist adapters.  New backends
                # implement CandidateGenerator.acquire and return CandidateSet.
                candidates = list(registered.implementation.ground(image_path, query))
            if not candidates:
                self.trace.append(
                    AcquisitionTrace(
                        len(self.trace),
                        "LOCALIZE",
                        node.node_id,
                        entity.entity_id,
                        registered.name,
                        {"query": query.to_dict()},
                        {
                            "candidate_count": 0,
                            "candidates": [],
                            "selected_candidate_index": None,
                            "candidate_pipeline_trace": getattr(
                                registered.implementation, "last_candidate_trace", None
                            ),
                            "candidate_set": candidate_set.to_dict() if candidate_set is not None else None,
                        },
                        ("QUESTION_TEXT",),
                    )
                )
                continue
            def _selection_key(item: Any) -> tuple[float, float]:
                raw_score = getattr(item, "score", None)
                score_key = float(raw_score) if raw_score is not None else float("-inf")
                return (score_key, -float(getattr(item, "bbox_xyxy_pixels", (0.0,))[0]))

            region = max(candidates, key=_selection_key)
            selected_candidate_index = next(
                index for index, candidate in enumerate(candidates) if candidate is region
            )
            self.entity_manager.bind_region(logical_ref, region)
            diagnostics = {
                "candidate_count": len(candidates),
                "native_score_available": bool(getattr(region, "score", None) is not None),
                "selection_key": "max(native_score; absent scores sort behind present scores, then smaller x0)",
            }
            if getattr(region, "score", None) is None:
                from harness.certification.evidence import EvidenceScore

                score = EvidenceScore.unsupported(
                    registered.name,
                    reason=(
                        "CERTIFICATION_SCORE_UNSUPPORTED: acquisition output exposes no numeric "
                        "region confidence; the region may be used for acquisition only."
                    ),
                    claimed_value=region,
                    diagnostics=diagnostics,
                    view_id="RAW",
                )
            else:
                from harness.certification.normalizers import normalize_confidence

                score = normalize_confidence(
                    registered.name,
                    region.score,
                    claimed_value=region,
                    diagnostics=diagnostics,
                    view_id="RAW",
                )
            observation = Observation(entity.entity_id, "REFERENT_TO_REGION", region, score, registered.name, ("QUESTION_TEXT",), "RAW", diagnostics)
            observations.append(observation)
            self.entity_manager.append_observation(logical_ref, observation)
            self.trace.append(
                AcquisitionTrace(
                    len(self.trace),
                    "LOCALIZE",
                    node.node_id,
                    entity.entity_id,
                    registered.name,
                    {"query": query.to_dict()},
                    {
                        "candidate_count": len(candidates),
                        "candidates": [candidate.__dict__ for candidate in candidates],
                        "selected_candidate_index": selected_candidate_index,
                        "selected_region": region.__dict__,
                        "candidate_pipeline_trace": getattr(
                            registered.implementation, "last_candidate_trace", None
                        ),
                        "candidate_set": candidate_set.to_dict() if candidate_set is not None else None,
                    },
                    ("QUESTION_TEXT", "TOOL_OBSERVATION"),
                )
            )
        return tuple(observations)

    def acquire_color(self, image_path: Path, logical_ref: str, *, color_processor: str = "color") -> Observation:
        entity = self.entity_manager.reuse(logical_ref, reason="color_reader_reuses_resolved_entity")
        if entity.region is None:
            raise RuntimeError(f"cannot read color before entity region exists: {logical_ref}")
        registered = self.registry.require(color_processor, "color")
        observation = registered.implementation.observe(image_path, entity.entity_id, entity.region, view_id="RAW")
        self.entity_manager.append_observation(logical_ref, observation)
        self.trace.append(AcquisitionTrace(len(self.trace), "READ_COLOR", "read_color", entity.entity_id, registered.name, {"view": "RAW"}, {"observation": observation.to_dict()}, ("PREVIOUS_ENTITY_STATE", "TOOL_OBSERVATION")))
        return observation
