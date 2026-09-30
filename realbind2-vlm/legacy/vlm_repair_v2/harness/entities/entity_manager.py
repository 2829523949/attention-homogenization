"""Persistent logical-reference to entity binding with re-grounding guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.processors.base import RegionPrediction
from harness.referents.referent_spec import GroundingQuery

from .entity_state import EntityState


@dataclass(frozen=True)
class EntityManagerEvent:
    event: str
    logical_ref: str
    entity_id: str
    reason: str | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "logical_ref": self.logical_ref,
            "entity_id": self.entity_id,
            "reason": self.reason,
            "source": self.source,
        }


class EntityManager:
    def __init__(self) -> None:
        self._by_ref: dict[str, EntityState] = {}
        self._next_id = 0
        self.events: list[EntityManagerEvent] = []

    def ensure_entity(self, logical_ref: str, query: GroundingQuery) -> EntityState:
        existing = self._by_ref.get(logical_ref)
        if existing is not None:
            if existing.grounding_query and existing.grounding_query.semantic_text != query.semantic_text:
                raise RuntimeError(f"logical reference {logical_ref} already has a different legal query")
            self.events.append(EntityManagerEvent("REUSE_ENTITY", logical_ref, existing.entity_id, None, "persistent_entity_manager"))
            return existing
        entity_id = f"E_{self._next_id}"
        self._next_id += 1
        entity = EntityState(entity_id, (logical_ref,), grounding_query=query, provenance=(f"logical_ref:{logical_ref}", "persistent_entity_manager"))
        self._by_ref[logical_ref] = entity
        self.events.append(EntityManagerEvent("CREATE_ENTITY", logical_ref, entity_id, None, "persistent_entity_manager"))
        return entity

    def get(self, logical_ref: str) -> EntityState:
        try:
            return self._by_ref[logical_ref]
        except KeyError as exc:
            raise KeyError(f"unknown logical reference: {logical_ref}") from exc

    def reuse(self, logical_ref: str, *, reason: str) -> EntityState:
        entity = self.get(logical_ref)
        self.events.append(EntityManagerEvent("REUSE_ENTITY", logical_ref, entity.entity_id, reason, "persistent_entity_manager"))
        return entity

    def all(self) -> tuple[EntityState, ...]:
        return tuple(sorted(self._by_ref.values(), key=lambda item: item.entity_id))

    def bind_region(self, logical_ref: str, region: RegionPrediction, *, reason: str | None = None, source: str = "specialist_grounding") -> EntityState:
        entity = self.get(logical_ref)
        if not isinstance(region, RegionPrediction):
            raise TypeError("entity binding accepts RegionPrediction only")
        if entity.region is not None and reason is None:
            raise RuntimeError(f"silent re-grounding prohibited for {entity.entity_id}; supply controlled re-resolution reason")
        if entity.region is not None:
            if not str(reason or "").strip():
                raise RuntimeError(f"controlled re-resolution of {entity.entity_id} requires a non-empty reason")
            self.events.append(EntityManagerEvent("CONTROLLED_RERESOLUTION", logical_ref, entity.entity_id, reason, source))
            # Stale observations/certificates refer to the old region. They
            # must not survive a controlled re-resolution.
            observations: tuple[Any, ...] = ()
            certificate_ids: tuple[str, ...] = ()
        else:
            self.events.append(EntityManagerEvent("BIND_REGION", logical_ref, entity.entity_id, None, source))
            observations = entity.observations
            certificate_ids = entity.certificate_ids
        updated = EntityState(
            entity.entity_id,
            entity.logical_refs,
            region,
            region.mask_path,
            entity.grounding_query,
            region.score,
            "RESOLVED_UNCERTIFIED",
            observations,
            certificate_ids,
            entity.provenance + (f"region:{source}",),
        )
        self._by_ref[logical_ref] = updated
        return updated

    def append_observation(self, logical_ref: str, observation: Any) -> EntityState:
        entity = self.get(logical_ref)
        updated = EntityState(
            entity.entity_id,
            entity.logical_refs,
            entity.region,
            entity.mask_path,
            entity.grounding_query,
            entity.grounding_score,
            entity.certification_status,
            entity.observations + (observation,),
            entity.certificate_ids,
            entity.provenance,
        )
        self._by_ref[logical_ref] = updated
        return updated

    def attach_certificate(self, logical_ref: str, certificate_id: str, status: str) -> EntityState:
        entity = self.get(logical_ref)
        updated = EntityState(
            entity.entity_id,
            entity.logical_refs,
            entity.region,
            entity.mask_path,
            entity.grounding_query,
            entity.grounding_score,
            status,
            entity.observations,
            entity.certificate_ids + (certificate_id,),
            entity.provenance,
        )
        self._by_ref[logical_ref] = updated
        return updated
