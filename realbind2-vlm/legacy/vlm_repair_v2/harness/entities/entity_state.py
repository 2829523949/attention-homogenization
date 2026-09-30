"""Persistent entity state shared by all specialist acquisitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.evaluators.leakage_checks import assert_inference_safe
from harness.referents.referent_spec import GroundingQuery


@dataclass(frozen=True)
class EntityState:
    entity_id: str
    logical_refs: tuple[str, ...]
    region: Any = None
    mask_path: str | None = None
    grounding_query: GroundingQuery | None = None
    grounding_score: float | None = None
    certification_status: str = "UNRESOLVED"
    observations: tuple[Any, ...] = ()
    certificate_ids: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        region = None
        if self.region is not None:
            region = self.region.__dict__.copy() if hasattr(self.region, "__dict__") else self.region
            if isinstance(region, dict):
                for key in ("bbox_xyxy_pixels", "bbox_xyxy_normalized"):
                    if key in region and isinstance(region[key], tuple):
                        region[key] = list(region[key])
        payload = {
            "entity_id": self.entity_id,
            "logical_refs": list(self.logical_refs),
            "region": region,
            "mask_path": self.mask_path,
            "grounding_query": self.grounding_query.to_dict() if self.grounding_query else None,
            "grounding_score": self.grounding_score,
            "certification_status": self.certification_status,
            "observations": [item.to_dict() if hasattr(item, "to_dict") else item for item in self.observations],
            "certificate_ids": list(self.certificate_ids),
            "provenance": list(self.provenance),
        }
        assert_inference_safe(payload, context="EntityState")
        return payload

