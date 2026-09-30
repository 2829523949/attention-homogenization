from __future__ import annotations

from .base import EntityVisualState, RegionPrediction, RelationState


def center(region: RegionPrediction) -> tuple[float, float]:
    x0, y0, x1, y1 = region.bbox_xyxy_normalized
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def horizontal_relation(source_id: str, source: RegionPrediction | None, target_id: str, target: RegionPrediction | None, epsilon: float = 1e-9) -> RelationState:
    if source is None or target is None:
        return RelationState(source_id, target_id, None, 0.0, "deterministic_geometry:missing_region")
    sx, _ = center(source)
    tx, _ = center(target)
    dx = sx - tx
    relation = "left" if dx < -epsilon else "right" if dx > epsilon else None
    confidence = min(1.0, abs(dx) * 2.0) if relation else 0.0
    return RelationState(source_id, target_id, relation, confidence, f"deterministic_geometry:dx={dx:.8f}")


def with_geometry(reference_id: str, region: RegionPrediction | None, *, category: str | None, category_confidence: float, color: str | None, color_confidence: float, provenance: tuple[str, ...]) -> EntityVisualState:
    xy = center(region) if region else (None, None)
    region_confidence = float(region.score) if region is not None and region.score is not None else 0.0
    return EntityVisualState(reference_id, region, region_confidence, category, category_confidence, color, color_confidence, xy[0], xy[1], provenance)


def derive_horizontal_fact(
    source_entity: str,
    source: RegionPrediction,
    target_entity: str,
    target: RegionPrediction,
    *,
    source_certificate_ids: tuple[str, ...],
    dag_node_id: str = "compare_horizontal",
    dependency_ids: tuple[str, ...] | None = None,
) -> "DerivedFact":
    """Return a deterministic relation; no visual model is called."""

    from harness.proof_state.compiler import DerivedFact

    sx, _ = center(source)
    tx, _ = center(target)
    if sx == tx:
        relation = "overlapping_center"
    else:
        relation = "left" if sx < tx else "right"
    return DerivedFact(
        f"derived_horizontal_{source_entity}_{target_entity}",
        "HORIZONTAL_RELATION",
        relation,
        (source_entity, target_entity),
        source_certificate_ids,
        "normalized_bbox_center_x_order_v1",
        ("PREVIOUS_ENTITY_STATE", "PREVIOUS_CERTIFIED_FACT"),
        dag_node_id,
        tuple(dependency_ids) if dependency_ids is not None else tuple(source_certificate_ids),
    )
