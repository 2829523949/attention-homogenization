from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.processors.base import VisualState


@dataclass(frozen=True)
class GroundTruthRecord:
    case_id: str
    bbox_A_xywh: tuple[float, float, float, float]
    bbox_B_xywh: tuple[float, float, float, float]
    color_A: str
    color_B: str
    left_is_A: bool


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    intersection = max(0., min(ax1, bx1) - max(ax0, bx0)) * max(0., min(ay1, by1) - max(ay0, by0))
    union = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - intersection
    return intersection / union if union > 0 else 0.


def evaluate_state(state: VisualState, gt: GroundTruthRecord) -> dict[str, Any]:
    entities = {row.reference_id: row for row in state.entities}
    per_entity = {}
    for ref in ("A", "B"):
        entity = entities.get(ref); gt_box = xywh_to_xyxy(getattr(gt, f"bbox_{ref}_xywh"))
        overlap = iou(entity.predicted_region.bbox_xyxy_pixels, gt_box) if entity and entity.predicted_region else 0.
        expected_color = getattr(gt, f"color_{ref}")
        per_entity[ref] = {"iou": overlap, "localized_iou_0_5": overlap >= .5, "color_correct_auto_region": bool(entity and entity.color == expected_color), "covered": bool(entity and entity.predicted_region)}
    relation = next((r for r in state.relations if r.source_entity == "A" and r.target_entity == "B"), None)
    expected_relation = "left" if gt.left_is_A else "right"
    required_exact = all(per_entity[x]["color_correct_auto_region"] for x in ("A", "B")) and bool(relation and relation.relation == expected_relation)
    return {"case_id": state.case_id, "entities": per_entity, "mean_iou": sum(x["iou"] for x in per_entity.values()) / 2, "localization_recall_iou_0_5": sum(x["localized_iou_0_5"] for x in per_entity.values()) / 2, "entity_mapping_accuracy": sum(x["localized_iou_0_5"] for x in per_entity.values()) / 2, "color_accuracy_automatic_region": sum(x["color_correct_auto_region"] for x in per_entity.values()) / 2, "relation_accuracy_automatic_regions": bool(relation and relation.relation == expected_relation), "required_fact_exact_match": required_exact, "coverage": sum(x["covered"] for x in per_entity.values()) / 2, "abstention_rate": sum(not x["covered"] for x in per_entity.values()) / 2}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("mean_iou", "localization_recall_iou_0_5", "entity_mapping_accuracy", "color_accuracy_gt_region", "color_accuracy_automatic_region", "relation_accuracy_automatic_regions", "required_fact_exact_match", "coverage", "abstention_rate")
    return {"N": len(rows), **{field: sum(float(row[field]) for row in rows) / len(rows) if rows else None for field in fields}}
