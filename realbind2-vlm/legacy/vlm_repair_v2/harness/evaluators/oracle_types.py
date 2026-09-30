"""Evaluator-only records with no automatic serialization path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluatorRecord:
    case_id: str
    ground_truth: Any
    gt_bbox: dict[str, Any]
    gt_category: dict[str, Any]
    gt_color: dict[str, Any]
    gt_relation: Any
    hidden_evaluator_ids: dict[str, Any]

    def to_evaluator_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ground_truth": self.ground_truth,
            "gt_bbox": self.gt_bbox,
            "gt_category": self.gt_category,
            "gt_color": self.gt_color,
            "gt_relation": self.gt_relation,
            "hidden_evaluator_ids": self.hidden_evaluator_ids,
            "namespace": "EVALUATOR_ONLY",
        }

    def __repr__(self) -> str:
        return f"EvaluatorRecord(case_id={self.case_id!r}, namespace='EVALUATOR_ONLY')"

