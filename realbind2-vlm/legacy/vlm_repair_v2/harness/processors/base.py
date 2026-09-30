from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FORBIDDEN_INFERENCE_KEYS = frozenset({"gt_bbox", "ground_truth_bbox", "bbox_gt", "oracle_bbox"})


def _assert_no_gt(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_INFERENCE_KEYS or str(key).casefold().startswith("gt_bbox"):
                raise AssertionError(f"GT localization leaked into inference state at {path}.{key}")
            _assert_no_gt(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_gt(item, f"{path}[{index}]")


@dataclass(frozen=True)
class RegionPrediction:
    bbox_xyxy_pixels: tuple[float, float, float, float]
    bbox_xyxy_normalized: tuple[float, float, float, float]
    # None means the model did not expose a numeric confidence for this
    # region. Callers must not replace that absence with 0.0/1.0 and then
    # present the result as model evidence.
    score: float | None
    label: str
    source: str
    mask_path: str | None = None

    @property
    def score_available(self) -> bool:
        return self.score is not None

    @classmethod
    def from_pixels(cls, bbox: list[float] | tuple[float, ...], width: int, height: int, *, score: float | None, label: str, source: str, mask_path: str | None = None) -> "RegionPrediction":
        if len(bbox) != 4 or width <= 0 or height <= 0:
            raise ValueError("invalid bbox/image dimensions")
        x0, y0, x1, y1 = (float(x) for x in bbox)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"bbox outside image: {bbox} vs {width}x{height}")
        return cls((x0, y0, x1, y1), (x0 / width, y0 / height, x1 / width, y1 / height), None if score is None else float(score), str(label), str(source), mask_path)


@dataclass(frozen=True)
class ColorPrediction:
    color: str | None
    confidence: float
    source: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityVisualState:
    reference_id: str
    predicted_region: RegionPrediction | None
    region_confidence: float
    category: str | None
    category_confidence: float
    color: str | None
    color_confidence: float
    center_x: float | None
    center_y: float | None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationState:
    source_entity: str
    target_entity: str
    relation: str | None
    confidence: float
    provenance: str


@dataclass(frozen=True)
class VisualState:
    case_id: str
    requested_task: str
    entities: tuple[EntityVisualState, ...]
    relations: tuple[RelationState, ...]
    acquisition_trace: tuple[dict[str, Any], ...]
    uncertainty: dict[str, Any]
    processor_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        _assert_no_gt(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VisualState":
        _assert_no_gt(payload)
        entities = []
        for entity in payload["entities"]:
            region = entity.get("predicted_region")
            if region is not None:
                region = RegionPrediction(**{**region, "bbox_xyxy_pixels": tuple(region["bbox_xyxy_pixels"]), "bbox_xyxy_normalized": tuple(region["bbox_xyxy_normalized"])})
            entities.append(EntityVisualState(**{**entity, "predicted_region": region, "provenance": tuple(entity.get("provenance", ())) }))
        relations = tuple(RelationState(**row) for row in payload.get("relations", []))
        return cls(str(payload["case_id"]), str(payload["requested_task"]), tuple(entities), relations, tuple(payload.get("acquisition_trace", ())), dict(payload.get("uncertainty", {})), dict(payload.get("processor_versions", {})))


class RegionProcessor:
    name = "abstract"
    version = "0"

    def predict(self, image_path: Path, **kwargs: Any) -> list[RegionPrediction]:
        raise NotImplementedError
