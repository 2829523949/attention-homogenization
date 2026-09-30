from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray")
COLOR_ALIASES = {"grey": "gray"}
MIN_AREA_RATIO = 0.02
MIN_DELTA_X = 0.20
MAX_IOU = 0.10


def normalize_color_labels(values: Iterable[object]) -> tuple[str | None, str | None]:
    found = set()
    for value in values:
        label = re.sub(r"\s+", " ", str(value).strip().lower())
        label = COLOR_ALIASES.get(label, label)
        if label in COLORS:
            found.add(label)
    ordered = [c for c in COLORS if c in found]
    if len(ordered) == 1:
        return ordered[0], None
    return None, "ATTRIBUTE_MISSING" if not ordered else "ATTRIBUTE_CONFLICT"


def bbox_valid(bbox: Sequence[float], width: int, height: int) -> bool:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return False
    x, y, w, h = map(float, bbox)
    return all(math.isfinite(v) for v in (x, y, w, h)) and x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def geometry_gate(a: Sequence[float], b: Sequence[float], width: int, height: int) -> tuple[dict, list[str]]:
    reasons: list[str] = []
    if not bbox_valid(a, width, height) or not bbox_valid(b, width, height):
        return {"area_ratio_A": None, "area_ratio_B": None, "delta_x": None, "iou": None}, ["INVALID_BBOX"]
    aa, ba = float(a[2]) * float(a[3]), float(b[2]) * float(b[3])
    ar_a, ar_b = aa / (width * height), ba / (width * height)
    dx = abs((float(a[0]) + float(a[2]) / 2) - (float(b[0]) + float(b[2]) / 2)) / width
    iou = bbox_iou(a, b)
    if ar_a < MIN_AREA_RATIO or ar_b < MIN_AREA_RATIO:
        reasons.append("OBJECT_TOO_SMALL")
    if dx < MIN_DELTA_X:
        reasons.append("AMBIGUOUS_GEOMETRY")
    if iou > MAX_IOU:
        reasons.append("HIGH_BBOX_OVERLAP")
    return {"area_ratio_A": ar_a, "area_ratio_B": ar_b, "delta_x": dx, "iou": iou, "left_is_A": float(a[0]) + float(a[2]) / 2 < float(b[0]) + float(b[2]) / 2}, reasons
