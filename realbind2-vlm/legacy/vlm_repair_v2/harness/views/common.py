"""Shared, provenance-safe helpers for counterfactual view construction."""

from __future__ import annotations

from typing import Any, Sequence

from harness.processors.base import RegionPrediction


def reject_oracle(region: Any, *, view_name: str) -> None:
    if not isinstance(region, RegionPrediction):
        raise TypeError(f"{view_name} requires a predicted RegionPrediction")
    source = str(getattr(region, "source", "")).casefold()
    if "gt" in source or "oracle" in source or "evaluator" in source:
        raise AssertionError(f"automatic {view_name} accepts predicted regions only")


def expanded_box_xyxy(region: RegionPrediction, size: tuple[int, int], expand_frac: float) -> tuple[float, float, float, float]:
    """Expand the acquired persistent entity region by a fixed fraction."""
    width, height = float(size[0]), float(size[1])
    x0, y0, x1, y1 = (float(value) for value in region.bbox_xyxy_pixels)
    dx, dy = (x1 - x0) * float(expand_frac), (y1 - y0) * float(expand_frac)
    return (
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(width, x1 + dx),
        min(height, y1 + dy),
    )


def region_to_dict(region: RegionPrediction) -> dict[str, Any]:
    payload = region.__dict__.copy()
    for key in ("bbox_xyxy_pixels", "bbox_xyxy_normalized"):
        if isinstance(payload.get(key), tuple):
            payload[key] = list(payload[key])
    return payload
