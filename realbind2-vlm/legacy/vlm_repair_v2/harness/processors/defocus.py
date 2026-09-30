from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .base import RegionPrediction


DEFOCUS_CONFIG = {"bbox_expand_frac": .10, "blur_radius_fraction_of_min_dimension": .025, "feather_radius_fraction_of_min_dimension": .010}


def _expanded(box: Sequence[float], width: int, height: int, fraction: float) -> list[float]:
    x0, y0, x1, y1 = map(float, box)
    dx, dy = (x1 - x0) * fraction, (y1 - y0) * fraction
    return [max(0., x0 - dx), max(0., y0 - dy), min(float(width), x1 + dx), min(float(height), y1 + dy)]


def _box_mask(size: tuple[int, int], box: Sequence[float], config: dict[str, float]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(tuple(box), fill=255)
    feather = config["feather_radius_fraction_of_min_dimension"] * min(size)
    return mask.filter(ImageFilter.GaussianBlur(radius=feather))


def _render_defocus(image_path: Path, boxes_xyxy: Sequence[Sequence[float]], output_path: Path, config: dict[str, float], mask_paths: Sequence[str | None] | None = None) -> dict:
    if not boxes_xyxy: raise ValueError("defocus requires at least one region")
    cfg = dict(config)
    image = Image.open(image_path).convert("RGB")
    masks = []
    boxes = []
    mask_paths = list(mask_paths or [None] * len(boxes_xyxy))
    mask_sources=[]
    for raw_box,mask_path in zip(boxes_xyxy,mask_paths):
        box = _expanded(raw_box, image.width, image.height, cfg["bbox_expand_frac"])
        boxes.append(box)
        if mask_path:
            predicted=Image.open(mask_path).convert("L")
            if predicted.size != image.size: raise ValueError(f"predicted mask/image size mismatch: {predicted.size} vs {image.size}")
            feather=cfg["feather_radius_fraction_of_min_dimension"]*min(image.size);masks.append(predicted.filter(ImageFilter.GaussianBlur(radius=feather)));mask_sources.append(str(mask_path))
        else:
            masks.append(_box_mask(image.size, box, cfg));mask_sources.append(None)
    union = masks[0]
    for mask in masks[1:]: union = ImageChops.lighter(union, mask)
    radius = cfg["blur_radius_fraction_of_min_dimension"] * min(image.size)
    result = Image.composite(image, image.filter(ImageFilter.GaussianBlur(radius=radius)), union).convert("RGB")
    if output_path.exists(): raise FileExistsError(f"refusing to overwrite output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True); result.save(output_path, format="PNG", compress_level=6)
    return {"expanded_boxes_xyxy": boxes, "predicted_mask_paths":mask_sources,"blur_radius": radius, "config": cfg, "output_path": str(output_path)}


def auto_defocus(image_path: Path, predicted_regions: Sequence[RegionPrediction], output_path: Path, *, config: dict[str, float] | None = None) -> dict:
    """Automatic-only API: accepts typed predictions and rejects oracle provenance."""
    if not predicted_regions: raise ValueError("Auto-Defocus requires at least one predicted region")
    if not all(isinstance(region, RegionPrediction) for region in predicted_regions): raise TypeError("Auto-Defocus accepts RegionPrediction only; evaluator GT records are forbidden")
    if any("oracle" in region.source.casefold() or "gt" in region.source.casefold() for region in predicted_regions): raise AssertionError("oracle/GT provenance forbidden in Auto-Defocus")
    cfg = dict(DEFOCUS_CONFIG if config is None else config)
    meta = _render_defocus(image_path, [r.bbox_xyxy_pixels for r in predicted_regions], output_path, cfg, [r.mask_path for r in predicted_regions])
    return {"mode": "auto", "source": "predicted_regions", **meta}


def oracle_defocus(image_path: Path, gt_boxes_xywh: Sequence[Sequence[float]], output_path: Path, *, config: dict[str, float] | None = None) -> dict:
    """Evaluator-only diagnostic; deliberately separate from auto_defocus."""
    boxes = []
    for box in gt_boxes_xywh:
        x, y, w, h = map(float, box)
        boxes.append([x, y, x + w, y + h])
    meta = _render_defocus(image_path, boxes, output_path, dict(DEFOCUS_CONFIG if config is None else config))
    return {"mode": "oracle", "source": "evaluator_gt_bbox", "diagnostic_only": True, **meta}
