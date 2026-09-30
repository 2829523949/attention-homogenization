"""One frozen deterministic crop policy for fixed-region evidence scoring.

Policy `exact_persistent_bbox_clamped_v1`:
  - use the exact persistent bbox (no context padding);
  - round coordinates to nearest integer;
  - clamp to image bounds;
  - no per-view adjustment, no re-detection, no aspect-ratio change by
    the harness; SigLIP's own processor performs its documented resize.

The crop is a processor-local input.  It does not replace or alter the
counterfactual view artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from harness.processors.base import RegionPrediction


CROP_POLICY_ID = "exact_persistent_bbox_clamped_v1"


def deterministic_crop_box(image_size: tuple[int, int], bbox_xyxy_pixels: Any) -> tuple[int, int, int, int]:
    width, height = int(image_size[0]), int(image_size[1])
    x0, y0, x1, y1 = (int(round(float(value))) for value in bbox_xyxy_pixels)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop after clamping: {(x0, y0, x1, y1)} for image {image_size}")
    return x0, y0, x1, y1


def crop_persistent_region(image_path: Path, region: RegionPrediction):
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        box = deterministic_crop_box(image.size, region.bbox_xyxy_pixels)
        return image.crop(box)


def save_persistent_crop(image_path: Path, region: RegionPrediction, output_path: Path) -> dict[str, Any]:
    """Materialize the exact crop used by the scorer, for visual audit only."""
    from PIL import Image

    image = crop_persistent_region(image_path, region)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite crop artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", compress_level=6)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    with Image.open(image_path) as opened:
        box = deterministic_crop_box(opened.size, region.bbox_xyxy_pixels)
    return {
        "crop_policy_id": CROP_POLICY_ID,
        "path": str(output_path),
        "output_image_sha256": digest,
        "crop_box_xyxy_pixels": list(box),
        "crop_size": [int(image.width), int(image.height)],
        "source_image": str(image_path),
    }
