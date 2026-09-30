"""Frozen historical RealBind pair-defocus transform.

Recovered byte-for-byte in semantics from
``experiments/realbind_phase2/qwen35_dah_oracle_eval.py`` and its certified
``qwen35_dah_oracle_v1`` artifacts.  Bounding boxes are ``[x, y, w, h]`` at
the original image resolution.  This module performs no target selection.
Callers must declare whether boxes are oracle/evaluator records or runtime
predictions; the historical DaH/VIR evidence used oracle family boxes.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter

DEFOCUS_CONFIG: dict[str, object] = {
    "bbox_expand_frac": 0.10,
    "blur_radius_fraction_of_min_dimension": 0.025,
    "feather_radius_fraction_of_min_dimension": 0.010,
    "blur_radius_unit": "pixels after reading original image resolution",
    "mask": "soft grayscale bbox masks; union by ImageChops.lighter; Gaussian feather",
    "output_format": "PNG",
    "png_optimize": False,
    "png_compress_level": 6,
    "resize": "none in transform; model processor resize occurs later",
    "crop": False,
    "draw_annotations": False,
    "target_support": "expanded A union B bboxes",
    "inside_operation": "preserve original pixels with feathered boundary",
    "outside_operation": "Pillow GaussianBlur",
    "deterministic": True,
}


def expanded_bbox(
    bbox_xywh: Sequence[float], width: int, height: int, expand_frac: float
) -> list[float]:
    x, y, w, h = [float(value) for value in bbox_xywh]
    dx = w * float(expand_frac)
    dy = h * float(expand_frac)
    return [
        max(0.0, x - dx),
        max(0.0, y - dy),
        min(float(width), x + w + dx),
        min(float(height), y + h + dy),
    ]


def _bbox_mask(
    size: tuple[int, int], bbox_xyxy: Sequence[float], config: Mapping[str, object]
) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(tuple(float(value) for value in bbox_xyxy), fill=255)
    feather = float(config["feather_radius_fraction_of_min_dimension"]) * min(size)
    return mask.filter(ImageFilter.GaussianBlur(radius=feather))


def render_pair_defocus(
    image: Image.Image,
    bbox_a_xywh: Sequence[float],
    bbox_b_xywh: Sequence[float],
    *,
    config: Mapping[str, object] | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    """Keep expanded A∪B clear and Gaussian-blur its complement."""

    cfg = dict(DEFOCUS_CONFIG if config is None else config)
    source = image.convert("RGB")
    expanded_a = expanded_bbox(
        bbox_a_xywh, source.width, source.height, float(cfg["bbox_expand_frac"])
    )
    expanded_b = expanded_bbox(
        bbox_b_xywh, source.width, source.height, float(cfg["bbox_expand_frac"])
    )
    mask_a = _bbox_mask(source.size, expanded_a, cfg)
    mask_b = _bbox_mask(source.size, expanded_b, cfg)
    union = ImageChops.lighter(mask_a, mask_b)
    radius = float(cfg["blur_radius_fraction_of_min_dimension"]) * min(source.size)
    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
    output = Image.composite(source, blurred, union).convert("RGB")
    return output, {
        "mask": "A_union_B",
        "bbox_A_xywh": [float(value) for value in bbox_a_xywh],
        "bbox_B_xywh": [float(value) for value in bbox_b_xywh],
        "expanded_bbox_A_xyxy": expanded_a,
        "expanded_bbox_B_xyxy": expanded_b,
        "blur_radius": radius,
        "feather_radius": float(cfg["feather_radius_fraction_of_min_dimension"])
        * min(source.size),
        "size": [source.width, source.height],
        "config": cfg,
    }


def png_bytes(image: Image.Image) -> bytes:
    """Encode using the exact historical artifact settings."""

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def save_pair_defocus(
    source_path: Path,
    output_path: Path,
    bbox_a_xywh: Sequence[float],
    bbox_b_xywh: Sequence[float],
    *,
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create one immutable PNG; an unequal existing file is an error."""

    with Image.open(source_path) as opened:
        output, metadata = render_pair_defocus(
            opened, bbox_a_xywh, bbox_b_xywh, config=config
        )
    payload = png_bytes(output)
    output.close()
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise FileExistsError(f"existing defocus output differs: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
    metadata["output_path"] = str(output_path)
    metadata["output_sha256"] = hashlib.sha256(payload).hexdigest()
    return metadata

