"""Target-suppressing REMOVE view.

REMOVE(E_i):
  - support = the *same* automatically acquired persistent entity region and
              the same expansion fraction used by FOCUS;
  - inside  = neutral mid-gray fill (target pixels are erased, including
              target colour and texture);
  - outside = original pixels preserved exactly (hard binary support mask);
  - no Gaussian background defocus, no crop/resize surrogate, no GT input.

This is a deterministic ablation intervention, not a generative inpainting
model.  It deliberately removes target evidence rather than preserving it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from harness.processors.base import RegionPrediction

from .common import expanded_box_xyxy, region_to_dict, reject_oracle
from .raw import ViewArtifact, image_sha256


def remove_view(image_path: Path, region: RegionPrediction, output_path: Path, *, entity_id: str, config: dict[str, Any]) -> ViewArtifact:
    reject_oracle(region, view_name="REMOVE")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    expand = float(config.get("bbox_expand_frac", 0.10))
    box = expanded_box_xyxy(region, image.size, expand)
    fill_rgb = tuple(int(value) for value in config.get("remove_fill_rgb", (127, 127, 127)))
    if len(fill_rgb) != 3 or any(value < 0 or value > 255 for value in fill_rgb):
        raise ValueError(f"invalid REMOVE fill colour: {fill_rgb!r}")

    neutral = Image.new("RGB", image.size, fill_rgb)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    result = Image.composite(neutral, image, mask).convert("RGB")

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite remove view: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG", compress_level=6)
    return ViewArtifact(
        "REMOVE",
        str(output_path),
        "target_region_neutral_fill_ablation_v2",
        entity_id,
        region_to_dict(region),
        image_sha256(image_path),
        image_sha256(output_path),
        {
            "suppression_box_xyxy": list(box),
            "suppression_mask": "hard_binary_expanded_bbox",
            "fill_rgb": list(fill_rgb),
            "inside_operation": "neutral_fill",
            "outside_operation": "preserve_original",
            "region_source": str(region.source),
            **{key: value for key, value in config.items() if key != "remove_fill_rgb"},
        },
        ("PREDICTED_REGION", "AUTO_VIEW", "TARGET_SUPPRESSION_NEUTRAL_FILL"),
    )
