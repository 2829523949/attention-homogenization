"""Target-preserving FOCUS view using the historical defocus composite.

FOCUS(E_i):
  - support  = automatically acquired persistent entity region, expanded by
               the configured fraction;
  - inside   = original pixels preserved (feathered boundary);
  - outside  = Gaussian-blurred background;
  - no crop/resize surrogate and no GT/oracle input.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from harness.processors.base import RegionPrediction

from .common import expanded_box_xyxy, region_to_dict, reject_oracle
from .raw import ViewArtifact, image_sha256


def focus_view(image_path: Path, region: RegionPrediction, output_path: Path, *, entity_id: str, config: dict[str, float]) -> ViewArtifact:
    reject_oracle(region, view_name="FOCUS")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    expand = float(config.get("bbox_expand_frac", 0.10))
    box = expanded_box_xyxy(region, image.size, expand)

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    feather = float(config.get("feather_radius_fraction_of_min_dimension", 0.010)) * min(image.size)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    blur_radius = float(config.get("blur_radius_fraction_of_min_dimension", 0.025)) * min(image.size)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    result = Image.composite(image, blurred, mask).convert("RGB")

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite focus view: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG", compress_level=6)
    return ViewArtifact(
        "FOCUS",
        str(output_path),
        "target_preserving_gaussian_defocus_outside_predicted_region_v2",
        entity_id,
        region_to_dict(region),
        image_sha256(image_path),
        image_sha256(output_path),
        {
            "expanded_box_xyxy": list(box),
            "blur_radius": blur_radius,
            "feather_radius": feather,
            "target_support": "preserve_original_with_feather",
            "outside_support": "gaussian_blur",
            "region_source": str(region.source),
            **config,
        },
        ("PREDICTED_REGION", "AUTO_VIEW", "TARGET_PRESERVING_DEFOCUS"),
    )
