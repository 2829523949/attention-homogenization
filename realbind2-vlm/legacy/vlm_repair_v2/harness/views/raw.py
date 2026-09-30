"""RAW view wrapper with immutable image provenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ViewArtifact:
    view_id: str
    path: str
    method: str
    source_entity: str | None
    predicted_region: dict[str, Any] | None
    source_image_sha256: str
    output_image_sha256: str
    parameters: dict[str, Any]
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "path": self.path,
            "method": self.method,
            "source_entity": self.source_entity,
            "predicted_region": self.predicted_region,
            "source_image_sha256": self.source_image_sha256,
            "output_image_sha256": self.output_image_sha256,
            "parameters": self.parameters,
            "provenance": list(self.provenance),
        }


def raw_view(image_path: Path, *, source_entity: str | None = None, predicted_region: Any = None) -> ViewArtifact:
    digest = image_sha256(image_path)
    region = None
    if predicted_region is not None:
        region = predicted_region.__dict__.copy()
        for key in ("bbox_xyxy_pixels", "bbox_xyxy_normalized"):
            if isinstance(region.get(key), tuple):
                region[key] = list(region[key])
    return ViewArtifact("RAW", str(image_path), "unaltered_input", source_entity, region, digest, digest, {}, ("RAW_IMAGE", "PREDICTED_REGION" if predicted_region is not None else "IMAGE_INPUT"))

