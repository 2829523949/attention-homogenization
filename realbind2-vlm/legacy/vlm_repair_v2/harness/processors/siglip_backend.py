"""Shared local SigLIP-2 inference backend.

Both the color reader and any fixed-region query scorer must use the same
loaded checkpoint, processor, device, and text-template transformation so that
scores are comparable across counterfactual views.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


class Siglip2Backend:
    def __init__(self, model_path: Path, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.model_path = Path(model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True).to(self.device).eval()
        self.version = "google_siglip2_base_patch16_224_local_v1"
        # Filename/size/config fingerprint of the local checkpoint tree.  This is
        # not a cryptographic hash of the weight tensor bytes.
        self.model_fingerprint = _directory_fingerprint(self.model_path)

    def _tensor_inputs(self, image: Any, texts: Sequence[str]) -> dict[str, Any]:
        inputs = self.processor(text=list(texts), images=image, padding="max_length", return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def image_text_logits(self, image: Any, texts: Sequence[str]) -> list[float]:
        """Return one real image-text logit per supplied text template."""
        import torch

        if not texts:
            raise ValueError("SigLIP scoring requires at least one text template")
        inputs = self._tensor_inputs(image, texts)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0].float().cpu().tolist()
        return [float(value) for value in logits]

    def logits_for_crop(self, image_path: Path, bbox_xyxy_pixels: Sequence[float], texts: Sequence[str]) -> list[float]:
        from PIL import Image

        from harness.processors.crop_policy import deterministic_crop_box

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            crop = image.crop(deterministic_crop_box(image.size, bbox_xyxy_pixels))
        return self.image_text_logits(crop, texts)

    def logits_for_region(self, image_path: Path, region: "RegionPrediction", texts: Sequence[str]) -> list[float]:
        return self.logits_for_crop(image_path, region.bbox_xyxy_pixels, texts)


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        if item.name in {"config.json", "preprocessor_config.json", "tokenizer_config.json"}:
            digest.update(item.read_bytes())
    return digest.hexdigest()
