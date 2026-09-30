"""Uncalibrated, full-image SigLIP2 query logits for contextual interventions."""

from pathlib import Path

from PIL import Image


class Siglip2FullImageQueryScorer:
    name = "siglip2_full_image_query"
    score_scope = "full_image"
    score_semantics = "image_query_compatibility_logit"
    calibrated = False

    def __init__(self, backend):
        self.backend = backend

    def score(self, image_path: Path, query_text: str) -> float:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            scores = self.backend.image_text_logits(image, [query_text])
        if len(scores) != 1:
            raise RuntimeError("Expected one full-image query logit")
        return float(scores[0])
