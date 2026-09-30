from __future__ import annotations

from pathlib import Path

from .base import RegionPrediction


class Sam2Refiner:
    """Optional bbox-prompted SAM2 adapter. It has no annotation-bearing API."""

    name = "sam2_bbox_refiner"
    version = "v1"

    def __init__(self, model_path: Path, device: str | None = None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Sam2Model.from_pretrained(model_path, local_files_only=True).to(self.device).eval()

    def refine(self, image_path: Path, predicted_region: RegionPrediction, output_path: Path) -> Path:
        import numpy as np
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        box = [[list(predicted_region.bbox_xyxy_pixels)]]
        inputs = self.processor(images=image, input_boxes=box, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        mask = (masks[0, 0].numpy() > 0).astype(np.uint8) * 255
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(output_path)
        return output_path
