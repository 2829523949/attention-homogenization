from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import ColorPrediction, RegionPrediction


REALBIND_COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray")
RGB_PROTOTYPES = {
    "red": (210, 35, 35), "orange": (230, 125, 30), "yellow": (225, 210, 45),
    "green": (45, 145, 55), "blue": (45, 90, 190), "purple": (125, 65, 155),
    "pink": (225, 145, 175), "brown": (120, 75, 45), "black": (20, 20, 20),
    "white": (235, 235, 235), "gray": (125, 125, 125),
}


class PixelColorReader:
    name = "pixel_lab_hsv"
    version = "realbind_vocab_v1"
    # Acquisition-only: the deterministic heuristic returns a label and a
    # local distance-separation proxy, but no validated fixed-candidate
    # referent/attribute score. It must not be used for certification.
    can_acquire = True
    supports_same_claim_rescoring = False
    supports_certification_score = False
    comparable_across_counterfactual_views = False
    score_semantics = (
        "CERTIFICATION_SCORE_UNSUPPORTED: heuristic median-color distance and "
        "achromatic rules, not a validated fixed-candidate model score."
    )
    score_normalization = "none"

    def __init__(self, vocabulary: Sequence[str] = REALBIND_COLORS):
        if tuple(vocabulary) != REALBIND_COLORS:
            raise ValueError("color vocabulary must be the frozen RealBind ordering")
        self.vocabulary = tuple(vocabulary)

    def read(self, image_path: Path, region: RegionPrediction, mask_path: Path | None = None) -> ColorPrediction:
        import cv2
        from PIL import Image

        image = np.asarray(Image.open(image_path).convert("RGB"))
        x0, y0, x1, y1 = (int(round(x)) for x in region.bbox_xyxy_pixels)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return ColorPrediction(None, 0.0, self.name, {"reason": "empty_region"})
        crop = image[y0:y1, x0:x1]
        if mask_path is not None and mask_path.is_file():
            mask = np.asarray(Image.open(mask_path).convert("L"))[y0:y1, x0:x1] > 127
        else:
            h, w = crop.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            margin_x, margin_y = max(1, int(w * .18)), max(1, int(h * .18))
            mask[margin_y:max(margin_y + 1, h - margin_y), margin_x:max(margin_x + 1, w - margin_x)] = True
        pixels = crop[mask]
        if len(pixels) < 9:
            pixels = crop.reshape(-1, 3)
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
        # Suppress likely highlights/deep shadows before robust median.
        valid = (hsv[:, 2] >= 18) & (hsv[:, 2] <= 245)
        if valid.sum() >= 9:
            pixels, hsv = pixels[valid], hsv[valid]
        med_rgb = np.median(pixels, axis=0).astype(np.uint8)
        med_hsv = np.median(hsv, axis=0)
        saturation, value = med_hsv[1] / 255.0, med_hsv[2] / 255.0
        if value < .16:
            color, confidence = "black", min(1.0, (.22 - value) / .12 + .55)
        elif saturation < .16 and value > .80:
            color, confidence = "white", min(1.0, (value - .72) / .24 + .45)
        elif saturation < .20:
            color, confidence = "gray", min(1.0, (.28 - saturation) / .20 + .45)
        else:
            lab = cv2.cvtColor(med_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(float)
            distances = {}
            for name, rgb in RGB_PROTOTYPES.items():
                proto = cv2.cvtColor(np.array(rgb, dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(float)
                distances[name] = float(np.linalg.norm(lab - proto))
            # Brown needs an explicit low-value orange/red prior.
            hue_deg = med_hsv[0] * 2.0
            if 8 <= hue_deg <= 45 and value < .58:
                distances["brown"] *= .55
            ranked = sorted(distances.items(), key=lambda item: (item[1], item[0]))
            color = ranked[0][0]
            confidence = float(np.clip((ranked[1][1] - ranked[0][1]) / max(12.0, ranked[1][1]), 0.05, 1.0))
        return ColorPrediction(color, float(confidence), self.name, {"median_rgb": [int(x) for x in med_rgb], "median_hsv_opencv": [round(float(x), 4) for x in med_hsv], "sample_pixels": int(len(pixels)), "used_mask": bool(mask_path and mask_path.is_file())})
