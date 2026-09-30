from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from .base import RegionPrediction, RegionProcessor


def _polygon_to_bbox(polygon: Any) -> list[float] | None:
    """Convert a Florence polygon output into an axis-aligned bbox."""
    points = polygon
    if isinstance(polygon, dict):
        points = polygon.get("points") or polygon.get("polygon") or []
    if not isinstance(points, (list, tuple)) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, dict):
            point = [point.get("x"), point.get("y")]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            return None
    if len(xs) < 3:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _boxes_labels_scores(payload: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """Read only fields that the bundled postprocessor actually emits.

    Returns ``(boxes, labels, scores)``.  Missing labels become empty strings
    and missing scores stay an empty list; neither is replaced by query text
    or a constant confidence.
    """

    if not isinstance(payload, dict):
        return [], [], []
    boxes = list(payload.get("bboxes") or [])
    labels = list(payload.get("bboxes_labels") or payload.get("labels") or [])
    if not boxes and payload.get("polygons"):
        converted = [_polygon_to_bbox(polygon) for polygon in payload.get("polygons") or []]
        boxes = [box for box in converted if box is not None]
        labels = list(payload.get("polygons_labels") or [])
    scores = list(payload.get("scores") or [])
    return boxes, labels, scores


class Florence2Processor(RegionProcessor):
    """Legacy generic Florence acquisition; query adapter is separate.

    This class is retained for historical engineering. The primary corrected
    path uses ``FlorenceQueryGroundingAdapter`` with a typed GroundingQuery;
    ``map_generic_regions`` is explicitly LEGACY_GENERIC_PROPOSAL_BASELINE.
    """

    name = "florence2_generic"
    version = "microsoft_Florence-2-base-ft_od_v1"

    def __init__(self, model_path: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        # Microsoft's checkpoint predates the native Transformers Florence
        # tokenizer metadata. Its bundled, locally downloaded processor/model
        # code is therefore the compatible implementation under Transformers 5.
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, torch_dtype=self.dtype).to(self.device).eval()
        self.model_path = model_path
        # Logging-only state consumed by the subprocess adapter.  It is not
        # consulted by decoding, filtering, or selection.
        self.last_candidate_trace: dict[str, Any] | None = None

    def _run_task(self, image: Any, task: str, max_new_tokens: int, *, task_token: str | None = None) -> dict[str, Any]:
        import torch

        inputs = self.processor(text=task, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True)
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(text, task=task_token or task, image_size=(image.width, image.height))

    def predict(self, image_path: Path, *, task: str = "<OD>", fallback_region_proposals: bool = True, max_new_tokens: int = 512) -> list[RegionPrediction]:
        """Legacy generic acquisition; never fabricates model scores."""
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        parsed = self._run_task(image, task, max_new_tokens)
        payload = parsed.get(task, parsed)
        boxes, labels, scores = _boxes_labels_scores(payload)
        if not boxes and fallback_region_proposals and task != "<REGION_PROPOSAL>":
            task = "<REGION_PROPOSAL>"
            parsed = self._run_task(image, task, max_new_tokens)
            payload = parsed.get(task, parsed)
            boxes, labels, scores = _boxes_labels_scores(payload)
        source = f"florence2:{task}" + ("" if scores else ":native_score_unavailable")
        result = []
        for index, box in enumerate(boxes):
            label = labels[index] if index < len(labels) else ""
            try:
                result.append(
                    RegionPrediction.from_pixels(
                        box,
                        image.width,
                        image.height,
                        score=(float(scores[index]) if index < len(scores) else None),
                        label=str(label).strip() or f"region_{index}",
                        source=source,
                    )
                )
            except ValueError:
                continue
        return result

    def predict_query(self, image_path: Path, query: str, *, max_new_tokens: int = 256) -> list[RegionPrediction]:
        """Query-conditioned acquisition.

        Florence-2's bundled ``<OPEN_VOCABULARY_DETECTION>`` postprocessor
        returns ``bboxes`` plus ``bboxes_labels`` (and optionally polygons).
        It does **not** return confidence scores. Missing scores therefore
        become ``score=None`` rather than a fabricated 1.0. Labels are never
        silently overwritten by the query text.
        """
        from PIL import Image
        if str(query).strip().casefold() in {"a", "b"}:
            raise AssertionError("logical A/B symbols are not legal Florence grounding queries")
        task_token = "<OPEN_VOCABULARY_DETECTION>"
        prompt = task_token + str(query).strip()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        parsed = self._run_task(image, prompt, max_new_tokens, task_token=task_token)
        payload = parsed.get(task_token, parsed)
        boxes, labels, scores = _boxes_labels_scores(payload)
        source = "florence2:open_vocabulary_detection" + ("" if scores else ":native_score_unavailable")
        result = []
        rejected: list[dict[str, Any]] = []
        accepted_source_indices: list[int] = []
        for index, box in enumerate(boxes):
            label = labels[index] if index < len(labels) else ""
            try:
                result.append(
                    RegionPrediction.from_pixels(
                        box,
                        image.width,
                        image.height,
                        score=(float(scores[index]) if index < len(scores) else None),
                        label=str(label).strip() or f"unlabeled_region_{index}",
                        source=source,
                    )
                )
                accepted_source_indices.append(index)
            except ValueError as exc:
                rejected.append(
                    {
                        "postprocessor_index": index,
                        "bbox_xyxy_pixels": list(box) if isinstance(box, (list, tuple)) else box,
                        "label": str(label),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
        self.last_candidate_trace = {
            "task": task_token,
            "query_text": str(query).strip(),
            "raw_model_candidate_count": None,
            "raw_model_candidate_count_status": (
                "UNAVAILABLE: the autoregressive backend exposes generated text, not a "
                "structured pre-postprocessor candidate list"
            ),
            "bundled_postprocessor_candidate_count": len(boxes),
            "accepted_candidate_count": len(result),
            "rejected_candidate_count": len(rejected),
            "accepted_postprocessor_indices": accepted_source_indices,
            "rejected_candidates": rejected,
            "native_score_count": len(scores),
            "native_score_supported": bool(scores),
            "harness_threshold_applied": False,
            "harness_nms_applied": False,
            "harness_filter": "RegionPrediction bounds/shape validation only",
        }
        return result


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.casefold())
    out = set(raw)
    for token in raw:
        if token.endswith("s") and len(token) > 3:
            out.add(token[:-1])
    return out


def map_generic_regions(regions: Sequence[RegionPrediction], references: dict[str, str]) -> tuple[dict[str, RegionPrediction | None], list[dict[str, Any]]]:
    """LEGACY_GENERIC_PROPOSAL_BASELINE: map anonymous labels to A/B.

    Logical A/B symbols are not semantic detector queries. This helper must
    never be selected by the corrected main-method configuration.
    """

    available = set(range(len(regions)))
    mapped: dict[str, RegionPrediction | None] = {}
    trace: list[dict[str, Any]] = []
    for ref_id, category in references.items():
        query = _tokens(category)
        ranked = []
        for index in available:
            label = _tokens(regions[index].label)
            overlap = len(query & label) / max(1, len(query | label))
            exact = int(category.casefold().strip() == regions[index].label.casefold().strip())
            ranked.append((exact, overlap, (regions[index].score if regions[index].score is not None else float("-inf")), -index, index))
        best = max(ranked) if ranked else None
        chosen = best[-1] if best and (best[0] or best[1] > 0) else None
        mapped[ref_id] = regions[chosen] if chosen is not None else None
        if chosen is not None:
            available.remove(chosen)
        trace.append({"operation": "map_generic_region", "reference_id": ref_id, "query_category": category, "selected_index": chosen, "selected_label": regions[chosen].label if chosen is not None else None, "lexical_match": float(best[1]) if best else 0.0})
    return mapped, trace
