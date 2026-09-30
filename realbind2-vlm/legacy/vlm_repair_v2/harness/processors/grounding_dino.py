from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.acquisition.candidates import (
    CandidateRegion,
    CandidateSet,
    GroundingDinoQuerySerializer,
    candidate_request_id,
)
from harness.processors.base import RegionPrediction, RegionProcessor
from harness.referents.referent_spec import GroundingQuery, ReferentSpec


@dataclass(frozen=True)
class GroundingDinoInferenceConfig:
    """Predeclared, unoptimized pilot settings."""

    box_threshold: float = 0.25
    text_threshold: float = 0.25
    nms_iou_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "threshold_policy": "standard_transformers_defaults_unoptimized_pilot",
        }


class GroundingDinoCandidateGenerator(RegionProcessor):
    """Text-conditioned region acquisition only; never a binding certifier."""

    name = "grounding_dino"
    version = "IDEA-Research/grounding-dino-base@12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    can_acquire = True
    supports_certification_score = False
    supports_same_claim_rescoring = False
    comparable_across_counterfactual_views = False
    score_semantics = "model_native_text_region_candidate_score"
    score_normalization = "none"

    def __init__(
        self,
        model_path: Path,
        device: str | None = None,
        *,
        config: GroundingDinoInferenceConfig | None = None,
        serializer: GroundingDinoQuerySerializer | None = None,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not Path(model_path).is_dir():
            raise FileNotFoundError(model_path)
        self.model_path = Path(model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config or GroundingDinoInferenceConfig()
        self.serializer = serializer or GroundingDinoQuerySerializer()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path, local_files_only=True
        ).to(self.device).eval()
        self.last_candidate_trace: dict[str, Any] | None = None

    @staticmethod
    def _bounded_box(box: list[float], width: int, height: int) -> tuple[list[float], bool]:
        bounded = [
            min(max(float(box[0]), 0.0), float(width)),
            min(max(float(box[1]), 0.0), float(height)),
            min(max(float(box[2]), 0.0), float(width)),
            min(max(float(box[3]), 0.0), float(height)),
        ]
        return bounded, any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(box, bounded))

    def acquire(self, image_path: Path, referent: ReferentSpec) -> CandidateSet:
        import torch
        from PIL import Image

        serialized = self.serializer.serialize(referent)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        # A singleton candidate-label list lets the official processor apply
        # its required punctuation without changing the stored legal query.
        inputs = self.processor(images=image, text=[[serialized.text]], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        raw_scores = torch.sigmoid(outputs.logits[0]).max(dim=-1).values
        raw_indices = torch.nonzero(raw_scores > self.config.box_threshold, as_tuple=False).flatten().tolist()
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = list(result.get("text_labels") or result.get("labels") or [])
        if not (len(boxes) == len(scores) == len(raw_indices)):
            raise RuntimeError("Grounding DINO postprocessor count/index mismatch")

        candidates: list[CandidateRegion] = []
        rejected: list[dict[str, Any]] = []
        for post_index, (box, score, raw_index) in enumerate(zip(boxes, scores, raw_indices)):
            bounded, clipped = self._bounded_box(box, image.width, image.height)
            if bounded[0] >= bounded[2] or bounded[1] >= bounded[3]:
                rejected.append(
                    {
                        "raw_candidate_index": int(raw_index),
                        "post_threshold_index": post_index,
                        "reason": "degenerate_after_bounds_clip",
                    }
                )
                continue
            region = RegionPrediction.from_pixels(
                bounded,
                image.width,
                image.height,
                score=float(score),
                label=str(labels[post_index]) if post_index < len(labels) else "",
                source="grounding_dino:text_conditioned_region_grounding",
            )
            candidates.append(
                CandidateRegion(
                    candidate_id=f"{serialized.logical_ref}:grounding_dino:{raw_index}",
                    bbox_xyxy_pixels=region.bbox_xyxy_pixels,
                    bbox_xyxy_normalized=region.bbox_xyxy_normalized,
                    native_grounding_score=region.score,
                    native_label_text=region.label or None,
                    source_backend=region.source,
                    query_hash=serialized.query_hash,
                    provenance=serialized.provenance + ("TOOL_OBSERVATION",),
                    raw_candidate_index=int(raw_index),
                    postprocessing_metadata={
                        "post_threshold_index": post_index,
                        "bounds_clipped": clipped,
                        "nms_applied": False,
                    },
                    score_semantics=self.score_semantics,
                )
            )

        trace = {
            "backend": "grounding_dino",
            "raw_candidate_count": int(outputs.pred_boxes.shape[1]),
            "post_threshold_count": len(boxes),
            "post_nms_count": len(boxes),
            "nms_applied": False,
            "rejected_invalid_count": len(rejected),
            "rejected_invalid_candidates": rejected,
            "final_candidate_set_count": len(candidates),
            "configuration": self.config.to_dict(),
            "query_hash": serialized.query_hash,
            "serializer_version": serialized.serializer_version,
        }
        self.last_candidate_trace = trace
        return CandidateSet(
            request_id=candidate_request_id(image_path, serialized, "grounding_dino"),
            logical_ref=referent.logical_ref,
            original_referent_spec=referent.to_dict(),
            legal_query=serialized.text,
            query_hash=serialized.query_hash,
            query_provenance=serialized.provenance,
            serializer_version=serialized.serializer_version,
            acquisition_backend="grounding_dino",
            model_revision=self.version,
            candidates=tuple(candidates),
            acquisition_trace=trace,
        )

    def ground(self, image_path: Path, query: GroundingQuery) -> list[RegionPrediction]:
        """Compatibility method for typed legacy callers."""
        if not isinstance(query, GroundingQuery):
            raise TypeError("grounding requires GroundingQuery")
        # GroundingQuery cannot reconstruct all ReferentSpec fields.  This
        # compatibility route uses only its already-legal semantic source.
        from harness.referents.referent_spec import Provenance, Provenanced

        source_provenance = next(iter(query.token_provenance.values()))[0]
        spec = ReferentSpec(
            logical_ref=query.logical_ref,
            category_constraint=Provenanced(query.semantic_text, source_provenance, query.source_spec),
        )
        return list(self.acquire(image_path, spec).regions())

    def predict(self, image_path: Path, *, query: str, threshold: float = 0.25) -> list[RegionPrediction]:
        """Old optional-processor API retained for compatibility tests."""
        from harness.referents.referent_spec import Provenance, Provenanced

        if threshold != self.config.box_threshold:
            raise ValueError("pilot thresholds are frozen; construct an explicit config instead")
        spec = ReferentSpec(
            logical_ref="QUERY",
            category_constraint=Provenanced(str(query), Provenance.PROTOCOL_INTERFACE, "predict compatibility"),
        )
        return list(self.acquire(image_path, spec).regions())


# Backward-compatible import name for the former optional thin processor.
GroundingDinoProcessor = GroundingDinoCandidateGenerator
