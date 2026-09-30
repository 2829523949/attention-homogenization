"""Single registry/factory for replaceable candidate acquisition backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from harness.acquisition.candidates import FlorenceCandidateGenerator
from harness.configs.paths import Paths
from harness.processors.florence_subprocess import FlorenceSubprocessGrounding
from harness.processors.grounding_dino import GroundingDinoCandidateGenerator, GroundingDinoInferenceConfig
from harness.processors.registry import ProcessorCapability, ProcessorRegistry


BACKEND_PROCESSOR_NAMES = {
    "florence2": "florence2_query_grounding",
    "grounding_dino": "grounding_dino",
}


def acquisition_processor_name(backend: str) -> str:
    try:
        return BACKEND_PROCESSOR_NAMES[backend]
    except KeyError as exc:
        raise KeyError(f"unknown acquisition backend: {backend!r}") from exc


def build_candidate_generator(
    backend: str,
    paths: Paths,
    *,
    device: str = "cpu",
    florence_device: str = "cpu",
) -> Any:
    factories: dict[str, Callable[[], Any]] = {
        "florence2": lambda: FlorenceCandidateGenerator(
            FlorenceSubprocessGrounding(paths.florence2, device=florence_device)
        ),
        "grounding_dino": lambda: GroundingDinoCandidateGenerator(
            paths.grounding_dino,
            device=device,
            config=GroundingDinoInferenceConfig(
                box_threshold=0.25,
                text_threshold=0.25,
                nms_iou_threshold=None,
            ),
        ),
    }
    try:
        return factories[backend]()
    except KeyError as exc:
        raise KeyError(f"unknown acquisition backend: {backend!r}") from exc


def register_candidate_backend(
    registry: ProcessorRegistry,
    backend: str,
    paths: Paths,
    *,
    device: str = "cpu",
    florence_device: str = "cpu",
) -> Any:
    generator = build_candidate_generator(
        backend, paths, device=device, florence_device=florence_device
    )
    processor_name = acquisition_processor_name(backend)
    native_score = backend == "grounding_dino"
    registry.register(
        processor_name,
        ProcessorCapability(
            primitive="grounding",
            legal_input_types=("image", "ReferentSpec"),
            output_schema="CandidateSet[CandidateRegion]",
            score_semantics=generator.score_semantics,
            model_revision=generator.version,
            model_path=str(generator.model_path if hasattr(generator, "model_path") else paths.florence2),
            deterministic=True,
            supports_evidence_scoring=False,
            notes=(
                "Candidate acquisition only. Native candidate scores, when present, are ranking/diagnostic "
                "signals and are not fixed-region binding-certification scores."
            ),
            can_acquire=True,
            supported_predicates=("REFERENT_TO_REGION",),
            supports_same_claim_rescoring=False,
            supports_certification_score=False,
            score_normalization="none",
            comparable_across_counterfactual_views=False,
            supported_primitives=("TEXT_CONDITIONED_REGION_GROUNDING",),
            returns_multiple_candidates=True,
            supports_native_candidate_score=native_score,
            supports_fixed_region_certification_score=False,
        ),
        generator,
    )
    return generator

