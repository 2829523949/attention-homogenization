from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from harness.acquisition.candidates import (
    CandidateGenerator,
    CandidateRegion,
    CandidateSet,
    FlorenceCandidateGenerator,
    GroundingDinoQuerySerializer,
)
from harness.acquisition.router import AcquisitionRouter
from harness.entities.entity_manager import EntityManager
from harness.planner.predicate_dag import build_predicate_dag
from harness.processors.base import RegionPrediction
from harness.processors.grounding_dino import GroundingDinoCandidateGenerator
from harness.processors.registry import ProcessorCapability, ProcessorRegistry
from harness.processors.siglip_region_query import Siglip2RegionQueryScorer
from harness.referents import Provenance, Provenanced, ReferentSpec, resolve_question
from harness.referents.region_query import build_region_query
from harness.views.focus import focus_view
from harness.views.remove import remove_view


def spec(noun: str = "racket") -> ReferentSpec:
    return ReferentSpec(
        "A",
        category_constraint=Provenanced(noun, Provenance.QUESTION_TEXT, "visible question noun"),
    )


def candidate(index: int, score: float | None, *, x0: float = 1.0) -> CandidateRegion:
    query = GroundingDinoQuerySerializer().serialize(spec())
    return CandidateRegion(
        candidate_id=f"A:grounding_dino:{index}",
        bbox_xyxy_pixels=(x0, 1.0, x0 + 3.0, 5.0),
        bbox_xyxy_normalized=(x0 / 10.0, 0.1, (x0 + 3.0) / 10.0, 0.5),
        native_grounding_score=score,
        native_label_text="racket",
        source_backend="grounding_dino:text_conditioned_region_grounding",
        query_hash=query.query_hash,
        provenance=("QUESTION_TEXT", "TOOL_OBSERVATION"),
        raw_candidate_index=index,
        postprocessing_metadata={"nms_applied": False},
        score_semantics=None if score is None else "model_native_text_region_candidate_score",
    )


def candidate_set(items: tuple[CandidateRegion, ...]) -> CandidateSet:
    referent = spec()
    query = GroundingDinoQuerySerializer().serialize(referent)
    return CandidateSet(
        request_id="ACQ_test",
        logical_ref="A",
        original_referent_spec=referent.to_dict(),
        legal_query=query.text,
        query_hash=query.query_hash,
        query_provenance=query.provenance,
        serializer_version=query.serializer_version,
        acquisition_backend="grounding_dino",
        model_revision="test",
        candidates=items,
        acquisition_trace={"raw_candidate_count": 900, "final_candidate_set_count": len(items)},
    )


class FakeGenerator:
    name = "fake_grounding_dino"
    version = "test"
    score_semantics = "model_native_text_region_candidate_score"
    last_candidate_trace = {"raw_candidate_count": 900}

    def __init__(self, result: CandidateSet):
        self.result = result

    def acquire(self, image_path: Path, referent: ReferentSpec) -> CandidateSet:
        return self.result


class FakeSiglipBackend:
    version = "fake"
    model_fingerprint = "fake"

    def logits_for_crop(self, image_path, bbox, texts):
        self.last_bbox = tuple(bbox)
        return [0.25]


class GroundingDinoAcquisitionTests(unittest.TestCase):
    def test_01_grounding_dino_satisfies_candidate_generator_interface(self):
        self.assertTrue(hasattr(GroundingDinoCandidateGenerator, "acquire"))
        self.assertIn("CandidateSet", str(inspect.signature(GroundingDinoCandidateGenerator.acquire)))
        self.assertTrue(isinstance(FakeGenerator(candidate_set((candidate(0, 0.7),))), CandidateGenerator))

    def test_02_query_serializer_preserves_legal_semantics_and_provenance(self):
        query = GroundingDinoQuerySerializer().serialize(spec("racket"))
        self.assertEqual(query.text, "racket")
        self.assertEqual(query.provenance, ("QUESTION_TEXT",))
        self.assertEqual(len(query.query_hash), 64)

    def test_03_hidden_gt_cannot_enter_query(self):
        with self.assertRaises(AssertionError):
            GroundingDinoQuerySerializer().serialize(
                ReferentSpec(
                    "A",
                    category_constraint=Provenanced("racket", Provenance.EVALUATOR_ONLY, "hidden category"),
                )
            )

    def test_04_candidate_set_can_preserve_multiple_candidates(self):
        result = candidate_set((candidate(0, 0.8), candidate(5, 0.6, x0=5.0)))
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual([row.raw_candidate_index for row in result.candidates], [0, 5])

    def test_05_genuine_native_score_is_preserved(self):
        row = candidate(0, 0.731)
        self.assertEqual(row.native_grounding_score, 0.731)
        self.assertEqual(row.score_semantics, "model_native_text_region_candidate_score")

    def test_06_missing_native_score_is_not_fabricated(self):
        row = candidate(0, None)
        self.assertIsNone(row.native_grounding_score)
        self.assertIsNone(row.as_region_prediction().score)

    def test_07_evaluator_iou_cannot_enter_or_reorder_inference_candidates(self):
        with self.assertRaises(AssertionError):
            CandidateRegion(
                **{
                    **candidate(0, 0.5).__dict__,
                    "postprocessing_metadata": {"gt_iou": 1.0},
                }
            )
        result = candidate_set((candidate(0, 0.4, x0=1.0), candidate(1, 0.9, x0=6.0)))
        self.assertEqual([row.raw_candidate_index for row in result.candidates], [0, 1])

    def test_08_florence_adapter_preserves_baseline_regions_and_missing_scores(self):
        class FakeFlorence:
            version = "baseline"
            score_semantics = "unsupported"
            last_candidate_trace = {"accepted_candidate_count": 2}

            def ground(self, image_path, query):
                return [
                    RegionPrediction.from_pixels([5, 1, 8, 5], 10, 10, score=None, label="racket", source="florence"),
                    RegionPrediction.from_pixels([1, 1, 4, 5], 10, 10, score=None, label="racket", source="florence"),
                ]

        result = FlorenceCandidateGenerator(FakeFlorence()).acquire(Path("unused"), spec())
        self.assertEqual([row.bbox_xyxy_pixels for row in result.candidates], [(5.0, 1.0, 8.0, 5.0), (1.0, 1.0, 4.0, 5.0)])
        self.assertTrue(all(row.native_grounding_score is None for row in result.candidates))

    def test_09_backend_switch_keeps_router_downstream_region_api(self):
        question = "What color is the racket?\nAnswer with only the color."
        resolved = resolve_question(question, "I2C_A")
        dag = build_predicate_dag(question, "I2C_A", resolved)
        result = candidate_set((candidate(0, 0.4), candidate(1, 0.9, x0=6.0)))
        registry = ProcessorRegistry()
        registry.register(
            "grounding_dino",
            ProcessorCapability(
                "grounding", ("image", "ReferentSpec"), "CandidateSet", "native", "test", None, True,
                supports_evidence_scoring=False,
            ),
            FakeGenerator(result),
        )
        manager = EntityManager()
        AcquisitionRouter(registry, manager).acquire_regions(Path("unused"), dag, resolved.specs, grounding_processor="grounding_dino")
        self.assertIsInstance(manager.get("A").region, RegionPrediction)
        self.assertEqual(manager.get("A").region.score, 0.9)

    def test_10_persistent_region_remains_fixed_after_acquisition(self):
        manager = EntityManager()
        query = GroundingDinoQuerySerializer().serialize(spec()).source_grounding_query
        manager.ensure_entity("A", query)
        region = candidate(0, 0.8).as_region_prediction()
        manager.bind_region("A", region, source="grounding_dino")
        self.assertIs(manager.reuse("A", reason="fixed scorer").region, region)
        with self.assertRaises(RuntimeError):
            manager.bind_region("A", candidate(1, 0.9, x0=5.0).as_region_prediction())

    def test_11_siglip_fixed_region_scorer_accepts_grounding_dino_region(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = FakeSiglipBackend()
            scorer = Siglip2RegionQueryScorer(backend)
            query = build_region_query(GroundingDinoQuerySerializer().serialize(spec()).source_grounding_query)
            region = candidate(0, 0.8).as_region_prediction()
            score = scorer.score_fixed_region_query(image_path, "E_0", region, query, view_id="RAW")
            self.assertEqual(backend.last_bbox, region.bbox_xyxy_pixels)
            self.assertEqual(score.support, "SUPPORTED")

    def test_12_focus_remove_reuse_exact_grounding_dino_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            region = candidate(0, 0.8).as_region_prediction()
            config = {"bbox_expand_frac": 0.1, "blur_radius_fraction_of_min_dimension": 0.025, "feather_radius_fraction_of_min_dimension": 0.01}
            focus = focus_view(image_path, region, root / "focus.png", entity_id="E_0", config=config)
            remove = remove_view(image_path, region, root / "remove.png", entity_id="E_0", config=config)
            self.assertEqual(focus.predicted_region["bbox_xyxy_pixels"], list(region.bbox_xyxy_pixels))
            self.assertEqual(remove.predicted_region["bbox_xyxy_pixels"], list(region.bbox_xyxy_pixels))

    def test_13_qwen_is_not_part_of_candidate_generator(self):
        source = inspect.getsource(GroundingDinoCandidateGenerator).casefold()
        self.assertNotIn("qwen", source)
        self.assertNotIn("reasoner", source)

    def test_14_no_sam2_dependency_introduced(self):
        modules = [
            Path("harness/acquisition/candidates.py"),
            Path("harness/acquisition/backend_registry.py"),
            Path("harness/processors/grounding_dino.py"),
        ]
        source = "\n".join(path.read_text(encoding="utf-8").casefold() for path in modules)
        self.assertNotIn("import sam2", source)
        self.assertNotIn("from sam2", source)

    def test_15_candidate_oracle_is_evaluator_only(self):
        fields = set(candidate(0, 0.8).to_dict())
        self.assertFalse({"gt_iou", "best_candidate_iou", "evaluator_rank", "selected_correctness"} & fields)
        self.assertNotIn("EVALUATOR_ONLY", json.dumps(candidate_set((candidate(0, 0.8),)).to_dict()))


if __name__ == "__main__":
    unittest.main()
