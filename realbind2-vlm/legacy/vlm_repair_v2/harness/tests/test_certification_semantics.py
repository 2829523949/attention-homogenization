from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from harness.certification.certificate import CertificateStatus
from harness.certification.certifier import CertificationConfig, EvidenceCertifier
from harness.certification.evidence import EvidenceScore
from harness.entities.entity_manager import EntityManager
from harness.evaluators.leakage_checks import assert_inference_safe, assert_no_image_payload
from harness.planner.predicate_dag import build_predicate_dag
from harness.processors.base import ColorPrediction, RegionPrediction
from harness.processors.color_siglip import SiglipColorReader
from harness.processors.florence2 import Florence2Processor
from harness.processors.registry import ColorReaderAdapter, ProcessorCapability, ProcessorRegistry
from harness.proof_state.compiler import DerivedFact, compile_minimal_proof_state
from harness.proof_state.controller import (
    ABSTAIN_UNCALIBRATED_BINDING,
    ABSTAIN_UNCERTIFIED_BINDING,
    PROOF_COMPLETE,
    decide_controller,
)
from harness.reasoners.hard_handoff_qwen import HardHandoffReasoner, HardHandoffRequest, Qwen35HardHandoffReasoner
from harness.refactored_pipeline import RefactoredCaseInput, run_refactored_case
from harness.referents import GroundingQuery, Provenance, Provenanced, ReferentSpec, resolve_question
from harness.views.focus import focus_view
from harness.views.remove import remove_view


def _config(view_id: str, processor: str, claimed: Any, normalized: float, *, normalizer: str = "test_norm_v1") -> EvidenceScore:
    return EvidenceScore(
        processor,
        "probability",
        float(normalized),
        float(normalized),
        claimed_value=claimed,
        normalizer_revision=normalizer,
        view_id=view_id,
    )


def _contract(processor: str, normalization: str = "test_norm_v1") -> dict[str, Any]:
    return {
        "processor": processor,
        "supports_certification_score": True,
        "supports_same_claim_rescoring": True,
        "comparable_across_counterfactual_views": True,
        "score_semantics": "scripted unit-test score",
        "score_normalization": normalization,
    }


class FakeSiglipBackend:
    """Small in-memory backend; no model load in unit tests."""

    model_path = Path("/mnt/d/vlm_repair_v2/models/siglip2/siglip2-base-patch16-224")
    version = "fake_siglip_unit_test"
    model_fingerprint = "fake"

    def __init__(self, logits_by_text: dict[str, float]):
        self.logits_by_text = logits_by_text

    def logits_for_crop(self, image_path: Path, bbox_xyxy_pixels: Any, texts: Any) -> list[float]:
        return [float(self.logits_by_text[text]) for text in texts]


class FakeFlorence(Florence2Processor):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def _run_task(self, image: Any, task: str, max_new_tokens: int, *, task_token: str | None = None) -> dict[str, Any]:
        return self.payload


class ScriptedGrounding:
    name = "scripted_grounding"
    version = "unit_test"

    def __init__(self, *, native_score: float | None):
        self.native_score = native_score

    def ground(self, image_path: Path, query: GroundingQuery) -> list[RegionPrediction]:
        if query.semantic_text == "cat":
            box = [2.0, 2.0, 10.0, 12.0]
        else:
            box = [22.0, 2.0, 30.0, 12.0]
        return [RegionPrediction.from_pixels(box, 32, 16, score=self.native_score, label=query.semantic_text, source="scripted_predicted")]

    def score(self, image_path: Path, query: GroundingQuery, region: RegionPrediction, *, view_id: str = "RAW") -> EvidenceScore:
        normalized = {"RAW": 0.90, "FOCUS": 0.95, "REMOVE": 0.10}[view_id]
        return _config(view_id, self.name, region, normalized, normalizer="scripted_region_norm_v1")


class ScriptedColorReader:
    name = "scripted_color"
    version = "unit_test"
    can_acquire = True
    supports_same_claim_rescoring = True
    supports_certification_score = True
    comparable_across_counterfactual_views = True
    score_semantics = "scripted fixed-candidate color score"
    score_normalization = "scripted_color_norm_v1"

    def read(self, image_path: Path, region: RegionPrediction) -> ColorPrediction:
        return ColorPrediction("red", 0.80, self.name, {"source": "scripted"})

    def score_claim(self, image_path: Path, region: RegionPrediction, claimed_value: str, *, view_id: str) -> EvidenceScore:
        normalized = {"RAW": 0.85, "FOCUS": 0.92, "REMOVE": 0.08}[view_id]
        return _config(view_id, self.name, claimed_value, normalized, normalizer="scripted_color_norm_v1")


class SpyReasoner(HardHandoffReasoner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def generate(self, question: str, proof_state: Any) -> dict[str, Any]:
        self.calls.append((question, proof_state))
        return {"reasoner": "spy", "answer": "ok", "raw_response": "ok", "image_received": False}


def _grounding_capability(*, supported: bool) -> ProcessorCapability:
    return ProcessorCapability(
        "grounding",
        ("image", "GroundingQuery"),
        "list[RegionPrediction]",
        "scripted fixed-claim region probability" if supported else "CERTIFICATION_SCORE_UNSUPPORTED: no native region score",
        "scripted_unit_test",
        None,
        True,
        supports_evidence_scoring=supported,
        notes="unit-test capability",
        can_acquire=True,
        supported_predicates=("REFERENT_TO_REGION",),
        supports_same_claim_rescoring=supported,
        supports_certification_score=supported,
        score_normalization="scripted_region_norm_v1" if supported else "none",
        comparable_across_counterfactual_views=supported,
    )


def _color_capability() -> ProcessorCapability:
    return ProcessorCapability(
        "color",
        ("image", "region"),
        "Observation(predicate=COLOR)",
        "scripted fixed-candidate color margin",
        "scripted_unit_test",
        None,
        True,
        supports_evidence_scoring=True,
        notes="unit-test capability",
        can_acquire=True,
        supported_predicates=("COLOR",),
        supports_same_claim_rescoring=True,
        supports_certification_score=True,
        score_normalization="scripted_color_norm_v1",
        comparable_across_counterfactual_views=True,
    )


def _registry(*, region_supported: bool) -> ProcessorRegistry:
    registry = ProcessorRegistry()
    grounder = ScriptedGrounding(native_score=0.9 if region_supported else None)
    registry.register("scripted_grounding", _grounding_capability(supported=region_supported), grounder)
    adapter = ColorReaderAdapter(ScriptedColorReader())
    registry.register("scripted_color", _color_capability(), adapter)
    return registry


class FocusRemoveSemanticsTests(unittest.TestCase):
    @staticmethod
    def _images(root: Path) -> tuple[Path, RegionPrediction, dict[str, float]]:
        # Structured scene: textured red target on a textured blue background.
        array = np.zeros((40, 60, 3), dtype=np.uint8)
        array[:, :, 2] = 180
        rr, cc = np.mgrid[0:40, 0:60]
        array[:, :, 0] = (10 + (cc * 7) % 120).astype(np.uint8)
        array[:, :, 1] = (5 + (rr * 11) % 90).astype(np.uint8)
        array[8:28, 12:36] = np.stack(
            [200 + (cc[8:28, 12:36] * 5) % 50, (rr[8:28, 12:36] * 9) % 70, (cc[8:28, 12:36] * 13) % 60], axis=-1
        ).astype(np.uint8)
        image_path = root / "scene.png"
        Image.fromarray(array, mode="RGB").save(image_path)
        region = RegionPrediction.from_pixels([12, 8, 36, 28], 60, 40, score=None, label="target", source="scripted_predicted")
        config = {"bbox_expand_frac": 0.05, "blur_radius_fraction_of_min_dimension": 0.08, "feather_radius_fraction_of_min_dimension": 0.01}
        return image_path, region, config

    def _views(self):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        root = Path(directory.name)
        image_path, region, config = self._images(root)
        focus = focus_view(image_path, region, root / "focus.png", entity_id="E_0", config=config)
        remove = remove_view(image_path, region, root / "remove.png", entity_id="E_0", config=config)
        return directory, root, region, focus, remove

    def test_focus_and_remove_are_not_identical(self):
        directory, root, region, focus, remove = self._views()
        try:
            self.assertNotEqual(focus.output_image_sha256, remove.output_image_sha256)
            self.assertNotEqual(Path(focus.path).read_bytes(), Path(remove.path).read_bytes())
            self.assertTrue(np.any(np.asarray(Image.open(focus.path)) != np.asarray(Image.open(remove.path))))
        finally:
            directory.cleanup()

    def test_focus_preserves_target_more_than_remove(self):
        directory, root, region, focus, remove = self._views()
        try:
            raw = np.asarray(Image.open(root / "scene.png").convert("RGB"), dtype=float)
            focus_array = np.asarray(Image.open(focus.path).convert("RGB"), dtype=float)
            remove_array = np.asarray(Image.open(remove.path).convert("RGB"), dtype=float)
            x0, y0, x1, y1 = (int(round(v)) for v in region.bbox_xyxy_pixels)
            target = np.s_[y0:y1, x0:x1]
            focus_mae = float(np.mean(np.abs(focus_array[target] - raw[target])))
            remove_mae = float(np.mean(np.abs(remove_array[target] - raw[target])))
            self.assertLess(focus_mae, 2.0)
            self.assertGreater(remove_mae, 20.0)
            self.assertGreater(remove_mae - focus_mae, 15.0)
        finally:
            directory.cleanup()

    def test_remove_preserves_background_more_than_focus(self):
        directory, root, region, focus, remove = self._views()
        try:
            raw = np.asarray(Image.open(root / "scene.png").convert("RGB"), dtype=float)
            focus_array = np.asarray(Image.open(focus.path).convert("RGB"), dtype=float)
            remove_array = np.asarray(Image.open(remove.path).convert("RGB"), dtype=float)
            mask = np.ones(raw.shape[:2], dtype=bool)
            sx0, sy0, sx1, sy1 = (int(round(v)) for v in remove.parameters["suppression_box_xyxy"])
            # Exclude a one-pixel rasterization boundary around the hard support.
            mask[max(0, sy0 - 1):sy1 + 2, max(0, sx0 - 1):sx1 + 2] = False
            focus_bg = float(np.mean(np.abs(focus_array[mask] - raw[mask])))
            remove_bg = float(np.mean(np.abs(remove_array[mask] - raw[mask])))
            self.assertLess(remove_bg, 0.5)
            self.assertGreater(focus_bg, 5.0)
            self.assertGreater(focus_bg - remove_bg, 5.0)
        finally:
            directory.cleanup()

    def test_same_entity_region_is_used_for_both_views(self):
        directory, root, region, focus, remove = self._views()
        try:
            self.assertEqual(focus.source_entity, "E_0")
            self.assertEqual(remove.source_entity, "E_0")
            self.assertEqual(focus.predicted_region["bbox_xyxy_pixels"], remove.predicted_region["bbox_xyxy_pixels"])
            self.assertEqual(focus.parameters["expanded_box_xyxy"], remove.parameters["suppression_box_xyxy"])
        finally:
            directory.cleanup()

    def test_automatic_views_reject_oracle_and_do_not_accept_gt_bbox(self):
        import inspect

        self.assertNotIn("gt", {name.casefold() for name in inspect.signature(remove_view).parameters})
        self.assertNotIn("bbox_gt", {name.casefold() for name in inspect.signature(focus_view).parameters})
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        try:
            root = Path(directory.name)
            image_path = root / "i.png"
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_path)
            oracle = RegionPrediction.from_pixels([2, 2, 10, 10], 20, 20, score=1.0, label="x", source="oracle_gt")
            with self.assertRaises(AssertionError):
                focus_view(image_path, oracle, root / "f.png", entity_id="E_0", config={})
            with self.assertRaises(AssertionError):
                remove_view(image_path, oracle, root / "r.png", entity_id="E_0", config={})
        finally:
            directory.cleanup()


class ScoreSemanticsTests(unittest.TestCase):
    def test_fixed_candidate_score_survives_non_argmax_candidate(self):
        logits = {
            "a red object": 1.90,
            "a orange object": -5.0,
            "a yellow object": -5.0,
            "a green object": -5.0,
            "a blue object": 2.00,
            "a purple object": -5.0,
            "a pink object": -5.0,
            "a brown object": -5.0,
            "a black object": -5.0,
            "a white object": -5.0,
            "a gray object": -5.0,
        }
        reader = SiglipColorReader(backend=FakeSiglipBackend(logits))
        region = RegionPrediction.from_pixels([1, 1, 5, 5], 10, 10, score=None, label="object", source="scripted_predicted")
        read = reader.read(Path("/mnt/d/vlm_repair_v2/tmp/nonexistent.png"), region)
        self.assertEqual(read.color, "blue")
        fixed = reader.score_claim(Path("/mnt/d/vlm_repair_v2/tmp/nonexistent.png"), region, "red", view_id="FOCUS")
        self.assertEqual(fixed.support, "SUPPORTED")
        self.assertGreater(fixed.normalized_score, 0.0)
        self.assertAlmostEqual(fixed.normalized_score, 1.0 / (1.0 + np.exp(0.10)), places=5)
        self.assertEqual(fixed.claimed_value, "red")

    def test_color_adapter_scores_same_fixed_claim_and_processor_on_all_views(self):
        class RecordingReader:
            name = "recording_color"
            version = "unit"
            can_acquire = True
            supports_same_claim_rescoring = True
            supports_certification_score = True
            comparable_across_counterfactual_views = True
            score_semantics = "recording fixed-candidate score"
            score_normalization = "recording_norm_v1"

            def __init__(self):
                self.calls = []

            def read(self, image_path, region):
                return ColorPrediction("blue", 0.9, self.name, {})

            def score_claim(self, image_path, region, claimed_value, *, view_id):
                self.calls.append((str(image_path), str(claimed_value), view_id))
                return _config(view_id, self.name, claimed_value, {"RAW": 0.4, "FOCUS": 0.8, "REMOVE": 0.1}[view_id], normalizer="recording_norm_v1")

        reader = RecordingReader()
        adapter = ColorReaderAdapter(reader)
        region = RegionPrediction.from_pixels([1, 1, 5, 5], 10, 10, score=None, label="object", source="scripted_predicted")
        scores = [
            adapter.score(Path(f"/mnt/d/vlm_repair_v2/tmp/{view_id}.png"), "E_0", region, "red", view_id=view_id)
            for view_id in ("RAW", "FOCUS", "REMOVE")
        ]
        self.assertEqual(len({score.processor for score in scores}), 1)
        self.assertEqual(len({score.normalizer_revision for score in scores}), 1)
        self.assertEqual([call[1] for call in reader.calls], ["red", "red", "red"])
        self.assertEqual([call[2] for call in reader.calls], ["RAW", "FOCUS", "REMOVE"])

    def test_unsupported_evidence_cannot_carry_a_normalized_score(self):
        with self.assertRaises(ValueError):
            EvidenceScore("p", "unsupported", None, 0.9, support="UNSUPPORTED", claimed_value="red")

    def test_reader_adapter_returns_unsupported_for_pixel_only_reader(self):
        from harness.processors.color_pixel import PixelColorReader

        adapter = ColorReaderAdapter(PixelColorReader())
        self.assertFalse(adapter.supports_certification_score)
        score = adapter.score(Path("/mnt/d/vlm_repair_v2/tmp/nonexistent.png"), "E_0", None, "red", view_id="RAW")
        self.assertEqual(score.support, "UNSUPPORTED")
        self.assertIsNone(score.normalized_score)

    def test_florence_parser_does_not_fabricate_labels_or_scores(self):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        try:
            root = Path(directory.name)
            image_path = root / "i.png"
            Image.new("RGB", (30, 20), (0, 0, 0)).save(image_path)
            payload = {"<OPEN_VOCABULARY_DETECTION>": {"bboxes": [[1, 1, 5, 5]], "bboxes_labels": ["racket"], "polygons": [], "polygons_labels": []}}
            regions = FakeFlorence(payload).predict_query(image_path, "racket")
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].label, "racket")
            self.assertIsNone(regions[0].score)
            self.assertIn("native_score_unavailable", regions[0].source)
            payload_with_scores = {"<OPEN_VOCABULARY_DETECTION>": {"bboxes": [[1, 1, 5, 5]], "bboxes_labels": ["racket"], "scores": [0.73]}}
            regions = FakeFlorence(payload_with_scores).predict_query(image_path, "racket")
            self.assertAlmostEqual(regions[0].score, 0.73)
            payload_without_labels = {"<OPEN_VOCABULARY_DETECTION>": {"bboxes": [[1, 1, 5, 5]], "polygons": []}}
            regions = FakeFlorence(payload_without_labels).predict_query(image_path, "racket")
            self.assertFalse(regions[0].label == "racket")
            self.assertTrue(regions[0].label.startswith("unlabeled_region"))
        finally:
            directory.cleanup()


class CertifierIntegrityTests(unittest.TestCase):
    def test_certifier_rejects_different_processors_and_unfixed_claims(self):
        certifier = EvidenceCertifier(CertificationConfig())
        raw = _config("RAW", "p1", "red", 0.9, normalizer="norm_v1")
        focus = _config("FOCUS", "p2", "red", 0.95, normalizer="norm_v1")
        remove = _config("REMOVE", "p1", "blue", 0.1, normalizer="norm_v1")
        cert = certifier.certify(entity_id="E_0", predicate="COLOR", proposed_value="red", binding_kind="entity_fact", raw=raw, focus=focus, remove=remove, score_contract=_contract("p1", "norm_v1"))
        self.assertEqual(cert.status, CertificateStatus.UNCERTAIN)
        self.assertIn(cert.diagnostics["failure_code"], {"SCORE_CONTRACT_MISMATCH", "FIXED_CLAIM_MISMATCH"})

    def test_fabricated_or_unsupported_scores_cannot_certify(self):
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="CALIBRATED_DEVELOPMENT", calibration_artifact_id="x"))
        raw = EvidenceScore.unsupported("p", reason="no score", claimed_value="red", view_id="RAW")
        fabricated = EvidenceScore("p", "probability", 0.99, 0.99, claimed_value="red", normalizer_revision="norm_v1", view_id="FOCUS")
        remove = _config("REMOVE", "p", "red", 0.1, normalizer="norm_v1")
        cert = certifier.certify(entity_id="E_0", predicate="COLOR", proposed_value="red", binding_kind="entity_fact", raw=raw, focus=fabricated, remove=remove, score_contract=_contract("p", "norm_v1"))
        self.assertEqual(cert.status, CertificateStatus.UNCERTAIN)
        self.assertEqual(cert.diagnostics["failure_code"], "CERTIFICATION_SCORE_UNSUPPORTED")
        self.assertIsNone(cert.necessity_delta)

    def test_processor_without_score_capability_cannot_certify(self):
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="CALIBRATED_DEVELOPMENT", calibration_artifact_id="x"))
        scores = [_config(v, "pixel", "red", 0.9, normalizer="norm_v1") for v in ("RAW", "FOCUS", "REMOVE")]
        contract = _contract("pixel", "norm_v1")
        contract["supports_certification_score"] = False
        contract["supports_same_claim_rescoring"] = False
        cert = certifier.certify(entity_id="E_0", predicate="COLOR", proposed_value="red", binding_kind="entity_fact", raw=scores[0], focus=scores[1], remove=scores[2], score_contract=contract)
        self.assertEqual(cert.status, CertificateStatus.UNCERTAIN)
        self.assertEqual(cert.diagnostics["failure_code"], "CERTIFICATION_SCORE_UNSUPPORTED")

    def test_capability_metadata_separates_acquisition_and_certification(self):
        capability = _grounding_capability(supported=False)
        payload = capability.to_dict()
        self.assertTrue(payload["can_acquire"])
        self.assertFalse(payload["supports_certification_score"])
        self.assertFalse(payload["supports_same_claim_rescoring"])
        self.assertFalse(payload["comparable_across_counterfactual_views"])
        self.assertIn("CERTIFICATION_SCORE_UNSUPPORTED", payload["score_semantics"])


class ProofStateAndControllerTests(unittest.TestCase):
    def _setup(self):
        question = "Between the cat and the dog,\nwhat color is the one on the left?\nAnswer with only the color."
        resolved = resolve_question(question, "REL2C_LEFT")
        dag = build_predicate_dag(question, "REL2C_LEFT", resolved)
        manager = EntityManager()
        for spec in resolved.specs:
            query = GroundingQuery.from_spec(spec)
            manager.ensure_entity(spec.logical_ref, query)
        region_a = RegionPrediction.from_pixels([2, 2, 10, 12], 32, 16, score=0.9, label="cat", source="scripted_predicted")
        region_b = RegionPrediction.from_pixels([22, 2, 30, 12], 32, 16, score=0.9, label="dog", source="scripted_predicted")
        manager.bind_region("A", region_a)
        manager.bind_region("B", region_b)
        return dag, manager, region_a, region_b

    def _region_certs(self, manager, region_a, region_b, *, status_mode="certified"):
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="CALIBRATED_DEVELOPMENT", calibration_artifact_id="unit-test") if status_mode == "certified" else CertificationConfig())
        contract = _contract("p", "norm_v1")

        def make(entity, region, node):
            raw = _config("RAW", "p", region, 0.9, normalizer="norm_v1")
            focus = _config("FOCUS", "p", region, 0.95, normalizer="norm_v1")
            remove = _config("REMOVE", "p", region, 0.1, normalizer="norm_v1")
            return certifier.certify(entity_id=entity.entity_id, predicate="REFERENT_TO_REGION", proposed_value=region, binding_kind="entity_binding", raw=raw, focus=focus, remove=remove, dag_node_id=node, dependency_ids=(f"localize_{node[-1]}",), score_contract=contract)

        return make(manager.get("A"), region_a, "certify_region_A"), make(manager.get("B"), region_b, "certify_region_B")

    def test_rejected_fact_never_enters_minimal_proof_state(self):
        dag, manager, region_a, region_b = self._setup()
        cert_a, cert_b = self._region_certs(manager, region_a, region_b, status_mode="certified")
        geometry = DerivedFact("relation", "HORIZONTAL_RELATION", "left", (manager.get("A").entity_id, manager.get("B").entity_id), (cert_a.certificate_id, cert_b.certificate_id), "normalized_bbox_center_x_order_v1", ("PREVIOUS_CERTIFIED_FACT",), "compare_horizontal", (cert_a.certificate_id, cert_b.certificate_id))
        selection = DerivedFact("selection", "SELECT_RELATION", {"logical_ref": "A"}, (manager.get("A").entity_id, manager.get("B").entity_id), (cert_a.certificate_id, cert_b.certificate_id), "deterministic_center_side_selection_v1", ("PREVIOUS_CERTIFIED_FACT",), "select_target", ("relation",))
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="CALIBRATED_DEVELOPMENT", calibration_artifact_id="unit-test"))
        contract = _contract("p", "norm_v1")
        rejected_color = certifier.certify(
            entity_id=manager.get("A").entity_id,
            predicate="COLOR",
            proposed_value="red",
            binding_kind="entity_fact",
            raw=_config("RAW", "p", "red", 0.8, normalizer="norm_v1"),
            focus=_config("FOCUS", "p", "red", 0.5, normalizer="norm_v1"),
            remove=_config("REMOVE", "p", "red", 0.4, normalizer="norm_v1"),
            dag_node_id="certify_selected_color",
            dependency_ids=(cert_a.certificate_id, "selection"),
            score_contract=contract,
        )
        self.assertEqual(rejected_color.status, CertificateStatus.REJECTED)
        state = compile_minimal_proof_state(dag, manager, (cert_a, cert_b, rejected_color), (geometry, selection), include_uncalibrated=False)
        self.assertFalse(any(getattr(fact, "predicate", "") == "COLOR" for fact in state.facts))
        self.assertFalse(state.proof_complete)

    def test_derived_fact_requires_satisfied_dependencies(self):
        dag, manager, region_a, region_b = self._setup()
        cert_a, cert_b = self._region_certs(manager, region_a, region_b, status_mode="certified")
        bad_geometry = DerivedFact("relation", "HORIZONTAL_RELATION", "left", (manager.get("A").entity_id, manager.get("B").entity_id), (cert_a.certificate_id, cert_b.certificate_id), "normalized_bbox_center_x_order_v1", ("PREVIOUS_CERTIFIED_FACT",), "compare_horizontal", ("missing-certificate",))
        state = compile_minimal_proof_state(dag, manager, (cert_a, cert_b), (bad_geometry,), include_uncalibrated=False)
        self.assertFalse(any(getattr(fact, "predicate", "") == "HORIZONTAL_RELATION" for fact in state.facts))
        self.assertFalse(state.dependency_closure_valid)
        self.assertFalse(state.proof_complete)

    def test_unresolved_required_certificate_blocks_proof_state_and_reasoner(self):
        dag, manager, region_a, region_b = self._setup()
        certifier = EvidenceCertifier(CertificationConfig())
        unsupported = EvidenceScore.unsupported("florence", reason="CERTIFICATION_SCORE_UNSUPPORTED: no score", claimed_value=region_a, view_id="RAW")
        unsupported_focus = EvidenceScore.unsupported("florence", reason="CERTIFICATION_SCORE_UNSUPPORTED: no score", claimed_value=region_a, view_id="FOCUS")
        unsupported_remove = EvidenceScore.unsupported("florence", reason="CERTIFICATION_SCORE_UNSUPPORTED: no score", claimed_value=region_a, view_id="REMOVE")
        cert_a = certifier.certify(entity_id=manager.get("A").entity_id, predicate="REFERENT_TO_REGION", proposed_value=region_a, binding_kind="entity_binding", raw=unsupported, focus=unsupported_focus, remove=unsupported_remove, dag_node_id="certify_region_A", dependency_ids=("localize_A",))
        state = compile_minimal_proof_state(dag, manager, (cert_a,), (), include_uncalibrated=False)
        self.assertFalse(state.proof_complete)
        self.assertIsNone(state.entities[0].region)
        decision = decide_controller(dag, manager, (cert_a,), (), calibration_mode="UNCALIBRATED_DIAGNOSTIC")
        self.assertEqual(decision.status, ABSTAIN_UNCERTIFIED_BINDING)
        self.assertEqual(decision.reason_code, "CERTIFICATION_SCORE_UNSUPPORTED")
        self.assertFalse(decision.allow_reasoner)


class PipelineGateTests(unittest.TestCase):
    def _run_case(self, *, region_supported: bool, calibration_mode: str, artifact: str | None, task_type: str = "REL2C_LEFT", question: str = "Between the cat and the dog,\nwhat color is the one on the left?\nAnswer with only the color."):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        root = Path(directory.name)
        image_path = root / "case.png"
        Image.new("RGB", (32, 16), (200, 40, 40)).save(image_path)
        reasoner = SpyReasoner()
        trace = run_refactored_case(
            RefactoredCaseInput(
                "unit_case",
                task_type,
                image_path,
                question,
            ),
            registry=_registry(region_supported=region_supported),
            reasoner=reasoner,
            output_root=root / "out",
            certification_config=CertificationConfig(calibration_mode=calibration_mode, calibration_artifact_id=artifact),
            acquisition_processor="scripted_grounding",
            color_processor="scripted_color",
        )
        return directory, trace, reasoner

    def test_incomplete_or_unsupported_proof_never_invokes_reasoner(self):
        directory, trace, reasoner = self._run_case(region_supported=False, calibration_mode="UNCALIBRATED_DIAGNOSTIC", artifact=None)
        try:
            self.assertEqual(len(reasoner.calls), 0)
            self.assertFalse(trace["reasoner_called"])
            self.assertIn(trace["controller_decision"]["status"], {ABSTAIN_UNCERTIFIED_BINDING})
            self.assertIn("CERTIFICATION_SCORE_UNSUPPORTED", str(trace["controller_decision"]))
            self.assertIsNone(trace["final_answer"])
        finally:
            directory.cleanup()

    def test_uncalibrated_diagnostic_never_invokes_reasoner(self):
        directory, trace, reasoner = self._run_case(region_supported=True, calibration_mode="UNCALIBRATED_DIAGNOSTIC", artifact=None)
        try:
            self.assertEqual(len(reasoner.calls), 0)
            self.assertEqual(trace["controller_decision"]["status"], ABSTAIN_UNCALIBRATED_BINDING)
            self.assertTrue(trace["diagnostic_proof_state"]["proof_complete"])
            self.assertIsNone(trace["final_answer"])
        finally:
            directory.cleanup()

    def test_qwen_adapter_is_never_invoked_when_proof_is_incomplete(self):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        try:
            root = Path(directory.name)
            image_path = root / "case.png"
            Image.new("RGB", (32, 16), (10, 20, 30)).save(image_path)
            calls: list[Any] = []
            original = Qwen35HardHandoffReasoner.generate

            def spy_generate(self, question, proof_state):
                calls.append((question, proof_state))
                return {"answer": "must-not-happen", "image_received": False}

            Qwen35HardHandoffReasoner.generate = spy_generate
            try:
                qwen_like = Qwen35HardHandoffReasoner.__new__(Qwen35HardHandoffReasoner)
                trace = run_refactored_case(
                    RefactoredCaseInput(
                        "unit_qwen_gate",
                        "REL2C_LEFT",
                        image_path,
                        "Between the cat and the dog,\nwhat color is the one on the left?\nAnswer with only the color.",
                    ),
                    registry=_registry(region_supported=False),
                    reasoner=qwen_like,
                    output_root=root / "out",
                    certification_config=CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"),
                    acquisition_processor="scripted_grounding",
                    color_processor="scripted_color",
                )
            finally:
                Qwen35HardHandoffReasoner.generate = original
            self.assertEqual(calls, [])
            self.assertFalse(trace["qwen_called"])
            self.assertFalse(trace["reasoner_called"])
        finally:
            directory.cleanup()

    def test_calibrated_complete_proof_invokes_reasoner_once(self):
        directory, trace, reasoner = self._run_case(region_supported=True, calibration_mode="CALIBRATED_DEVELOPMENT", artifact="unit-test-artifact")
        try:
            self.assertEqual(len(reasoner.calls), 1)
            self.assertEqual(trace["controller_decision"]["status"], PROOF_COMPLETE)
            self.assertTrue(trace["reasoner_called"])
            self.assertEqual(trace["final_answer"], "ok")
        finally:
            directory.cleanup()

    def test_rel2i_proof_state_requires_no_color_node(self):
        directory, trace, reasoner = self._run_case(
            region_supported=True,
            calibration_mode="CALIBRATED_DEVELOPMENT",
            artifact="unit-test-artifact",
            task_type="REL2I_LEFT",
            question="Between the cat and the dog,\nwhich one is on the left?\nAnswer with only the object name.",
        )
        try:
            color_nodes = [node for node in trace["predicate_dag"]["nodes"] if node["operation"] == "READ_COLOR"]
            self.assertEqual(color_nodes, [])
            self.assertEqual(trace["controller_decision"]["status"], PROOF_COMPLETE)
            self.assertEqual(len(reasoner.calls), 1)
            color_facts = [fact for fact in trace["minimal_certified_proof_state"]["facts"] if fact["predicate"] == "COLOR"]
            self.assertEqual(color_facts, [])
        finally:
            directory.cleanup()


class HardHandoffAndLeakageTests(unittest.TestCase):
    def test_hard_handoff_rejects_incomplete_state_and_image_payload(self):
        contract = _contract("p", "norm_v1")
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="CALIBRATED_DEVELOPMENT", calibration_artifact_id="x"))
        raw = _config("RAW", "p", "red", 0.9, normalizer="norm_v1")
        focus = _config("FOCUS", "p", "red", 0.95, normalizer="norm_v1")
        remove = _config("REMOVE", "p", "red", 0.1, normalizer="norm_v1")
        cert = certifier.certify(entity_id="E_0", predicate="COLOR", proposed_value="red", binding_kind="entity_fact", raw=raw, focus=focus, remove=remove, score_contract=contract)
        # Structurally complete but empty state is still not proof-complete.
        from harness.proof_state.compiler import ProofState

        empty = ProofState("s", "REL2C_LEFT", "d", (), (), {}, proof_complete=False, certified_proof_complete=False)
        with self.assertRaises(AssertionError):
            HardHandoffRequest("q", empty).to_dict()
        with self.assertRaises(AssertionError):
            assert_no_image_payload({"image": b"bytes"})
        with self.assertRaises(AssertionError):
            assert_no_image_payload({"image_tensor": object()})

    def test_evaluator_only_fields_cannot_enter_automatic_objects(self):
        with self.assertRaises(AssertionError):
            ReferentSpec("A", Provenanced("cat", Provenance.EVALUATOR_ONLY, "family category"))
        with self.assertRaises(AssertionError):
            assert_inference_safe({"gt_bbox": [1, 2, 3, 4]}, context="unit-test")
        with self.assertRaises(AssertionError):
            assert_inference_safe({"inner": {"left_is_a": True}}, context="unit-test")

    def test_persistent_entity_reuse_and_no_silent_regrounding(self):
        spec = ReferentSpec("A", Provenanced("cat", Provenance.QUESTION_TEXT, "visible noun"), Provenanced("cat", Provenance.QUESTION_TEXT, "visible noun"))
        manager = EntityManager()
        query = GroundingQuery.from_spec(spec)
        first = manager.ensure_entity("A", query)
        second = manager.ensure_entity("A", query)
        self.assertEqual(first.entity_id, second.entity_id)
        region = RegionPrediction.from_pixels([1, 1, 4, 4], 10, 10, score=None, label="cat", source="predicted")
        manager.bind_region("A", region)
        with self.assertRaises(RuntimeError):
            manager.bind_region("A", region)
        with self.assertRaises(RuntimeError):
            manager.bind_region("A", region, reason="   ")
        manager.bind_region("A", region, reason="controlled diagnostic re-resolution")
        self.assertEqual(manager.get("A").region, region)
        self.assertTrue(any(event.event == "CONTROLLED_RERESOLUTION" for event in manager.events))


if __name__ == "__main__":
    unittest.main()
