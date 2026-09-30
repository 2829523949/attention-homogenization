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
from harness.processors.base import ColorPrediction, RegionPrediction
from harness.processors.crop_policy import CROP_POLICY_ID, save_persistent_crop
from harness.processors.registry import ColorReaderAdapter, FlorenceQueryGroundingAdapter, ProcessorCapability, ProcessorRegistry
from harness.processors.siglip_region_query import Siglip2RegionQueryScorer, region_identity
from harness.reasoners.hard_handoff_qwen import HardHandoffReasoner
from harness.refactored_pipeline import RefactoredCaseInput, run_refactored_case
from harness.referents import GroundingQuery, Provenance, Provenanced, ReferentSpec, resolve_question
from harness.referents.region_query import RegionQuery, build_region_query


class RecordingSiglipBackend:
    model_path = Path("/mnt/d/vlm_repair_v2/models/siglip2/siglip2-base-patch16-224")
    version = "unit_test_siglip_region_query"
    model_fingerprint = "unit_test"

    def __init__(self, logit: float = 3.0):
        self.logit = float(logit)
        self.calls: list[dict[str, Any]] = []

    def logits_for_crop(self, image_view: Path, bbox_xyxy_pixels: Any, texts: Any) -> list[float]:
        self.calls.append({"image_view": str(image_view), "bbox": list(bbox_xyxy_pixels), "texts": list(texts)})
        return [self.logit]


class ContentAwareBackend:
    """Controlled fixture: logit = scaled mean red channel of the exact crop."""

    model_path = Path("/mnt/d/vlm_repair_v2/models/siglip2/siglip2-base-patch16-224")
    version = "content_aware_unit_test"
    model_fingerprint = "content_aware"

    def logits_for_crop(self, image_view: Path, bbox_xyxy_pixels: Any, texts: Any) -> list[float]:
        from harness.processors.crop_policy import deterministic_crop_box

        with Image.open(image_view) as opened:
            image = opened.convert("RGB")
            crop = image.crop(deterministic_crop_box(image.size, bbox_xyxy_pixels))
        array = np.asarray(crop, dtype=float)
        mean_red = float(array[:, :, 0].mean())
        return [mean_red / 32.0]


class FakeFlorenceProcessor:
    def predict_query(self, image_path: Path, query: str) -> list[RegionPrediction]:
        return [RegionPrediction.from_pixels([1, 1, 5, 5], 10, 10, score=None, label=query, source="fake_florence")]


class FakeAcquisition:
    name = "fake_florence_acquisition"
    version = "unit"
    can_acquire = True
    supports_certification_score = False
    supports_same_claim_rescoring = False
    comparable_across_counterfactual_views = False
    score_semantics = "CERTIFICATION_SCORE_UNSUPPORTED: unit-test acquisition only"
    score_normalization = "none"

    def ground(self, image_path: Path, query: GroundingQuery) -> list[RegionPrediction]:
        return [RegionPrediction.from_pixels([2, 2, 12, 12], 32, 16, score=None, label=query.semantic_text, source="fake_florence")]

    def score(self, image_path: Path, query: GroundingQuery, region: RegionPrediction, *, view_id: str = "RAW") -> EvidenceScore:
        return EvidenceScore.unsupported(self.name, reason="CERTIFICATION_SCORE_UNSUPPORTED: unit-test", claimed_value=region, view_id=view_id)


class SpyReasoner(HardHandoffReasoner):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, proof_state: Any) -> dict[str, Any]:
        self.calls += 1
        return {"answer": "must-not-happen", "image_received": False}


def _legal_query() -> GroundingQuery:
    spec = ReferentSpec(
        "A",
        Provenanced("bottle", Provenance.QUESTION_TEXT, "visible noun"),
        Provenanced("bottle", Provenance.QUESTION_TEXT, "visible noun"),
    )
    return GroundingQuery.from_spec(spec)


def _region() -> RegionPrediction:
    return RegionPrediction.from_pixels([2, 2, 12, 12], 32, 16, score=None, label="bottle", source="fake_florence")


def _semantic_contract(processor: str = "siglip2_region_query") -> dict[str, Any]:
    return {
        "processor": processor,
        "supports_certification_score": True,
        "supports_same_claim_rescoring": True,
        "comparable_across_counterfactual_views": True,
        "score_semantics": "fixed_region_query_compatibility",
        "score_normalization": "siglip2_query_logit_sigmoid_v1",
        "requires_fixed_region": True,
        "requires_fixed_query": True,
        "evidence_kind": "FIXED_REGION_QUERY_COMPATIBILITY",
    }


class FixedRegionScorerTests(unittest.TestCase):
    def test_same_region_query_template_and_processor_across_views(self):
        backend = RecordingSiglipBackend(logit=2.0)
        scorer = Siglip2RegionQueryScorer(backend)
        query = build_region_query(_legal_query())
        region = _region()
        paths = [Path(f"/mnt/d/vlm_repair_v2/tmp/{view}.png") for view in ("RAW", "FOCUS", "REMOVE")]
        scores = [
            scorer.score_fixed_region_query(path, "E_0", region, query, view_id=view)
            for path, view in zip(paths, ("RAW", "FOCUS", "REMOVE"))
        ]
        self.assertEqual(len({score.processor for score in scores}), 1)
        self.assertEqual(len({score.normalizer_revision for score in scores}), 1)
        region_ids = {score.diagnostics["region_id"] for score in scores}
        query_hashes = {score.diagnostics["query_hash"] for score in scores}
        template_ids = {score.diagnostics["query_template_id"] for score in scores}
        crop_policies = {score.diagnostics["crop_policy_id"] for score in scores}
        bboxes = {tuple(score.diagnostics["region_bbox_xyxy_pixels"]) for score in scores}
        queries = {score.diagnostics["query_text"] for score in scores}
        self.assertEqual(len(region_ids), 1)
        self.assertEqual(len(query_hashes), 1)
        self.assertEqual(len(template_ids), 1)
        self.assertEqual(len(crop_policies), 1)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(len(queries), 1)
        self.assertEqual(region_ids.pop(), region_identity("E_0", region))
        self.assertEqual([call["bbox"] for call in backend.calls], [[2.0, 2.0, 12.0, 12.0]] * 3)
        self.assertEqual({call["texts"][0] for call in backend.calls}, {query.text})

    def test_fixed_claim_survives_non_argmax_candidate(self):
        backend = RecordingSiglipBackend(logit=-4.0)
        scorer = Siglip2RegionQueryScorer(backend)
        query = build_region_query(_legal_query())
        score = scorer.score_fixed_region_query(Path("/mnt/d/vlm_repair_v2/tmp/non_argmax.png"), "E_0", _region(), query, view_id="FOCUS")
        self.assertEqual(score.support, "SUPPORTED")
        self.assertEqual(score.raw_score, -4.0)
        self.assertGreater(score.normalized_score, 0.0)
        self.assertLess(score.normalized_score, 0.5)
        self.assertEqual(score.claimed_value["query_hash"], query.query_hash)
        # The scorer never asks the model for an arg-max or alternative label.
        self.assertEqual(backend.calls[0]["texts"], [query.text])

    def test_no_gt_can_enter_query_construction(self):
        with self.assertRaises(AssertionError):
            ReferentSpec(
                "A",
                Provenanced("bottle", Provenance.EVALUATOR_ONLY, "hidden family category"),
            )
        query = build_region_query(_legal_query())
        self.assertIn("bottle", query.text)
        self.assertNotIn("gt", str(query.to_dict()).casefold())
        self.assertNotIn("evaluator", str(query.to_dict()).casefold())

    def test_remove_evidence_drop_on_controlled_fixture(self):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        try:
            root = Path(directory.name)
            raw_path = root / "raw.png"
            focus_path = root / "focus.png"
            remove_path = root / "remove.png"
            Image.new("RGB", (32, 16), (240, 20, 20)).save(raw_path)
            Image.new("RGB", (32, 16), (240, 20, 20)).save(focus_path)
            Image.new("RGB", (32, 16), (127, 127, 127)).save(remove_path)
            scorer = Siglip2RegionQueryScorer(ContentAwareBackend())
            query = build_region_query(_legal_query())
            region = _region()
            raw = scorer.score_fixed_region_query(raw_path, "E_0", region, query, view_id="RAW")
            focus = scorer.score_fixed_region_query(focus_path, "E_0", region, query, view_id="FOCUS")
            remove = scorer.score_fixed_region_query(remove_path, "E_0", region, query, view_id="REMOVE")
            binding_evidence = focus.normalized_score - remove.normalized_score
            self.assertGreater(binding_evidence, 0.0)
            self.assertLess(remove.normalized_score, focus.normalized_score)
            self.assertAlmostEqual(focus.raw_score, raw.raw_score, places=6)  # target-preserving fixture
        finally:
            directory.cleanup()

    def test_normalization_metadata_is_explicitly_non_calibrated(self):
        scorer = Siglip2RegionQueryScorer(RecordingSiglipBackend(logit=0.0))
        score = scorer.score_fixed_region_query(Path("/mnt/d/vlm_repair_v2/tmp/x.png"), "E_0", _region(), build_region_query(_legal_query()), view_id="RAW")
        self.assertFalse(score.calibrated)
        self.assertIn("not_calibrated", score.normalization_semantics)
        self.assertEqual(score.diagnostics["normalization_semantics"], "monotonic_sigmoid_transform_not_calibrated_probability")
        self.assertEqual(score.diagnostics["normalization_id"], "siglip2_query_logit_sigmoid_v1")
        self.assertAlmostEqual(score.normalized_score, 0.5, places=6)


class SeparationAndGateTests(unittest.TestCase):
    def test_florence_acquisition_still_cannot_issue_certification_score(self):
        adapter = FlorenceQueryGroundingAdapter(FakeFlorenceProcessor())
        score = adapter.score(Path("/mnt/d/vlm_repair_v2/tmp/x.png"), _legal_query(), _region(), view_id="RAW")
        self.assertEqual(score.support, "UNSUPPORTED")
        self.assertIsNone(score.normalized_score)
        self.assertEqual(score.processor, "florence2_query_grounding")

    def test_certifier_rejects_fixed_region_or_query_violation(self):
        certifier = EvidenceCertifier(CertificationConfig())
        scorer = Siglip2RegionQueryScorer(RecordingSiglipBackend(logit=1.0))
        query = build_region_query(_legal_query())
        region = _region()
        scores = {
            view: scorer.score_fixed_region_query(Path(f"/mnt/d/vlm_repair_v2/tmp/{view}.png"), "E_0", region, query, view_id=view)
            for view in ("RAW", "FOCUS", "REMOVE")
        }
        # Deliberately corrupt one region id.
        corrupted = scores["REMOVE"]
        corrupted_diagnostics = dict(corrupted.diagnostics)
        corrupted_diagnostics["region_id"] = "region_OTHER"
        scores["REMOVE"] = EvidenceScore(
            corrupted.processor, corrupted.score_kind, corrupted.raw_score, corrupted.normalized_score,
            corrupted.margin, corrupted.claimed_value, corrupted.support, corrupted.normalizer_revision,
            corrupted_diagnostics, corrupted.view_id, corrupted.claim_fingerprint,
            corrupted.normalization_semantics, corrupted.calibrated, corrupted.evidence_kind,
        )
        cert = certifier.certify(
            entity_id="E_0",
            predicate="SEMANTIC_REGION_SUPPORT",
            proposed_value=scores["RAW"].claimed_value,
            binding_kind="region_content_evidence",
            raw=scores["RAW"], focus=scores["FOCUS"], remove=scores["REMOVE"],
            score_contract=_semantic_contract(),
            evidence_kind="FIXED_REGION_QUERY_COMPATIBILITY",
            identity_layer="SEMANTIC_REGION_SUPPORT",
        )
        self.assertEqual(cert.status, CertificateStatus.UNCERTAIN)
        self.assertEqual(cert.diagnostics["failure_code"], "FIXED_REGION_CONTRACT_VIOLATION")

    def test_no_certified_status_without_calibration_artifact(self):
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"))
        scorer = Siglip2RegionQueryScorer(RecordingSiglipBackend(logit=3.0))
        query = build_region_query(_legal_query())
        region = _region()
        scores = {view: scorer.score_fixed_region_query(Path(f"/mnt/d/vlm_repair_v2/tmp/{view}.png"), "E_0", region, query, view_id=view) for view in ("RAW", "FOCUS", "REMOVE")}
        cert = certifier.certify(
            entity_id="E_0",
            predicate="SEMANTIC_REGION_SUPPORT",
            proposed_value=scores["RAW"].claimed_value,
            binding_kind="region_content_evidence",
            raw=scores["RAW"], focus=scores["FOCUS"], remove=scores["REMOVE"],
            score_contract=_semantic_contract(),
            evidence_kind="FIXED_REGION_QUERY_COMPATIBILITY",
            identity_layer="SEMANTIC_REGION_SUPPORT",
        )
        self.assertNotEqual(cert.status, CertificateStatus.CERTIFIED)
        self.assertFalse(cert.calibrated)

    def test_controller_abstains_with_uncalibrated_semantic_evidence(self):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        try:
            root = Path(directory.name)
            image_path = root / "case.png"
            Image.new("RGB", (32, 16), (200, 30, 30)).save(image_path)
            backend = RecordingSiglipBackend(logit=4.0)
            scorer = Siglip2RegionQueryScorer(backend)
            registry = ProcessorRegistry()
            acquisition = FakeAcquisition()
            registry.register(
                "fake_florence_acquisition",
                ProcessorCapability(
                    "grounding", ("image", "GroundingQuery"), "list[RegionPrediction]",
                    acquisition.score_semantics, "unit", None, True,
                    supports_evidence_scoring=False, notes="acquisition-only fake",
                    can_acquire=True, supported_predicates=("REFERENT_TO_REGION",),
                    supports_same_claim_rescoring=False, supports_certification_score=False,
                    score_normalization="none", comparable_across_counterfactual_views=False,
                ),
                acquisition,
            )
            registry.register(
                "siglip2_region_query",
                ProcessorCapability(
                    "region_evidence", ("image", "region", "RegionQuery"), "EvidenceScore",
                    "fixed_region_query_compatibility", scorer.version, None, True,
                    supports_evidence_scoring=True, notes="fixed-region semantic scorer",
                    can_acquire=False, supported_predicates=("SEMANTIC_REGION_SUPPORT",),
                    supports_same_claim_rescoring=True, supports_certification_score=True,
                    score_normalization="siglip2_query_logit_sigmoid_v1",
                    comparable_across_counterfactual_views=True,
                    requires_fixed_region=True, requires_fixed_query=True,
                ),
                scorer,
            )
            reasoner = SpyReasoner()
            trace = run_refactored_case(
                RefactoredCaseInput(
                    "semantic_evidence_case", "REL2C_LEFT", image_path,
                    "Between the bottle and the cup,\nwhat color is the one on the left?\nAnswer with only the color.",
                ),
                registry=registry,
                reasoner=reasoner,
                output_root=root / "out",
                certification_config=CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"),
                acquisition_processor="fake_florence_acquisition",
                grounding_score_processor="siglip2_region_query",
                color_processor="siglip2_color",
            )
            self.assertEqual(reasoner.calls, 0)
            self.assertFalse(trace["reasoner_called"])
            self.assertFalse(trace["qwen_called"])
            evidence = trace["region_query_evidence"]
            self.assertGreaterEqual(len(evidence), 1)
            record = evidence[0]
            self.assertEqual(record["acquisition_processor"], "fake_florence_acquisition")
            self.assertEqual(record["score_processor"], "siglip2_region_query")
            self.assertEqual(record["evidence_kind"], "FIXED_REGION_QUERY_COMPATIBILITY")
            self.assertEqual(record["identity_layer"], "SEMANTIC_REGION_SUPPORT")
            self.assertEqual(record["controller_decision"], "ABSTAIN_UNCERTIFIED_BINDING")
            self.assertIsNotNone(record["s_raw"])
            self.assertIsNotNone(record["s_focus"])
            self.assertIsNotNone(record["s_remove"])
            self.assertIsNotNone(record["binding_evidence"])
            self.assertEqual(record["evidence_status"], "UNCALIBRATED")
            self.assertIn("diagnostic_pattern_pass", record)
            self.assertNotEqual(record["full_logical_identity_status"], "CERTIFIED")
        finally:
            directory.cleanup()

    def test_exact_crop_policy_materialization(self):
        directory = tempfile.TemporaryDirectory(dir="/mnt/d/vlm_repair_v2/tmp")
        try:
            root = Path(directory.name)
            image_path = root / "image.png"
            Image.new("RGB", (20, 10), (1, 2, 3)).save(image_path)
            region = RegionPrediction.from_pixels([2, 2, 9, 8], 20, 10, score=None, label="x", source="pred")
            crop = save_persistent_crop(image_path, region, root / "crop.png")
            self.assertEqual(crop["crop_policy_id"], CROP_POLICY_ID)
            self.assertEqual(crop["crop_box_xyxy_pixels"], [2, 2, 9, 8])
            self.assertTrue(Path(crop["path"]).is_file())
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
