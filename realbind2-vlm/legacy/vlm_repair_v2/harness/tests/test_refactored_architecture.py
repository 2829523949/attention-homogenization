from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from harness.acquisition.observations import Observation
from harness.acquisition.router import AcquisitionRouter
from harness.certification.certificate import CertificateStatus
from harness.certification.certifier import CertificationConfig, EvidenceCertifier
from harness.certification.evidence import EvidenceScore
from harness.entities.entity_manager import EntityManager
from harness.evaluators.leakage_checks import assert_inference_safe, assert_no_image_payload
from harness.planner.predicate_dag import Operation, build_predicate_dag
from harness.processors.base import RegionPrediction
from harness.processors.registry import ProcessorCapability, ProcessorRegistry
from harness.proof_state.compiler import DerivedFact, compile_minimal_proof_state
from harness.proof_state.serializer import serialize_proof_state
from harness.reasoners.hard_handoff_qwen import DeterministicProofStateReasoner, HardHandoffReasoner, hard_handoff_signature_is_image_free
from harness.referents import GroundingQuery, Provenance, Provenanced, ReferentSpec, resolve_question
from harness.views.focus import focus_view
from harness.views.remove import remove_view


class RefactoredArchitectureTests(unittest.TestCase):
    def test_candidate_logging_does_not_change_legacy_selection(self):
        class FakeGrounder:
            score_semantics = "test"
            last_candidate_trace = {"bundled_postprocessor_candidate_count": 3}

            def ground(self, image_path, query):
                # All scores are unsupported, so the frozen selector chooses
                # the smallest x0.  Input order deliberately differs.
                return [
                    RegionPrediction.from_pixels([6, 1, 9, 5], 10, 10, score=None, label="object", source="test"),
                    RegionPrediction.from_pixels([1, 1, 4, 5], 10, 10, score=None, label="object", source="test"),
                    RegionPrediction.from_pixels([3, 1, 5, 5], 10, 10, score=None, label="object", source="test"),
                ]

        question = "What color is the object?\nAnswer with only the color."
        resolved = resolve_question(question, "I2C_A")
        dag = build_predicate_dag(question, "I2C_A", resolved)
        registry = ProcessorRegistry()
        grounder = FakeGrounder()
        registry.register(
            "test_grounder",
            ProcessorCapability(
                "grounding", ("image", "GroundingQuery"), "list[RegionPrediction]",
                "unsupported", "test-v1", None, True, False,
            ),
            grounder,
        )
        manager = EntityManager()
        router = AcquisitionRouter(registry, manager)
        router.acquire_regions(Path("unused.png"), dag, resolved.specs, grounding_processor="test_grounder")

        selected = manager.get("A").region
        self.assertEqual(selected.bbox_xyxy_pixels, (1.0, 1.0, 4.0, 5.0))
        outputs = router.trace[0].outputs
        self.assertEqual(outputs["selected_candidate_index"], 1)
        self.assertEqual(len(outputs["candidates"]), 3)
        self.assertEqual(outputs["selected_region"], outputs["candidates"][1])
        self.assertEqual(outputs["candidate_pipeline_trace"], grounder.last_candidate_trace)

    def test_visible_rel2c_referents_and_dag(self):
        question = "Between the racket and the shirt,\nwhat color is the one on the left?\nAnswer with only the color."
        resolved = resolve_question(question, "REL2C_LEFT")
        self.assertEqual(resolved.status.value, "RESOLVABLE_FROM_TEXT")
        self.assertEqual([spec.category_constraint.value for spec in resolved.specs], ["racket", "shirt"])
        dag = build_predicate_dag(question, "REL2C_LEFT", resolved)
        self.assertIn(Operation.SELECT_RELATION, [node.operation for node in dag.nodes])
        self.assertIn(Operation.READ_COLOR, [node.operation for node in dag.nodes])
        self.assertNotIn("white", str(dag.to_dict()).casefold())

    def test_main_config_is_not_legacy_route(self):
        config = Path("/mnt/d/vlm_repair_v2/harness/configs/main_method.yaml").read_text(encoding="utf-8")
        self.assertIn("method_id: MAIN_HARD_HANDOFF", config)
        self.assertIn("legacy_route_enabled: false", config)
        self.assertIn("final_reasoner_accepts_image: false", config)

    def test_logical_symbol_grounding_is_rejected(self):
        with self.assertRaises(AssertionError):
            GroundingQuery("A", "A", (), ("A",), {"A": (Provenance.QUESTION_TEXT,)}, "test")
        with self.assertRaises(AssertionError):
            ReferentSpec("A", Provenanced("hidden", Provenance.EVALUATOR_ONLY, "gt category"))

    def test_persistent_entity_binding_and_controlled_reresolution(self):
        spec = ReferentSpec("A", Provenanced("racket", Provenance.QUESTION_TEXT, "visible noun"), Provenanced("racket", Provenance.QUESTION_TEXT, "visible noun"))
        query = GroundingQuery.from_spec(spec)
        manager = EntityManager()
        first = manager.ensure_entity("A", query)
        second = manager.ensure_entity("A", query)
        self.assertEqual(first.entity_id, second.entity_id)
        region = RegionPrediction.from_pixels([1, 1, 5, 5], 10, 10, score=0.8, label="racket", source="predicted_query")
        manager.bind_region("A", region)
        with self.assertRaises(RuntimeError):
            manager.bind_region("A", region)
        manager.bind_region("A", region, reason="controlled diagnostic re-resolution")
        self.assertTrue(any(event.event == "CONTROLLED_RERESOLUTION" for event in manager.events))

    def test_specialist_observation_cannot_be_final_answer(self):
        score = EvidenceScore("test", "probability", 0.8, 0.8, claimed_value="red")
        with self.assertRaises(AssertionError):
            Observation("E_0", "OUTPUT", "red", score, "test", ("TOOL_OBSERVATION",))
        capability = ProcessorCapability("color", ("region/mask",), "Observation", "confidence", "test-v1", None, True, True)
        self.assertEqual(capability.primitive, "color")
        with self.assertRaises(AssertionError):
            ProcessorCapability("color", ("region/mask",), "final_answer", "confidence", "test-v1", None, True, True)

    def test_counterfactual_status_is_uncalibrated_without_dev_labels(self):
        certifier = EvidenceCertifier(CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC", minimum_base_score=0.4, tau_remove=0.1))
        contract = {"processor": "test", "supports_certification_score": True, "supports_same_claim_rescoring": True, "comparable_across_counterfactual_views": True, "score_semantics": "unit-test score", "score_normalization": "test_norm_v1"}
        raw = EvidenceScore("test", "probability", 0.8, 0.8, claimed_value="red", normalizer_revision="test_norm_v1", view_id="RAW")
        focus = EvidenceScore("test", "probability", 0.9, 0.9, claimed_value="red", normalizer_revision="test_norm_v1", view_id="FOCUS")
        remove = EvidenceScore("test", "probability", 0.1, 0.1, claimed_value="red", normalizer_revision="test_norm_v1", view_id="REMOVE")
        certificate = certifier.certify(entity_id="E_0", predicate="COLOR", proposed_value="red", binding_kind="entity_fact", raw=raw, focus=focus, remove=remove, score_contract=contract)
        self.assertEqual(certificate.status, CertificateStatus.UNCALIBRATED)
        self.assertFalse(certificate.calibrated)

    def test_proof_state_is_small_and_answer_free(self):
        spec_a = ReferentSpec("A", Provenanced("racket", Provenance.QUESTION_TEXT, "visible noun"), Provenanced("racket", Provenance.QUESTION_TEXT, "visible noun"))
        spec_b = ReferentSpec("B", Provenanced("shirt", Provenance.QUESTION_TEXT, "visible noun"), Provenanced("shirt", Provenance.QUESTION_TEXT, "visible noun"))
        query_a, query_b = GroundingQuery.from_spec(spec_a), GroundingQuery.from_spec(spec_b)
        manager = EntityManager()
        a, b = manager.ensure_entity("A", query_a), manager.ensure_entity("B", query_b)
        region_a = RegionPrediction.from_pixels([1, 1, 3, 3], 10, 10, score=0.8, label="racket", source="predicted")
        region_b = RegionPrediction.from_pixels([7, 1, 9, 3], 10, 10, score=0.8, label="shirt", source="predicted")
        manager.bind_region("A", region_a)
        manager.bind_region("B", region_b)
        question = "Between the racket and the shirt,\nwhat color is the one on the left?\nAnswer with only the color."
        dag = build_predicate_dag(question, "REL2C_LEFT", resolve_question(question, "REL2C_LEFT"))
        certifier = EvidenceCertifier(CertificationConfig())
        contract = {"processor": "test", "supports_certification_score": True, "supports_same_claim_rescoring": True, "comparable_across_counterfactual_views": True, "score_semantics": "unit-test score", "score_normalization": "test_norm_v1"}
        def scores(claimed):
            return (
                EvidenceScore("test", "probability", 0.8, 0.8, claimed_value=claimed, normalizer_revision="test_norm_v1", view_id="RAW"),
                EvidenceScore("test", "probability", 0.9, 0.9, claimed_value=claimed, normalizer_revision="test_norm_v1", view_id="FOCUS"),
                EvidenceScore("test", "probability", 0.1, 0.1, claimed_value=claimed, normalizer_revision="test_norm_v1", view_id="REMOVE"),
            )
        raw_a, focus_a, remove_a = scores(region_a)
        raw_b, focus_b, remove_b = scores(region_b)
        cert_a = certifier.certify(entity_id=a.entity_id, predicate="REFERENT_TO_REGION", proposed_value=region_a, binding_kind="entity_binding", raw=raw_a, focus=focus_a, remove=remove_a, dag_node_id="certify_region_A", dependency_ids=("localize_A",), score_contract=contract)
        cert_b = certifier.certify(entity_id=b.entity_id, predicate="REFERENT_TO_REGION", proposed_value=region_b, binding_kind="entity_binding", raw=raw_b, focus=focus_b, remove=remove_b, dag_node_id="certify_region_B", dependency_ids=("localize_B",), score_contract=contract)
        raw_c, focus_c, remove_c = scores("red")
        derived = DerivedFact("relation", "HORIZONTAL_RELATION", "left", (a.entity_id, b.entity_id), (cert_a.certificate_id, cert_b.certificate_id), "normalized_bbox_center_x_order_v1", ("PREVIOUS_CERTIFIED_FACT",), "compare_horizontal", (cert_a.certificate_id, cert_b.certificate_id))
        selection = DerivedFact("selection", "SELECT_RELATION", {"logical_ref": "A"}, (a.entity_id, b.entity_id), (cert_a.certificate_id, cert_b.certificate_id), "deterministic_center_side_selection_v1", ("PREVIOUS_CERTIFIED_FACT",), "select_target", ("relation",))
        cert_color = certifier.certify(entity_id=a.entity_id, predicate="COLOR", proposed_value="red", binding_kind="entity_fact", raw=raw_c, focus=focus_c, remove=remove_c, dag_node_id="certify_selected_color", dependency_ids=(cert_a.certificate_id, "selection"), score_contract=contract)
        state = compile_minimal_proof_state(dag, manager, (cert_a, cert_b, cert_color), (derived, selection), include_uncalibrated=True)
        payload = state.to_dict()
        self.assertEqual(len(payload["entities"]), 2)
        self.assertEqual(len(payload["facts"]), 3)
        self.assertTrue(state.proof_complete)
        self.assertFalse(state.certified_proof_complete)
        self.assertNotIn("gt_bbox", str(payload))
        self.assertNotIn("final_answer", str(payload))
        self.assertEqual(serialize_proof_state(state), serialize_proof_state(state))

    def test_hard_handoff_has_no_image_argument(self):
        self.assertTrue(hard_handoff_signature_is_image_free())
        self.assertNotIn("image_path", inspect.signature(HardHandoffReasoner.generate).parameters)
        with self.assertRaises(AssertionError):
            assert_no_image_payload({"image_tensor": b"pixels"})

    def test_views_accept_predicted_regions_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (20, 20), (255, 255, 255)).save(image_path)
            predicted = RegionPrediction.from_pixels([3, 3, 10, 10], 20, 20, score=0.9, label="object", source="predicted")
            config = {"bbox_expand_frac": 0.1, "blur_radius_fraction_of_min_dimension": 0.025, "feather_radius_fraction_of_min_dimension": 0.01}
            focus = focus_view(image_path, predicted, root / "focus.png", entity_id="E_0", config=config)
            remove = remove_view(image_path, predicted, root / "remove.png", entity_id="E_0", config=config)
            self.assertEqual(focus.view_id, "FOCUS")
            self.assertEqual(remove.view_id, "REMOVE")
            self.assertNotEqual(focus.output_image_sha256, remove.output_image_sha256)
            self.assertTrue(remove.method.startswith("target_region_neutral_fill"))
            oracle = RegionPrediction.from_pixels([3, 3, 10, 10], 20, 20, score=1.0, label="object", source="oracle_gt")
            with self.assertRaises(AssertionError):
                focus_view(image_path, oracle, root / "bad.png", entity_id="E_0", config={})
            with self.assertRaises(AssertionError):
                remove_view(image_path, oracle, root / "bad_remove.png", entity_id="E_0", config={})


if __name__ == "__main__":
    unittest.main()
