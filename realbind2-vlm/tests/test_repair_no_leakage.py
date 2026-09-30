import pytest

from src.repair.strong_integrated.contracts import assert_final_handoff, assert_inference_safe


def test_selector_cannot_receive_ground_truth():
    with pytest.raises(AssertionError):
        assert_inference_safe({"selected_provider": "R4", "ground_truth_bbox": [0, 0, 1, 1]})


def test_attribute_cannot_receive_composition_gt():
    with pytest.raises(AssertionError):
        assert_inference_safe({"reader": "crop_qwen35", "composition_gt": "blue"})


def test_final_reasoner_is_image_free():
    assert_final_handoff({"minimal_state": "color=blue", "image_present": False})
    with pytest.raises(AssertionError):
        assert_final_handoff({"minimal_state": "color=blue", "image_path": "original.jpg"})

