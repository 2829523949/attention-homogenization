from __future__ import annotations

FORBIDDEN_RUNTIME_KEYS = {
    "ground_truth",
    "ground_truth_bbox",
    "gt_bbox",
    "composition_gt",
    "final_answer",
    "oracle_bbox",
    "true_target",
}


def assert_inference_safe(record: dict) -> None:
    hits: list[str] = []

    def visit(value, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                new_path = f"{path}.{key}" if path else str(key)
                if str(key).casefold() in FORBIDDEN_RUNTIME_KEYS:
                    hits.append(new_path)
                visit(child, new_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(record, "")
    if hits:
        raise AssertionError(f"runtime repair contains evaluator-only keys: {hits}")


def assert_final_handoff(record: dict) -> None:
    assert_inference_safe(record)
    if record.get("image_path") is not None or record.get("image_present") is True:
        raise AssertionError("Strong Integrated final reasoner must not receive an image")
    if "minimal_state" not in record:
        raise AssertionError("final handoff requires minimal_state")

