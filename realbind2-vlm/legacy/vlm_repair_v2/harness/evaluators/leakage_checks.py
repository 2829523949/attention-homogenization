"""Runtime guardrails shared by automatic inference and evaluator code."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any


FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "gt_bbox",
        "ground_truth_bbox",
        "bbox_gt",
        "oracle_bbox",
        "gt_category",
        "ground_truth_category",
        "gt_color",
        "ground_truth_color",
        "gt_relation",
        "ground_truth_relation",
        "ground_truth",
        "instance_id",
        "hidden_evaluator_id",
        "evaluator_id",
        "left_is_a",
        "right_is_a",
        "candidate_option_order",
        "target_identity",
        "target_color",
        "final_answer",
    }
)

FORBIDDEN_IMAGE_KEYS = frozenset(
    {
        "image",
        "image_path",
        "raw_image",
        "pixel_input",
        "image_tensor",
        "image_embedding",
        "visual_embedding",
        "caption_dump",
        "raw_caption",
    }
)


def _key_token(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")


def _as_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _as_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: _as_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_as_value(item) for item in value]
    if hasattr(value, "value") and value.__class__.__module__ != "builtins":
        try:
            return value.value
        except Exception:
            pass
    return value


def _walk(value: Any, path: str = "root") -> list[tuple[str, str, Any]]:
    value = _as_value(value)
    found: list[tuple[str, str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append((path, _key_token(key), item))
            found.extend(_walk(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk(item, f"{path}[{index}]"))
    return found


def assert_inference_safe(value: Any, *, context: str = "inference") -> None:
    """Reject evaluator-only names and provenance anywhere in an auto object."""

    for path, token, item in _walk(value):
        if token in FORBIDDEN_INFERENCE_KEYS or token.startswith("gt_") or token.startswith("oracle_"):
            raise AssertionError(f"GT/evaluator field leaked into {context} at {path}.{token}")
        if isinstance(item, str) and item.casefold() in {"evaluator_only", "oracle", "evaluator_gt"}:
            raise AssertionError(f"evaluator-only provenance leaked into {context} at {path}.{token}")


def assert_no_final_answer(value: Any, *, context: str = "proof state") -> None:
    for path, token, _ in _walk(value):
        if token in {"answer", "final_answer", "ground_truth", "prediction", "derived_answer"}:
            raise AssertionError(f"answer leakage into {context} at {path}.{token}")


def assert_no_image_payload(value: Any, *, context: str = "hard handoff") -> None:
    for path, token, item in _walk(value):
        if token in FORBIDDEN_IMAGE_KEYS or token.endswith("_image") or token.endswith("_image_path"):
            raise AssertionError(f"raw image payload leaked into {context} at {path}.{token}")
        if isinstance(item, (bytes, bytearray, memoryview)):
            raise AssertionError(f"binary image-like payload leaked into {context} at {path}")


def assert_d_drive_path(path: Path, *, artifact_kind: str = "artifact") -> None:
    resolved = path.expanduser().resolve()
    if not str(resolved).startswith("/mnt/d/"):
        raise AssertionError(f"new {artifact_kind} must be stored on D drive: {resolved}")

