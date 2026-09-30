from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT = Path("/root/Qwen3-VL")
SCRIPT_DIR = PROJECT / "experiments/realbind_phase2/scripts"
for path in (PROJECT, SCRIPT_DIR):
    if str(path) not in sys.path: sys.path.insert(0, str(path))


def score(response: str, answer_type: str, ground_truth: str, family: dict[str, Any], semantic_vocabulary: dict[str, list[str]]) -> dict[str, Any]:
    from experiments.realbind_phase2.pair_direct_v1_1_baseline import SCORING_REVISION, parse_semantic_answer
    result = parse_semantic_answer(response, answer_type, ground_truth, family, semantic_vocabulary)
    result["correct"] = bool(not result["invalid_prediction"] and result["normalized_prediction"] == ground_truth)
    result["scoring_revision"] = SCORING_REVISION
    return result
