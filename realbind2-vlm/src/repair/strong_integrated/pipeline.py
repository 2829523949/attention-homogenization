"""Dataset-scale Strong Integrated assembly and image-free final execution.

Grounding, selection, SAM3 isolation, and crop-Qwen acquisition are cached in
separate append-only files.  Assembly is deterministic and only consumes those
runtime records.  Evaluator labels are joined later by analysis code.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.common.dataset import RealBind2Dataset, read_jsonl
from src.common.logging_utils import JsonlRunLog
from src.common.manifests import stable_hash
from src.common.scorer import score_prediction
from src.models import ADAPTERS
from src.repair.strong_integrated.cache import StageCache
from src.repair.strong_integrated.contracts import assert_final_handoff, assert_inference_safe

ENTITY_CACHE_FILES = {
    "candidates": ("candidates.jsonl", ("family_id", "logical_ref")),
    "selected": ("selected_targets.jsonl", ("family_id", "logical_ref")),
    "attributes": ("attributes.jsonl", ("family_id", "logical_ref")),
}


def load_entity_caches(output_root: Path) -> dict[str, StageCache]:
    return {
        name: StageCache(output_root / filename, fields)
        for name, (filename, fields) in ENTITY_CACHE_FILES.items()
    }


def audit_cache(output_root: Path) -> dict:
    caches = load_entity_caches(output_root)
    selected = caches["selected"].rows
    attributes = caches["attributes"].rows
    errors: list[str] = []
    for key, row in selected.items():
        try:
            assert_inference_safe(row)
        except AssertionError as exc:
            errors.append(f"selected:{key}:{exc}")
        provider = str(row.get("selected_provider", ""))
        if not provider.startswith("R4_MM_GDINO_L_QWEN_SELECTOR"):
            errors.append(f"selected:{key}:primary provider is not frozen R4")
    for key, row in attributes.items():
        try:
            assert_inference_safe(row)
        except AssertionError as exc:
            errors.append(f"attribute:{key}:{exc}")
        if row.get("reader") != "crop_qwen35":
            errors.append(f"attribute:{key}:reader is not crop_qwen35")
    return {
        "passed": not errors,
        "errors": errors,
        "rows": {name: len(cache.rows) for name, cache in caches.items()},
        "final_repair": "Strong Integrated",
        "sam3_consolidation_is_final": False,
    }


def assemble_states(dataset_root: Path, output_root: Path) -> int:
    dataset = RealBind2Dataset(dataset_root)
    families = {str(row["family_id"]): row for row in dataset.families()}
    questions = dataset.questions()
    caches = load_entity_caches(output_root)
    selected, attributes = caches["selected"].rows, caches["attributes"].rows
    states = StageCache(output_root / "minimal_states.jsonl", ("qa_id",))
    for qa in questions:
        if qa["qa_type"] not in {"REL2C_LEFT", "REL2C_RIGHT"}:
            continue
        family_id = str(qa["family_id"])
        family = families[family_id]
        a = selected[(family_id, "A")]
        b = selected[(family_id, "B")]
        ax = sum(float(x) for x in a["bbox_xyxy"][0::2]) / 2.0
        bx = sum(float(x) for x in b["bbox_xyxy"][0::2]) / 2.0
        left_ref, right_ref = ("A", "B") if ax < bx else ("B", "A")
        target = left_ref if qa["qa_type"].endswith("LEFT") else right_ref
        attribute = attributes[(family_id, target)]
        record = {
            "qa_id": str(qa["qa_id"]),
            "case_id": str(qa["qa_id"]),
            "family_id": family_id,
            "qa_type": str(qa["qa_type"]),
            "question": str(qa["question"]),
            "logical_target": target,
            "minimal_state": f"[TARGET_ENTITY] {target}\n[TARGET_ATTRIBUTE] color = {attribute['color']}",
            "image_present": False,
            "state_source": "MM-GDINO-L -> Qwen selector -> SAM3 isolation -> crop-Qwen",
            "selected_target_hash": stable_hash(selected[(family_id, target)]),
            "attribute_hash": stable_hash(attribute),
            "protocol": "strong_integrated_v1",
        }
        assert_final_handoff(record)
        states.append(record)
    return len(states.rows)


def run_final(dataset_root: Path, output_root: Path, model_key: str, model_path: Path, runtime: dict) -> int:
    dataset = RealBind2Dataset(dataset_root)
    qa = {str(row["qa_id"]): row for row in dataset.questions()}
    states = list(read_jsonl(output_root / "minimal_states.jsonl"))
    adapter = ADAPTERS[model_key](model_path, **runtime)
    log = JsonlRunLog(output_root / f"final_{model_key}.jsonl")
    config_hash = stable_hash({"model": adapter.get_model_metadata(), "protocol": "strong_integrated_v1"})
    for state in states:
        case_id = str(state["qa_id"])
        if log.is_complete(case_id):
            continue
        assert_final_handoff(state)
        prompt = state["question"] + "\n\nAcquired visual facts:\n" + state["minimal_state"]
        started = time.monotonic()
        raw = adapter.generate(adapter.prepare_prompt(prompt, image_path=None))
        scored = score_prediction(raw, qa[case_id]["ground_truth"], qa[case_id]["answer_type"])
        log.append({
            "case_id": case_id,
            "family_id": state["family_id"],
            "image_id": None,
            "model": model_key,
            "experiment": "strong_integrated_final",
            "condition": "text_only_minimal_state",
            "prediction_raw": raw,
            "prediction_normalized": scored.normalized,
            "ground_truth": scored.ground_truth,
            "correct": scored.correct,
            "valid": scored.valid,
            "runtime_sec": time.monotonic() - started,
            "status": "ok",
            "config_hash": config_hash,
            "dataset_hash": stable_hash({"qa_count": len(qa)}),
            "image_present": False,
        })
    return len(log.completed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("audit-cache", "assemble", "final"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", choices=tuple(ADAPTERS))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "audit-cache":
        result = audit_cache(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if args.stage == "assemble":
        print(f"ASSEMBLED {assemble_states(args.dataset_root, args.output_root)}")
        return 0
    if not args.model_key or not args.model_path:
        parser.error("--model-key and --model-path are required for --stage final")
    print(f"FINAL_COMPLETE {run_final(args.dataset_root, args.output_root, args.model_key, args.model_path, {'device': args.device})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
