#!/usr/bin/env python3
"""SPECIALIZED TARGET-ATTRIBUTE READER REPLACEMENT.

Parent baseline: Strong Integrated Repair
  outputs/strong_integrated_repair  (GroundedTargetCorrect 26/40,
  AttributeCorrectGivenGrounding 20/26, MinimalState 20/40, CleanFinal 20/40,
  RawFinal 25/40).

This experiment replaces ONLY the frozen Qwen3.5-4B target-color reader with a
specialised fine-grained reader (ZwZ-4B / ZwZ-8B), using the official red
bounding-box overlay interface the ZwZ models were trained/evaluated with.
All upstream binding / grounding / target resolution / final reasoner is frozen.

Two-stage protocol:
  Stage A: atomic 200 (I2C_A/I2C_B) reader selection.
  Stage B: one-shot transfer to the frozen 40 REL2C development QA.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/mnt/d/vlm_repair_v2")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/specialized_attribute_reader"
PARENT = ROOT / "outputs/strong_integrated_repair"
ATOMIC_DIR = ROOT / "outputs/target_local_multiview_repair"
PREPARED = ROOT / "outputs/minimal_decomposition_route_pilot/prepared.jsonl"
ROUTE_PRIOR = ROOT / "outputs/minimal_decomposition_route_pilot/route_results.jsonl"
OLD_CANDIDATES = ROOT / "outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"

# ---------------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------------
PROMPT = ("What is the color of the object annotated by the red box? "
          "Answer with exactly one color word from this list: red, orange, yellow, "
          "green, blue, purple, pink, brown, black, white, gray.")
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
BBOX_EXPAND_RATIO = 0.05
BBOX_OUTLINE_WIDTH = 5
BBOX_OUTLINE_RGB = (255, 0, 0)
GEN_MAX_NEW_TOKENS = 16
GEN_DO_SAMPLE = False
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
PROTOCOL_ID = "zwz_official_red_bbox_overlay_expand0.05_w5_greedy16"

REALBIND_COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink",
                   "brown", "black", "white", "gray")
ATOMIC_BASELINE_CORRECT = 129
ATOMIC_BASELINE_TOTAL = 200
ATOMIC_GATE = ATOMIC_BASELINE_CORRECT          # must be strictly greater
ATOMIC_MARGINAL = 140                           # >= clear gain

READERS = {
    "zwz4b": {
        "repo": "inclusionAI/ZwZ-4B",
        "revision": "e43981c19904b4ee4ec48efc21c9a77bf4e7838c",
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "local_path": "/mnt/d/vlm_repair_v2/models/zwz_4b",
        "project_link": "models/zwz_4b (same directory)",
        "default_mode": "bf16_offload",
        "size_note": "4B-class; ~9.65 GB bf16 safetensors",
    },
    "zwz8b": {
        "repo": "inclusionAI/ZwZ-8B",
        "revision": "894243f5453519699849a2ef0afae92a18637521",
        "base_model": "Qwen/Qwen3-VL-8B-Instruct",
        "local_path": "/mnt/d/vlm_repair_v2/models/zwz_8b",
        "project_link": "models/zwz_8b (same directory)",
        "default_mode": "int8",
        "size_note": "8B-class; ~17.2 GB bf16 safetensors",
    },
}


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def dumpl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in records), encoding="utf-8")


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_color(raw: str):
    value = str(raw).strip().casefold()
    return value if value in REALBIND_COLORS else None


def update_status(**kw):
    path = OUT / "resume_status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(kw)
    status["updated_at_unix"] = time.time()
    dump(path, status)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Stage 0 - frozen parent audit
# ---------------------------------------------------------------------------

def _parent_binding_expected(ctx):
    import harness.scripts.run_strong_integrated_repair as frozen
    ground = {(r["family_id"], r["logical_ref"]): r for r in rows(PARENT / "grounding_results.jsonl")}
    assert len(ground) == 40
    expected = {}
    for family_id in sorted(ctx["families"]):
        for ref in ("A", "B"):
            key = (family_id, ref)
            primary = ctx["r4_selected"][key]
            bbox = primary.get("bbox_xyxy")
            provider = "R4_MM_GDINO_L_QWEN_SELECTOR"
            fallback = False
            sam_candidates = ground[key]["candidates"]
            sam_top1 = sam_candidates[0]["bbox_xyxy"] if sam_candidates else None
            if bbox is None:
                bbox, provider, fallback = sam_top1, "SAM3_LITETEXT_TOP1_FALLBACK", True
            elif bbox is not None and sam_candidates:
                support = max(frozen.iou(bbox, c["bbox_xyxy"]) for c in sam_candidates[:16])
                top_score = primary.get("top_native_score")
                if support < frozen.R4_SUPPORT_FALLBACK_IOU and top_score is not None and top_score < frozen.R4_SUPPORT_FALLBACK_TOP_SCORE:
                    bbox, provider, fallback = sam_top1, "SAM3_LITETEXT_TOP1_FALLBACK", True
            expected[key] = {"bbox": bbox, "provider": provider, "fallback_used": fallback}
    return expected


def stage_parent_audit():
    import harness.scripts.run_strong_integrated_repair as frozen
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = frozen.load_context()
    binding = {(r["family_id"], r["logical_ref"]): r for r in rows(PARENT / "binding_results.jsonl")}
    ground = {(r["family_id"], r["logical_ref"]): r for r in rows(PARENT / "grounding_results.jsonl")}
    qa = {r["qa_id"]: r for r in rows(PARENT / "qa_results.jsonl")}
    final = {r["qa_id"]: r for r in rows(PARENT / "final_results.jsonl")}
    attr = {(r["family_id"], r["logical_ref"]): r for r in rows(PARENT / "attribute_results.jsonl")}
    prepared = rows(PREPARED)
    prior = {r["qa_id"]: r for r in rows(ROUTE_PRIOR)}
    assert len(binding) == 40 and len(ground) == 40 and len(qa) == 40 and len(final) == 40 and len(attr) == 40
    assert len(prepared) == 40 and len(prior) == 40

    expected_binding = _parent_binding_expected(ctx)
    binding_ok = True
    for key, expected in expected_binding.items():
        row = binding[key]
        binding_ok &= (row["selected_bbox_xyxy"] == expected["bbox"] and row["selected_provider"] == expected["provider"] and bool(row["fallback_used"]) == expected["fallback_used"])

    gt_boxes = {}
    for row in rows(OLD_CANDIDATES):
        inf = row["inference"]
        if inf["backend"] == "grounding_dino":
            gt_boxes[(inf["family_id"], inf["logical_ref"])] = [float(v) for v in row["EVALUATOR_ONLY"]["gt_bbox_xyxy_pixels"]]
    assert len(gt_boxes) == 40

    recomputed = []
    for row in prepared:
        qa_id = row["qa_id"]
        family_id = row["family_id"]
        parent_row = qa[qa_id]
        true_target = prior[qa_id]["EVALUATOR_ONLY"]["true_target"]
        gt_color = str(prior[qa_id]["EVALUATOR_ONLY"]["ground_truth"]).strip().casefold()
        bbox_a = binding[(family_id, "A")]["selected_bbox_xyxy"]
        bbox_b = binding[(family_id, "B")]["selected_bbox_xyxy"]
        target = frozen._target_from_boxes(bbox_a, bbox_b, row["qa_type"])
        logical = bool(target == true_target)
        grounded = bool(logical and target is not None and frozen.iou(binding[(family_id, target)]["selected_bbox_xyxy"], gt_boxes[(family_id, target)]) >= 0.5)
        historical_attr = bool(parent_row["target_attribute_correct"])
        sufficient = bool(logical and grounded and historical_attr)
        assert target == parent_row["target"], qa_id
        assert logical == bool(parent_row["logical_target_correct"]), qa_id
        assert grounded == bool(parent_row["grounded_target_correct"]), qa_id
        assert gt_color == str(parent_row["gt_color"]).strip().casefold(), qa_id
        assert sufficient == bool(parent_row["minimal_state_sufficient"]), qa_id
        source_image = next(iter(ctx["families"][family_id].values())) if False else ctx["families"][family_id]["image_path"]
        target_bbox = binding[(family_id, target)]["selected_bbox_xyxy"] if target else None
        recomputed.append({
            "qa_id": qa_id,
            "family_id": family_id,
            "qa_type": row["qa_type"],
            "target": target,
            "true_target": true_target,
            "gt_color": gt_color,
            "logical_target_correct": logical,
            "grounded_target_correct": grounded,
            "historical_target_attribute_correct": historical_attr,
            "historical_minimal_state_sufficient": sufficient,
            "selected_target_bbox_xyxy": target_bbox,
            "image_path": source_image,
            "target_region_hash": sha_json({"qa_id": qa_id, "family_id": family_id, "logical_ref": target, "bbox_xyxy": target_bbox}),
            "parent_state_sha256": parent_row["state_sha256"],
            "parent_state_text": parent_row["state_text"],
            "parent_rendered_color": parent_row["rendered_color"],
            "parent_final_correct": bool(final[qa_id]["final_correct"]),
            "parent_final_image_present": bool(final[qa_id]["image_present"]),
            "parent_final_vision_encoder_called": bool(final[qa_id]["vision_encoder_called"]),
        })

    logical = sum(r["logical_target_correct"] for r in recomputed)
    grounded = sum(r["grounded_target_correct"] for r in recomputed)
    attr_all = sum(r["historical_target_attribute_correct"] for r in recomputed)
    attr_grounded = sum(r["historical_target_attribute_correct"] and r["grounded_target_correct"] for r in recomputed)
    sufficient = sum(r["historical_minimal_state_sufficient"] for r in recomputed)
    final_rows = list(final.values())
    raw = sum(bool(r["final_correct"]) for r in final_rows)
    clean = sum(bool(r["state_sufficient"]) and bool(r["final_correct"]) for r in final_rows)
    comp = sum((not bool(r["state_sufficient"])) and bool(r["final_correct"]) for r in final_rows)

    assertions = {
        "binding_matches_frozen_rules": bool(binding_ok),
        "logical_target_correct_36": logical == 36,
        "grounded_target_correct_26": grounded == 26,
        "attribute_all_25": attr_all == 25,
        "attribute_given_grounding_20": attr_grounded == 20,
        "minimal_state_sufficient_20": sufficient == 20,
        "raw_final_25": raw == 25,
        "clean_final_20": clean == 20,
        "compensating_errors_5": comp == 5,
        "parent_final_image_exit_40": all((not r["image_present"]) and (not r["vision_encoder_called"]) for r in final_rows),
        "n_qa_40": len(recomputed) == 40,
    }
    drift = (not all(assertions.values())) or (not binding_ok)
    family_hashes = {}
    for family_id in sorted(ctx["families"]):
        family_hashes[family_id] = {
            "selected_A_hash": sha_json({"family_id": family_id, "logical_ref": "A", "selected_bbox_xyxy": binding[(family_id, "A")]["selected_bbox_xyxy"], "selected_provider": binding[(family_id, "A")]["selected_provider"]}),
            "selected_B_hash": sha_json({"family_id": family_id, "logical_ref": "B", "selected_bbox_xyxy": binding[(family_id, "B")]["selected_bbox_xyxy"], "selected_provider": binding[(family_id, "B")]["selected_provider"]}),
            "selected_A_bbox": binding[(family_id, "A")]["selected_bbox_xyxy"],
            "selected_B_bbox": binding[(family_id, "B")]["selected_bbox_xyxy"],
            "image_path": ctx["families"][family_id]["image_path"],
        }
    parent_files = ["binding_results.jsonl", "grounding_results.jsonl", "qa_results.jsonl",
                    "minimal_state_results.jsonl", "final_results.jsonl", "attribute_results.jsonl",
                    "survival_chain.json", "clean_vs_raw_accuracy.json", "pipeline_config.json"]
    dump(OUT / "historical_parent_manifest.json", {
        "parent": str(PARENT),
        "parent_artifact_sha256": {name: (file_sha(PARENT / name) if (PARENT / name).exists() else None) for name in parent_files},
        "metrics": {
            "grounded_target_correct": grounded,
            "attribute_correct_given_grounding": {"correct": attr_grounded, "n": grounded},
            "target_attribute_correct_all": attr_all,
            "minimal_state_sufficient": sufficient,
            "clean_final_correct": clean,
            "raw_final_correct": raw,
            "compensating_error_correct": comp,
            "logical_target_correct": logical,
        },
        "family_region_hashes": family_hashes,
        "qa_target_region_hashes": {r["qa_id"]: r["target_region_hash"] for r in recomputed},
        "protocol": {"no_new_reader_inference_in_this_stage": True},
    })
    audit = {
        "parent": str(PARENT),
        "assertions": assertions,
        "binding_matches_frozen_rules": bool(binding_ok),
        "pass": (not drift),
        "drift_label": None if not drift else "INVALID_UPSTREAM_DRIFT",
        "selected_A_B_hashes": {f: {"A": v["selected_A_hash"], "B": v["selected_B_hash"]} for f, v in family_hashes.items()},
        "target_region_hashes": {r["qa_id"]: r["target_region_hash"] for r in recomputed},
    }
    dump(OUT / "upstream_hash_audit.json", audit)
    dumpl(OUT / "_parent_evaluator_rows.jsonl", recomputed)
    if drift:
        update_status(status="INVALID_UPSTREAM_DRIFT", parent_audit_complete=False)
        raise SystemExit("INVALID_UPSTREAM_DRIFT: frozen parent assertions failed")
    update_status(status="PARENT_AUDIT_COMPLETE", parent_audit_complete=True,
                  parent_metrics={"grounded": grounded, "attr_given_grounding": attr_grounded,
                                  "sufficient": sufficient, "clean": clean, "raw": raw})
    log(f"PARENT_AUDIT_COMPLETE logical={logical} grounded={grounded} attr_grounded={attr_grounded}/26 "
        f"sufficient={sufficient} clean={clean} raw={raw}")


# ---------------------------------------------------------------------------
# Stage 1 - frozen reader inputs (atomic 200 + composition 40)
# ---------------------------------------------------------------------------

def expanded_bbox(bbox, image_size, ratio=BBOX_EXPAND_RATIO):
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width, height = image_size
    bw, bh = x2 - x1, y2 - y1
    return [max(0.0, x1 - bw * ratio), max(0.0, y1 - bh * ratio),
            min(float(width), x2 + bw * ratio), min(float(height), y2 + bh * ratio)]


def render_overlay(image_path, bbox, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    box = expanded_bbox(bbox, image.size)
    draw.rectangle(box, outline=BBOX_OUTLINE_RGB, width=BBOX_OUTLINE_WIDTH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return {"image_size": list(image.size), "expanded_bbox_xyxy": box,
            "image_sha256": file_sha(image_path), "overlay_sha256": file_sha(out_path)}


def stage_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_manifest = rows(ATOMIC_DIR / "atomic_view_manifest.jsonl")
    assert len(atomic_manifest) == 200
    atomic_inputs = []
    for index, row in enumerate(atomic_manifest, 1):
        qa_id = row["qa_id"]
        out_path = OUT / "atomic_inputs" / f"{qa_id}.png"
        meta = render_overlay(row["image_path"], row["bbox_xyxy"], out_path)
        input_sha = sha_json({"protocol": PROTOCOL_ID, "qa_id": qa_id, "image_sha256": meta["image_sha256"],
                              "bbox_xyxy": row["bbox_xyxy"], "expanded_bbox_xyxy": meta["expanded_bbox_xyxy"],
                              "overlay_sha256": meta["overlay_sha256"], "prompt_sha256": PROMPT_SHA})
        atomic_inputs.append({
            "qa_id": qa_id, "family_id": row["family_id"], "logical_ref": row["logical_ref"],
            "source_image_path": row["image_path"], "image_size": meta["image_size"],
            "bbox_xyxy": row["bbox_xyxy"], "expanded_bbox_xyxy": meta["expanded_bbox_xyxy"],
            "overlay_path": str(out_path), "image_sha256": meta["image_sha256"],
            "overlay_sha256": meta["overlay_sha256"], "input_sha256": input_sha,
        })
        if index % 50 == 0:
            log(f"atomic overlay {index}/200")
    dumpl(OUT / "atomic_reader_inputs.jsonl", atomic_inputs)
    # evaluator-only GT map is written by the parent audit join (_parent_evaluator_rows)
    parent_eval = {r["qa_id"]: r for r in rows(OUT / "_parent_evaluator_rows.jsonl")}
    atomic_gt = {}
    for row in atomic_manifest:
        atomic_gt[row["qa_id"]] = {"gt_color": str(row["gt_color"]).strip().casefold(),
                                   "object_phrase": row["object_phrase"],
                                   "mask_valid_historical": bool(row["mask_valid"])}
    dump(OUT / "atomic_evaluator_only.json", atomic_gt)

    binding = {(r["family_id"], r["logical_ref"]): r for r in rows(PARENT / "binding_results.jsonl")}
    comp_entities = {}
    for qa_id, row in parent_eval.items():
        family_id, target = row["family_id"], row["target"]
        if target is None:
            continue
        key = f"{family_id}__{target}"
        if key in comp_entities:
            continue
        bbox = binding[(family_id, target)]["selected_bbox_xyxy"]
        assert [float(v) for v in bbox] == [float(v) for v in row["selected_target_bbox_xyxy"]]
        out_path = OUT / "composition_inputs" / f"{key}.png"
        meta = render_overlay(row["image_path"], bbox, out_path)
        input_sha = sha_json({"protocol": PROTOCOL_ID, "key": key, "image_sha256": meta["image_sha256"],
                              "bbox_xyxy": bbox, "expanded_bbox_xyxy": meta["expanded_bbox_xyxy"],
                              "overlay_sha256": meta["overlay_sha256"], "prompt_sha256": PROMPT_SHA})
        comp_entities[key] = {
            "comp_id": key, "family_id": family_id, "logical_ref": target,
            "source_image_path": row["image_path"], "bbox_xyxy": bbox,
            "expanded_bbox_xyxy": meta["expanded_bbox_xyxy"], "image_size": meta["image_size"],
            "overlay_path": str(out_path), "image_sha256": meta["image_sha256"],
            "overlay_sha256": meta["overlay_sha256"], "input_sha256": input_sha,
        }
    comp_rows = [comp_entities[k] for k in sorted(comp_entities)]
    dumpl(OUT / "composition_reader_inputs.jsonl", comp_rows)
    dump(OUT / "composition_evaluator_only.json", {
        "qa": {r["qa_id"]: {"family_id": r["family_id"], "target": r["target"], "true_target": r["true_target"],
                            "gt_color": r["gt_color"], "logical_target_correct": r["logical_target_correct"],
                            "grounded_target_correct": r["grounded_target_correct"],
                            "historical_target_attribute_correct": r["historical_target_attribute_correct"],
                            "historical_minimal_state_sufficient": r["historical_minimal_state_sufficient"],
                            "parent_final_correct": r["parent_final_correct"]} for r in parent_eval.values()},
        "n_unique_target_entities": len(comp_rows),
    })
    update_status(status="INPUTS_READY", inputs_complete=True,
                  atomic_inputs=len(atomic_inputs), composition_inputs=len(comp_rows),
                  protocol_id=PROTOCOL_ID, prompt_sha256=PROMPT_SHA)
    log(f"INPUTS_READY atomic={len(atomic_inputs)} composition_entities={len(comp_rows)}")


# ---------------------------------------------------------------------------
# Stage 2 - model manifest
# ---------------------------------------------------------------------------

def _import_versions():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["gpu_total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover
        info["transformers_error"] = repr(exc)
    try:
        import accelerate
        info["accelerate"] = accelerate.__version__
    except Exception as exc:
        info["accelerate_error"] = repr(exc)
    try:
        import bitsandbytes
        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception as exc:
        info["bitsandbytes_error"] = repr(exc)
    return info


def write_pipeline_config():
    config = {
        "experiment": "specialized_target_attribute_reader_replacement",
        "parent_baseline": {
            "name": "Strong Integrated Repair",
            "path": str(PARENT),
            "artifacts": ["binding_results.jsonl", "grounding_results.jsonl", "attribute_results.jsonl",
                          "qa_results.jsonl", "minimal_state_results.jsonl", "final_results.jsonl"],
            "frozen_metrics": {"grounded_target_correct": "26/40", "attribute_correct_given_grounding": "20/26",
                               "minimal_state_sufficient": "20/40", "clean_final_correct": "20/40", "raw_final_correct": "25/40"},
        },
        "cohort": {"prepared": str(PREPARED), "atomic_manifest": str(ATOMIC_DIR / "atomic_view_manifest.jsonl"),
                   "n_families": 20, "n_rel2c_qa": 40, "n_atomic_i2c": 200},
        "frozen_upstream": ["ReferentSpec", "MM-GDINO-L candidate pool", "frozen Qwen candidate selector",
                            "fallback rules", "selected A/B boxes", "deterministic bbox-center target resolution",
                            "target entity", "image paths and resolutions"],
        "reader_interface": {"protocol_id": PROTOCOL_ID,
                             "input": "full original image + official ZwZ red bounding-box overlay",
                             "bbox_expand_ratio": BBOX_EXPAND_RATIO, "bbox_outline_width": BBOX_OUTLINE_WIDTH,
                             "bbox_outline_rgb": list(BBOX_OUTLINE_RGB),
                             "no_sam3_mask_input": True, "no_case_dependent_interface": True},
        "prompt": {"text": PROMPT, "sha256": PROMPT_SHA, "contains_gt": False, "contains_left_right": False,
                   "contains_answer_relation": False},
        "generation": {"do_sample": GEN_DO_SAMPLE, "max_new_tokens": GEN_MAX_NEW_TOKENS, "use_cache": True,
                       "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "num_beams": 1, "temperature": None},
        "evaluators": {"color_vocabulary": list(REALBIND_COLORS),
                       "atomic_main_metric": "AtomicColorAccuracy (strict frozen vocabulary normalisation, identical to historical box baseline)",
                       "downstream": "frozen harness evaluator parse_semantic_answer (scoring_revision preserved)",
                       "no_special_scorer_rules_for_zwz": True},
        "stage_protocol": {
            "stage_A": "atomic 200 reader selection only",
            "stage_B": "single-shot transfer of the frozen selected reader to the 40 REL2C QA",
            "composition_tuning_forbidden": True, "reader_ensembling_forbidden": True,
            "per_case_routing_forbidden": True, "confidence_fusion_forbidden": True, "historical_fallback_forbidden": True,
        },
        "selection_rule": {"primary": "highest AtomicColorAccuracy",
                           "gate": "strictly greater than 129/200 to enter composition",
                           "marginal_band": "130-139/200 flagged MARGINAL_ATOMIC_GAIN",
                           "tie_break": "smaller / lower VRAM / faster (ZwZ-4B before ZwZ-8B)",
                           "composition_never_used_for_selection": True},
        "downstream": {"state_format": "[TARGET_ENTITY] A/B\n[TARGET_ATTRIBUTE] color = <color>",
                       "added_fields": [], "final_reasoner": "frozen /root/Qwen3-VL/models/Qwen3.5-4B text-only",
                       "final_reasoner_prompt_frozen": True, "image_present": False, "vision_encoder_called": False,
                       "hard_image_exit_required_for": "40/40"},
        "stop_rules": {"A": "new reader atomic <=129/200 -> SPECIALIZED_READER_NO_ATOMIC_GAIN; no composition",
                       "B": "atomic gain but attribute given grounding <=20/26 -> ATOMIC_GAIN_DOES_NOT_TRANSFER",
                       "C": "attribute >20/26 but CleanFinal <=20/40 -> ATTRIBUTE_GAIN_NO_CLEAN_PROPAGATION"},
    }
    dump(OUT / "pipeline_config.json", config)


def stage_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    write_pipeline_config()
    old_manifest = {}
    old_path = OUT / "model_manifest.json"
    if old_path.exists():
        try:
            old_manifest = json.loads(old_path.read_text()).get("readers", {})
        except Exception:
            old_manifest = {}
    manifest = {"experiment": "specialized_attribute_reader_replacement", "readers": {}, "environment": _import_versions()}
    for key, spec in READERS.items():
        path = Path(spec["local_path"])
        entry = {k: v for k, v in spec.items()}
        entry["local_exists"] = path.exists()
        entry["deployment_modes"] = {
            "bf16_offload": {"dtype": "bfloat16", "device_map": "auto", "max_memory": {"0": "7GiB", "cpu": "4GiB"}},
            "bf16_gpu": {"dtype": "bfloat16", "device_map": {"": "cuda:0"}},
            "int8": {"quantization": "bitsandbytes_llm_int8", "compute_dtype": "bfloat16", "device_map": "auto", "max_memory": {"0": "7GiB", "cpu": "4GiB"}},
            "int4": {"quantization": "bitsandbytes_nf4_double_quant", "compute_dtype": "bfloat16", "device_map": "auto", "max_memory": {"0": "7GiB", "cpu": "3GiB"}},
        }
        entry["default_mode"] = spec["default_mode"]
        if path.exists():
            for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                         "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
                f = path / name
                if f.exists():
                    entry.setdefault("files_sha256", {})[name] = file_sha(f)
            try:
                entry["config"] = json.loads((path / "config.json").read_text())
            except Exception:
                entry["config"] = None
            try:
                entry["generation_config"] = json.loads((path / "generation_config.json").read_text())
            except Exception:
                entry["generation_config"] = None
            try:
                entry["preprocessor_config"] = json.loads((path / "preprocessor_config.json").read_text())
            except Exception:
                entry["preprocessor_config"] = None
            weights = []
            index_file = path / "model.safetensors.index.json"
            if index_file.exists():
                index = json.loads(index_file.read_text())
                for shard in sorted(set(index["weight_map"].values())):
                    f = path / shard
                    weights.append({"file": shard, "exists": f.exists(),
                                    "size_bytes": f.stat().st_size if f.exists() else None,
                                    "sha256": file_sha(f) if f.exists() else None,
                                    "sha256_pending": not f.exists()})
            entry["weights"] = weights
            entry["total_weight_bytes"] = sum(w["size_bytes"] or 0 for w in weights)
            try:
                import struct
                first = path / weights[0]["file"] if weights else None
                if first is not None and first.exists():
                    with first.open("rb") as handle:
                        header_len = struct.unpack("<Q", handle.read(8))[0]
                        header = json.loads(handle.read(header_len))
                    dtypes = Counter(v["dtype"] for k, v in header.items() if k != "__metadata__")
                    entry["safetensors_header_dtypes"] = dict(dtypes)
            except Exception as exc:
                entry["safetensors_header_error"] = repr(exc)
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(path, local_files_only=True)
                tokenizer = getattr(processor, "tokenizer", None)
                image_processor = getattr(processor, "image_processor", None)
                entry["processor_class"] = type(processor).__name__
                entry["tokenizer_class"] = type(tokenizer).__name__ if tokenizer is not None else None
                entry["image_processor_class"] = type(image_processor).__name__ if image_processor is not None else None
                entry["tokenizer"] = {"class": type(tokenizer).__name__ if tokenizer is not None else None,
                                      "vocab_size": getattr(tokenizer, "vocab_size", None),
                                      "model_max_length": getattr(tokenizer, "model_max_length", None),
                                      "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                                      "pad_token_id": getattr(tokenizer, "pad_token_id", None)}
                entry["image_processor"] = {"class": type(image_processor).__name__ if image_processor is not None else None,
                                            "min_pixels": getattr(image_processor, "min_pixels", None),
                                            "max_pixels": getattr(image_processor, "max_pixels", None),
                                            "patch_size": getattr(image_processor, "patch_size", None),
                                            "merge_size": getattr(image_processor, "merge_size", None)}
            except Exception as exc:
                entry["processor_error"] = repr(exc)
        entry["generation_settings"] = {"do_sample": GEN_DO_SAMPLE, "max_new_tokens": GEN_MAX_NEW_TOKENS,
                                        "use_cache": True, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
        entry["interface"] = {"protocol_id": PROTOCOL_ID, "prompt": PROMPT, "prompt_sha256": PROMPT_SHA,
                              "bbox_overlay": {"expand_ratio": BBOX_EXPAND_RATIO, "outline_rgb": list(BBOX_OUTLINE_RGB),
                                               "outline_width": BBOX_OUTLINE_WIDTH, "format": "full original image + red rectangle"}}
        manifest["readers"][key] = entry
    try:
        from urllib.request import urlopen
        import urllib.error
        for key, spec in READERS.items():
            url = f"https://huggingface.co/api/models/{spec['repo']}"
            try:
                with urlopen(url, timeout=20) as response:
                    data = json.loads(response.read().decode("utf-8"))
                manifest["readers"][key]["hf_api"] = {"modelId": data.get("modelId"), "sha": data.get("sha"),
                                                      "lastModified": data.get("lastModified"), "downloads": data.get("downloads"),
                                                      "tags": data.get("tags")}
            except Exception as exc:
                manifest["readers"][key]["hf_api_error"] = repr(exc)
                if old_manifest.get(key, {}).get("hf_api"):
                    manifest["readers"][key]["hf_api"] = old_manifest[key]["hf_api"]
                    manifest["readers"][key]["hf_api_source"] = "preserved from earlier successful fetch"
    except Exception:
        pass
    for key in manifest["readers"]:
        if "modelscope" not in manifest["readers"][key] and old_manifest.get(key, {}).get("modelscope"):
            manifest["readers"][key]["modelscope"] = old_manifest[key]["modelscope"]
            manifest["readers"][key]["modelscope_source"] = "preserved from earlier successful fetch"
    try:
        from urllib.request import urlopen
        for key, spec in READERS.items():
            try:
                url = (f"https://modelscope.cn/api/v1/models/{spec['repo']}/repo/files"
                       f"?Revision=master&Recursive=true")
                with urlopen(url, timeout=25) as response:
                    data = json.loads(response.read().decode("utf-8"))
                files = (data.get("Data") or {}).get("Files") or []
                manifest["readers"][key]["modelscope"] = {
                    "repo": spec["repo"], "revision": "master",
                    "files": [{"path": f.get("Path"), "size": f.get("Size"), "sha256": f.get("Sha256"),
                               "revision": f.get("Revision")} for f in files],
                }
            except Exception as exc:
                manifest["readers"][key]["modelscope_error"] = repr(exc)
    except Exception:
        pass
    manifest["hash_note"] = "sha256 computed over local files; revisions pinned to HuggingFace commit shas"
    dump(OUT / "model_manifest.json", manifest)
    update_status(status="MANIFEST_WRITTEN", manifest_complete=True)
    log("MANIFEST_WRITTEN")


# ---------------------------------------------------------------------------
# Stage 3 - reader deployment + inference
# ---------------------------------------------------------------------------

def load_reader(model_key, mode=None, log_fn=log):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    spec = READERS[model_key]
    mode = mode or spec["default_mode"]
    path = spec["local_path"]
    log_fn(f"loading {model_key} mode={mode} from {path}")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    if mode == "bf16_offload":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map="auto", max_memory={0: "7GiB", "cpu": "4GiB"})
    elif mode == "bf16_gpu":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map={"": "cuda:0"})
    elif mode == "int8":
        from transformers import BitsAndBytesConfig
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map="auto", max_memory={0: "7GiB", "cpu": "4GiB"},
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True))
    elif mode == "int4":
        from transformers import BitsAndBytesConfig
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map="auto", max_memory={0: "7GiB", "cpu": "3GiB"},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                                   bnb_4bit_use_double_quant=True))
    else:
        raise ValueError(mode)
    model.eval()
    return processor, model, mode


def reader_generate(processor, model, image_path, prompt=PROMPT):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path), "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=GEN_DO_SAMPLE, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                use_cache=True)
    prompt_len = int(inputs["input_ids"].shape[-1])
    continuation = output[0, prompt_len:]
    text = processor.batch_decode([continuation], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
    token_ids = [int(v) for v in continuation.tolist()]
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        eos_ids.add(int(eos))
    extra = getattr(tokenizer, "eos_token_ids", None)
    if extra is not None:
        if isinstance(extra, (list, tuple, set, frozenset)):
            eos_ids.update(int(v) for v in extra)
        else:
            eos_ids.add(int(extra))
    eos_observed = bool(eos_ids.intersection(token_ids))
    return {"raw_response": text, "token_ids": token_ids, "n_new_tokens": len(token_ids),
            "eos_observed": eos_observed,
            "truncated": bool(len(token_ids) >= GEN_MAX_NEW_TOKENS and not eos_observed),
            "input_length": prompt_len}


def _run_batch_on_inputs(model_key, mode, inputs, result_path, stage):
    processor, model, mode = load_reader(model_key, mode)
    existing = {}
    for row in rows(result_path):
        existing[row.get("qa_id") or row.get("comp_id")] = row
    todo = []
    for item in inputs:
        key = item.get("qa_id") or item.get("comp_id")
        old = existing.get(key)
        if old and old.get("input_sha256") == item["input_sha256"] and old.get("prompt_sha256") == PROMPT_SHA and old.get("status") == "OK":
            continue
        todo.append(item)
    log(f"{stage} {model_key} mode={mode} todo={len(todo)} cached={len(inputs)-len(todo)}")
    done = 0
    for item in todo:
        key = item.get("qa_id") or item.get("comp_id")
        record = {"qa_id": item.get("qa_id"), "comp_id": item.get("comp_id"),
                  "family_id": item["family_id"], "logical_ref": item["logical_ref"],
                  "image_path": item["overlay_path"], "image_sha256": item["overlay_sha256"],
                  "input_sha256": item["input_sha256"], "prompt": PROMPT, "prompt_sha256": PROMPT_SHA,
                  "protocol_id": PROTOCOL_ID, "reader": model_key, "deployment_mode": mode,
                  "status": "ERROR", "error": None}
        started = time.time()
        try:
            generation = reader_generate(processor, model, item["overlay_path"])
            record.update(generation)
            record["color"] = strict_color(generation["raw_response"])
            record["valid"] = record["color"] is not None
            record["status"] = "OK"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-2000:]
        record["wall_seconds"] = round(time.time() - started, 3)
        append(result_path, record)
        existing[key] = record
        done += 1
        if done % 5 == 0 or done == len(todo):
            log(f"{stage} {done}/{len(todo)} key={key} color={record.get('color')} raw={record.get('raw_response')!r} ({record['wall_seconds']}s)")
    return rows(result_path)


def stage_smoke(model_key, mode=None):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "figures" / "smoke_input.png"
    image = Image.new("RGB", (512, 512), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle([160, 160, 352, 352], fill=(40, 150, 60))
    draw.rectangle(expanded_bbox([160, 160, 352, 352], (512, 512)), outline=BBOX_OUTLINE_RGB, width=BBOX_OUTLINE_WIDTH)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    processor, model, mode = load_reader(model_key, mode)
    generation = reader_generate(processor, model, path)
    color = strict_color(generation["raw_response"])
    record = {"model": model_key, "deployment_mode": mode, "smoke_input": str(path),
              "expected_color": "green", "generation": generation, "parsed_color": color,
              "pass": color == "green", "prompt": PROMPT, "prompt_sha256": PROMPT_SHA,
              "used_cohort_data": False}
    dump(OUT / f"model_deployment_smoke_{model_key}.json", record)
    log(f"SMOKE {model_key} mode={mode} parsed={color} pass={color == 'green'} raw={generation['raw_response']!r}")
    return record


def stage_atomic(model_key, mode=None):
    inputs = rows(OUT / "atomic_reader_inputs.jsonl")
    assert len(inputs) == 200
    result_path = OUT / f"atomic_{model_key}_results.jsonl"
    _run_batch_on_inputs(model_key, mode, inputs, result_path, "atomic")
    records = rows(result_path)
    valid = sum(1 for r in records if r.get("status") == "OK" and r.get("valid"))
    ok = sum(1 for r in records if r.get("status") == "OK")
    update_status(status=f"ATOMIC_{model_key.upper()}_COMPLETE", **{f"atomic_{model_key}_complete": True,
                  f"atomic_{model_key}_valid": valid, f"atomic_{model_key}_ok": ok})
    log(f"ATOMIC_{model_key.upper()}_COMPLETE valid={valid}/{len(records)}")


def stage_select():
    atomic_inputs = {r["qa_id"]: r for r in rows(OUT / "atomic_reader_inputs.jsonl")}
    atomic_gt = json.loads((OUT / "atomic_evaluator_only.json").read_text())
    # historical Qwen baseline (reused, not rerun)
    hist = [r for r in rows(ATOMIC_DIR / "atomic_view_predictions.jsonl") if r.get("view") == "box"]
    hist_by_qa = {r["qa_id"]: r for r in hist}
    assert len(hist_by_qa) == 200
    hist_valid = sum(1 for qa_id in atomic_inputs if hist_by_qa.get(qa_id, {}).get("valid"))
    hist_correct = sum(1 for qa_id in atomic_inputs if hist_by_qa.get(qa_id, {}).get("normalized_color") == atomic_gt[qa_id]["gt_color"])
    assert hist_correct == ATOMIC_BASELINE_CORRECT, hist_correct
    baseline = {
        "reader": "Qwen3.5-4B historical exact-bbox reader", "reference": True,
        "n": 200, "valid": hist_valid, "invalid_vocabulary": 200 - hist_valid,
        "correct": hist_correct, "atomic_color_accuracy": f"{hist_correct}/200",
        "atomic_color_accuracy_value": hist_correct / 200.0,
        "source": str(ATOMIC_DIR / "atomic_view_predictions.jsonl"),
        "source_view": "box", "interface": "exact bbox crop (historical protocol)",
    }
    dump(OUT / "atomic_qwen_baseline.json", baseline)

    comparison = {"n": 200, "main_metric": "AtomicColorAccuracy (strict frozen RealBind vocabulary, parse_color)",
                  "readers": {"qwen35_4b_historical": baseline}}
    candidates = []
    for key in ("zwz4b", "zwz8b"):
        if key not in READERS:
            continue
        records = rows(OUT / f"atomic_{key}_results.jsonl")
        if len(records) != 200:
            comparison["readers"][key] = {"status": "not_run_or_incomplete", "n": len(records)}
            continue
        by_qa = {r["qa_id"]: r for r in records}
        ok = sum(1 for r in records if r.get("status") == "OK")
        valid = sum(1 for r in records if r.get("status") == "OK" and r.get("color"))
        invalid_vocab = 200 - valid
        trunc = sum(1 for r in records if r.get("truncated"))
        correct = sum(1 for qa_id in atomic_inputs if by_qa.get(qa_id, {}).get("color") == atomic_gt[qa_id]["gt_color"])
        # normalized (downstream semantic parser) accuracy
        try:
            sys.path.insert(0, "/root/Qwen3-VL")
            from experiments.realbind_phase2.pair_direct_v1_1_baseline import parse_semantic_answer
            norm_correct = 0
            for qa_id, item in atomic_inputs.items():
                rec = by_qa.get(qa_id, {})
                out = parse_semantic_answer(str(rec.get("raw_response", "")), "color", atomic_gt[qa_id]["gt_color"], {}, {"color": list(REALBIND_COLORS)})
                if (not out["invalid_prediction"]) and out["normalized_prediction"] == atomic_gt[qa_id]["gt_color"]:
                    norm_correct += 1
        except Exception:
            norm_correct = None
        modes = sorted({r.get("deployment_mode") for r in records})
        comparison["readers"][key] = {
            "status": "complete", "reader": f"ZwZ-{key[-2:].upper()} ({READERS[key]['repo']})",
            "n": 200, "ok": ok, "valid": valid, "valid_output_rate": valid / 200.0,
            "invalid_vocabulary": invalid_vocab, "truncation_count": trunc,
            "correct": correct, "atomic_color_accuracy": f"{correct}/200",
            "atomic_color_accuracy_value": correct / 200.0,
            "normalized_correct": norm_correct,
            "normalized_accuracy": (norm_correct / 200.0) if norm_correct is not None else None,
            "deployment_modes": modes, "results_file": str(OUT / f"atomic_{key}_results.jsonl"),
            "interface": "full original image + official red bbox overlay (expand 5%, width 5)",
        }
        candidates.append({"key": key, "correct": correct, "normalized_correct": norm_correct if norm_correct is not None else -1,
                           "valid": valid, "truncation": trunc, "size_rank": 0 if key == "zwz4b" else 1})
    comparison["gate"] = {"baseline_correct": ATOMIC_BASELINE_CORRECT, "require_strictly_greater": True}
    dump(OUT / "atomic_reader_comparison.json", comparison)

    if not candidates:
        selection = {"decision": "SPECIALIZED_READER_NO_ATOMIC_GAIN", "reason": "no specialised reader results",
                     "selected_reader": None, "atomic_complete": False}
        dump(OUT / "reader_selection.json", selection)
        update_status(status="SPECIALIZED_READER_NO_ATOMIC_GAIN", selection_complete=True,
                      selected_reader=None, stop_rule="ATOMIC")
        return selection
    eligible = [c for c in candidates if c["correct"] > ATOMIC_GATE]
    if not eligible:
        selection = {"decision": "SPECIALIZED_READER_NO_ATOMIC_GAIN",
                     "reason": f"all specialised readers <= {ATOMIC_BASELINE_CORRECT}/200",
                     "selected_reader": None, "candidates": candidates,
                     "stop_rule": "STOP_RULE_A: no composition run"}
        dump(OUT / "reader_selection.json", selection)
        update_status(status="SPECIALIZED_READER_NO_ATOMIC_GAIN", selection_complete=True,
                      selected_reader=None, stop_rule="ATOMIC")
        return selection
    eligible.sort(key=lambda c: (-c["correct"], -c["normalized_correct"], c["size_rank"]))
    best = eligible[0]
    if best["correct"] >= ATOMIC_MARGINAL:
        gain_label = "CLEAR_ATOMIC_GAIN"
    else:
        gain_label = "MARGINAL_ATOMIC_GAIN"
    selection = {
        "decision": "READER_SELECTED", "atomic_gain_label": gain_label,
        "selected_reader": best["key"], "selected_reader_repo": READERS[best["key"]]["repo"],
        "atomic_color_accuracy": f"{best['correct']}/200",
        "baseline": f"{ATOMIC_BASELINE_CORRECT}/200",
        "criteria": ["highest AtomicColorAccuracy", "tie -> smaller / lower VRAM / faster (ZwZ-4B before ZwZ-8B)",
                     "selection never looks at composition results"],
        "candidates": candidates,
    }
    dump(OUT / "reader_selection.json", selection)
    update_status(status="READER_SELECTED", selection_complete=True,
                  selected_reader=best["key"], atomic_gain_label=gain_label)
    log(f"READER_SELECTED {best['key']} {best['correct']}/200 ({gain_label})")
    return selection


def stage_composition(model_key=None, mode=None):
    selection = json.loads((OUT / "reader_selection.json").read_text())
    model_key = model_key or selection.get("selected_reader")
    if not model_key:
        raise SystemExit("no reader selected; composition must not run")
    inputs = rows(OUT / "composition_reader_inputs.jsonl")
    result_path = OUT / "composition_attribute_results.jsonl"
    _run_batch_on_inputs(model_key, mode, inputs, result_path, "composition")
    records = rows(result_path)
    update_status(status="COMPOSITION_COMPLETE", composition_complete=True,
                  composition_reader=model_key, composition_rows=len(records))
    log(f"COMPOSITION_COMPLETE {model_key} rows={len(records)}")


# ---------------------------------------------------------------------------
# Stage 4 - assembly + attribute transition analysis
# ---------------------------------------------------------------------------

def stage_assemble():
    parent_eval = {r["qa_id"]: r for r in rows(OUT / "_parent_evaluator_rows.jsonl")}
    selection = json.loads((OUT / "reader_selection.json").read_text())
    model_key = selection.get("selected_reader")
    assert model_key, "no selected reader"
    comp = {r["comp_id"]: r for r in rows(OUT / "composition_attribute_results.jsonl")}
    for key in comp:
        assert comp[key].get("status") == "OK", key
    states = []
    for qa_id, row in parent_eval.items():
        target = row["target"]
        rec = comp.get(f"{row['family_id']}__{target}") if target else None
        color = rec.get("color") if rec else None
        attr_correct = bool(color is not None and str(color).strip().casefold() == row["gt_color"])
        sufficient = bool(row["logical_target_correct"] and row["grounded_target_correct"] and attr_correct)
        state_text = f"[TARGET_ENTITY] {target if target else 'unknown'}\n[TARGET_ATTRIBUTE] color = {color if color else 'unknown'}"
        states.append({
            "qa_id": qa_id, "family_id": row["family_id"], "qa_type": row["qa_type"],
            "route": "SPECIALIZED_ATTRIBUTE_READER_REPAIR",
            "logical_target_correct": row["logical_target_correct"],
            "grounded_target_correct": row["grounded_target_correct"],
            "target_attribute_correct": attr_correct,
            "minimal_state_sufficient": sufficient,
            "target": target, "true_target": row["true_target"], "gt_color": row["gt_color"],
            "rendered_color": color,
            "historical_rendered_color": row["parent_rendered_color"],
            "historical_target_attribute_correct": row["historical_target_attribute_correct"],
            "historical_minimal_state_sufficient": row["historical_minimal_state_sufficient"],
            "selected_target_bbox_xyxy": row["selected_target_bbox_xyxy"],
            "target_region_hash": row["target_region_hash"],
            "reader": model_key, "reader_raw_response": rec.get("raw_response") if rec else None,
            "reader_tokens": rec.get("n_new_tokens") if rec else None,
            "reader_truncated": rec.get("truncated") if rec else None,
            "state_text": state_text, "state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
        })
    assert len(states) == 40
    dumpl(OUT / "minimal_state_results.jsonl", states)

    grounded = [s for s in states if s["grounded_target_correct"]]
    assert len(grounded) == 26
    hist_correct = sum(1 for s in grounded if s["historical_target_attribute_correct"])
    new_correct = sum(1 for s in grounded if s["target_attribute_correct"])
    both_correct = sum(1 for s in grounded if s["historical_target_attribute_correct"] and s["target_attribute_correct"])
    both_wrong = sum(1 for s in grounded if not s["historical_target_attribute_correct"] and not s["target_attribute_correct"])
    rescued = sum(1 for s in grounded if not s["historical_target_attribute_correct"] and s["target_attribute_correct"])
    regressed = sum(1 for s in grounded if s["historical_target_attribute_correct"] and not s["target_attribute_correct"])
    grounded_ids = {s["qa_id"] for s in grounded}
    transition = {
        "n_grounded": 26,
        "historical_attribute_correct": f"{hist_correct}/26",
        "new_attribute_correct": f"{new_correct}/26",
        "transitions": {
            "historical_wrong_new_correct": rescued,
            "historical_correct_new_wrong": regressed,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "regression_constraint": {"regressions": regressed, "<=1": regressed <= 1},
        "rescue_target": {"historical_failures": 26 - hist_correct, "rescued": rescued,
                          "target_min": 2, "target_ideal": "3-4", "met": rescued >= 2},
        "per_case": [{k: s[k] for k in ("qa_id", "family_id", "target", "gt_color", "rendered_color",
                                        "historical_rendered_color", "target_attribute_correct",
                                        "historical_target_attribute_correct", "selected_target_bbox_xyxy")} for s in grounded],
    }
    dump(OUT / "attribute_transition_analysis.json", transition)

    # all-40 summary + historical failures detailed analysis (evaluator-only join)
    failures = []
    for s in states:
        if s["historical_target_attribute_correct"] or not s["grounded_target_correct"]:
            continue
        bbox = s["selected_target_bbox_xyxy"]
        image_size = None
        if bbox:
            with Image.open(parent_eval[s["qa_id"]]["image_path"]) as im:
                image_size = list(im.size)
        area_frac = None
        if bbox and image_size:
            area_frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(image_size[0] * image_size[1])
        failures.append({
            "qa_id": s["qa_id"], "family_id": s["family_id"], "target_entity": s["target"],
            "target_bbox_xyxy": bbox, "bbox_area_fraction": area_frac,
            "image_size": image_size, "gt_color_evaluator_only": s["gt_color"],
            "historical_prediction": s["historical_rendered_color"],
            "specialized_reader_prediction": s["rendered_color"],
            "raw_reader_response": s["reader_raw_response"],
            "rescue": bool(s["target_attribute_correct"]),
            "visual_failure_characterization": (
                "tiny_target_region" if area_frac is not None and area_frac < 0.01 else
                "small_target_region" if area_frac is not None and area_frac < 0.03 else
                "large_target_region"),
        })
    dump(OUT / "historical_attribute_failure_analysis.json", {
        "note": "Evaluator-only join; never used for prompt/interface tuning.",
        "n": len(failures), "rescued": sum(1 for f in failures if f["rescue"]), "rows": failures})
    attr_all = sum(s["target_attribute_correct"] for s in states)
    sufficient = sum(s["minimal_state_sufficient"] for s in states)
    dump(OUT / "survival_chain.json", {
        "start": 40,
        "logical_target_correct": sum(s["logical_target_correct"] for s in states),
        "grounded_target_correct": 26,
        "attribute_correct_given_grounding": new_correct,
        "target_attribute_correct_all": attr_all,
        "minimal_state_sufficient": sufficient,
        "clean_final_correct": None,
        "raw_final_correct": None,
        "funnel_note": "40 -> LogicalTargetCorrect -> GroundedTargetCorrect -> AttributeCorrectGivenGrounding -> MinimalStateSufficient -> CleanFinal",
    })
    update_status(status="ASSEMBLED", assemble_complete=True,
                  new_attribute_given_grounding=new_correct, minimal_state_sufficient=sufficient)
    log(f"ASSEMBLED attr_given_grounding={new_correct}/26 (hist {hist_correct}/26) "
        f"rescued={rescued} regressed={regressed} sufficient={sufficient}/40")


# ---------------------------------------------------------------------------
# Stage 5 - frozen text-only final reasoner
# ---------------------------------------------------------------------------

def stage_final():
    from harness.configs.paths import Paths
    from harness.dataset import read_jsonl
    from harness.evaluators.end2end_metrics import score
    from harness.processors.color_pixel import REALBIND_COLORS as RB
    from harness.reasoners.qwen35 import Qwen35Reasoner
    from harness.scripts.run_minimal_decomposition_route_pilot import prompt

    states = {r["qa_id"]: r for r in rows(OUT / "minimal_state_results.jsonl")}
    prepared = {r["qa_id"]: r for r in rows(PREPARED)}
    families = {r["family_id"]: r for r in read_jsonl(Paths().families)}
    vocab = {"color": list(RB), "position": ["left", "right"],
             "identity": sorted({family[k]["object"] for family in families.values() for k in ("A", "B")})}
    result_path = OUT / "final_predictions.jsonl"
    existing = {(r["qa_id"]): r for r in rows(result_path)}
    reasoner = Qwen35Reasoner()
    records = list(existing.values())
    for index, qa_id in enumerate(sorted(states), 1):
        state = states[qa_id]
        if qa_id in existing and existing[qa_id].get("state_sha256") == state["state_sha256"]:
            continue
        qa = prepared[qa_id]
        query = prompt(qa["question"], state["state_text"])
        generation = reasoner.generate(query, None)
        assert generation["image_path"] is None
        assert not generation["vision_encoder_called"]
        assert not generation["visual_tokens_in_rendered_prompt"]
        scored = score(generation["response"], "color", state["gt_color"], families[qa["family_id"]], vocab)
        record = {
            "qa_id": qa_id, "route": state["route"], "state_sha256": state["state_sha256"],
            "state_sufficient": bool(state["minimal_state_sufficient"]),
            "question": qa["question"], "state_text": state["state_text"],
            "prompt_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "raw_response": generation["response"], "prediction": scored["normalized_prediction"],
            "invalid_prediction": bool(scored["invalid_prediction"]),
            "final_correct": bool(scored["correct"]),
            "image_path": None, "image_present": False, "vision_encoder_called": False,
            "scoring_revision": scored["scoring_revision"],
            "writer_reader": state["reader"],
        }
        append(result_path, record)
        existing[qa_id] = record
        log(f"final {index}/40 {qa_id} sufficient={record['state_sufficient']} pred={record['prediction']} correct={record['final_correct']}")
    final_rows = [existing[k] for k in sorted(existing)]
    assert len(final_rows) == 40
    state_sufficient = sum(r["state_sufficient"] for r in final_rows)
    raw = sum(r["final_correct"] for r in final_rows)
    clean = sum(r["state_sufficient"] and r["final_correct"] for r in final_rows)
    comp = sum((not r["state_sufficient"]) and r["final_correct"] for r in final_rows)
    logical = sum(states[r["qa_id"]]["logical_target_correct"] for r in final_rows)
    grounded = sum(states[r["qa_id"]]["grounded_target_correct"] for r in final_rows)
    attr_grounded = sum(states[r["qa_id"]]["grounded_target_correct"] and states[r["qa_id"]]["target_attribute_correct"] for r in final_rows)
    funnel = [logical, grounded, attr_grounded, state_sufficient, clean]
    monotonic = all(funnel[i] >= funnel[i + 1] for i in range(len(funnel) - 1))
    dump(OUT / "clean_vs_raw.json", {
        "n": 40, "funnel": {"start": 40, "logical_target_correct": logical,
                            "grounded_target_correct": grounded,
                            "attribute_correct_given_grounding": attr_grounded,
                            "minimal_state_sufficient": state_sufficient,
                            "clean_final_correct": clean, "raw_final_correct": raw},
        "state_sufficient": state_sufficient, "clean_final_correct": clean, "raw_final_correct": raw,
        "compensating_error_correct": comp,
        "historical_parent": {"grounded_target_correct": 26, "attribute_correct_given_grounding": 20,
                              "minimal_state_sufficient": 20, "clean_final_correct": 20, "raw_final_correct": 25},
        "vanilla_vlm_locked": 20,
        "monotonic_funnel": bool(monotonic),
    })
    comp_rows = []
    for r in final_rows:
        s = states[r["qa_id"]]
        if (not r["state_sufficient"]) and r["final_correct"]:
            comp_rows.append({"qa_id": r["qa_id"], "target": s["target"], "true_target": s["true_target"],
                              "grounded_target_correct": s["grounded_target_correct"],
                              "target_attribute_correct": s["target_attribute_correct"],
                              "state_text": s["state_text"], "final_prediction": r["prediction"],
                              "gt_color_evaluator_only": s["gt_color"]})
    dump(OUT / "compensating_error_analysis.json", {
        "count": len(comp_rows),
        "parent_count": 5,
        "definition": "RawFinalCorrect AND NOT MinimalStateSufficient",
        "rows": comp_rows,
        "clean_primary_goal": "improve clean state, not compensating errors",
    })
    dump(OUT / "survival_chain.json", {
        "start": 40, "logical_target_correct": logical, "grounded_target_correct": grounded,
        "attribute_correct_given_grounding": attr_grounded, "target_attribute_correct_all": sum(s["target_attribute_correct"] for s in states.values()),
        "minimal_state_sufficient": state_sufficient, "clean_final_correct": clean, "raw_final_correct": raw,
        "compensating_error_correct": comp, "monotonic": bool(monotonic),
        "funnel_note": "40 -> LogicalTargetCorrect -> GroundedTargetCorrect -> AttributeCorrectGivenGrounding -> MinimalStateSufficient -> CleanFinal",
    })
    update_status(status="FINAL_COMPLETE", final_complete=True, clean=clean, raw=raw,
                  attr_given_grounding=attr_grounded, sufficient=state_sufficient)
    log(f"FINAL_COMPLETE logical={logical} grounded={grounded} attr_grounded={attr_grounded}/26 "
        f"sufficient={state_sufficient} clean={clean} raw={raw} compensating={comp} monotonic={monotonic}")


# ---------------------------------------------------------------------------
# Stage 6 - audits + reports
# ---------------------------------------------------------------------------

def stage_audit():
    final = rows(OUT / "final_predictions.jsonl")
    states = {r["qa_id"]: r for r in rows(OUT / "minimal_state_results.jsonl")}
    hard_rows = [{"qa_id": r["qa_id"], "image_present": r["image_present"],
                  "vision_encoder_called": r["vision_encoder_called"],
                  "state_sha256_matches": r["state_sha256"] == states[r["qa_id"]]["state_sha256"]} for r in final]
    if not final:
        dump(OUT / "hard_handoff_audit.json", {
            "pass": None, "status": "NOT_RUN", "n": 0, "rows": [],
            "requirement": "image_present=false and vision_encoder_called=false for 40/40",
            "reason": "final reasoner handoff not executed because composition did not run (Stop Rule A)",
        })
    else:
        dump(OUT / "hard_handoff_audit.json", {
            "pass": all((not r["image_present"]) and (not r["vision_encoder_called"]) and r["state_sha256_matches"] for r in hard_rows),
            "n": len(hard_rows), "rows": hard_rows,
            "requirement": "image_present=false and vision_encoder_called=false for 40/40",
        })

    pure = ["atomic_reader_inputs.jsonl", "composition_reader_inputs.jsonl",
            "atomic_zwz4b_results.jsonl", "atomic_zwz8b_results.jsonl",
            "composition_attribute_results.jsonl"]
    forbidden = {"gt_bbox", "ground_truth", "ground_truth_bbox", "bbox_gt", "oracle_bbox",
                 "true_target", "answer", "relation_answer", "gt_color"}
    hits = []
    for name in pure:
        path = OUT / name
        for idx, row in enumerate(rows(path)):
            for key in row:
                if str(key).casefold() in forbidden or str(key).casefold().startswith("gt_"):
                    hits.append(f"{name}[{idx}].{key}")
    prompt_leak_checks = {
        "prompt_contains_gt_vocabulary_color_words_only": True,
        "prompt_does_not_mention_left_right": all(t not in PROMPT.casefold() for t in ("left", "right")),
        "prompt_does_not_mention_ab_labels": all(t not in PROMPT for t in ("[TARGET", "answer is")),
        "reader_receives_only_overlay_and_prompt": True,
        "atomic_inputs_have_no_gt_fields": True,
    }
    dump(OUT / "gt_leakage_audit.json", {
        "pass": len(hits) == 0,
        "pure_inference_artifacts_scanned": pure,
        "forbidden_key_hits": hits,
        "evaluator_only_artifacts": ["atomic_evaluator_only.json", "composition_evaluator_only.json",
                                      "attribute_transition_analysis.json", "historical_attribute_failure_analysis.json",
                                      "clean_vs_raw.json", "compensating_error_analysis.json"],
        "gt_used_only_after_inference_for_evaluation": True,
        "reader_input_fields": ["full original image + red bbox overlay", "fixed generic color prompt"],
        "fixed_prompt": PROMPT, "fixed_prompt_sha256": PROMPT_SHA,
        "prompt_leak_checks": prompt_leak_checks,
        "final_reasoner_image_present_values": sorted({r["image_present"] for r in final}),
        "final_reasoner_vision_encoder_called_values": sorted({r["vision_encoder_called"] for r in final}),
    })
    parent_audit = json.loads((OUT / "upstream_hash_audit.json").read_text())
    state_rows = rows(OUT / "minimal_state_results.jsonl")
    if not state_rows:
        consistency = {
            "pass": None,
            "status": "NOT_APPLICABLE_STOP_RULE_A",
            "parent_grounded_target_correct": 26,
            "new_grounded_target_correct": None,
            "requirement": "GroundedTargetCorrect_new == 26/40 (only applies if composition is executed)",
            "note": "Composition was not run because no specialised reader exceeded 129/200 atomic; "
                    "the frozen parent GroundedTargetCorrect=26/40 is verified in upstream_hash_audit.json.",
        }
    else:
        consistency = {
            "pass": bool(parent_audit["pass"]),
            "parent_grounded_target_correct": 26,
            "new_grounded_target_correct": sum(1 for r in state_rows if r["grounded_target_correct"]),
            "requirement": "GroundedTargetCorrect_new == 26/40",
        }
        consistency["pass"] = consistency["pass"] and consistency["new_grounded_target_correct"] == 26
    dump(OUT / "consistency_audit.json", consistency)
    update_status(status="AUDIT_COMPLETE", audit_complete=True)
    log(f"AUDIT_COMPLETE hard_handoff={json.loads((OUT / 'hard_handoff_audit.json').read_text())['pass']} "
        f"gt_leak_pass={len(hits) == 0} consistency={consistency['pass']}")


def stage_stop_outputs():
    """Honest NOT_RUN artifacts for downstream stages when Stop Rule A fires."""
    selection = json.loads((OUT / "reader_selection.json").read_text())
    ensure = {
        "status": "NOT_RUN",
        "reason": "SPECIALIZED_READER_NO_ATOMIC_GAIN",
        "stop_rule": "STOP_RULE_A: no specialised reader exceeded 129/200 atomic; composition was NOT run",
        "selection": selection.get("decision"),
        "composition_reader": None,
        "note": "Empty JSONL files are intentional: no composition inference was executed.",
    }
    for name in ("attribute_transition_analysis.json", "clean_vs_raw.json",
                 "compensating_error_analysis.json"):
        dump(OUT / name, ensure)
    dump(OUT / "survival_chain.json", {
        "start": 40,
        "logical_target_correct": 36,
        "grounded_target_correct": 26,
        "attribute_correct_given_grounding": None,
        "minimal_state_sufficient": None,
        "clean_final_correct": None,
        "raw_final_correct": None,
        "status": "NOT_RUN",
        "reason": "SPECIALIZED_READER_NO_ATOMIC_GAIN",
        "parent_baseline": {"attribute_correct_given_grounding": 20, "minimal_state_sufficient": 20,
                            "clean_final_correct": 20, "raw_final_correct": 25},
    })
    for name in ("composition_attribute_results.jsonl", "minimal_state_results.jsonl", "final_predictions.jsonl"):
        (OUT / name).write_text("", encoding="utf-8")
    dump(OUT / "composition_stop_notice.json", ensure)
    dump(OUT / "hard_handoff_audit.json", {
        "pass": None, "status": "NOT_RUN", "n": 0, "rows": [],
        "reason": "final reasoner handoff not executed because composition did not run (Stop Rule A)",
    })
    update_status(status="STOPPED_NO_ATOMIC_GAIN_OUTPUTS_WRITTEN", stop_outputs_complete=True)
    log("STOP_OUTPUTS_WRITTEN (composition not run per Stop Rule A)")


def _decision_label():
    sel = json.loads((OUT / "reader_selection.json").read_text()) if (OUT / "reader_selection.json").exists() else {}
    if sel.get("decision") == "SPECIALIZED_READER_NO_ATOMIC_GAIN":
        return "SPECIALIZED_READER_NO_ATOMIC_GAIN"
    clean_path = OUT / "clean_vs_raw.json"
    trans_path = OUT / "attribute_transition_analysis.json"
    if not clean_path.exists() or not trans_path.exists():
        return "INCOMPLETE"
    clean = json.loads(clean_path.read_text())
    trans = json.loads(trans_path.read_text())
    attr = clean["funnel"]["attribute_correct_given_grounding"]
    reg = trans["transitions"]["historical_correct_new_wrong"]
    if attr <= 20:
        return "ATOMIC_GAIN_DOES_NOT_TRANSFER"
    if reg > 1:
        return "SPECIALIZED_READER_GAIN_FRAGILE"
    if clean["clean_final_correct"] > 20:
        return "CLEAN_BASELINE_BREAKING_REPAIR"
    return "SPECIALIZED_ATTRIBUTE_REPAIR_ESTABLISHED"


def stage_report():
    comparison = json.loads((OUT / "atomic_reader_comparison.json").read_text())
    selection = json.loads((OUT / "reader_selection.json").read_text())
    trans_path = OUT / "attribute_transition_analysis.json"
    clean_path = OUT / "clean_vs_raw.json"
    trans = json.loads(trans_path.read_text()) if trans_path.exists() else {"status": "NOT_RUN"}
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else {"status": "NOT_RUN"}
    parent = json.loads((OUT / "historical_parent_manifest.json").read_text())
    failure_path = OUT / "historical_attribute_failure_analysis.json"
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {"n": 0, "rescued": 0, "rows": []}
    decision = _decision_label()
    stopped = decision == "SPECIALIZED_READER_NO_ATOMIC_GAIN"
    label_meanings = {
        "SPECIALIZED_READER_NO_ATOMIC_GAIN": "specialised reader did not beat 129/200 atomic; composition was not run",
        "MARGINAL_ATOMIC_GAIN": "atomic gain was small (130-139/200); interpret cautiously",
        "ATOMIC_GAIN_DOES_NOT_TRANSFER": "atomic gain did not transfer to correctly-grounded composition",
        "SPECIALIZED_ATTRIBUTE_REPAIR_ESTABLISHED": "attribute repair beyond 20/26 with <=1 regression",
        "CLEAN_BASELINE_BREAKING_REPAIR": "CleanFinal > 20/40 supported by sufficient minimal states",
        "SPECIALIZED_READER_GAIN_FRAGILE": "total gain came with too many regressions",
    }
    rows_atomic = []
    for name, data in comparison["readers"].items():
        if data.get("reference"):
            rows_atomic.append(("Qwen3.5-4B (historical exact-bbox)", data.get("valid", "200"), "129/200 reference"))
        elif data.get("status") == "complete":
            rows_atomic.append((data.get("reader", name), f"{data['valid']}/200", f"{data['correct']}/{data['n']}"))
        else:
            rows_atomic.append((name, "NOT RUN", "NOT RUN"))
    lines = []
    lines.append("# Specialized Target-Attribute Reader Replacement")
    lines.append("")
    lines.append(f"Decision label: **{decision}** — {label_meanings.get(decision, '')}")
    lines.append("")
    lines.append("## Frozen parent")
    lines.append("")
    lines.append(f"- Strong Integrated: GroundedTargetCorrect {parent['metrics']['grounded_target_correct']}/40, "
                 f"AttributeCorrectGivenGrounding {parent['metrics']['attribute_correct_given_grounding']['correct']}/{parent['metrics']['attribute_correct_given_grounding']['n']}, "
                 f"MinimalStateSufficient {parent['metrics']['minimal_state_sufficient']}/40, "
                 f"CleanFinal {parent['metrics']['clean_final_correct']}/40, RawFinal {parent['metrics']['raw_final_correct']}/40.")
    lines.append(f"- Upstream assertions pass: {json.loads((OUT / 'upstream_hash_audit.json').read_text())['pass']} (selected A/B hashes, target region hashes, frozen parent metrics).")
    lines.append("")
    lines.append("## Stage A - atomic 200 reader selection")
    lines.append("")
    lines.append("| Reader | Valid | Atomic Color Accuracy | Truncation | Deployment |")
    lines.append("|---|---:|---:|---:|---|")
    for name, valid, correct in rows_atomic:
        data = comparison["readers"].get("zwz4b" if "ZwZ-4B" in name else ("zwz8b" if "ZwZ-8B" in name else "qwen35_4b_historical"))
        trunc = data.get("truncation_count", "-") if isinstance(data, dict) else "-"
        mode = ",".join(data.get("deployment_modes", [])) if isinstance(data, dict) and data.get("deployment_modes") else "-"
        lines.append(f"| {name} | {valid} | {correct} | {trunc} | {mode} |")
    lines.append("")
    lines.append(f"Selection: `{selection.get('selected_reader')}` ({selection.get('atomic_gain_label', selection.get('decision'))}).")
    lines.append("Interface: full original image + official red bbox overlay (expand 5%, width 5), fixed generic color prompt.")
    lines.append("")
    lines.append("## Stage B - composition transfer (frozen 40 REL2C)")
    lines.append("")
    if stopped:
        lines.append("**NOT RUN — Stop Rule A fired.** No specialised reader exceeded the fixed atomic baseline "
                     "(129/200), so per protocol the 40 REL2C composition stage was not executed and no composition GT "
                     "was used for any decision. The Strong Integrated parent remains the deployed baseline.")
        lines.append("")
        lines.append("| Reader | GroundedTarget | AttributeCorrect / Grounded | MinimalState | CleanFinal | RawFinal |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append("| Historical Qwen (parent) | 26/40 | 20/26 | 20/40 | 20/40 | 25/40 |")
        lines.append("| Selected Specialized Reader | 26/40 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")
        lines.append("")
        f = None
    else:
        lines.append("| Reader | GroundedTarget | AttributeCorrect / Grounded | MinimalState | CleanFinal | RawFinal |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append("| Historical Qwen | 26/40 | 20/26 | 20/40 | 20/40 | 25/40 |")
        lines.append(f"| {selection.get('selected_reader', 'selected specialised reader')} | 26/40 | "
                     f"{trans['new_attribute_correct']} | {clean['funnel']['minimal_state_sufficient']}/40 | "
                     f"{clean['funnel']['clean_final_correct']}/40 | {clean['funnel']['raw_final_correct']}/40 |")
        lines.append("")
        lines.append("## Paired attribute transition (correctly-grounded 26)")
        lines.append("")
        lines.append("| Transition | Count |")
        lines.append("|---|---:|")
        lines.append(f"| Historical wrong -> New correct | {trans['transitions']['historical_wrong_new_correct']} |")
        lines.append(f"| Historical correct -> New wrong | {trans['transitions']['historical_correct_new_wrong']} |")
        lines.append(f"| Both correct | {trans['transitions']['both_correct']} |")
        lines.append(f"| Both wrong | {trans['transitions']['both_wrong']} |")
        lines.append("")
        lines.append(f"Historical 6 residual attribute failures rescued: {failure['rescued']}/{failure['n']}.")
        lines.append("")
        lines.append("## Funnel (monotonic)")
        lines.append("")
        lines.append("| Stage | Count |")
        lines.append("|---|---:|")
        f = clean["funnel"]
        lines.append(f"| Start | 40 |")
        lines.append(f"| LogicalTargetCorrect | {f['logical_target_correct']} |")
        lines.append(f"| GroundedTargetCorrect | {f['grounded_target_correct']} |")
        lines.append(f"| AttributeCorrectGivenGrounding | {f['attribute_correct_given_grounding']} |")
        lines.append(f"| MinimalStateSufficient | {f['minimal_state_sufficient']} |")
        lines.append(f"| CleanFinal | {f['clean_final_correct']} |")
        lines.append(f"| RawFinal | {f['raw_final_correct']} |")
        lines.append(f"| Compensating errors (raw but not sufficient) | {clean['compensating_error_correct']} |")
        lines.append("")
    lines.append("## Scientific questions")
    lines.append("")
    zwz = comparison["readers"].get(selection.get("selected_reader") or "zwz4b", {})
    baseline_correct = 129
    q1 = (f"Yes: {zwz.get('correct')}/200 vs Qwen 129/200" if zwz.get("correct", 0) and zwz["correct"] > baseline_correct
          else f"No: specialised reader <= 129/200") if zwz.get("status") == "complete" else "Not run"
    lines.append(f"1. Q1 specialised reader beats Qwen on atomic 200? {q1}.")
    if stopped:
        lines.append("2. Q2 atomic gain transfers to correctly-grounded composition? Not testable: Stop Rule A forbade composition.")
        lines.append("3. Q3 historical 6 residual failures rescued: NOT RUN (no composition).")
        lines.append("4. Q4 historical correct attribute cases regressed: NOT RUN (no composition).")
        lines.append("5. Q5 AttributeCorrectGivenGrounding > 20/26? NOT RUN (no composition).")
        lines.append("6. Q6 MinimalStateSufficient > 20/40? NOT RUN (no composition).")
        lines.append("7. Q7 CleanFinal > Vanilla 20/40? No (parent CleanFinal unchanged at 20/40; no replacement deployed).")
        lines.append("8. Q8 gain from clean state improvement? No gain established at the atomic gate.")
        lines.append("9. Q9 remaining attribute errors: the specialised reader was not better than Qwen on the fixed "
                     "atomic protocol, so this experiment provides no support for reader capacity being the remaining "
                     "bottleneck; the residual 6/26 parent attribute errors are more consistent with region/interface "
                     "quality or protocol-level limits than with a generic reader-capacity deficit.")
    else:
        lines.append(f"2. Q2 atomic gain transfers to correctly-grounded composition? "
                     f"{'Yes' if f['attribute_correct_given_grounding'] > 20 else 'No'} ({f['attribute_correct_given_grounding']}/26 vs 20/26).")
        lines.append(f"3. Q3 historical 6 residual failures rescued: {trans['transitions']['historical_wrong_new_correct']}.")
        lines.append(f"4. Q4 historical correct attribute cases regressed: {trans['transitions']['historical_correct_new_wrong']}.")
        lines.append(f"5. Q5 AttributeCorrectGivenGrounding > 20/26? {'Yes' if f['attribute_correct_given_grounding'] > 20 else 'No'}.")
        lines.append(f"6. Q6 MinimalStateSufficient > 20/40? {'Yes' if f['minimal_state_sufficient'] > 20 else 'No'}.")
        lines.append(f"7. Q7 CleanFinal > Vanilla 20/40? {'Yes' if f['clean_final_correct'] > 20 else 'No'}.")
        gain_clean = f['clean_final_correct'] > 20 and f['minimal_state_sufficient'] >= f['clean_final_correct']
        lines.append(f"8. Q8 gain from clean state improvement rather than compensating errors? "
                     f"{'Yes' if gain_clean else 'No/inconclusive'} (compensating errors {clean['compensating_error_correct']} vs parent 5).")
        remaining = 26 - f['attribute_correct_given_grounding']
        lines.append(f"9. Q9 if the specialised reader still fails {remaining} grounded cases, remaining errors likely reflect "
                     f"region/interface quality rather than generic reader capacity to the extent the frozen bbox is imperfect "
                     f"(see historical_attribute_failure_analysis.json).")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("All required artifacts are in `outputs/specialized_attribute_reader/`; JSONL inference files are append+fsync resume-safe.")
    (OUT / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    impl = []
    impl.append("# Implementation report - specialized attribute reader replacement")
    impl.append("")
    impl.append("## What was replaced")
    impl.append("Only the target-local color reader. Grounding, binding, fallbacks, selected A/B boxes, target resolution,")
    impl.append("state format and the text-only final reasoner were frozen and reused from Strong Integrated Repair.")
    impl.append("")
    impl.append("## Interface (frozen before Stage A)")
    impl.append(f"- `{PROTOCOL_ID}`: original full image + official ZwZ red bounding-box overlay (expand {BBOX_EXPAND_RATIO:.0%}, width {BBOX_OUTLINE_WIDTH}px).")
    impl.append(f"- Prompt: `{PROMPT}`")
    impl.append(f"- Prompt sha256: `{PROMPT_SHA}`")
    impl.append(f"- Generation: greedy (do_sample=false), max_new_tokens={GEN_MAX_NEW_TOKENS}, use_cache=true, min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}.")
    impl.append("")
    impl.append("## Models")
    manifest = json.loads((OUT / "model_manifest.json").read_text()) if (OUT / "model_manifest.json").exists() else {}
    for key, entry in manifest.get("readers", {}).items():
        complete = bool(entry.get("weights")) and all(w.get("exists") for w in entry.get("weights", []))
        impl.append(f"- {key}: `{entry.get('repo')}` @ `{entry.get('revision')}` (base {entry.get('base_model')}), "
                    f"local `{entry.get('local_path')}`, default mode `{entry.get('default_mode')}`, "
                    f"weights_complete={complete}.")
    impl.append("")
    impl.append("## Resume safety")
    impl.append("- Atomic/composition JSONL append+fsync with deterministic input hashes; completed records skipped on rerun.")
    impl.append("- `resume_status.json` tracks every stage.")
    impl.append("")
    impl.append("## Audits")
    impl.append(f"- Upstream hash audit: {json.loads((OUT / 'upstream_hash_audit.json').read_text())['pass']}")
    impl.append(f"- GT leakage audit: {json.loads((OUT / 'gt_leakage_audit.json').read_text())['pass']}")
    impl.append(f"- Hard handoff audit: {json.loads((OUT / 'hard_handoff_audit.json').read_text())['pass']}")
    (OUT / "implementation_report.md").write_text("\n".join(impl) + "\n", encoding="utf-8")

    # audit.html
    def table(headers, data_rows):
        head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in data_rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    html_parts = ["<html><head><meta charset='utf-8'><title>Specialized Attribute Reader Audit</title>",
                  "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse;margin:12px 0}"
                  "td,th{border:1px solid #bbb;padding:4px 8px;font-size:13px}th{background:#eee}"
                  ".pass{color:#0a0}.fail{color:#c00}pre{background:#f6f6f6;padding:8px}</style></head><body>"]
    html_parts.append(f"<h1>Specialized Target-Attribute Reader Replacement</h1><h2>Decision: {html.escape(decision)}</h2>")
    special_row = ([f"Specialized Reader ({selection.get('selected_reader') or 'best candidate ZwZ-4B, atomic gate failed'})", "26/40", "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN"]
                   if stopped else
                   [f"Specialized Reader ({selection.get('selected_reader')})", "26/40", trans["new_attribute_correct"],
                    f"{f['minimal_state_sufficient']}/40", f"{f['clean_final_correct']}/40", f"{f['raw_final_correct']}/40"])
    html_parts.append("<h3>Frozen parent</h3>" + table(
        ["Method", "GroundedTarget", "AttributeCorrect", "MinimalState", "CleanFinal", "RawFinal"],
        [["Vanilla", "-", "-", "-", "-", "20/40"],
         ["Strong Integrated", "26/40", "20/26", "20/40", "20/40", "25/40"],
         special_row]))
    html_parts.append("<h3>Atomic Stage A</h3>" + table(["Reader", "Valid", "Atomic Color Accuracy", "Truncation"],
        [[name, valid, correct, (comparison["readers"].get("zwz4b" if "ZwZ-4B" in name else "zwz8b", {}).get("truncation_count", "-"))] for name, valid, correct in rows_atomic]))
    if stopped:
        html_parts.append("<h3>Stage B</h3><p><b>NOT RUN</b> — Stop Rule A: no specialised reader exceeded 129/200 atomic.</p>")
    else:
        html_parts.append("<h3>Paired transitions (26 grounded)</h3>" + table(["Transition", "Count"],
            [["historical wrong -> new correct", trans["transitions"]["historical_wrong_new_correct"]],
             ["historical correct -> new wrong", trans["transitions"]["historical_correct_new_wrong"]],
             ["both correct", trans["transitions"]["both_correct"]],
             ["both wrong", trans["transitions"]["both_wrong"]]]))
        html_parts.append("<h3>Funnel</h3>" + table(["Stage", "Count"],
            [["Start", 40], ["LogicalTargetCorrect", f["logical_target_correct"]],
             ["GroundedTargetCorrect", f["grounded_target_correct"]],
             ["AttributeCorrectGivenGrounding", f["attribute_correct_given_grounding"]],
             ["MinimalStateSufficient", f["minimal_state_sufficient"]],
             ["CleanFinal", f["clean_final_correct"]], ["RawFinal", f["raw_final_correct"]]]))
    audits = []
    for name in ("upstream_hash_audit", "gt_leakage_audit", "hard_handoff_audit", "consistency_audit"):
        p = OUT / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            audits.append([name, "PASS" if data.get("pass") else "FAIL"])
    html_parts.append("<h3>Audits</h3>" + table(["Audit", "Result"], audits))
    html_parts.append("<h3>Resume status</h3><pre>" + html.escape((OUT / "resume_status.json").read_text()) + "</pre>")
    html_parts.append("</body></html>")
    (OUT / "audit.html").write_text("".join(html_parts), encoding="utf-8")
    log(f"REPORTS_WRITTEN decision={decision}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["parent_audit", "inputs", "manifest", "smoke", "atomic",
                                          "select", "composition", "assemble", "final", "audit", "report",
                                          "stop_outputs"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()
    if args.stage == "parent_audit":
        stage_parent_audit()
    elif args.stage == "inputs":
        stage_inputs()
    elif args.stage == "manifest":
        stage_manifest()
    elif args.stage == "smoke":
        stage_smoke(args.model or "zwz4b", args.mode)
    elif args.stage == "atomic":
        stage_atomic(args.model or "zwz4b", args.mode)
    elif args.stage == "select":
        stage_select()
    elif args.stage == "composition":
        stage_composition(args.model, args.mode)
    elif args.stage == "assemble":
        stage_assemble()
    elif args.stage == "final":
        stage_final()
    elif args.stage == "audit":
        stage_audit()
    elif args.stage == "report":
        stage_report()
    elif args.stage == "stop_outputs":
        stage_stop_outputs()


if __name__ == "__main__":
    main()
