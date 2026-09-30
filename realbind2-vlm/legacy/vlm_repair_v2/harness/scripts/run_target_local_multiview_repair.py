#!/usr/bin/env python3
"""Baseline-preserving target-local multi-view attribute repair.

Parent route is frozen Strong Integrated Repair.  Binding, grounding, target
resolution, minimal-state format and final text-only reasoner are untouched.
Only the target-region -> target-color observation interface changes:

  V_box     : exact frozen target bbox crop
  V_mask    : frozen Strong Integrated SAM3 isolated view
  V_context : frozen target bbox expanded by fixed padding 12.5%

Primary fusion is 2-of-3 majority with conservative fallback to the
historical Strong Integrated color when no majority exists.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import html
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_DXG = Path("/dev/dxg")
if not _DXG.exists() and Path("/proc/1/root/dev/dxg").exists():
    try:
        _DXG.symlink_to("/proc/1/root/dev/dxg")
    except OSError:
        pass

ROOT = Path("/mnt/d/vlm_repair_v2")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/target_local_multiview_repair"
HIST = ROOT / "outputs/strong_integrated_repair"
PREPARED = ROOT / "outputs/minimal_decomposition_route_pilot/prepared.jsonl"
PRIOR = ROOT / "outputs/minimal_decomposition_route_pilot/route_results.jsonl"
REALBIND = Path("/mnt/d/datasets/RealBind/realbind")
QWEN_MODEL = Path("/root/Qwen3-VL/models/Qwen3.5-4B")

VLM_PROMPT = "What color is the main object or region shown?\nAnswer with only one color from this list: red, orange, yellow, green, blue, purple, pink, brown, black, white, gray."
PADDING = 0.125
EXPERIMENT = "target_local_multiview_repair_v1"
LEGAL_COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray")
VIEWS = ("box", "mask", "context")


def rows(path: Path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_color(raw: str):
    value = str(raw).strip().casefold()
    return value if value in LEGAL_COLORS else None


def iou_bbox(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def context_box(image_size, bbox, padding=PADDING):
    width, height = image_size
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    px, py = w * padding, h * padding
    cx0, cy0, cx1, cy1 = x0 - px, y0 - py, x1 + px, y1 + py
    cx0, cy0 = max(0.0, min(float(width), cx0)), max(0.0, min(float(height), cy0))
    cx1, cy1 = max(0.0, min(float(width), cx1)), max(0.0, min(float(height), cy1))
    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError("empty context box")
    return [int(round(cx0)), int(round(cy0)), int(round(cx1)), int(round(cy1))]


def exact_box(image_size, bbox):
    width, height = image_size
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty bbox")
    return [x0, y0, x1, y1]


def load_prepared():
    prep = {r["qa_id"]: r for r in rows(PREPARED)}
    prior = {r["qa_id"]: r for r in rows(PRIOR)}
    assert len(prep) == len(prior) == 40
    return prep, prior


def resume_update(**kwargs):
    path = OUT / "resume_status.json"
    status = read_json(path) if path.exists() else {}
    status.update({"experiment": EXPERIMENT, "updated_unix": time.time()})
    status.update(kwargs)
    write_json(path, status)


# ---------------------------------------------------------------------------
# preparation and upstream freeze
# ---------------------------------------------------------------------------

def stage_prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "views").mkdir(exist_ok=True)
    (OUT / "context_crops").mkdir(exist_ok=True)
    binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "binding_results.jsonl")}
    masks = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "mask_results.jsonl")}
    attrs = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "attribute_results.jsonl")}
    qa = {r["qa_id"]: r for r in rows(HIST / "qa_results.jsonl")}
    finals = {r["qa_id"]: r for r in rows(HIST / "final_results.jsonl")}
    assert len(binding) == len(masks) == len(attrs) == 40
    assert len(qa) == len(finals) == 40
    manifest = []
    for key in sorted(binding):
        f, ref = key
        b, m, a = binding[key], masks[key], attrs[key]
        manifest.append({
            "family_id": f,
            "logical_ref": ref,
            "referent_phrase": b["referent_phrase"],
            "selected_bbox_xyxy": b["selected_bbox_xyxy"],
            "region_hash": sha256_json({"bbox": b["selected_bbox_xyxy"], "policy": "frozen_strong_integrated_selected_region"}),
            "mask_path": m.get("mask_path"),
            "mask_valid": bool(m.get("mask_valid")),
            "isolated_path": m.get("isolated_path"),
            "crop_path": a.get("crop_path"),
            "historical_primary_color": a.get("primary_color"),
            "bbox_view_color": a.get("bbox_crop_color"),
            "mask_view_color": a.get("mask_isolated_color"),
            "parent_files": {
                "binding_results": str(HIST / "binding_results.jsonl"),
                "mask_results": str(HIST / "mask_results.jsonl"),
                "attribute_results": str(HIST / "attribute_results.jsonl"),
                "qa_results": str(HIST / "qa_results.jsonl"),
                "final_results": str(HIST / "final_results.jsonl"),
            },
        })
    write_json(OUT / "historical_parent_manifest.json", {
        "parent": "Strong Integrated Repair",
        "n_entities": 40,
        "n_qa": 40,
        "parent_hashes": {
            "binding_results": sha256_file(HIST / "binding_results.jsonl"),
            "mask_results": sha256_file(HIST / "mask_results.jsonl"),
            "attribute_results": sha256_file(HIST / "attribute_results.jsonl"),
            "qa_results": sha256_file(HIST / "qa_results.jsonl"),
            "final_results": sha256_file(HIST / "final_results.jsonl"),
        },
        "entities": manifest,
    })
    config = {
        "experiment": EXPERIMENT,
        "development_only": True,
        "parent_route": "Strong Integrated Repair",
        "frozen_components": [
            "20-family / 40-QA subset",
            "ReferentSpec",
            "MM-GDINO-L proposal generation",
            "historical Qwen selector",
            "selected A/B regions",
            "deterministic geometry and target resolution",
            "minimal-state format",
            "frozen text-only final reasoner",
        ],
        "allowed_change": "target region -> target color observation interface only",
        "views": {
            "box": "exact frozen target bbox crop",
            "mask": "frozen Strong Integrated SAM3 isolated view",
            "context": f"target bbox expanded by fixed padding {PADDING}",
        },
        "view_padding": PADDING,
        "fusion": "2-of-3 majority; no majority -> historical Strong Integrated color",
        "fallback": "HISTORICAL_STRONG_INTEGRATED_COLOR",
        "color_prompt": VLM_PROMPT,
        "color_prompt_sha256": sha256_text(VLM_PROMPT),
        "color_vocabulary": list(LEGAL_COLORS),
        "qwen_model": str(QWEN_MODEL),
        "qwen_generation": {"do_sample": False, "use_cache": False, "max_new_tokens": 16, "enable_thinking": False, "batch_size": 1},
        "final_reasoner": {"image_present": False, "vision_encoder_called": False},
    }
    write_json(OUT / "pipeline_config.json", config)
    resume_update(status="PREPARED", prepare_complete=True, n_entities=40)
    print("[prepare] frozen parent manifest written", flush=True)


# ---------------------------------------------------------------------------
# development views
# ---------------------------------------------------------------------------

def _view_record(f, ref, view, image_path, source, raw_response, color, valid, extra=None):
    rec = {
        "family_id": f,
        "logical_ref": ref,
        "view": view,
        "image_path": str(image_path) if image_path else None,
        "image_sha256": sha256_file(image_path) if image_path and Path(image_path).exists() else None,
        "source": source,
        "raw_response": raw_response,
        "normalized_color": color,
        "valid": bool(valid),
    }
    if extra:
        rec.update(extra)
    return rec


def stage_dev_views():
    OUT.mkdir(parents=True, exist_ok=True)
    binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "binding_results.jsonl")}
    masks = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "mask_results.jsonl")}
    attrs = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "attribute_results.jsonl")}
    manifest = []
    box_predictions = []
    mask_predictions = []
    for key in sorted(binding):
        f, ref = key
        b = binding[key]
        m = masks[key]
        a = attrs[key]
        # Resolve source image through the historical grounding record.
        grounding = next(r for r in rows(HIST / "grounding_results.jsonl") if r["family_id"] == f and r["logical_ref"] == ref)
        image_path = grounding["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        bbox = b["selected_bbox_xyxy"]
        box_path = OUT / "views" / f"{f}__{ref}__box.png"
        box_path.write_bytes(Path(a["crop_path"]).read_bytes()) if a.get("crop_path") and Path(a["crop_path"]).exists() else image.crop(exact_box(image_size, bbox)).save(box_path)
        ctx_path = OUT / "context_crops" / f"{f}__{ref}__context.png"
        image.crop(context_box(image_size, bbox)).save(ctx_path, format="PNG")
        box_valid = a.get("bbox_crop_color") in LEGAL_COLORS and box_path.exists()
        mask_valid = bool(m.get("mask_valid")) and a.get("mask_isolated_color") in LEGAL_COLORS and str(a.get("isolated_path") or "")
        mask_path = Path(a["isolated_path"]) if a.get("isolated_path") else None
        mask_pred_valid = bool(mask_valid and mask_path is not None and mask_path.exists())
        manifest.append({
            "family_id": f,
            "logical_ref": ref,
            "referent_phrase": b["referent_phrase"],
            "selected_bbox_xyxy": bbox,
            "views": {
                "box": {"path": str(box_path), "source": "exact_bbox_crop_regenerated", "valid_shift": box_valid},
                "mask": {"path": str(mask_path) if mask_pred_valid else None, "source": "strong_integrated_frozen_isolated_view", "mask_valid": mask_pred_valid},
                "context": {"path": str(ctx_path), "source": "fixed_padding_12.5pct", "valid_shift": True},
            },
        })
        box_predictions.append({
            "family_id": f, "logical_ref": ref, "view": "box",
            "image_path": str(box_path), "image_sha256": sha256_file(box_path),
            "source": "strong_integrated_reused_prediction",
            "raw_response": (a.get("bbox_crop_generation") or {}).get("raw_response"),
            "normalized_color": a.get("bbox_crop_color") if a.get("bbox_crop_color") in LEGAL_COLORS else None,
            "valid": box_valid,
        })
        mask_predictions.append({
            "family_id": f, "logical_ref": ref, "view": "mask",
            "image_path": str(mask_path) if mask_pred_valid else None,
            "image_sha256": sha256_file(mask_path) if mask_pred_valid else None,
            "source": "strong_integrated_reused_prediction",
            "raw_response": (a.get("mask_isolated_generation") or {}).get("raw_response") if mask_pred_valid else None,
            "normalized_color": a.get("mask_isolated_color") if (mask_pred_valid and a.get("mask_isolated_color") in LEGAL_COLORS) else None,
            "valid": mask_pred_valid,
        })
    write_jsonl(OUT / "view_manifest.jsonl", manifest)
    write_jsonl(OUT / "bbox_view_predictions.jsonl", box_predictions)
    write_jsonl(OUT / "mask_view_predictions.jsonl", mask_predictions)
    resume_update(status="DEV_VIEWS_READY", dev_views_complete=True, n_entities=len(manifest))
    print(f"[dev_views] prepared {len(manifest)} entities", flush=True)


# ---------------------------------------------------------------------------
# development context view inference
# ---------------------------------------------------------------------------

def stage_dev_context():
    from harness.reasoners.qwen35 import Qwen35Reasoner
    manifest = rows(OUT / "view_manifest.jsonl")
    existing = {(r["family_id"], r["logical_ref"], r["view"]): r for r in rows(OUT / "context_view_predictions.jsonl")}
    todo = []
    for m in manifest:
        key = (m["family_id"], m["logical_ref"], "context")
        if key in existing and existing[key].get("status") == "OK":
            continue
        todo.append(m)
    print(f"[dev_context] loading frozen Qwen color reader; todo={len(todo)}", flush=True)
    q = Qwen35Reasoner()
    for index, m in enumerate(todo, 1):
        f, ref = m["family_id"], m["logical_ref"]
        path = Path(m["views"]["context"]["path"])
        gen = q.generate(VLM_PROMPT, path)
        assert gen["vision_encoder_called"] and Path(gen["image_path"]) == path
        color = parse_color(gen["response"])
        record = {
            "family_id": f, "logical_ref": ref, "view": "context",
            "image_path": str(path), "image_sha256": sha256_file(path),
            "source": "new_frozen_qwen_context_inference",
            "raw_response": gen["response"],
            "normalized_color": color,
            "valid": color in LEGAL_COLORS,
            "status": "OK",
            "generation": {"do_sample": False, "use_cache": False, "enable_thinking": False, "max_new_tokens": 16},
        }
        append_jsonl(OUT / "context_view_predictions.jsonl", record)
        existing[(f, ref, "context")] = record
        print(f"[dev_context] {index}/{len(todo)} {f} {ref} -> {color}", flush=True)
    resume_update(status="DEV_CONTEXT_COMPLETE", dev_context_complete=True, context_predictions=len(existing))
    print(f"[dev_context] complete {len(existing)} context predictions", flush=True)


# ---------------------------------------------------------------------------
# primary fusion and downstream
# ---------------------------------------------------------------------------

def _pattern(box, mask, context, mask_valid):
    if not mask_valid:
        return "invalid_view_exists"
    if box is None or mask is None or context is None:
        return "invalid_view_exists"
    if box == mask == context:
        return "box_eq_mask_eq_context"
    if box == mask and box != context:
        return "box_eq_mask_ne_context"
    if box == context and box != mask:
        return "box_eq_context_ne_mask"
    if mask == context and mask != box:
        return "mask_eq_context_ne_box"
    return "all_different"


def _majority(box, mask, context, mask_valid):
    values = []
    if box in LEGAL_COLORS:
        values.append(("box", box))
    if mask_valid and mask in LEGAL_COLORS:
        values.append(("mask", mask))
    if context in LEGAL_COLORS:
        values.append(("context", context))
    counts = Counter(v for _, v in values)
    for color, count in counts.most_common(1):
        if count >= 2:
            return color, "MAJORITY"
    return None, "NO_MAJORITY"


def stage_dev_fuse():
    binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "binding_results.jsonl")}
    attrs = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "attribute_results.jsonl")}
    view_rows = {(r["family_id"], r["logical_ref"], r["view"]): r for r in rows(OUT / "bbox_view_predictions.jsonl") + rows(OUT / "mask_view_predictions.jsonl") + rows(OUT / "context_view_predictions.jsonl")}
    # ensure every entity has all three view rows
    fuse = []
    for key in sorted(binding):
        f, ref = key
        box = view_rows.get((f, ref, "box"), {})
        mask = view_rows.get((f, ref, "mask"), {})
        ctx = view_rows.get((f, ref, "context"), {})
        box_color = box.get("normalized_color") if box.get("valid") else None
        mask_valid = bool(mask.get("valid")) and mask.get("normalized_color") in LEGAL_COLORS
        mask_color = mask.get("normalized_color") if mask_valid else None
        ctx_color = ctx.get("normalized_color") if ctx.get("valid") else None
        hist_color = attrs[key].get("primary_color")
        maj_color, maj_status = _majority(box_color, mask_color, ctx_color, mask_valid)
        if maj_color is None:
            final_color = hist_color
            fallback = "HISTORICAL_STRONG_INTEGRATED_COLOR"
        else:
            final_color = maj_color
            fallback = None
        fuse.append({
            "family_id": f, "logical_ref": ref,
            "referent_phrase": binding[key].get("referent_phrase"),
            "selected_bbox_xyxy": binding[key]["selected_bbox_xyxy"],
            "box_color": box_color, "mask_color": mask_color, "context_color": ctx_color,
            "box_valid": bool(box.get("valid")), "mask_valid": mask_valid, "context_valid": bool(ctx.get("valid")),
            "raw_box_response": box.get("raw_response"),
            "raw_mask_response": mask.get("raw_response"),
            "raw_context_response": ctx.get("raw_response"),
            "disagreement_pattern": _pattern(box_color, mask_color, ctx_color, mask_valid),
            "majority_status": maj_status,
            "majority_color": maj_color,
            "historical_color": hist_color,
            "new_color": final_color,
            "fallback_reason": fallback,
            "changed_from_historical": bool(final_color != hist_color),
        })
    # Primary fusion choices are written before any evaluator annotation is opened.
    write_jsonl(OUT / "majority_fusion_results.jsonl", fuse)
    # Composition projection: one row per QA, using frozen historical target resolution.
    qa_hist = {r["qa_id"]: r for r in rows(HIST / "qa_results.jsonl")}
    fuse_by_key = {(r["family_id"], r["logical_ref"]): r for r in fuse}
    composition = []
    for qa_id, qa in qa_hist.items():
        target = qa.get("target")
        key = (qa["family_id"], target) if target else None
        ff = fuse_by_key.get(key) if key else None
        composition.append({
            "qa_id": qa_id,
            "family_id": qa["family_id"],
            "qa_type": qa["qa_type"],
            "target": target,
            "true_target": qa.get("true_target"),
            "logical_target_correct": bool(qa["logical_target_correct"]),
            "grounded_target_correct": bool(qa["grounded_target_correct"]),
            "historical_color": ff["historical_color"] if ff else None,
            "box_color": ff["box_color"] if ff else None,
            "mask_color": ff["mask_color"] if ff else None,
            "context_color": ff["context_color"] if ff else None,
            "majority_color": ff["majority_color"] if ff else None,
            "new_color": ff["new_color"] if ff else None,
            "fallback_reason": ff["fallback_reason"] if ff else "NO_TARGET_ENTITY",
            "disagreement_pattern": ff["disagreement_pattern"] if ff else None,
            "target_mask_valid": bool(qa.get("target_mask_valid")),
            "historical_state_text": qa.get("state_text"),
        })
    write_jsonl(OUT / "composition_results.jsonl", composition)
    resume_update(status="DEV_FUSION_COMPLETE", dev_fusion_complete=True, n_changed=sum(r["changed_from_historical"] for r in fuse))
    print(f"[dev_fuse] entities={len(fuse)} changed={sum(r['changed_from_historical'] for r in fuse)}", flush=True)


def _load_vocab():
    from harness.configs.paths import Paths
    from harness.dataset import read_jsonl
    from harness.processors.color_pixel import REALBIND_COLORS
    families = {r["family_id"]: r for r in read_jsonl(Paths().families)}
    vocab = {
        "color": list(REALBIND_COLORS),
        "position": ["left", "right"],
        "identity": sorted({family[k]["object"] for family in families.values() for k in ("A", "B")}),
    }
    return families, vocab


def stage_dev_final():
    from harness.evaluators.end2end_metrics import score
    from harness.reasoners.qwen35 import Qwen35Reasoner
    from harness.scripts.run_minimal_decomposition_route_pilot import prompt as final_prompt

    prepared, prior = load_prepared()
    qa_hist = {r["qa_id"]: r for r in rows(HIST / "qa_results.jsonl")}
    final_hist = {r["qa_id"]: r for r in rows(HIST / "final_results.jsonl")}
    fusion = {r["qa_id"]: r for r in rows(OUT / "composition_results.jsonl")}
    fam, vocab = _load_vocab()
    gt = {qa_id: str(qa_hist[qa_id]["gt_color"]).strip().casefold() for qa_id in qa_hist}
    # evaluator annotation is joined only after fusion and composition files are frozen
    state_rows = []
    final_rows = []
    reasoner = None
    for qa_id in sorted(qa_hist):
        qa = qa_hist[qa_id]
        c = fusion[qa_id]
        target = c["target"]
        new_color = c["new_color"]
        state_text = f"[TARGET_ENTITY] {target if target else 'unknown'}\n[TARGET_ATTRIBUTE] color = {new_color if new_color else 'unknown'}"
        attr_correct = bool(new_color is not None and new_color == gt[qa_id])
        state_sufficient = bool(qa["logical_target_correct"] and qa["grounded_target_correct"] and attr_correct)
        state = {
            "qa_id": qa_id,
            "family_id": qa["family_id"],
            "qa_type": qa["qa_type"],
            "target": target,
            "true_target": qa.get("true_target"),
            "target_bbox_xyxy": qa.get("selected_target_bbox_xyxy"),
            "target_region_hash": sha256_json({"bbox": qa.get("selected_target_bbox_xyxy"), "policy": "frozen_strong_integrated_selected_region"}),
            "logical_target_correct": bool(qa["logical_target_correct"]),
            "grounded_target_correct": bool(qa["grounded_target_correct"]),
            "target_mask_valid": bool(qa.get("target_mask_valid")),
            "historical_color": c.get("historical_color"),
            "new_color": new_color,
            "target_attribute_correct": attr_correct,
            "minimal_state_sufficient": state_sufficient,
            "state_text": state_text,
            "state_sha256": sha256_text(state_text),
            "box_color": c.get("box_color"),
            "mask_color": c.get("mask_color"),
            "context_color": c.get("context_color"),
            "majority_color": c.get("majority_color"),
            "fallback_reason": c.get("fallback_reason"),
            "disagreement_pattern": c.get("disagreement_pattern"),
        }
        state_rows.append(state)
        query = final_prompt(prepared[qa_id]["question"], state_text)
        hist_final = final_hist[qa_id]
        if hist_final.get("state_text") == state_text:
            rec = {
                "qa_id": qa_id,
                "route": "TARGET_LOCAL_MULTIVIEW_REPAIR",
                "state_text": state_text,
                "state_sha256": state["state_sha256"],
                "state_sufficient": state_sufficient,
                "raw_response": hist_final.get("raw_response"),
                "prediction": hist_final.get("prediction"),
                "invalid_prediction": bool(hist_final.get("invalid_prediction")),
                "final_correct": bool(hist_final.get("final_correct")),
                "image_present": False,
                "vision_encoder_called": False,
                "final_cache_reused_from_historical": True,
            }
        else:
            if reasoner is None:
                reasoner = Qwen35Reasoner()
            gen = reasoner.generate(query, None)
            assert gen["image_path"] is None and not gen["vision_encoder_called"]
            scored = score(gen["response"], "color", gt[qa_id], fam[qa["family_id"]], vocab)
            rec = {
                "qa_id": qa_id,
                "route": "TARGET_LOCAL_MULTIVIEW_REPAIR",
                "state_text": state_text,
                "state_sha256": state["state_sha256"],
                "state_sufficient": state_sufficient,
                "raw_response": gen["response"],
                "prediction": scored["normalized_prediction"],
                "invalid_prediction": bool(scored["invalid_prediction"]),
                "final_correct": bool(scored["correct"]),
                "image_present": False,
                "vision_encoder_called": False,
                "final_cache_reused_from_historical": False,
            }
        final_rows.append(rec)
    write_jsonl(OUT / "minimal_state_results.jsonl", state_rows)
    write_jsonl(OUT / "final_predictions.jsonl", final_rows)
    # metrics and analyses
    result = _evaluate_dev(state_rows, final_rows, qa_hist, final_hist, fusion, gt)
    resume_update(status="DEV_FINAL_COMPLETE", dev_final_complete=True, **{k: result["metrics"][k] for k in ("minimal_state_sufficient", "clean_final_correct", "raw_final_correct")})
    print(f"[dev_final] state={result['metrics']['minimal_state_sufficient']} clean={result['metrics']['clean_final_correct']} raw={result['metrics']['raw_final_correct']}", flush=True)
    return result


def _mask_quality_metrics():
    from scipy import ndimage
    binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "binding_results.jsonl")}
    masks = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "mask_results.jsonl")}
    out = []
    for key in sorted(binding):
        f, ref = key
        m = masks[key]
        bbox = binding[key]["selected_bbox_xyxy"]
        rec = {
            "family_id": f, "logical_ref": ref,
            "selected_bbox_xyxy": bbox,
            "mask_valid_historical": bool(m.get("mask_valid")),
            "mask_area_ratio": None,
            "connected_components": None,
            "largest_component_fraction": None,
            "boundary_touch_fraction": None,
        }
        if not m.get("mask_valid") or not m.get("mask_path") or not Path(m["mask_path"]).exists():
            out.append(rec)
            continue
        image = np.asarray(Image.open(m["mask_path"]).convert("L")) > 0
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
        h, w = image.shape
        x0, y0 = max(0, min(w, x0)), max(0, min(h, y0))
        x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
        bbox_area = max(0, x1 - x0) * max(0, y1 - y0)
        area = int(image.sum())
        rec["mask_area_ratio"] = area / bbox_area if bbox_area else None
        labels, count = ndimage.label(image)
        rec["connected_components"] = int(count)
        if area > 0 and count > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            rec["largest_component_fraction"] = float(sizes.max() / area)
        # fixed 2-pixel inner boundary band of the bbox; fraction of mask area touching it
        band = np.zeros_like(image, dtype=bool)
        if x1 > x0 and y1 > y0:
            band[max(0, y0):min(h, y0 + 2), x0:x1] = True
            band[max(0, y1 - 2):min(h, y1), x0:x1] = True
            band[y0:y1, max(0, x0):min(w, x0 + 2)] = True
            band[y0:y1, max(0, x1 - 2):min(w, x1)] = True
        if area > 0:
            rec["boundary_touch_fraction"] = float((image & band).sum() / area)
        out.append(rec)
    write_jsonl(OUT / "mask_quality_metrics.jsonl", out)
    return out


def _leakage_audit():
    forbidden = {"gt_color", "ground_truth", "true_target", "target_attribute_correct", "grounded_target_correct", "final_correct", "EVALUATOR_ONLY", "evaluator_only", "answer"}
    names = ["view_manifest.jsonl", "bbox_view_predictions.jsonl", "mask_view_predictions.jsonl", "context_view_predictions.jsonl", "majority_fusion_results.jsonl", "pipeline_config.json"]
    violations = []
    scanned = 0
    for name in names:
        path = OUT / name
        if not path.exists():
            continue
        recs = rows(path) if path.suffix == ".jsonl" else [read_json(path)]
        for rec in recs:
            stack = [rec]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        scanned += 1
                        if k in forbidden:
                            violations.append({"artifact": name, "key": k})
                        stack.append(v)
                elif isinstance(obj, list):
                    stack.extend(obj)
    report = {
        "passed": len(violations) == 0,
        "forbidden_keys": sorted(forbidden),
        "violations": violations[:100],
        "scanned_values": scanned,
        "notes": [
            "Fusion and composition color choices are written before evaluator color labels are joined.",
            "The frozen upstream selected target region is read only from Strong Integrated Repair artifacts.",
        ],
    }
    write_json(OUT / "gt_leakage_audit.json", report)
    return report


def _evaluate_dev(state_rows, final_rows, qa_hist, final_hist, fusion, gt):
    state_by_qa = {r["qa_id"]: r for r in state_rows}
    final_by_qa = {r["qa_id"]: r for r in final_rows}
    composition_by_qa = fusion
    logical = sum(r["logical_target_correct"] for r in state_rows)
    grounded = sum(r["grounded_target_correct"] for r in state_rows)
    assert grounded == 26, f"GroundedTargetCorrect drift: {grounded} != 26"
    observation_available = sum(
        1 for r in state_rows
        if r["logical_target_correct"] and r["grounded_target_correct"] and composition_by_qa[r["qa_id"]]["target"] is not None
    )
    attr_correct = sum(r["target_attribute_correct"] for r in state_rows)
    minimal = sum(r["minimal_state_sufficient"] for r in state_rows)
    raw = sum(r["final_correct"] for r in final_rows)
    clean = sum(state_by_qa[r["qa_id"]]["minimal_state_sufficient"] and r["final_correct"] for r in final_rows)
    compensating = sum((not state_by_qa[r["qa_id"]]["minimal_state_sufficient"]) and r["final_correct"] for r in final_rows)
    metrics = {
        "n_qa": 40,
        "logical_target_correct": logical,
        "grounded_target_correct": grounded,
        "target_observation_available": observation_available,
        "target_attribute_correct": attr_correct,
        "minimal_state_sufficient": minimal,
        "clean_final_correct": clean,
        "raw_final_correct": raw,
        "compensating_error_correct": compensating,
        "raw_minus_clean": raw - clean,
    }
    # grounded target subset view accuracy
    grounded_ids = [r["qa_id"] for r in state_rows if r["logical_target_correct"] and r["grounded_target_correct"]]
    view_acc = {v: 0 for v in ("historical", "box", "mask", "context", "majority_defined", "final_fusion")}
    view_den = {v: 0 for v in view_acc}
    mask_valid_n = 0
    mask_invalid_n = 0
    for qa_id in grounded_ids:
        c = composition_by_qa[qa_id]
        truth = gt[qa_id]
        for v, color in (("historical", c.get("historical_color")), ("box", c.get("box_color")), ("context", c.get("context_color")), ("final_fusion", c.get("new_color"))):
            view_den[v] += 1
            if color is not None and str(color).strip().casefold() == truth:
                view_acc[v] += 1
        if c.get("mask_color") is not None:
            mask_valid_n += 1
            view_den["mask"] += 1
            if str(c.get("mask_color")).strip().casefold() == truth:
                view_acc["mask"] += 1
        else:
            mask_invalid_n += 1
        if c.get("majority_color") is not None:
            view_den["majority_defined"] += 1
            if str(c.get("majority_color")).strip().casefold() == truth:
                view_acc["majority_defined"] += 1
    # disagreement patterns over all 40 composition QA
    patterns = Counter()
    for qa_id in sorted(state_by_qa):
        c = composition_by_qa[qa_id]
        patterns[c.get("disagreement_pattern")] += 1
    # clean transitions
    transitions = Counter()
    clean_success_cases = []
    for qa_id in sorted(state_by_qa):
        h = bool(final_hist[qa_id]["state_sufficient"] and final_hist[qa_id]["final_correct"])
        n = bool(state_by_qa[qa_id]["minimal_state_sufficient"] and final_by_qa[qa_id]["final_correct"])
        if h and n:
            transitions["both_clean_correct"] += 1
            clean_success_cases.append({"qa_id": qa_id, "status": "preserved" if final_hist[qa_id]["prediction"] == final_by_qa[qa_id]["prediction"] else "changed_but_still_correct", "historical_prediction": final_hist[qa_id]["prediction"], "new_prediction": final_by_qa[qa_id]["prediction"]})
        elif h and not n:
            transitions["historical_clean_correct_new_clean_wrong"] += 1
            clean_success_cases.append({"qa_id": qa_id, "status": "regressed", "historical_prediction": final_hist[qa_id]["prediction"], "new_prediction": final_by_qa[qa_id]["prediction"], "new_state_sufficient": state_by_qa[qa_id]["minimal_state_sufficient"], "new_final_correct": final_by_qa[qa_id]["final_correct"]})
        elif (not h) and n:
            transitions["historical_clean_wrong_new_clean_correct"] += 1
        else:
            transitions["both_clean_wrong"] += 1
    regression_count = transitions["historical_clean_correct_new_clean_wrong"]
    # failure rescue
    failure_tax = {r["qa_id"]: r for r in read_json(HIST / "failure_attribution.json")["rows"]}
    failure_types = ["ISOLATION_FAILURE", "ATTRIBUTE_FAILURE", "OTHER"]
    rescue = {t: {"historical_failures": 0, "rescued": 0, "regressions": 0, "cases": []} for t in failure_types}
    for qa_id in sorted(state_by_qa):
        ftype = failure_tax[qa_id]["first_failure"]
        bucket = ftype if ftype in ("ISOLATION_FAILURE", "ATTRIBUTE_FAILURE") else "OTHER"
        h = bool(final_hist[qa_id]["state_sufficient"] and final_hist[qa_id]["final_correct"])
        n = bool(state_by_qa[qa_id]["minimal_state_sufficient"] and final_by_qa[qa_id]["final_correct"])
        rescue[bucket]["historical_failures"] += 1
        if (not h) and n:
            rescue[bucket]["rescued"] += 1
        if h and (not n):
            rescue[bucket]["regressions"] += 1
        c = composition_by_qa[qa_id]
        rescue[bucket]["cases"].append({
            "qa_id": qa_id, "first_failure": ftype,
            "historical_color": c.get("historical_color"),
            "box_color": c.get("box_color"), "mask_color": c.get("mask_color"), "context_color": c.get("context_color"),
            "majority_color": c.get("majority_color"), "new_color": c.get("new_color"),
            "gt_color": gt[qa_id], "historical_clean": h, "new_clean": n,
            "rescued": bool((not h) and n), "regressed": bool(h and (not n)),
        })
    # compensating errors
    comp = [{
        "qa_id": qa_id,
        "state_sufficient": state_by_qa[qa_id]["minimal_state_sufficient"],
        "final_correct": final_by_qa[qa_id]["final_correct"],
        "compensating_error": bool((not state_by_qa[qa_id]["minimal_state_sufficient"]) and final_by_qa[qa_id]["final_correct"]),
    } for qa_id in sorted(state_by_qa)]
    # survival chain: cumulative given previous conditions
    prev_ids = {r["qa_id"] for r in state_rows if r["logical_target_correct"] and r["grounded_target_correct"]}
    observation = sum(1 for qa_id in prev_ids if composition_by_qa[qa_id]["new_color"] is not None)
    attr_given = sum(1 for qa_id in prev_ids if state_by_qa[qa_id]["target_attribute_correct"])
    minimal_given = sum(1 for qa_id in prev_ids if state_by_qa[qa_id]["minimal_state_sufficient"])
    clean_given = sum(1 for qa_id in prev_ids if state_by_qa[qa_id]["minimal_state_sufficient"] and final_by_qa[qa_id]["final_correct"])
    chain = [
        {"stage": "start", "count": 40, "condition": "locked 40 REL2C QA"},
        {"stage": "LogicalTargetCorrect", "count": logical, "condition": "historical deterministic target identity"},
        {"stage": "GroundedTargetCorrect", "count": grounded, "condition": "historical selected target region IoU>=0.5"},
        {"stage": "TargetObservationAvailable", "count": observation, "condition": "grounded target entity has at least one fused local view color"},
        {"stage": "TargetAttributeCorrectGivenPrevious", "count": attr_given, "condition": "new multi-view target color equals evaluator target color"},
        {"stage": "MinimalStateSufficient", "count": minimal_given, "condition": "logical target, grounded region and new target color correct"},
        {"stage": "CleanFinal", "count": clean_given, "condition": "minimal state sufficient and frozen text-only final answer correct"},
    ]
    counts = [x["count"] for x in chain]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts
    write_json(OUT / "clean_transition_analysis.json", {
        "counts": dict(transitions),
        "historical_clean_success_n": sum(1 for qa_id in state_by_qa if final_hist[qa_id]["state_sufficient"] and final_hist[qa_id]["final_correct"]),
        "regressions": regression_count,
        "cases": clean_success_cases,
        "primary_requirement_regression_le_1": regression_count <= 1,
    })
    write_json(OUT / "failure_rescue_analysis.json", {
        "failure_types": rescue,
        "definitions": {
            "historical_failures": "first-failure count in Strong Integrated failure_attribution.json",
            "rescued": "historical clean false -> new clean true",
            "regressions": "historical clean true -> new clean false",
        },
        "evaluator_only": True,
    })
    write_json(OUT / "compensating_error_analysis.json", {
        "raw_final_correct": raw,
        "clean_final_correct": clean,
        "raw_minus_clean": raw - clean,
        "compensating_error_count": compensating,
        "rows": comp,
    })
    write_json(OUT / "survival_chain.json", {"monotonic_funnel": chain, "unconditional_components": metrics})
    # upstream hash audit
    parent = read_json(OUT / "historical_parent_manifest.json")
    parent_by_key = {(r["family_id"], r["logical_ref"]): r for r in parent["entities"]}
    drift = []
    for r in state_rows:
        key = (r["family_id"], r["target"])
        if key not in parent_by_key:
            drift.append({"qa_id": r["qa_id"], "reason": "target_entity_missing_parent"})
            continue
        p = parent_by_key[key]
        if p["region_hash"] != r["target_region_hash"]:
            drift.append({"qa_id": r["qa_id"], "reason": "target_region_hash_changed", "parent": p["region_hash"], "new": r["target_region_hash"]})
    hist_binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST / "binding_results.jsonl")}
    parent_region = {(r["family_id"], r["logical_ref"]): r["region_hash"] for r in parent["entities"]}
    ab_hashes = {}
    ab_unchanged = True
    for key in sorted(hist_binding):
        h = sha256_json({"bbox": hist_binding[key]["selected_bbox_xyxy"], "policy": "frozen_strong_integrated_selected_region"})
        ab_hashes[f"{key[0]}|{key[1]}"] = {"historical": h, "new": parent_region[key], "unchanged": h == parent_region[key]}
        ab_unchanged = ab_unchanged and (h == parent_region[key])
    audit = {
        "passed": len(drift) == 0 and ab_unchanged,
        "grounded_target_correct_historical": 26,
        "grounded_target_correct_new": grounded,
        "grounded_target_correct_unchanged": grounded == 26,
        "selected_region_hashes_unchanged": len(drift) == 0,
        "A_B_selected_region_hashes_unchanged": ab_unchanged,
        "A_B_region_hashes": ab_hashes,
        "violations": drift,
        "invalid_upstream_drift": bool(drift or grounded != 26 or not ab_unchanged),
    }
    write_json(OUT / "upstream_hash_audit.json", audit)
    if audit["invalid_upstream_drift"]:
        resume_update(status="INVALID_UPSTREAM_DRIFT", invalid_upstream_drift=True, **metrics)
        raise RuntimeError("INVALID_UPSTREAM_DRIFT")
    # hard handoff
    hard = {
        "pass": all((not r["image_present"]) and (not r["vision_encoder_called"]) for r in final_rows),
        "rows": [{"qa_id": r["qa_id"], "image_present": r["image_present"], "vision_encoder_called": r["vision_encoder_called"], "state_sha256": r["state_sha256"]} for r in final_rows],
        "reused_historical_final": sum(bool(r.get("final_cache_reused_from_historical")) for r in final_rows),
    }
    write_json(OUT / "hard_handoff_audit.json", hard)
    leakage = _leakage_audit()
    _mask_quality_metrics()
    result = {
        "metrics": metrics,
        "view_accuracy_on_grounded_subset": {v: {"correct": view_acc[v], "n": view_den[v]} for v in view_acc},
        "grounded_mask_valid_n": mask_valid_n,
        "grounded_mask_invalid_n": mask_invalid_n,
        "disagreement_patterns": dict(patterns),
        "clean_transitions": dict(transitions),
        "failure_rescue": rescue,
        "compensating_error_count": compensating,
        "survival_chain": chain,
        "upstream_hash_audit": audit,
        "hard_handoff_audit": hard,
        "gt_leakage_audit": leakage,
    }
    write_json(OUT / "dev_metrics.json", result)
    return result


# ---------------------------------------------------------------------------
# atomic calibration (200 I2C atomic attribute QA)
# ---------------------------------------------------------------------------

def _atomic_targets():
    families = {r["family_id"]: r for r in rows(REALBIND / "families/dryrun_100.jsonl")}
    qas = rows(REALBIND / "qa/dryrun_100_qa.jsonl")
    targets = []
    for qa in qas:
        if qa.get("qa_type") not in ("I2C_A", "I2C_B"):
            continue
        family = families[qa["family_id"]]
        ref = qa["qa_type"].split("_")[1]
        entity = family[ref]
        bbox_xywh = [float(v) for v in entity["bbox"]]
        bbox = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
        targets.append({
            "qa_id": qa["qa_id"],
            "family_id": qa["family_id"],
            "logical_ref": ref,
            "object_phrase": entity["object"],
            "bbox_xyxy": bbox,
            "image_path": family["image_path"],
            "gt_color": qa["ground_truth"],
        })
    assert len(targets) == 200, len(targets)
    return targets


def stage_atomic_prepare():
    import torch
    import harness.scripts.run_strong_integrated_repair as frozen

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("atomic_views/box", "atomic_views/mask", "atomic_views/context"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    targets = _atomic_targets()
    manifest_path = OUT / "atomic_view_manifest.jsonl"
    done = {r["qa_id"] for r in rows(manifest_path)}
    if len(done) == len(targets):
        print("[atomic_prepare] recovered from cache", flush=True)
        return
    processor, model = frozen.load_sam3()
    by_family = defaultdict(list)
    for t in targets:
        by_family[t["family_id"]].append(t)
    records = []
    for index, (family_id, items) in enumerate(sorted(by_family.items()), 1):
        image_path = items[0]["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_size = list(image.size)
        with torch.inference_mode():
            image_inputs = processor(images=image, return_tensors="pt")
            vision_embeds = model.get_vision_features(pixel_values=image_inputs["pixel_values"].to("cuda"))
        for t in items:
            ref = t["logical_ref"]
            bbox = t["bbox_xyxy"]
            box_path = OUT / "atomic_views/box" / f"{family_id}__{ref}.png"
            image.crop(exact_box(image.size, bbox)).save(box_path, format="PNG")
            ctx_path = OUT / "atomic_views/context" / f"{family_id}__{ref}.png"
            image.crop(context_box(image.size, bbox)).save(ctx_path, format="PNG")
            mask_path = None
            mask_valid = False
            mask_stats = None
            cx = (bbox[0] + bbox[2]) / 2.0 / image.width
            cy = (bbox[1] + bbox[3]) / 2.0 / image.height
            w = (bbox[2] - bbox[0]) / image.width
            h = (bbox[3] - bbox[1]) / image.height
            if w > 0 and h > 0:
                text_inputs = processor(text=[t["object_phrase"]], return_tensors="pt")
                inputs = {
                    "vision_embeds": vision_embeds,
                    "input_ids": text_inputs["input_ids"].to("cuda"),
                    "attention_mask": text_inputs["attention_mask"].to("cuda"),
                    "input_boxes": torch.tensor([[[cx, cy, w, h]]], dtype=torch.float32, device="cuda"),
                    "input_boxes_labels": torch.tensor([[1]], dtype=torch.long, device="cuda"),
                }
                with torch.inference_mode():
                    outputs = model(**inputs)
                res = frozen._sam3_postprocess(processor, outputs, image.size)
                best = None
                best_value = -1e9
                for score, m, mb in zip(res["scores"], res["masks"], res["boxes"]):
                    mask_np = m.to(torch.uint8).cpu().numpy() > 0
                    stats = frozen._mask_stats(mask_np, bbox, image.size)
                    value = float(score) + 0.2 * (stats.get("mask_fraction_inside_bbox") or 0.0)
                    if value > best_value:
                        best_value = value
                        best = (float(score), mask_np, [float(v) for v in mb.tolist()], stats)
                if best is not None and frozen._mask_valid(best[3]):
                    mask_score, mask_np, mb, mask_stats = best
                    mask_path = OUT / "atomic_views/mask" / f"{family_id}__{ref}.png"
                    Image.fromarray((mask_np.astype(np.uint8) * 255)).save(mask_path, format="PNG")
                    mask_valid = True
            records.append({
                "qa_id": t["qa_id"],
                "family_id": family_id,
                "logical_ref": ref,
                "object_phrase": t["object_phrase"],
                "bbox_xyxy": bbox,
                "image_path": image_path,
                "image_size": image_size,
                "box_path": str(box_path),
                "mask_path": str(mask_path) if mask_valid else None,
                "context_path": str(ctx_path),
                "mask_valid": mask_valid,
                "mask_stats": mask_stats,
                "gt_color": t["gt_color"],
            })
        del vision_embeds
        torch.cuda.empty_cache()
        print(f"[atomic_prepare] {index}/{len(by_family)} families", flush=True)
    # order by original target order
    by_qa = {r["qa_id"]: r for r in records}
    ordered = [by_qa[t["qa_id"]] for t in targets]
    write_jsonl(manifest_path, ordered)
    resume_update(status="ATOMIC_PREPARED", atomic_prepare_complete=True, atomic_targets=len(ordered), atomic_mask_valid=sum(r["mask_valid"] for r in ordered))
    print(f"[atomic_prepare] complete {len(ordered)} targets mask_valid={sum(r['mask_valid'] for r in ordered)}", flush=True)


def stage_atomic_infer():
    from harness.reasoners.qwen35 import Qwen35Reasoner
    manifest = rows(OUT / "atomic_view_manifest.jsonl")
    existing = {(r["qa_id"], r["view"]): r for r in rows(OUT / "atomic_view_predictions.jsonl")}
    todo = []
    for m in manifest:
        for view, path_key, valid in (("box", "box_path", True), ("mask", "mask_path", m["mask_valid"]), ("context", "context_path", True)):
            if not valid or not m.get(path_key):
                continue
            key = (m["qa_id"], view)
            if key in existing and existing[key].get("status") == "OK":
                continue
            todo.append((m, view, m[path_key]))
    print(f"[atomic_infer] qwen color reader todo={len(todo)}", flush=True)
    q = Qwen35Reasoner()
    for index, (m, view, path) in enumerate(todo, 1):
        gen = q.generate(VLM_PROMPT, Path(path))
        color = parse_color(gen["response"])
        rec = {
            "qa_id": m["qa_id"], "family_id": m["family_id"], "logical_ref": m["logical_ref"], "view": view,
            "image_path": path, "image_sha256": sha256_file(path),
            "raw_response": gen["response"], "normalized_color": color, "valid": color in LEGAL_COLORS,
            "status": "OK",
        }
        append_jsonl(OUT / "atomic_view_predictions.jsonl", rec)
        existing[(m["qa_id"], view)] = rec
        if index % 20 == 0:
            print(f"[atomic_infer] {index}/{len(todo)}", flush=True)
    resume_update(status="ATOMIC_INFER_COMPLETE", atomic_infer_complete=True, atomic_predictions=len(existing))
    print(f"[atomic_infer] complete {len(existing)} predictions", flush=True)


def stage_atomic_report():
    manifest = {r["qa_id"]: r for r in rows(OUT / "atomic_view_manifest.jsonl")}
    preds = defaultdict(dict)
    for r in rows(OUT / "atomic_view_predictions.jsonl"):
        preds[r["qa_id"]][r["view"]] = r
    counts = Counter()
    view_correct = Counter()
    view_den = Counter()
    patterns = Counter()
    majority_defined = 0
    majority_correct = 0
    fallback_box_correct = 0
    mask_quality = []
    for qa_id, m in sorted(manifest.items()):
        colors = {
            "box": (preds[qa_id].get("box") or {}).get("normalized_color"),
            "mask": (preds[qa_id].get("mask") or {}).get("normalized_color") if m["mask_valid"] else None,
            "context": (preds[qa_id].get("context") or {}).get("normalized_color"),
        }
        valid = {k: (v in LEGAL_COLORS) for k, v in colors.items()}
        truth = str(m["gt_color"]).strip().casefold()
        for v, color in colors.items():
            if v == "mask" and not m["mask_valid"]:
                continue
            view_den[v] += 1
            if color == truth:
                view_correct[v] += 1
        counts["targets"] += 1
        counts["mask_valid"] += int(m["mask_valid"])
        counts["box_valid"] += int(valid["box"])
        counts["context_valid"] += int(valid["context"])
        vals = [c for c in colors.values() if c in LEGAL_COLORS]
        ctr = Counter(vals)
        maj = None
        for color, count in ctr.most_common(1):
            if count >= 2:
                maj = color
        if maj is not None:
            majority_defined += 1
            majority_correct += int(maj == truth)
            final = maj
        else:
            final = colors["box"] if colors["box"] in LEGAL_COLORS else None
        if final == truth:
            fallback_box_correct += 1
        if not m["mask_valid"]:
            patterns["invalid_view_exists"] += 1
        else:
            bx, mk, ct = colors["box"], colors["mask"], colors["context"]
            if bx is not None and bx == mk == ct:
                patterns["box_eq_mask_eq_context"] += 1
            elif bx == mk and bx != ct:
                patterns["box_eq_mask_ne_context"] += 1
            elif bx == ct and bx != mk:
                patterns["box_eq_context_ne_mask"] += 1
            elif mk == ct and mk != bx:
                patterns["mask_eq_context_ne_box"] += 1
            else:
                patterns["all_different"] += 1
        if m["mask_valid"] and m.get("mask_stats"):
            stats = m["mask_stats"]
            mask_quality.append({
                "mask_area_ratio_inside": stats.get("mask_area_over_bbox_area"),
                "mask_fraction_inside_bbox": stats.get("mask_fraction_inside_bbox"),
            })
    report = {
        "n_atomic_targets": counts["targets"],
        "n_families": 100,
        "atomic_qa_types": ["I2C_A", "I2C_B"],
        "view_valid_counts": {"box": counts["box_valid"], "mask": counts["mask_valid"], "context": counts["context_valid"]},
        "view_accuracy_raw": {v: {"correct": view_correct[v], "n": view_den[v]} for v in ("box", "mask", "context")},
        "view_accuracy_denominators": {"box": "all 200 targets", "mask": "97 valid-mask targets", "context": "all 200 targets"},
        "mask_valid_fraction": counts["mask_valid"] / max(1, counts["targets"]),
        "disagreement_patterns": dict(patterns),
        "majority_defined": majority_defined,
        "majority_coverage": majority_defined / max(1, counts["targets"]),
        "majority_accuracy_when_defined": majority_correct / max(1, majority_defined),
        "box_fallback_accuracy_all_targets": fallback_box_correct / max(1, counts["targets"]),
        "mask_validity_stats": {
            "n": len(mask_quality),
            "mean_mask_area_ratio_inside": sum((x["mask_area_ratio_inside"] or 0) for x in mask_quality) / max(1, len(mask_quality)),
            "mean_mask_fraction_inside_bbox": sum((x["mask_fraction_inside_bbox"] or 0) for x in mask_quality) / max(1, len(mask_quality)),
        },
        "protocol_validation": "The three fixed views were constructible, the frozen prompt was usable, mask validity used the existing deterministic rule, and disagreement patterns were measured without training or composition-set tuning.",
        "no_training": True,
        "no_composition_tuning": True,
    }
    write_json(OUT / "atomic_attribute_calibration.json", report)
    resume_update(status="ATOMIC_REPORT_COMPLETE", atomic_report_complete=True)
    print("[atomic_report] written", flush=True)


# ---------------------------------------------------------------------------
# final report
# ---------------------------------------------------------------------------

def _decision(dev, clean_trans, upstream):
    if upstream.get("invalid_upstream_drift"):
        return "INVALID_UPSTREAM_DRIFT"
    m = dev["metrics"]
    clean = m["clean_final_correct"]
    regression = clean_trans.get("regressions", 0)
    if clean >= 22 and regression <= 1:
        return "TARGET_LOCAL_REPAIR_ESTABLISHED"
    if clean > 20:
        if regression <= 1 and m["minimal_state_sufficient"] >= m["clean_final_correct"]:
            return "CLEAN_BASELINE_BREAKING_REPAIR"
        return "TARGET_LOCAL_GAIN_FRAGILE"
    if clean == 21:
        return "TARGET_LOCAL_GAIN_FRAGILE"
    return "TARGET_LOCAL_MULTIVIEW_NO_GAIN"


def stage_report():
    dev = read_json(OUT / "dev_metrics.json")
    atomic = read_json(OUT / "atomic_attribute_calibration.json")
    clean_trans = read_json(OUT / "clean_transition_analysis.json")
    rescue = read_json(OUT / "failure_rescue_analysis.json")
    upstream = read_json(OUT / "upstream_hash_audit.json")
    comp = read_json(OUT / "compensating_error_analysis.json")
    decision = _decision(dev, clean_trans, upstream)
    m = dev["metrics"]
    view = dev["view_accuracy_on_grounded_subset"]
    patterns = dev["disagreement_patterns"]
    lines = []
    lines.append("# Baseline-preserving target-local multi-view attribute repair")
    lines.append("")
    lines.append("**DEVELOPMENT-ONLY FROZEN COHORT.** The 20-family / 40-REL2C-QA subset is reused. This is not held-out paper generalization evidence.")
    lines.append("")
    lines.append("## Main comparison")
    lines.append("")
    lines.append("| Method | GroundedTargetCorrect | MinimalStateSufficient | CleanFinal | RawFinal |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append("| Vanilla VLM | — | — | — | 20/40 |")
    lines.append("| Strong Integrated | 26/40 | 20/40 | 20/40 | 25/40 |")
    lines.append(f"| Multi-View Attribute Repair | {m['grounded_target_correct']}/40 | {m['minimal_state_sufficient']}/40 | {m['clean_final_correct']}/40 | {m['raw_final_correct']}/40 |")
    lines.append("")
    lines.append("## Required view table (evaluator-confirmed correct target subset, n = 26)")
    lines.append("")
    lines.append("| View | Color Correct |")
    lines.append("|---|---:|")
    for key, label in (("historical", "Historical current"), ("box", "Exact bbox"), ("mask", "SAM3 isolated (valid-mask targets)"), ("context", "Local context"), ("majority_defined", "2-of-3 majority (defined cases)"), ("final_fusion", "Final fusion (with historical fallback)")):
        v = view[key]
        lines.append(f"| {label} | {v['correct']}/{v['n']} |")
    lines.append("")
    lines.append("## Required disagreement table (all 40 target QA)")
    lines.append("")
    lines.append("| Pattern | Count |")
    lines.append("|---|---:|")
    for key in ("box_eq_mask_eq_context", "box_eq_mask_ne_context", "box_eq_context_ne_mask", "mask_eq_context_ne_box", "all_different", "invalid_view_exists"):
        lines.append(f"| {key} | {patterns.get(key, 0)} |")
    lines.append("")
    lines.append("## Required rescue table")
    lines.append("")
    lines.append("| Failure Type | Historical Failures | Rescued | Regressions |")
    lines.append("|---|---:|---:|---:|")
    for key in ("ISOLATION_FAILURE", "ATTRIBUTE_FAILURE", "OTHER"):
        v = rescue["failure_types"][key]
        lines.append(f"| {key} | {v['historical_failures']} | {v['rescued']} | {v['regressions']} |")
    lines.append("")
    lines.append("## Required clean transition table")
    lines.append("")
    lines.append("| Transition | Count |")
    lines.append("|---|---:|")
    for key in ("historical_clean_wrong_new_clean_correct", "historical_clean_correct_new_clean_wrong", "both_clean_correct", "both_clean_wrong"):
        lines.append(f"| {key} | {clean_trans['counts'].get(key, 0)} |")
    lines.append(f"Historical clean-success regressions: {clean_trans['regressions']}/20 (primary requirement <=1: {clean_trans['primary_requirement_regression_le_1']}).")
    lines.append("")
    lines.append("## Required state table")
    lines.append("")
    lines.append("| Method | MinimalStateSufficient | CleanFinal | RawFinal |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| Vanilla | — | — | 20/40 |")
    lines.append("| Strong Integrated | 20/40 | 20/40 | 25/40 |")
    lines.append(f"| Multi-View Repair | {m['minimal_state_sufficient']}/40 | {m['clean_final_correct']}/40 | {m['raw_final_correct']}/40 |")
    lines.append("")
    lines.append("## Survival chain")
    lines.append("")
    lines.append("| Stage | Count | Condition |")
    lines.append("|---|---:|---|")
    for step in dev["survival_chain"]:
        lines.append(f"| {step['stage']} | {step['count']} | {step['condition']} |")
    lines.append("")
    lines.append("## Atomic calibration")
    lines.append("")
    lines.append(f"- 200 I2C atomic attribute QA: box {atomic['view_accuracy_raw']['box']['correct']}/{atomic['view_accuracy_raw']['box']['n']}, mask {atomic['view_accuracy_raw']['mask']['correct']}/{atomic['view_accuracy_raw']['mask']['n']}, context {atomic['view_accuracy_raw']['context']['correct']}/{atomic['view_accuracy_raw']['context']['n']}.")
    lines.append(f"- Mask valid fraction: {atomic['mask_valid_fraction']:.3f}; majority defined coverage: {atomic['majority_coverage']:.3f}; majority accuracy when defined: {atomic['majority_accuracy_when_defined']:.3f}.")
    lines.append("")
    lines.append("## Required scientific questions")
    lines.append("")
    lines.append(f"**Q1. Do different views of the same correct target produce substantially different color predictions?** Disagreement patterns over 40 composition targets: {patterns}. This is reported without GT.")
    lines.append(f"**Q2. Can bbox/context views correct SAM3 isolated errors?** On grounded targets, mask view {view['mask']['correct']}/{view['mask']['n']} (valid masks only), box view {view['box']['correct']}/{view['box']['n']}, context view {view['context']['correct']}/{view['context']['n']}; final fusion {view['final_fusion']['correct']}/{view['final_fusion']['n']}.")
    lines.append(f"**Q3. Can SAM3 mask view correct bbox background interference?** Mask view target-subset accuracy and case-level rescue analysis are in `failure_rescue_analysis.json`.")
    lines.append(f"**Q4. Does 2-of-3 majority beat the historical single-view strategy?** Historical current {view['historical']['correct']}/{view['historical']['n']} vs majority-defined {view['majority_defined']['correct']}/{view['majority_defined']['n']} and final fusion {view['final_fusion']['correct']}/{view['final_fusion']['n']} on grounded targets; full clean final moves 20/40 -> {m['clean_final_correct']}/40.")
    lines.append(f"**Q5. Does improvement come mainly from the original 3 isolation + 3 attribute failures?** Rescue table: isolation {rescue['failure_types']['ISOLATION_FAILURE']['rescued']}/{rescue['failure_types']['ISOLATION_FAILURE']['historical_failures']}, attribute {rescue['failure_types']['ATTRIBUTE_FAILURE']['rescued']}/{rescue['failure_types']['ATTRIBUTE_FAILURE']['historical_failures']}.")
    lines.append(f"**Q6. How many of the original 20 clean-success QA regress?** {clean_trans['regressions']}/20.")
    lines.append(f"**Q7. Does MinimalStateSufficient exceed 20/40?** {m['minimal_state_sufficient']}/40.")
    lines.append(f"**Q8. Does CleanFinal truly exceed Vanilla 20/40?** {'Yes' if m['clean_final_correct'] > 20 else 'No'} ({m['clean_final_correct']}/40).")
    lines.append(f"**Q9. Does RawFinal stay at least 25/40?** {'Yes' if m['raw_final_correct'] >= 25 else 'No'} ({m['raw_final_correct']}/40).")
    lines.append("")
    lines.append("## Decision label")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append("## Compensating errors")
    lines.append("")
    lines.append(f"RawFinal {m['raw_final_correct']}/40, CleanFinal {m['clean_final_correct']}/40, compensating errors {m['compensating_error_correct']} (Raw - Clean = {m['raw_minus_clean']}).")
    lines.append("")
    lines.append("## Upstream audit")
    lines.append("")
    lines.append(f"GroundedTargetCorrect_new={upstream['grounded_target_correct_new']}/40; selected region hashes unchanged={upstream['selected_region_hashes_unchanged']}; invalid drift={upstream['invalid_upstream_drift']}.")
    (OUT / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # implementation report
    impl = [
        "# Implementation report: baseline-preserving target-local multi-view repair",
        "",
        "## Scope",
        "",
        "The experiment starts from the frozen Strong Integrated Repair parent and changes only the target-region -> target-color observation interface. Binding, grounding, target resolution, selected regions, minimal-state format and the frozen text-only final reasoner are reused exactly.",
        "",
        "## Views",
        "",
        "- V_box: exact frozen target bbox crop.",
        "- V_mask: frozen Strong Integrated SAM3 isolated view (same checkpoint, box prompt, thresholds, 16 px dilation and neutral background policy).",
        f"- V_context: target bbox expanded by fixed padding {PADDING}; clipped to image bounds.",
        "- Development V_box/V_mask color predictions are reused from the frozen Strong Integrated attribute artifacts; V_context is newly inferred with the same reader and prompt.",
        "- Atomic calibration runs all three views freshly on the 200 I2C atomic attribute QA and uses them only for protocol validation.",
        "- All views use the same frozen Qwen3.5-4B color reader and the same generic legal-color prompt.",
        "",
        "## Fusion",
        "",
        "- Primary rule: 2-of-3 majority among valid views.",
        "- Conservative fallback: no majority or insufficient valid views -> historical Strong Integrated target color.",
        "- No per-case routing, no threshold search, no prompt search, no GT-conditioned decisions.",
        "",
        "## Atomic calibration",
        "",
        f"- 200 I2C atomic attribute QA used only for protocol validation: {atomic['view_accuracy_raw']}, mask valid fraction {atomic['mask_valid_fraction']:.3f}.",
        "",
        "## Result",
        "",
        f"- CleanFinal {m['clean_final_correct']}/40, MinimalStateSufficient {m['minimal_state_sufficient']}/40, RawFinal {m['raw_final_correct']}/40.",
        f"- Historical clean-success regressions: {clean_trans['regressions']}/20.",
        f"- Decision: `{decision}`.",
        "",
        "## Audits",
        "",
        f"- GT leakage audit passed: {dev['gt_leakage_audit']['passed']}.",
        f"- Hard handoff audit passed: {dev['hard_handoff_audit']['pass']}.",
        f"- Upstream hash audit passed: {upstream['passed']}.",
    ]
    (OUT / "implementation_report.md").write_text("\n".join(impl) + "\n", encoding="utf-8")
    _write_audit_html(dev, atomic, clean_trans, rescue, decision)
    resume_update(status="COMPLETE", report_complete=True, decision_label=decision, clean_final_correct=m["clean_final_correct"], minimal_state_sufficient=m["minimal_state_sufficient"], raw_final_correct=m["raw_final_correct"])
    print(f"[report] decision={decision}", flush=True)


def _write_audit_html(dev, atomic, clean_trans, rescue, decision):
    def table(headers, body):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in body) + "</tbody></table>"
    m = dev["metrics"]
    view = dev["view_accuracy_on_grounded_subset"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Target-local multi-view audit</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #bbb;padding:.35rem .55rem}th{background:#eee}code{background:#f3f3f3;padding:.1rem .25rem}</style></head><body>",
        f"<h1>Target-local multi-view attribute repair</h1><p><b>Development-only frozen subset.</b> Decision: <code>{html.escape(decision)}</code></p>",
        "<h2>Main table</h2>",
        table(["Method", "GroundedTargetCorrect", "MinimalStateSufficient", "CleanFinal", "RawFinal"], [
            ["Vanilla", "—", "—", "—", "20/40"],
            ["Strong Integrated", "26/40", "20/40", "20/40", "25/40"],
            ["Multi-View Repair", f"{m['grounded_target_correct']}/40", f"{m['minimal_state_sufficient']}/40", f"{m['clean_final_correct']}/40", f"{m['raw_final_correct']}/40"],
        ]),
        "<h2>View accuracy on grounded targets</h2>",
        table(["View", "Correct"], [[k, f"{v['correct']}/{v['n']}"] for k, v in view.items()]),
        "<h2>Clean transitions</h2>",
        table(["Transition", "Count"], sorted(clean_trans["counts"].items())),
        "<h2>Failure rescue</h2>",
        table(["Failure", "Historical", "Rescued", "Regressions"], [[k, v["historical_failures"], v["rescued"], v["regressions"]] for k, v in rescue["failure_types"].items()]),
        "<h2>Atomic calibration</h2>",
        table(["View", "Correct"], [[k, f"{v['correct']}/{v['n']}"] for k, v in atomic["view_accuracy_raw"].items()]),
        "<h2>Artifacts</h2><ul>" + "".join(f"<li><a href='{html.escape(n)}'>{html.escape(n)}</a></li>" for n in ["comparison_report.md", "implementation_report.md", "pipeline_config.json", "historical_parent_manifest.json", "upstream_hash_audit.json", "view_manifest.jsonl", "bbox_view_predictions.jsonl", "mask_view_predictions.jsonl", "context_view_predictions.jsonl", "majority_fusion_results.jsonl", "mask_quality_metrics.jsonl", "atomic_attribute_calibration.json", "composition_results.jsonl", "clean_transition_analysis.json", "failure_rescue_analysis.json", "compensating_error_analysis.json", "minimal_state_results.jsonl", "final_predictions.jsonl", "survival_chain.json", "gt_leakage_audit.json", "hard_handoff_audit.json", "resume_status.json"]) + "</ul></body></html>",
    ]
    (OUT / "audit.html").write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "dev_views", "dev_context", "dev_fuse", "dev_final", "atomic_prepare", "atomic_infer", "atomic_report", "report", "all"])
    args = parser.parse_args()
    if args.stage == "prepare":
        stage_prepare()
    elif args.stage == "dev_views":
        stage_dev_views()
    elif args.stage == "dev_context":
        stage_dev_context()
    elif args.stage == "dev_fuse":
        stage_dev_fuse()
    elif args.stage == "dev_final":
        stage_dev_final()
    elif args.stage == "atomic_prepare":
        stage_atomic_prepare()
    elif args.stage == "atomic_infer":
        stage_atomic_infer()
    elif args.stage == "atomic_report":
        stage_atomic_report()
    elif args.stage == "report":
        stage_report()
    elif args.stage == "all":
        stage_prepare()
        stage_atomic_prepare()
        stage_atomic_infer()
        stage_atomic_report()
        stage_dev_views()
        stage_dev_context()
        stage_dev_fuse()
        stage_dev_final()
        stage_report()


if __name__ == "__main__":
    main()
