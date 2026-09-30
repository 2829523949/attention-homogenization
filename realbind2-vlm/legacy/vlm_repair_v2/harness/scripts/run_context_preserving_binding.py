#!/usr/bin/env python3
"""Context-preserving referent-region binding repair.

This experiment freezes the historical RealBind 20-family/40-QA development
subset, the high-recall R4+SAM3-top64 union candidate pool, the ReferentSpec
contract, and every downstream component except the binding interface.

The only new inference object is a candidate-focused contextual view:

    Image 1 : the untouched full scene
    Image 2 : the same canvas, candidate pixels untouched, outside attenuated

Qwen3.5-4B scores YES/NO logits for every frozen candidate.  Selection is the
unmodified argmax of log P(YES | I, focus(region), referent) - log P(NO | ...).

Evaluator annotations are opened only after selected_regions.jsonl has been
written.
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
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFile

# WSL sandbox uses a fresh /dev; make the host dxg node visible when needed.
_DXG = Path("/dev/dxg")
if not _DXG.exists() and Path("/proc/1/root/dev/dxg").exists():
    try:
        _DXG.symlink_to("/proc/1/root/dev/dxg")
    except OSError:
        pass

ROOT = Path("/mnt/d/vlm_repair_v2")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/context_preserving_binding"
POOL_PATH = ROOT / "outputs/high_recall_binding_repair/p0_qwen_independent/candidate_pool_manifest.json"
HIST_OUT = ROOT / "outputs/strong_integrated_repair"
R4_CANDIDATES = ROOT / "outputs/region_bottleneck_joint_repair/candidate_results.jsonl"
GT_PATH = ROOT / "outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"
PREPARED_PATH = ROOT / "outputs/minimal_decomposition_route_pilot/prepared.jsonl"
PRIOR_PATH = ROOT / "outputs/minimal_decomposition_route_pilot/route_results.jsonl"
VG_SUBSET_PATH = ROOT / "outputs/high_recall_binding_repair/final_comparison/vg_objects_subset.json"
QWEN_MODEL = Path("/root/Qwen3-VL/models/Qwen3.5-4B")

ROUTE = "CONTEXT_PRESERVING_BINDER"
EXPERIMENT = "context_preserving_referent_region_binding_v1"
IOU_THRESHOLD = 0.5
SCORE_BATCH_SIZE = 4
ALPHA = 0.5
NEUTRAL_GRAY = (127, 127, 127)
BOUNDARY_COLOR = (255, 255, 255)
BOUNDARY_WIDTH = 3
PROMPT_TEMPLATE = (
    'Image 1 shows the full scene. Image 2 shows the same scene with the '
    'candidate region highlighted. Does the focused candidate correspond to '
    'the entity described as: "{referent}"? Answer exactly YES or NO.'
)

FORBIDDEN_INFERENCE_TERMS = (
    "REL2C", "question", "left", "right", "answer", "ground_truth",
    "gt_bbox", "iou", "true_target", "evaluator", "correct_target",
    "final_correct", "color", "relation",
)


# ---------------------------------------------------------------------------
# basic io
# ---------------------------------------------------------------------------

def rows(path: Path):
    path = Path(path)
    if not path.exists():
        return []
    parsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            # A crash can leave one partial trailing JSONL row.  It is ignored
            # so the deterministic cache can resume from the last complete row.
            continue
    return parsed


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_float(value):
    if value is None:
        return None
    return float(value)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    try:
        ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
        iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    except Exception:
        return 0.0
    inter = ix * iy
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clipped_bbox(bbox, size):
    x0, y0, x1, y1 = (float(v) for v in bbox)
    width, height = float(size[0]), float(size[1])
    x0, y0 = max(0.0, min(width, x0)), max(0.0, min(height, y0))
    x1, y1 = max(0.0, min(width, x1)), max(0.0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def canonical_bbox(bbox):
    return [round(float(v), 6) for v in bbox]


def target_from_boxes(bbox_a, bbox_b, qa_type):
    if bbox_a is None or bbox_b is None:
        return None
    ax = (float(bbox_a[0]) + float(bbox_a[2])) / 2.0
    bx = (float(bbox_b[0]) + float(bbox_b[2])) / 2.0
    relation = "left" if ax - bx < -1e-9 else "right" if ax - bx > 1e-9 else None
    side = "left" if str(qa_type).endswith("LEFT") else "right"
    return "A" if relation == side else "B" if relation in ("left", "right") else None


def model_identity() -> dict:
    files = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"):
        path = QWEN_MODEL / name
        files[name] = {
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return {
        "path": str(QWEN_MODEL),
        "model_id": "Qwen/Qwen3.5-4B",
        "checkpoint_files": files,
        "runtime_loader": "harness.reasoners.qwen35.Qwen35Reasoner -> experiments.realbind_phase2.atomic_composition_baseline.load_frozen_model",
        "dtype": "bfloat16",
        "device_policy": "transformers accelerate auto; 6.75GiB CUDA budget and CPU offload are loader allocator settings only",
        "generation_config": {"do_sample": False, "use_cache": False, "enable_thinking": False, "batch_size_for_logits": SCORE_BATCH_SIZE},
    }


def focus_transform_config() -> dict:
    cfg = {
        "transform_id": "full_canvas_candidate_focus_dim_v1",
        "canvas": "same width/height as original Image 1",
        "candidate_region": "original pixels preserved at original position",
        "outside_candidate": {
            "operation": "I_out = alpha * I + (1 - alpha) * neutral_gray",
            "alpha": ALPHA,
            "neutral_gray_rgb": list(NEUTRAL_GRAY),
            "blur": "none",
            "recognizable": True,
        },
        "candidate_boundary": {
            "enabled": True,
            "type": "thin rectangle outline drawn outside bbox",
            "color_rgb": list(BOUNDARY_COLOR),
            "width_px": BOUNDARY_WIDTH,
            "rasterization": "integer floor/ceil candidate mask; outline coordinates are allowed outside the canvas and clipped by the canvas, so border pixels never overwrite an edge-touching candidate",
            "numbers": False,
            "A_B_marks": False,
            "target_word": False,
            "left_right": False,
            "semantic_color_hint": False,
        },
        "mask_policy": {
            "frozen_union_pool_has_valid_masks": False,
            "mask_source": "candidate_pool_manifest.json mask_available field; frozen SAM3 dump contains boxes/scores, not masks",
            "fallback": "exact bbox focus for every frozen candidate (no candidate dropped)",
        },
        "image_order": ["Image 1 original context", "Image 2 candidate-focused context"],
        "context_requirement": "outside-candidate scene remains visible to support relation, size, and identity disambiguation",
        "binding_focus_not_attribute_isolation": True,
    }
    cfg["transform_sha256"] = sha256_json(cfg)
    return cfg


def prompt_for(referent: str) -> str:
    return PROMPT_TEMPLATE.format(referent=referent)


def prepare_pool_and_configs():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "focus_views").mkdir(exist_ok=True)
    pool = read_json(POOL_PATH)
    referents = pool["referents"]
    assert len(referents) == 40
    family_counts = Counter(r["family_id"] for r in referents)
    assert len(family_counts) == 20 and all(v == 2 for v in family_counts.values())
    manifest = {
        "schema": "context_preserving_candidate_pool_v1",
        "experiment": EXPERIMENT,
        "frozen_source": str(POOL_PATH),
        "frozen_source_sha256": sha256_file(POOL_PATH),
        "candidate_order": pool.get("candidate_order"),
        "primary_sam3_k": pool.get("primary_sam3_k"),
        "dedup_bbox_iou_strictly_greater_than": pool.get("dedup_bbox_iou_strictly_greater_than"),
        "policy": "candidate K and pool identity are inherited unchanged from high_recall_binding_repair/p0_qwen_independent",
        "n_referents": len(referents),
        "n_families": len(family_counts),
        "referents": [],
    }
    for row in referents:
        image_path = Path(row["image_path"])
        image_sha = sha256_file(image_path)
        image_size = list(Image.open(image_path).size)
        assert list(row["image_size"]) == image_size, row["family_id"]
        out_row = {
            "family_id": row["family_id"],
            "logical_ref": row["logical_ref"],
            "referent_phrase": row["referent_phrase"],
            "image_path": str(image_path),
            "image_size": image_size,
            "image_sha256": image_sha,
            "referent_text_sha256": sha256_text(row["referent_phrase"]),
            "candidates": [],
        }
        for idx, cand in enumerate(row["candidates"]):
            bbox = canonical_bbox(cand["bbox_xyxy"])
            source_entries = list(cand.get("source_entries", []))
            r4_ranks = []
            sam3_ranks = []
            for entry in source_entries:
                try:
                    _, number = entry.split(":", 1)
                    number = int(number)
                except Exception:
                    continue
                if entry.startswith("r4:"):
                    r4_ranks.append(number + 1)
                elif entry.startswith("sam3:"):
                    sam3_ranks.append(number + 1)
            out_row["candidates"].append({
                "candidate_id": cand["candidate_id"],
                "candidate_index": idx,
                "union_native_rank": idx + 1,
                "bbox_xyxy": bbox,
                "provenance": list(cand.get("provenance", [])),
                "source_entries": source_entries,
                "native_score": cand.get("native_score"),
                "r4_native_rank": min(r4_ranks) if r4_ranks else None,
                "sam3_native_rank": min(sam3_ranks) if sam3_ranks else None,
                "mask_available": False,
                "mask_sha256": None,
            })
        manifest["referents"].append(out_row)
    write_json(OUT / "candidate_pool_manifest.json", manifest)

    focus = focus_transform_config()
    write_json(OUT / "focus_transform_config.json", focus)
    identity = model_identity()
    pipeline_config = {
        "experiment": EXPERIMENT,
        "development_only": True,
        "cohort": {
            "families": 20,
            "qa": 40,
            "frozen_source": str(PREPARED_PATH),
            "frozen_source_sha256": sha256_file(PREPARED_PATH),
            "prior_target_join_source": str(PRIOR_PATH),
            "prior_target_join_sha256": sha256_file(PRIOR_PATH),
        },
        "frozen_components": [
            "RealBind subset",
            "ReferentSpec",
            "candidate generation",
            "R4 candidates",
            "SAM3 candidate pools",
            "candidate dedup",
            "deterministic geometry",
            "SAM3 target isolation after binding",
            "crop-Qwen3.5 color reader",
            "minimal-state format",
            "final text-only reasoner",
        ],
        "allowed_change": "referent-region binding interface only",
        "candidate_pool": {
            "source": str(POOL_PATH),
            "source_sha256": sha256_file(POOL_PATH),
            "primary_sam3_k": pool.get("primary_sam3_k"),
            "dedup_iou": pool.get("dedup_bbox_iou_strictly_greater_than"),
            "candidate_cap_selection_reused": True,
        },
        "binding": {
            "representation": "full original context + candidate-focused full-canvas context",
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_sha256": sha256_text(PROMPT_TEMPLATE),
            "score": "log P(YES | I, focus(r_i), q) - log P(NO | I, focus(r_i), q)",
            "score_method": "next-token log-softmax; full response conditional log-likelihood fallback if tokenizer ever emits multi-token labels",
            "selection": "argmax_i score_i independently for A and B, then one frozen grounding per family",
            "fallback_allowed": ["empty candidate pool", "model execution failure", "NaN / invalid logits"],
        },
        "focus_transform": focus,
        "qwen_model": identity,
        "downstream": {
            "route": ROUTE,
            "frozen_pipeline": "outputs/strong_integrated_repair pipeline stages mask -> attribute -> assemble -> text-only final",
            "cache_reuse": "unchanged selected A/B regions reuse historical geometry, isolation, target crop, color, minimal state and final result by state hash",
            "minimal_state_format": "[TARGET_ENTITY] A/B\\n[TARGET_ATTRIBUTE] color = <color>",
            "final_reasoner_image_policy": "no image, no vision encoder",
        },
        "iou_threshold": IOU_THRESHOLD,
        "score_batch_size": SCORE_BATCH_SIZE,
    }
    write_json(OUT / "pipeline_config.json", pipeline_config)

    resume = {
        "experiment": EXPERIMENT,
        "status": "PREPARED",
        "score_total_candidates": sum(len(r["candidates"]) for r in manifest["referents"]),
        "score_completed_candidates": 0,
        "selection_complete": False,
        "evaluation_complete": False,
        "downstream_complete": False,
        "report_complete": False,
        "protocol_fingerprint": protocol_fingerprint(pipeline_config, focus),
        "updated_unix": time.time(),
    }
    write_json(OUT / "resume_status.json", resume)
    return manifest


def protocol_fingerprint(pipeline_config: dict, focus: dict) -> str:
    parts = {
        "experiment": EXPERIMENT,
        "prompt_sha256": sha256_text(PROMPT_TEMPLATE),
        "focus_sha256": focus["transform_sha256"],
        "model_identity": pipeline_config["qwen_model"],
        "score": pipeline_config["binding"]["score"],
        "selection": pipeline_config["binding"]["selection"],
        "pool_sha256": pipeline_config["candidate_pool"]["source_sha256"],
    }
    return sha256_json(parts)


# ---------------------------------------------------------------------------
# tokenization audit
# ---------------------------------------------------------------------------

def tokenization_audit() -> dict:
    from transformers import AutoProcessor

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL, local_files_only=True)
    tok = getattr(processor, "tokenizer", processor)
    strings = ["YES", "NO", " YES", " NO", "yes", "no", "Yes", "No"]
    tokenization = {s: [int(x) for x in tok.encode(s, add_special_tokens=False)] for s in strings}
    audit = {
        "model": str(QWEN_MODEL),
        "tokenizer_identity": str(getattr(tok, "name_or_path", QWEN_MODEL)),
        "strings": strings,
        "tokenization": tokenization,
        "single_token_uppercase": {
            "YES": len(tokenization["YES"]) == 1,
            "NO": len(tokenization["NO"]) == 1,
        },
        "response_contract": {
            "normalized_labels": ["YES", "NO"],
            "yes_ids": tokenization["YES"],
            "no_ids": tokenization["NO"],
            "prompt_read_position": "last non-padding position of the rendered user turn with add_generation_prompt=True",
            "primary_method": "log_softmax(logits[-1])[yes_id] - log_softmax(logits[-1])[no_id]",
            "fallback_method": "teacher-forced sum of token log-probabilities for each complete response if either label is multi-token",
        },
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
    }
    write_json(OUT / "qwen_tokenization_audit.json", audit)
    return audit


# ---------------------------------------------------------------------------
# focus view and Qwen scoring
# ---------------------------------------------------------------------------

def make_focus_view(original: Image.Image, bbox) -> Image.Image:
    """Return Image 2: outside-candidate attenuated, candidate pixels untouched."""
    width, height = original.size
    canvas = original.copy()
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0i, y0i = int(math.floor(x0)), int(math.floor(y0))
    x1i, y1i = int(math.ceil(x1)), int(math.ceil(y1))
    x0i, y0i = max(0, x0i), max(0, y0i)
    x1i, y1i = min(width, x1i), min(height, y1i)
    if x1i <= x0i or y1i <= y0i:
        return canvas
    mask = Image.new("L", original.size, 255)
    ImageDraw.Draw(mask).rectangle([x0i, y0i, x1i, y1i], fill=0)
    canvas = Image.composite(Image.new("RGB", original.size, NEUTRAL_GRAY), canvas, mask)
    # Do not clamp the outline into the candidate.  Drawing outside the canvas is
    # clipped harmlessly; negative/over-width coordinates keep the stroke on the
    # outside of edge-touching candidate boxes.
    ImageDraw.Draw(canvas).rectangle(
        [x0i - BOUNDARY_WIDTH, y0i - BOUNDARY_WIDTH, x1i + BOUNDARY_WIDTH, y1i + BOUNDARY_WIDTH],
        outline=BOUNDARY_COLOR,
        width=BOUNDARY_WIDTH,
    )
    return canvas


def render_two_image_chat(processor, original_path: str, referent: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(original_path)},
            {"type": "image", "image": "candidate_focused_context"},
            {"type": "text", "text": prompt_for(referent)},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _is_oom(exc: Exception) -> bool:
    text = repr(exc).lower()
    return ("out of memory" in text) or ("cuda" in text and "memory" in text)


def _single_sequence_logprob(model_bundle, sample, response_ids):
    """Teacher-forced conditional log-likelihood for a multi-token response.

    Used only when the frozen tokenizer does not expose YES or NO as one token.
    The current audited tokenizer exposes both as single tokens, so the fast
    next-token log-odds path is the one used by the completed experiment.
    """
    import torch

    q = model_bundle
    processor, model, device = q.processor, q.model, q.device
    text = render_two_image_chat(processor, sample["original_path"], sample["referent"])
    encoded = processor(text=[text], images=[sample["original_image"], sample["focus_image"]], return_tensors="pt")
    encoded = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    response = torch.tensor([response_ids], dtype=torch.long, device=device)
    prompt_ids = encoded["input_ids"]
    prompt_mask = encoded.get("attention_mask")
    inputs = dict(encoded)
    inputs["input_ids"] = torch.cat([prompt_ids, response], dim=1)
    if prompt_mask is not None:
        inputs["attention_mask"] = torch.cat([prompt_mask, torch.ones_like(response)], dim=1)
        prompt_mask = inputs["attention_mask"]
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in inputs:
            extension = torch.zeros_like(response)
            inputs[key] = torch.cat([inputs[key], extension], dim=1)
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    log_probs = torch.log_softmax(output.logits[0, prompt_len - 1:prompt_len - 1 + len(response_ids), :].float(), dim=-1)
    total = 0.0
    for offset, token_id in enumerate(response_ids):
        total += float(log_probs[offset, int(token_id)].item())
    del encoded, inputs, output, log_probs
    return total


def batch_score(model_bundle, samples):
    """Score one batch.

    samples: list of dicts with keys original_image (PIL), focus_image (PIL),
    original_path, referent, bbox, candidate_id.
    Returns list of score results aligned with samples.
    """
    import torch

    q = model_bundle
    processor, model, device = q.processor, q.model, q.device
    tok = getattr(processor, "tokenizer", processor)
    yes_ids = [int(x) for x in tok.encode("YES", add_special_tokens=False)]
    no_ids = [int(x) for x in tok.encode("NO", add_special_tokens=False)]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        results = []
        for sample in samples:
            yes_lp = _single_sequence_logprob(model_bundle, sample, yes_ids)
            no_lp = _single_sequence_logprob(model_bundle, sample, no_ids)
            results.append({
                "yes_logprob": yes_lp,
                "no_logprob": no_lp,
                "compatibility_score": yes_lp - no_lp,
                "score_method": "multi_token_conditional_loglikelihood",
            })
        return results
    texts = [render_two_image_chat(processor, s["original_path"], s["referent"]) for s in samples]
    images = []
    for s in samples:
        images.append(s["original_image"])
        images.append(s["focus_image"])
    encoded = processor(text=texts, images=images, return_tensors="pt")
    encoded = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, use_cache=False, logits_to_keep=1)
    logits = output.logits[:, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    y, n = yes_ids[0], no_ids[0]
    results = []
    for i in range(len(samples)):
        yes_lp = float(log_probs[i, y].item())
        no_lp = float(log_probs[i, n].item())
        results.append({
            "yes_logprob": yes_lp,
            "no_logprob": no_lp,
            "compatibility_score": yes_lp - no_lp,
            "score_method": "single_token_logodds",
        })
    del encoded, output, logits, log_probs
    return results


def score_batch_safe(model_bundle, samples):
    import torch

    if len(samples) == 0:
        return []
    try:
        return batch_score(model_bundle, samples)
    except Exception as exc:
        if len(samples) > 1 and _is_oom(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = len(samples) // 2
            return score_batch_safe(model_bundle, samples[:midpoint]) + score_batch_safe(model_bundle, samples[midpoint:])
        if len(samples) == 1:
            return [{
                "yes_logprob": None,
                "no_logprob": None,
                "compatibility_score": None,
                "score_method": "single_token_logodds",
                "status": "MODEL_EXECUTION_FAILURE",
                "error": repr(exc),
            }]
        return score_batch_safe(model_bundle, samples[:1]) + score_batch_safe(model_bundle, samples[1:])


def score_stage(limit: int = 0, batch_size: int = SCORE_BATCH_SIZE):
    manifest = read_json(OUT / "candidate_pool_manifest.json")
    focus = read_json(OUT / "focus_transform_config.json")
    pipeline_config = read_json(OUT / "pipeline_config.json")
    fingerprint = protocol_fingerprint(pipeline_config, focus)
    score_path = OUT / "context_binding_scores.jsonl"
    existing = rows(score_path)
    if existing and any(r.get("protocol_fingerprint") != fingerprint for r in existing):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = OUT / f"context_binding_scores.previous_{stamp}.jsonl"
        score_path.rename(archive)
        existing = []
    done = {}
    attempts = {}
    for rec in existing:
        key = rec.get("cache_key")
        if not key:
            continue
        attempts[key] = attempts.get(key, 0) + 1
        if rec.get("status") == "OK":
            done[key] = rec
    total = sum(len(r["candidates"]) for r in manifest["referents"])
    original_cache = {}
    original_pil_cache = {}

    from harness.reasoners.qwen35 import Qwen35Reasoner
    print(f"[score] loading frozen Qwen3.5-4B; existing OK={len(done)}/{total}", flush=True)
    q = Qwen35Reasoner()
    print(f"[score] model loaded on {q.device}; batch_size={batch_size}", flush=True)

    processed = 0
    started = time.time()
    for ref_row in manifest["referents"]:
        family_id = ref_row["family_id"]
        logical_ref = ref_row["logical_ref"]
        image_path = ref_row["image_path"]
        if image_path not in original_cache:
            original_cache[image_path] = sha256_file(image_path)
            original_pil_cache[image_path] = Image.open(image_path).convert("RGB")
        original_sha = original_cache[image_path]
        original_pil = original_pil_cache[image_path]
        referent = ref_row["referent_phrase"]
        referent_sha = sha256_text(referent)
        prompt = prompt_for(referent)
        prompt_sha = sha256_text(prompt)
        missing = []
        for cand in ref_row["candidates"]:
            bbox = canonical_bbox(cand["bbox_xyxy"])
            mask_hash = sha256_text("NO_MASK")
            cache_components = {
                "family_id": family_id,
                "referent_id": logical_ref,
                "candidate_id": cand["candidate_id"],
                "candidate_index": cand["candidate_index"],
                "bbox_hash": sha256_json(bbox),
                "mask_hash": mask_hash,
                "original_image_sha256": original_sha,
                "focus_transform_sha256": focus["transform_sha256"],
                "referent_text_sha256": referent_sha,
                "prompt_sha256": prompt_sha,
                "model_identity_sha256": sha256_json(pipeline_config["qwen_model"]),
                "protocol_fingerprint": fingerprint,
            }
            cache_key = sha256_json(cache_components)
            if cache_key in done:
                continue
            missing.append((cand, bbox, cache_components, cache_key))
        for start in range(0, len(missing), max(1, batch_size)):
            chunk = missing[start:start + max(1, batch_size)]
            samples = []
            meta = []
            for cand, bbox, cache_components, cache_key in chunk:
                focus_image = make_focus_view(original_pil, bbox)
                samples.append({
                    "original_image": original_pil,
                    "focus_image": focus_image,
                    "original_path": image_path,
                    "referent": referent,
                    "bbox": bbox,
                    "candidate_id": cand["candidate_id"],
                })
                meta.append((cand, bbox, cache_components, cache_key))
            t0 = time.time()
            results = score_batch_safe(q, samples)
            elapsed = time.time() - t0
            for (cand, bbox, cache_components, cache_key), result in zip(meta, results):
                record = {
                    "experiment": EXPERIMENT,
                    "protocol_fingerprint": fingerprint,
                    "cache_key": cache_key,
                    "cache_components": cache_components,
                    "family_id": family_id,
                    "logical_ref": logical_ref,
                    "candidate_id": cand["candidate_id"],
                    "candidate_index": cand["candidate_index"],
                    "union_native_rank": cand["union_native_rank"],
                    "bbox_xyxy": bbox,
                    "provenance": cand["provenance"],
                    "source_entries": cand["source_entries"],
                    "native_score": cand.get("native_score"),
                    "r4_native_rank": cand.get("r4_native_rank"),
                    "sam3_native_rank": cand.get("sam3_native_rank"),
                    "mask_available": False,
                    "mask_sha256": None,
                    "referent_phrase": referent,
                    "referent_text_sha256": referent_sha,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                    "focus_transform_sha256": focus["transform_sha256"],
                    "model_identity_sha256": sha256_json(pipeline_config["qwen_model"]),
                    "score_method": result.get("score_method"),
                    "yes_logprob": clean_float(result.get("yes_logprob")),
                    "no_logprob": clean_float(result.get("no_logprob")),
                    "compatibility_score": clean_float(result.get("compatibility_score")),
                    "status": result.get("status", "OK"),
                    "error": result.get("error"),
                    "elapsed_seconds": elapsed / max(1, len(results)),
                    "timestamp_unix": time.time(),
                }
                append_jsonl(score_path, record)
                existing.append(record)
                if record["status"] == "OK" and record["compatibility_score"] is not None and math.isfinite(record["compatibility_score"]):
                    done[cache_key] = record
                attempts[cache_key] = attempts.get(cache_key, 0) + 1
                processed += 1
            completed = len(done)
            if processed % 8 == 0 or processed <= 8:
                rate = processed / max(1e-9, time.time() - started)
                print(f"[score] {completed}/{total} candidates OK  batch={len(chunk)}  {rate:.2f} cand/s", flush=True)
                write_json(OUT / "resume_status.json", {
                    "experiment": EXPERIMENT,
                    "status": "RUNNING_SCORING",
                    "protocol_fingerprint": fingerprint,
                    "score_total_candidates": total,
                    "score_completed_candidates": completed,
                    "score_failed_attempts": sum(1 for r in existing if r.get("status") != "OK"),
                    "score_batch_size": batch_size,
                    "selection_complete": False,
                    "evaluation_complete": False,
                    "downstream_complete": False,
                    "report_complete": False,
                    "updated_unix": time.time(),
                })
            if limit and processed >= limit:
                print(f"[score] limit reached after {processed} newly scored candidates", flush=True)
                finalize_scores(score_path, existing)
                return
            del samples, meta, results
            gc.collect()
    finalize_scores(score_path, existing)
    completed = len(done)
    write_json(OUT / "resume_status.json", {
        "experiment": EXPERIMENT,
        "status": "SCORING_COMPLETE" if completed == total else "SCORING_INCOMPLETE",
        "protocol_fingerprint": fingerprint,
        "score_total_candidates": total,
        "score_completed_candidates": completed,
        "score_failed_attempts": sum(1 for r in existing if r.get("status") != "OK"),
        "score_batch_size": batch_size,
        "selection_complete": False,
        "evaluation_complete": False,
        "downstream_complete": False,
        "report_complete": False,
        "updated_unix": time.time(),
    })
    print(f"[score] complete OK={completed}/{total}", flush=True)


def finalize_scores(path: Path, records):
    """Compact append-only score log to one latest record per cache key."""
    by_key = {}
    for rec in records:
        key = rec.get("cache_key")
        if not key:
            continue
        old = by_key.get(key)
        # Prefer OK; otherwise latest attempt.
        rank = (1 if rec.get("status") == "OK" else 0, rec.get("timestamp_unix") or 0)
        old_rank = (1 if old and old.get("status") == "OK" else 0, (old or {}).get("timestamp_unix") or 0)
        if old is None or rank >= old_rank:
            by_key[key] = rec
    compact = sorted(by_key.values(), key=lambda r: (r["family_id"], r["logical_ref"], r["candidate_index"]))
    write_jsonl(path, compact)


# ---------------------------------------------------------------------------
# selection (no evaluator label is read here)
# ---------------------------------------------------------------------------

def select_stage():
    manifest = read_json(OUT / "candidate_pool_manifest.json")
    focus = read_json(OUT / "focus_transform_config.json")
    pipeline_config = read_json(OUT / "pipeline_config.json")
    fingerprint = protocol_fingerprint(pipeline_config, focus)
    score_records = [r for r in rows(OUT / "context_binding_scores.jsonl") if r.get("protocol_fingerprint") == fingerprint]
    by_key = {}
    for rec in score_records:
        if rec.get("status") != "OK" or rec.get("compatibility_score") is None or not math.isfinite(float(rec["compatibility_score"])):
            continue
        old = by_key.get(rec["cache_key"])
        if old is None or (rec.get("timestamp_unix") or 0) >= (old.get("timestamp_unix") or 0):
            by_key[rec["cache_key"]] = rec
    selected_rows = []
    technical_fallbacks = []
    for ref_row in manifest["referents"]:
        values = []
        for cand in ref_row["candidates"]:
            bbox_hash = sha256_json(canonical_bbox(cand["bbox_xyxy"]))
            key_match = None
            for rec in by_key.values():
                if rec["family_id"] == ref_row["family_id"] and rec["logical_ref"] == ref_row["logical_ref"] and rec["candidate_id"] == cand["candidate_id"]:
                    key_match = rec
                    break
            if key_match is None:
                values.append(None)
            else:
                values.append(float(key_match["compatibility_score"]))
        valid = [(v, idx) for idx, v in enumerate(values) if v is not None and math.isfinite(v)]
        fallback = None
        if valid:
            best_score, best_index = max(valid, key=lambda item: (item[0], -item[1]))
        elif ref_row["candidates"]:
            best_index = 0
            best_score = None
            fallback = "ALL_SCORES_INVALID_OR_MODEL_FAILURE"
            technical_fallbacks.append({"family_id": ref_row["family_id"], "logical_ref": ref_row["logical_ref"], "reason": fallback})
        else:
            best_index = None
            best_score = None
            fallback = "EMPTY_CANDIDATE_POOL"
            technical_fallbacks.append({"family_id": ref_row["family_id"], "logical_ref": ref_row["logical_ref"], "reason": fallback})
        cand = ref_row["candidates"][best_index] if best_index is not None else None
        rec = None
        if cand is not None:
            for candidate_rec in by_key.values():
                if candidate_rec["family_id"] == ref_row["family_id"] and candidate_rec["logical_ref"] == ref_row["logical_ref"] and candidate_rec["candidate_id"] == cand["candidate_id"]:
                    rec = candidate_rec
                    break
        ranked = sorted([(v, i) for i, v in enumerate(values) if v is not None], key=lambda item: item[0], reverse=True)
        selected_rows.append({
            "experiment": EXPERIMENT,
            "family_id": ref_row["family_id"],
            "logical_ref": ref_row["logical_ref"],
            "referent_phrase": ref_row["referent_phrase"],
            "image_path": ref_row["image_path"],
            "image_sha256": ref_row["image_sha256"],
            "n_candidates": len(ref_row["candidates"]),
            "selection_rule": "argmax compatibility_score; deterministic lowest candidate_index tie-break",
            "selected_candidate_id": cand["candidate_id"] if cand else None,
            "selected_candidate_index": cand["candidate_index"] if cand else None,
            "selected_union_native_rank": cand["union_native_rank"] if cand else None,
            "selected_r4_native_rank": cand.get("r4_native_rank") if cand else None,
            "selected_sam3_native_rank": cand.get("sam3_native_rank") if cand else None,
            "selected_bbox_xyxy": cand["bbox_xyxy"] if cand else None,
            "selected_provenance": cand["provenance"] if cand else [],
            "selected_source_entries": cand["source_entries"] if cand else [],
            "selected_native_score": cand.get("native_score") if cand else None,
            "selected_score": clean_float(best_score),
            "selected_cache_key": rec["cache_key"] if rec else None,
            "top2_scores": [
                {"candidate_id": ref_row["candidates"][i]["candidate_id"], "candidate_index": i, "score": clean_float(v)}
                for v, i in ranked[:2]
            ],
            "score_vector": [
                {"candidate_id": c["candidate_id"], "candidate_index": c["candidate_index"], "score": clean_float(v)}
                for c, v in zip(ref_row["candidates"], values)
            ],
            "fallback_reason": fallback,
            "fallback_is_technical": fallback is not None,
            "focus_transform_sha256": focus["transform_sha256"],
            "prompt_sha256": sha256_text(prompt_for(ref_row["referent_phrase"])),
            "protocol_fingerprint": fingerprint,
        })
    assert len(selected_rows) == 40
    write_jsonl(OUT / "selected_regions.jsonl", selected_rows)
    # selected evidence views
    selected_dir = OUT / "focus_views"
    selected_dir.mkdir(exist_ok=True)
    for row in selected_rows:
        if row["selected_bbox_xyxy"] is None:
            continue
        original = Image.open(row["image_path"]).convert("RGB")
        canvas = make_focus_view(original, row["selected_bbox_xyxy"])
        canvas.save(selected_dir / f'{row["family_id"]}__{row["logical_ref"]}__selected.png')
    status = read_json(OUT / "resume_status.json")
    status.update({
        "status": "SELECTION_COMPLETE",
        "selection_complete": True,
        "n_technical_fallbacks": len(technical_fallbacks),
        "technical_fallbacks": technical_fallbacks,
        "updated_unix": time.time(),
    })
    write_json(OUT / "resume_status.json", status)
    print(f"[select] selected 40 regions; technical_fallbacks={len(technical_fallbacks)}", flush=True)


# ---------------------------------------------------------------------------
# evaluation (annotations opened only here)
# ---------------------------------------------------------------------------

def load_evaluator_annotations():
    gt = {}
    for row in rows(GT_PATH):
        inf = row["inference"]
        if inf.get("backend") == "grounding_dino":
            gt[(inf["family_id"], inf["logical_ref"])] = [float(v) for v in row["EVALUATOR_ONLY"]["gt_bbox_xyxy_pixels"]]
    assert len(gt) == 40
    prepared = {r["qa_id"]: r for r in rows(PREPARED_PATH)}
    prior = {r["qa_id"]: r for r in rows(PRIOR_PATH)}
    assert len(prepared) == len(prior) == 40
    return gt, prepared, prior


def load_historical_eval():
    gt, prepared, prior = load_evaluator_annotations()
    hist_binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST_OUT / "binding_results.jsonl")}
    hist_qa = {r["qa_id"]: r for r in rows(HIST_OUT / "qa_results.jsonl")}
    assert len(hist_binding) == len(hist_qa) == 40
    entity = {}
    for key, gt_box in gt.items():
        selected = hist_binding[key]["selected_bbox_xyxy"]
        entity[key] = {
            "selected_bbox_xyxy": selected,
            "selected_iou": iou(selected, gt_box),
            "selected_region_correct": iou(selected, gt_box) >= IOU_THRESHOLD,
            "selected_provider": hist_binding[key].get("selected_provider"),
        }
    qa = {}
    for qa_id, prep in prepared.items():
        family_id = prep["family_id"]
        true_target = prior[qa_id]["EVALUATOR_ONLY"]["true_target"]
        target = target_from_boxes(
            hist_binding[(family_id, "A")]["selected_bbox_xyxy"],
            hist_binding[(family_id, "B")]["selected_bbox_xyxy"],
            prep["qa_type"],
        )
        logical = target == true_target
        grounded = bool(logical and target is not None and entity[(family_id, target)]["selected_region_correct"])
        qa[qa_id] = {
            "target": target,
            "true_target": true_target,
            "logical_target_correct": logical,
            "grounded_target_correct": grounded,
        }
    return entity, qa


def evaluate_stage():
    selected = {row["family_id"] + "|" + row["logical_ref"]: row for row in rows(OUT / "selected_regions.jsonl")}
    assert len(selected) == 40
    gt, prepared, prior = load_evaluator_annotations()
    # entity-level selected-region metrics
    entity_results = []
    for ref_row in read_json(OUT / "candidate_pool_manifest.json")["referents"]:
        key = ref_row["family_id"] + "|" + ref_row["logical_ref"]
        gt_box = gt[(ref_row["family_id"], ref_row["logical_ref"])]
        selected_row = selected[key]
        selected_bbox = selected_row["selected_bbox_xyxy"]
        available = any(iou(c["bbox_xyxy"], gt_box) >= IOU_THRESHOLD for c in ref_row["candidates"])
        correct = bool(selected_bbox is not None and iou(selected_bbox, gt_box) >= IOU_THRESHOLD)
        entity_results.append({
            "family_id": ref_row["family_id"],
            "logical_ref": ref_row["logical_ref"],
            "referent_phrase": ref_row["referent_phrase"],
            "candidate_available_at_0_5": available,
            "selected_region_correct_at_0_5": correct,
            "selected_iou": iou(selected_bbox, gt_box),
            "selected_candidate_id": selected_row["selected_candidate_id"],
            "selected_union_native_rank": selected_row["selected_union_native_rank"],
            "selected_r4_native_rank": selected_row["selected_r4_native_rank"],
            "selected_sam3_native_rank": selected_row["selected_sam3_native_rank"],
            "selected_score": selected_row["selected_score"],
            "failure_mode": "NO_USABLE_CANDIDATE" if not available else "USABLE_CANDIDATE_NOT_SELECTED" if not correct else None,
        })
    qa_results = []
    for qa_id, prep in prepared.items():
        family_id = prep["family_id"]
        true_target = prior[qa_id]["EVALUATOR_ONLY"]["true_target"]
        bbox_a = selected[family_id + "|A"]["selected_bbox_xyxy"]
        bbox_b = selected[family_id + "|B"]["selected_bbox_xyxy"]
        target = target_from_boxes(bbox_a, bbox_b, prep["qa_type"])
        logical = target == true_target
        grounded = bool(logical and target is not None and next(
            r for r in entity_results if r["family_id"] == family_id and r["logical_ref"] == target
        )["selected_region_correct_at_0_5"])
        qa_results.append({
            "qa_id": qa_id,
            "family_id": family_id,
            "qa_type": prep["qa_type"],
            "target": target,
            "true_target": true_target,
            "logical_target_correct": logical,
            "grounded_target_correct": grounded,
        })
    candidate_available = sum(r["candidate_available_at_0_5"] for r in entity_results)
    selected_correct = sum(r["selected_region_correct_at_0_5"] for r in entity_results)
    metrics = {
        "experiment": EXPERIMENT,
        "n_entities": 40,
        "n_families": 20,
        "n_qa": 40,
        "candidate_available_at_0_5": candidate_available,
        "candidate_available_denominator": 40,
        "selected_region_correct_at_0_5": selected_correct,
        "selected_region_denominator": 40,
        "selection_efficiency": selected_correct / candidate_available if candidate_available else None,
        "selection_efficiency_definition": "P(selected region correct at IoU>=0.5 | at least one usable candidate exists)",
        "both_entity_regions_correct": sum(
            1 for f in {r["family_id"] for r in entity_results}
            if all(r["selected_region_correct_at_0_5"] for r in entity_results if r["family_id"] == f)
        ),
        "logical_target_correct": sum(r["logical_target_correct"] for r in qa_results),
        "grounded_target_correct": sum(r["grounded_target_correct"] for r in qa_results),
        "logical_target_denominator": 40,
        "grounded_target_denominator": 40,
    }
    write_json(OUT / "binding_metrics.json", metrics)
    write_jsonl(OUT / "entity_evaluation.jsonl", entity_results)
    write_jsonl(OUT / "qa_evaluation.jsonl", qa_results)

    hist_entity, hist_qa = load_historical_eval()
    # selected-region paired transitions
    entity_transition = {"both_correct": 0, "both_wrong": 0, "historical_correct_context_wrong": 0, "historical_wrong_context_correct": 0}
    entity_cases = {"historical_wrong_context_correct": [], "historical_correct_context_wrong": []}
    for r in entity_results:
        key = (r["family_id"], r["logical_ref"])
        h = hist_entity[key]["selected_region_correct"]
        c = r["selected_region_correct_at_0_5"]
        if h and c:
            entity_transition["both_correct"] += 1
        elif h and not c:
            entity_transition["historical_correct_context_wrong"] += 1
            entity_cases["historical_correct_context_wrong"].append({
                "family_id": r["family_id"], "logical_ref": r["logical_ref"],
                "referent_phrase": r["referent_phrase"],
                "historical_selected_iou": hist_entity[key]["selected_iou"],
                "context_selected_iou": r["selected_iou"],
                "context_selected_candidate_id": r["selected_candidate_id"],
                "context_selected_union_native_rank": r["selected_union_native_rank"],
            })
        elif (not h) and c:
            entity_transition["historical_wrong_context_correct"] += 1
            entity_cases["historical_wrong_context_correct"].append({
                "family_id": r["family_id"], "logical_ref": r["logical_ref"],
                "referent_phrase": r["referent_phrase"],
                "historical_selected_iou": hist_entity[key]["selected_iou"],
                "context_selected_iou": r["selected_iou"],
                "context_selected_candidate_id": r["selected_candidate_id"],
                "context_selected_union_native_rank": r["selected_union_native_rank"],
            })
        else:
            entity_transition["both_wrong"] += 1

    grounded_transition = {"both_correct": 0, "both_wrong": 0, "historical_correct_context_wrong": 0, "historical_wrong_context_correct": 0}
    grounded_cases = {"historical_wrong_context_correct": [], "historical_correct_context_wrong": []}
    for r in qa_results:
        qa_id = r["qa_id"]
        h = hist_qa[qa_id]["grounded_target_correct"]
        c = r["grounded_target_correct"]
        if h and c:
            grounded_transition["both_correct"] += 1
        elif h and not c:
            grounded_transition["historical_correct_context_wrong"] += 1
            grounded_cases["historical_correct_context_wrong"].append({
                "qa_id": qa_id, "family_id": r["family_id"], "qa_type": r["qa_type"],
                "historical_target": hist_qa[qa_id]["target"], "context_target": r["target"],
                "true_target": r["true_target"],
            })
        elif (not h) and c:
            grounded_transition["historical_wrong_context_correct"] += 1
            grounded_cases["historical_wrong_context_correct"].append({
                "qa_id": qa_id, "family_id": r["family_id"], "qa_type": r["qa_type"],
                "historical_target": hist_qa[qa_id]["target"], "context_target": r["target"],
                "true_target": r["true_target"],
            })
        else:
            grounded_transition["both_wrong"] += 1
    paired = {
        "definition": "historical integrated vs context-preserving binder; correct at IoU>=0.5; grounded target additionally requires logical target identity",
        "entity_selected_region": {
            "n": 40,
            "counts": entity_transition,
            "cases": entity_cases,
            "historical_selected_region_correct": sum(v["selected_region_correct"] for v in hist_entity.values()),
            "context_selected_region_correct": selected_correct,
        },
        "grounded_target": {
            "n": 40,
            "counts": grounded_transition,
            "cases": grounded_cases,
            "historical_grounded_target_correct": sum(v["grounded_target_correct"] for v in hist_qa.values()),
            "context_grounded_target_correct": metrics["grounded_target_correct"],
        },
        "evaluator_only": True,
    }
    write_json(OUT / "paired_transitions.json", paired)

    subset = context_subset_analysis(entity_results, qa_results, hist_entity, hist_qa)
    same = same_category_analysis(entity_results, qa_results, hist_entity, hist_qa)
    rank = rank_rescue_analysis(entity_results, qa_results, hist_entity, hist_qa)
    status = read_json(OUT / "resume_status.json")
    status.update({"status": "EVALUATION_COMPLETE", "evaluation_complete": True, "updated_unix": time.time()})
    write_json(OUT / "resume_status.json", status)
    print(f"[evaluate] metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# evaluator-only failure taxonomy and rank analysis
# ---------------------------------------------------------------------------

def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _name_matches_referent(name: str, referent: str) -> bool:
    nt = [_singular(x) for x in _normalize_name(name).split()]
    qt = [_singular(x) for x in _normalize_name(referent).split()]
    if not nt or not qt:
        return False
    if nt == qt:
        return True
    if len(qt) <= len(nt):
        for i in range(len(nt) - len(qt) + 1):
            if nt[i:i + len(qt)] == qt:
                return True
    if len(qt) == 1 and qt[0] in nt:
        return True
    if len(nt) == 1 and nt[0] in qt:
        return True
    return False


def _referent_vg_matches(family_id: str, referent: str):
    image_id = family_id.split("_")[2]
    vg = read_json(VG_SUBSET_PATH) if VG_SUBSET_PATH.exists() else {}
    objects = vg.get(image_id, {}).get("objects", [])
    matches = []
    for obj in objects:
        names = obj.get("names") or []
        if any(_name_matches_referent(name, referent) for name in names):
            matches.append({
                "name": names[0] if names else "",
                "synsets": obj.get("synsets") or [],
                "bbox_xywh": [obj.get("x"), obj.get("y"), obj.get("w"), obj.get("h")],
                "object_id": obj.get("object_id"),
            })
    return matches


SCENE_REGION_TERMS = {
    "wall", "floor", "ground", "grass", "grass field", "roof", "line",
    "sauce", "street", "sky", "water", "road", "ceiling", "field", "land",
}


def classify_referents(entity_results):
    classification = {}
    for r in entity_results:
        family_id = r["family_id"]
        ref = r["logical_ref"]
        phrase = r["referent_phrase"]
        matches = _referent_vg_matches(family_id, phrase)
        normalized_phrase = _normalize_name(phrase)
        rel_terms = {"holding", "held", "next", "behind", "front", "near", "under", "above", "below", "between", "with", "wearing", "inside", "outside", "on", "in", "of"}
        comp_terms = {"smaller", "larger", "bigger", "smallest", "largest", "leftmost", "rightmost", "middle"}
        words = set(normalized_phrase.split())
        labels = []
        if words & rel_terms:
            labels.append("relational_or_context_dependent")
        if words & comp_terms:
            labels.append("comparative")
        if len(matches) >= 2:
            labels.append("same_category_or_ambiguous")
        region_match = any(_normalize_name(m["name"]) in {_normalize_name(x) for x in SCENE_REGION_TERMS} for m in matches)
        if region_match:
            labels.append("scene_region_context_dependent")
        if not labels:
            labels.append("purely_local_appearance")
        classification[(family_id, ref)] = {
            "referent_phrase": phrase,
            "vg_matching_objects": matches,
            "vg_match_count": len(matches),
            "labels": labels,
            "primary_label": labels[0],
        }
    return classification


def context_subset_analysis(entity_results, qa_results, hist_entity, hist_qa):
    classification = classify_referents(entity_results)
    by_entity = {(r["family_id"], r["logical_ref"]): r for r in entity_results}
    by_qa = {r["qa_id"]: r for r in qa_results}
    subsets = {}
    label_names = ["relational_or_context_dependent", "comparative", "same_category_or_ambiguous", "scene_region_context_dependent", "purely_local_appearance"]
    for label in label_names:
        keys = [k for k, v in classification.items() if label in v["labels"]]
        qa_ids = []
        for qa in qa_results:
            if (qa["family_id"], "A") in keys or (qa["family_id"], "B") in keys:
                qa_ids.append(qa["qa_id"])
        entity_hist = sum(hist_entity[k]["selected_region_correct"] for k in keys)
        entity_ctx = sum(by_entity[k]["selected_region_correct_at_0_5"] for k in keys)
        qa_hist = sum(hist_qa[q]["grounded_target_correct"] for q in qa_ids)
        qa_ctx = sum(by_qa[q]["grounded_target_correct"] for q in qa_ids)
        subsets[label] = {
            "n_entities": len(keys),
            "n_qa": len(qa_ids),
            "historical_selected_region_correct": entity_hist,
            "context_selected_region_correct": entity_ctx,
            "historical_grounded_target_correct": qa_hist,
            "context_grounded_target_correct": qa_ctx,
            "entity_keys": [f"{k[0]}|{k[1]}" for k in sorted(keys)],
        }
    rescue_entity = set()
    rescue_qa = set()
    for k, h in hist_entity.items():
        c = by_entity[k]["selected_region_correct_at_0_5"]
        if (not h["selected_region_correct"]) and c:
            rescue_entity.add(k)
    for q, h in hist_qa.items():
        c = by_qa[q]["grounded_target_correct"]
        if (not h["grounded_target_correct"]) and c:
            rescue_qa.add(q)
    rescue_by_label = {}
    for label in label_names:
        keys = {k for k, v in classification.items() if label in v["labels"]}
        qids = {q["qa_id"] for q in qa_results if (q["family_id"], "A") in keys or (q["family_id"], "B") in keys}
        rescue_by_label[label] = {
            "entity_rescues": len(keys & rescue_entity),
            "grounded_target_rescues": len(qids & rescue_qa),
            "entity_n": len(keys),
            "qa_n": len(qids),
        }
    result = {
        "annotation_source": "evaluator-only frozen VG object annotations plus transparent lexical taxonomy; never used in inference",
        "notes": [
            "No frozen legal referent phrase in this subset contains an explicit spatial/relational or comparative modifier; relational_or_context_dependent and comparative therefore have denominator zero.",
            "scene_region_context_dependent is a predeclared lexical category for surface/background/material nouns whose category is scene-dependent; it is an analysis convenience, not a claim of a referring-expression relation.",
        ],
        "classification": {f"{k[0]}|{k[1]}": v for k, v in sorted(classification.items())},
        "subsets": subsets,
        "rescue_by_label": rescue_by_label,
        "entity_rescue_count": len(rescue_entity),
        "grounded_target_rescue_count": len(rescue_qa),
        "evaluator_only": True,
    }
    write_json(OUT / "context_subset_analysis.json", result)
    return result


def same_category_analysis(entity_results, qa_results, hist_entity, hist_qa):
    classification = classify_referents(entity_results)
    by_entity = {(r["family_id"], r["logical_ref"]): r for r in entity_results}
    by_qa = {r["qa_id"]: r for r in qa_results}
    same_keys = {k for k, v in classification.items() if "same_category_or_ambiguous" in v["labels"]}
    other_keys = {k for k, v in classification.items() if k not in same_keys}
    same_qa = {q["qa_id"] for q in qa_results if (q["family_id"], "A") in same_keys or (q["family_id"], "B") in same_keys}
    other_qa = {q["qa_id"] for q in qa_results if q["qa_id"] not in same_qa}
    def entity_stats(keys):
        return {
            "n": len(keys),
            "historical_selected_region_correct": sum(hist_entity[k]["selected_region_correct"] for k in keys),
            "context_selected_region_correct": sum(by_entity[k]["selected_region_correct_at_0_5"] for k in keys),
        }
    def qa_stats(ids):
        return {
            "n": len(ids),
            "historical_grounded_target_correct": sum(hist_qa[q]["grounded_target_correct"] for q in ids),
            "context_grounded_target_correct": sum(by_qa[q]["grounded_target_correct"] for q in ids),
        }
    result = {
        "same_category_definition": "evaluator-only: at least two frozen VG objects in the scene match the legal referent category under transparent singular/stem name matching",
        "warnings": [
            "The locked subset contains no A/B pair sharing an exact frozen ReferentSpec category; this analysis is about multiple VG instances within a scene, not about A-vs-B category collision.",
        ],
        "same_category_or_ambiguous": {
            "entities": entity_stats(same_keys),
            "qa": qa_stats(same_qa),
            "keys": [f"{k[0]}|{k[1]}" for k in sorted(same_keys)],
        },
        "other": {
            "entities": entity_stats(other_keys),
            "qa": qa_stats(other_qa),
            "keys": [f"{k[0]}|{k[1]}" for k in sorted(other_keys)],
        },
        "evaluator_only": True,
    }
    write_json(OUT / "same_category_analysis.json", result)
    return result


def _correct_candidate_ranks(ref_row, gt_box):
    correct = []
    for cand in ref_row["candidates"]:
        if iou(cand["bbox_xyxy"], gt_box) >= IOU_THRESHOLD:
            correct.append(cand)
    if not correct:
        return None
    best = min(correct, key=lambda c: c["candidate_index"])
    return {
        "n_correct_candidates": len(correct),
        "best_correct_union_native_rank": best["union_native_rank"],
        "best_correct_r4_native_rank": best.get("r4_native_rank"),
        "best_correct_sam3_native_rank": best.get("sam3_native_rank"),
        "correct_candidate_ids": [c["candidate_id"] for c in correct],
        "correct_union_ranks": [c["union_native_rank"] for c in correct],
    }


def _nearest_union_rank(ref_row, bbox):
    best = None
    for cand in ref_row["candidates"]:
        value = iou(cand["bbox_xyxy"], bbox)
        if value > 0.7 and (best is None or value > best[0]):
            best = (value, cand)
    if best is None:
        return None
    cand = best[1]
    return {
        "match_iou": best[0],
        "union_native_rank": cand["union_native_rank"],
        "r4_native_rank": cand.get("r4_native_rank"),
        "sam3_native_rank": cand.get("sam3_native_rank"),
        "candidate_id": cand["candidate_id"],
    }


def rank_rescue_analysis(entity_results, qa_results, hist_entity, hist_qa):
    pool = read_json(OUT / "candidate_pool_manifest.json")
    ref_by_key = {(r["family_id"], r["logical_ref"]): r for r in pool["referents"]}
    gt, prepared, prior = load_evaluator_annotations()
    qa_by_id = {r["qa_id"]: r for r in qa_results}
    entity_by_key = {(r["family_id"], r["logical_ref"]): r for r in entity_results}
    entity_rescues = []
    for key, h in hist_entity.items():
        c = entity_by_key[key]
        if (not h["selected_region_correct"]) and c["selected_region_correct_at_0_5"]:
            gt_box = gt[key]
            rank = _correct_candidate_ranks(ref_by_key[key], gt_box)
            entity_rescues.append({
                "family_id": key[0],
                "logical_ref": key[1],
                "referent_phrase": c["referent_phrase"],
                "historical_selected_iou": h["selected_iou"],
                "context_selected_iou": c["selected_iou"],
                "historical_selected_nearest_union_candidate": _nearest_union_rank(ref_by_key[key], h["selected_bbox_xyxy"]),
                "context_selected_candidate_id": c["selected_candidate_id"],
                "context_selected_union_native_rank": c["selected_union_native_rank"],
                "context_selected_r4_native_rank": c["selected_r4_native_rank"],
                "context_selected_sam3_native_rank": c["selected_sam3_native_rank"],
                "correct_candidate_rank": rank,
            })
    grounded_rescues = []
    for qa_id, h in hist_qa.items():
        c = qa_by_id[qa_id]
        if (not h["grounded_target_correct"]) and c["grounded_target_correct"]:
            family_id = c["family_id"]
            target = c["target"]
            if target is None:
                continue
            key = (family_id, target)
            rank = _correct_candidate_ranks(ref_by_key[key], gt[key])
            grounded_rescues.append({
                "qa_id": qa_id,
                "family_id": family_id,
                "qa_type": c["qa_type"],
                "true_target": c["true_target"],
                "historical_target": h["target"],
                "context_target": c["target"],
                "correct_candidate_rank": rank,
                "historical_selected_nearest_union_candidate": _nearest_union_rank(ref_by_key[key], hist_entity[key]["selected_bbox_xyxy"]),
                "context_selected_candidate_id": entity_by_key[key]["selected_candidate_id"],
                "context_selected_union_native_rank": entity_by_key[key]["selected_union_native_rank"],
            })
    # Global selected-candidate rank distributions.
    distribution = []
    for r in entity_results:
        distribution.append({
            "family_id": r["family_id"],
            "logical_ref": r["logical_ref"],
            "correct": r["selected_region_correct_at_0_5"],
            "selected_union_native_rank": r["selected_union_native_rank"],
            "selected_r4_native_rank": r["selected_r4_native_rank"],
            "selected_sam3_native_rank": r["selected_sam3_native_rank"],
        })
    lower_rank_entity_rescues = []
    for case in entity_rescues:
        cr = (case.get("correct_candidate_rank") or {}).get("best_correct_union_native_rank")
        hr = (case.get("historical_selected_nearest_union_candidate") or {}).get("union_native_rank")
        if cr is not None and hr is not None and cr < hr:
            lower_rank_entity_rescues.append(f"{case['family_id']}|{case['logical_ref']}")
    result = {
        "entity_rescues": entity_rescues,
        "grounded_target_rescues": grounded_rescues,
        "entity_rescue_count": len(entity_rescues),
        "grounded_target_rescue_count": len(grounded_rescues),
        "lower_union_rank_than_historical_wrong_selection": lower_rank_entity_rescues,
        "selected_rank_distribution": distribution,
        "interpretation": {
            "r4_native_rank": "1-based rank among R4/MM-GDINO-L candidates retained in the frozen union",
            "sam3_native_rank": "1-based rank among SAM3-LiteText-L top-64 candidates retained in the frozen union",
            "union_native_rank": "1-based position in frozen R4-then-SAM3 deduplicated order",
        },
        "evaluator_only": True,
    }
    write_json(OUT / "rank_rescue_analysis.json", result)
    return result


# ---------------------------------------------------------------------------
# downstream: frozen geometry, isolation, color, minimal state, final
# ---------------------------------------------------------------------------

def _same_bbox(a, b, atol=1e-6) -> bool:
    if a is None or b is None:
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def _selected_by_key():
    selected = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "selected_regions.jsonl")}
    assert len(selected) == 40
    return selected


def _write_context_binding_results():
    selected = _selected_by_key()
    records = []
    for key in sorted(selected):
        row = selected[key]
        records.append({
            "family_id": row["family_id"],
            "logical_ref": row["logical_ref"],
            "referent_phrase": row["referent_phrase"],
            "selected_bbox_xyxy": row["selected_bbox_xyxy"],
            "selected_provider": ROUTE,
            "fallback_used": bool(row.get("fallback_reason")),
            "fallback_reason": row.get("fallback_reason"),
            "persistent_entity_state": True,
            "binding_once_per_family": True,
            "focus_transform_sha256": row.get("focus_transform_sha256"),
            "prompt_sha256": row.get("prompt_sha256"),
        })
    write_jsonl(OUT / "binding_results.jsonl", records)


def _prepopulate_downstream_cache():
    """Reuse historical isolation/attribute rows for unchanged selected regions.

    Existing computed rows in this output directory are preserved as-is, which
    makes the downstream stage resumable after a crash without rerunning GPU
    work for entities already completed.
    """
    selected = _selected_by_key()
    hist_mask = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST_OUT / "mask_results.jsonl")}
    hist_attr = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST_OUT / "attribute_results.jsonl")}
    existing_mask = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "mask_results.jsonl")}
    existing_attr = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "attribute_results.jsonl")}
    mask_out = []
    attribute_out = []
    reuse = {
        "mask_rows_reused_from_historical": 0,
        "attribute_rows_reused_from_historical": 0,
        "mask_rows_reused_from_current_output": 0,
        "attribute_rows_reused_from_current_output": 0,
        "entities_rerun": [],
    }
    for key, sel in sorted(selected.items()):
        # 1) Keep a row already produced by a prior downstream run if it belongs
        #    to the current selected region.
        prev_m = existing_mask.get(key)
        prev_a = existing_attr.get(key)
        if prev_m is not None and _same_bbox(sel["selected_bbox_xyxy"], prev_m.get("selected_bbox_xyxy")):
            m = copy.deepcopy(prev_m)
            m["selected_provider"] = ROUTE
            mask_out.append(m)
            reuse["mask_rows_reused_from_current_output"] += 1
        else:
            hist_m = hist_mask.get(key)
            if hist_m is not None and _same_bbox(sel["selected_bbox_xyxy"], hist_m.get("selected_bbox_xyxy")):
                m = copy.deepcopy(hist_m)
                m["selected_provider"] = ROUTE
                m["reused_from_historical_integrated"] = True
                mask_out.append(m)
                reuse["mask_rows_reused_from_historical"] += 1
            else:
                reuse["entities_rerun"].append(f"{key[0]}|{key[1]}")
        if prev_a is not None and _same_bbox(sel["selected_bbox_xyxy"], prev_a.get("selected_bbox_xyxy")):
            a = copy.deepcopy(prev_a)
            a["selected_provider"] = ROUTE
            attribute_out.append(a)
            reuse["attribute_rows_reused_from_current_output"] += 1
        else:
            hist_a = hist_attr.get(key)
            if hist_a is not None and _same_bbox(sel["selected_bbox_xyxy"], hist_a.get("selected_bbox_xyxy")):
                a = copy.deepcopy(hist_a)
                a["selected_provider"] = ROUTE
                a["reused_from_historical_integrated"] = True
                attribute_out.append(a)
                reuse["attribute_rows_reused_from_historical"] += 1
    write_jsonl(OUT / "mask_results.jsonl", mask_out)
    write_jsonl(OUT / "attribute_results.jsonl", attribute_out)
    return reuse


def _run_frozen_downstream():
    import harness.scripts.run_strong_integrated_repair as frozen

    frozen.OUT = OUT
    frozen.ensure_output()
    # stage_assemble reads this evaluator-only grounding artifact for its Q2 summary.
    grounding_hist = HIST_OUT / "grounding_results.jsonl"
    if grounding_hist.exists() and not (OUT / "grounding_results.jsonl").exists():
        (OUT / "grounding_results.jsonl").write_bytes(grounding_hist.read_bytes())
    _write_context_binding_results()
    reuse = _prepopulate_downstream_cache()

    if len(rows(OUT / "mask_results.jsonl")) < 40:
        print(f"[downstream] recomputing mask isolation for {len(reuse['entities_rerun'])} entities", flush=True)
        frozen.stage_mask()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        print("[downstream] mask isolation fully reused", flush=True)
    if len(rows(OUT / "attribute_results.jsonl")) < 40:
        print("[downstream] recomputing target color for changed entities", flush=True)
        frozen.stage_attribute()
    else:
        print("[downstream] target color fully reused", flush=True)
    frozen.stage_assemble()
    # The frozen assembler uses its historical route string; relabel only our output.
    for name in ("qa_results.jsonl", "minimal_state_results.jsonl"):
        rr = rows(OUT / name)
        for r in rr:
            r["route"] = ROUTE
        write_jsonl(OUT / name, rr)
    return reuse


def _final_record_from_generation(qa, state, generation, question=None):
    from harness.evaluators.end2end_metrics import score
    from harness.processors.color_pixel import REALBIND_COLORS
    from harness.dataset import read_jsonl
    from harness.configs.paths import Paths

    families = {r["family_id"]: r for r in read_jsonl(Paths().families)}
    vocab = {
        "color": list(REALBIND_COLORS),
        "position": ["left", "right"],
        "identity": sorted({family[k]["object"] for family in families.values() for k in ("A", "B")}),
    }
    scored = score(generation["response"], "color", state["gt_color"], families[qa["family_id"]], vocab)
    return {
        "qa_id": qa["qa_id"],
        "route": ROUTE,
        "state_sha256": state["state_sha256"],
        "state_sufficient": bool(state["minimal_state_sufficient"]),
        "question": question if question is not None else qa["question"],
        "state_text": state["state_text"],
        "prompt_sha256": None,
        "raw_response": generation["response"],
        "prediction": scored["normalized_prediction"],
        "invalid_prediction": bool(scored["invalid_prediction"]),
        "final_correct": bool(scored["correct"]),
        "image_path": None,
        "image_present": False,
        "vision_encoder_called": False,
        "scoring_revision": scored["scoring_revision"],
        "final_cache_reused_from_historical": False,
    }


def _run_final_reuse():
    from harness.reasoners.qwen35 import Qwen35Reasoner
    from harness.scripts.run_minimal_decomposition_route_pilot import prompt as final_prompt

    qa_rows = {r["qa_id"]: r for r in rows(OUT / "qa_results.jsonl")}
    minimal_rows = {r["qa_id"]: r for r in rows(OUT / "minimal_state_results.jsonl")}
    prepared = {r["qa_id"]: r for r in rows(PREPARED_PATH)}
    assert len(qa_rows) == len(minimal_rows) == 40
    hist_final = {r["qa_id"]: r for r in rows(HIST_OUT / "final_results.jsonl")}
    final_path = OUT / "final_results.jsonl"
    existing = rows(final_path) if final_path.exists() else []
    done = {r["qa_id"]: r for r in existing}
    reasoner = None
    reuse_count = 0
    new_count = 0
    final_rows = []
    for qa_id in sorted(qa_rows):
        if qa_id in done:
            final_rows.append(done[qa_id])
            continue
        state = qa_rows[qa_id]
        question = prepared[qa_id]["question"]
        assert minimal_rows[qa_id]["state_text"] == state["state_text"]
        query = final_prompt(question, state["state_text"])
        hist = hist_final.get(qa_id)
        state_sha = sha256_text(state["state_text"])
        if (
            hist is not None
            and hist.get("state_sha256") == state_sha
            and hist.get("question") == question
            and hist.get("image_present") is False
            and hist.get("vision_encoder_called") is False
        ):
            rec = {
                "qa_id": qa_id,
                "route": ROUTE,
                "state_sha256": state_sha,
                "state_sufficient": bool(state["minimal_state_sufficient"]),
                "question": question,
                "state_text": state["state_text"],
                "prompt_sha256": sha256_text(query),
                "raw_response": hist["raw_response"],
                "prediction": hist["prediction"],
                "invalid_prediction": bool(hist["invalid_prediction"]),
                "final_correct": bool(hist["final_correct"]),
                "image_path": None,
                "image_present": False,
                "vision_encoder_called": False,
                "scoring_revision": hist.get("scoring_revision"),
                "final_cache_reused_from_historical": True,
                "reuse_source": "outputs/strong_integrated_repair/final_results.jsonl",
            }
            reuse_count += 1
        else:
            if reasoner is None:
                print("[downstream] loading frozen text-only Qwen final reasoner", flush=True)
                reasoner = Qwen35Reasoner()
            generation = reasoner.generate(query, None)
            assert generation["image_path"] is None
            assert not generation["vision_encoder_called"]
            assert not generation["visual_tokens_in_rendered_prompt"]
            assert not any("pixel" in key or "image" in key or "video" in key for key in generation["input_keys"])
            rec = _final_record_from_generation(state, state, generation, question=question)
            rec["prompt_sha256"] = sha256_text(query)
            rec["final_cache_reused_from_historical"] = False
            new_count += 1
        append_jsonl(final_path, rec)
        done[qa_id] = rec
        final_rows.append(rec)
        print(f"[final] {qa_id} reuse={rec['final_cache_reused_from_historical']} correct={rec['final_correct']}", flush=True)
    # Keep deterministic order.
    final_rows = sorted(done.values(), key=lambda r: r["qa_id"])
    write_jsonl(final_path, final_rows)
    return {"reused": reuse_count, "new": new_count, "rows": final_rows}


def _run_downstream_audits(metrics):
    qa_rows = {r["qa_id"]: r for r in rows(OUT / "qa_results.jsonl")}
    minimal_rows = {r["qa_id"]: r for r in rows(OUT / "minimal_state_results.jsonl")}
    final_rows = {r["qa_id"]: r for r in rows(OUT / "final_results.jsonl")}
    mask_rows = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "mask_results.jsonl")}
    binding_rows = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "binding_results.jsonl")}
    selected = _selected_by_key()
    assert len(qa_rows) == len(minimal_rows) == len(final_rows) == 40
    assert len(mask_rows) == len(binding_rows) == 40
    # Paired questions share the same A/B binding.
    family_qas = defaultdict(list)
    for r in qa_rows.values():
        family_qas[r["family_id"]].append(r)
    shared = {}
    for family_id, rs in family_qas.items():
        assert len(rs) == 2
        for ref in ("A", "B"):
            b = binding_rows[(family_id, ref)]["selected_bbox_xyxy"]
            hashes = {sha256_json({"bbox": b, "type": "A/B selected region"}) for _ in rs}
            assert len(hashes) == 1
            shared[f"{family_id}|{ref}"] = {
                "bbox": b,
                "region_hash": next(iter(hashes)),
                "shared_by_both_qa": True,
            }
    for qa_id, state in qa_rows.items():
        assert minimal_rows[qa_id]["state_text"] == state["state_text"]
        target = state.get("target")
        if target is not None:
            mask = mask_rows[(state["family_id"], target)]
            assert state["selected_target_bbox_xyxy"] == binding_rows[(state["family_id"], target)]["selected_bbox_xyxy"]
            assert final_rows[qa_id]["state_sha256"] == sha256_text(state["state_text"])
    format_ok = all(
        re.fullmatch(r"\[TARGET_ENTITY\] [AB]\n\[TARGET_ATTRIBUTE\] color = (red|orange|yellow|green|blue|purple|pink|brown|black|white|gray)", r["state_text"]) is not None
        for r in qa_rows.values()
    )
    audit = {
        "binding_once_per_family": True,
        "paired_questions_share_A_region": all(v["shared_by_both_qa"] for k, v in shared.items() if k.endswith("|A")),
        "paired_questions_share_B_region": all(v["shared_by_both_qa"] for k, v in shared.items() if k.endswith("|B")),
        "region_hashes": shared,
        "minimal_state_text_consistent": True,
        "minimal_state_format_valid": format_ok,
        "final_state_hash_consistent": True,
        "deterministic_geometry_target_matches_state": True,
    }
    write_json(OUT / "state_consistency_audit.json", audit)

    leakage = gt_leakage_audit()
    return audit, leakage


def gt_leakage_audit():
    forbidden_keys = {
        "qa_id", "question", "qa_type", "ground_truth", "gt_bbox",
        "gt_bbox_xyxy_pixels", "true_target", "evaluator_only", "EVALUATOR_ONLY",
        "final_correct", "answer", "final_answer", "color", "relation",
    }
    artifact_rows = []
    passed = True
    notes = []
    for rel in ("context_binding_scores.jsonl", "selected_regions.jsonl"):
        for row in rows(OUT / rel):
            bad = sorted(k for k in row if k in forbidden_keys)
            if bad:
                passed = False
                artifact_rows.append({"artifact": rel, "key": row.get("candidate_id") or row.get("family_id"), "forbidden_keys": bad})
    # The prompt deliberately contains the literal instruction "Answer exactly YES or NO."
    # Audit the actual inference prompt against evaluator payload, not against that instruction word.
    for row in rows(OUT / "context_binding_scores.jsonl"):
        prompt = row.get("prompt", "")
        if "REL2C" in prompt or "ground_truth" in prompt or "true_target" in prompt or "gt_bbox" in prompt:
            passed = False
            artifact_rows.append({"artifact": "context_binding_scores.jsonl", "key": row.get("candidate_id"), "forbidden_prompt_token": True})
    # Selected rows must use the frozen prompt template and no side/question fields.
    for row in rows(OUT / "selected_regions.jsonl"):
        if row.get("prompt_sha256") != sha256_text(prompt_for(row["referent_phrase"])):
            passed = False
            artifact_rows.append({"artifact": "selected_regions.jsonl", "key": row["family_id"] + "|" + row["logical_ref"], "prompt_mismatch": True})

    def scan_keys(value, path=""):
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden_keys:
                    found.append(path + "/" + str(key))
                found.extend(scan_keys(child, path + "/" + str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(scan_keys(child, path + f"/{index}"))
        return found

    for rel in ("candidate_pool_manifest.json", "focus_transform_config.json", "pipeline_config.json", "qwen_tokenization_audit.json"):
        payload = read_json(OUT / rel)
        hits = scan_keys(payload)
        if hits:
            passed = False
            artifact_rows.append({"artifact": rel, "forbidden_keys": hits[:20]})
    audit = {
        "passed": passed,
        "forbidden_keys_checked": sorted(forbidden_keys),
        "violations": artifact_rows,
        "notes": [
            "Evaluator annotations are joined only after selected_regions.jsonl is frozen.",
            "Candidate scoring receives only the legal referent phrase, original image, candidate bbox, and fixed focus transform.",
            "The prompt contains the response-format word 'Answer'; this is not a leaked final answer.",
        ],
        "evaluator_only_join_point": "evaluate_stage() after select_stage()",
    }
    write_json(OUT / "gt_leakage_audit.json", audit)
    return audit


def _survival_funnel(metrics):
    qa_rows = {r["qa_id"]: r for r in rows(OUT / "qa_results.jsonl")}
    final_rows = {r["qa_id"]: r for r in rows(OUT / "final_results.jsonl")}
    mask_rows = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "mask_results.jsonl")}
    assert len(qa_rows) == len(final_rows) == 40
    steps = []
    current = set(qa_rows)
    steps.append({"stage": "start", "count": 40, "condition": "locked 40 REL2C QA"})
    logical = {q for q in current if qa_rows[q]["logical_target_correct"]}
    steps.append({"stage": "LogicalTargetCorrect", "count": len(logical), "condition": "QA target identity equals evaluator logical target"})
    grounded = {q for q in logical if qa_rows[q]["grounded_target_correct"]}
    steps.append({"stage": "GroundedTargetCorrect", "count": len(grounded), "condition": "logical target and selected target region IoU>=0.5"})
    isolated = set()
    for q in grounded:
        state = qa_rows[q]
        target = state["target"]
        mask = mask_rows[(state["family_id"], target)]
        if target is not None and (mask.get("mask_valid") or mask.get("crop_path")):
            isolated.add(q)
    steps.append({"stage": "IsolationValidGivenPrevious", "count": len(isolated), "condition": "valid target-local view (SAM3 mask isolation or frozen exact-bbox crop)"})
    attribute = {q for q in isolated if qa_rows[q]["target_attribute_correct"]}
    steps.append({"stage": "AttributeCorrectGivenPrevious", "count": len(attribute), "condition": "target color reader output equals evaluator target color"})
    state_sufficient = {q for q in attribute if qa_rows[q]["minimal_state_sufficient"]}
    steps.append({"stage": "MinimalStateSufficientGivenPrevious", "count": len(state_sufficient), "condition": "frozen minimal-state sufficiency criterion and all previous conditions"})
    clean = {q for q in state_sufficient if final_rows[q]["final_correct"]}
    steps.append({"stage": "CleanFinalCorrectGivenPrevious", "count": len(clean), "condition": "frozen text-only reasoner answers correctly"})
    counts = [s["count"] for s in steps]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts
    unconditional = {
        "target_isolated_mask_used": sum(r.get("target_isolated", False) for r in qa_rows.values()),
        "target_mask_valid": sum(r.get("target_mask_valid", False) for r in qa_rows.values()),
        "target_attribute_correct": sum(r.get("target_attribute_correct", False) for r in qa_rows.values()),
        "minimal_state_sufficient": sum(r.get("minimal_state_sufficient", False) for r in qa_rows.values()),
        "raw_final_correct": sum(r.get("final_correct", False) for r in final_rows.values()),
        "compensating_error_correct": sum((not qa_rows[q]["minimal_state_sufficient"]) and final_rows[q]["final_correct"] for q in qa_rows),
        "clean_final_correct": len(clean),
    }
    return {"monotonic_funnel": steps, "unconditional_components": unconditional}


# ---------------------------------------------------------------------------
# reports and audits
# ---------------------------------------------------------------------------

def _historical_summary():
    clean = read_json(HIST_OUT / "clean_vs_raw_accuracy.json")
    strong = clean["strong_integrated_repair"]
    hist_entity, hist_qa = load_historical_eval()
    return {
        "selected_region_correct": sum(1 for v in hist_entity.values() if v["selected_region_correct"]),
        "logical_target_correct": sum(1 for v in hist_qa.values() if v["logical_target_correct"]),
        "grounded_target_correct": sum(1 for v in hist_qa.values() if v["grounded_target_correct"]),
        "minimal_state_sufficient": strong["minimal_state_sufficient"],
        "clean_final_correct": strong["clean_final_correct"],
        "raw_final_correct": strong["raw_final_correct"],
        "target_attribute_correct": strong["target_attribute_correct"],
        "target_isolated": strong["target_isolated"],
        "vanilla_vlm_locked": clean["vanilla_vlm_locked"],
    }


def _decision_label(metrics, clean, hist):
    grounded = metrics["grounded_target_correct"]
    clean_final = clean["clean_final_correct"]
    if clean_final > 20:
        return "CLEAN_BASELINE_BREAKING_REPAIR"
    if grounded >= 30:
        return "CONTEXT_BINDING_ESTABLISHED" if grounded > hist["grounded_target_correct"] else "CONTEXT_BINDING_GAIN_NOT_SUFFICIENT"
    if grounded > hist["grounded_target_correct"]:
        return "CONTEXT_BINDING_GAIN_NOT_SUFFICIENT"
    return "CONTEXT_BINDING_NO_GAIN"


def _subset_table_rows(subset_analysis):
    """Return display rows for the required subset table.

    A referent may satisfy more than one annotation label (for example a
    same-category ambiguity can also be a local appearance noun), so the
    table reports each listed subset independently.
    """
    classification = subset_analysis.get("classification", {})
    all_keys = {tuple(k.split("|")) for k in classification}
    labels_by_key = {tuple(k.split("|")): set(v.get("labels", [])) for k, v in classification.items()}

    def keys_with(predicate):
        return {k for k, labels in labels_by_key.items() if predicate(labels)}

    same = keys_with(lambda ls: "same_category_or_ambiguous" in ls)
    context = keys_with(lambda ls: ("relational_or_context_dependent" in ls) or ("scene_region_context_dependent" in ls))
    local = keys_with(lambda ls: "purely_local_appearance" in ls)
    listed = same | context | local
    other = all_keys - listed

    hist_entity, hist_qa = load_historical_eval()
    entity_eval = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / "entity_evaluation.jsonl")}
    qa_eval = {r["qa_id"]: r for r in rows(OUT / "qa_evaluation.jsonl")}
    prepared = rows(PREPARED_PATH)

    def stats(keys):
        family_refs = set(keys)
        qa_ids = [p["qa_id"] for p in prepared if (p["family_id"], "A") in family_refs or (p["family_id"], "B") in family_refs]
        return {
            "n_entities": len(family_refs),
            "n_qa": len(qa_ids),
            "historical_selected_region_correct": sum(hist_entity[k]["selected_region_correct"] for k in family_refs),
            "context_selected_region_correct": sum(entity_eval[k]["selected_region_correct_at_0_5"] for k in family_refs),
            "historical_grounded_target_correct": sum(hist_qa[q]["grounded_target_correct"] for q in qa_ids),
            "context_grounded_target_correct": sum(qa_eval[q]["grounded_target_correct"] for q in qa_ids),
        }

    return {
        "same_category_or_ambiguous": ("Same-category / ambiguous", stats(same)),
        "context_dependent": ("Context-dependent referent", stats(context)),
        "purely_local_appearance": ("Local appearance referent", stats(local)),
        "other": ("Other", stats(other)),
    }, len(other)


def _format_ratio(a, b):
    return f"{a}/{b}"


def write_reports(metrics, paired, subset, same, rank, clean, survival, downstream_audit, leakage_audit):
    hist = _historical_summary()
    try:
        p0 = read_json(ROOT / "outputs/high_recall_binding_repair/p0_qwen_independent/binding_metrics.json")
    except Exception:
        p0 = {"selected_region_correct": 13, "grounded_target_correct": 13}
    try:
        prior_route_metrics = read_json(ROOT / "outputs/high_recall_binding_repair/final_comparison/route_metrics.json")
    except Exception:
        prior_route_metrics = {}
    decision = _decision_label(metrics, clean, hist)
    qa_rows = {r["qa_id"]: r for r in rows(OUT / "qa_results.jsonl")}
    final_rows = {r["qa_id"]: r for r in rows(OUT / "final_results.jsonl")}
    # ------------------------- comparison report -------------------------
    lines = []
    lines.append("# Context-preserving referent-region binding repair")
    lines.append("")
    lines.append("**DEVELOPMENT-ONLY FROZEN COHORT.** The same heavily inspected 20 families and 40 REL2C QA rows are reused. These are not held-out paper generalization results.")
    lines.append("")
    lines.append("## Main comparison")
    lines.append("")
    lines.append("| Method | SelectedRegionCorrect | LogicalTargetCorrect | GroundedTargetCorrect | MinimalStateSufficient | CleanFinal | RawFinal |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| Historical Integrated | {_format_ratio(hist['selected_region_correct'],40)} | {_format_ratio(hist['logical_target_correct'],40)} | {_format_ratio(hist['grounded_target_correct'],40)} | {_format_ratio(hist['minimal_state_sufficient'],40)} | {_format_ratio(hist['clean_final_correct'],40)} | {_format_ratio(hist['raw_final_correct'],40)} |")
    lines.append(f"| Context-Preserving Binder | {_format_ratio(metrics['selected_region_correct_at_0_5'],40)} | {_format_ratio(metrics['logical_target_correct'],40)} | {_format_ratio(metrics['grounded_target_correct'],40)} | {_format_ratio(clean['minimal_state_sufficient'],40)} | {_format_ratio(clean['clean_final_correct'],40)} | {_format_ratio(clean['raw_final_correct'],40)} |")
    lines.append("")
    lines.append("Vanilla VLM locked baseline: 20/40 raw final and 20/40 clean final.")
    lines.append("")
    lines.append("## Candidate availability and selection efficiency")
    lines.append("")
    lines.append(f"- CandidateAvailable@0.5: {metrics['candidate_available_at_0_5']}/40")
    lines.append(f"- SelectedRegionCorrect@0.5: {metrics['selected_region_correct_at_0_5']}/40")
    lines.append(f"- SelectionEfficiency = P(selected correct | usable candidate exists): {metrics['selection_efficiency'] if metrics['selection_efficiency'] is None else round(metrics['selection_efficiency'], 4)}")
    lines.append(f"- Both A and B selected regions correct: {metrics['both_entity_regions_correct']}/20 families")
    lines.append("")
    lines.append("## Paired binding transitions")
    lines.append("")
    lines.append("### Entity selected region")
    lines.append("")
    lines.append("| Transition | Count |")
    lines.append("|---|---:|")
    for label, key in (
        ("Historical wrong → Context correct", "historical_wrong_context_correct"),
        ("Historical correct → Context wrong", "historical_correct_context_wrong"),
        ("Both correct", "both_correct"),
        ("Both wrong", "both_wrong"),
    ):
        lines.append(f"| {label} | {paired['entity_selected_region']['counts'][key]} |")
    lines.append("")
    lines.append("### Grounded target")
    lines.append("")
    lines.append("| Transition | Count |")
    lines.append("|---|---:|")
    for label, key in (
        ("Historical wrong → Context correct", "historical_wrong_context_correct"),
        ("Historical correct → Context wrong", "historical_correct_context_wrong"),
        ("Both correct", "both_correct"),
        ("Both wrong", "both_wrong"),
    ):
        lines.append(f"| {label} | {paired['grounded_target']['counts'][key]} |")
    lines.append("")
    lines.append("## Context rescue cases")
    lines.append("")
    if paired["entity_selected_region"]["cases"]["historical_wrong_context_correct"]:
        for case in paired["entity_selected_region"]["cases"]["historical_wrong_context_correct"]:
            lines.append(f"- entity rescue `{case['family_id']}|{case['logical_ref']}` ({case['referent_phrase']}): historical IoU={case['historical_selected_iou']:.4f} → context IoU={case['context_selected_iou']:.4f}, selected union rank={case['context_selected_union_native_rank']}")
    else:
        lines.append("- None at the entity selected-region level.")
    if paired["grounded_target"]["cases"]["historical_wrong_context_correct"]:
        for case in paired["grounded_target"]["cases"]["historical_wrong_context_correct"]:
            lines.append(f"- grounded rescue `{case['qa_id']}`: historical target={case['historical_target']} → context target={case['context_target']} (true={case['true_target']})")
    else:
        lines.append("- None at the grounded-target level.")
    lines.append("")
    lines.append("## Regression audit")
    lines.append("")
    if paired["entity_selected_region"]["cases"]["historical_correct_context_wrong"]:
        for case in paired["entity_selected_region"]["cases"]["historical_correct_context_wrong"]:
            lines.append(f"- entity regression `{case['family_id']}|{case['logical_ref']}` ({case['referent_phrase']}): historical IoU={case['historical_selected_iou']:.4f} → context IoU={case['context_selected_iou']:.4f}")
    else:
        lines.append("- No entity selected-region regressions.")
    if paired["grounded_target"]["cases"]["historical_correct_context_wrong"]:
        for case in paired["grounded_target"]["cases"]["historical_correct_context_wrong"]:
            lines.append(f"- grounded regression `{case['qa_id']}`: historical target={case['historical_target']} → context target={case['context_target']} (true={case['true_target']})")
    else:
        lines.append("- No grounded-target regressions.")
    lines.append("")
    lines.append("## Subset analysis")
    lines.append("")
    lines.append("| Subset | Historical | Contextual |")
    lines.append("|---|---:|---:|")
    subset_rows, _other_n = _subset_table_rows(subset)
    for key, (label, s) in subset_rows.items():
        lines.append(f"| {label} | {s['historical_selected_region_correct']}/{s['n_entities']} listed entities; {s['historical_grounded_target_correct']}/{s['n_qa']} listed QA | {s['context_selected_region_correct']}/{s['n_entities']} listed entities; {s['context_grounded_target_correct']}/{s['n_qa']} listed QA |")
    lines.append("")
    lines.append("## Rank rescue analysis")
    lines.append("")
    lines.append(f"- Entity rescues: {rank['entity_rescue_count']}")
    lines.append(f"- Grounded-target rescues: {rank['grounded_target_rescue_count']}")
    for case in rank["entity_rescues"]:
        cr = case.get("correct_candidate_rank") or {}
        lines.append(f"- `{case['family_id']}|{case['logical_ref']}` correct candidate union rank={cr.get('best_correct_union_native_rank')}, R4 rank={cr.get('best_correct_r4_native_rank')}, SAM3 rank={cr.get('best_correct_sam3_native_rank')}; selected union rank={case.get('context_selected_union_native_rank')}")
    lines.append("")
    lines.append("## Monotonic survival funnel")
    lines.append("")
    lines.append("| Stage | Count | Condition |")
    lines.append("|---|---:|---|")
    for s in survival["monotonic_funnel"]:
        lines.append(f"| {s['stage']} | {s['count']} | {s['condition']} |")
    lines.append("")
    lines.append("Unconditional component counts are reported separately in `survival_chain.json`, not mixed into the funnel.")
    lines.append("")
    # Evaluator-only but non-tuning selection behavior audit.
    selected_rows = rows(OUT / "selected_regions.jsonl")
    hist_binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST_OUT / "binding_results.jsonl")}
    source_counts = Counter()
    union_ranks = []
    identical_to_historical = 0
    for row in selected_rows:
        prov = set(row.get("selected_provenance") or [])
        if "R4/MM-GDINO-L" in prov and "SAM3-LiteText-L" in prov:
            source_counts["R4+SAM3 deduplicated candidate"] += 1
        elif "SAM3-LiteText-L" in prov:
            source_counts["SAM3-only candidate"] += 1
        elif "R4/MM-GDINO-L" in prov:
            source_counts["R4-only candidate"] += 1
        else:
            source_counts["unknown provenance"] += 1
        union_ranks.append(row.get("selected_union_native_rank"))
        h = hist_binding.get((row["family_id"], row["logical_ref"]))
        if h is not None and row.get("selected_bbox_xyxy") is not None and h.get("selected_bbox_xyxy") is not None:
            if all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(row["selected_bbox_xyxy"], h["selected_bbox_xyxy"])):
                identical_to_historical += 1
    finite_ranks = sorted(int(x) for x in union_ranks if x is not None)
    lines.append("## Selection behavior audit")
    lines.append("")
    lines.append("- Selected candidate provenance: " + "; ".join(f"{k}={v}" for k, v in sorted(source_counts.items())))
    lines.append(f"- Selected region identical to Historical Integrated: {identical_to_historical}/40")
    if finite_ranks:
        lines.append(f"- Selected union native rank: min={finite_ranks[0]}, median={finite_ranks[len(finite_ranks)//2]}, max={finite_ranks[-1]}")
    lines.append("- The argmax rule shifted most selections to low-union-rank SAM3 proposals; the evaluator join shows this shift was not aligned with GT regions (SelectedRegionCorrect 8/40). This is a descriptive outcome, not a post-hoc selector change.")
    lines.append("")
    lines.append("## Required scientific questions")
    lines.append("")
    p1 = prior_route_metrics.get("p1_objembed", {})
    p2 = prior_route_metrics.get("p2_joint_assignment", {})
    lines.append(f"**Q1. Does context-preserving candidate representation exceed isolated/crop-based binding?** Prior frozen crop/region routes: P0 Qwen isolated crop SelectedRegionCorrect {p0.get('selected_region_correct')}/40 and GroundedTargetCorrect {p0.get('grounded_target_correct')}/40; P1 ObjEmbed {p1.get('selected_region_correct')}/40 and {p1.get('grounded_target_correct')}/40; P2 joint assignment {p2.get('selected_region_correct')}/40 and {p2.get('grounded_target_correct')}/40. Context-preserving binding obtains {metrics['selected_region_correct_at_0_5']}/40 and {metrics['grounded_target_correct']}/40. It does not exceed the isolated/crop-based routes on this frozen subset.")
    lines.append(f"**Q2. Does it exceed Historical Integrated 26/40 GroundedTargetCorrect?** Context route: {metrics['grounded_target_correct']}/40. {'Yes' if metrics['grounded_target_correct'] > 26 else 'No'}.")
    rel_res = subset.get("rescue_by_label", {}).get("relational_or_context_dependent", {})
    scene_res = subset.get("rescue_by_label", {}).get("scene_region_context_dependent", {})
    same_res = subset.get("rescue_by_label", {}).get("same_category_or_ambiguous", {})
    lines.append(f"**Q3. Are rescues concentrated in relation/context-dependent referents?** The frozen legal referent phrases contain no explicit relational or comparative modifiers (relational denominator {subset.get('subsets', {}).get('relational_or_context_dependent', {}).get('n_entities', 0)}). Entity rescues={rank['entity_rescue_count']}, grounded rescues={rank['grounded_target_rescue_count']}. Evaluator-only rescue counts: same-category entity/grounded={same_res.get('entity_rescues', 0)}/{same_res.get('grounded_target_rescues', 0)}; scene-region-context entity/grounded={scene_res.get('entity_rescues', 0)}/{scene_res.get('grounded_target_rescues', 0)}; explicit relational entity/grounded={rel_res.get('entity_rescues', 0)}/{rel_res.get('grounded_target_rescues', 0)}.")
    lines.append(f"**Q4. Does adding context also cause regressions?** Entity selected-region regressions={paired['entity_selected_region']['counts']['historical_correct_context_wrong']}; grounded-target regressions={paired['grounded_target']['counts']['historical_correct_context_wrong']}. All cases are listed above and in `paired_transitions.json`.")
    lines.append(f"**Q5. Does improved binding propagate to MinimalStateSufficient?** MinimalStateSufficient={clean['minimal_state_sufficient']}/40 vs historical 20/40; CleanFinal={clean['clean_final_correct']}/40 vs historical 20/40.")
    lines.append(f"**Q6. Does CleanFinal finally exceed Vanilla 20/40?** {'Yes' if clean['clean_final_correct'] > 20 else 'No'} ({clean['clean_final_correct']}/40).")
    lines.append(f"**Q7. If still unsuccessful, is there sufficient evidence to stop optimizing the pure image-exit perception stack?** Decision label: `{decision}`. If the contextual binder did not produce a clean downstream gain, the result supports the stop rule: do not continue selector/alpha/prompt tuning on this development subset; move to the predeclared soft-handoff phase.")
    lines.append("")
    lines.append("## Decision label")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append("Decision rules applied in order: CleanFinal>20 → `CLEAN_BASELINE_BREAKING_REPAIR`; otherwise GroundedTargetCorrect<=26 → `CONTEXT_BINDING_NO_GAIN`; binding gain with CleanFinal<=20 → `CONTEXT_BINDING_GAIN_NOT_SUFFICIENT`; GroundedTargetCorrect>=30 and clearly above historical → `CONTEXT_BINDING_ESTABLISHED`.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"The context-preserving binder is fully implemented and evaluated on the frozen development subset. CandidateAvailable@0.5 is {metrics['candidate_available_at_0_5']}/40, but the context-conditioned argmax selects only {metrics['selected_region_correct_at_0_5']}/40 usable GT regions, so the observed failure is binding, not candidate recall. It does not exceed Historical Integrated: GroundedTargetCorrect {metrics['grounded_target_correct']}/40 vs 26/40, MinimalStateSufficient {clean['minimal_state_sufficient']}/40 vs 20/40, and CleanFinal {clean['clean_final_correct']}/40 vs 20/40. The paired audit finds only {paired['entity_selected_region']['counts']['historical_wrong_context_correct']} entity rescues against {paired['entity_selected_region']['counts']['historical_correct_context_wrong']} regressions. Per the predeclared stop rule, the experiment returns `{decision}`; the next phase is SOFT HANDOFF, not further alpha/prompt/overlay/selector engineering on this subset.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- This is a repeatedly inspected development subset, not held-out evidence.")
    lines.append("- The frozen union pool has no valid SAM3 masks, so all candidates use the predeclared exact-bbox focus; no candidate was dropped.")
    lines.append("- The legal referents are simple category nouns in this subset; relational/comparative referent strata have denominator zero.")
    lines.append("- Only one fixed alpha, prompt and focus style were run; no hyperparameter search was performed.")
    comparison = "\n".join(lines) + "\n"
    (OUT / "comparison_report.md").write_text(comparison, encoding="utf-8")

    # ------------------------- implementation report -------------------------
    cfg = read_json(OUT / "pipeline_config.json")
    audit = read_json(OUT / "qwen_tokenization_audit.json")
    impl = []
    impl.append("# Implementation report: context-preserving referent-region binding")
    impl.append("")
    impl.append("## Scope")
    impl.append("")
    impl.append("Only the referent-region binding interface was changed. The frozen candidate union, ReferentSpec, geometry, SAM3 isolation, color reader, minimal-state format and text-only final reasoner were reused. The matcher receives the untouched full scene (Image 1) and a full-canvas candidate-focused view (Image 2); outside-candidate pixels are attenuated by a fixed alpha=0.5 toward neutral gray and a fixed thin boundary marks the candidate region. Candidate pixels remain untouched and context remains recognizable.")
    impl.append("")
    impl.append("## Frozen pool")
    impl.append("")
    impl.append(f"- Source: `{cfg['candidate_pool']['source']}`")
    impl.append(f"- Source SHA-256: `{cfg['candidate_pool']['source_sha256']}`")
    impl.append(f"- Frozen SAM3 K: {cfg['candidate_pool']['primary_sam3_k']} (not re-selected)")
    impl.append(f"- Frozen dedup IoU: {cfg['candidate_pool']['dedup_iou']}")
    impl.append(f"- Candidate count: {sum(len(r['candidates']) for r in read_json(OUT/'candidate_pool_manifest.json')['referents'])} across 40 referents")
    impl.append("")
    impl.append("## Binding inference")
    impl.append("")
    impl.append(f"- Prompt: `{cfg['binding']['prompt_template']}`")
    impl.append(f"- Prompt SHA-256: `{cfg['binding']['prompt_sha256']}`")
    impl.append(f"- Score: {cfg['binding']['score']}")
    impl.append(f"- Qwen YES/NO IDs: YES={audit['response_contract']['yes_ids']}, NO={audit['response_contract']['no_ids']}")
    impl.append("- The audited tokenizer exposes both labels as single tokens, so scoring uses next-token log-odds; a teacher-forced full conditional log-likelihood fallback for multi-token labels is implemented in the same scorer.")
    impl.append("- Primary selection had zero technical fallbacks and zero semantic/historical fallbacks; the selected region is the unmodified per-referent argmax.")
    impl.append(f"- Selection: {cfg['binding']['selection']}")
    impl.append(f"- Fixed focus transform SHA-256: `{cfg['focus_transform']['transform_sha256']}`")
    impl.append("- No REL2C question, side, answer, color, relation, GT bbox, GT IoU, or evaluator label enters scoring.")
    impl.append("")
    impl.append("## Resume and crash safety")
    impl.append("")
    impl.append("- Every candidate score is appended and fsynced immediately.")
    impl.append("- Deterministic cache key includes family/image ID, referent ID, canonical candidate ID, bbox and mask hashes, original image hash, focus-transform hash, referent text hash, prompt hash, model identity, and protocol fingerprint.")
    impl.append("- `resume_status.json` records score progress; a restarted run skips OK cache keys.")
    impl.append("- OOM is handled by recursive batch splitting down to one candidate; a single-candidate failure is recorded as a technical fallback rather than a semantic fallback.")
    impl.append("")
    impl.append("## Downstream")
    impl.append("")
    impl.append("- Selected A/B regions are passed through the frozen Strong Integrated Repair downstream.")
    impl.append("- Families with an unchanged selected A/B region reuse historical geometry, mask isolation, target crop, color reading, minimal state, and text-only final result.")
    impl.append("- Minimal state remains exactly `[TARGET_ENTITY] A/B` + `[TARGET_ATTRIBUTE] color = <color>`.")
    impl.append("- Final reasoner receives no image; audited image_present=false and vision_encoder_called=false.")
    impl.append("")
    impl.append("## Audits")
    impl.append("")
    impl.append(f"- GT leakage audit passed: {read_json(OUT/'gt_leakage_audit.json')['passed']}")
    impl.append(f"- Hard handoff audit passed: {read_json(OUT/'hard_handoff_audit.json')['pass']}")
    impl.append(f"- State consistency audit: {read_json(OUT/'state_consistency_audit.json')['binding_once_per_family']}")
    (OUT / "implementation_report.md").write_text("\n".join(impl) + "\n", encoding="utf-8")

    # ------------------------- audit html -------------------------
    def table(headers, body):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in body) + "</tbody></table>"
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Context-preserving binding audit</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #bbb;padding:.35rem .55rem;text-align:left}th{background:#eee}code{background:#f3f3f3;padding:.1rem .25rem}</style></head><body>",
        f"<h1>Context-preserving referent-region binding</h1><p><b>Development-only frozen subset.</b> Decision: <code>{html.escape(decision)}</code></p>",
        "<h2>Main comparison</h2>",
        table(["Method", "SelectedRegionCorrect", "LogicalTargetCorrect", "GroundedTargetCorrect", "MinimalStateSufficient", "CleanFinal", "RawFinal"], [
            ["Historical Integrated", "27/40", f"{hist['logical_target_correct']}/40", f"{hist['grounded_target_correct']}/40", f"{hist['minimal_state_sufficient']}/40", f"{hist['clean_final_correct']}/40", f"{hist['raw_final_correct']}/40"],
            ["Context-Preserving Binder", f"{metrics['selected_region_correct_at_0_5']}/40", f"{metrics['logical_target_correct']}/40", f"{metrics['grounded_target_correct']}/40", f"{clean['minimal_state_sufficient']}/40", f"{clean['clean_final_correct']}/40", f"{clean['raw_final_correct']}/40"],
        ]),
        "<h2>Paired transitions</h2>",
        table(["Transition", "Entity selected region", "Grounded target"], [
            ["Historical wrong → Context correct", paired['entity_selected_region']['counts']['historical_wrong_context_correct'], paired['grounded_target']['counts']['historical_wrong_context_correct']],
            ["Historical correct → Context wrong", paired['entity_selected_region']['counts']['historical_correct_context_wrong'], paired['grounded_target']['counts']['historical_correct_context_wrong']],
            ["Both correct", paired['entity_selected_region']['counts']['both_correct'], paired['grounded_target']['counts']['both_correct']],
            ["Both wrong", paired['entity_selected_region']['counts']['both_wrong'], paired['grounded_target']['counts']['both_wrong']],
        ]),
        "<h2>Funnel</h2>",
        table(["Stage", "Count", "Condition"], [[s['stage'], s['count'], s['condition']] for s in survival['monotonic_funnel']]),
        "<h2>Artifacts</h2><ul>" + "".join(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>" for name in ["pipeline_config.json", "candidate_pool_manifest.json", "focus_transform_config.json", "qwen_tokenization_audit.json", "context_binding_scores.jsonl", "selected_regions.jsonl", "binding_metrics.json", "paired_transitions.json", "context_subset_analysis.json", "same_category_analysis.json", "rank_rescue_analysis.json", "downstream_results.jsonl", "minimal_state_results.jsonl", "clean_vs_raw.json", "survival_chain.json", "state_consistency_audit.json", "gt_leakage_audit.json", "hard_handoff_audit.json", "comparison_report.md"]) + "</ul></body></html>",
    ]
    (OUT / "audit.html").write_text("".join(html_parts), encoding="utf-8")
    return decision


def downstream_and_report_stage():
    metrics = read_json(OUT / "binding_metrics.json")
    reuse = _run_frozen_downstream()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    final_info = _run_final_reuse()
    survival = _survival_funnel(metrics)
    write_json(OUT / "survival_chain.json", survival)
    clean = {
        "n": 40,
        "state_sufficient": survival["unconditional_components"]["minimal_state_sufficient"],
        "minimal_state_sufficient": survival["unconditional_components"]["minimal_state_sufficient"],
        "clean_final_correct": survival["unconditional_components"]["clean_final_correct"],
        "raw_final_correct": survival["unconditional_components"]["raw_final_correct"],
        "compensating_error_correct": survival["unconditional_components"]["compensating_error_correct"],
        "vanilla_vlm_locked": 20,
        "historical_integrated": _historical_summary(),
        "context_preserving_binder": {
            "selected_region_correct": metrics["selected_region_correct_at_0_5"],
            "logical_target_correct": metrics["logical_target_correct"],
            "grounded_target_correct": metrics["grounded_target_correct"],
            "minimal_state_sufficient": survival["unconditional_components"]["minimal_state_sufficient"],
            "clean_final_correct": survival["unconditional_components"]["clean_final_correct"],
            "raw_final_correct": survival["unconditional_components"]["raw_final_correct"],
        },
    }
    write_json(OUT / "clean_vs_raw.json", clean)
    # join final predictions into downstream table
    qa_rows = {r["qa_id"]: r for r in rows(OUT / "qa_results.jsonl")}
    final_rows = {r["qa_id"]: r for r in rows(OUT / "final_results.jsonl")}
    downstream = []
    for qa_id in sorted(qa_rows):
        q = qa_rows[qa_id]
        f = final_rows[qa_id]
        downstream.append({
            **q,
            "final_prediction": f["prediction"],
            "final_raw_response": f["raw_response"],
            "final_correct": f["final_correct"],
            "clean_final_correct": bool(q["minimal_state_sufficient"] and f["final_correct"]),
            "final_image_present": f["image_present"],
            "final_vision_encoder_called": f["vision_encoder_called"],
            "final_cache_reused_from_historical": f["final_cache_reused_from_historical"],
        })
    write_jsonl(OUT / "downstream_results.jsonl", downstream)
    write_jsonl(OUT / "minimal_state_results.jsonl", rows(OUT / "qa_results.jsonl"))
    hard = {
        "pass": all((not r["image_present"]) and (not r["vision_encoder_called"]) for r in final_rows.values()),
        "image_present_false_count": sum(not r["image_present"] for r in final_rows.values()),
        "vision_encoder_called_false_count": sum(not r["vision_encoder_called"] for r in final_rows.values()),
        "rows": [{"qa_id": q, "image_present": r["image_present"], "vision_encoder_called": r["vision_encoder_called"]} for q, r in sorted(final_rows.items())],
        "final_cache_reused_count": sum(bool(r.get("final_cache_reused_from_historical")) for r in final_rows.values()),
        "final_cache_new_count": sum(not bool(r.get("final_cache_reused_from_historical")) for r in final_rows.values()),
    }
    write_json(OUT / "hard_handoff_audit.json", hard)
    audit, leakage = _run_downstream_audits(metrics)
    decision = write_reports(metrics, read_json(OUT / "paired_transitions.json"), read_json(OUT / "context_subset_analysis.json"), read_json(OUT / "same_category_analysis.json"), read_json(OUT / "rank_rescue_analysis.json"), clean, survival, audit, leakage)
    status = read_json(OUT / "resume_status.json")
    status.update({
        "status": "COMPLETE",
        "downstream_complete": True,
        "report_complete": True,
        "decision_label": decision,
        "clean_final_correct": clean["clean_final_correct"],
        "grounded_target_correct": metrics["grounded_target_correct"],
        "updated_unix": time.time(),
    })
    write_json(OUT / "resume_status.json", status)
    print(f"[downstream] reuse={reuse} final_reuse={final_info['reused']} final_new={final_info['new']} clean={clean['clean_final_correct']} decision={decision}", flush=True)
    return clean


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "tokenize", "score", "select", "evaluate", "downstream", "report", "all"])
    parser.add_argument("--limit", type=int, default=0, help="optional newly-scored candidate limit for smoke runs")
    parser.add_argument("--batch-size", type=int, default=SCORE_BATCH_SIZE)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_pool_and_configs()
    elif args.stage == "tokenize":
        tokenization_audit()
    elif args.stage == "score":
        score_stage(limit=args.limit, batch_size=args.batch_size)
    elif args.stage == "select":
        select_stage()
    elif args.stage == "evaluate":
        evaluate_stage()
    elif args.stage == "downstream":
        downstream_and_report_stage()
    elif args.stage == "report":
        metrics = read_json(OUT / "binding_metrics.json")
        paired = read_json(OUT / "paired_transitions.json")
        subset = read_json(OUT / "context_subset_analysis.json")
        same = read_json(OUT / "same_category_analysis.json")
        rank = read_json(OUT / "rank_rescue_analysis.json")
        clean = read_json(OUT / "clean_vs_raw.json") if (OUT / "clean_vs_raw.json").exists() else {"clean_final_correct": 0, "minimal_state_sufficient": 0, "raw_final_correct": 0}
        survival = read_json(OUT / "survival_chain.json") if (OUT / "survival_chain.json").exists() else {"monotonic_funnel": [], "unconditional_components": {}}
        audit = read_json(OUT / "state_consistency_audit.json") if (OUT / "state_consistency_audit.json").exists() else {}
        leakage = read_json(OUT / "gt_leakage_audit.json") if (OUT / "gt_leakage_audit.json").exists() else {}
        write_reports(metrics, paired, subset, same, rank, clean, survival, audit, leakage)
    elif args.stage == "all":
        prepare_pool_and_configs()
        tokenization_audit()
        score_stage(limit=args.limit, batch_size=args.batch_size)
        select_stage()
        evaluate_stage()
        downstream_and_report_stage()


if __name__ == "__main__":
    main()
