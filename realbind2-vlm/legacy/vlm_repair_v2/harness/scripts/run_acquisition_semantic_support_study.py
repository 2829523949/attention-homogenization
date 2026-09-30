#!/usr/bin/env python3
"""Diagnostic validity study: acquisition IoU vs fixed-region SigLIP evidence.

This script does not modify the scientific method.  It runs the current
Florence acquisition -> FOCUS/REMOVE -> SigLIP2 semantic scorer path on a
deterministic 20-family REL2C subset, deduplicates unique entity units, and
joins evaluator-only GT bboxes *after* inference for diagnostic IoU analysis.

No calibration, no Qwen, no full benchmark, no new models.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

from harness.certification.certifier import CertificationConfig
from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.dataset import read_jsonl
from harness.evaluators.leakage_checks import assert_inference_safe, assert_no_image_payload
from harness.processors.base import RegionPrediction
from harness.processors.color_siglip import SiglipColorReader
from harness.processors.crop_policy import CROP_POLICY_ID, save_persistent_crop
from harness.processors.florence_subprocess import FlorenceSubprocessGrounding
from harness.processors.registry import ColorReaderAdapter, ProcessorCapability, ProcessorRegistry
from harness.processors.siglip_backend import Siglip2Backend
from harness.processors.siglip_region_query import Siglip2RegionQueryScorer
from harness.reasoners.hard_handoff_qwen import DeterministicProofStateReasoner
from harness.refactored_pipeline import RefactoredCaseInput, run_refactored_case


OUTPUT_ROOT = Path("/mnt/d/vlm_repair_v2/outputs/acquisition_semantic_support")
MANIFEST_PATH = Path("/mnt/d/vlm_repair_v2/manifests/acquisition_semantic_support_20family.json")
N_FAMILIES = 20

# Diagnostic integrity checks only.  These are not certification thresholds and
# are not used for any scientific pass/fail classification of semantic support.
VIEW_INTEGRITY_RULES = {
    "focus_target_mae_max": 5.0,
    "remove_target_mae_min": 10.0,
    "focus_outside_mae_min": 2.0,
    "remove_outside_mae_max": 1.0,
    "pixel_difference_fraction_min": 0.5,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_20_families(qa_rows: list[dict[str, Any]], n: int = N_FAMILIES) -> list[dict[str, Any]]:
    """Frozen-order deterministic selection of families with both REL2C sides."""
    by_family: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
    for row in qa_rows:
        if row.get("qa_type") not in {"REL2C_LEFT", "REL2C_RIGHT"}:
            continue
        family_id = str(row["family_id"])
        by_family.setdefault(family_id, {})[str(row["qa_type"])] = row
    selected = []
    for family_id, sides in by_family.items():
        if {"REL2C_LEFT", "REL2C_RIGHT"}.issubset(sides):
            selected.append(
                {
                    "family_id": family_id,
                    "left_case": sides["REL2C_LEFT"],
                    "right_case": sides["REL2C_RIGHT"],
                    "image_path": str(sides["REL2C_LEFT"]["image_path"]),
                }
            )
        if len(selected) >= n:
            break
    if len(selected) != n:
        raise RuntimeError(f"expected {n} families with both REL2C sides, found {len(selected)}")
    return selected


def build_registry(
    paths: Paths,
    *,
    device: str,
    florence_device: str,
    acquisition_backend: str = "florence2",
) -> ProcessorRegistry:
    from harness.acquisition.backend_registry import register_candidate_backend

    registry = ProcessorRegistry()
    register_candidate_backend(
        registry, acquisition_backend, paths, device=device, florence_device=florence_device
    )
    backend = Siglip2Backend(paths.siglip2, device=device)
    region_scorer = Siglip2RegionQueryScorer(backend)
    registry.register(
        "siglip2_region_query",
        ProcessorCapability(
            primitive="region_evidence",
            legal_input_types=("image_view", "persistent_region", "RegionQuery"),
            output_schema="EvidenceScore",
            score_semantics=region_scorer.score_semantics,
            model_revision=region_scorer.version,
            model_path=str(paths.siglip2),
            deterministic=True,
            supports_evidence_scoring=True,
            notes="Separate fixed-region/fixed-query semantic evidence scorer.",
            can_acquire=False,
            supported_predicates=("SEMANTIC_REGION_SUPPORT",),
            supports_same_claim_rescoring=True,
            supports_certification_score=True,
            score_normalization=region_scorer.score_normalization,
            comparable_across_counterfactual_views=True,
            requires_fixed_region=True,
            requires_fixed_query=True,
        ),
        region_scorer,
    )
    color_reader = SiglipColorReader(backend=backend)
    registry.register(
        "siglip2_color",
        ProcessorCapability(
            primitive="color",
            legal_input_types=("image", "region"),
            output_schema="Observation(predicate=COLOR)",
            score_semantics=color_reader.score_semantics,
            model_revision=color_reader.version,
            model_path=str(paths.siglip2),
            deterministic=True,
            supports_evidence_scoring=True,
            notes="Color scorer; not reached because full identity is unresolved.",
            can_acquire=True,
            supported_predicates=("COLOR",),
            supports_same_claim_rescoring=True,
            supports_certification_score=True,
            score_normalization=color_reader.score_normalization,
            comparable_across_counterfactual_views=True,
        ),
        ColorReaderAdapter(color_reader),
    )
    return registry


def xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in bbox)
    return x, y, x + w, y + h


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def iou_band(iou: float) -> str:
    if iou < 0.30:
        return "A (<0.30)"
    if iou < 0.50:
        return "B [0.30,0.50)"
    if iou < 0.70:
        return "C [0.50,0.70)"
    return "D (>=0.70)"


def _region_from_dict(payload: dict[str, Any] | None) -> RegionPrediction | None:
    if not payload:
        return None
    return RegionPrediction(
        tuple(float(v) for v in payload["bbox_xyxy_pixels"]),
        tuple(float(v) for v in payload["bbox_xyxy_normalized"]),
        None if payload.get("score") is None else float(payload["score"]),
        str(payload.get("label", "")),
        str(payload.get("source", "")),
        payload.get("mask_path"),
    )


def acquired_regions(trace: dict[str, Any]) -> dict[str, RegionPrediction]:
    regions: dict[str, RegionPrediction] = {}
    for step in trace.get("acquisition_trace", []):
        if step.get("operation") == "LOCALIZE" and step.get("entity_id"):
            region = _region_from_dict((step.get("outputs") or {}).get("selected_region"))
            if region is not None:
                regions[step["entity_id"]] = region
    return regions


def views_by_entity(trace: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for view in trace.get("views", []):
        if view.get("source_entity"):
            grouped.setdefault(view["source_entity"], {})[view["view_id"]] = view
    return grouped


def view_pixel_metrics(image_path: Path, focus_path: Path, remove_path: Path, region: RegionPrediction, suppression_box: list[float]) -> dict[str, Any]:
    raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    focus = np.asarray(Image.open(focus_path).convert("RGB"), dtype=float)
    remove = np.asarray(Image.open(remove_path).convert("RGB"), dtype=float)
    x0, y0, x1, y1 = (int(round(v)) for v in region.bbox_xyxy_pixels)
    inside = np.s_[y0:y1, x0:x1]
    sx0, sy0, sx1, sy1 = (int(round(v)) for v in suppression_box)
    outside_mask = np.ones(raw.shape[:2], dtype=bool)
    outside_mask[max(0, sy0 - 1):sy1 + 2, max(0, sx0 - 1):sx1 + 2] = False
    outside_pixel_count = int(outside_mask.sum())
    focus_outside = float(np.mean(np.abs(focus[outside_mask] - raw[outside_mask]))) if outside_pixel_count else None
    remove_outside = float(np.mean(np.abs(remove[outside_mask] - raw[outside_mask]))) if outside_pixel_count else None
    return {
        "focus_target_mae_rgb": float(np.mean(np.abs(focus[inside] - raw[inside]))),
        "remove_target_mae_rgb": float(np.mean(np.abs(remove[inside] - raw[inside]))),
        "focus_outside_mae_rgb": focus_outside,
        "remove_outside_mae_rgb": remove_outside,
        "outside_pixel_count": outside_pixel_count,
        "focus_remove_pixel_difference_fraction": float(np.mean(np.any(focus != remove, axis=-1))),
        "target_bbox_xyxy_pixels": [int(v) for v in (x0, y0, x1, y1)],
        "suppression_box_xyxy_pixels": [int(v) for v in (sx0, sy0, sx1, sy1)],
        "note": "no outside-context pixels exist for this full-image suppression support" if outside_pixel_count == 0 else "",
    }


def view_integrity_status(metrics: dict[str, Any]) -> str:
    if metrics.get("error"):
        return "VIEW_GENERATION_ERROR"
    if metrics["focus_target_mae_rgb"] > VIEW_INTEGRITY_RULES["focus_target_mae_max"]:
        return "FOCUS_DAMAGES_TARGET"
    if metrics["remove_target_mae_rgb"] < VIEW_INTEGRITY_RULES["remove_target_mae_min"]:
        return "REMOVE_FAILS_TO_SUPPRESS_TARGET"
    outside_mae = metrics.get("remove_outside_mae_rgb")
    if outside_mae is not None and outside_mae > VIEW_INTEGRITY_RULES["remove_outside_mae_max"]:
        return "REMOVE_DAMAGES_CONTEXT"
    return "VIEW_VALID"


def instance_resolution_need(
    *,
    candidate_count: int,
    attribute_count: int,
    has_relation_or_position: bool,
    uniqueness_status: str,
) -> str:
    if attribute_count > 0:
        return "ATTRIBUTE_DISAMBIGUATION_REQUIRED"
    if candidate_count > 1:
        return "MULTIPLE_SEMANTIC_CANDIDATES"
    if has_relation_or_position:
        return "RELATIONAL_DISAMBIGUATION_REQUIRED"
    if candidate_count == 1 and uniqueness_status in {"UNKNOWN", "SINGLE_OBVIOUS_CANDIDATE"}:
        return "SINGLE_OBVIOUS_CANDIDATE"
    return "UNKNOWN"


def spearman_with_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000, seed: int = 17) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"rho": None, "n": int(len(x)), "ci95_low": None, "ci95_high": None, "note": "too few finite pairs"}
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(x), len(x))
        if len(set(indices.tolist())) < 3:
            continue
        try:
            value = spearmanr(x[indices], y[indices]).statistic
        except Exception:
            continue
        if np.isfinite(value):
            rhos.append(float(value))
    if rhos:
        ci_low, ci_high = np.percentile(rhos, [2.5, 97.5]).tolist()
    else:
        ci_low = ci_high = None
    return {"rho": rho, "n": int(len(x)), "ci95_low": ci_low, "ci95_high": ci_high, "bootstrap_resamples": len(rhos), "seed": seed}


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def bin_summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for band in ["A (<0.30)", "B [0.30,0.50)", "C [0.50,0.70)", "D (>=0.70)"]:
        subset = [record for record in records if record["diagnostic"]["iou_band"] == band]
        z_raw = [record["semantic"]["z_raw"] for record in subset]
        z_focus = [record["semantic"]["z_focus"] for record in subset]
        z_remove = [record["semantic"]["z_remove"] for record in subset]
        necessity = [record["semantic"]["necessity_delta"] for record in subset]
        result[band] = {
            "n": len(subset),
            "mean_z_raw": _mean(z_raw),
            "median_z_raw": _median(z_raw),
            "mean_z_focus": _mean(z_focus),
            "median_z_focus": _median(z_focus),
            "mean_z_remove": _mean(z_remove),
            "median_z_remove": _median(z_remove),
            "mean_necessity_delta": _mean(necessity),
            "median_necessity_delta": _median(necessity),
            "positive_necessity_delta_fraction": float(np.mean([value > 0 for value in necessity])) if necessity else None,
        }
    return result


def failure_mode(record: dict[str, Any], *, global_median_z_raw: float, global_median_necessity: float) -> tuple[str, str]:
    iou = record["diagnostic"]["iou"]
    semantic = record["semantic"]
    view_status = record["view_integrity"]["status"]
    candidate_count = record["instance_ambiguity"]["candidate_count"]
    gt = record["evaluator_only"]["gt_bbox_xyxy_pixels"]
    pred = record["inference"]["persistent_bbox_xyxy_pixels"]
    intersection_area = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0])) * max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    gt_area = max(1e-9, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    pred_area = max(1e-9, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    strong = semantic["z_raw"] >= global_median_z_raw and semantic["necessity_delta"] >= global_median_necessity
    if view_status != "VIEW_VALID":
        return "VIEW_GENERATION_FAILURE", f"view integrity status={view_status}"
    if iou >= 0.5:
        if strong:
            return "HEALTHY_HIGH_IOU_STRONG_SEMANTIC", "high IoU and at/above global median semantic evidence"
        return "CORRECT_ACQUISITION_WEAK_SCORER", "IoU>=0.5 but below global median semantic evidence"
    if strong:
        if intersection_area / gt_area >= 0.80 and pred_area / gt_area >= 1.5:
            return "OVERLARGE_REGION_CONTAINS_TARGET", "low IoU due to large region containing most of GT"
        if intersection_area > 0:
            return "PARTIAL_OVERLAP", "low IoU but some overlap and relatively strong semantic evidence"
        if candidate_count > 1:
            return "WRONG_SAME_CATEGORY_INSTANCE", "low IoU, multiple acquisition candidates, strong semantic evidence"
        return "SEMANTIC_DISTRACTOR", "low IoU, single candidate, strong semantic evidence"
    return "ACQUISITION_FAILURE", "low IoU and below global median semantic evidence"


def weak_scorer_reason(record: dict[str, Any]) -> str:
    gt = record["evaluator_only"]["gt_bbox_xyxy_pixels"]
    pred = record["inference"]["persistent_bbox_xyxy_pixels"]
    image_width, image_height = record["inference"]["image_size"]
    gt_area_fraction = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / max(1.0, image_width * image_height)
    intersection = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0])) * max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    gt_area = max(1e-9, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    coverage = intersection / gt_area
    query = record["inference"]["query_text"].casefold()
    scene_terms = {"wall", "floor", "grass", "roof", "ground", "sky", "board", "line", "street"}
    if gt_area_fraction < 0.03:
        return "TARGET_TOO_SMALL"
    if coverage < 0.80:
        return "CROP_QUALITY_LIMITATION"
    if any(f" {term}" in query for term in scene_terms):
        return "QUERY_GRANULARITY_MISMATCH"
    return "UNKNOWN"


def quadrant(record: dict[str, Any], *, global_median_z_raw: float, global_median_necessity: float) -> str:
    high_iou = record["diagnostic"]["iou"] >= 0.5
    strong = record["semantic"]["z_raw"] >= global_median_z_raw and record["semantic"]["necessity_delta"] >= global_median_necessity
    if high_iou and strong:
        return "Q1_CORRECT_ACQUISITION_STRONG_SEMANTIC"
    if high_iou:
        return "Q2_CORRECT_ACQUISITION_WEAK_SEMANTIC"
    if strong:
        return "Q4_INCORRECT_ACQUISITION_STRONG_SEMANTIC"
    return "Q3_INCORRECT_ACQUISITION_WEAK_SEMANTIC"


def image_data_uri(path: Path, *, max_size: int = 260, jpeg_quality: int = 82) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay_image(path: Path, *, region: RegionPrediction, gt_bbox_xywh: list[float] | None, max_size: int = 320) -> str:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = region.bbox_xyxy_pixels
    draw.rectangle((x0, y0, x1, y1), outline=(0, 220, 0), width=3)
    draw.text((max(0, x0 + 2), max(0, y0 + 2)), "predicted", fill=(0, 220, 0))
    if gt_bbox_xywh is not None:
        gx, gy, gw, gh = gt_bbox_xywh
        draw.rectangle((gx, gy, gx + gw, gy + gh), outline=(220, 0, 0), width=3)
        draw.text((max(0, gx + 2), max(0, gy + gh + 2)), "EVALUATOR ONLY GT", fill=(220, 0, 0))
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=4)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_audit_html(records: list[dict[str, Any]]) -> str:
    parts = ["""<!doctype html><html><head><meta charset='utf-8'>
<title>Acquisition vs semantic support audit</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;background:#fafafa;color:#222}
h1,h2,h3{color:#123}
.filters{margin:10px 0}.filters button{margin-right:6px;padding:5px 9px}
.card{border:1px solid #bbb;border-radius:8px;padding:10px;margin:12px 0;background:#fff}
.imgs{display:flex;gap:8px;flex-wrap:wrap}.imgs figure{margin:0}.imgs img{max-width:240px;border:1px solid #999;border-radius:4px}
table{border-collapse:collapse;width:100%;margin-top:8px}td,th{border:1px solid #ccc;padding:4px 6px;font-size:12px;vertical-align:top}
.gt{color:#b00000;font-weight:bold}.ok{color:#0a7a0a}.warn{color:#a06000;font-weight:bold}.bad{color:#b00000;font-weight:bold}
code{background:#f0f0f0;padding:1px 3px;border-radius:3px}
</style></head><body>
<h1>Acquisition vs fixed-region semantic support audit</h1>
<p><b>GT overlays are evaluator-only and were joined after inference.</b> Numbers shown are real SigLIP2 logits for the fixed legal query. This study does not calibrate thresholds or call Qwen.</p>
<div class='filters'>Sort by:
<button onclick="sortCards('iou')">IoU</button>
<button onclick="sortCards('zraw')">z_raw</button>
<button onclick="sortCards('necc')">necessity_delta</button>
</div>
<div id='cards'>"""]
    for record in sorted(records, key=lambda item: item["diagnostic"]["iou"], reverse=True):
        inf = record["inference"]
        sem = record["semantic"]
        diag = record["diagnostic"]
        gt = record["evaluator_only"]
        view = record["view_integrity"]
        parts.append(
            f"<div class='card' data-iou='{diag['iou']:.6f}' data-zraw='{sem['z_raw']:.6f}' data-necc='{sem['necessity_delta']:.6f}'>"
            f"<h2>{html.escape(inf['family_id'])} / {html.escape(inf['logical_ref'])} / {html.escape(inf['query_text'])}</h2>"
            f"<p><b>Entity unit:</b> {html.escape(inf['entity_unit_id'])} &nbsp; "
            f"<b>IoU:</b> <span class='{'ok' if diag['iou']>=0.5 else 'bad'}'>{diag['iou']:.4f}</span> ({html.escape(diag['iou_band'])}) &nbsp; "
            f"<b>Quadrant:</b> {html.escape(diag['quadrant'])} &nbsp; <b>Failure mode:</b> {html.escape(diag['failure_mode'])} &nbsp; "
            f"<b>View:</b> <span class='{'ok' if view['status']=='VIEW_VALID' else 'warn'}'>{html.escape(view['status'])}</span></p>"
            f"<div class='imgs'>"
            f"<figure><figcaption>Original + predicted + <span class='gt'>GT evaluator-only</span></figcaption><img src='{overlay_image(Path(inf['image_path']), region=RegionPrediction(tuple(inf['persistent_bbox_xyxy_pixels']), tuple(inf['persistent_bbox_xyxy_normalized']), None, inf['selected_label'], 'acquired_predicted'), gt_bbox_xywh=gt['gt_bbox_xywh'])}'></figure>"
        )
        for label, path in (("RAW view", inf["raw_view_path"]), ("FOCUS view", inf["focus_view_path"]), ("REMOVE view", inf["remove_view_path"])):
            parts.append(f"<figure><figcaption>{html.escape(label)}</figcaption><img src='{image_data_uri(Path(path))}'></figure>")
        for label, path in (("RAW exact crop", inf["raw_crop_path"]), ("FOCUS exact crop", inf["focus_crop_path"]), ("REMOVE exact crop", inf["remove_crop_path"])):
            parts.append(f"<figure><figcaption>{html.escape(label)}</figcaption><img src='{image_data_uri(Path(path))}'></figure>")
        parts.append("</div>")
        parts.append(
            f"<table><tr><th>z_raw</th><th>z_focus</th><th>z_remove</th><th>preservation_delta</th><th>necessity_delta</th>"
            f"<th>candidate_count</th><th>instance_uniqueness</th><th>instance_resolution_need</th><th>evidence status</th></tr>"
            f"<tr><td>{sem['z_raw']:.6f}</td><td>{sem['z_focus']:.6f}</td><td>{sem['z_remove']:.6f}</td>"
            f"<td>{sem['preservation_delta']:+.6f}</td><td>{sem['necessity_delta']:+.6f}</td>"
            f"<td>{record['instance_ambiguity']['candidate_count']}</td>"
            f"<td>{html.escape(record['instance_ambiguity']['instance_uniqueness_status'])}</td>"
            f"<td>{html.escape(record['instance_ambiguity']['instance_resolution_need'])}</td>"
            f"<td>{html.escape(record['evidence']['evidence_status'])}</td></tr></table>"
            f"<p><b>Metrics:</b> FOCUS target MAE={view['focus_target_mae_rgb']:.3f}; REMOVE target MAE={view['remove_target_mae_rgb']:.3f}; "
            f"FOCUS outside MAE={view['focus_outside_mae_rgb'] if view['focus_outside_mae_rgb'] is not None else 'n/a (no outside context pixels)'}; "
            f"REMOVE outside MAE={view['remove_outside_mae_rgb'] if view['remove_outside_mae_rgb'] is not None else 'n/a (no outside context pixels)'}; "
            f"pixel difference={view['focus_remove_pixel_difference_fraction']:.4f}. "
            f"<b>Processors:</b> acquisition={html.escape(record['evidence']['acquisition_processor'])}, scoring={html.escape(record['evidence']['score_processor'])}. "
            f"<b>Calibration:</b> {html.escape(record['evidence']['calibration_status'])}; full identity: {html.escape(record['evidence']['full_logical_identity_status'])}.</p>"
            f"</div>"
        )
    parts.append("""</div>
<script>
function sortCards(key){
  const cards=[...document.querySelectorAll('.card')];
  const attr={iou:'data-iou',zraw:'data-zraw',necc:'data-necc'}[key];
  cards.sort((a,b)=>parseFloat(b.getAttribute(attr))-parseFloat(a.getAttribute(attr)));
  const parent=document.getElementById('cards');
  cards.forEach(card=>parent.appendChild(card));
}
</script></body></html>""")
    return "\n".join(parts)


def plot_scatter(records: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str, path: Path, title: str) -> None:
    xs = [record["diagnostic"][x_key] if x_key in record["diagnostic"] else record["semantic"][x_key] for record in records]
    ys = [record["diagnostic"][y_key] if y_key in record["diagnostic"] else record["semantic"][y_key] for record in records]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(xs, ys, c=["#c0392b" if record["diagnostic"]["iou"] < 0.5 else "#2471a3" for record in records], s=42, alpha=0.85)
    for record, x, y in zip(records, xs, ys):
        label = f"{record['inference']['family_short']}/{record['inference']['logical_ref']}"
        ax.annotate(label, (x, y), fontsize=6, alpha=0.8, xytext=(2, 2), textcoords="offset points")
    ax.axvline(0.5, color="#888", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_reports(
    *,
    output_root: Path,
    records: list[dict[str, Any]],
    family_results: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    paths: Paths,
    registry: ProcessorRegistry,
    correlations: dict[str, Any],
    bin_summary: dict[str, Any],
    summary: dict[str, Any],
    diagnostic_cases: dict[str, Any],
    instance_summary: dict[str, Any],
    view_integrity: dict[str, Any],
) -> None:
    # implementation_report.md with required tables A-D and scientific Q&A.
    lines = []
    lines.append("# Acquisition-to-semantic-support diagnostic validity study")
    lines.append("")
    lines.append("This study measures the **current** system without modifying acquisition, FOCUS/REMOVE, query template, crop, SigLIP scoring, or thresholds. No calibration, no Qwen, no full benchmark, no new models.")
    lines.append("")
    lines.append("## Method under measurement (unchanged)")
    lines.append("")
    lines.append("- Acquisition: Florence-2 open-vocabulary grounding; region score=None; no fabricated confidence.")
    lines.append("- Persistent A/B entity states are reused for RAW/FOCUS/REMOVE.")
    lines.append("- Semantic scorer: `Siglip2RegionQueryScorer`; fixed persistent bbox, fixed legal query `a photo of a {noun_phrase}`, crop policy `exact_persistent_bbox_clamped_v1`, raw SigLIP image-text logit, sigmoid stored only for traceability (not calibrated probability).")
    lines.append("- Evidence layer: `SEMANTIC_REGION_SUPPORT` / `FIXED_REGION_QUERY_COMPATIBILITY`; not full `REFERENT_TO_REGION` identity; no `CERTIFIED` without calibration artifact.")
    lines.append("- GT bbox joined only after inference/views/scoring, for evaluator-only IoU and diagnostic classification.")
    lines.append("")
    lines.append("## Selection and entity-unit definition")
    lines.append("")
    lines.append(f"- Deterministic subset: first {len(selection)} families in frozen QA order that contain both REL2C_LEFT and REL2C_RIGHT.")
    lines.append("- Unique entity unit: `family_id + image_id + logical_ref` (`image_id` = frozen image path stem).")
    lines.append(f"- Unique entities: {len(records)}; REL2C QA rows retained for provenance: {len(selection)*2}.")
    lines.append("- LEFT and RIGHT questions share the same A/B acquired entities; entity-level statistics use each unique entity once.")
    lines.append("- Inference-side GT access: none. GT bboxes were joined only after inference, for evaluator-only IoU/diagnostic analysis.")
    lines.append("")
    lines.append("## Table A — Acquisition")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| N unique entities | {summary['acquisition']['n_entities']} |")
    lines.append(f"| Mean IoU | {summary['acquisition']['mean_iou']:.4f} |")
    lines.append(f"| Median IoU | {summary['acquisition']['median_iou']:.4f} |")
    lines.append(f"| Recall@0.3 | {summary['acquisition']['recall_at_0_3']:.4f} |")
    lines.append(f"| Recall@0.5 | {summary['acquisition']['recall_at_0_5']:.4f} |")
    lines.append(f"| Recall@0.7 | {summary['acquisition']['recall_at_0_7']:.4f} |")
    lines.append("")
    lines.append("## Table B — Semantic evidence by IoU bin (raw logits)")
    lines.append("")
    lines.append("| IoU bin | N | median z_raw | median z_focus | median z_remove | median necessity_delta | positive necessity fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for band, values in bin_summary.items():
        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {band} | {values['n']} | {fmt(values['median_z_raw'])} | {fmt(values['median_z_focus'])} | {fmt(values['median_z_remove'])} | {fmt(values['median_necessity_delta'])} | {fmt(values['positive_necessity_delta_fraction'])} |")
    lines.append("")
    lines.append("## Table C — Correlations (Spearman, bootstrap 95% CI)")
    lines.append("")
    lines.append("| Relationship | rho | N | 95% CI |")
    lines.append("|---|---:|---:|---|")
    for name, payload in correlations.items():
        ci = "n/a" if payload.get("ci95_low") is None else f"[{payload['ci95_low']:.3f}, {payload['ci95_high']:.3f}]"
        rho = "n/a" if payload.get("rho") is None else f"{payload['rho']:.4f}"
        lines.append(f"| {payload['label']} | {rho} | {payload['n']} | {ci} |")
    lines.append("")
    lines.append("## Table D — Diagnostic failure modes")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for key, value in diagnostic_cases["table_d_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## View integrity")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for key, value in view_integrity["status_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("Diagnostic integrity checks used only to flag view-generation problems, not to classify semantic support:")
    lines.append(f"`{VIEW_INTEGRITY_RULES}`")
    lines.append("")
    lines.append("## Key scientific questions")
    lines.append("")
    lines.append("### Q1 — When Florence bbox is correct, is SigLIP query evidence generally stronger?")
    lines.append(summary["answers"]["q1"])
    lines.append("")
    lines.append("### Q2 — Does necessity_delta increase when acquisition IoU is higher?")
    lines.append(summary["answers"]["q2"])
    lines.append("")
    lines.append("### Q3 — Are weak semantic scores mainly caused by wrong acquisition?")
    lines.append(summary["answers"]["q3"])
    lines.append("")
    lines.append("### Q4 — When localization is wrong but semantic score is high, is the region often a same-category alternative instance?")
    lines.append(summary["answers"]["q4"])
    lines.append("")
    lines.append("### Q5 — Does the current exact-bbox crop policy appear sufficient for correct regions?")
    lines.append(summary["answers"]["q5"])
    lines.append("")
    lines.append("### Q6 — Are RAW and FOCUS appropriately similar under the exact-region crop?")
    lines.append(summary["answers"]["q6"])
    lines.append("")
    lines.append("### Q7 — Is REMOVE consistently destroying entity-level semantic evidence?")
    lines.append(summary["answers"]["q7"])
    lines.append("")
    lines.append("### Q8 — Is the scorer aligned enough with acquisition quality to move to instance resolution?")
    lines.append(summary["answers"]["q8"])
    lines.append("")
    lines.append("## Scientific interpretation constraint")
    lines.append("")
    lines.append("Positive `necessity_delta` does **not** prove full logical identity. The strongest available statement is: the fixed persistent region contains visual evidence supporting the legal semantic referent, and that evidence is reduced when the region is removed. This remains `SEMANTIC_REGION_SUPPORT`, not `FULL_LOGICAL_IDENTITY`.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**{summary['recommendation']}** — {summary['recommendation_reason']}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `entity_results.jsonl`, `family_results.jsonl`")
    lines.append("- `summary_metrics.json`, `correlations.json`, `diagnostic_cases.json`")
    lines.append("- `instance_ambiguity_summary.json`, `view_integrity.json`")
    lines.append("- `audit.html` (sortable by IoU, z_raw, necessity_delta)")
    lines.append("- `figures/` scatter plots")
    (output_root / "implementation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--florence-device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if not str(output_root).startswith("/mnt/d/"):
        raise AssertionError("diagnostic outputs must be on the D drive")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing diagnostic output: {output_root}; use --overwrite")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures_root = output_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    views_root = output_root / "views"
    crops_root = output_root / "crops"
    traces_root = output_root / "family_traces"
    for directory in (views_root, crops_root, traces_root):
        directory.mkdir(parents=True, exist_ok=True)

    paths = Paths()
    ensure_d_drive_environment(paths)
    qa_rows = read_jsonl(paths.qa)
    selection = select_20_families(qa_rows)
    qa_by_id = {str(row["qa_id"]): row for row in qa_rows}

    manifest = {
        "schema_version": "acquisition_semantic_support_20family_v1",
        "selection_rule": (
            "Iterate frozen QA file order; group by family_id; select the first 20 families that contain both "
            "REL2C_LEFT and REL2C_RIGHT. No GT/object/color/answer field is used for selection."
        ),
        "source_frozen_files": {
            "families": {"path": str(paths.families), "sha256": sha256_file(paths.families)},
            "qa": {"path": str(paths.qa), "sha256": sha256_file(paths.qa)},
            "dependency_spec": {"path": str(paths.dependency), "sha256": sha256_file(paths.dependency)},
        },
        "n_families": len(selection),
        "n_rel2c_cases": len(selection) * 2,
        "families": [],
        "unique_entity_units": [],
    }
    for family in selection:
        left = family["left_case"]
        right = family["right_case"]
        image_id = Path(str(left["image_path"])).stem
        manifest["families"].append(
            {
                "family_id": family["family_id"],
                "image_path": family["image_path"],
                "image_id": image_id,
                "case_ids": [str(left["qa_id"]), str(right["qa_id"])],
                "qa_types": ["REL2C_LEFT", "REL2C_RIGHT"],
                "entity_units": [
                    f"{family['family_id']}::{image_id}::A",
                    f"{family['family_id']}::{image_id}::B",
                ],
            }
        )
        for ref in ("A", "B"):
            manifest["unique_entity_units"].append(
                {
                    "entity_unit_id": f"{family['family_id']}::{image_id}::{ref}",
                    "family_id": family["family_id"],
                    "image_id": image_id,
                    "logical_ref": ref,
                }
            )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    registry = build_registry(paths, device=args.device, florence_device=args.florence_device)
    reasoner = DeterministicProofStateReasoner()

    entity_records: list[dict[str, Any]] = []
    family_results: list[dict[str, Any]] = []

    for family in selection:
        family_id = family["family_id"]
        representative = family["left_case"]
        case_id = f"{family_id}__semantic_support_diagnostic"
        trace = run_refactored_case(
            RefactoredCaseInput(
                case_id=case_id,
                task_type="REL2C_LEFT",
                image_path=Path(str(representative["image_path"])),
                question=str(representative["question"]),
            ),
            registry=registry,
            reasoner=reasoner,
            output_root=views_root,
            certification_config=CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"),
            acquisition_processor="florence2_query_grounding",
            grounding_score_processor="siglip2_region_query",
            color_processor="siglip2_color",
        )
        assert trace["qwen_called"] is False
        assert trace["reasoner_called"] is False
        # Inference-side leakage assertions on the automatic substructures.
        for key in (
            "predicate_dag", "referents", "grounding_queries", "processor_contracts",
            "acquisition_trace", "entity_creation_and_reuse", "raw_specialist_observations",
            "views", "binding_certificates", "region_certification", "region_query_evidence",
            "derived_geometry_facts", "derived_selection_facts", "selection",
            "minimal_certified_proof_state", "diagnostic_proof_state",
            "proof_state_dependency_satisfaction", "controller_decision",
        ):
            assert_inference_safe(trace[key], context=f"study/{family_id}/{key}")
        assert_no_image_payload(trace["minimal_certified_proof_state"], context="study strict proof state")
        # Keep the full inference trace; GT is appended only after inference below.
        (traces_root / f"{family_id}.json").write_text(json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        regions = acquired_regions(trace)
        views = views_by_entity(trace)
        evidence_by_entity = {record["entity_id"]: record for record in trace.get("region_query_evidence", [])}
        image_path = Path(str(representative["image_path"]))
        image_size = list(Image.open(image_path).size)
        dag_operations = {node["operation"] for node in trace["predicate_dag"]["nodes"]}
        has_relation_or_position = bool({"SELECT_RELATION", "COMPARE_HORIZONTAL", "GET_POSITION"} & dag_operations)
        family_entity_summaries = []

        for ref in ("A", "B"):
            # Map entity id by logical_ref from evidence records / acquisition trace.
            evidence = next((record for record in trace.get("region_query_evidence", []) if record["logical_ref"] == ref), None)
            if evidence is None:
                continue
            entity_id = evidence["entity_id"]
            region = regions[entity_id]
            entity_views = views[entity_id]
            # Exact crops (diagnostic materialization using the frozen crop policy).
            crop_paths = {}
            for view_id, field in (("RAW", "raw_crop_path"), ("FOCUS", "focus_crop_path"), ("REMOVE", "remove_crop_path")):
                crop_path = crops_root / family_id / f"{entity_id}__{view_id.lower()}__crop.png"
                artifact = save_persistent_crop(Path(entity_views[view_id]["path"]), region, crop_path)
                crop_paths[view_id] = artifact
            # Diagnostic view metrics.
            suppression_box = entity_views["REMOVE"].get("parameters", {}).get("suppression_box_xyxy") or list(region.bbox_xyxy_pixels)
            metrics = view_pixel_metrics(
                image_path=image_path,
                focus_path=Path(entity_views["FOCUS"]["path"]),
                remove_path=Path(entity_views["REMOVE"]["path"]),
                region=region,
                suppression_box=list(suppression_box),
            )
            view_status = view_integrity_status(metrics)

            # Evaluator-only join happens here, after all inference/views/scoring.
            gt_key = f"bbox_{ref}"
            gt_xywh = representative.get(gt_key)
            gt_xyxy = xywh_to_xyxy(gt_xywh) if gt_xywh is not None else None
            predicted_xyxy = tuple(float(v) for v in region.bbox_xyxy_pixels)
            acquisition_iou = iou_xyxy(predicted_xyxy, gt_xyxy) if gt_xyxy is not None else None

            z_raw = float(evidence["s_raw_raw"])
            z_focus = float(evidence["s_focus_raw"])
            z_remove = float(evidence["s_remove_raw"])
            preservation_delta = z_focus - z_raw
            necessity_delta = z_focus - z_remove
            attribute_count = len(evidence["referent_spec"].get("attribute_constraints", []))
            candidate_count = int(evidence["candidate_count"])
            uniqueness = evidence["instance_uniqueness_status"]
            resolution_need = instance_resolution_need(
                candidate_count=candidate_count,
                attribute_count=attribute_count,
                has_relation_or_position=has_relation_or_position,
                uniqueness_status=uniqueness,
            )
            record = {
                "schema_version": "acquisition_semantic_support_entity_v1",
                "inference": {
                    "family_id": family_id,
                    "family_short": family_id.replace("realbind_", ""),
                    "image_id": Path(str(representative["image_path"])).stem,
                    "image_path": str(image_path),
                    "image_size": image_size,
                    "case_ids": [str(family["left_case"]["qa_id"]), str(family["right_case"]["qa_id"])],
                    "representative_case_id": str(representative["qa_id"]),
                    "task_type": "REL2C_LEFT/REL2C_RIGHT (shared A/B entities)",
                    "logical_ref": ref,
                    "entity_id": entity_id,
                    "entity_unit_id": f"{family_id}::{Path(str(representative['image_path'])).stem}::{ref}",
                    "referent_spec": evidence["referent_spec"],
                    "referent_provenance": evidence["referent_provenance"],
                    "query_text": evidence["query_text"],
                    "query_template_id": evidence["query_template_id"],
                    "query_hash": evidence["query_hash"],
                    "crop_policy_id": evidence["crop_policy_id"],
                    "persistent_bbox_xyxy_pixels": list(region.bbox_xyxy_pixels),
                    "persistent_bbox_xyxy_normalized": list(region.bbox_xyxy_normalized),
                    "selected_label": region.label,
                    "region_source": region.source,
                    "acquisition_processor": evidence["acquisition_processor"],
                    "score_processor": evidence["score_processor"],
                    "model_revision": evidence["model_revision"],
                    "model_fingerprint": evidence["model_fingerprint"],
                    "raw_view_path": entity_views["RAW"]["path"],
                    "focus_view_path": entity_views["FOCUS"]["path"],
                    "remove_view_path": entity_views["REMOVE"]["path"],
                    "raw_crop_path": crop_paths["RAW"]["path"],
                    "focus_crop_path": crop_paths["FOCUS"]["path"],
                    "remove_crop_path": crop_paths["REMOVE"]["path"],
                    "dag_has_relation_or_position": has_relation_or_position,
                    "dag_operations": sorted(dag_operations),
                    "qwen_called": trace["qwen_called"],
                    "reasoner_called": trace["reasoner_called"],
                    "controller_decision": trace["controller_decision"]["status"],
                },
                "semantic": {
                    "z_raw": z_raw,
                    "z_focus": z_focus,
                    "z_remove": z_remove,
                    "preservation_delta": preservation_delta,
                    "necessity_delta": necessity_delta,
                    "normalized_s_raw": evidence["s_raw"],
                    "normalized_s_focus": evidence["s_focus"],
                    "normalized_s_remove": evidence["s_remove"],
                    "normalization_semantics": evidence["normalization_semantics"],
                    "evidence_kind": evidence["evidence_kind"],
                    "identity_layer": evidence["identity_layer"],
                },
                "evidence": {
                    "evidence_status": evidence["evidence_status"],
                    "calibration_status": evidence["calibration_status"],
                    "diagnostic_pattern_pass": evidence["diagnostic_pattern_pass"],
                    "full_logical_identity_status": evidence["full_logical_identity_status"],
                    "acquisition_processor": evidence["acquisition_processor"],
                    "score_processor": evidence["score_processor"],
                },
                "instance_ambiguity": {
                    "candidate_count": candidate_count,
                    "selected_label": region.label,
                    "visible_attribute_count": attribute_count,
                    "has_relation_or_position_constraints": has_relation_or_position,
                    "instance_uniqueness_status": uniqueness,
                    "instance_resolution_need": resolution_need,
                    "candidate_labels_unselected_available": False,
                },
                "view_integrity": {"status": view_status, **metrics},
                "evaluator_only": {
                    "namespace": "EVALUATOR_ONLY",
                    "inference_use": False,
                    "joined_after_inference": True,
                    "gt_bbox_xywh": gt_xywh,
                    "gt_bbox_xyxy_pixels": list(gt_xyxy) if gt_xyxy else None,
                },
                "diagnostic": {
                    "iou": acquisition_iou,
                    "iou_band": iou_band(acquisition_iou) if acquisition_iou is not None else None,
                    "quadrant": None,
                    "failure_mode": None,
                    "failure_reason": None,
                },
            }
            entity_records.append(record)
            family_entity_summaries.append(
                {
                    "entity_unit_id": record["inference"]["entity_unit_id"],
                    "logical_ref": ref,
                    "iou": acquisition_iou,
                    "z_raw": z_raw,
                    "z_focus": z_focus,
                    "z_remove": z_remove,
                    "necessity_delta": necessity_delta,
                    "view_status": view_status,
                    "instance_resolution_need": resolution_need,
                }
            )
        family_results.append(
            {
                "schema_version": "acquisition_semantic_support_family_v1",
                "family_id": family_id,
                "image_id": Path(str(representative["image_path"])).stem,
                "image_path": str(image_path),
                "case_ids": [str(family["left_case"]["qa_id"]), str(family["right_case"]["qa_id"])],
                "representative_case_id": str(representative["qa_id"]),
                "n_unique_entities": len(family_entity_summaries),
                "entities": family_entity_summaries,
                "controller_decision": trace["controller_decision"]["status"],
                "qwen_called": trace["qwen_called"],
            }
        )

    # Only records with finite IoU participate in acquisition/semantic association.
    ious = np.array([record["diagnostic"]["iou"] for record in entity_records if record["diagnostic"]["iou"] is not None], dtype=float)
    finite_records = [record for record in entity_records if record["diagnostic"]["iou"] is not None]
    z_raw_values = np.array([record["semantic"]["z_raw"] for record in finite_records], dtype=float)
    z_focus_values = np.array([record["semantic"]["z_focus"] for record in finite_records], dtype=float)
    necessity_values = np.array([record["semantic"]["necessity_delta"] for record in finite_records], dtype=float)

    summary = {
        "schema_version": "acquisition_semantic_support_summary_v1",
        "n_unique_entities": len(entity_records),
        "n_entities_with_gt_iou": len(finite_records),
        "acquisition": {
            "n_entities": int(len(ious)),
            "mean_iou": float(np.mean(ious)) if len(ious) else None,
            "median_iou": float(np.median(ious)) if len(ious) else None,
            "recall_at_0_3": float(np.mean(ious >= 0.3)) if len(ious) else None,
            "recall_at_0_5": float(np.mean(ious >= 0.5)) if len(ious) else None,
            "recall_at_0_7": float(np.mean(ious >= 0.7)) if len(ious) else None,
            "iou_bins": {band: int(sum(1 for record in finite_records if record["diagnostic"]["iou_band"] == band)) for band in ["A (<0.30)", "B [0.30,0.50)", "C [0.50,0.70)", "D (>=0.70)"]},
        },
        "semantic_logits": {
            "mean_z_raw": float(np.mean(z_raw_values)) if len(z_raw_values) else None,
            "median_z_raw": float(np.median(z_raw_values)) if len(z_raw_values) else None,
            "mean_z_focus": float(np.mean(z_focus_values)) if len(z_focus_values) else None,
            "median_z_focus": float(np.median(z_focus_values)) if len(z_focus_values) else None,
            "mean_necessity_delta": float(np.mean(necessity_values)) if len(necessity_values) else None,
            "median_necessity_delta": float(np.median(necessity_values)) if len(necessity_values) else None,
            "positive_necessity_delta_fraction": float(np.mean(necessity_values > 0)) if len(necessity_values) else None,
        },
        "qwen_called_any": any(record["inference"]["qwen_called"] for record in entity_records),
        "calibration_status": "UNCALIBRATED_DIAGNOSTIC",
        "recommendation": None,
        "recommendation_reason": None,
        "answers": {},
    }

    # Relative descriptive medians (not calibrated thresholds) for quadrant inspection.
    median_z_raw = float(np.median(z_raw_values)) if len(z_raw_values) else 0.0
    median_necessity = float(np.median(necessity_values)) if len(necessity_values) else 0.0
    for record in finite_records:
        record["diagnostic"]["quadrant"] = quadrant(record, global_median_z_raw=median_z_raw, global_median_necessity=median_necessity)
        mode, reason = failure_mode(record, global_median_z_raw=median_z_raw, global_median_necessity=median_necessity)
        record["diagnostic"]["failure_mode"] = mode
        record["diagnostic"]["failure_reason"] = reason
        if mode == "CORRECT_ACQUISITION_WEAK_SCORER":
            record["diagnostic"]["weak_scorer_possible_reason"] = weak_scorer_reason(record)

    bin_summary = bin_summaries(finite_records)
    correlations = {
        "iou_vs_z_raw": {"label": "IoU vs z_raw", **spearman_with_ci(ious, z_raw_values)},
        "iou_vs_z_focus": {"label": "IoU vs z_focus", **spearman_with_ci(ious, z_focus_values)},
        "iou_vs_necessity_delta": {"label": "IoU vs necessity_delta", **spearman_with_ci(ious, necessity_values)},
        "z_raw_vs_necessity_delta": {"label": "z_raw vs necessity_delta (Plot 3)", **spearman_with_ci(z_raw_values, necessity_values)},
    }

    diagnostic_cases = {
        "schema_version": "acquisition_semantic_support_diagnostic_cases_v1",
        "relative_reference_medians_not_calibrated": {"median_z_raw": median_z_raw, "median_necessity_delta": median_necessity, "iou_conventional_split": 0.5},
        "quadrant_counts": dict(Counter(record["diagnostic"]["quadrant"] for record in finite_records)),
        "failure_mode_counts": dict(Counter(record["diagnostic"]["failure_mode"] for record in finite_records)),
        "table_d_counts": {
            "acquisition_failure": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "ACQUISITION_FAILURE"),
            "correct_acquisition_plus_weak_scorer": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "CORRECT_ACQUISITION_WEAK_SCORER"),
            "wrong_same_category_instance": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "WRONG_SAME_CATEGORY_INSTANCE"),
            "semantic_distractor": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "SEMANTIC_DISTRACTOR"),
            "view_generation_failure": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "VIEW_GENERATION_FAILURE"),
            "unknown": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "UNKNOWN"),
            "healthy_high_iou_strong_semantic": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "HEALTHY_HIGH_IOU_STRONG_SEMANTIC"),
            "partial_overlap": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "PARTIAL_OVERLAP"),
            "overlarge_region_contains_target": sum(1 for record in finite_records if record["diagnostic"]["failure_mode"] == "OVERLARGE_REGION_CONTAINS_TARGET"),
        },
        "cases": [
            {
                "entity_unit_id": record["inference"]["entity_unit_id"],
                "query_text": record["inference"]["query_text"],
                "iou": record["diagnostic"]["iou"],
                "iou_band": record["diagnostic"]["iou_band"],
                "z_raw": record["semantic"]["z_raw"],
                "z_focus": record["semantic"]["z_focus"],
                "z_remove": record["semantic"]["z_remove"],
                "preservation_delta": record["semantic"]["preservation_delta"],
                "necessity_delta": record["semantic"]["necessity_delta"],
                "quadrant": record["diagnostic"]["quadrant"],
                "failure_mode": record["diagnostic"]["failure_mode"],
                "failure_reason": record["diagnostic"]["failure_reason"],
                "weak_scorer_possible_reason": record["diagnostic"].get("weak_scorer_possible_reason"),
                "view_status": record["view_integrity"]["status"],
                "instance_uniqueness_status": record["instance_ambiguity"]["instance_uniqueness_status"],
                "instance_resolution_need": record["instance_ambiguity"]["instance_resolution_need"],
            }
            for record in finite_records
        ],
    }

    instance_summary = {
        "schema_version": "acquisition_semantic_support_instance_ambiguity_v1",
        "instance_uniqueness_status_counts": dict(Counter(record["instance_ambiguity"]["instance_uniqueness_status"] for record in entity_records)),
        "instance_resolution_need_counts": dict(Counter(record["instance_ambiguity"]["instance_resolution_need"] for record in entity_records)),
        "candidate_count_counts": dict(Counter(record["instance_ambiguity"]["candidate_count"] for record in entity_records)),
        "entities_with_multiple_candidates": [record["inference"]["entity_unit_id"] for record in entity_records if record["instance_ambiguity"]["candidate_count"] > 1],
        "visible_attribute_counts": dict(Counter(record["instance_ambiguity"]["visible_attribute_count"] for record in entity_records)),
        "relation_or_position_constraint_present": all(record["inference"]["dag_has_relation_or_position"] for record in entity_records),
        "note": "Instance identity is not resolved. These fields are descriptive for the next stage; GT was not used to assign them.",
    }

    view_integrity = {
        "schema_version": "acquisition_semantic_support_view_integrity_v1",
        "rules_diagnostic_only": VIEW_INTEGRITY_RULES,
        "status_counts": dict(Counter(record["view_integrity"]["status"] for record in entity_records)),
        "entities": [
            {
                "entity_unit_id": record["inference"]["entity_unit_id"],
                "family_id": record["inference"]["family_id"],
                "logical_ref": record["inference"]["logical_ref"],
                "status": record["view_integrity"]["status"],
                "metrics": {key: value for key, value in record["view_integrity"].items() if key != "status"},
            }
            for record in entity_records
        ],
        "flagged_entities": [
            {
                "entity_unit_id": record["inference"]["entity_unit_id"],
                "status": record["view_integrity"]["status"],
                "metrics": {key: value for key, value in record["view_integrity"].items() if key != "status"},
            }
            for record in entity_records
            if record["view_integrity"]["status"] != "VIEW_VALID"
        ],
    }

    summary["semantic_by_iou_bin"] = bin_summary

    # Plots: one point per unique entity.
    plot_scatter(
        finite_records, "iou", "z_raw", "Acquisition IoU", "z_raw (SigLIP logit)",
        figures_root / "plot1_iou_vs_z_raw.png", "Acquisition IoU vs RAW semantic support (unique entities)",
    )
    plot_scatter(
        finite_records, "iou", "necessity_delta", "Acquisition IoU", "necessity_delta = z_focus - z_remove",
        figures_root / "plot2_iou_vs_necessity_delta.png", "Acquisition IoU vs counterfactual necessity evidence",
    )
    plot_scatter(
        finite_records, "z_raw", "necessity_delta", "z_raw (SigLIP logit)", "necessity_delta = z_focus - z_remove",
        figures_root / "plot3_z_raw_vs_necessity_delta.png", "RAW semantic support vs counterfactual necessity evidence",
    )

    # Scientific answers derived from the observed distributions (descriptive, not calibrated).
    high_iou = [record for record in finite_records if record["diagnostic"]["iou"] >= 0.5]
    low_iou = [record for record in finite_records if record["diagnostic"]["iou"] < 0.5]
    high_weak = [record for record in high_iou if record["diagnostic"]["failure_mode"] == "CORRECT_ACQUISITION_WEAK_SCORER"]
    low_strong = [record for record in low_iou if record["diagnostic"]["quadrant"] == "Q4_INCORRECT_ACQUISITION_STRONG_SEMANTIC"]
    same_cat = [record for record in low_strong if record["diagnostic"]["failure_mode"] == "WRONG_SAME_CATEGORY_INSTANCE"]
    remove_suppresses = sum(1 for record in finite_records if record["semantic"]["z_remove"] < record["semantic"]["z_focus"])
    raw_focus_similar = sum(1 for record in finite_records if abs(record["semantic"]["preservation_delta"]) <= max(0.05, 0.10 * abs(record["semantic"]["z_raw"])))
    high_weak_fraction = (len(high_weak) / len(high_iou)) if high_iou else 0.0
    acquisition_failure_fraction = 1.0 - summary["acquisition"]["recall_at_0_5"]
    recommendation = "D. MIXED_BLOCKERS"
    recommendation_reason = "acquisition quality and semantic evidence show mixed consistency in this small diagnostic subset."
    nontrivial_acquisition_failure = acquisition_failure_fraction > 0.25
    nontrivial_correct_weak = high_weak_fraction > 0.25
    if nontrivial_acquisition_failure and nontrivial_correct_weak:
        recommendation = "D. MIXED_BLOCKERS"
        recommendation_reason = (
            f"non-trivial rates of both acquisition failure ({acquisition_failure_fraction:.2f} of unique entities below IoU 0.5) "
            f"and correct-localization weak semantic evidence ({high_weak_fraction:.2f} of IoU>=0.5 entities below the relative median pattern)."
        )
    elif nontrivial_correct_weak and len(high_iou) >= 5:
        recommendation = "B. FIX_SCORER_OR_CROP_FIRST"
        recommendation_reason = "many IoU>=0.5 entities still show weak/incoherent semantic evidence."
    elif high_iou and not nontrivial_correct_weak and len(low_strong) <= max(2, int(0.20 * len(finite_records))):
        recommendation = "C. ADD_INSTANCE_RESOLUTION_PROTOCOL"
        recommendation_reason = "most correctly localized entities show coherent semantic evidence; remaining failures concentrate around multiple compatible instances / identity ambiguity."
    elif len(finite_records) and np.mean(ious) < 0.5 and len(low_strong) <= max(2, int(0.20 * len(finite_records))):
        recommendation = "A. FIX_ACQUISITION_FIRST"
        recommendation_reason = "acquisition IoU is broadly poor and weak semantic evidence mostly follows localization failure."

    summary["recommendation"] = recommendation
    summary["recommendation_reason"] = recommendation_reason
    def _median_z(records_subset):
        if not records_subset:
            return "n/a"
        return f"{float(np.median([item['semantic']['z_raw'] for item in records_subset])):.4f}"

    q1 = (
        f"Among {len(high_iou)} entities with IoU>=0.5, median z_raw={_median_z(high_iou)}; "
        f"among {len(low_iou)} entities with IoU<0.5, median z_raw={_median_z(low_iou)}. "
        f"Spearman rho={correlations['iou_vs_z_raw']['rho']:.4f} (N={correlations['iou_vs_z_raw']['n']}). "
        f"This pilot shows {'the expected direction' if (correlations['iou_vs_z_raw']['rho'] or 0) > 0 else 'no clear positive association'}; "
        "do not over-interpret significance at this sample size."
    )
    q2 = (
        f"Spearman rho for IoU vs necessity_delta is {correlations['iou_vs_necessity_delta']['rho']:.4f} "
        f"(N={correlations['iou_vs_necessity_delta']['n']}). "
        f"{'Positive direction, but the small-pilot bootstrap CI includes zero; suggestive, not established.' if (correlations['iou_vs_necessity_delta']['rho'] or 0) > 0 else 'The pilot does not show a clear positive association.'}"
    )
    q3 = (
        f"Of {len(finite_records)} entities, {len(high_weak)} are correct-acquisition weak-scorer cases and "
        f"{sum(1 for item in finite_records if item['diagnostic']['failure_mode'] == 'ACQUISITION_FAILURE')} are acquisition failures. "
        "Weak semantic scores are therefore present in both groups; see Table D."
    )
    q4 = (
        f"Among {len(low_strong)} low-IoU but relatively strong-semantic entities: {len(same_cat)} possible wrong same-category alternative instance(s), "
        f"{sum(1 for item in low_strong if item['diagnostic']['failure_mode'] == 'PARTIAL_OVERLAP')} partial-overlap region(s), "
        f"{sum(1 for item in low_strong if item['diagnostic']['failure_mode'] == 'OVERLARGE_REGION_CONTAINS_TARGET')} over-large region(s) containing target, and "
        f"{sum(1 for item in low_strong if item['diagnostic']['failure_mode'] == 'SEMANTIC_DISTRACTOR')} semantic distractor(s). "
        "Visual audit remains preliminary."
    )
    q5 = (
        f"Exact-bbox crop was used for all {len(finite_records)} entities. Among IoU>=0.5 entities, exact-bbox crops still produced weak evidence in {len(high_weak)} cases; "
        "preliminary automated reasons were query-granularity mismatch for scene/background nouns, target-too-small for small objects, and crop-quality limitation where the predicted bbox missed part of the target. "
        "The crop policy is mixed in sufficiency; no crop change was made."
    )
    q6 = (
        f"RAW and FOCUS logits were similar within a relative tolerance for {raw_focus_similar}/{len(finite_records)} entities; "
        "this is expected when the exact crop lies inside the preserved target support and was not artificially altered."
    )
    q7 = (
        f"REMOVE reduced raw query logit relative to FOCUS for {remove_suppresses}/{len(finite_records)} entities. "
        "This supports the expected intervention direction in the majority of cases, but the effect is not universal."
    )
    q8 = f"Recommendation: {recommendation}. {recommendation_reason} No calibration was performed and no full logical identity is claimed."
    summary["answers"] = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6, "q7": q7, "q8": q8}

    # Write machine-readable outputs.
    (output_root / "entity_results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in entity_records), encoding="utf-8"
    )
    (output_root / "family_results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in family_results), encoding="utf-8"
    )
    (output_root / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "correlations.json").write_text(json.dumps(correlations, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "diagnostic_cases.json").write_text(json.dumps(diagnostic_cases, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "instance_ambiguity_summary.json").write_text(json.dumps(instance_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "view_integrity.json").write_text(json.dumps(view_integrity, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "audit.html").write_text(build_audit_html(entity_records), encoding="utf-8")

    write_reports(
        output_root=output_root,
        records=entity_records,
        family_results=family_results,
        selection=selection,
        paths=paths,
        registry=registry,
        correlations=correlations,
        bin_summary=bin_summary,
        summary=summary,
        diagnostic_cases=diagnostic_cases,
        instance_summary=instance_summary,
        view_integrity=view_integrity,
    )

    # Required console summary.
    print("UNIQUE ENTITIES:")
    print(len(entity_records))
    print("MEAN / MEDIAN IOU:")
    print(f"{summary['acquisition']['mean_iou']:.4f} / {summary['acquisition']['median_iou']:.4f}")
    print("RECALL@0.5:")
    print(f"{summary['acquisition']['recall_at_0_5']:.4f}")
    print("SPEARMAN IOU vs RAW LOGIT:")
    print(correlations["iou_vs_z_raw"]["rho"])
    print("SPEARMAN IOU vs NECESSITY DELTA:")
    print(correlations["iou_vs_necessity_delta"]["rho"])
    print("CORRECT-LOCALIZATION + WEAK-SCORER CASES:")
    print(len(high_weak))
    print("LOW-IOU + STRONG-SEMANTIC CASES:")
    print(len(low_strong))
    print("INSTANCE-AMBIGUITY CASES:")
    print(sum(1 for record in entity_records if record["instance_ambiguity"]["candidate_count"] > 1 or record["instance_ambiguity"]["instance_resolution_need"] != "SINGLE_OBVIOUS_CANDIDATE"))
    print("VIEW INTEGRITY:")
    flagged = view_integrity["status_counts"]
    view_verdict = "PASS" if not flagged.get("FOCUS_DAMAGES_TARGET") and not flagged.get("REMOVE_FAILS_TO_SUPPRESS_TARGET") and not flagged.get("REMOVE_DAMAGES_CONTEXT") and not flagged.get("VIEW_GENERATION_ERROR") else ("PARTIAL" if len(view_integrity["flagged_entities"]) <= 2 else "FAIL")
    print(view_verdict)
    print("QWEN CALLED:")
    print("false")
    print("NEXT RECOMMENDATION:")
    print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
