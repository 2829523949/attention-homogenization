#!/usr/bin/env python3
"""Tiny corrected RealBind smoke runner (2-5 legally resolvable REL2C cases).

This is not a benchmark.  It records:
  * real Florence-2 acquisition (isolated dependency worker),
  * real automatic FOCUS and neutral-fill REMOVE views,
  * explicit certification refusals when the acquisition backend exposes no
    valid referent-region score,
  * real SigLIP-2 fixed-candidate colour diagnostics on the acquired views,
  * controller decisions and whether the final reasoner was invoked, and
  * evaluator-only GT fields clearly isolated after inference.

The main pipeline is fed only case_id/task_type/image_path/question.  GT joins
happen after inference solely for the audit HTML.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from harness.certification.certifier import CertificationConfig
from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.dataset import read_jsonl
from harness.processors.base import RegionPrediction
from harness.processors.color_siglip import SiglipColorReader
from harness.processors.florence_subprocess import FlorenceSubprocessGrounding
from harness.processors.registry import ColorReaderAdapter, ProcessorCapability, ProcessorRegistry
from harness.reasoners.hard_handoff_qwen import DeterministicProofStateReasoner
from harness.refactored_pipeline import RefactoredCaseInput, run_refactored_case

OUTPUT_ROOT = Path("/mnt/d/vlm_repair_v2/outputs/certification_repair")


def build_registry(paths: Paths, *, device: str, florence_device: str) -> ProcessorRegistry:
    from harness.processors.siglip_backend import Siglip2Backend

    registry = ProcessorRegistry()
    grounder = FlorenceSubprocessGrounding(paths.florence2, device=florence_device)
    registry.register(
        "florence2_query_grounding",
        ProcessorCapability(
            primitive="grounding",
            legal_input_types=("image", "GroundingQuery"),
            output_schema="list[RegionPrediction]",
            score_semantics=grounder.score_semantics,
            model_revision=grounder.version,
            model_path=str(paths.florence2),
            deterministic=True,
            supports_evidence_scoring=False,
            notes=(
                "ACQUISITION ONLY. Florence-2 OVD returns boxes/labels but no "
                "referent-region confidence; certification score unsupported."
            ),
            can_acquire=True,
            supported_predicates=("REFERENT_TO_REGION",),
            supports_same_claim_rescoring=False,
            supports_certification_score=False,
            score_normalization="none",
            comparable_across_counterfactual_views=False,
        ),
        grounder,
    )
    backend = Siglip2Backend(paths.siglip2, device=device)
    reader = SiglipColorReader(backend=backend)
    adapter = ColorReaderAdapter(reader)
    registry.register(
        "siglip2_color",
        ProcessorCapability(
            primitive="color",
            legal_input_types=("image", "region"),
            output_schema="Observation(predicate=COLOR)",
            score_semantics=reader.score_semantics,
            model_revision=reader.version,
            model_path=str(paths.siglip2),
            deterministic=True,
            supports_evidence_scoring=True,
            notes="Fixed-candidate SigLIP-2 colour logit margin; same claim scored on RAW/FOCUS/REMOVE.",
            can_acquire=True,
            supported_predicates=("COLOR",),
            supports_same_claim_rescoring=True,
            supports_certification_score=True,
            score_normalization=reader.score_normalization,
            comparable_across_counterfactual_views=True,
        ),
        adapter,
    )
    return registry


def select_smoke_rows(qa_rows: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    if not 2 <= max_cases <= 5:
        raise ValueError("tiny smoke must use 2-5 cases")
    eligible = [row for row in qa_rows if row.get("qa_type") in {"REL2C_LEFT", "REL2C_RIGHT"}]
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    per_side = max(1, max_cases // 2)
    for task_type in ("REL2C_LEFT", "REL2C_RIGHT"):
        count = 0
        for row in eligible:
            if row.get("qa_type") != task_type:
                continue
            image_path = str(row.get("image_path"))
            if image_path in used_images:
                continue
            selected.append(row)
            used_images.add(image_path)
            count += 1
            if count >= per_side or len(selected) >= max_cases:
                break
        if len(selected) >= max_cases:
            break
    for row in eligible:
        if len(selected) >= max_cases:
            break
        image_path = str(row.get("image_path"))
        if image_path in used_images:
            continue
        selected.append(row)
        used_images.add(image_path)
    if len(selected) < 2:
        raise RuntimeError("could not select at least two distinct legally resolvable REL2C smoke rows")
    return selected


def _region_from_dict(payload: dict[str, Any] | None) -> RegionPrediction | None:
    if not payload:
        return None
    return RegionPrediction(
        tuple(float(item) for item in payload["bbox_xyxy_pixels"]),
        tuple(float(item) for item in payload["bbox_xyxy_normalized"]),
        None if payload.get("score") is None else float(payload["score"]),
        str(payload.get("label", "")),
        str(payload.get("source", "")),
        payload.get("mask_path"),
    )


def _views_by_entity(trace: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for view in trace.get("views", []):
        entity_id = view.get("source_entity")
        if entity_id:
            grouped.setdefault(entity_id, {})[view["view_id"]] = view
    return grouped


def _acquired_regions(trace: dict[str, Any]) -> dict[str, RegionPrediction]:
    regions: dict[str, RegionPrediction] = {}
    for step in trace.get("acquisition_trace", []):
        if step.get("operation") != "LOCALIZE":
            continue
        entity_id = step.get("entity_id")
        selected = (step.get("outputs") or {}).get("selected_region")
        region = _region_from_dict(selected)
        if entity_id and region is not None:
            regions[entity_id] = region
    return regions


def _provisional_selection(trace: dict[str, Any]) -> dict[str, Any]:
    region_by_ref: dict[str, RegionPrediction] = {}
    for step in trace.get("acquisition_trace", []):
        if step.get("operation") == "LOCALIZE" and step.get("node_id", "").startswith("localize_"):
            ref = step["node_id"].split("_", 1)[1]
            region = _region_from_dict((step.get("outputs") or {}).get("selected_region"))
            if region is not None:
                region_by_ref[ref] = region
    if not {"A", "B"}.issubset(region_by_ref):
        return {"status": "NOT_AVAILABLE", "eligible_for_proof": False}
    ax = sum(region_by_ref["A"].bbox_xyxy_normalized[0::2]) / 2.0
    bx = sum(region_by_ref["B"].bbox_xyxy_normalized[0::2]) / 2.0
    if abs(ax - bx) < 1e-9:
        return {"status": "OVERLAPPING_CENTER", "eligible_for_proof": False, "center_x": {"A": ax, "B": bx}}
    desired = "left" if str(trace.get("task_type", "")).endswith("LEFT") else "right"
    relation = "left" if ax < bx else "right"
    selected_ref = "A" if relation == desired else "B"
    return {
        "status": "PROVISIONAL_DIAGNOSTIC_ONLY",
        "eligible_for_proof": False,
        "reason": "region binding is GROUNDING_ACQUIRED_BUT_NOT_SCORE_CERTIFIABLE",
        "relation": relation,
        "requested_side": desired,
        "logical_ref": selected_ref,
        "center_x": {"A": ax, "B": bx},
    }


def _score_payload(score: Any) -> dict[str, Any]:
    return {
        "processor": score.processor,
        "score_kind": score.score_kind,
        "raw_score": score.raw_score,
        "normalized_score": score.normalized_score,
        "margin": score.margin,
        "support": score.support,
        "normalizer_revision": score.normalizer_revision,
        "score_semantics": score.diagnostics.get("score_semantics"),
        "score_definition": score.diagnostics.get("score_definition"),
        "view_id": score.view_id,
        "diagnostics": score.diagnostics,
    }


def diagnostic_color_probe(
    *,
    trace: dict[str, Any],
    reader: SiglipColorReader,
    views: dict[str, dict[str, dict[str, Any]]],
    regions: dict[str, RegionPrediction],
    selection: dict[str, Any],
) -> dict[str, Any]:
    target_ref = selection.get("logical_ref")
    if target_ref is None:
        return {"status": "BLOCKED_NO_SELECTION", "eligible_for_proof": False}
    target_entity = None
    for step in trace.get("entity_creation_and_reuse", []):
        if step.get("logical_ref") == target_ref:
            target_entity = step.get("entity_id")
            break
    if target_entity is None or target_entity not in regions or target_entity not in views:
        return {"status": "BLOCKED_NO_ENTITY", "eligible_for_proof": False}
    region = regions[target_entity]
    raw_path = Path(views[target_entity]["RAW"]["path"])
    candidate = reader.read(raw_path, region)
    scores: dict[str, Any] = {}
    for view_id in ("RAW", "FOCUS", "REMOVE"):
        score = reader.score_claim(Path(views[target_entity][view_id]["path"]), region, candidate.color, view_id=view_id)
        scores[view_id] = _score_payload(score)
    focus_delta = None
    binding_score = None
    if all(scores[view]["normalized_score"] is not None for view in ("RAW", "FOCUS", "REMOVE")):
        focus_delta = scores["FOCUS"]["normalized_score"] - scores["RAW"]["normalized_score"]
        binding_score = scores["FOCUS"]["normalized_score"] - scores["REMOVE"]["normalized_score"]
    return {
        "status": "SCORED_DIAGNOSTIC_ONLY",
        "eligible_for_proof": False,
        "reason": "region binding is GROUNDING_ACQUIRED_BUT_NOT_SCORE_CERTIFIABLE; scores are audit diagnostics only",
        "entity_id": target_entity,
        "logical_ref": target_ref,
        "candidate_claim": f"COLOR({target_entity})={candidate.color}",
        "candidate_color": candidate.color,
        "candidate_read_confidence": candidate.confidence,
        "processor": reader.name,
        "score_semantics": reader.score_semantics,
        "score_normalization": reader.score_normalization,
        "s_raw": scores["RAW"]["normalized_score"],
        "s_focus": scores["FOCUS"]["normalized_score"],
        "s_remove": scores["REMOVE"]["normalized_score"],
        "focus_delta": focus_delta,
        "binding_score": binding_score,
        "view_scores": scores,
    }


def view_semantic_metrics(
    *,
    image_path: Path,
    focus_path: Path,
    remove_path: Path,
    region: RegionPrediction,
    suppression_box: list[float],
) -> dict[str, Any]:
    raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    focus = np.asarray(Image.open(focus_path).convert("RGB"), dtype=float)
    remove = np.asarray(Image.open(remove_path).convert("RGB"), dtype=float)
    x0, y0, x1, y1 = (int(round(v)) for v in region.bbox_xyxy_pixels)
    inside = np.s_[y0:y1, x0:x1]
    sx0, sy0, sx1, sy1 = (int(round(v)) for v in suppression_box)
    outside_mask = np.ones(raw.shape[:2], dtype=bool)
    outside_mask[max(0, sy0 - 1):sy1 + 2, max(0, sx0 - 1):sx1 + 2] = False
    return {
        "target_bbox_xyxy_pixels": [int(v) for v in (x0, y0, x1, y1)],
        "suppression_box_xyxy_pixels": [int(v) for v in (sx0, sy0, sx1, sy1)],
        "focus_target_mae_rgb": float(np.mean(np.abs(focus[inside] - raw[inside]))),
        "remove_target_mae_rgb": float(np.mean(np.abs(remove[inside] - raw[inside]))),
        "focus_outside_mae_rgb": float(np.mean(np.abs(focus[outside_mask] - raw[outside_mask]))),
        "remove_outside_mae_rgb": float(np.mean(np.abs(remove[outside_mask] - raw[outside_mask]))),
        "focus_remove_pixel_difference_fraction": float(np.mean(np.any(focus != remove, axis=-1))),
    }


def evaluator_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "namespace": "EVALUATOR_ONLY",
        "inference_use": False,
        "ground_truth": row.get("ground_truth"),
        "bbox_A_xywh": row.get("bbox_A"),
        "bbox_B_xywh": row.get("bbox_B"),
        "object_A": row.get("object_A"),
        "object_B": row.get("object_B"),
        "color_A": row.get("color_A"),
        "color_B": row.get("color_B"),
        "left_is_A": row.get("left_is_A"),
    }


def build_certificate_trace(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        for cert in trace.get("binding_certificates", []):
            records.append(
                {
                    "schema_version": "certification_repair_certificate_trace_v1",
                    "case_id": trace["case_id"],
                    "task_type": trace["task_type"],
                    "entity_id": cert["entity_id"],
                    "predicate": cert["predicate"],
                    "binding_kind": cert["binding_kind"],
                    "candidate_value": cert["proposed_value"],
                    "dag_node_id": cert.get("dag_node_id"),
                    "dependency_ids": cert.get("dependency_ids", []),
                    "raw_score": cert["raw_score"],
                    "focus_score": cert["focus_score"],
                    "remove_score": cert["remove_score"],
                    "s_raw": cert["raw_score"].get("normalized_score"),
                    "s_focus": cert["focus_score"].get("normalized_score"),
                    "s_remove": cert["remove_score"].get("normalized_score"),
                    "focus_delta": cert["focus_delta"],
                    "binding_score": cert["necessity_delta"],
                    "status": cert["status"],
                    "rationale": cert["rationale"],
                    "score_semantics": cert.get("diagnostics", {}).get("score_semantics"),
                    "score_processor": cert.get("diagnostics", {}).get("score_processor"),
                    "acquisition_processor": cert.get("diagnostics", {}).get("acquisition_processor"),
                    "failure_code": cert.get("diagnostics", {}).get("failure_code"),
                    "controller_decision": trace.get("controller_decision", {}).get("status"),
                    "reasoner_called": trace.get("reasoner_called"),
                    "qwen_called": trace.get("qwen_called"),
                }
            )
    return records


def _image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _overlay(path: Path, region: RegionPrediction | None, *, gt_xywh: list[float] | None = None, title: str = "") -> str:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if region is not None:
        x0, y0, x1, y1 = region.bbox_xyxy_pixels
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=3)
        draw.text((max(0, x0 + 2), max(0, y0 + 2)), "predicted entity region", fill=(0, 255, 0))
    if gt_xywh is not None:
        x, y, w, h = gt_xywh
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
        draw.text((max(0, x + 2), max(0, y + h + 2)), "EVALUATOR ONLY GT bbox", fill=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_html(traces: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Certification repair smoke audit</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:20px} .case{border:1px solid #bbb;border-radius:8px;padding:12px;margin:18px 0}"
        ".entity{border-top:1px solid #ddd;margin-top:12px;padding-top:8px}.imgs{display:flex;gap:12px;flex-wrap:wrap}"
        ".imgs figure{margin:0}.imgs img{max-width:300px;border:1px solid #999}table{border-collapse:collapse;margin-top:8px}"
        "td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}.gt{color:#b00000;font-weight:bold}.bad{color:#b00000}.ok{color:#0a7a0a}"
        "pre{background:#f6f6f6;padding:8px;overflow:auto}</style></head><body>",
        "<h1>Certification repair smoke audit</h1>",
        "<p><b>GT overlays are evaluator-only and were never passed to inference.</b></p>",
    ]
    for trace in traces:
        row = rows_by_id.get(trace["case_id"], {})
        gt = evaluator_payload(row)
        parts.append(f"<div class='case'><h2>{html.escape(trace['case_id'])}</h2>")
        parts.append(f"<p><b>Task:</b> {html.escape(trace['task_type'])} &nbsp; <b>Question:</b> {html.escape(trace['question']).replace(chr(10),'<br>')}</p>")
        parts.append(f"<p><b>Controller:</b> <span class='bad'>{html.escape(trace['controller_decision']['status'])}</span> "
                     f"&nbsp; proof_complete={trace['controller_decision']['proof_complete']} &nbsp; Qwen called={trace['qwen_called']}</p>")
        regions = _acquired_regions(trace)
        views = _views_by_entity(trace)
        probe = trace.get("diagnostic_color_probe") or {}
        for entity_id in sorted(regions):
            region = regions[entity_id]
            entity_views = views.get(entity_id, {})
            ref = next((r for r in ("A", "B") if row.get(f"bbox_{r}") is not None and entity_id in str(trace.get("selection"))), "?")
            gt_bbox = None
            for ref_letter in ("A", "B"):
                if row.get(f"bbox_{ref_letter}") is not None and f"E_{'0' if ref_letter=='A' else '1'}" == entity_id:
                    gt_bbox = row.get(f"bbox_{ref_letter}")
            parts.append(f"<div class='entity'><h3>{html.escape(entity_id)}</h3>")
            parts.append("<div class='imgs'>")
            parts.append(f"<figure><figcaption>Original + predicted region (green) + <span class='gt'>GT bbox (red, evaluator only)</span></figcaption><img src='{_overlay(Path(trace['image_path']), region, gt_xywh=gt_bbox)}'></figure>")
            if "FOCUS" in entity_views:
                parts.append(f"<figure><figcaption>FOCUS: target preserved, background defocused</figcaption><img src='{_overlay(Path(entity_views['FOCUS']['path']), region)}'></figure>")
            if "REMOVE" in entity_views:
                parts.append(f"<figure><figcaption>REMOVE: target region neutral-filled, outside preserved</figcaption><img src='{_overlay(Path(entity_views['REMOVE']['path']), region)}'></figure>")
            parts.append("</div>")
            certs = [c for c in trace.get("binding_certificates", []) if c["entity_id"] == entity_id]
            parts.append("<table><tr><th>certificate</th><th>s_raw</th><th>s_focus</th><th>s_remove</th><th>binding score</th><th>status</th><th>failure code</th></tr>")
            for cert in certs:
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(cert['predicate'])} {html.escape(str(cert['dag_node_id']))}</td>"
                    f"<td>{cert['raw_score'].get('normalized_score')}</td>"
                    f"<td>{cert['focus_score'].get('normalized_score')}</td>"
                    f"<td>{cert['remove_score'].get('normalized_score')}</td>"
                    f"<td>{cert['necessity_delta']}</td>"
                    f"<td class='bad'>{html.escape(cert['status'])}</td>"
                    f"<td>{html.escape(str(cert.get('diagnostics', {}).get('failure_code')))}</td>"
                    "</tr>"
                )
            parts.append("</table>")
            if probe.get("entity_id") == entity_id:
                parts.append(
                    f"<p><b>Diagnostic-only colour claim:</b> {html.escape(str(probe.get('candidate_claim')))} "
                    f"score={probe.get('s_focus')} focus / {probe.get('s_remove')} remove / binding={probe.get('binding_score')} "
                    f"<i>({html.escape(str(probe.get('reason')))}; not in proof state)</i></p>"
                )
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu", help="SigLIP-2 device")
    parser.add_argument("--florence-device", default="cpu")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if not str(output_root).startswith("/mnt/d/"):
        raise AssertionError("certification repair outputs must be on the D drive")
    paths = Paths()
    ensure_d_drive_environment(paths)
    rows = select_smoke_rows(read_jsonl(paths.qa), args.max_cases)
    registry = build_registry(paths, device=args.device, florence_device=args.florence_device)
    reader = registry.require("siglip2_color", "color").implementation.reader
    reasoner = DeterministicProofStateReasoner()
    view_root = output_root / "smoke_views"
    if view_root.exists():
        import shutil
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        trace = run_refactored_case(
            RefactoredCaseInput(
                case_id=str(row["qa_id"]),
                task_type=str(row["qa_type"]),
                image_path=Path(str(row["image_path"])),
                question=str(row["question"]),
            ),
            registry=registry,
            reasoner=reasoner,
            output_root=view_root,
            certification_config=CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"),
            acquisition_processor="florence2_query_grounding",
            grounding_score_processor=None,
            color_processor="siglip2_color",
        )
        regions = _acquired_regions(trace)
        views = _views_by_entity(trace)
        selection = _provisional_selection(trace)
        trace["diagnostic_selection"] = selection
        trace["diagnostic_color_probe"] = diagnostic_color_probe(trace=trace, reader=reader, views=views, regions=regions, selection=selection)
        metrics: dict[str, Any] = {}
        for entity_id, region in regions.items():
            entity_views = views.get(entity_id, {})
            if {"FOCUS", "REMOVE"}.issubset(entity_views):
                suppression_box = entity_views["REMOVE"].get("parameters", {}).get("suppression_box_xyxy") or region.bbox_xyxy_pixels
                metrics[entity_id] = view_semantic_metrics(
                    image_path=Path(str(row["image_path"])),
                    focus_path=Path(entity_views["FOCUS"]["path"]),
                    remove_path=Path(entity_views["REMOVE"]["path"]),
                    region=region,
                    suppression_box=list(suppression_box),
                )
        trace["view_semantic_metrics"] = metrics
        trace["evaluator_only_after_inference"] = evaluator_payload(row)
        trace["gt_answer_EVALUATOR_ONLY"] = {"value": row.get("ground_truth"), "namespace": "EVALUATOR_ONLY"}
        traces.append(trace)
        print(json.dumps({"case_id": trace["case_id"], "controller": trace["controller_decision"]["status"], "qwen_called": trace["qwen_called"]}, ensure_ascii=False))

    smoke_path = output_root / "smoke_trace.jsonl"
    certificate_path = output_root / "certificate_trace.jsonl"
    smoke_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in traces), encoding="utf-8")
    certificate_records = build_certificate_trace(traces)
    certificate_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in certificate_records), encoding="utf-8")
    diagnostic_records = []
    for trace in traces:
        probe = trace.get("diagnostic_color_probe") or {}
        if probe.get("status") != "SCORED_DIAGNOSTIC_ONLY":
            continue
        diagnostic_records.append(
            {
                "record_type": "DIAGNOSTIC_COLOR_SCORE_NOT_A_CERTIFICATE",
                "eligible_for_proof": False,
                "case_id": trace["case_id"],
                "task_type": trace["task_type"],
                "entity_id": probe["entity_id"],
                "logical_ref": probe["logical_ref"],
                "candidate_claim": probe["candidate_claim"],
                "candidate_color": probe["candidate_color"],
                "processor": probe["processor"],
                "score_semantics": probe["score_semantics"],
                "score_normalization": probe["score_normalization"],
                "s_raw": probe["s_raw"],
                "s_focus": probe["s_focus"],
                "s_remove": probe["s_remove"],
                "focus_delta": probe["focus_delta"],
                "binding_score": probe["binding_score"],
                "blocker": "GROUNDING_ACQUIRED_BUT_NOT_SCORE_CERTIFIABLE",
                "controller_decision": trace["controller_decision"]["status"],
                "qwen_called": trace["qwen_called"],
            }
        )
    diagnostic_path = output_root / "diagnostic_evidence_trace.jsonl"
    diagnostic_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in diagnostic_records), encoding="utf-8")
    html_path = output_root / "smoke_audit.html"
    rows_by_id = {str(row["qa_id"]): row for row in rows}
    html_path.write_text(build_html(traces, rows_by_id), encoding="utf-8")
    print(json.dumps({"smoke_trace": str(smoke_path), "certificate_trace": str(certificate_path), "diagnostic_evidence_trace": str(diagnostic_path), "html": str(html_path), "cases": len(traces), "full_benchmark_launched": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
