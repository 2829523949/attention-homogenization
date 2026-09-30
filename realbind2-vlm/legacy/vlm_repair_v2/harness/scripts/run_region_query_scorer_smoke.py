#!/usr/bin/env python3
"""Tiny smoke for the separate SigLIP2 fixed-region query scorer.

Florence acquires persistent regions.  SigLIP2 independently scores the
fixed region against the fixed legal query on RAW / FOCUS / REMOVE.  The
semantic evidence layer does NOT satisfy the full logical identity binding,
so the strict controller still abstains and Qwen is not called.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from harness.certification.certifier import CertificationConfig
from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.dataset import read_jsonl
from harness.processors.crop_policy import CROP_POLICY_ID, save_persistent_crop
from harness.processors.florence_subprocess import FlorenceSubprocessGrounding
from harness.processors.registry import ColorReaderAdapter, ProcessorCapability, ProcessorRegistry
from harness.processors.siglip_backend import Siglip2Backend
from harness.processors.color_siglip import SiglipColorReader
from harness.processors.siglip_region_query import Siglip2RegionQueryScorer
from harness.reasoners.hard_handoff_qwen import DeterministicProofStateReasoner
from harness.refactored_pipeline import RefactoredCaseInput, run_refactored_case
from harness.scripts.run_certification_repair_smoke import (
    _acquired_regions,
    _image_data_uri,
    _overlay,
    _region_from_dict,
    _views_by_entity,
    evaluator_payload,
    select_smoke_rows,
)

OUTPUT_ROOT = Path("/mnt/d/vlm_repair_v2/outputs/region_query_scorer")


def build_registry(paths: Paths, *, device: str, florence_device: str) -> ProcessorRegistry:
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
            notes="FLORENCE ACQUISITION ONLY. No referent-region certification score.",
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
            notes=(
                "SEPARATE SigLIP2 scoring stage for fixed persistent region + fixed legal query; "
                "not Florence confidence, not detector confidence, not full logical identity."
            ),
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
            notes="Fixed-candidate colour scoring; not reached in this smoke because full identity is unresolved.",
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


def _crop_artifact_name(view_id: str) -> str:
    return f"{view_id.lower()}__crop.png"


def enrich_evidence_records_with_crops(trace: dict[str, Any], case_root: Path) -> list[dict[str, Any]]:
    regions = _acquired_regions(trace)
    views = _views_by_entity(trace)
    records = trace.get("region_query_evidence", [])
    for record in records:
        entity_id = record["entity_id"]
        region = regions.get(entity_id)
        if region is None:
            continue
        for view_id, field in (("RAW", "raw_crop_path"), ("FOCUS", "focus_crop_path"), ("REMOVE", "remove_crop_path")):
            entity_views = views.get(entity_id, {})
            view_payload = entity_views.get(view_id)
            if view_payload is None:
                continue
            crop_path = case_root / f"{entity_id}__{view_id.lower()}__crop.png"
            artifact = save_persistent_crop(Path(view_payload["path"]), region, crop_path)
            record[field] = artifact["path"]
            record.setdefault("crop_artifacts", {})[view_id] = artifact
            record["crop_policy_id"] = artifact["crop_policy_id"]
    return records


def evidence_trace_records(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        for record in trace.get("region_query_evidence", []):
            records.append(
                {
                    "schema_version": "fixed_region_query_evidence_trace_v1",
                    "case_id": trace["case_id"],
                    "task_type": trace["task_type"],
                    "logical_ref": record["logical_ref"],
                    "entity_id": record["entity_id"],
                    "referent_spec": record["referent_spec"],
                    "referent_provenance": record["referent_provenance"],
                    "acquisition_processor": record["acquisition_processor"],
                    "acquisition_output": record["acquisition_output"],
                    "persistent_bbox_xyxy_pixels": record["persistent_bbox_xyxy_pixels"],
                    "region_id": record["region_id"],
                    "instance_uniqueness_status": record["instance_uniqueness_status"],
                    "candidate_count": record["candidate_count"],
                    "query_text": record["query_text"],
                    "query_template_id": record["query_template_id"],
                    "query_hash": record["query_hash"],
                    "query_provenance": record["query_provenance"],
                    "crop_policy_id": record["crop_policy_id"],
                    "score_processor": record["score_processor"],
                    "model_revision": record["model_revision"],
                    "model_fingerprint": record["model_fingerprint"],
                    "raw_crop_path": record["raw_crop_path"],
                    "focus_crop_path": record["focus_crop_path"],
                    "remove_crop_path": record["remove_crop_path"],
                    "s_raw_raw": record["s_raw_raw"],
                    "s_focus_raw": record["s_focus_raw"],
                    "s_remove_raw": record["s_remove_raw"],
                    "s_raw": record["s_raw"],
                    "s_focus": record["s_focus"],
                    "s_remove": record["s_remove"],
                    "focus_delta": record["focus_delta"],
                    "binding_evidence": record["binding_evidence"],
                    "evidence_kind": record["evidence_kind"],
                    "identity_layer": record["identity_layer"],
                    "normalization_semantics": record["normalization_semantics"],
                    "calibration_status": record["calibration_status"],
                    "evidence_status": record["evidence_status"],
                    "failure_code": record["failure_code"],
                    "diagnostic_pattern_pass": record.get("diagnostic_pattern_pass"),
                    "diagnostic_pattern_note": record.get("diagnostic_pattern_note"),
                    "full_logical_identity_status": record["full_logical_identity_status"],
                    "controller_decision": record["controller_decision"],
                    "reasoner_called": record["reasoner_called"],
                    "qwen_called": record["qwen_called"],
                }
            )
    return records


def build_html(traces: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Fixed-region query scorer audit</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:20px}.case{border:1px solid #bbb;border-radius:8px;padding:12px;margin:18px 0}"
        ".entity{border-top:1px solid #ddd;margin-top:12px;padding-top:8px}.imgs{display:flex;gap:10px;flex-wrap:wrap}"
        ".imgs figure{margin:0}.imgs img{max-width:240px;border:1px solid #999}table{border-collapse:collapse;margin-top:8px}"
        "td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}.stage{font-weight:bold}.gt{color:#b00000}.bad{color:#b00000}.ok{color:#0a7a0a}"
        "pre{background:#f6f6f6;padding:8px;overflow:auto}</style></head><body>",
        "<h1>Separate fixed-region query scorer audit</h1>",
        "<p><b>Florence acquisition</b> and <b>SigLIP2 semantic scoring</b> are distinct stages. GT overlays are evaluator-only and were never passed to inference.</p>",
    ]
    for trace in traces:
        row = rows_by_id.get(trace["case_id"], {})
        regions = _acquired_regions(trace)
        views = _views_by_entity(trace)
        evidence_by_entity = {record["entity_id"]: record for record in trace.get("region_query_evidence", [])}
        parts.append(f"<div class='case'><h2>{trace['case_id']}</h2>")
        parts.append(f"<p><b>Question:</b> {trace['question'].replace(chr(10),'<br>')}</p>")
        parts.append(
            f"<p><span class='stage'>Florence acquisition:</span> persistent region from legal noun query. "
            f"<span class='stage'>SigLIP2 scoring:</span> fixed region + fixed legal query, identical crop policy across views. "
            f"<b>Controller:</b> <span class='bad'>{trace['controller_decision']['status']}</span>; "
            f"Qwen called={trace['qwen_called']}.</p>"
        )
        for entity_id in sorted(regions):
            region = regions[entity_id]
            record = evidence_by_entity.get(entity_id, {})
            parts.append(f"<div class='entity'><h3>{entity_id}</h3>")
            legal_query = record.get("query_text", "(no semantic evidence record)")
            parts.append(f"<p><b>Legal query:</b> <code>{legal_query}</code> &nbsp; <b>template:</b> {record.get('query_template_id')} &nbsp; <b>query hash:</b> {record.get('query_hash')}</p>")
            parts.append(f"<p><b>Persistent bbox:</b> {record.get('persistent_bbox_xyxy_pixels')} &nbsp; <b>crop policy:</b> {record.get('crop_policy_id')} &nbsp; <b>instance uniqueness:</b> {record.get('instance_uniqueness_status')}</p>")
            parts.append("<div class='imgs'>")
            gt_bbox = None
            for ref in ("A", "B"):
                if row.get(f"bbox_{ref}") is not None and f"E_{'0' if ref == 'A' else '1'}" == entity_id:
                    gt_bbox = row.get(f"bbox_{ref}")
            parts.append(f"<figure><figcaption>Original + persistent bbox (green) + <span class='gt'>GT (red, evaluator only)</span></figcaption><img src='{_overlay(Path(trace['image_path']), region, gt_xywh=gt_bbox)}'></figure>")
            for view_id in ("RAW", "FOCUS", "REMOVE"):
                view_payload = views.get(entity_id, {}).get(view_id)
                if view_payload:
                    parts.append(f"<figure><figcaption>{view_id} view</figcaption><img src='{_overlay(Path(view_payload['path']), region)}'></figure>")
            for field in ("raw_crop_path", "focus_crop_path", "remove_crop_path"):
                crop_path = record.get(field)
                if crop_path and Path(crop_path).is_file():
                    parts.append(f"<figure><figcaption>{field.replace('_path','').upper()} exact crop</figcaption><img src='{_image_data_uri(Path(crop_path))}'></figure>")
            parts.append("</div>")
            parts.append("<table><tr><th>s_raw_raw</th><th>s_focus_raw</th><th>s_remove_raw</th><th>s_raw</th><th>s_focus</th><th>s_remove</th><th>focus_delta</th><th>binding_evidence</th><th>evidence status</th></tr>")
            parts.append(
                f"<tr><td>{record.get('s_raw_raw')}</td><td>{record.get('s_focus_raw')}</td><td>{record.get('s_remove_raw')}</td>"
                f"<td>{record.get('s_raw')}</td><td>{record.get('s_focus')}</td><td>{record.get('s_remove')}</td>"
                f"<td>{record.get('focus_delta')}</td><td>{record.get('binding_evidence')}</td><td class='bad'>{record.get('evidence_status')}</td></tr>"
            )
            parts.append("</table>")
            parts.append(
                f"<p><b>Acquisition processor:</b> {record.get('acquisition_processor')} &nbsp; "
                f"<b>Score processor:</b> {record.get('score_processor')} &nbsp; "
                f"<b>Model revision:</b> {record.get('model_revision')} &nbsp; "
                f"<b>Evidence kind:</b> {record.get('evidence_kind')} &nbsp; "
                f"<b>Calibration:</b> {record.get('calibration_status')} &nbsp; "
                f"<b>Full logical identity:</b> {record.get('full_logical_identity_status')}</p>"
            )
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu", help="SigLIP2 device")
    parser.add_argument("--florence-device", default="cpu")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if not str(output_root).startswith("/mnt/d/"):
        raise AssertionError("region-query scorer outputs must be on the D drive")
    paths = Paths()
    ensure_d_drive_environment(paths)
    rows = select_smoke_rows(read_jsonl(paths.qa), args.max_cases)
    registry = build_registry(paths, device=args.device, florence_device=args.florence_device)
    reasoner = DeterministicProofStateReasoner()

    if (output_root / "smoke_views").exists():
        shutil.rmtree(output_root / "smoke_views")
    if (output_root / "smoke_crops").exists():
        shutil.rmtree(output_root / "smoke_crops")
    view_root = output_root / "smoke_views"
    crop_root = output_root / "smoke_crops"
    view_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    for row in rows:
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
            grounding_score_processor="siglip2_region_query",
            color_processor="siglip2_color",
        )
        enrich_evidence_records_with_crops(trace, crop_root / str(row["qa_id"]))
        trace["evaluator_only_after_inference"] = evaluator_payload(row)
        trace["gt_answer_EVALUATOR_ONLY"] = {"value": row.get("ground_truth"), "namespace": "EVALUATOR_ONLY"}
        traces.append(trace)
        print(json.dumps({"case_id": trace["case_id"], "controller": trace["controller_decision"]["status"], "qwen_called": trace["qwen_called"], "evidence_records": len(trace.get("region_query_evidence", []))}, ensure_ascii=False))

    smoke_path = output_root / "smoke_trace.jsonl"
    evidence_path = output_root / "evidence_trace.jsonl"
    smoke_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in traces), encoding="utf-8")
    evidence_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in evidence_trace_records(traces)), encoding="utf-8")
    rows_by_id = {str(row["qa_id"]): row for row in rows}
    html_path = output_root / "smoke_audit.html"
    html_path.write_text(build_html(traces, rows_by_id), encoding="utf-8")
    print(json.dumps({"smoke_trace": str(smoke_path), "evidence_trace": str(evidence_path), "html": str(html_path), "cases": len(traces), "full_benchmark_launched": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
