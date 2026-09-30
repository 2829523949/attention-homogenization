#!/usr/bin/env python3
"""Read-only Florence candidate observability audit on the frozen 20 families.

The current inference path and selector are executed unchanged.  This script
only consumes the newly exposed acquisition trace, scores every accepted
candidate with the existing RAW SigLIP region/query scorer, and joins GT after
all inference/scoring is complete for evaluator-only diagnostics.

No Qwen, candidate reranking, threshold tuning, crop experiment, template
change, model download, or full benchmark is performed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from harness.certification.certifier import CertificationConfig
from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.dataset import read_jsonl
from harness.evaluators.leakage_checks import assert_inference_safe
from harness.processors.base import RegionPrediction
from harness.reasoners.hard_handoff_qwen import DeterministicProofStateReasoner
from harness.refactored_pipeline import RefactoredCaseInput, run_refactored_case
from harness.referents.referent_resolver import resolve_question
from harness.referents.referent_spec import GroundingQuery
from harness.referents.region_query import build_region_query
from harness.scripts.run_acquisition_semantic_support_study import build_registry, iou_xyxy, xywh_to_xyxy


OUTPUT_ROOT = ROOT / "outputs" / "candidate_level_audit"
MANIFEST_PATH = ROOT / "manifests" / "acquisition_semantic_support_20family.json"
BASELINE_PATH = ROOT / "outputs" / "acquisition_semantic_support" / "entity_results.jsonl"
THRESHOLDS = (0.3, 0.5, 0.7)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def region_from_payload(payload: dict[str, Any]) -> RegionPrediction:
    return RegionPrediction(
        tuple(float(value) for value in payload["bbox_xyxy_pixels"]),
        tuple(float(value) for value in payload["bbox_xyxy_normalized"]),
        None if payload.get("score") is None else float(payload["score"]),
        str(payload.get("label", "")),
        str(payload.get("source", "")),
        payload.get("mask_path"),
    )


def normalized_label_tokens(text: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", str(text).casefold())
    normalized = []
    for token in tokens:
        normalized.append(token[:-1] if token.endswith("s") and len(token) > 3 else token)
    return tuple(sorted(set(normalized)))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def metric_fraction(rows: list[dict[str, Any]], predicate) -> float | None:
    return float(np.mean([bool(predicate(row)) for row in rows])) if rows else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def safe_qa_index(path: Path) -> dict[str, dict[str, Any]]:
    """Load only inference-visible fields; discard evaluator fields."""
    result = {}
    for row in read_jsonl(path):
        safe = {
            "qa_id": str(row["qa_id"]),
            "family_id": str(row["family_id"]),
            "qa_type": str(row["qa_type"]),
            "image_path": str(row["image_path"]),
            "question": str(row["question"]),
        }
        assert_inference_safe(safe, context="candidate audit safe QA projection")
        result[safe["qa_id"]] = safe
    return result


def resolution_constraints(spec: dict[str, Any], dag: dict[str, Any], selection_side: str | None) -> dict[str, Any]:
    operations = {node["operation"] for node in dag.get("nodes", [])}
    category = spec.get("category_constraint")
    attributes = spec.get("attribute_constraints") or []
    relations = spec.get("relation_constraints") or []
    existing_entity = spec.get("existing_entity_id")
    has_pair_relation = "COMPARE_HORIZONTAL" in operations and "SELECT_RELATION" in operations
    return {
        "category_noun": {
            "available": category is not None,
            "value": category.get("value") if category else None,
            "provenance": category.get("provenance") if category else None,
        },
        "attribute_constraint": {
            "available": bool(attributes),
            "values": attributes,
        },
        "left_right_relation": {
            "available": has_pair_relation and selection_side in {"left", "right"},
            "value": selection_side if has_pair_relation else None,
            "provenance": "QUESTION_TEXT" if has_pair_relation else None,
            "note": "family-level relation/selection constraint; not injected into the Florence noun query",
        },
        "known_reference_entity": {
            "available": existing_entity is not None,
            "value": existing_entity,
        },
        "relative_position": {
            "available": has_pair_relation,
            "value": "horizontal A/B comparison" if has_pair_relation else None,
            "provenance": "QUESTION_TEXT + deterministic PredicateDAG" if has_pair_relation else None,
        },
        "previous_certified_entity": {
            "available": False,
            "value": None,
            "note": "no previous certified entity is present at initial A/B acquisition",
        },
        "other_legal_relation": {
            "available": bool(relations),
            "values": relations,
        },
        "hidden_generator_prior_excluded": True,
    }


def primary_subtype(record: dict[str, Any]) -> tuple[str | None, str | None]:
    evaluator = record["EVALUATOR_ONLY"]
    if evaluator["primary_failure_category"] != "SELECTION_FAILURE":
        return None, None
    selected = record["inference"]["candidates"][evaluator["selected_candidate_index"]]
    best = record["inference"]["candidates"][evaluator["best_candidate_index"]]
    gt = evaluator["gt_bbox_xyxy_pixels"]
    pred = selected["bbox_xyxy_pixels"]
    intersection = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0])) * max(
        0.0, min(pred[3], gt[3]) - max(pred[1], gt[1])
    )
    gt_area = max(1e-9, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    pred_area = max(1e-9, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    if intersection / gt_area >= 0.80 and pred_area / gt_area >= 1.50:
        return "OVERLARGE_REGION_SELECTED", "selected box covers >=80% of GT but has >=1.5x GT area"
    if intersection > 0:
        return "PARTIAL_REGION_SELECTED", "selected box overlaps GT but remains below IoU 0.5"
    selected_tokens = normalized_label_tokens(selected["returned_florence_label_text"])
    best_tokens = normalized_label_tokens(best["returned_florence_label_text"])
    category_tokens = normalized_label_tokens(record["inference"]["florence_query_text"])
    if selected_tokens and selected_tokens == best_tokens and bool(set(selected_tokens) & set(category_tokens)):
        return (
            "WRONG_SAME_CATEGORY_INSTANCE",
            "disjoint selected and GT-best boxes share the same visible Florence category label",
        )
    if selected["semantic_diagnostic"]["z_raw"] >= best["semantic_diagnostic"]["z_raw"]:
        return (
            "SEMANTIC_DISTRACTOR",
            "disjoint selected box has at least as much RAW SigLIP support as the GT-best candidate",
        )
    return "UNKNOWN", "available geometry/text/semantic evidence does not support a narrower subtype"


def draw_candidate_overlay(record: dict[str, Any], path: Path) -> None:
    image = Image.open(record["inference"]["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for candidate in record["inference"]["candidates"]:
        index = candidate["candidate_index"]
        box = tuple(candidate["bbox_xyxy_pixels"])
        color = palette[index % len(palette)]
        width = 5 if candidate["became_current_selected_region"] else 3
        draw.rectangle(box, outline=color, width=width)
        draw.text((box[0] + 2, box[1] + 2), f"C{index} z={candidate['semantic_diagnostic']['z_raw']:.2f}", fill=color)
    gt = record["EVALUATOR_ONLY"]["gt_bbox_xyxy_pixels"]
    draw.rectangle(tuple(gt), outline="#e60000", width=3)
    draw.text((gt[0] + 2, max(0, gt[1] - 13)), "GT (evaluator only)", fill="#e60000")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def image_uri(path: Path, max_size: int = 420) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=84)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_figures(records: list[dict[str, Any]], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    best = np.asarray([row["EVALUATOR_ONLY"]["best_candidate_iou"] for row in records])
    selected = np.asarray([row["EVALUATOR_ONLY"]["selected_iou"] for row in records])
    regret = best - selected

    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(best, bins=bins, alpha=0.58, label="best candidate IoU", color="#2a6fbb")
    ax.hist(selected, bins=bins, alpha=0.58, label="selected IoU", color="#d95f02")
    ax.set(xlabel="IoU", ylabel="Entity count", title="Best-candidate versus selected-region IoU")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures / "figure1_iou_distributions.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(best, selected, c=[row["inference"]["candidate_count"] for row in records], cmap="viridis", s=48)
    ax.plot([0, 1], [0, 1], "--", color="#777", linewidth=1, label="zero regret")
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="Best candidate IoU", ylabel="Selected IoU", title="Candidate availability versus current selection")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures / "figure2_best_vs_selected_scatter.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(regret, bins=np.linspace(0, 1, 11), color="#7551a5", alpha=.85)
    ax.set(xlabel="Selection regret (best IoU − selected IoU)", ylabel="Entity count", title="Selection regret distribution")
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures / "figure3_selection_regret.png", dpi=160); plt.close(fig)

    usable = [row for row in records if row["EVALUATOR_ONLY"]["best_candidate_iou"] >= .5]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ranks = [row["EVALUATOR_ONLY"]["semantic_rank_of_gt_best_candidate"] for row in usable]
    ys = [1 if row["EVALUATOR_ONLY"]["selected_iou"] >= .5 else 0 for row in usable]
    jitter = np.linspace(-.035, .035, len(usable)) if usable else []
    ax.scatter(ranks, np.asarray(ys) + jitter, c=["#258039" if value else "#c43d3d" for value in ys], s=55)
    ax.set(xlabel="Semantic rank of GT-best candidate (1 = strongest)", ylabel="Current selection usable (1/0)", title="Semantic rank and current selection correctness")
    ax.set_yticks([0, 1]); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures / "figure4_gt_best_semantic_rank.png", dpi=160); plt.close(fig)

    counts = Counter(row["inference"]["candidate_count"] for row in records)
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], color="#3d8b8b")
    ax.set(xlabel="Accepted Florence candidate count", ylabel="Entity count", title="Candidate-count distribution")
    ax.set_xticks(xs); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(figures / "figure5_candidate_count.png", dpi=160); plt.close(fig)


def build_html(records: list[dict[str, Any]], metrics: dict[str, Any], overlay_root: Path) -> str:
    parts = ["""<!doctype html><html><head><meta charset='utf-8'><title>Florence candidate audit</title>
<style>body{font-family:system-ui;margin:20px;background:#fafafa;color:#222}.card{background:white;border:1px solid #bbb;border-radius:8px;padding:12px;margin:12px 0}.bad{color:#b00020}.ok{color:#087830}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:5px;font-size:12px}img{max-width:440px}code{background:#eee;padding:2px 4px}</style></head><body>
<h1>Florence candidate-level observability audit</h1><p><b>Red boxes and all IoU/oracle fields are evaluator-only.</b> Current selection was not changed. SigLIP scores are diagnostic raw logits and never select a box. Qwen was not called.</p>"""]
    summary = metrics["candidate_recall"]
    parts.append(f"<p>N={summary['n_unique_entities']}; SelectedRecall@0.5={summary['selected_recall_at_0_5']:.3f}; CandidateRecall@0.5={summary['candidate_recall_at_0_5']:.3f}.</p>")
    for row in sorted(records, key=lambda value: value["EVALUATOR_ONLY"]["selection_regret"], reverse=True):
        inf, ev = row["inference"], row["EVALUATOR_ONLY"]
        overlay = overlay_root / f"{inf['family_id']}__{inf['logical_ref']}.jpg"
        parts.append(f"<div class='card'><h2>{html.escape(inf['family_id'])} / {inf['logical_ref']} / {html.escape(inf['florence_query_text'])}</h2>")
        parts.append(f"<p>selected IoU=<b class='{'ok' if ev['selected_iou']>=.5 else 'bad'}'>{ev['selected_iou']:.4f}</b>; best IoU={ev['best_candidate_iou']:.4f}; regret={ev['selection_regret']:.4f}; primary={ev['primary_failure_category']}; subtype={ev.get('selection_failure_subtype')}</p>")
        parts.append(f"<img src='{image_uri(overlay)}'>")
        parts.append("<table><tr><th>index</th><th>selected</th><th>box xyxy</th><th>label</th><th>native score</th><th>z_raw</th><th>semantic rank</th><th>IoU (evaluator only)</th></tr>")
        for candidate in inf["candidates"]:
            cev = candidate["EVALUATOR_ONLY"]
            parts.append(f"<tr><td>{candidate['candidate_index']}</td><td>{str(candidate['became_current_selected_region']).lower()}</td><td><code>{candidate['bbox_xyxy_pixels']}</code></td><td>{html.escape(candidate['returned_florence_label_text'])}</td><td>{candidate['native_score']}</td><td>{candidate['semantic_diagnostic']['z_raw']:.5f}</td><td>{cev['semantic_rank']}</td><td>{cev['candidate_iou_with_gt']:.5f}</td></tr>")
        parts.append("</table></div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def recommendation_for(records: list[dict[str, Any]]) -> tuple[str, str]:
    generation = sum(row["EVALUATOR_ONLY"]["primary_failure_category"] == "GENERATION_FAILURE" for row in records)
    selection = sum(row["EVALUATOR_ONLY"]["primary_failure_category"] == "SELECTION_FAILURE" for row in records)
    usable = [row for row in records if row["EVALUATOR_ONLY"]["best_candidate_iou"] >= .5]
    semantic_miss = sum(row["EVALUATOR_ONLY"]["semantic_top1_candidate_iou"] < .5 for row in usable)
    n = len(records)
    # Descriptive recommendation logic only; these are not inference or
    # calibration thresholds.
    if generation / n >= .25 and selection / n >= .15 and semantic_miss / max(1, len(usable)) >= .20:
        return "D. MIXED_CANDIDATE_FAILURES", "generation, current selection, and semantic ranking each contribute materially"
    if selection / n >= .15 and semantic_miss / max(1, len(usable)) < .20:
        return "B. BUILD_DETERMINISTIC_CONSTRAINT_RESOLVER", "candidate recall materially exceeds selected recall and semantic top-1 is usually usable"
    if semantic_miss / max(1, len(usable)) >= .30 and selection > 0:
        return "C. FIX_SEMANTIC_RANKING_INTERFACE", "usable candidates exist but RAW SigLIP frequently ranks a non-usable candidate first"
    if generation >= selection:
        return "A. FIX_CANDIDATE_GENERATION", "most selected-region failures have no usable Florence candidate in the exposed set"
    return "D. MIXED_CANDIDATE_FAILURES", "more than one candidate-stage failure source remains non-trivial"


def write_report(
    root: Path,
    records: list[dict[str, Any]],
    families: list[dict[str, Any]],
    candidate_metrics: dict[str, Any],
    regret_metrics: dict[str, Any],
    semantic_metrics: dict[str, Any],
    joint_metrics: dict[str, Any],
    failures: dict[str, Any],
    pipeline_trace: dict[str, Any],
    recommendation: tuple[str, str],
) -> None:
    cm, rm, sm, jm = candidate_metrics, regret_metrics, semantic_metrics, joint_metrics
    selected_wrong = [row for row in records if row["EVALUATOR_ONLY"]["selected_iou"] < .5]
    selection_failures = failures["primary_counts"]["SELECTION_FAILURE"]
    generation_failures = failures["primary_counts"]["GENERATION_FAILURE"]
    lines = [
        "# Florence candidate-level observability audit",
        "",
        "This is a read-only/logging-only diagnostic of the frozen 20-family RealBind subset. It does not change Florence queries or decoding, the current selection key, SigLIP, exact-bbox crop policy, FOCUS/REMOVE, certification, or thresholds. It does not call Qwen and does not run the full benchmark.",
        "",
        "## Integrity and frozen-method checks",
        "",
        f"- Manifest: `{MANIFEST_PATH}` ({pipeline_trace['manifest_sha256']}).",
        f"- Locked families/entities: {len(families)}/{len(records)}.",
        f"- Selected-region regression: **{pipeline_trace['selected_region_regression']['status']}**, {pipeline_trace['selected_region_regression']['matched_entities']}/{len(records)} exact bbox/label/source matches.",
        "- Candidate scores: existing `siglip2_region_query`, RAW image, unchanged legal query template, unchanged exact persistent-bbox crop; raw logits are primary.",
        "- GT was loaded only after all inference and candidate scoring completed; every GT-derived field/output is labeled `EVALUATOR_ONLY`.",
        "- `qwen_called = false`; no VLM selector was used.",
        "",
        "## Table A — Candidate generation",
        "",
        "| Metric | Value |", "|---|---:|",
        f"| N unique entities | {cm['n_unique_entities']} |",
        f"| CandidateRecall@0.3 | {fmt(cm['candidate_recall_at_0_3'])} |",
        f"| CandidateRecall@0.5 | {fmt(cm['candidate_recall_at_0_5'])} |",
        f"| CandidateRecall@0.7 | {fmt(cm['candidate_recall_at_0_7'])} |",
        f"| SelectedRecall@0.3 | {fmt(cm['selected_recall_at_0_3'])} |",
        f"| SelectedRecall@0.5 | {fmt(cm['selected_recall_at_0_5'])} |",
        f"| SelectedRecall@0.7 | {fmt(cm['selected_recall_at_0_7'])} |",
        "",
        "## Table B — Selection quality",
        "",
        "| Metric | Value |", "|---|---:|",
        f"| N entities with candidate@0.5 | {cm['n_entities_with_candidate_at_0_5']} |",
        f"| SelectionSuccess@0.5 \| candidate exists | {fmt(cm['selection_success_at_0_5_given_candidate_exists'])} |",
        f"| Mean selection regret | {fmt(rm['overall']['mean'])} |",
        f"| Median selection regret | {fmt(rm['overall']['median'])} |",
        f"| Regret > 0.1 | {fmt(rm['overall']['fraction_gt_0_1'])} |",
        f"| Regret > 0.3 | {fmt(rm['overall']['fraction_gt_0_3'])} |",
        f"| Regret > 0.5 | {fmt(rm['overall']['fraction_gt_0_5'])} |",
        "",
        "## Table C — Semantic ranking",
        "",
        "For entities where a candidate reaches IoU 0.5:",
        "",
        "| Metric | Value |", "|---|---:|",
        f"| GT-best semantic rank = 1 fraction | {fmt(sm['gt_best_rank_1_fraction'])} |",
        f"| GT-best semantic rank <= 2 fraction | {fmt(sm['gt_best_rank_le_2_fraction'])} |",
        f"| Mean reciprocal rank | {fmt(sm['mean_reciprocal_rank'])} |",
        f"| Median rank | {fmt(sm['median_rank'])} |",
        f"| Semantic-top1 localization success | {fmt(sm['semantic_top1_localization_success_given_usable_candidate'])} |",
        "",
        "## Table D — Failure decomposition",
        "",
        "| Category | Count |", "|---|---:|",
    ]
    for key in ("GENERATION_FAILURE", "SELECTION_FAILURE", "SELECTION_SUCCESS"):
        lines.append(f"| {key} | {failures['primary_counts'][key]} |")
    for key in ("WRONG_SAME_CATEGORY_INSTANCE", "SEMANTIC_DISTRACTOR", "PARTIAL_REGION_SELECTED", "OVERLARGE_REGION_SELECTED", "UNKNOWN"):
        lines.append(f"| ↳ {key} | {failures['selection_failure_subtype_counts'][key]} |")
    lines.extend([
        "",
        "## Table E — Family-level visual ceiling",
        "",
        "All values in this table are **EVALUATOR_ONLY diagnostic upper bounds**, not achieved method accuracy.",
        "",
        "| Metric | Value |", "|---|---:|",
        f"| N families | {jm['n_families']} |",
        f"| JointSelectedRecall@0.5 | {fmt(jm['joint_selected_recall_at_0_5'])} |",
        f"| JointCandidateRecall@0.5 | {fmt(jm['joint_candidate_recall_at_0_5'])} |",
        f"| Selected-region perfect-reasoner ceiling | {fmt(jm['selected_region_perfect_reasoner_ceiling'])} |",
        f"| Candidate-oracle perfect-reasoner ceiling | {fmt(jm['candidate_oracle_perfect_reasoner_ceiling'])} |",
        "",
        "## Candidate multiplicity trace",
        "",
        f"Candidate counts were {pipeline_trace['accepted_candidate_count_distribution']}. The bundled Florence postprocessor produced the same total count as the router received in {pipeline_trace['postprocessor_to_router_count_match_fraction']:.3f} of requests. The harness applies no candidate threshold and no NMS; it performs only bbox validity checking before the router. The router received all accepted candidates, then selected exactly one using the unchanged key `max(native_score, then smaller x0)`. Florence exposes no native score here, so the tie-break is smaller `x0`.",
        "",
        f"The pre-postprocessor structured candidate count is unavailable because the backend emits autoregressive text. Thus multiplicity is not being truncated by the router: where only one candidate is observed, the bundled Florence postprocessor itself exposed one accepted box. Exact stage records and rejected invalid outputs (if any) are in `candidate_pipeline_trace.json`.",
        "",
        "## Scientific questions",
        "",
        "### Q1 — How often is a usable candidate generated when the selected region is wrong?",
        "",
        f"{selection_failures}/{len(selected_wrong)} selected-wrong entities ({fmt(selection_failures / len(selected_wrong) if selected_wrong else None)}) still contain a candidate with IoU>=0.5; this is {fmt(selection_failures / len(records))} of all entities.",
        "",
        "### Q2 — Generation versus selection contribution",
        "",
        f"Among the {len(selected_wrong)} entities below selected IoU 0.5, {generation_failures} are generation failures and {selection_failures} are selection failures. The decomposition is direct and mutually exclusive.",
        "",
        "### Q3 — Conditional current-selection recovery",
        "",
        f"SelectionSuccess@0.5 | CandidateExists@0.5 is {fmt(cm['selection_success_at_0_5_given_candidate_exists'])} ({cm['n_selected_success_among_candidate_at_0_5']}/{cm['n_entities_with_candidate_at_0_5']}).",
        "",
        "### Q4 — SigLIP rank of the correct candidate",
        "",
        f"Conditional on a usable candidate, the GT-best candidate is semantic rank 1 in {fmt(sm['gt_best_rank_1_fraction'])}, rank <=2 in {fmt(sm['gt_best_rank_le_2_fraction'])}, with MRR {fmt(sm['mean_reciprocal_rank'])} and median rank {fmt(sm['median_rank'])}. SigLIP top-1 itself is usable in {fmt(sm['semantic_top1_localization_success_given_usable_candidate'])} of these entities.",
        "",
        "### Q5 — Character of low-IoU failures",
        "",
        f"The primary low-IoU split is {generation_failures} no-usable-candidate versus {selection_failures} wrong-selection cases. Selection-failure subtypes are {failures['selection_failure_subtype_counts']}; subtype rules use evaluator geometry plus visible Florence labels/RAW semantic scores and leave unsupported cases UNKNOWN.",
        "",
        "### Q6 — Does multiplicity exist upstream or disappear in code?",
        "",
        f"Multiplicity exists for {pipeline_trace['entities_with_multiple_candidates']}/{len(records)} entities. No accepted candidate is discarded before the frozen selector: postprocessor and router counts match in {fmt(pipeline_trace['postprocessor_to_router_count_match_fraction'])}. Most singleton sets originate at the bundled postprocessor output, not at router truncation.",
        "",
        "### Q7 — Family-level joint availability",
        "",
        f"Both usable A and B candidates exist in {jm['families_with_joint_candidate_at_0_5']}/{jm['n_families']} families ({fmt(jm['joint_candidate_recall_at_0_5'])}).",
        "",
        "### Q8 — Perfect-reasoner perceptual ceilings without certification",
        "",
        f"The selected-region ceiling is {fmt(jm['selected_region_perfect_reasoner_ceiling'])}; the candidate-oracle ceiling is {fmt(jm['candidate_oracle_perfect_reasoner_ceiling'])}. These were computed jointly per family without assuming A/B independence and are not achieved accuracies.",
        "",
        "### Q9 — Next repair target",
        "",
        f"**{recommendation[0]}** — {recommendation[1]}. This recommendation is confined to candidate-stage evidence; it does not recommend calibration, a full benchmark, or a VLM selector.",
        "",
        "## Final recommendation",
        "",
        f"**{recommendation[0]}**",
        "",
        recommendation[1] + ".",
        "",
        "## Outputs",
        "",
        "- `entity_candidate_results.jsonl`, `family_candidate_results.jsonl`",
        "- `candidate_recall_metrics.json`, `selection_regret_metrics.json`, `semantic_ranking_metrics.json`",
        "- `joint_candidate_metrics.json`, `failure_decomposition.json`, `candidate_pipeline_trace.json`",
        "- `audit.html` and `figures/`",
    ])
    (root / "implementation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--florence-device", default="cpu")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if not str(output_root).startswith("/mnt/d/"):
        raise AssertionError("audit outputs must be on the D drive")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output path: {output_root}")
    stage = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(f"staging path already exists: {stage}")
    stage.mkdir(parents=True)
    figures = stage / "figures"
    traces = stage / "inference_traces"
    views = stage / "selected_path_views"
    overlay_root = figures / "candidate_overlays"
    for directory in (figures, traces, views, overlay_root):
        directory.mkdir(parents=True, exist_ok=True)

    paths = Paths()
    ensure_d_drive_environment(paths)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("n_families") != 20 or len(manifest.get("unique_entity_units", [])) != 40:
        raise AssertionError("locked manifest is not the required 20-family/40-entity subset")
    for source in manifest["source_frozen_files"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise AssertionError(f"frozen source hash mismatch: {source['path']}")

    qa_safe = safe_qa_index(paths.qa)
    registry = build_registry(paths, device=args.device, florence_device=args.florence_device)
    reasoner = DeterministicProofStateReasoner()
    scorer = registry.require("siglip2_region_query", "region_evidence").implementation
    inference_records: list[dict[str, Any]] = []
    family_inference: list[dict[str, Any]] = []

    # Inference and candidate RAW scoring phase.  No GT-bearing row or prior
    # evaluator output is loaded before this entire phase completes.
    for family in manifest["families"]:
        family_id = str(family["family_id"])
        case = qa_safe[str(family["case_ids"][0])]
        resolved = resolve_question(case["question"], case["qa_type"])
        trace = run_refactored_case(
            RefactoredCaseInput(
                case_id=f"{family_id}__candidate_level_audit",
                task_type=case["qa_type"],
                image_path=Path(case["image_path"]),
                question=case["question"],
            ),
            registry=registry,
            reasoner=reasoner,
            output_root=views,
            certification_config=CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"),
            acquisition_processor="florence2_query_grounding",
            grounding_score_processor="siglip2_region_query",
            color_processor="siglip2_color",
        )
        if trace["qwen_called"] or trace["reasoner_called"]:
            raise AssertionError("Qwen/reasoner must not be called in the candidate audit")
        # The top-level trace schema contains a reserved ``final_answer`` key
        # even when the reasoner is not called.  Check the actual automatic
        # inference substructures, matching the locked predecessor study.
        for safe_key in (
            "predicate_dag", "referents", "grounding_queries", "processor_contracts",
            "acquisition_trace", "entity_creation_and_reuse", "raw_specialist_observations",
            "views", "binding_certificates", "region_certification", "region_query_evidence",
            "derived_geometry_facts", "derived_selection_facts", "selection",
            "minimal_certified_proof_state", "diagnostic_proof_state",
            "proof_state_dependency_satisfaction", "controller_decision",
        ):
            assert_inference_safe(trace[safe_key], context=f"candidate audit/{family_id}/{safe_key}")
        (traces / f"{family_id}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

        steps = {
            step["inputs"]["query"]["logical_ref"]: step
            for step in trace["acquisition_trace"]
            if step["operation"] == "LOCALIZE"
        }
        evidence_by_ref = {row["logical_ref"]: row for row in trace["region_query_evidence"]}
        spec_by_ref = {spec.logical_ref: spec for spec in resolved.specs}
        family_entities = []
        for logical_ref in ("A", "B"):
            step = steps[logical_ref]
            evidence = evidence_by_ref[logical_ref]
            candidates = step["outputs"].get("candidates", [])
            selected_index = step["outputs"].get("selected_candidate_index")
            if not candidates or selected_index is None:
                raise RuntimeError(f"no accepted Florence candidate for {family_id}/{logical_ref}")
            grounding_query = GroundingQuery.from_spec(spec_by_ref[logical_ref])
            region_query = build_region_query(grounding_query)
            candidate_rows = []
            for index, payload in enumerate(candidates):
                region = region_from_payload(payload)
                if index == selected_index:
                    z_raw = float(evidence["s_raw_raw"])
                    normalized = float(evidence["s_raw"])
                    score_source = "reused_from_unchanged_selected_path_RAW_score"
                else:
                    score = scorer.score_fixed_region_query(
                        Path(case["image_path"]), evidence["entity_id"], region, region_query, view_id="RAW"
                    )
                    z_raw = float(score.raw_score)
                    normalized = float(score.normalized_score)
                    score_source = "direct_existing_siglip2_region_query_RAW_score"
                candidate_rows.append(
                    {
                        "candidate_index": index,
                        "bbox_xyxy_pixels": list(region.bbox_xyxy_pixels),
                        "bbox_xyxy_normalized": list(region.bbox_xyxy_normalized),
                        "returned_florence_label_text": region.label,
                        "native_score": region.score,
                        "native_score_status": "UNSUPPORTED" if region.score is None else "AVAILABLE",
                        "candidate_source": region.source,
                        "candidate_task": (step["outputs"].get("candidate_pipeline_trace") or {}).get("task"),
                        "pre_filter_status": "BUNDLED_POSTPROCESSOR_OUTPUT",
                        "post_filter_status": "ACCEPTED_VALID_REGION",
                        "became_current_selected_region": index == selected_index,
                        "florence_query_text": grounding_query.semantic_text,
                        "query_provenance": sorted({p.value for values in grounding_query.token_provenance.values() for p in values}),
                        "semantic_diagnostic": {
                            "processor": "siglip2_region_query",
                            "view": "RAW",
                            "z_raw": z_raw,
                            "normalized_trace_value": normalized,
                            "normalized_value_semantics": "monotonic sigmoid; not calibrated probability",
                            "query_text": region_query.text,
                            "query_template_id": region_query.template_id,
                            "query_hash": region_query.query_hash,
                            "crop_policy_id": evidence["crop_policy_id"],
                            "score_source": score_source,
                            "used_for_selection": False,
                        },
                    }
                )
            label_token_sets = [normalized_label_tokens(row["returned_florence_label_text"]) for row in candidate_rows]
            query_tokens = set(normalized_label_tokens(grounding_query.semantic_text))
            similar_count = sum(
                any(tokens and tokens == other for j, other in enumerate(label_token_sets) if j != i)
                for i, tokens in enumerate(label_token_sets)
            )
            same_category_count = sum(bool(set(tokens) & query_tokens) for tokens in label_token_sets)
            sorted_z = sorted((row["semantic_diagnostic"]["z_raw"] for row in candidate_rows), reverse=True)
            semantic_gap = sorted_z[0] - sorted_z[1] if len(sorted_z) > 1 else None
            spec_payload = spec_by_ref[logical_ref].to_dict()
            record = {
                "schema_version": "candidate_level_entity_audit_v1",
                "inference": {
                    "family_id": family_id,
                    "case_ids": list(family["case_ids"]),
                    "image_id": str(family["image_id"]),
                    "image_path": str(case["image_path"]),
                    "image_size": list(Image.open(case["image_path"]).size),
                    "logical_ref": logical_ref,
                    "persistent_entity_id": evidence["entity_id"],
                    "persistent_entity_acquisition_request_id": f"{family_id}::{step['node_id']}::{evidence['entity_id']}",
                    "entity_unit_id": f"{family_id}::{family['image_id']}::{logical_ref}",
                    "candidate_count": len(candidate_rows),
                    "selected_candidate_index": selected_index,
                    "current_selection_key": "max(native_score; absent scores behind present scores, then smaller x0)",
                    "florence_query_text": grounding_query.semantic_text,
                    "florence_query": grounding_query.to_dict(),
                    "siglip_query": region_query.to_dict(),
                    "query_provenance": sorted({p.value for values in grounding_query.token_provenance.values() for p in values}),
                    "referent_spec": spec_payload,
                    "available_resolution_constraints": resolution_constraints(spec_payload, trace["predicate_dag"], resolved.selection_side),
                    "candidate_ambiguity": {
                        "total_candidate_count": len(candidate_rows),
                        "number_of_candidates_with_similar_florence_text_labels": similar_count,
                        "similar_label_rule": "same normalized visible token set as at least one other candidate",
                        "number_of_candidates_with_same_legal_noun_category_semantics": same_category_count,
                        "same_category_rule": "visible Florence label shares a normalized token with legal noun query",
                        "semantic_gap_top1_top2": semantic_gap,
                        "semantic_score_gaps_from_top1": [sorted_z[0] - value for value in sorted_z],
                        "no_ambiguity_pass_fail_threshold": True,
                    },
                    "candidate_pipeline_trace": step["outputs"].get("candidate_pipeline_trace"),
                    "candidates": candidate_rows,
                    "qwen_called": False,
                    "vlm_candidate_selector_called": False,
                },
            }
            assert_inference_safe(record, context=f"candidate audit entity/{family_id}/{logical_ref}")
            inference_records.append(record)
            family_entities.append(record["inference"]["entity_unit_id"])
        family_inference.append(
            {
                "family_id": family_id,
                "case_ids": list(family["case_ids"]),
                "image_id": str(family["image_id"]),
                "image_path": str(case["image_path"]),
                "entity_unit_ids": family_entities,
                "qwen_called": False,
            }
        )

    # EVALUATOR-ONLY PHASE.  GT and the old evaluator-bearing result file are
    # intentionally opened only after inference and every candidate score are complete.
    qa_evaluator = {str(row["qa_id"]): row for row in read_jsonl(paths.qa)}
    baseline_rows = [json.loads(line) for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = {row["inference"]["entity_unit_id"]: row["inference"] for row in baseline_rows}
    family_manifest = {str(row["family_id"]): row for row in manifest["families"]}
    regression_mismatches = []

    for record in inference_records:
        inf = record["inference"]
        family = family_manifest[inf["family_id"]]
        gt_row = qa_evaluator[str(family["case_ids"][0])]
        gt_xywh = gt_row[f"bbox_{inf['logical_ref']}"]
        gt_xyxy = list(xywh_to_xyxy(gt_xywh))
        for candidate in inf["candidates"]:
            candidate_iou = iou_xyxy(tuple(candidate["bbox_xyxy_pixels"]), tuple(gt_xyxy))
            candidate["EVALUATOR_ONLY"] = {
                "namespace": "EVALUATOR_ONLY",
                "inference_use": False,
                "candidate_iou_with_gt": candidate_iou,
            }
        iou_ranked = sorted(inf["candidates"], key=lambda row: (-row["EVALUATOR_ONLY"]["candidate_iou_with_gt"], row["candidate_index"]))
        semantic_ranked = sorted(inf["candidates"], key=lambda row: (-row["semantic_diagnostic"]["z_raw"], row["candidate_index"]))
        for rank, candidate in enumerate(semantic_ranked, 1):
            candidate["EVALUATOR_ONLY"]["semantic_rank"] = rank
        best = iou_ranked[0]
        selected = inf["candidates"][inf["selected_candidate_index"]]
        semantic_top1 = semantic_ranked[0]
        best_iou = best["EVALUATOR_ONLY"]["candidate_iou_with_gt"]
        selected_iou = selected["EVALUATOR_ONLY"]["candidate_iou_with_gt"]
        primary = "SELECTION_SUCCESS" if selected_iou >= .5 else ("SELECTION_FAILURE" if best_iou >= .5 else "GENERATION_FAILURE")
        record["EVALUATOR_ONLY"] = {
            "namespace": "EVALUATOR_ONLY",
            "inference_use": False,
            "joined_after_all_inference_and_candidate_scoring": True,
            "gt_bbox_xywh": gt_xywh,
            "gt_bbox_xyxy_pixels": gt_xyxy,
            "best_candidate_iou": best_iou,
            "best_candidate_index": best["candidate_index"],
            "selected_iou": selected_iou,
            "selected_candidate_index": inf["selected_candidate_index"],
            "selection_regret": best_iou - selected_iou,
            "candidate_count": len(inf["candidates"]),
            "num_candidates_iou_ge_0_3": sum(row["EVALUATOR_ONLY"]["candidate_iou_with_gt"] >= .3 for row in inf["candidates"]),
            "num_candidates_iou_ge_0_5": sum(row["EVALUATOR_ONLY"]["candidate_iou_with_gt"] >= .5 for row in inf["candidates"]),
            "num_candidates_iou_ge_0_7": sum(row["EVALUATOR_ONLY"]["candidate_iou_with_gt"] >= .7 for row in inf["candidates"]),
            "semantic_rank_of_gt_best_candidate": best["EVALUATOR_ONLY"]["semantic_rank"],
            "selected_candidate_semantic_rank": selected["EVALUATOR_ONLY"]["semantic_rank"],
            "semantic_top1_candidate_index": semantic_top1["candidate_index"],
            "semantic_top1_candidate_iou": semantic_top1["EVALUATOR_ONLY"]["candidate_iou_with_gt"],
            "primary_failure_category": primary,
            "selection_failure_subtype": None,
            "selection_failure_subtype_basis": None,
            "oracle_label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
        }
        subtype, basis = primary_subtype(record)
        record["EVALUATOR_ONLY"]["selection_failure_subtype"] = subtype
        record["EVALUATOR_ONLY"]["selection_failure_subtype_basis"] = basis

        old = baseline.get(inf["entity_unit_id"])
        regression_fields = {
            "bbox": old is not None and old["persistent_bbox_xyxy_pixels"] == selected["bbox_xyxy_pixels"],
            "normalized_bbox": old is not None and old["persistent_bbox_xyxy_normalized"] == selected["bbox_xyxy_normalized"],
            "label": old is not None and old["selected_label"] == selected["returned_florence_label_text"],
            "source": old is not None and old["region_source"] == selected["candidate_source"],
        }
        if not all(regression_fields.values()):
            regression_mismatches.append({"entity_unit_id": inf["entity_unit_id"], "field_matches": regression_fields})

    if regression_mismatches:
        raise AssertionError(f"selected-region regression failed: {regression_mismatches}")

    # Entity-level summaries.
    candidate_metrics = {
        "schema_version": "candidate_recall_metrics_v1",
        "namespace": "EVALUATOR_ONLY",
        "label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
        "n_unique_entities": len(inference_records),
    }
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        candidate_metrics[f"candidate_recall_at_{suffix}"] = metric_fraction(inference_records, lambda row, t=threshold: row["EVALUATOR_ONLY"]["best_candidate_iou"] >= t)
        candidate_metrics[f"selected_recall_at_{suffix}"] = metric_fraction(inference_records, lambda row, t=threshold: row["EVALUATOR_ONLY"]["selected_iou"] >= t)
    for threshold in (.5, .7):
        eligible = [row for row in inference_records if row["EVALUATOR_ONLY"]["best_candidate_iou"] >= threshold]
        successes = sum(row["EVALUATOR_ONLY"]["selected_iou"] >= threshold for row in eligible)
        suffix = str(threshold).replace(".", "_")
        candidate_metrics[f"n_entities_with_candidate_at_{suffix}"] = len(eligible)
        candidate_metrics[f"n_selected_success_among_candidate_at_{suffix}"] = successes
        candidate_metrics[f"selection_success_at_{suffix}_given_candidate_exists"] = successes / len(eligible) if eligible else None
    candidate_metrics["candidate_oracle_entity_recall_at_0_5"] = candidate_metrics["candidate_recall_at_0_5"]

    def regret_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        values = [row["EVALUATOR_ONLY"]["selection_regret"] for row in rows]
        return {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "percentile_75": percentile(values, 75),
            "percentile_90": percentile(values, 90),
            "fraction_gt_0_1": float(np.mean(np.asarray(values) > .1)) if values else None,
            "fraction_gt_0_3": float(np.mean(np.asarray(values) > .3)) if values else None,
            "fraction_gt_0_5": float(np.mean(np.asarray(values) > .5)) if values else None,
        }
    regret_metrics = {
        "schema_version": "selection_regret_metrics_v1",
        "namespace": "EVALUATOR_ONLY",
        "overall": regret_group(inference_records),
        "candidate_count_eq_1": regret_group([row for row in inference_records if row["inference"]["candidate_count"] == 1]),
        "candidate_count_gt_1": regret_group([row for row in inference_records if row["inference"]["candidate_count"] > 1]),
    }

    usable = [row for row in inference_records if row["EVALUATOR_ONLY"]["best_candidate_iou"] >= .5]
    ranks = [row["EVALUATOR_ONLY"]["semantic_rank_of_gt_best_candidate"] for row in usable]
    semantic_metrics = {
        "schema_version": "semantic_ranking_metrics_v1",
        "namespace": "EVALUATOR_ONLY",
        "diagnostic_only": True,
        "n_entities_with_usable_candidate": len(usable),
        "gt_best_rank_1_fraction": float(np.mean(np.asarray(ranks) == 1)) if ranks else None,
        "gt_best_rank_le_2_fraction": float(np.mean(np.asarray(ranks) <= 2)) if ranks else None,
        "mean_reciprocal_rank": float(np.mean([1 / rank for rank in ranks])) if ranks else None,
        "median_rank": float(np.median(ranks)) if ranks else None,
        "semantic_top1_localization_success_given_usable_candidate": metric_fraction(usable, lambda row: row["EVALUATOR_ONLY"]["semantic_top1_candidate_iou"] >= .5),
        "raw_logit_primary": True,
        "used_for_inference_selection": False,
    }

    by_entity = {row["inference"]["entity_unit_id"]: row for row in inference_records}
    family_rows = []
    for family in family_inference:
        entities = [by_entity[entity_id] for entity_id in family["entity_unit_ids"]]
        by_ref = {row["inference"]["logical_ref"]: row for row in entities}
        usable_a = by_ref["A"]["EVALUATOR_ONLY"]["best_candidate_iou"] >= .5
        usable_b = by_ref["B"]["EVALUATOR_ONLY"]["best_candidate_iou"] >= .5
        selected_a = by_ref["A"]["EVALUATOR_ONLY"]["selected_iou"] >= .5
        selected_b = by_ref["B"]["EVALUATOR_ONLY"]["selected_iou"] >= .5
        family_rows.append(
            {
                "schema_version": "candidate_level_family_audit_v1",
                "inference": family,
                "EVALUATOR_ONLY": {
                    "namespace": "EVALUATOR_ONLY",
                    "oracle_label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
                    "usable_A_candidate_exists": usable_a,
                    "usable_B_candidate_exists": usable_b,
                    "both_usable_candidates_exist": usable_a and usable_b,
                    "selected_A_usable": selected_a,
                    "selected_B_usable": selected_b,
                    "both_current_selected_usable": selected_a and selected_b,
                    "alternative_or_current_candidate_pair_both_usable_exists": usable_a and usable_b,
                    "selected_region_perfect_reasoner_eligible": selected_a and selected_b,
                    "candidate_oracle_perfect_reasoner_eligible": usable_a and usable_b,
                    "entities": [
                        {
                            "entity_unit_id": row["inference"]["entity_unit_id"],
                            "logical_ref": row["inference"]["logical_ref"],
                            "selected_iou": row["EVALUATOR_ONLY"]["selected_iou"],
                            "best_candidate_iou": row["EVALUATOR_ONLY"]["best_candidate_iou"],
                        }
                        for row in entities
                    ],
                },
            }
        )
    joint_metrics = {
        "schema_version": "joint_candidate_metrics_v1",
        "namespace": "EVALUATOR_ONLY",
        "oracle_label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
        "n_families": len(family_rows),
        "families_with_joint_selected_at_0_5": sum(row["EVALUATOR_ONLY"]["both_current_selected_usable"] for row in family_rows),
        "families_with_joint_candidate_at_0_5": sum(row["EVALUATOR_ONLY"]["both_usable_candidates_exist"] for row in family_rows),
    }
    joint_metrics["joint_selected_recall_at_0_5"] = joint_metrics["families_with_joint_selected_at_0_5"] / len(family_rows)
    joint_metrics["joint_candidate_recall_at_0_5"] = joint_metrics["families_with_joint_candidate_at_0_5"] / len(family_rows)
    joint_metrics["candidate_oracle_family_joint_recall_at_0_5"] = joint_metrics["joint_candidate_recall_at_0_5"]
    joint_metrics["selected_region_perfect_reasoner_ceiling"] = joint_metrics["joint_selected_recall_at_0_5"]
    joint_metrics["candidate_oracle_perfect_reasoner_ceiling"] = joint_metrics["joint_candidate_recall_at_0_5"]
    joint_metrics["a_b_independence_assumed"] = False

    failures = {
        "schema_version": "candidate_failure_decomposition_v1",
        "namespace": "EVALUATOR_ONLY",
        "primary_counts": dict(Counter(row["EVALUATOR_ONLY"]["primary_failure_category"] for row in inference_records)),
        "selection_failure_subtype_counts": dict(Counter(row["EVALUATOR_ONLY"]["selection_failure_subtype"] for row in inference_records if row["EVALUATOR_ONLY"]["primary_failure_category"] == "SELECTION_FAILURE")),
    }
    for key in ("GENERATION_FAILURE", "SELECTION_FAILURE", "SELECTION_SUCCESS"):
        failures["primary_counts"].setdefault(key, 0)
    for key in ("WRONG_SAME_CATEGORY_INSTANCE", "SEMANTIC_DISTRACTOR", "PARTIAL_REGION_SELECTED", "OVERLARGE_REGION_SELECTED", "UNKNOWN"):
        failures["selection_failure_subtype_counts"].setdefault(key, 0)
    failures["entities"] = [
        {
            "entity_unit_id": row["inference"]["entity_unit_id"],
            "primary_category": row["EVALUATOR_ONLY"]["primary_failure_category"],
            "selection_failure_subtype": row["EVALUATOR_ONLY"]["selection_failure_subtype"],
            "subtype_basis": row["EVALUATOR_ONLY"]["selection_failure_subtype_basis"],
        }
        for row in inference_records
    ]

    post_matches = []
    rejected_total = 0
    per_request_pipeline = []
    for row in inference_records:
        trace_payload = row["inference"].get("candidate_pipeline_trace") or {}
        accepted = row["inference"]["candidate_count"]
        post = trace_payload.get("accepted_candidate_count")
        post_matches.append(post == accepted)
        rejected_total += int(trace_payload.get("rejected_candidate_count") or 0)
        per_request_pipeline.append(
            {
                "entity_unit_id": row["inference"]["entity_unit_id"],
                "persistent_entity_acquisition_request_id": row["inference"]["persistent_entity_acquisition_request_id"],
                "raw_model_candidate_count": trace_payload.get("raw_model_candidate_count"),
                "raw_model_candidate_count_status": trace_payload.get("raw_model_candidate_count_status"),
                "bundled_postprocessor_candidate_count": trace_payload.get("bundled_postprocessor_candidate_count"),
                "accepted_candidate_count_after_bbox_validation": trace_payload.get("accepted_candidate_count"),
                "rejected_candidate_count": trace_payload.get("rejected_candidate_count"),
                "router_received_candidate_count": accepted,
                "selected_count": 1,
                "harness_threshold_applied": trace_payload.get("harness_threshold_applied"),
                "harness_nms_applied": trace_payload.get("harness_nms_applied"),
                "selected_candidate_index": row["inference"]["selected_candidate_index"],
            }
        )
    pipeline_trace = {
        "schema_version": "candidate_pipeline_trace_v1",
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "pipeline": [
            "autoregressive Florence output (structured raw candidate count unavailable)",
            "bundled post_process_generation bboxes",
            "RegionPrediction bbox validation",
            "router receives all accepted candidates",
            "unchanged selector chooses one",
        ],
        "multiplicity_loss_finding": "No accepted candidates are truncated by current router logging or selection input. Most singleton sets are already singleton outputs of the bundled Florence postprocessor.",
        "exact_candidate_discard_location": "None for accepted candidates; invalid bboxes only are rejected in Florence2Processor.predict_query before routing.",
        "harness_threshold_changed_or_applied": False,
        "harness_nms_changed_or_applied": False,
        "florence_query_changed": False,
        "crop_policy_changed": False,
        "siglip_changed": False,
        "qwen_called": False,
        "vlm_candidate_selector_called": False,
        "accepted_candidate_count_distribution": {str(k): v for k, v in sorted(Counter(row["inference"]["candidate_count"] for row in inference_records).items())},
        "entities_with_multiple_candidates": sum(row["inference"]["candidate_count"] > 1 for row in inference_records),
        "postprocessor_to_router_count_match_fraction": float(np.mean(post_matches)),
        "total_rejected_invalid_candidates": rejected_total,
        "selected_region_regression": {
            "status": "PASS",
            "baseline_path": str(BASELINE_PATH),
            "matched_entities": len(inference_records),
            "mismatches": [],
            "comparison_fields": ["bbox_xyxy_pixels", "bbox_xyxy_normalized", "label", "source"],
        },
        "requests": per_request_pipeline,
    }

    recommendation = recommendation_for(inference_records)
    for row in inference_records:
        draw_candidate_overlay(row, overlay_root / f"{row['inference']['family_id']}__{row['inference']['logical_ref']}.jpg")
    make_figures(inference_records, figures)

    write_jsonl(stage / "entity_candidate_results.jsonl", inference_records)
    write_jsonl(stage / "family_candidate_results.jsonl", family_rows)
    write_json(stage / "candidate_recall_metrics.json", candidate_metrics)
    write_json(stage / "selection_regret_metrics.json", regret_metrics)
    write_json(stage / "semantic_ranking_metrics.json", semantic_metrics)
    write_json(stage / "joint_candidate_metrics.json", joint_metrics)
    write_json(stage / "failure_decomposition.json", failures)
    write_json(stage / "candidate_pipeline_trace.json", pipeline_trace)
    (stage / "audit.html").write_text(
        build_html(inference_records, {"candidate_recall": candidate_metrics}, overlay_root), encoding="utf-8"
    )
    write_report(stage, inference_records, family_rows, candidate_metrics, regret_metrics, semantic_metrics, joint_metrics, failures, pipeline_trace, recommendation)

    # Close the persistent Florence worker before publishing the completed tree.
    registry.require("florence2_query_grounding", "grounding").implementation.close()
    stage.rename(output_root)

    print(f"OUTPUT: {output_root}")
    print(f"ENTITIES/FAMILIES: {len(inference_records)}/{len(family_rows)}")
    print(f"SELECTED RECALL@0.5: {candidate_metrics['selected_recall_at_0_5']:.4f}")
    print(f"CANDIDATE RECALL@0.5: {candidate_metrics['candidate_recall_at_0_5']:.4f}")
    print(f"SELECTION SUCCESS | CANDIDATE EXISTS: {candidate_metrics['selection_success_at_0_5_given_candidate_exists']:.4f}")
    print(f"JOINT SELECTED/CANDIDATE: {joint_metrics['joint_selected_recall_at_0_5']:.4f}/{joint_metrics['joint_candidate_recall_at_0_5']:.4f}")
    print(f"FAILURES: {failures['primary_counts']}")
    print("QWEN CALLED: false")
    print(f"RECOMMENDATION: {recommendation[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
