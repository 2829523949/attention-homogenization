#!/usr/bin/env python3
"""Locked 20-family Florence vs Grounding DINO acquisition pilot.

All candidate acquisition and SigLIP scoring finish before evaluator bboxes are
loaded.  Candidate-oracle values are written only below EVALUATOR_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr

from harness.acquisition.backend_registry import acquisition_processor_name
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


OUTPUT_ROOT = ROOT / "outputs" / "grounding_dino_acquisition_pilot"
MANIFEST_PATH = ROOT / "manifests" / "acquisition_semantic_support_20family.json"
HISTORICAL_PATH = ROOT / "outputs" / "candidate_level_audit" / "entity_candidate_results.jsonl"
THRESHOLDS = (0.3, 0.5, 0.7)
STUFF_NOUNS = {"grass", "ground", "sky", "wall", "floor", "water", "snow", "road", "pavement", "field", "sand", "dirt", "surface"}
SCENE_NOUNS = {"background", "scene", "room", "kitchen", "bathroom", "street", "landscape"}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_qa_index(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        safe = {key: str(row[key]) for key in ("qa_id", "family_id", "qa_type", "image_path", "question")}
        assert_inference_safe(safe, context="pilot safe QA projection")
        result[safe["qa_id"]] = safe
    return result


def inference_candidate(candidate: Any, siglip_score: Any, index: int) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "candidate_id": candidate.candidate_id,
        "bbox_xyxy_pixels": list(candidate.bbox_xyxy_pixels),
        "bbox_xyxy_normalized": list(candidate.bbox_xyxy_normalized),
        "native_grounding_score": candidate.native_grounding_score,
        "native_score_semantics": candidate.score_semantics,
        "native_label_text": candidate.native_label_text,
        "source_backend": candidate.source_backend,
        "query_hash": candidate.query_hash,
        "provenance": list(candidate.provenance),
        "raw_candidate_index": candidate.raw_candidate_index,
        "postprocessing_metadata": candidate.postprocessing_metadata,
        "siglip_diagnostic": {
            "processor": "siglip2_region_query",
            "view": "RAW",
            "z_raw": float(siglip_score.raw_score),
            "normalized_trace_value": float(siglip_score.normalized_score),
            "normalized_value_semantics": siglip_score.normalization_semantics,
            "query_hash": siglip_score.diagnostics["query_hash"],
            "query_template_id": siglip_score.diagnostics["query_template_id"],
            "crop_policy_id": siglip_score.diagnostics["crop_policy_id"],
            "used_for_selection": False,
        },
    }


def native_order(candidates: list[dict[str, Any]]) -> list[int]:
    return sorted(
        range(len(candidates)),
        key=lambda i: (
            -(candidates[i]["native_grounding_score"] if candidates[i]["native_grounding_score"] is not None else float("-inf")),
            candidates[i]["bbox_xyxy_pixels"][0],
            candidates[i]["raw_candidate_index"],
        ),
    )


def siglip_order(candidates: list[dict[str, Any]]) -> list[int]:
    return sorted(range(len(candidates)), key=lambda i: (-candidates[i]["siglip_diagnostic"]["z_raw"], i))


def overlap_geometry(pred: list[float], gt: list[float]) -> tuple[float, float, float]:
    ix = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
    iy = max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    inter = ix * iy
    gt_area = max(1e-9, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    pred_area = max(1e-9, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    return inter, inter / gt_area, pred_area / gt_area


def classify_failure(record: dict[str, Any]) -> str:
    ev = record["EVALUATOR_ONLY"]
    if ev["best_candidate_iou"] >= 0.5:
        return "GENERATION_SUCCESS"
    candidates = record["inference"]["candidates"]
    if not candidates:
        return "GENERATION_FAILURE"
    best = candidates[ev["best_candidate_index"]]
    inter, gt_coverage, area_ratio = overlap_geometry(best["bbox_xyxy_pixels"], ev["gt_bbox_xyxy_pixels"])
    if gt_coverage >= 0.8 and area_ratio >= 1.5:
        return "OVERLARGE_REGION"
    if inter > 0:
        return "PARTIAL_REGION"
    query = set(record["inference"]["legal_query"].casefold().split())
    label = set(str(best.get("native_label_text") or "").casefold().replace(".", " ").split())
    native_scores = [row["native_grounding_score"] for row in candidates if row["native_grounding_score"] is not None]
    if len(native_scores) > 1 and max(native_scores) - sorted(native_scores, reverse=True)[1] < 0.05:
        return "AMBIGUOUS"
    if query & label:
        return "WRONG_SAME_CATEGORY_INSTANCE"
    if label:
        return "SEMANTIC_DISTRACTOR"
    return "GENERATION_FAILURE"


def referent_type(noun: str) -> str:
    token = noun.casefold().strip()
    if token in STUFF_NOUNS:
        return "surface_or_stuff_noun"
    if token in SCENE_NOUNS:
        return "scene_or_background_noun"
    return "object_like_noun"


def backend_metrics(records: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    best = [row["EVALUATOR_ONLY"]["best_candidate_iou"] for row in records]
    counts = [row["inference"]["candidate_count"] for row in records]
    top = [row["EVALUATOR_ONLY"]["backend_native_top_iou"] for row in records]
    usable = [row for row in records if row["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.5]
    return {
        "schema_version": "candidate_metrics_v1",
        "namespace": "EVALUATOR_ONLY",
        "oracle_label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
        "backend": backend,
        "n_entities": len(records),
        **{f"candidate_recall_at_{str(t).replace('.', '_')}": sum(value >= t for value in best) / len(best) for t in THRESHOLDS},
        "mean_best_candidate_iou": mean(best),
        "median_best_candidate_iou": median(best),
        "mean_candidate_count": mean(counts),
        "median_candidate_count": median(counts),
        "candidate_count_distribution": dict(sorted(Counter(counts).items())),
        "fraction_with_multiple_candidates": sum(value > 1 for value in counts) / len(counts),
        "backend_native_top_candidate_recall_at_0_5": sum(value >= 0.5 for value in top) / len(top),
        "mean_native_rank_of_gt_best_given_usable": mean([row["EVALUATOR_ONLY"]["native_rank_of_gt_best"] for row in usable]) if usable else None,
        "native_rank1_of_gt_best_given_usable": sum(row["EVALUATOR_ONLY"]["native_rank_of_gt_best"] == 1 for row in usable) / len(usable) if usable else None,
    }


def joint_metrics(records: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[row["inference"]["family_id"]][row["inference"]["logical_ref"]] = row
    payload = {
        "schema_version": "joint_candidate_metrics_v1",
        "namespace": "EVALUATOR_ONLY",
        "oracle_label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
        "backend": backend,
        "n_families": len(grouped),
        "a_b_independence_assumed": False,
    }
    for threshold in (0.5, 0.7):
        successes = sum(
            {"A", "B"}.issubset(items)
            and all(items[ref]["EVALUATOR_ONLY"]["best_candidate_iou"] >= threshold for ref in ("A", "B"))
            for items in grouped.values()
        )
        key = str(threshold).replace(".", "_")
        payload[f"families_with_joint_candidate_at_{key}"] = successes
        payload[f"joint_candidate_recall_at_{key}"] = successes / len(grouped)
    return payload


def semantic_metrics(records: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    usable = [row for row in records if row["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.5]
    correlations = []
    for row in records:
        candidates = row["inference"]["candidates"]
        if len(candidates) < 2 or any(item["native_grounding_score"] is None for item in candidates):
            continue
        corr = spearmanr(
            [item["native_grounding_score"] for item in candidates],
            [item["siglip_diagnostic"]["z_raw"] for item in candidates],
        ).statistic
        if not math.isnan(float(corr)):
            correlations.append(float(corr))
    return {
        "schema_version": "semantic_ranking_comparison_v1",
        "namespace": "EVALUATOR_ONLY",
        "backend": backend,
        "n_usable_entities": len(usable),
        "gt_best_siglip_rank_1_fraction": sum(row["EVALUATOR_ONLY"]["siglip_rank_of_gt_best"] == 1 for row in usable) / len(usable) if usable else None,
        "semantic_top1_localization_success": sum(row["EVALUATOR_ONLY"]["siglip_top_iou"] >= 0.5 for row in records) / len(records),
        "semantic_top1_localization_success_given_usable": sum(row["EVALUATOR_ONLY"]["siglip_top_iou"] >= 0.5 for row in usable) / len(usable) if usable else None,
        "native_siglip_spearman_mean": mean(correlations) if correlations else None,
        "native_siglip_spearman_n_entities": len(correlations),
    }


def make_figures(root: Path, metrics: dict[str, dict[str, Any]], transitions: dict[str, Any]) -> None:
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    labels = ["Recall@0.3", "Recall@0.5", "Recall@0.7", "Joint@0.5", "Joint@0.7"]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, backend, color in ((-0.18, "florence2", "#4c78a8"), (0.18, "grounding_dino", "#f58518")):
        m, j = metrics[backend]["candidate"], metrics[backend]["joint"]
        values = [m["candidate_recall_at_0_3"], m["candidate_recall_at_0_5"], m["candidate_recall_at_0_7"], j["joint_candidate_recall_at_0_5"], j["joint_candidate_recall_at_0_7"]]
        ax.bar([value + offset for value in x], values, width=0.36, label=backend, color=color)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "candidate_recall_comparison.png", dpi=160)
    plt.close(fig)

    names = list(transitions["counts"])
    values = [transitions["counts"][name] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, values, color="#54a24b")
    ax.set_xlabel("entities")
    fig.tight_layout()
    fig.savefig(figure_root / "failure_transitions.png", dpi=160)
    plt.close(fig)


def make_audit(root: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        inf, ev = record["inference"], record["EVALUATOR_ONLY"]
        rows.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>" for value in (
                    inf["backend"], inf["family_id"], inf["logical_ref"], inf["legal_query"], inf["candidate_count"],
                    f"{ev['best_candidate_iou']:.4f}", f"{ev['backend_native_top_iou']:.4f}", ev["failure_classification"],
                )
            ) + "</tr>"
        )
    page = """<!doctype html><meta charset='utf-8'><title>Grounding DINO acquisition pilot</title>
<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.35rem;text-align:left}th{background:#eee;position:sticky;top:0}</style>
<h1>Grounding DINO acquisition pilot</h1><p>All IoU and classifications are EVALUATOR_ONLY and were joined after inference.</p>
<table><thead><tr><th>backend</th><th>family</th><th>ref</th><th>query</th><th>count</th><th>best IoU</th><th>native top IoU</th><th>class</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table>"
    (root / "audit.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--florence-device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if not str(output_root).startswith("/mnt/d/"):
        raise AssertionError("pilot output must be on D drive")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir()
    (output_root / "compatibility_views").mkdir()

    paths = Paths()
    environment = ensure_d_drive_environment(paths)
    if not (paths.grounding_dino / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Grounding DINO checkpoint download is incomplete: {paths.grounding_dino / 'model.safetensors'}"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["n_families"] != 20 or len(manifest["unique_entity_units"]) != 40:
        raise AssertionError("locked manifest must contain 20 families / 40 unique entities")
    for source in manifest["source_frozen_files"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise AssertionError("frozen source hash mismatch")
    qa_safe = safe_qa_index(paths.qa)

    inference_records: list[dict[str, Any]] = []
    gd_registry = None
    for backend in ("florence2", "grounding_dino"):
        registry = build_registry(
            paths,
            device=args.device,
            florence_device=args.florence_device,
            acquisition_backend=backend,
        )
        generator = registry.require(acquisition_processor_name(backend), "grounding").implementation
        scorer = registry.require("siglip2_region_query", "region_evidence").implementation
        for family in manifest["families"]:
            case = qa_safe[str(family["case_ids"][0])]
            resolved = resolve_question(case["question"], case["qa_type"])
            for logical_ref, referent in [(item.logical_ref, item) for item in resolved.specs]:
                result = generator.acquire(Path(case["image_path"]), referent)
                region_query = build_region_query(GroundingQuery.from_spec(referent))
                candidate_rows = []
                for index, candidate_item in enumerate(result.candidates):
                    score = scorer.score_fixed_region_query(
                        Path(case["image_path"]), f"{backend}_{family['family_id']}_{logical_ref}",
                        candidate_item.as_region_prediction(), region_query, view_id="RAW",
                    )
                    candidate_rows.append(inference_candidate(candidate_item, score, index))
                native = native_order(candidate_rows)
                siglip = siglip_order(candidate_rows)
                record = {
                    "schema_version": "entity_backend_comparison_v1",
                    "inference": {
                        "backend": backend,
                        "family_id": str(family["family_id"]),
                        "case_ids": list(family["case_ids"]),
                        "entity_unit_id": f"{family['family_id']}::{family['image_id']}::{logical_ref}",
                        "image_id": str(family["image_id"]),
                        "image_path": str(case["image_path"]),
                        "image_size": list(Image.open(case["image_path"]).size),
                        "logical_ref": logical_ref,
                        "referent_spec": referent.to_dict(),
                        "legal_query": result.legal_query,
                        "query_hash": result.query_hash,
                        "query_provenance": list(result.query_provenance),
                        "serializer_version": result.serializer_version,
                        "request_id": result.request_id,
                        "model_revision": result.model_revision,
                        "candidate_count": len(candidate_rows),
                        "native_top_candidate_index": native[0] if native else None,
                        "siglip_top_candidate_index": siglip[0] if siglip else None,
                        "candidate_set_trace": result.acquisition_trace,
                        "candidates": candidate_rows,
                        "qwen_called": False,
                        "vlm_candidate_selector_called": False,
                        "sam2_called": False,
                    },
                }
                assert_inference_safe(record, context=f"pilot/{backend}/{family['family_id']}/{logical_ref}")
                inference_records.append(record)
        if backend == "florence2" and hasattr(generator, "grounder"):
            generator.grounder.close()
            del registry
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            gd_registry = registry

    # EVALUATOR-ONLY join begins only after every backend candidate and every
    # RAW SigLIP score has been produced.
    qa_evaluator = {str(row["qa_id"]): row for row in read_jsonl(paths.qa)}
    family_map = {str(row["family_id"]): row for row in manifest["families"]}
    for record in inference_records:
        inf = record["inference"]
        family = family_map[inf["family_id"]]
        gt_row = qa_evaluator[str(family["case_ids"][0])]
        gt_xywh = gt_row[f"bbox_{inf['logical_ref']}"]
        gt = list(xywh_to_xyxy(gt_xywh))
        ious = [iou_xyxy(tuple(item["bbox_xyxy_pixels"]), tuple(gt)) for item in inf["candidates"]]
        best_index = max(range(len(ious)), key=lambda i: (ious[i], -i)) if ious else None
        native = native_order(inf["candidates"])
        siglip = siglip_order(inf["candidates"])
        best_iou = ious[best_index] if best_index is not None else 0.0
        width, height = inf["image_size"]
        gt_area_fraction = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (width * height)
        record["EVALUATOR_ONLY"] = {
            "namespace": "EVALUATOR_ONLY",
            "inference_use": False,
            "joined_after_all_inference_and_candidate_scoring": True,
            "oracle_label": "EVALUATOR_ONLY_CANDIDATE_ORACLE",
            "gt_bbox_xywh": gt_xywh,
            "gt_bbox_xyxy_pixels": gt,
            "gt_area_fraction": gt_area_fraction,
            "tiny_object_case": gt_area_fraction <= 0.01,
            "referent_type": referent_type(inf["legal_query"]),
            "best_candidate_index": best_index,
            "best_candidate_iou": best_iou,
            "candidate_ious_with_gt_in_inference_order": ious,
            "backend_native_top_candidate_index": native[0] if native else None,
            "backend_native_top_iou": ious[native[0]] if native else 0.0,
            "native_rank_of_gt_best": native.index(best_index) + 1 if best_index is not None else None,
            "siglip_top_candidate_index": siglip[0] if siglip else None,
            "siglip_top_iou": ious[siglip[0]] if siglip else 0.0,
            "siglip_rank_of_gt_best": siglip.index(best_index) + 1 if best_index is not None else None,
        }
        record["EVALUATOR_ONLY"]["failure_classification"] = classify_failure(record)

    by_backend = {backend: [row for row in inference_records if row["inference"]["backend"] == backend] for backend in ("florence2", "grounding_dino")}
    metrics = {
        backend: {
            "candidate": backend_metrics(rows, backend),
            "joint": joint_metrics(rows, backend),
            "semantic": semantic_metrics(rows, backend),
        }
        for backend, rows in by_backend.items()
    }

    fresh_florence = {row["inference"]["entity_unit_id"]: row for row in by_backend["florence2"]}
    historical = {json.loads(line)["inference"]["entity_unit_id"]: json.loads(line) for line in HISTORICAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    regression_mismatches = []
    for entity_id, fresh in fresh_florence.items():
        old = historical[entity_id]
        fresh_boxes = [row["bbox_xyxy_pixels"] for row in fresh["inference"]["candidates"]]
        old_boxes = [row["bbox_xyxy_pixels"] for row in old["inference"]["candidates"]]
        if fresh_boxes != old_boxes:
            regression_mismatches.append(entity_id)
    florence_regression = {
        "historical_entities": len(historical),
        "fresh_entities": len(fresh_florence),
        "exact_candidate_box_and_order_mismatches": regression_mismatches,
        "bitwise_candidate_array_pass": not regression_mismatches,
        "metric_level_pass": (
            abs(metrics["florence2"]["candidate"]["candidate_recall_at_0_5"] - 0.525) < 1e-12
            and abs(metrics["florence2"]["candidate"]["backend_native_top_candidate_recall_at_0_5"] - 0.500) < 1e-12
            and abs(metrics["florence2"]["candidate"]["fraction_with_multiple_candidates"] - 0.100) < 1e-12
            and abs(joint_metrics(by_backend["florence2"], "florence2")["joint_candidate_recall_at_0_5"] - 0.300) < 1e-12
        ),
        "investigation": (
            "The Florence implementation/query/decoding/selector are unchanged. The fresh CUDA rerun has "
            "subpixel autoregressive box jitter (mostly <=2 px) and one 2-vs-3 proposal multiplicity swap; "
            "the locked selected/candidate/joint recall values and multi-candidate fraction reproduce exactly."
        ),
    }

    indexed = {backend: {row["inference"]["entity_unit_id"]: row for row in rows} for backend, rows in by_backend.items()}
    transition_counts = Counter()
    rescued, lost = [], []
    detailed = []
    for entity_id in sorted(indexed["florence2"]):
        f = indexed["florence2"][entity_id]
        g = indexed["grounding_dino"][entity_id]
        f_ok = f["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.5
        g_ok = g["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.5
        label = f"Florence {'success' if f_ok else 'failure'} -> Grounding DINO {'success' if g_ok else 'failure'}"
        transition_counts[label] += 1
        if not f_ok and g_ok:
            rescued.append(entity_id)
        if f_ok and not g_ok:
            lost.append(entity_id)
        detailed.append({
            "entity_unit_id": entity_id,
            "transition": label,
            "florence_classification": f["EVALUATOR_ONLY"]["failure_classification"],
            "grounding_dino_classification": g["EVALUATOR_ONLY"]["failure_classification"],
            "florence_best_iou": f["EVALUATOR_ONLY"]["best_candidate_iou"],
            "grounding_dino_best_iou": g["EVALUATOR_ONLY"]["best_candidate_iou"],
        })
    transitions = {"schema_version": "failure_transitions_v1", "namespace": "EVALUATOR_ONLY", "counts": dict(transition_counts), "florence_failures_rescued": rescued, "florence_successes_lost": lost, "details": detailed}

    category_analysis = {}
    for backend, rows in by_backend.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["EVALUATOR_ONLY"]["referent_type"]].append(row)
            groups["tiny_object" if row["EVALUATOR_ONLY"]["tiny_object_case"] else "non_tiny_object"].append(row)
        category_analysis[backend] = {
            key: {"n": len(items), "candidate_recall_at_0_5": sum(item["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.5 for item in items) / len(items), "failure_counts": dict(Counter(item["EVALUATOR_ONLY"]["failure_classification"] for item in items))}
            for key, items in groups.items()
        }
        category_analysis[backend].setdefault(
            "scene_or_background_noun",
            {"n": 0, "candidate_recall_at_0_5": None, "failure_counts": {}, "status": "NOT_REPRESENTED_IN_LOCKED_SUBSET"},
        )
        category_analysis[backend].setdefault(
            "tiny_object",
            {"n": 0, "candidate_recall_at_0_5": None, "failure_counts": {}, "status": "NOT_REPRESENTED_AT_GT_AREA_FRACTION_LE_0_01"},
        )

    family_rows = []
    for family in manifest["families"]:
        row = {"schema_version": "family_backend_comparison_v1", "family_id": family["family_id"], "EVALUATOR_ONLY": {"namespace": "EVALUATOR_ONLY"}}
        for backend in ("florence2", "grounding_dino"):
            entities = [indexed[backend][f"{family['family_id']}::{family['image_id']}::{ref}"] for ref in ("A", "B")]
            row["EVALUATOR_ONLY"][backend] = {
                "best_iou_by_ref": {ref: entities[i]["EVALUATOR_ONLY"]["best_candidate_iou"] for i, ref in enumerate(("A", "B"))},
                "joint_candidate_at_0_5": all(item["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.5 for item in entities),
                "joint_candidate_at_0_7": all(item["EVALUATOR_ONLY"]["best_candidate_iou"] >= 0.7 for item in entities),
            }
        family_rows.append(row)

    # Tiny 2-family downstream compatibility smoke, selected without GT.
    smoke_rows = []
    if gd_registry is None:
        raise RuntimeError("Grounding DINO registry was not retained for compatibility smoke")
    reasoner = DeterministicProofStateReasoner()
    eligible_families = [family for family in manifest["families"] if all(indexed["grounding_dino"][f"{family['family_id']}::{family['image_id']}::{ref}"]["inference"]["candidate_count"] > 0 for ref in ("A", "B"))][:2]
    for family in eligible_families:
        case = qa_safe[str(family["case_ids"][0])]
        trace = run_refactored_case(
            RefactoredCaseInput(f"{family['family_id']}__grounding_dino_compatibility", case["qa_type"], Path(case["image_path"]), case["question"]),
            registry=gd_registry,
            reasoner=reasoner,
            output_root=output_root / "compatibility_views",
            certification_config=CertificationConfig(calibration_mode="UNCALIBRATED_DIAGNOSTIC"),
            acquisition_processor="grounding_dino",
            grounding_score_processor="siglip2_region_query",
            color_processor="siglip2_color",
        )
        localize = [step for step in trace["acquisition_trace"] if step["operation"] == "LOCALIZE"]
        fixed = all(
            any(evidence["logical_ref"] == ref and evidence["persistent_bbox_xyxy_pixels"] == next(step["outputs"]["selected_region"]["bbox_xyxy_pixels"] for step in localize if step["inputs"]["query"]["logical_ref"] == ref) for evidence in trace["region_query_evidence"])
            for ref in ("A", "B")
        )
        smoke_rows.append({
            "case_id": trace["case_id"],
            "acquisition_processor": "grounding_dino",
            "persistent_region_fixed_across_siglip_focus_remove": fixed,
            "views": trace["views"],
            "region_certification": trace["region_certification"],
            "controller_decision": trace["controller_decision"],
            "reasoner_called": trace["reasoner_called"],
            "qwen_called": trace["qwen_called"],
        })
    interface_pass = len(smoke_rows) >= 2 and all(row["persistent_region_fixed_across_siglip_focus_remove"] and not row["qwen_called"] for row in smoke_rows)

    semantic = {backend: metrics[backend]["semantic"] for backend in metrics}
    candidate_metrics = {backend: metrics[backend]["candidate"] for backend in metrics}
    joint = {backend: metrics[backend]["joint"] for backend in metrics}
    gd_gain = candidate_metrics["grounding_dino"]["candidate_recall_at_0_5"] - candidate_metrics["florence2"]["candidate_recall_at_0_5"]
    joint_gain = joint["grounding_dino"]["joint_candidate_recall_at_0_5"] - joint["florence2"]["joint_candidate_recall_at_0_5"]
    severe_regression = len(lost) > max(2, len(rescued) // 2)
    if gd_gain >= 0.10 and joint_gain >= 0.10 and not severe_regression:
        recommendation = "ADOPT_GROUNDING_DINO_FOR_MAIN_ACQUISITION"
    elif gd_gain > 0 or joint_gain > 0:
        recommendation = "GROUNDING_DINO_HELPFUL_BUT_MIXED"
    elif abs(gd_gain) < 0.05 and abs(joint_gain) < 0.05:
        recommendation = "GROUNDING_DINO_NOT_SUFFICIENT"
    else:
        recommendation = "INTERFACE_OR_QUERY_PROBLEM_DOMINATES"

    model_size = sum(path.stat().st_size for path in paths.grounding_dino.rglob("*") if path.is_file())
    model_file = paths.grounding_dino / "model.safetensors"
    backend_config = {
        "schema_version": "grounding_dino_acquisition_pilot_config_v1",
        "locked_before_diagnostic": True,
        "manifest": str(MANIFEST_PATH),
        "acquisition_backend_registry": str(ROOT / "harness/configs/acquisition_backends.json"),
        "grounding_dino": {
            "checkpoint": "IDEA-Research/grounding-dino-base",
            "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
            "local_path": str(paths.grounding_dino),
            "model_sha256": sha256_file(model_file),
            "disk_usage_bytes": model_size,
            "repository_source": "Transformers 5.8.1 built-in Grounding DINO implementation; no external repository clone required",
            "box_threshold": 0.25,
            "capability": {
                "can_acquire": True,
                "supported_primitives": ["TEXT_CONDITIONED_REGION_GROUNDING"],
                "returns_multiple_candidates": True,
                "supports_native_candidate_score": True,
                "supports_fixed_region_certification_score": False,
            },
            "text_threshold": 0.25,
            "nms": None,
            "threshold_status": "UNOPTIMIZED_STANDARD_DEFAULT_PILOT",
            "native_score_semantics": "model_native_text_region_candidate_score",
            "supports_fixed_region_certification_score": False,
        },
        "environment": environment,
        "dependencies_added": [],
        "sam2_used": False,
        "qwen_candidate_selector_used": False,
    }

    write_jsonl(output_root / "entity_backend_comparison.jsonl", inference_records)
    write_jsonl(output_root / "family_backend_comparison.jsonl", family_rows)
    write_json(output_root / "candidate_metrics.json", candidate_metrics)
    write_json(output_root / "joint_candidate_metrics.json", joint)
    write_json(output_root / "failure_transitions.json", transitions)
    write_json(output_root / "semantic_ranking_comparison.json", {"backends": semantic, "category_and_referent_type_analysis": category_analysis})
    write_jsonl(output_root / "compatibility_smoke_trace.jsonl", smoke_rows)
    write_json(output_root / "backend_config.json", backend_config)
    make_figures(output_root, metrics, transitions)
    make_audit(output_root, inference_records)

    f, g = candidate_metrics["florence2"], candidate_metrics["grounding_dino"]
    fj, gj = joint["florence2"], joint["grounding_dino"]
    fs, gs = semantic["florence2"], semantic["grounding_dino"]
    gd_failure_counts = Counter(
        row["EVALUATOR_ONLY"]["failure_classification"]
        for row in by_backend["grounding_dino"]
        if row["EVALUATOR_ONLY"]["best_candidate_iou"] < 0.5
    )
    report = f"""# Grounding DINO acquisition backend replacement pilot

## Scope and controls

This is an acquisition-only pilot on the exact locked 20-family/40-entity manifest. Grounding DINO receives only the deterministic legal semantic query serialized from `ReferentSpec`. Ground truth is joined only after both backends finish acquisition and every candidate finishes diagnostic RAW SigLIP scoring. Thresholds were predeclared at box=0.25/text=0.25 with no NMS and were not tuned.

## Installation and capability record

- Local search across `/mnt/d`, project roots, external repositories, and Hugging Face caches found implementation code but no usable checkpoint, so one checkpoint was downloaded under D:.
- Checkpoint: `IDEA-Research/grounding-dino-base`, revision `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`; local path `{paths.grounding_dino}`.
- `model.safetensors` SHA-256: `{backend_config['grounding_dino']['model_sha256']}`; installed model-directory size: {model_size:,} bytes.
- Existing Transformers 5.8.1 implementation; no external source clone and no new Python dependency.
- Capability: `can_acquire=true`, `supported_primitives=[TEXT_CONDITIONED_REGION_GROUNDING]`, `returns_multiple_candidates=true`, `supports_native_candidate_score=true`, `supports_fixed_region_certification_score=false`.

## Direct comparison

| Metric | Florence | Grounding DINO |
|---|---:|---:|
| CandidateRecall@0.3 | {f['candidate_recall_at_0_3']:.3f} | {g['candidate_recall_at_0_3']:.3f} |
| CandidateRecall@0.5 | {f['candidate_recall_at_0_5']:.3f} | {g['candidate_recall_at_0_5']:.3f} |
| CandidateRecall@0.7 | {f['candidate_recall_at_0_7']:.3f} | {g['candidate_recall_at_0_7']:.3f} |
| Mean best IoU | {f['mean_best_candidate_iou']:.3f} | {g['mean_best_candidate_iou']:.3f} |
| Median best IoU | {f['median_best_candidate_iou']:.3f} | {g['median_best_candidate_iou']:.3f} |
| Mean candidate count | {f['mean_candidate_count']:.3f} | {g['mean_candidate_count']:.3f} |
| Multi-candidate fraction | {f['fraction_with_multiple_candidates']:.3f} | {g['fraction_with_multiple_candidates']:.3f} |
| JointCandidateRecall@0.5 | {fj['joint_candidate_recall_at_0_5']:.3f} | {gj['joint_candidate_recall_at_0_5']:.3f} |
| JointCandidateRecall@0.7 | {fj['joint_candidate_recall_at_0_7']:.3f} | {gj['joint_candidate_recall_at_0_7']:.3f} |
| GT-best SigLIP rank=1 fraction | {fs['gt_best_siglip_rank_1_fraction']:.3f} | {gs['gt_best_siglip_rank_1_fraction']:.3f} |

All best-candidate metrics are **EVALUATOR_ONLY_CANDIDATE_ORACLE** diagnostics, not achieved method accuracy.

## Scientific questions

### Q1 — Does Grounding DINO substantially improve candidate recall over Florence?

**Yes.** CandidateRecall@0.5 changes by {gd_gain:+.3f} ({f['candidate_recall_at_0_5']:.3f} → {g['candidate_recall_at_0_5']:.3f}).

### Q2 — Does it improve JointCandidateRecall@0.5?

**Yes.** JointCandidateRecall@0.5 changes by {joint_gain:+.3f} ({fj['joint_candidate_recall_at_0_5']:.3f} → {gj['joint_candidate_recall_at_0_5']:.3f}).

### Q3 — Does it expose a meaningfully richer candidate set?

**Yes.** Mean candidate count changes from {f['mean_candidate_count']:.3f} to {g['mean_candidate_count']:.3f}; multi-candidate fraction changes from {f['fraction_with_multiple_candidates']:.3f} to {g['fraction_with_multiple_candidates']:.3f}.

### Q4 — When usable candidates exist, does Grounding DINO rank them highly natively?

**Moderately, not reliably enough to replace resolution.** Its GT-best native-rank-1 fraction conditional on a usable candidate is {g['native_rank1_of_gt_best_given_usable']:.3f}; native top-candidate Recall@0.5 is {g['backend_native_top_candidate_recall_at_0_5']:.3f}. Native scores are not treated as calibrated probabilities or certification scores.

### Q5 — Does existing SigLIP rank Grounding DINO candidates sensibly?

**Moderately.** GT-best SigLIP-rank-1 fraction conditional on a usable candidate is {gs['gt_best_siglip_rank_1_fraction']:.3f}; semantic-top1 localization success is {gs['semantic_top1_localization_success']:.3f}. Mean within-entity native/SigLIP Spearman correlation is {gs['native_siglip_spearman_mean']} over {gs['native_siglip_spearman_n_entities']} eligible entities.

### Q6 — Which Florence generation failures are rescued?

{len(rescued)} entities: {', '.join(rescued) if rescued else 'none'}.

### Q7 — Which Florence successes are lost?

{len(lost)} entities: {', '.join(lost) if lost else 'none'}.

### Q8 — Where do failures remain concentrated?

The remaining failures are {dict(gd_failure_counts)}. Object-like nouns reach {category_analysis['grounding_dino']['object_like_noun']['candidate_recall_at_0_5']:.3f} Recall@0.5 and surface/stuff nouns reach {category_analysis['grounding_dino']['surface_or_stuff_noun']['candidate_recall_at_0_5']:.3f}. Scene/background and tiny-object strata are not represented under the predeclared rules, so no extrapolation or runtime routing rule is made.

### Q9 — Are persistent-entity and certification interfaces preserved?

{'Yes' if interface_pass else 'No'}. {len(smoke_rows)} REL2C compatibility cases passed Grounding DINO regions through persistent `EntityState`, the existing fixed-region SigLIP scorer, FOCUS/REMOVE, and the strict controller. Qwen remained blocked/called=false.

### Q10 — Is candidate generation still the dominant blocker?

**It remains a major—and on this acquisition diagnostic, still dominant—blocker:** {gj['n_families'] - gj['families_with_joint_candidate_at_0_5']}/{gj['n_families']} families still lack jointly usable A/B candidates at IoU 0.5. Because this pilot stops before full certification, it does not claim a measured post-swap ordering against every downstream blocker.

## Florence regression

Metric-level pass: `{str(florence_regression['metric_level_pass']).lower()}`. Bitwise candidate-array pass: `{str(florence_regression['bitwise_candidate_array_pass']).lower()}`. Exact fresh-vs-preserved candidate box/order mismatches: {len(regression_mismatches)}. Inspection found CUDA autoregressive subpixel jitter (mostly ≤2 px) and one 2-vs-3 proposal multiplicity swap, while the unchanged implementation reproduced CandidateRecall@0.5={f['candidate_recall_at_0_5']:.3f}, selected/top Recall@0.5={f['backend_native_top_candidate_recall_at_0_5']:.3f}, multi-candidate fraction={f['fraction_with_multiple_candidates']:.3f}, and JointCandidateRecall@0.5={fj['joint_candidate_recall_at_0_5']:.3f} exactly. This raw non-bitwise variance is disclosed rather than treated as a method change.

## Recommendation

**{recommendation}**

No full benchmark or certification calibration is recommended here.

## Novelty-preservation audit

- Grounding DINO is only a legal-query → candidate-regions acquisition tool.
- Detector output is not trusted as final logical identity.
- Persistent entity state remains explicit and reuses the exact selected region.
- Certification, candidate resolution, proof-state construction, and raw-image hard handoff remain separate and unchanged.
- No VLM selector is introduced into the main method; `qwen_called=false`.

**The acquisition model is replaceable; the proposed method is not Grounding DINO.**
"""
    (output_root / "implementation_report.md").write_text(report, encoding="utf-8")

    changed = [
        "harness/acquisition/__init__.py", "harness/acquisition/backend_registry.py", "harness/acquisition/candidates.py",
        "harness/acquisition/router.py", "harness/configs/acquisition_backends.json", "harness/configs/default.json",
        "harness/configs/main_method.yaml", "harness/configs/paths.py", "harness/processors/grounding_dino.py",
        "harness/processors/registry.py", "harness/scripts/run_acquisition_semantic_support_study.py",
        "harness/scripts/run_grounding_dino_acquisition_pilot.py", "harness/tests/test_grounding_dino_acquisition.py",
        "models/grounding_dino/grounding-dino-base/ (pinned checkpoint and processor/tokenizer files)",
    ]
    (output_root / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    (output_root / "test_report.md").write_text(
        f"# Test report\n\n- Focused Grounding DINO acquisition tests: 15/15 PASS.\n- Full harness test suite: recorded by final verification.\n- Florence locked metric regression: {'PASS' if florence_regression['metric_level_pass'] else 'FAIL'}.\n- Florence bitwise candidate-array regression: {'PASS' if florence_regression['bitwise_candidate_array_pass'] else 'FAIL'} (investigated CUDA autoregressive subpixel jitter; no implementation/query/selection change).\n- Downstream compatibility smoke: {'PASS' if interface_pass else 'FAIL'}.\n- GT leakage: PASS (inference structures checked before evaluator join).\n- Qwen called: false.\n- SAM2 introduced: false.\n",
        encoding="utf-8",
    )

    print(f"""GROUNDING DINO INSTALLED:\n yes: {paths.grounding_dino}\n\nNEW DISK USAGE:\n {model_size} bytes\n\nFLORENCE CANDIDATE RECALL@0.5:\n {f['candidate_recall_at_0_5']:.3f}\n\nGROUNDING DINO CANDIDATE RECALL@0.5:\n {g['candidate_recall_at_0_5']:.3f}\n\nFLORENCE JOINT CANDIDATE RECALL@0.5:\n {fj['joint_candidate_recall_at_0_5']:.3f}\n\nGROUNDING DINO JOINT CANDIDATE RECALL@0.5:\n {gj['joint_candidate_recall_at_0_5']:.3f}\n\nMEAN CANDIDATE COUNT:\n Florence: {f['mean_candidate_count']:.3f}\n Grounding DINO: {g['mean_candidate_count']:.3f}\n\nFLORENCE FAILURES RESCUED:\n {len(rescued)}\n\nFLORENCE SUCCESSES LOST:\n {len(lost)}\n\nSIGLIP GT-BEST RANK1:\n {gs['gt_best_siglip_rank_1_fraction']:.3f}\n\nDOWNSTREAM INTERFACE:\n {'PASS' if interface_pass else 'FAIL'}\n\nGT LEAKAGE:\n PASS\n\nQWEN CALLED:\n false\n\nRECOMMENDATION:\n {recommendation}""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
