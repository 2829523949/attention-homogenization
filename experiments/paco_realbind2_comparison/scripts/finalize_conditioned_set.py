#!/usr/bin/env python3
"""Stage 3: freeze selector-conditioned PACO evaluation population.

Strict RealBind2 membership rule:
    ref_A != ref_B  AND  R4-predicted A/B bbox IoU < 0.1
No outcome-based filtering and no annotation-box substitution are allowed.
The annotation boxes remain in the family record only as evaluation metadata;
inference scripts expose referents/questions, never colors or target sides.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import (
    COLOR_PROMPT, atomic_json, code_identity, iou_xyxy, jsonl, load_yaml, r4_protocol, sha256_file, stable_hash,
    write_jsonl_atomic,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--families", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--selected-targets", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=None,
                    help="Must remain 0.1 for the frozen RealBind2 protocol.")
    args = ap.parse_args()
    paths = load_yaml(args.paths)
    threshold = float(paths["conditioning"]["predicted_ab_iou_strictly_less_than"])
    if args.iou_threshold is not None and float(args.iou_threshold) != threshold:
        raise ValueError("conditioning threshold is frozen; edit paths.yaml only for an explicit new protocol")
    if threshold != 0.1:
        raise ValueError("strict RealBind2 reproduction requires predicted A/B IoU < 0.1")
    if not (threshold > 0 and threshold < 1):
        raise ValueError("IoU threshold must be in (0,1)")
    primary_tag = paths['conditioning']['primary_selector_tag']
    selection_identity_path = args.selected_targets.parent / 'identity.json'
    if selection_identity_path.is_file():
        selection_identity = json.loads(selection_identity_path.read_text())
        if selection_identity.get('model_tag') != primary_tag:
            raise ValueError(
                f"conditioning selector tag mismatch: expected {primary_tag}, "
                f"found {selection_identity.get('model_tag')}"
            )
    protocol = r4_protocol()
    source_spec_path = args.questions.parent / "split_spec.json"
    source_spec = json.loads(source_spec_path.read_text()) if source_spec_path.is_file() else {}
    task_semantics = source_spec.get("task_semantics", {
        "question_color_semantics": "color",
        "attribute_prompt": COLOR_PROMPT,
        "question_templates": "unknown",
    })

    families = jsonl(args.families)
    questions = jsonl(args.questions)
    family_by_id = {f["family_id"]: f for f in families}
    q_by_family = {}
    for q in questions:
        q_by_family.setdefault(q["family_id"], []).append(q)
    selected = {r["task_id"]: r for r in jsonl(args.selected_targets)}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    audit = []
    kept_families = []
    kept_ids = set()
    stats = Counter()
    for family in families:
        fid = family["family_id"]
        stats["candidate_families"] += 1
        a = selected.get(f"{fid}|A")
        b = selected.get(f"{fid}|B")
        if a is None or b is None:
            stats["excluded_missing_selection"] += 1
            audit.append({"family_id": fid, "ref_A": family["referent_A"], "ref_B": family["referent_B"],
                          "decision": "EXCLUDE", "reason": "missing_selection"})
            continue
        if a.get("status") != "ok" or b.get("status") != "ok":
            stats["excluded_selection_not_ok"] += 1
            audit.append({"family_id": fid, "ref_A": family["referent_A"], "ref_B": family["referent_B"],
                          "selection_status_A": a.get("status"), "selection_status_B": b.get("status"),
                          "decision": "EXCLUDE", "reason": "selection_not_ok"})
            continue
        if a.get("selection_model_id") != b.get("selection_model_id"):
            raise ValueError(f"mixed selector models for {fid}: {a.get('selection_model_id')} vs {b.get('selection_model_id')}")
        box_a = a.get("selected_bbox_xyxy")
        box_b = b.get("selected_bbox_xyxy")
        if not box_a or not box_b:
            stats["excluded_missing_bbox"] += 1
            audit.append({"family_id": fid, "ref_A": family["referent_A"], "ref_B": family["referent_B"],
                          "decision": "EXCLUDE", "reason": "missing_selected_bbox"})
            continue
        same_description = family["referent_A"] == family["referent_B"]
        ab_iou = iou_xyxy([float(v) for v in box_a], [float(v) for v in box_b])
        geometry = protocol["geometry"](box_a, box_b, int(family["width"]), protocol["EPSILON"])
        passed = (not same_description) and (ab_iou < threshold)
        reason = "retained" if passed else ("same_description" if same_description else "predicted_ab_iou_ge_threshold")
        audit_row = {
            "family_id": fid,
            "paco_split": family["paco_split"],
            "ref_A": family["referent_A"],
            "ref_B": family["referent_B"],
            "same_description": same_description,
            "selected_A_bbox_xyxy": box_a,
            "selected_B_bbox_xyxy": box_b,
            "ab_bbox_iou": ab_iou,
            "geometry_status": geometry["status"],
            "geometry_valid": geometry["status"] == "VALID",
            "normalized_center_difference": geometry.get("normalized_center_difference"),
            "iou_threshold_strictly_less_than": threshold,
            "decision": "KEEP" if passed else "EXCLUDE",
            "reason": reason,
            "selection_model_id_A": a.get("selection_model_id"),
            "selection_status_A": a.get("qwen_status"),
            "selection_status_B": b.get("qwen_status"),
        }
        audit.append(audit_row)
        if passed:
            stats["retained_families"] += 1
            kept_ids.add(fid)
            kept = dict(family)
            kept.update({
                "r4_selected_bbox_A_xyxy": box_a,
                "r4_selected_bbox_B_xyxy": box_b,
                "r4_predicted_ab_iou": ab_iou,
                "r4_geometry_status": geometry["status"],
                "r4_geometry_valid": geometry["status"] == "VALID",
                "conditioning_rule": {
                    "ref_A_ne_ref_B": True,
                    "r4_predicted_ab_iou_strictly_less_than": threshold,
                },
            })
            kept_families.append(kept)
    stats["candidate_questions"] = len(questions)
    kept_questions = [q for q in questions if q["family_id"] in kept_ids]
    stats["retained_questions"] = len(kept_questions)
    stats["excluded_questions"] = stats["candidate_questions"] - stats["retained_questions"]
    stats["retained_unique_images"] = len({f["image_relpath"] for f in kept_families})
    if stats["retained_families"] == 0:
        raise RuntimeError("conditioned set is empty; inspect selection/candidate outputs")

    write_jsonl_atomic(out / "families_conditioned.jsonl", sorted(kept_families, key=lambda r: r["family_id"]))
    write_jsonl_atomic(out / "questions_conditioned.jsonl", sorted(kept_questions, key=lambda r: r["qa_id"]))
    write_jsonl_atomic(out / "benchmark.jsonl", sorted(kept_questions, key=lambda r: r["qa_id"]))
    write_jsonl_atomic(out / "family_conditioning_audit.jsonl", sorted(audit, key=lambda r: r["family_id"]))
    audit_by_family = {r["family_id"]: r for r in audit}
    qa_audit = []
    for q in kept_questions:
        a = audit_by_family[q["family_id"]]
        qa_audit.append({
            "qa_id": q["qa_id"],
            "family_id": q["family_id"],
            "paco_split": q["paco_split"],
            "ref_A": a["ref_A"],
            "ref_B": a["ref_B"],
            "same_description": a["same_description"],
            "ab_bbox_iou": a["ab_bbox_iou"],
            "geometry_status": a["geometry_status"],
            "geometry_valid": a["geometry_valid"],
            "normalized_center_difference": a.get("normalized_center_difference"),
            "iou_threshold_strictly_less_than": threshold,
            "decision": "KEEP",
            "reason": "retained",
        })
    write_jsonl_atomic(out / "selection_audit.jsonl", sorted(qa_audit, key=lambda r: r["qa_id"]))
    spec = {
        "schema": "paco-realbind2-conditioned-v1",
        "scope": "PACO-LVIS; retrospectively method-conditioned by the primary R4 selector; not human-curated",
        "selection_rule": {"ref_A_ne_ref_B": True, "r4_predicted_A_B_IoU_strictly_less_than": threshold},
        "primary_selector_tag": paths["conditioning"]["primary_selector_tag"],
        "counts": {
            "candidate_families": stats["candidate_families"],
            "retained_families": stats["retained_families"],
            "candidate_questions": stats["candidate_questions"],
            "retained_questions": stats["retained_questions"],
            "retained_unique_images": stats["retained_unique_images"],
            "excluded_missing_selection": stats["excluded_missing_selection"],
            "excluded_selection_not_ok": stats["excluded_selection_not_ok"],
            "excluded_missing_bbox": stats["excluded_missing_bbox"],
        },
        "qa_type_counts": dict(sorted(Counter(q["qa_type"] for q in kept_questions).items())),
        "split_counts": dict(sorted(Counter(q["paco_split"] for q in kept_questions).items())),
        "task_semantics": task_semantics,
        "files": {
            "families_conditioned.jsonl": sha256_file(out / "families_conditioned.jsonl"),
            "questions_conditioned.jsonl": sha256_file(out / "questions_conditioned.jsonl"),
            "benchmark.jsonl": sha256_file(out / "benchmark.jsonl"),
            "selection_audit.jsonl": sha256_file(out / "selection_audit.jsonl"),
            "family_conditioning_audit.jsonl": sha256_file(out / "family_conditioning_audit.jsonl"),
        },
        "sources": {
            "families_sha256": sha256_file(args.families),
            "questions_sha256": sha256_file(args.questions),
            "selected_targets_sha256": sha256_file(args.selected_targets),
        },
        "code_identity": code_identity(),
    }
    atomic_json(out / "split_spec.json", spec)
    print(json.dumps(spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
