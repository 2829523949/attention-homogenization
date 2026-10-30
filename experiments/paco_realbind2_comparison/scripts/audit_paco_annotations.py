#!/usr/bin/env python3
"""Audit PACO candidate-family annotations before selector/model runs.

Checks the assumptions of the PACO RealBind2 runner:
  - A/B annotation identity and split consistency;
  - whether bare category referents are unique in the image;
  - annotation-box overlap/containment;
  - tiny/border boxes and out-of-image coordinates;
  - dominant-color and color-attribute ambiguity;
  - other annotated objects overlapping A/B.

This is an audit, not an outcome-based filter. It writes flags so a benchmark
version can pre-register a stricter inclusion rule if needed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import atomic_json, canonical_color, display_category, iou_xyxy, jsonl, sha256_file, write_jsonl_atomic


def bbox_xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return [x, y, x + w, y + h]


def area_xyxy(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(inner, outer):
    inter = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return inter / area_xyxy(inner) if area_xyxy(inner) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest", type=Path,
                    default=Path("/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k/manifests/candidate_families_test_val.jsonl"))
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/root/Qwen3-VL/outputs/paco_realbind2_comparison/annotation_audit"))
    args = ap.parse_args()
    root = args.dataset_root.resolve()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = jsonl(args.candidate_manifest)
    split_data = {}
    for split in sorted({r["paco_split"] for r in rows}):
        ann_path = root / "annotations" / f"paco_lvis_v1_{split}.json"
        data = json.loads(ann_path.read_text())
        cats = {int(c["id"]): c for c in data["categories"]}
        attrs = {int(a["id"]): a["name"] for a in data["attributes"]}
        color_ids = set(data.get("attr_type_to_attr_idxs", {}).get("color", []))
        annotations = {}
        obj_by_image = defaultdict(list)
        for a in data["annotations"]:
            annotations[int(a["id"])] = a
            if int(a["id"]) == int(a["obj_ann_id"]):
                obj_by_image[int(a["image_id"])].append(a)
        images = {int(im["id"]): im for im in data["images"]}
        split_data[split] = {"cats": cats, "attrs": attrs, "color_ids": color_ids,
                             "annotations": annotations, "obj_by_image": obj_by_image,
                             "images": images, "sha256": sha256_file(ann_path)}

    audit = []
    for r in rows:
        split = r["paco_split"]; sd = split_data[split]
        img = sd["images"][int(r["image_id"])]
        a = sd["annotations"].get(int(r["A_ann_id"]))
        b = sd["annotations"].get(int(r["B_ann_id"]))
        flags = []
        if a is None or b is None:
            audit.append({"family_id": r["family_id"], "paco_split": split,
                          "flags": ["missing_annotation"], "decision": "EXCLUDE"})
            continue
        cat_a = sd["cats"][int(a["category_id"])]["name"]
        cat_b = sd["cats"][int(b["category_id"])]["name"]
        box_a = bbox_xywh_to_xyxy(a["bbox"]); box_b = bbox_xywh_to_xyxy(b["bbox"])
        W, H = float(img["width"]), float(img["height"])
        inter = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        union = area_xyxy(box_a) + area_xyxy(box_b) - inter
        ab_iou = inter / union if union > 0 else 1.0
        contain_ab = containment(box_a, box_b)  # fraction of A inside B
        contain_ba = containment(box_b, box_a)
        area_a = area_xyxy(box_a) / (W * H)
        area_b = area_xyxy(box_b) / (H * W)
        objs = sd["obj_by_image"][int(r["image_id"])]
        cat_counts = Counter(sd["cats"][int(o["category_id"])]["name"] for o in objs)
        count_a = cat_counts[cat_a]; count_b = cat_counts[cat_b]
        if count_a > 1: flags.append("A_category_duplicate")
        if count_b > 1: flags.append("B_category_duplicate")
        if area_a < 0.005: flags.append("A_bbox_lt_0.5pct")
        if area_b < 0.005: flags.append("B_bbox_lt_0.5pct")
        if ab_iou >= 0.1: flags.append("annotation_iou_ge_0.1")
        if ab_iou >= 0.3: flags.append("annotation_iou_ge_0.3")
        if ab_iou >= 0.5: flags.append("annotation_iou_ge_0.5")
        if contain_ab >= 0.5 and contain_ab > contain_ba: flags.append("B_contains_A_ge_0.5")
        if contain_ba >= 0.5 and contain_ba > contain_ab: flags.append("A_contains_B_ge_0.5")
        for ref, box in (("A", box_a), ("B", box_b)):
            if box[0] <= 1 or box[1] <= 1 or box[2] >= W - 1 or box[3] >= H - 1:
                flags.append(f"{ref}_bbox_touches_border")
            if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                flags.append(f"{ref}_bbox_out_of_bounds")
        # Other object boxes overlapping each referent.
        for ref, box in (("A", box_a), ("B", box_b)):
            n = 0
            for other in objs:
                if int(other["id"]) in {int(a["id"]), int(b["id"])}:
                    continue
                if iou_xyxy(box, bbox_xywh_to_xyxy(other["bbox"])) >= 0.5:
                    n += 1
            if n:
                flags.append(f"{ref}_other_iou_ge_0.5")
        # Color-label ambiguity checks.
        for ref, ann in (("A", a), ("B", b)):
            dom = [int(x) for x in ann.get("dom_color_ids", [])]
            all_color = [int(x) for x in ann.get("attribute_ids", []) if int(x) in sd["color_ids"]]
            if len(dom) != 1: flags.append(f"{ref}_dom_color_count_{len(dom)}")
            if len(all_color) > 1: flags.append(f"{ref}_multi_color_attributes")
            if int(ann.get("unknown_color", 0)) != 0: flags.append(f"{ref}_unknown_color")
            if int(dom[0]) == 29 if dom else False: flags.append(f"{ref}_other_color")
        same_referent = display_category(cat_a) == display_category(cat_b)
        if same_referent: flags.append("same_display_referent")
        # A stricter benchmark candidate subset can be defined from these flags.
        strict_unique = count_a == 1 and count_b == 1
        strict_low_overlap = ab_iou < 0.1
        strict_area = area_a >= 0.005 and area_b >= 0.005
        strict_color = not any("multi_color" in x or "dom_color_count" in x or "unknown_color" in x or "other_color" in x for x in flags)
        strict = strict_unique and strict_low_overlap and strict_area and strict_color
        audit.append({
            "family_id": r["family_id"], "paco_split": split, "image_id": r["image_id"], "image_file": r["image_file"],
            "width": W, "height": H, "A_ann_id": int(a["id"]), "B_ann_id": int(b["id"]),
            "A_category": cat_a, "B_category": cat_b,
            "A_color_raw": sd["attrs"][int(a["dom_color_ids"][0])] if a.get("dom_color_ids") else None,
            "B_color_raw": sd["attrs"][int(b["dom_color_ids"][0])] if b.get("dom_color_ids") else None,
            "A_bbox_xyxy": box_a, "B_bbox_xyxy": box_b,
            "A_bbox_area_ratio": area_a, "B_bbox_area_ratio": area_b,
            "ab_annotation_iou": ab_iou, "A_inside_B_fraction": contain_ab, "B_inside_A_fraction": contain_ba,
            "horizontal_center_separation": r["horizontal_center_separation"],
            "A_category_count_in_image": count_a, "B_category_count_in_image": count_b,
            "n_object_annotations_in_image": len(objs),
            "referent_unique_by_category": strict_unique,
            "flags": sorted(set(flags)),
            "strict_high_quality_candidate": bool(strict),
        })
    audit.sort(key=lambda x: x["family_id"])
    write_jsonl_atomic(out / "family_annotation_audit.jsonl", audit)
    with (out / "family_annotation_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        w.writeheader()
        for row in audit:
            rr = dict(row)
            rr["flags"] = ";".join(rr["flags"])
            rr["A_bbox_xyxy"] = json.dumps(rr["A_bbox_xyxy"])
            rr["B_bbox_xyxy"] = json.dumps(rr["B_bbox_xyxy"])
            w.writerow(rr)
    summary = {"schema": "paco-annotation-audit-v1", "candidate_manifest": str(args.candidate_manifest.resolve()),
               "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
               "source_annotations": {k: v["sha256"] for k, v in split_data.items()}, "by_split": {},
               "all_candidates": len(audit), "strict_high_quality_candidates": sum(x.get("strict_high_quality_candidate", False) for x in audit)}
    for split in sorted({x["paco_split"] for x in audit}):
        subset = [x for x in audit if x["paco_split"] == split]
        flag_counts = Counter(flag for x in subset for flag in x.get("flags", []))
        summary["by_split"][split] = {
            "candidates": len(subset),
            "unique_images": len({x["image_id"] for x in subset}),
            "referent_unique_by_category": sum(x.get("referent_unique_by_category", False) for x in subset),
            "annotation_iou_ge_0.1": sum(x.get("ab_annotation_iou", 0) >= 0.1 for x in subset),
            "annotation_iou_ge_0.3": sum(x.get("ab_annotation_iou", 0) >= 0.3 for x in subset),
            "annotation_iou_ge_0.5": sum(x.get("ab_annotation_iou", 0) >= 0.5 for x in subset),
            "A_contains_B_ge_0.5": sum(x.get("A_inside_B_fraction", 0) >= 0.5 or x.get("B_inside_A_fraction", 0) >= 0.5 for x in subset),
            "A_bbox_lt_0.5pct": sum(x.get("A_bbox_area_ratio", 1) < 0.005 for x in subset),
            "B_bbox_lt_0.5pct": sum(x.get("B_bbox_area_ratio", 1) < 0.005 for x in subset),
            "strict_high_quality_candidates": sum(x.get("strict_high_quality_candidate", False) for x in subset),
            "flag_counts": dict(sorted(flag_counts.items())),
        }
    atomic_json(out / "annotation_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
