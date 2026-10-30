#!/usr/bin/env python3
"""Build a RealBind2-shaped PACO benchmark from candidate families.

Default mode is an annotation-audited benchmark:
  - annotation A/B IoU < 0.1;
  - max bbox containment fraction < 0.5;
  - each bbox area >= 0.5% of image;
  - A/B referents unique without using the queried color:
      * bare category when the category is unique in the image;
      * category + a unique non-color attribute when repeated;
      * exclude the family if no color-free unique description exists;
  - dominant-color semantics (PACO has multi-color objects and dom_color_ids);
  - exact RealBind2 templates are available with --color-semantics color.

Raw candidate mode remains available only for diagnostics:
    --curation-mode raw --color-semantics color
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    atomic_json, canonical_color, code_identity, display_category, ensure_release_on_path,
    iou_xyxy, load_yaml, referent_phrase, render_question, sha256_file, stable_hash,
    write_jsonl_atomic,
)

DOMINANT_COLOR_PROMPT = (
    "What is the dominant color of the main object or region shown?\n"
    "Answer with only one color from this list: red, orange, yellow, green, "
    "blue, purple, pink, brown, black, white, gray."
)
EXACT_REALBIND_COLOR_PROMPT = (
    "What color is the main object or region shown?\n"
    "Answer with only one color from this list: red, orange, yellow, green, "
    "blue, purple, pink, brown, black, white, gray."
)
DOMINANT_TEMPLATES = {
    "REL2C_LEFT": "Between {A} and {B}, what is the dominant color of the one on the LEFT? Answer with one color.",
    "REL2C_RIGHT": "Between {A} and {B}, what is the dominant color of the one on the RIGHT? Answer with one color.",
}


def area_xyxy(b):
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def containment(inner, outer):
    inter = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return inter / area_xyxy(inner) if area_xyxy(inner) > 0 else 0.0


def load_split_annotations(dataset_root: Path, split: str) -> dict:
    path = dataset_root / "annotations" / f"paco_lvis_v1_{split}.json"
    data = json.loads(path.read_text())
    cats = {int(c["id"]): c for c in data["categories"]}
    attrs = {int(a["id"]): a for a in data["attributes"]}
    ranges = data.get("attr_type_to_attr_idxs", {})
    anns = {int(a["id"]): a for a in data["annotations"]}
    objs_by_image = defaultdict(list)
    for a in data["annotations"]:
        if int(a["id"]) == int(a["obj_ann_id"]):
            objs_by_image[int(a["image_id"])].append(a)
    return {"path": path, "sha256": sha256_file(path), "cats": cats, "attrs": attrs,
            "ranges": ranges, "anns": anns, "objs_by_image": objs_by_image}


def unique_referent(ann: dict, split_ann: dict) -> tuple[str | None, dict]:
    """Return a color-free unique referent or None if unavailable."""
    cat = split_ann["cats"][int(ann["category_id"])]["name"]
    cat_display = display_category(cat)
    same_cat = [o for o in split_ann["objs_by_image"][int(ann["image_id"])]
                if int(o["category_id"]) == int(ann["category_id"])]
    if len(same_cat) == 1:
        return referent_phrase(cat), {"basis": "unique_category", "attribute": None}
    pattern_ids = set(split_ann["ranges"].get("pattern_marking", []))
    material_ids = set(split_ann["ranges"].get("material", []))
    transparency_ids = set(split_ann["ranges"].get("transparency", []))
    noncolor_ids = pattern_ids | material_ids | transparency_ids
    my = {int(x) for x in ann.get("attribute_ids", []) if int(x) in noncolor_ids}
    other_union = set()
    for other in same_cat:
        if int(other["id"]) == int(ann["id"]):
            continue
        other_union |= {int(x) for x in other.get("attribute_ids", []) if int(x) in noncolor_ids}
    unique = []
    cat_tokens = set(cat_display.casefold().replace("(", " ").replace(")", " ").split())
    for aid in my - other_union:
        name = split_ann["attrs"][aid]["name"]
        if name.startswith("other(") or name.casefold() in cat_tokens:
            continue
        if aid in material_ids:
            priority = 0
        elif aid in pattern_ids:
            priority = 1
        else:
            priority = 2
        unique.append((priority, name, aid))
    if not unique:
        return None, {"basis": "ambiguous", "attribute": None, "category_count": len(same_cat)}
    _, attr, aid = sorted(unique)[0]
    return f"the {attr} {cat_display}", {"basis": "unique_noncolor_attribute", "attribute": attr, "attribute_id": int(aid), "category_count": len(same_cat)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--candidate-manifest", type=Path,
                    default=Path("/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k/manifests/candidate_families_test_val.jsonl"))
    ap.add_argument("--image-checksums", type=Path,
                    default=Path("/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k/manifests/image_checksums.csv"))
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/root/Qwen3-VL/outputs/paco_realbind2_comparison/benchmark"))
    ap.add_argument("--splits", default="test", help="comma-separated PACO splits, e.g. test or test,val")
    ap.add_argument("--max-families", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--curation-mode", choices=("quality", "raw"), default="quality")
    ap.add_argument("--max-annotation-iou", type=float, default=0.1)
    ap.add_argument("--max-containment", type=float, default=0.5)
    ap.add_argument("--min-bbox-area-ratio", type=float, default=0.005)
    ap.add_argument("--color-semantics", choices=("dominant", "color"), default="dominant")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    release = ensure_release_on_path(Path(paths["release_root"]))
    rows = [json.loads(line) for line in args.candidate_manifest.read_text().splitlines() if line.strip()]
    splits = {x.strip() for x in args.splits.split(",") if x.strip()}
    rows = [r for r in rows if r["paco_split"] in splits]
    split_ann = {s: load_split_annotations(args.dataset_root, s) for s in sorted(splits)}

    image_checks = {}
    if args.image_checksums.is_file():
        import csv
        with args.image_checksums.open(encoding="utf-8", newline="") as f:
            image_checks = {r["image_file"]: r for r in csv.DictReader(f)}

    if args.curation_mode == "quality":
        if args.max_annotation_iou != 0.1 or args.max_containment != 0.5 or args.min_bbox_area_ratio != 0.005:
            raise ValueError("quality curation is frozen to annotation IoU<0.1, containment<0.5, area>=0.005")
    else:
        if args.color_semantics != "color":
            raise ValueError("raw mode must preserve exact RealBind2 color wording")
    if args.color_semantics == "dominant":
        attribute_prompt = DOMINANT_COLOR_PROMPT
        def render_q(qa_type, a, b):
            return DOMINANT_TEMPLATES[qa_type].format(A=a, B=b)
    else:
        attribute_prompt = EXACT_REALBIND_COLOR_PROMPT
        def render_q(qa_type, a, b):
            return render_question(qa_type, a, b)

    curated = []
    audit = []
    for r in rows:
        split = r["paco_split"]
        sa = split_ann[split]
        a = sa["anns"].get(int(r["A_ann_id"]))
        b = sa["anns"].get(int(r["B_ann_id"]))
        if a is None or b is None:
            audit.append({"family_id": r["family_id"], "paco_split": split, "decision": "EXCLUDE", "reason": "missing_annotation"})
            continue
        ax, ay, aw, ah = map(float, a["bbox"]); bx, by, bw, bh = map(float, b["bbox"])
        a_xyxy = [ax, ay, ax + aw, ay + ah]; b_xyxy = [bx, by, bx + bw, by + bh]
        img = next(im for im in json.loads(sa["path"].read_text())["images"] if int(im["id"]) == int(r["image_id"])) if False else None
        # Image dimensions are already in the candidate manifest and checked against the files.
        W, H = float(r["image_width"]), float(r["image_height"])
        inter = max(0.0, min(a_xyxy[2], b_xyxy[2]) - max(a_xyxy[0], b_xyxy[0])) * max(0.0, min(a_xyxy[3], b_xyxy[3]) - max(a_xyxy[1], b_xyxy[1]))
        union = area_xyxy(a_xyxy) + area_xyxy(b_xyxy) - inter
        ann_iou = inter / union if union > 0 else 1.0
        contain_ab = containment(a_xyxy, b_xyxy)
        contain_ba = containment(b_xyxy, a_xyxy)
        area_a = area_xyxy(a_xyxy) / (W * H)
        area_b = area_xyxy(b_xyxy) / (W * H)
        reason = None
        if args.curation_mode == "quality":
            if ann_iou >= args.max_annotation_iou:
                reason = "annotation_iou_ge_threshold"
            elif max(contain_ab, contain_ba) >= args.max_containment:
                reason = "containment_ge_threshold"
            elif area_a < args.min_bbox_area_ratio or area_b < args.min_bbox_area_ratio:
                reason = "bbox_area_below_threshold"
        if reason:
            audit.append({"family_id": r["family_id"], "paco_split": split, "decision": "EXCLUDE", "reason": reason,
                          "ab_annotation_iou": ann_iou, "A_inside_B_fraction": contain_ab,
                          "B_inside_A_fraction": contain_ba, "A_bbox_area_ratio": area_a, "B_bbox_area_ratio": area_b})
            continue
        if args.curation_mode == "quality":
            ref_a, info_a = unique_referent(a, sa)
            ref_b, info_b = unique_referent(b, sa)
            if ref_a is None or ref_b is None:
                audit.append({"family_id": r["family_id"], "paco_split": split, "decision": "EXCLUDE",
                              "reason": "ambiguous_referent_without_color", "A_referent_info": info_a,
                              "B_referent_info": info_b, "ab_annotation_iou": ann_iou})
                continue
        else:
            ref_a = referent_phrase(sa["cats"][int(a["category_id"])]["name"])
            ref_b = referent_phrase(sa["cats"][int(b["category_id"])]["name"])
            info_a = {"basis": "bare_category", "attribute": None}
            info_b = {"basis": "bare_category", "attribute": None}
        ca = canonical_color(sa["attrs"][int(a["dom_color_ids"][0])]["name"])
        cb = canonical_color(sa["attrs"][int(b["dom_color_ids"][0])]["name"])
        if ref_a == ref_b:
            audit.append({"family_id": r["family_id"], "paco_split": split, "decision": "EXCLUDE",
                          "reason": "same_referent", "referent_A": ref_a, "referent_B": ref_b})
            continue
        curated.append({
            "candidate_row": r, "split_annotation": sa, "a": a, "b": b,
            "a_xyxy": a_xyxy, "b_xyxy": b_xyxy, "ann_iou": ann_iou,
            "contain_ab": contain_ab, "contain_ba": contain_ba,
            "area_a": area_a, "area_b": area_b, "ref_a": ref_a, "ref_b": ref_b,
            "info_a": info_a, "info_b": info_b, "color_a": ca, "color_b": cb,
        })
        audit.append({"family_id": r["family_id"], "paco_split": split, "decision": "KEEP", "reason": "retained",
                      "ab_annotation_iou": ann_iou, "A_inside_B_fraction": contain_ab,
                      "B_inside_A_fraction": contain_ba, "A_bbox_area_ratio": area_a,
                      "B_bbox_area_ratio": area_b, "referent_A": ref_a, "referent_B": ref_b,
                      "A_referent_info": info_a, "B_referent_info": info_b})

    if args.max_families is not None:
        if args.max_families <= 0:
            raise ValueError("--max-families must be positive")
        if args.max_families < len(curated):
            rng = random.Random(args.seed)
            buckets = defaultdict(list)
            for r in curated:
                buckets[(r["candidate_row"]["paco_split"], r["candidate_row"]["A_category"])].append(r)
            chosen = []; keys = sorted(buckets)
            while len(chosen) < args.max_families and keys:
                nxt = []
                for key in keys:
                    bucket = buckets[key]
                    if bucket and len(chosen) < args.max_families:
                        chosen.append(bucket.pop(rng.randrange(len(bucket))))
                    if bucket:
                        nxt.append(key)
                keys = nxt
            curated = sorted(chosen, key=lambda x: x["candidate_row"]["family_id"])
    if not curated:
        raise RuntimeError("no curated families")

    families = []
    questions = []
    for item in curated:
        r = item["candidate_row"]
        cs = image_checks.get(r["image_file"])
        if cs is None:
            raise RuntimeError(f"image checksum missing for {r['image_file']}")
        img_path = args.dataset_root / cs["local_path"]
        if not img_path.is_file() or sha256_file(img_path) != cs["sha256"]:
            raise ValueError(f"image/checksum mismatch: {img_path}")
        family = {
            "family_id": r["family_id"],
            "paco_split": r["paco_split"],
            "image_id": r["image_id"],
            "image_relpath": "images/" + r["image_file"],
            "image_sha256": cs["sha256"],
            "width": int(r["image_width"]),
            "height": int(r["image_height"]),
            "referent_A": item["ref_a"], "referent_B": item["ref_b"],
            "referent_A_info": item["info_a"], "referent_B_info": item["info_b"],
            "bbox_A_xyxy": item["a_xyxy"], "bbox_B_xyxy": item["b_xyxy"],
            "color_A": item["color_a"], "color_B": item["color_b"],
            "A_ann_id": r["A_ann_id"], "B_ann_id": r["B_ann_id"],
            "A_category": r["A_category"], "B_category": r["B_category"],
            "annotation_ab_iou": item["ann_iou"],
            "A_inside_B_fraction": item["contain_ab"], "B_inside_A_fraction": item["contain_ba"],
            "A_bbox_area_ratio": item["area_a"], "B_bbox_area_ratio": item["area_b"],
            "horizontal_center_separation": float(r["horizontal_center_separation"]),
            "curation_mode": args.curation_mode,
        }
        left_is_a = item["a_xyxy"][0] + item["a_xyxy"][2] < item["b_xyxy"][0] + item["b_xyxy"][2]
        answers = {"REL2C_LEFT": item["color_a"] if left_is_a else item["color_b"],
                   "REL2C_RIGHT": item["color_b"] if left_is_a else item["color_a"]}
        families.append(family)
        for qa_type in ("REL2C_LEFT", "REL2C_RIGHT"):
            q = render_q(qa_type, item["ref_a"], item["ref_b"])
            questions.append({
                "qa_id": f"{r['family_id']}__{qa_type}",
                "family_id": r["family_id"],
                "paco_split": r["paco_split"],
                "qa_type": qa_type,
                "question": q,
                "answer": answers[qa_type],
                "answer_type": "color",
                "question_sha256": stable_hash(q),
                "image_relpath": "images/" + r["image_file"],
                "image_sha256": cs["sha256"],
                "referent_A": item["ref_a"], "referent_B": item["ref_b"],
            })

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(out / "families_all.jsonl", families)
    write_jsonl_atomic(out / "questions_all.jsonl", questions)
    write_jsonl_atomic(out / "benchmark_all.jsonl", questions)
    write_jsonl_atomic(out / "curation_audit.jsonl", audit)
    # A shorter summary of exclusion reasons.
    decisions = Counter(a["decision"] for a in audit)
    reasons = Counter(a.get("reason") for a in audit if a["decision"] == "EXCLUDE")
    split_counts = Counter(q["paco_split"] for q in questions)
    spec = {
        "schema": "paco-realbind2-benchmark-v2",
        "source": {
            "candidate_manifest": str(args.candidate_manifest.resolve()),
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "image_checksums": str(args.image_checksums.resolve()),
            "image_checksums_sha256": sha256_file(args.image_checksums),
            "release_root": str(release),
            "annotations": {k: v["sha256"] for k, v in split_ann.items()},
        },
        "curation_mode": args.curation_mode,
        "curation": {
            "max_annotation_iou_strictly_less_than": args.max_annotation_iou,
            "max_containment_fraction_strictly_less_than": args.max_containment,
            "min_bbox_area_ratio": args.min_bbox_area_ratio,
            "referent_mode": "unique_category_or_unique_noncolor_attribute" if args.curation_mode == "quality" else "bare_category",
            "question_color_semantics": args.color_semantics,
        },
        "task_semantics": {
            "question_color_semantics": args.color_semantics,
            "attribute_prompt": attribute_prompt,
            "question_templates": DOMINANT_TEMPLATES if args.color_semantics == "dominant" else "imported_exact_realbind2",
        },
        "selection_rule": {"ref_A_ne_ref_B": True, "r4_predicted_A_B_IoU_strictly_less_than": 0.1},
        "candidate_pool": {
            "splits": sorted(splits), "families": len(families), "questions": len(questions),
            "qa_types": dict(sorted(Counter(q["qa_type"] for q in questions).items())),
            "split_counts": dict(sorted(split_counts.items())),
            "unique_images": len({f["image_relpath"] for f in families}),
            "decisions": dict(sorted(decisions.items())),
            "exclusion_reasons": dict(sorted(reasons.items())),
        },
        "files": {
            "families_all.jsonl": sha256_file(out / "families_all.jsonl"),
            "questions_all.jsonl": sha256_file(out / "questions_all.jsonl"),
            "benchmark_all.jsonl": sha256_file(out / "benchmark_all.jsonl"),
            "curation_audit.jsonl": sha256_file(out / "curation_audit.jsonl"),
        },
        "code_identity": code_identity(),
    }
    atomic_json(out / "split_spec.json", spec)
    print(json.dumps(spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
