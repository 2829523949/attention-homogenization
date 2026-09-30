from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .gates import COLORS, geometry_gate, normalize_color_labels
from .referents import construct_referent, leakage_reasons
from .state_db import connect, event


def _json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): digest.update(chunk)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _canon_category(x: str) -> str:
    text = " ".join(str(x).strip().lower().split())
    aliases = {"people": "person", "men": "man", "women": "woman", "children": "child", "shorts": "shorts", "leaves": "leaf", "buses": "bus", "glasses": "glass"}
    if text in aliases: return aliases[text]
    if len(text) > 4 and text.endswith("s") and not text.endswith(("ss", "us", "is")): return text[:-1]
    return text


def _candidate_id(image_id: str, a: dict, b: dict) -> str:
    first, second = sorted((a["instance_id"], b["instance_id"]))
    return f"vg{image_id}_{first}_{second}"


def mine(root: Path, source_root: Path, limit_images: int | None = None, force: bool = False) -> dict:
    """Metadata-only mining over local VAW and VG annotations."""
    prior = root / "manifests" / "mine_summary.json"
    if prior.is_file() and (root / "state" / "construction_state.sqlite").is_file() and not force:
        existing = json.loads(prior.read_text(encoding="utf-8"))
        existing["resume"] = "existing mining state preserved; use a new root to change protocol"
        return existing
    vaw_root = source_root / "vaw" / "data"
    vg_root = source_root / "vg" / "annotations"
    image_data = {str(r["image_id"]): r for r in _json(vg_root / "image_data.json") if r.get("image_id") is not None}
    vg_records = _json(vg_root / "objects.json")
    vg_by_image = {str(r["image_id"]): r for r in vg_records if r.get("image_id") is not None}
    dev_file = source_root / "realbind" / "families" / "dryrun_100_pair_direct_v1_1_frozen.jsonl"
    dev_ids = set()
    if dev_file.is_file():
        for line in dev_file.open(encoding="utf-8"):
            if line.strip(): dev_ids.add(str(json.loads(line)["image_id"]))
    rows = []
    object_reject_counts = Counter()
    raw_counts = Counter()
    for name in ("train_part1.json", "train_part2.json", "val.json", "test.json"):
        path = vaw_root / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f: records = json.load(f)
        raw_counts[name] = len(records)
        for rec in records:
            image_id = str(rec.get("image_id", ""))
            if not image_id or image_id in dev_ids: continue
            im = image_data.get(image_id)
            if not im:
                object_reject_counts["DIMENSION_MISMATCH"] += 1
                continue
            attrs = rec.get("positive_attributes") or []
            color, reason = normalize_color_labels(attrs)
            if reason:
                object_reject_counts[reason] += 1
                continue
            bbox = rec.get("instance_bbox")
            try: bbox = [float(v) for v in bbox]
            except (TypeError, ValueError):
                object_reject_counts["INVALID_BBOX"] += 1
                continue
            rows.append({"image_id": image_id, "instance_id": str(rec.get("instance_id", "")),
                         "bbox": bbox, "category": _canon_category(rec.get("object_name", "")),
                         "color": color, "attrs": [str(v) for v in attrs], "source_split": name.removesuffix(".json"),
                         "width": int(im["width"]), "height": int(im["height"]), "url": im.get("url")})
    by_image = defaultdict(list)
    for row in rows: by_image[row["image_id"]].append(row)
    all_families, reject_counts, image_payloads, pair_reject_rows = [], Counter(), {}, []
    for image_id in sorted(by_image, key=lambda x: (int(x) if x.isdigit() else 0, x)):
        objects = by_image[image_id]
        instance_counts = Counter(o["instance_id"] for o in objects)
        width, height = objects[0]["width"], objects[0]["height"]
        vg = vg_by_image.get(image_id, {}).get("objects", [])
        vg_by_id = defaultdict(list)
        vg_name_counts = Counter()
        for o in vg: vg_by_id[str(o.get("object_id"))].append(o)
        for o in vg:
            for name in o.get("names", []) if isinstance(o.get("names", []), list) else [o.get("names", "")]:
                vg_name_counts[_canon_category(name)] += 1
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                if a["instance_id"] == b["instance_id"] or instance_counts[a["instance_id"]] > 1 or instance_counts[b["instance_id"]] > 1:
                    reject_counts["DUPLICATE_OBJECT_ANNOTATION"] += 1
                    pair_reject_rows.append({"candidate_family_id": _candidate_id(image_id, a, b), "image_id": image_id, "object_A_id": a["instance_id"], "object_B_id": b["instance_id"], "bbox_A": a["bbox"], "bbox_B": b["bbox"], "category_A": a["category"], "category_B": b["category"], "color_A": a["color"], "color_B": b["color"], "reject_reasons": ["DUPLICATE_OBJECT_ANNOTATION"]}); continue
                if a["color"] == b["color"]:
                    reject_counts["SAME_TARGET_ATTRIBUTE"] += 1
                    pair_reject_rows.append({"candidate_family_id": _candidate_id(image_id, a, b), "image_id": image_id, "object_A_id": a["instance_id"], "object_B_id": b["instance_id"], "bbox_A": a["bbox"], "bbox_B": b["bbox"], "category_A": a["category"], "category_B": b["category"], "color_A": a["color"], "color_B": b["color"], "reject_reasons": ["SAME_TARGET_ATTRIBUTE"]}); continue
                g, reasons = geometry_gate(a["bbox"], b["bbox"], width, height)
                if reasons:
                    reject_counts.update(reasons)
                    pair_reject_rows.append({"candidate_family_id": _candidate_id(image_id, a, b), "image_id": image_id, "object_A_id": a["instance_id"], "object_B_id": b["instance_id"], "bbox_A": a["bbox"], "bbox_B": b["bbox"], "category_A": a["category"], "category_B": b["category"], "color_A": a["color"], "color_B": b["color"], "geometry_metrics": g, "reject_reasons": reasons}); continue
                if not vg_name_counts[a["category"]] or not vg_name_counts[b["category"]]:
                    reject_counts["ANNOTATION_CONFLICT"] += 1
                    pair_reject_rows.append({"candidate_family_id": _candidate_id(image_id, a, b), "image_id": image_id, "object_A_id": a["instance_id"], "object_B_id": b["instance_id"], "bbox_A": a["bbox"], "bbox_B": b["bbox"], "category_A": a["category"], "category_B": b["category"], "color_A": a["color"], "color_B": b["color"], "reject_reasons": ["ANNOTATION_CONFLICT"]}); continue
                attrs_a = [x for x in a["attrs"] if x.strip().lower() not in COLORS]
                attrs_b = [x for x in b["attrs"] if x.strip().lower() not in COLORS]
                a_count = sum(1 for o in objects if o["category"] == a["category"])
                b_count = sum(1 for o in objects if o["category"] == b["category"])
                ref_a, uniqueness_a = construct_referent(a["category"], attrs_a, a_count)
                ref_b, uniqueness_b = construct_referent(b["category"], attrs_b, b_count)
                leakage = leakage_reasons(ref_a, a["color"]) + leakage_reasons(ref_b, b["color"])
                if leakage:
                    reject_counts.update(leakage)
                    pair_reject_rows.append({"candidate_family_id": _candidate_id(image_id, a, b), "image_id": image_id, "object_A_id": a["instance_id"], "object_B_id": b["instance_id"], "bbox_A": a["bbox"], "bbox_B": b["bbox"], "category_A": a["category"], "category_B": b["category"], "color_A": a["color"], "color_B": b["color"], "referent_A": ref_a, "referent_B": ref_b, "reject_reasons": sorted(set(leakage))}); continue
                # Canonical endpoint assignment is deterministic and independent of side.
                A, B = (a, b) if a["instance_id"] < b["instance_id"] else (b, a)
                ra, rb = (ref_a, ref_b) if A is a else (ref_b, ref_a)
                status_a, status_b = (uniqueness_a, uniqueness_b) if A is a else (uniqueness_b, uniqueness_a)
                ga, gr = geometry_gate(A["bbox"], B["bbox"], width, height)
                cid = f"vg{image_id}_{A['instance_id']}_{B['instance_id']}"
                family = {"candidate_family_id": cid, "image_id": image_id, "source": "VAW+VisualGenome",
                          "source_split": sorted({A["source_split"], B["source_split"]}), "width": width, "height": height,
                          "object_A_id": A["instance_id"], "object_B_id": B["instance_id"], "bbox_A": A["bbox"], "bbox_B": B["bbox"],
                          "category_A": A["category"], "category_B": B["category"], "referent_A": ra, "referent_B": rb,
                          "referent_status_A": status_a, "referent_status_B": status_b,
                          "vg_category_count_A": vg_name_counts[A["category"]], "vg_category_count_B": vg_name_counts[B["category"]],
                          "color_A": A["color"], "color_B": B["color"], "same_category": A["category"] == B["category"],
                          "geometry_metrics": ga, "url": A["url"], "metadata_gate_results": {"M1_bbox": "PASS", "M2_color": "PASS", "M3_distinct_color": "PASS", "M4_size": "PASS", "M5_geometry": "PASS", "M6_overlap": "PASS", "M7_annotation_consistency": "PASS", "referent_leakage": "PASS", "referent_uniqueness": "HUMAN_UNIQUENESS_CHECK_REQUIRED"},
                          "expected_download_source": A["url"], "protocol_version": "RealBind-v2.0-construction-pre1"}
                all_families.append(family)
        # metadata-only image manifest includes all valid pairs and source URL.
        image_payloads[image_id] = {"image_id": image_id, "source": "Visual Genome", "width": width, "height": height,
                                    "url": objects[0]["url"], "expected_download_source": objects[0]["url"], "candidate_family_ids": [], "selected_family_ids": [], "protocol_version": "RealBind-v2.0-construction-pre1"}
    # rank and cap at two pairs/image, without model data; geometry, area, and annotation completeness only.
    dedup = {}
    for fam in all_families:
        prior_fam = dedup.get(fam["candidate_family_id"])
        if prior_fam is None:
            dedup[fam["candidate_family_id"]] = fam
        else:
            reject_counts["DUPLICATE_FAMILY"] += 1
    all_families = list(dedup.values())
    fam_by_image = defaultdict(list)
    for fam in all_families: fam_by_image[fam["image_id"]].append(fam)
    selected = []
    for image_id, fs in sorted(fam_by_image.items()):
        fs.sort(key=lambda f: (-f["geometry_metrics"]["delta_x"],
                               -min(f["geometry_metrics"]["area_ratio_A"], f["geometry_metrics"]["area_ratio_B"]),
                               f["geometry_metrics"]["iou"], f["same_category"], f["candidate_family_id"]))
        kept = fs[:2]
        selected.extend(kept)
        image_payloads[image_id]["candidate_family_ids"] = [f["candidate_family_id"] for f in fs]
        image_payloads[image_id]["selected_family_ids"] = [f["candidate_family_id"] for f in kept]
        image_payloads[image_id]["all_metadata_valid_family_count"] = len(fs)
        image_payloads[image_id]["family_ranking_rule"] = "descending center separation; descending smaller-object area ratio; ascending IoU; deterministic category-diversity tie break; ascending candidate_family_id; cap 2/image"
    # Avoid all-results RAM after this point by transactional inserts and streamed exports.
    # Persist image-level rejects too, so state/status statistics cover the
    # complete VG metadata inventory and no failed image is silently omitted.
    candidate_image_ids = set(fam_by_image)
    for image_id, im in image_data.items():
        if image_id not in image_payloads:
            image_payloads[image_id] = {"image_id": image_id, "source": "Visual Genome", "width": int(im.get("width", 0)),
                                        "height": int(im.get("height", 0)), "url": im.get("url"), "expected_download_source": im.get("url"), "candidate_family_ids": [], "selected_family_ids": [],
                                        "all_metadata_valid_family_count": 0, "metadata_reject_reason": "NO_VALID_PAIR",
                                        "protocol_version": "RealBind-v2.0-construction-pre1"}
    root.mkdir(parents=True, exist_ok=True)
    db = connect(root / "state" / "construction_state.sqlite")
    with db:
        for image_id, data in sorted(image_payloads.items()):
            status = "PENDING_DOWNLOAD" if data["selected_family_ids"] else "METADATA_REJECTED"
            db.execute("INSERT OR IGNORE INTO images(image_id,source,width,height,url,status,metadata_json) VALUES(?,?,?,?,?,?,?)",
                       (image_id, data["source"], data["width"], data["height"], data["url"], status, json.dumps(data, sort_keys=True)))
            db.execute("UPDATE images SET metadata_json=? WHERE image_id=?", (json.dumps(data, sort_keys=True), image_id))
            for fam in fam_by_image.get(image_id, []):
                selected_flag = fam["candidate_family_id"] in data["selected_family_ids"]
                if selected_flag:
                    db.execute("INSERT OR IGNORE INTO families(candidate_family_id,image_id,status,payload_json) VALUES(?,?,?,?)",
                               (fam["candidate_family_id"], image_id, "METADATA_VALID", json.dumps(fam, sort_keys=True)))
                    db.execute("UPDATE families SET payload_json=? WHERE candidate_family_id=?", (json.dumps(fam, sort_keys=True), fam["candidate_family_id"]))
        db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('mining_protocol',?)", ("RealBind-v2.0-construction-pre1",))
        event(db, None, None, "METADATA_MINED", {"source_record_counts": dict(raw_counts), "candidate_pairs_before_image_cap": len(all_families), "candidate_families": len(selected), "reject_counts": dict(reject_counts), "v1_1_dev_image_count_excluded": len(dev_ids)})
    manifest = (root / "manifests" / "metadata_candidate_images.jsonl")
    _jsonl(manifest, (image_payloads[k] for k in sorted(image_payloads) if image_payloads[k]["candidate_family_ids"]))
    fam_path = root / "manifests" / "metadata_candidate_families.jsonl"
    selected_ids = {f["candidate_family_id"] for f in selected}
    _jsonl(fam_path, ({**f, "selected_for_stream": f["candidate_family_id"] in selected_ids} for f in all_families))
    _jsonl(root / "manifests" / "metadata_pair_rejects.jsonl", pair_reject_rows)
    # A separate immutable metadata snapshot provides direct SQLite queries
    # without mixing candidate metadata with the mutable construction state.
    meta_db_path = root / "metadata" / "metadata_candidates.sqlite"
    meta_db_path.parent.mkdir(parents=True, exist_ok=True)
    meta_db = sqlite3.connect(meta_db_path)
    with meta_db:
        meta_db.executescript("CREATE TABLE IF NOT EXISTS candidate_images(image_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS candidate_families(candidate_family_id TEXT PRIMARY KEY,image_id TEXT NOT NULL,selected_for_stream INTEGER NOT NULL,payload_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS pair_rejects(candidate_family_id TEXT PRIMARY KEY,image_id TEXT NOT NULL,reject_reasons TEXT NOT NULL,payload_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS mining_summary(key TEXT PRIMARY KEY,value TEXT NOT NULL);")
        meta_db.execute("DELETE FROM candidate_images"); meta_db.execute("DELETE FROM candidate_families"); meta_db.execute("DELETE FROM mining_summary")
        meta_db.executemany("INSERT INTO candidate_images VALUES(?,?)", ((k, json.dumps(v, sort_keys=True)) for k,v in image_payloads.items() if v["candidate_family_ids"]))
        meta_db.executemany("INSERT INTO candidate_families VALUES(?,?,?,?)", ((f["candidate_family_id"],f["image_id"],int(f["candidate_family_id"] in selected_ids),json.dumps(f,sort_keys=True)) for f in all_families))
        meta_db.execute("DELETE FROM pair_rejects")
        meta_db.executemany("INSERT OR REPLACE INTO pair_rejects VALUES(?,?,?,?)", ((r["candidate_family_id"],r["image_id"],json.dumps(r["reject_reasons"]),json.dumps(r,sort_keys=True)) for r in pair_reject_rows))
    source_files = [p for p in (vaw_root / "train_part1.json", vaw_root / "train_part2.json", vaw_root / "val.json", vaw_root / "test.json", vg_root / "image_data.json", vg_root / "objects.json") if p.is_file()]
    summary = {"protocol_version": "RealBind-v2.0-construction-pre1", "source_record_counts": dict(raw_counts),
               "source_color_eligible_instances": len(rows), "metadata_candidate_pairs_before_image_cap": len(all_families),
               "metadata_candidate_families": len(selected), "candidate_images": sum(bool(x["candidate_family_ids"]) for x in image_payloads.values()),
               "metadata_total_images": len(image_data), "metadata_rejected_images": len(image_data) - sum(bool(x["candidate_family_ids"]) for x in image_payloads.values()),
               "vg_object_annotation_records": sum(len(r.get("objects", [])) for r in vg_records), "source_fingerprints": [_fingerprint(p) for p in source_files],
               "v1_1_dev_image_count_excluded": len(dev_ids), "reject_counts": dict(reject_counts), "no_vlm_performance_used": True,
               "object_reject_counts": dict(object_reject_counts), "metadata_pair_reject_records": len(pair_reject_rows),
               "same_category_families": sum(bool(f["same_category"]) for f in selected),
               "cross_category_families": sum(not f["same_category"] for f in selected),
               "color_endpoint_distribution": dict(Counter(f[c] for f in selected for c in ("color_A", "color_B"))),
               "left_is_A_distribution": {"true": sum(bool(f["geometry_metrics"]["left_is_A"]) for f in selected), "false": sum(not f["geometry_metrics"]["left_is_A"] for f in selected)},
               "referent_uniqueness": "HUMAN_UNIQUENESS_CHECK_REQUIRED for every candidate"}
    with meta_db:
        meta_db.execute("INSERT INTO mining_summary VALUES('summary_json',?)", (json.dumps(summary, sort_keys=True),))
    meta_db.close()
    (root / "manifests" / "mine_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
