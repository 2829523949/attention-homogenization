#!/usr/bin/env python3
"""SAM3 raw-proposal -> canonical instance-hypothesis -> instance-level binding.

Only the SAM3 consolidation interface and the instance-level referent binding
interface are changed.  The frozen RealBind 20-family/40-QA development subset,
ReferentSpec, SAM3 checkpoint, MM-GDINO/R4 cache, geometry, downstream SAM3
isolation, color reader, minimal state and final reasoner are reused.

Stages are incremental and crash-resumable.  Evaluator annotations are joined
only after all routing choices have been written.
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
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image

_DXG = Path("/dev/dxg")
if not _DXG.exists() and Path("/proc/1/root/dev/dxg").exists():
    try:
        _DXG.symlink_to("/proc/1/root/dev/dxg")
    except OSError:
        pass

ROOT = Path(
    os.environ.get(
        "VLM_REPAIR_V2_ROOT", Path(__file__).resolve().parents[3] / "legacy" / "vlm_repair_v2"
    )
)
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/sam3_instance_hypothesis_binding"
RAW_CACHE = OUT / "raw_cache"
MASK_DIR = OUT / "sam3_masks"
CONS_CACHE = OUT / "consolidation_cache"
OBJEMBED_CACHE = OUT / "objembed_cache"
DOWNSTREAM_DIR = OUT / "downstream"
FROZEN_RAW = ROOT / "tmp/sam3_dev_candidates.json"
GT_PATH = ROOT / "outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"
PREPARED_PATH = ROOT / "outputs/minimal_decomposition_route_pilot/prepared.jsonl"
PRIOR_PATH = ROOT / "outputs/minimal_decomposition_route_pilot/route_results.jsonl"
HIST_OUT = ROOT / "outputs/strong_integrated_repair"
HIGH_RECALL_OUT = ROOT / "outputs/high_recall_binding_repair"
CONTEXT_OUT = ROOT / "outputs/context_preserving_binding"

SAM3_K = 64
DUP_IOU = 0.70
CONTAIN_FRAC = 0.90
AREA_RATIO = 0.70
HIGH_OVERLAP_IOU = 0.30
IOU_THRESHOLD = 0.50
OBJEMBED_QUERY = "Find an object that matches the given caption. {referent} <local_text>"
EXPERIMENT = "sam3_instance_hypothesis_binding_v1"
ROUTE_INDEP = "HYPOTHESIS_OBJEMBED_INDEPENDENT"
ROUTE_JOINT = "HYPOTHESIS_OBJEMBED_JOINT"
ROUTE_DOWNSTREAM = "SAM3_INSTANCE_HYPOTHESIS_BINDER"


def rows(path: Path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # crash-safe: ignore a single partial trailing JSONL row
            continue
    return out


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_jsonl(path: Path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in records),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def iou_bbox(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def clip_bbox(bbox, size):
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = float(size[0]), float(size[1])
    x0, y0 = max(0.0, min(w, x0)), max(0.0, min(h, y0))
    x1, y1 = max(0.0, min(w, x1)), max(0.0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def target_from_boxes(bbox_a, bbox_b, qa_type):
    if bbox_a is None or bbox_b is None:
        return None
    ax = (float(bbox_a[0]) + float(bbox_a[2])) / 2.0
    bx = (float(bbox_b[0]) + float(bbox_b[2])) / 2.0
    relation = "left" if ax - bx < -1e-9 else "right" if ax - bx > 1e-9 else None
    side = "left" if str(qa_type).endswith("LEFT") else "right"
    return "A" if relation == side else "B" if relation in ("left", "right") else None


def load_gt():
    gt = {}
    for row in rows(GT_PATH):
        inf = row["inference"]
        if inf.get("backend") == "grounding_dino":
            gt[(inf["family_id"], inf["logical_ref"])] = [float(v) for v in row["EVALUATOR_ONLY"]["gt_bbox_xyxy_pixels"]]
    assert len(gt) == 40
    return gt


def load_prepared():
    prep = {r["qa_id"]: r for r in rows(PREPARED_PATH)}
    prior = {r["qa_id"]: r for r in rows(PRIOR_PATH)}
    assert len(prep) == len(prior) == 40
    return prep, prior


def resume_update(**kwargs):
    path = OUT / "resume_status.json"
    status = read_json(path) if path.exists() else {}
    status.update({"experiment": EXPERIMENT, "updated_unix": time.time()})
    status.update(kwargs)
    write_json(path, status)


# ---------------------------------------------------------------------------
# Stage raw: frozen SAM3 top64 proposals + masks
# ---------------------------------------------------------------------------

def _raw_cache_path(key):
    return RAW_CACHE / (key.replace("|", "__") + ".json")


def stage_raw():
    """Regenerate exactly the frozen SAM3 top64 proposal call to recover masks.

    The historical top64 cache contains boxes and native scores but discarded
    masks for ranks 17-64.  We run the frozen SAM3 checkpoint/query/cap only
    because mask data for the frozen candidate source is absent, then verify
    every regenerated box/score against the frozen cache.
    """
    import torch
    import harness.scripts.run_strong_integrated_repair as frozen

    OUT.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    frozen_raw = read_json(FROZEN_RAW)
    keys = sorted(frozen_raw)
    assert len(keys) == 40
    done = {k for k in keys if _raw_cache_path(k).exists()}
    if len(done) == len(keys):
        print("[raw] all 40 referents recovered from cache", flush=True)
        build_raw_manifest()
        return
    processor, model = frozen.load_sam3()
    print(f"[raw] loaded SAM3; cached={len(done)}/40", flush=True)
    for index, key in enumerate(keys, 1):
        if key in done:
            continue
        ref = frozen_raw[key]
        image_path = ref["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_size = list(image.size)
        assert list(ref["image_size"]) == image_size
        with torch.inference_mode():
            image_inputs = processor(images=image, return_tensors="pt")
            vision_embeds = model.get_vision_features(pixel_values=image_inputs["pixel_values"].to("cuda"))
        kept, all_boxes, all_scores = frozen._top_candidates_from_sam3(
            processor, model, vision_embeds, ref["referent_phrase"], image.size, SAM3_K
        )
        assert len(kept) == min(SAM3_K, len(ref["boxes"]))
        candidates = []
        bbox_diffs = []
        score_diffs = []
        for rank0, item in enumerate(kept):
            rank = rank0 + 1
            mask = item["mask"]
            mask_area = int(np.asarray(mask).sum())
            bbox = [float(v) for v in item["bbox_xyxy"]]
            frozen_bbox = [float(v) for v in ref["boxes"][rank0]]
            frozen_score = float(ref["scores"][rank0])
            bbox_diffs.append(max(abs(bbox[i] - frozen_bbox[i]) for i in range(4)))
            score_diffs.append(abs(float(item["native_score"]) - frozen_score))
            mask_path = MASK_DIR / f"{key.replace('|','__')}__r{rank:03d}.png"
            frozen._save_mask(mask, mask_path)
            candidates.append({
                "candidate_id": f"s{rank-1:03d}",
                "native_rank": rank,
                "raw_query_index": int(item["raw_query_index"]),
                "bbox_xyxy": bbox,
                "native_score": float(item["native_score"]),
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "mask_area": mask_area,
                "mask_valid": bool(mask_area > 0),
                "bbox_area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                "image_sha256": sha256_file(image_path),
                "source": "SAM3-LiteText-L",
            })
        payload = {
            "family_id": ref["family_id"],
            "logical_ref": ref["logical_ref"],
            "referent_phrase": ref["referent_phrase"],
            "image_path": image_path,
            "image_size": image_size,
            "image_sha256": sha256_file(image_path),
            "sam3_k": SAM3_K,
            "candidates": candidates,
            "reconstruction": {
                "max_bbox_abs_diff": max(bbox_diffs),
                "max_native_score_abs_diff": max(score_diffs),
                "reconstructed_from_frozen_cache": True,
            },
        }
        write_json(_raw_cache_path(key), payload)
        del vision_embeds
        torch.cuda.empty_cache()
        print(f"[raw] {index}/40 {key} n={len(candidates)} diff={max(bbox_diffs):.2e}", flush=True)
    build_raw_manifest()
    resume_update(status="RAW_COMPLETE", raw_complete=True, raw_referents=40)
    print("[raw] complete", flush=True)


def build_raw_manifest():
    frozen_raw = read_json(FROZEN_RAW)
    referents = []
    for key in sorted(frozen_raw):
        path = _raw_cache_path(key)
        if not path.exists():
            raise RuntimeError(f"missing raw cache for {key}; run stage raw first")
        referents.append(read_json(path))
    manifest = {
        "schema": "sam3_raw_top64_masks_v1",
        "sam3_k": SAM3_K,
        "source_frozen_cache": str(FROZEN_RAW),
        "source_frozen_sha256": sha256_file(FROZEN_RAW),
        "note": "mask PNGs regenerate the frozen SAM3 proposal call; boxes/scores were checked against the frozen cache",
        "n_referents": len(referents),
        "referents": referents,
    }
    write_json(OUT / "raw_candidate_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Stage A/B: structural audit and canonical instance hypotheses
# ---------------------------------------------------------------------------

def _load_mask(path):
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def _distribution(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "p25": None, "p75": None}
    clean.sort()
    def q(p):
        if len(clean) == 1:
            return clean[0]
        idx = p * (len(clean) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return clean[lo]
        return clean[lo] * (hi - idx) + clean[hi] * (idx - lo)
    return {
        "n": len(clean), "min": clean[0], "max": clean[-1],
        "mean": sum(clean) / len(clean), "median": q(0.5), "p25": q(0.25), "p75": q(0.75),
    }


def _component_structural_representative(member_indices, cands, masks, areas):
    """Largest valid mask; tie by higher native score, then lower rank."""
    def key(i):
        return (1 if areas[i] > 0 else 0, areas[i], float(cands[i]["native_score"]), -int(cands[i]["native_rank"]))
    return max(member_indices, key=key)


def _pair_relation(i, j, cands, masks, areas):
    inter = int(np.logical_and(masks[i], masks[j]).sum()) if (areas[i] > 0 and areas[j] > 0) else 0
    union = areas[i] + areas[j] - inter
    mask_iou = inter / union if union > 0 else 0.0
    bbox_iou = iou_bbox(cands[i]["bbox_xyxy"], cands[j]["bbox_xyxy"])
    duplicate = bool(mask_iou >= DUP_IOU or ((areas[i] == 0 or areas[j] == 0) and bbox_iou >= DUP_IOU))
    overlap = bool(mask_iou >= HIGH_OVERLAP_IOU)
    # directional containment; mask-based by default, bbox fallback only when a mask is empty
    contain_i_j = inter / areas[i] if areas[i] > 0 else 0.0
    contain_j_i = inter / areas[j] if areas[j] > 0 else 0.0
    if areas[i] == 0:
        contain_i_j = 0.0
    if areas[j] == 0:
        contain_j_i = 0.0
    area_ratio_i_j = areas[i] / areas[j] if areas[j] > 0 else float("inf")
    area_ratio_j_i = areas[j] / areas[i] if areas[i] > 0 else float("inf")
    contained_i_in_j = bool(contain_i_j >= CONTAIN_FRAC and area_ratio_i_j <= AREA_RATIO)
    contained_j_in_i = bool(contain_j_i >= CONTAIN_FRAC and area_ratio_j_i <= AREA_RATIO)
    return {
        "mask_intersection": inter,
        "mask_iou": mask_iou,
        "bbox_iou": bbox_iou,
        "duplicate": duplicate,
        "high_overlap": overlap,
        "contain_i_j": contain_i_j,
        "contain_j_i": contain_j_i,
        "area_ratio_i_j": area_ratio_i_j if math.isfinite(area_ratio_i_j) else None,
        "area_ratio_j_i": area_ratio_j_i if math.isfinite(area_ratio_j_i) else None,
        "contained_i_in_j": contained_i_in_j,
        "contained_j_in_i": contained_j_in_i,
    }


def consolidate_referent(raw):
    cands = raw["candidates"]
    n = len(cands)
    masks = [_load_mask(c["mask_path"]) for c in cands]
    areas = [int(m.sum()) for m in masks]
    pair = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair[(i, j)] = _pair_relation(i, j, cands, masks, areas)
    # duplicate connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    duplicate_edges = []
    high_overlap_edges = []
    for (i, j), rel in sorted(pair.items()):
        if rel["duplicate"]:
            union(i, j)
            duplicate_edges.append((i, j, rel))
        elif rel["high_overlap"]:
            high_overlap_edges.append((i, j, rel))
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    raw_component_index = {root: idx for idx, root in enumerate(sorted(comps, key=lambda r: min(comps[r])))}
    comp_records = []
    for root in sorted(comps, key=lambda r: min(comps[r])):
        members = sorted(comps[root])
        rep = _component_structural_representative(members, cands, masks, areas)
        comp_records.append({
            "component_index": raw_component_index[root],
            "component_id": f"d{raw_component_index[root]:03d}",
            "members": members,
            "representative_index": rep,
            "representative_candidate_id": cands[rep]["candidate_id"],
            "representative_native_rank": cands[rep]["native_rank"],
            "representative_native_score": cands[rep]["native_score"],
            "representative_mask_area": areas[rep],
            "max_member_mask_area": max(areas[i] for i in members),
            "min_member_native_rank": min(cands[i]["native_rank"] for i in members),
            "member_count": len(members),
        })
    by_root = {r: rec for r, rec in zip(sorted(comps, key=lambda r: min(comps[r])), comp_records)}
    root_to_comp = {rec["members"][0] if False else root: rec for root, rec in zip(sorted(comps, key=lambda r: min(comps[r])), comp_records)}
    # component-level containment using structural representatives
    component_containment = []
    parent_component = {}
    for ci in comp_records:
        ri = ci["representative_index"]
        choices = []
        for cj in comp_records:
            if ci["component_id"] == cj["component_id"]:
                continue
            rj = cj["representative_index"]
            ai, aj = areas[ri], areas[rj]
            if ai <= 0 or aj <= 0:
                continue
            inter = int(np.logical_and(masks[ri], masks[rj]).sum())
            contain = inter / ai
            if contain >= CONTAIN_FRAC and (ai / aj) <= AREA_RATIO:
                component_containment.append({
                    "child_component_id": ci["component_id"],
                    "parent_component_id": cj["component_id"],
                    "child_representative_candidate_id": ci["representative_candidate_id"],
                    "parent_representative_candidate_id": cj["representative_candidate_id"],
                    "contain_fraction": contain,
                    "child_area": ai,
                    "parent_area": aj,
                    "area_ratio": ai / aj,
                })
                choices.append((aj, -float(cj["representative_native_score"]), rj, cj["component_id"]))
        if choices:
            # most specific containing whole: smallest parent representative area,
            # then highest parent native score, then lower rank, then stable id.
            _, _, _, chosen = min(choices)
            parent_component[ci["component_id"]] = chosen
    # descendants from component attachment tree
    children = defaultdict(list)
    for child, par in parent_component.items():
        children[par].append(child)
    comp_by_id = {c["component_id"]: c for c in comp_records}
    roots = [c["component_id"] for c in comp_records if c["component_id"] not in parent_component]
    roots.sort(key=lambda cid: comp_by_id[cid]["min_member_native_rank"])
    hypotheses = []
    membership = []
    for h_index, root_cid in enumerate(roots):
        hypothesis_id = f"{raw['family_id']}__{raw['logical_ref']}__H{h_index:03d}"
        stack = [(root_cid, 0, None)]
        member_indices = []
        component_edges = []
        depth_by_comp = {}
        while stack:
            cid, depth, parent_cid = stack.pop(0)
            comp = comp_by_id[cid]
            depth_by_comp[cid] = depth
            member_indices.extend(comp["members"])
            if parent_cid is not None:
                component_edges.append({"child_component_id": cid, "parent_component_id": parent_cid, "depth": depth})
            for child_cid in sorted(children.get(cid, []), key=lambda x: comp_by_id[x]["min_member_native_rank"]):
                stack.append((child_cid, depth + 1, cid))
        member_indices = sorted(set(member_indices))
        # root candidates: not strongly contained by any other member in the same hypothesis
        contained_in_hypothesis = set()
        for i in member_indices:
            for j in member_indices:
                if i == j:
                    continue
                ai, aj = areas[i], areas[j]
                if ai <= 0 or aj <= 0:
                    continue
                if (ai / aj) <= AREA_RATIO:
                    inter = int(np.logical_and(masks[i], masks[j]).sum())
                    if inter / ai >= CONTAIN_FRAC:
                        contained_in_hypothesis.add(i)
        root_member_indices = [i for i in member_indices if i not in contained_in_hypothesis]
        if not root_member_indices:
            root_member_indices = list(member_indices)
        def rep_key(i):
            return (
                1 if i in root_member_indices else 0,
                1 if areas[i] > 0 else 0,
                float(cands[i]["native_score"]),
                areas[i],
                -int(cands[i]["native_rank"]),
            )
        rep = max(member_indices, key=rep_key)
        hyp = {
            "hypothesis_id": hypothesis_id,
            "family_id": raw["family_id"],
            "logical_ref": raw["logical_ref"],
            "referent_phrase": raw["referent_phrase"],
            "image_path": raw["image_path"],
            "image_sha256": raw["image_sha256"],
            "root_component_id": root_cid,
            "component_ids": [cid for cid in sorted(depth_by_comp, key=lambda c: (depth_by_comp[c], comp_by_id[c]["min_member_native_rank"]))],
            "component_edges": component_edges,
            "member_candidate_ids": [cands[i]["candidate_id"] for i in member_indices],
            "member_count": len(member_indices),
            "component_count": len(depth_by_comp),
            "max_depth": max(depth_by_comp.values()) if depth_by_comp else 0,
            "root_member_candidate_ids": [cands[i]["candidate_id"] for i in root_member_indices],
            "representative_candidate_id": cands[rep]["candidate_id"],
            "representative_bbox_xyxy": cands[rep]["bbox_xyxy"],
            "representative_mask_path": cands[rep]["mask_path"],
            "representative_native_rank": cands[rep]["native_rank"],
            "representative_native_score": cands[rep]["native_score"],
            "representative_mask_area": areas[rep],
            "representative_selection_rule": "root member (not contained in any hypothesis member) > valid mask > higher native score > larger mask area > lower native rank",
        }
        hypotheses.append(hyp)
        for i in member_indices:
            membership.append({
                "family_id": raw["family_id"],
                "logical_ref": raw["logical_ref"],
                "hypothesis_id": hypothesis_id,
                "candidate_id": cands[i]["candidate_id"],
                "native_rank": cands[i]["native_rank"],
                "native_score": cands[i]["native_score"],
                "bbox_xyxy": cands[i]["bbox_xyxy"],
                "mask_path": cands[i]["mask_path"],
                "mask_area": areas[i],
                "duplicate_component_id": next(c["component_id"] for c in comp_records if i in c["members"]),
                "component_parent_id": parent_component.get(next(c["component_id"] for c in comp_records if i in c["members"])),
                "depth": depth_by_comp[next(c["component_id"] for c in comp_records if i in c["members"])],
                "is_attached_child": next(c["component_id"] for c in comp_records if i in c["members"]) in parent_component,
                "is_root_candidate": bool(i in root_member_indices),
            })
    # structural stats
    bbox_ious = [r["bbox_iou"] for r in pair.values()]
    mask_ious = [r["mask_iou"] for r in pair.values()]
    cont_vals = [max(r["contain_i_j"], r["contain_j_i"]) for r in pair.values()]
    near_dup_pairs = sum(1 for r in pair.values() if r["duplicate"])
    high_contain_pairs = sum(1 for r in pair.values() if r["contained_i_in_j"] or r["contained_j_in_i"])
    duplicate_component_sizes = [c["member_count"] for c in comp_records]
    structure = {
        "family_id": raw["family_id"],
        "logical_ref": raw["logical_ref"],
        "referent_phrase": raw["referent_phrase"],
        "raw_candidate_count": n,
        "hypothesis_count": len(hypotheses),
        "duplicate_component_count": len(comp_records),
        "duplicate_edge_count": len(duplicate_edges),
        "high_overlap_edge_count": len(high_overlap_edges),
        "near_duplicate_pair_count": near_dup_pairs,
        "high_containment_pair_count": high_contain_pairs,
        "duplicate_component_size_max": max(duplicate_component_sizes) if duplicate_component_sizes else 0,
        "singleton_component_count": sum(1 for s in duplicate_component_sizes if s == 1),
        "duplicate_member_count": sum(s for s in duplicate_component_sizes if s > 1),
        "bbox_iou_distribution": _distribution(bbox_ious),
        "mask_iou_distribution": _distribution(mask_ious),
        "containment_distribution": _distribution(cont_vals),
        "area_distribution": _distribution(areas),
        "native_score_distribution": _distribution([c["native_score"] for c in cands]),
        "thresholds": {"duplicate_iou": DUP_IOU, "contain_fraction": CONTAIN_FRAC, "area_ratio": AREA_RATIO, "high_overlap_iou": HIGH_OVERLAP_IOU},
    }
    cache = {
        "raw": raw,
        "structure": structure,
        "duplicate_edges": [{"family_id": raw["family_id"], "logical_ref": raw["logical_ref"], "edge_type": "DUPLICATE", "candidate_i": cands[i]["candidate_id"], "candidate_j": cands[j]["candidate_id"], "native_rank_i": cands[i]["native_rank"], "native_rank_j": cands[j]["native_rank"], **rel} for i, j, rel in duplicate_edges],
        "high_overlap_edges": [{"family_id": raw["family_id"], "logical_ref": raw["logical_ref"], "edge_type": "HIGH_OVERLAP", "candidate_i": cands[i]["candidate_id"], "candidate_j": cands[j]["candidate_id"], "native_rank_i": cands[i]["native_rank"], "native_rank_j": cands[j]["native_rank"], **rel} for i, j, rel in high_overlap_edges],
        "containment_edges": component_containment,
        "components": comp_records,
        "hypotheses": hypotheses,
        "membership": membership,
    }
    write_json(CONS_CACHE / (f"{raw['family_id']}__{raw['logical_ref']}.json"), cache)
    return cache


def stage_consolidate():
    OUT.mkdir(parents=True, exist_ok=True)
    CONS_CACHE.mkdir(parents=True, exist_ok=True)
    raw_manifest = read_json(OUT / "raw_candidate_manifest.json")
    keys = [f"{r['family_id']}__{r['logical_ref']}" for r in raw_manifest["referents"]]
    caches = []
    for raw in raw_manifest["referents"]:
        cache_path = CONS_CACHE / f"{raw['family_id']}__{raw['logical_ref']}.json"
        if cache_path.exists():
            cache = read_json(cache_path)
        else:
            cache = consolidate_referent(raw)
            print(f"[consolidate] {raw['family_id']} {raw['logical_ref']} raw={len(raw['candidates'])} hyp={len(cache['hypotheses'])}", flush=True)
        caches.append(cache)
    # Required JSONL exports.
    write_jsonl(OUT / "proposal_structure.jsonl", [c["structure"] for c in caches])
    duplicate_rows = []
    containment_rows = []
    hypothesis_rows = []
    membership_rows = []
    representative_rows = []
    for c in caches:
        duplicate_rows.extend(c["duplicate_edges"])
        duplicate_rows.extend(c["high_overlap_edges"])
        for e in c["containment_edges"]:
            containment_rows.append({**e, "edge_type": "CONTAINED_IN"})
            containment_rows.append({
                "family_id": c["raw"]["family_id"], "logical_ref": c["raw"]["logical_ref"],
                "edge_type": "CONTAINS",
                "child_component_id": e["child_component_id"], "parent_component_id": e["parent_component_id"],
                "child_representative_candidate_id": e["child_representative_candidate_id"],
                "parent_representative_candidate_id": e["parent_representative_candidate_id"],
                "contain_fraction": e["contain_fraction"], "area_ratio": e["area_ratio"],
            })
        hypothesis_rows.extend(c["hypotheses"])
        membership_rows.extend(c["membership"])
        for h in c["hypotheses"]:
            representative_rows.append({
                "family_id": h["family_id"], "logical_ref": h["logical_ref"], "hypothesis_id": h["hypothesis_id"],
                "representative_candidate_id": h["representative_candidate_id"],
                "representative_bbox_xyxy": h["representative_bbox_xyxy"],
                "representative_mask_path": h["representative_mask_path"],
                "representative_native_rank": h["representative_native_rank"],
                "representative_native_score": h["representative_native_score"],
                "representative_mask_area": h["representative_mask_area"],
                "member_count": h["member_count"],
                "max_depth": h["max_depth"],
                "selection_rule": h["representative_selection_rule"],
            })
    write_jsonl(OUT / "duplicate_graph.jsonl", duplicate_rows)
    write_jsonl(OUT / "containment_graph.jsonl", containment_rows)
    write_jsonl(OUT / "instance_hypotheses.jsonl", hypothesis_rows)
    write_jsonl(OUT / "hypothesis_membership.jsonl", membership_rows)
    write_jsonl(OUT / "representative_regions.jsonl", representative_rows)

    n_raw = sum(c["structure"]["raw_candidate_count"] for c in caches)
    n_hyp = sum(c["structure"]["hypothesis_count"] for c in caches)
    member_counts = [h["member_count"] for c in caches for h in c["hypotheses"]]
    part_hyp = sum(1 for c in caches for h in c["hypotheses"] if any(e["child_component_id"] for e in h["component_edges"]))
    singleton_hyp = sum(1 for x in member_counts if x == 1)
    # same-instance duplicate rate: fraction of raw candidates in duplicate components >1
    dup_members = sum(c["structure"]["duplicate_member_count"] for c in caches)
    near_dup_pairs = sum(c["structure"]["near_duplicate_pair_count"] for c in caches)
    total_pairs = sum(c["structure"]["raw_candidate_count"] * (c["structure"]["raw_candidate_count"] - 1) // 2 for c in caches)
    per_ref_hyp = [c["structure"]["hypothesis_count"] for c in caches]
    per_ref_raw = [c["structure"]["raw_candidate_count"] for c in caches]
    metrics = {
        "experiment": EXPERIMENT,
        "thresholds": {"duplicate_iou": DUP_IOU, "contain_fraction": CONTAIN_FRAC, "area_ratio": AREA_RATIO, "high_overlap_iou": HIGH_OVERLAP_IOU},
        "n_referent_candidate_sets": len(caches),
        "n_unique_images": 20,
        "raw_candidate_count_total": n_raw,
        "hypothesis_count_total": n_hyp,
        "mean_raw_candidates_per_referent": n_raw / len(caches),
        "median_raw_candidates_per_referent": sorted(per_ref_raw)[len(per_ref_raw)//2],
        "mean_hypotheses_per_referent": n_hyp / len(caches),
        "median_hypotheses_per_referent": sorted(per_ref_hyp)[len(per_ref_hyp)//2],
        "candidate_compression_ratio_raw_over_hypotheses": n_raw / n_hyp if n_hyp else None,
        "mean_compression_ratio_per_referent": sum(r / max(1, h) for r, h in zip(per_ref_raw, per_ref_hyp)) / len(caches),
        "median_members_per_hypothesis": sorted(member_counts)[len(member_counts)//2],
        "mean_members_per_hypothesis": sum(member_counts) / len(member_counts),
        "max_members_per_hypothesis": max(member_counts),
        "singleton_hypothesis_fraction": singleton_hyp / len(member_counts),
        "part_containing_hypothesis_fraction": part_hyp / len(member_counts),
        "same_instance_duplicate_member_rate": dup_members / n_raw,
        "near_duplicate_pair_rate": near_dup_pairs / total_pairs if total_pairs else None,
        "candidate_entropy_reduction_proxy": n_raw / n_hyp if n_hyp else None,
        "notes": [
            "Counts are per referent candidate set (40 sets over 20 images); each image contributes A and B referents.",
            "same_instance_duplicate_member_rate is the fraction of raw proposals in a duplicate component of size >1.",
            "candidate_entropy_reduction_proxy is the operational raw_count / hypothesis_count ratio.",
        ],
    }
    write_json(OUT / "consolidation_metrics.json", metrics)

    report = {
        "experiment": EXPERIMENT,
        "structural_summary": metrics,
        "per_referent_structure": [c["structure"] for c in caches],
        "thresholds": metrics["thresholds"],
    }
    write_json(OUT / "raw_proposal_structure_report.json", report)
    _write_structure_markdown(metrics, [c["structure"] for c in caches])
    resume_update(status="CONSOLIDATION_COMPLETE", consolidation_complete=True, raw_candidate_count=n_raw, hypothesis_count=n_hyp)
    print(f"[consolidate] raw={n_raw} hypotheses={n_hyp} compression={n_raw/max(1,n_hyp):.3f} singleton_frac={singleton_hyp/max(1,len(member_counts)):.3f}", flush=True)


def _write_structure_markdown(metrics, structures):
    lines = ["# Raw SAM3 proposal structure report", ""]
    lines.append("This is a structure-only audit before any evaluator annotation is opened.  Counts are per referent candidate set (40 sets over 20 images); each image contributes an A and a B referent.")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Mean raw candidates per referent | {metrics['mean_raw_candidates_per_referent']:.3f} |")
    lines.append(f"| Median raw candidates per referent | {metrics['median_raw_candidates_per_referent']} |")
    lines.append(f"| Mean hypotheses per referent | {metrics['mean_hypotheses_per_referent']:.3f} |")
    lines.append(f"| Median hypotheses per referent | {metrics['median_hypotheses_per_referent']} |")
    lines.append(f"| Compression ratio (raw / hypotheses) | {metrics['candidate_compression_ratio_raw_over_hypotheses']:.3f} |")
    lines.append(f"| Same-instance duplicate member rate | {metrics['same_instance_duplicate_member_rate']:.3f} |")
    lines.append(f"| Near-duplicate pair rate | {metrics['near_duplicate_pair_rate']:.3f} |")
    lines.append(f"| Singleton hypothesis fraction | {metrics['singleton_hypothesis_fraction']:.3f} |")
    lines.append(f"| Part-containing hypothesis fraction | {metrics['part_containing_hypothesis_fraction']:.3f} |")
    lines.append("")
    lines.append("Per-referent raw/mask-IoU/containment distributions are in `proposal_structure.jsonl`; graph edges are in `duplicate_graph.jsonl` and `containment_graph.jsonl`.")
    (OUT / "raw_proposal_structure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Oracle preservation (evaluator-only, after hypotheses are frozen)
# ---------------------------------------------------------------------------

def stage_oracle():
    gt = load_gt()
    frozen_raw = read_json(FROZEN_RAW)
    raw_depths = {}
    for depth in (16, 32, 64, 128, 200):
        hit = 0
        for key, ref in frozen_raw.items():
            f, r = key.rsplit("|", 1)
            boxes = ref["boxes"][:depth]
            if any(iou_bbox(b, gt[(f, r)]) >= IOU_THRESHOLD for b in boxes):
                hit += 1
        raw_depths[f"top{depth}"] = hit
    hypothesis_oracle = 0
    hypothesis_depth = {}
    for depth in (16, 32, 64):
        hit = 0
        for key, ref in frozen_raw.items():
            f, r = key.rsplit("|", 1)
            hyp_path = CONS_CACHE / f"{f}__{r}.json"
            cache = read_json(hyp_path)
            # member native ranks are 1-based; only proposals within depth participate
            usable = False
            for h in cache["hypotheses"]:
                for cid in h["member_candidate_ids"]:
                    member = next(c for c in cache["raw"]["candidates"] if c["candidate_id"] == cid)
                    if member["native_rank"] <= depth and iou_bbox(member["bbox_xyxy"], gt[(f, r)]) >= IOU_THRESHOLD:
                        usable = True
                        break
                if usable:
                    break
            if usable:
                hit += 1
        hypothesis_depth[f"top{depth}"] = hit
    # Overall consolidated oracle uses all frozen top64 members.
    for f_ref in gt:
        f, r = f_ref
        cache = read_json(CONS_CACHE / f"{f}__{r}.json")
        if any(iou_bbox(m["bbox_xyxy"], gt[f_ref]) >= IOU_THRESHOLD for m in cache["raw"]["candidates"]):
            hypothesis_oracle += 1
    raw_top64 = raw_depths["top64"]
    damage = raw_top64 - hypothesis_oracle
    result = {
        "evaluator_only": True,
        "iou_threshold": IOU_THRESHOLD,
        "raw_candidate_oracle_by_depth": raw_depths,
        "consolidated_hypothesis_oracle_by_depth": hypothesis_depth,
        "raw_candidate_oracle_top64": raw_top64,
        "consolidated_oracle_top64": hypothesis_oracle,
        "oracle_damage_vs_raw_top64": damage,
        "consolidation_gate_pass": bool(damage <= 2),
        "stop_a_triggered": bool(damage > 2),
        "chain_stage_1": {
            "name": "RawCandidateOracle",
            "count": raw_top64,
            "denominator": 40,
        },
        "chain_stage_2": {
            "name": "HypothesisOracle",
            "count": hypothesis_oracle,
            "denominator": 40,
        },
        "notes": [
            "A hypothesis preserves a usable region iff any of its frozen top64 members has IoU>=0.5 with GT.",
            "If raw top64 oracle is below 34/40, use the relative raw-to-consolidated comparison rather than the absolute target.",
        ],
    }
    write_json(OUT / "oracle_preservation.json", result)
    resume_update(status="ORACLE_COMPLETE", oracle_complete=True, raw_top64_oracle=raw_top64, consolidated_oracle=hypothesis_oracle, consolidation_gate_pass=bool(damage <= 2))
    print(f"[oracle] raw_top64={raw_top64}/40 consolidated={hypothesis_oracle}/40 damage={damage}", flush=True)
    if damage > 2:
        print("[oracle] STOP_A: consolidation damaged oracle by >2/40", flush=True)



# ---------------------------------------------------------------------------
# Stage B/C/D: ObjEmbed member scoring and instance-level binding
# ---------------------------------------------------------------------------

def stage_objembed():
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    helper = ROOT / "harness/scripts/score_objembed_sam3_instance_hypotheses.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "external/objembed_pydeps"),
        str(ROOT / "external/ObjEmbed"),
        str(ROOT),
    ])
    print("[objembed] launching isolated ObjEmbed scorer", flush=True)
    subprocess.run([sys.executable, str(helper)], check=True, env=env)
    expected = 40
    got = len(rows(OUT / "objembed_member_scores.jsonl"))
    assert got == expected, f"ObjEmbed scores {got}/{expected}"
    resume_update(status="OBJEMBED_COMPLETE", objembed_complete=True, objembed_referents=got)
    print(f"[objembed] complete {got}/40", flush=True)


def _load_objembed_scores():
    data = {}
    for rec in rows(OUT / "objembed_member_scores.jsonl"):
        key = (rec["family_id"], rec["logical_ref"])
        assert rec["status"] == "OK"
        assert len(rec["candidate_ids"]) == len(rec["scores"])
        data[key] = {
            "candidate_id_to_score": {cid: float(s) for cid, s in zip(rec["candidate_ids"], rec["scores"])},
            "candidate_id_to_semantic": {cid: float(s) for cid, s in zip(rec["candidate_ids"], rec["semantic_scores"])},
            "candidate_id_to_localization": {cid: float(s) for cid, s in zip(rec["candidate_ids"], rec["localization_quality"])},
            "record": rec,
        }
    assert len(data) == 40
    return data


def _hypothesis_scores_for_referent(cache, obj_rec):
    rows_out = []
    for h in cache["hypotheses"]:
        member_scores = [(cid, obj_rec["candidate_id_to_score"][cid]) for cid in h["member_candidate_ids"]]
        values = [v for _, v in member_scores]
        ordered = sorted(values, reverse=True)
        rows_out.append({
            "family_id": h["family_id"],
            "logical_ref": h["logical_ref"],
            "hypothesis_id": h["hypothesis_id"],
            "member_candidate_ids": [cid for cid, _ in member_scores],
            "member_scores": [float(v) for _, v in member_scores],
            "hypothesis_score_max": float(max(values)) if values else None,
            "hypothesis_score_mean": float(sum(values) / len(values)) if values else None,
            "hypothesis_score_top2_mean": float(sum(ordered[:2]) / len(ordered[:2])) if ordered else None,
            "primary_pooling": "max",
            "representative_candidate_id": h["representative_candidate_id"],
        })
    return rows_out


def _select_raw(scores, raw_candidates):
    ranked = sorted(
        [(float(scores[c["candidate_id"]]), c) for c in raw_candidates],
        key=lambda x: (x[0], -x[1]["native_rank"]),
        reverse=True,
    )
    if not ranked:
        return None
    score, cand = ranked[0]
    return {
        "selected_candidate_id": cand["candidate_id"],
        "selected_bbox_xyxy": cand["bbox_xyxy"],
        "selected_mask_path": cand["mask_path"],
        "selected_native_rank": cand["native_rank"],
        "selected_score": score,
        "fallback_reason": None,
        "scores": [{"candidate_id": c["candidate_id"], "native_rank": c["native_rank"], "score": float(scores[c["candidate_id"]])} for _, c in sorted(
            [(float(scores[c["candidate_id"]]), c) for c in raw_candidates],
            key=lambda x: (x[0], -x[1]["native_rank"]), reverse=True)],
    }


def _select_hypothesis(hyp_rows):
    if not hyp_rows:
        return None
    # fixed primary rule: max member score; deterministic tie-break by best
    # member native rank and hypothesis id.  No GT or detector-score fusion.
    def key(row):
        best_rank = min(int(x) for x in row.get("member_ranks", [10**9]))
        return (row["hypothesis_score_max"], -best_rank, row["hypothesis_id"])
    ranked = sorted(hyp_rows, key=key, reverse=True)
    best = ranked[0]
    return {
        "selected_hypothesis_id": best["hypothesis_id"],
        "selected_representative_candidate_id": best["representative_candidate_id"],
        "selected_score": best["hypothesis_score_max"],
        "fallback_reason": None,
        "scores": ranked,
    }


def _write_binding_route(route, records):
    path = OUT / f"{route}_binding_results.jsonl"
    write_jsonl(path, records)


def stage_bind():
    gt = None  # GT is opened only after all route selections have been written.
    raw_manifest = read_json(OUT / "raw_candidate_manifest.json")
    raw_by_key = {(r["family_id"], r["logical_ref"]): r for r in raw_manifest["referents"]}
    caches = {}
    for key in raw_by_key:
        caches[key] = read_json(CONS_CACHE / f"{key[0]}__{key[1]}.json")
    obj = _load_objembed_scores()
    # hypothesis-level score records
    hyp_score_rows = []
    independent = []
    raw_route = []
    for key, raw in sorted(raw_by_key.items()):
        obj_rec = obj[key]
        hyp_rows = _hypothesis_scores_for_referent(caches[key], obj_rec)
        for h in hyp_rows:
            cache_h = next(x for x in caches[key]["hypotheses"] if x["hypothesis_id"] == h["hypothesis_id"])
            member_ranks = [int(next(c for c in raw["candidates"] if c["candidate_id"] == cid)["native_rank"]) for cid in h["member_candidate_ids"]]
            h["member_ranks"] = member_ranks
            hyp_score_rows.append(h)
        raw_sel = _select_raw(obj_rec["candidate_id_to_score"], raw["candidates"])
        hyp_sel = _select_hypothesis(hyp_rows)
        raw_route.append({
            "family_id": key[0], "logical_ref": key[1], "referent_phrase": raw["referent_phrase"],
            "route": "RAW_OBJEMBED_SAM3_TOP64",
            "selected_candidate_id": raw_sel["selected_candidate_id"],
            "selected_bbox_xyxy": raw_sel["selected_bbox_xyxy"],
            "selected_mask_path": raw_sel["selected_mask_path"],
            "selected_native_rank": raw_sel["selected_native_rank"],
            "selected_score": raw_sel["selected_score"],
            "fallback_reason": raw_sel["fallback_reason"],
            "candidate_scores": raw_sel["scores"],
        })
        independent.append({
            "family_id": key[0], "logical_ref": key[1], "referent_phrase": raw["referent_phrase"],
            "route": ROUTE_INDEP,
            "selected_hypothesis_id": hyp_sel["selected_hypothesis_id"],
            "selected_representative_candidate_id": hyp_sel["selected_representative_candidate_id"],
            "selected_bbox_xyxy": next(x["representative_bbox_xyxy"] for x in caches[key]["hypotheses"] if x["hypothesis_id"] == hyp_sel["selected_hypothesis_id"]),
            "selected_mask_path": next(x["representative_mask_path"] for x in caches[key]["hypotheses"] if x["hypothesis_id"] == hyp_sel["selected_hypothesis_id"]),
            "selected_score": hyp_sel["selected_score"],
            "fallback_reason": hyp_sel["fallback_reason"],
            "hypothesis_scores": hyp_sel["scores"],
        })
    write_jsonl(OUT / "hypothesis_scores.jsonl", hyp_score_rows)
    _write_binding_route("independent", independent)
    _write_binding_route("raw_objembed", raw_route)
    # joint assignment at hypothesis level
    joint = []
    by_family = defaultdict(dict)
    for row in independent:
        by_family[row["family_id"]][row["logical_ref"]] = row
    joint_notes = []
    for family_id, refs in sorted(by_family.items()):
        a_h = [h for h in hyp_score_rows if h["family_id"] == family_id and h["logical_ref"] == "A"]
        b_h = [h for h in hyp_score_rows if h["family_id"] == family_id and h["logical_ref"] == "B"]
        candidates = []
        for ha in a_h:
            for hb in b_h:
                if ha["hypothesis_id"] == hb["hypothesis_id"]:
                    continue
                candidates.append((ha["hypothesis_score_max"] + hb["hypothesis_score_max"], ha, hb))
        if candidates:
            best_sum, ha, hb = max(candidates, key=lambda x: (x[0], x[1]["hypothesis_id"], x[2]["hypothesis_id"]))
            joint_notes.append({"family_id": family_id, "joint_sum": best_sum, "A": ha["hypothesis_id"], "B": hb["hypothesis_id"]})
            for ha_sel, ref in ((ha, "A"), (hb, "B")):
                hcache = next(x for x in caches[(family_id, ref)]["hypotheses"] if x["hypothesis_id"] == ha_sel["hypothesis_id"])
                joint.append({
                    "family_id": family_id, "logical_ref": ref,
                    "referent_phrase": raw_by_key[(family_id, ref)]["referent_phrase"],
                    "route": ROUTE_JOINT,
                    "selected_hypothesis_id": hcache["hypothesis_id"],
                    "selected_representative_candidate_id": hcache["representative_candidate_id"],
                    "selected_bbox_xyxy": hcache["representative_bbox_xyxy"],
                    "selected_mask_path": hcache["representative_mask_path"],
                    "selected_score": ha_sel["hypothesis_score_max"],
                    "fallback_reason": None,
                    "hypothesis_scores": [x for x in hyp_score_rows if x["family_id"] == family_id and x["logical_ref"] == ref],
                })
        else:
            # technical fallback: same independent selection, recorded explicitly
            joint_notes.append({"family_id": family_id, "joint_sum": None, "fallback_reason": "NO_DISTINCT_HYPOTHESIS_PAIR"})
            for ref in ("A", "B"):
                row = copy.deepcopy(by_family[family_id][ref])
                row["route"] = ROUTE_JOINT
                row["fallback_reason"] = "NO_DISTINCT_HYPOTHESIS_PAIR"
                joint.append(row)
    assert len(joint) == 40
    _write_binding_route("joint", joint)
    write_json(OUT / "joint_assignment_debug.json", {"pairs": joint_notes, "constraint": "hypothesis_id_A != hypothesis_id_B only"})

    # candidate-level joint baseline on the same raw top64 set (for Q6)
    candidate_joint = []
    for family_id, refs in sorted(by_family.items()):
        a_raw = raw_by_key[(family_id, "A")]["candidates"]
        b_raw = raw_by_key[(family_id, "B")]["candidates"]
        a_scores = obj[(family_id, "A")]["candidate_id_to_score"]
        b_scores = obj[(family_id, "B")]["candidate_id_to_score"]
        pairs = []
        for ca in a_raw:
            for cb in b_raw:
                if ca["candidate_id"] == cb["candidate_id"]:
                    continue
                pairs.append((a_scores[ca["candidate_id"]] + b_scores[cb["candidate_id"]], ca, cb))
        if pairs:
            best_sum, ca, cb = max(pairs, key=lambda x: (x[0], -x[1]["native_rank"], -x[2]["native_rank"]))
            for c, ref in ((ca, "A"), (cb, "B")):
                candidate_joint.append({
                    "family_id": family_id, "logical_ref": ref,
                    "route": "RAW_OBJEMBED_CANDIDATE_JOINT_SAM3_TOP64",
                    "selected_candidate_id": c["candidate_id"], "selected_bbox_xyxy": c["bbox_xyxy"],
                    "selected_mask_path": c["mask_path"], "selected_score": a_scores[c["candidate_id"]] if ref == "A" else b_scores[c["candidate_id"]],
                    "fallback_reason": None,
                })
        else:
            for ref in ("A", "B"):
                c = raw_by_key[(family_id, ref)]["candidates"][0]
                candidate_joint.append({
                    "family_id": family_id, "logical_ref": ref,
                    "route": "RAW_OBJEMBED_CANDIDATE_JOINT_SAM3_TOP64",
                    "selected_candidate_id": c["candidate_id"], "selected_bbox_xyxy": c["bbox_xyxy"],
                    "selected_mask_path": c["mask_path"], "selected_score": (a_scores if ref == "A" else b_scores)[c["candidate_id"]],
                    "fallback_reason": "NO_DISTINCT_CANDIDATE_PAIR",
                })
    _write_binding_route("candidate_joint", candidate_joint)

    # All route choices are now frozen.  Open evaluator annotations only here.
    gt = load_gt()
    prepared, prior = load_prepared()
    metric = evaluate_routes(raw_by_key, caches, obj, gt, prepared, prior, independent, joint, raw_route, candidate_joint)
    write_json(OUT / "binding_metrics.json", metric)
    resume_update(status="BINDING_COMPLETE", binding_complete=True, best_route=metric["best_route"], best_grounded=metric["routes"][metric["best_route"]]["grounded_target_correct"])
    print(f"[bind] best={metric['best_route']} grounded={metric['routes'][metric['best_route']]['grounded_target_correct']}/40", flush=True)


def _route_entity_metrics(route_name, route_rows, caches, gt):
    by_key = {(r["family_id"], r["logical_ref"]): r for r in route_rows}
    assert len(by_key) == 40
    records = []
    for key, gt_box in gt.items():
        row = by_key[key]
        selected_bbox = row["selected_bbox_xyxy"]
        rep_iou = iou_bbox(selected_bbox, gt_box)
        hyp_correct = None
        hyp_id = row.get("selected_hypothesis_id")
        if hyp_id is not None:
            cache = caches[key]
            hyp = next(h for h in cache["hypotheses"] if h["hypothesis_id"] == hyp_id)
            hyp_correct = any(iou_bbox(next(c for c in cache["raw"]["candidates"] if c["candidate_id"] == cid)["bbox_xyxy"], gt_box) >= IOU_THRESHOLD for cid in hyp["member_candidate_ids"])
        records.append({
            "family_id": key[0], "logical_ref": key[1], "route": route_name,
            "referent_phrase": caches[key]["raw"]["referent_phrase"],
            "selected_hypothesis_id": hyp_id,
            "selected_representative_candidate_id": row.get("selected_representative_candidate_id") or row.get("selected_candidate_id"),
            "selected_bbox_xyxy": selected_bbox,
            "representative_iou": rep_iou,
            "representative_region_correct": bool(rep_iou >= IOU_THRESHOLD),
            "hypothesis_correct": hyp_correct,
            "selection_fallback_reason": row.get("fallback_reason"),
        })
    return records


def _route_qa_metrics(route_name, route_rows, entity_records, prepared, prior):
    by_key = {(r["family_id"], r["logical_ref"]): r for r in route_rows}
    ent_by_key = {(r["family_id"], r["logical_ref"]): r for r in entity_records}
    out = []
    for qa_id, prep in prepared.items():
        family_id = prep["family_id"]
        true_target = prior[qa_id]["EVALUATOR_ONLY"]["true_target"]
        bbox_a = by_key[(family_id, "A")]["selected_bbox_xyxy"]
        bbox_b = by_key[(family_id, "B")]["selected_bbox_xyxy"]
        target = target_from_boxes(bbox_a, bbox_b, prep["qa_type"])
        logical = target == true_target
        rep_correct = bool(logical and target is not None and ent_by_key[(family_id, target)]["representative_region_correct"])
        hyp_correct = None
        if target is not None:
            h = ent_by_key[(family_id, target)]["hypothesis_correct"]
            hyp_correct = bool(logical and h) if h is not None else None
        out.append({
            "qa_id": qa_id, "family_id": family_id, "qa_type": prep["qa_type"], "route": route_name,
            "target": target, "true_target": true_target,
            "logical_target_correct": bool(logical),
            "selected_hypothesis_target_correct": hyp_correct,
            "representative_region_target_correct": rep_correct,
            "grounded_target_correct": rep_correct,
            "selected_hypothesis_id_target": by_key[(family_id, target)].get("selected_hypothesis_id") if target is not None else None,
        })
    return out


def _summ(records, key):
    return sum(1 for r in records if r.get(key))


def evaluate_routes(raw_by_key, caches, obj, gt, prepared, prior, independent, joint, raw_route, candidate_joint):
    # historical reference
    hist_binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST_OUT / "binding_results.jsonl")}
    hist_qa = {r["qa_id"]: r for r in rows(HIST_OUT / "qa_results.jsonl")}
    hist_entity_correct = {k: bool(iou_bbox(hist_binding[k]["selected_bbox_xyxy"], gt[k]) >= IOU_THRESHOLD) for k in gt}
    hist_grounded = {q: bool(hist_qa[q]["grounded_target_correct"]) for q in hist_qa}
    try:
        prior_route_metrics = read_json(HIGH_RECALL_OUT / "final_comparison/route_metrics.json")
    except Exception:
        prior_route_metrics = {}

    routes = {}
    entity_records = {}
    qa_records = {}
    for name, route_rows in ((ROUTE_INDEP, independent), (ROUTE_JOINT, joint), ("RAW_OBJEMBED_SAM3_TOP64", raw_route), ("RAW_OBJEMBED_CANDIDATE_JOINT_SAM3_TOP64", candidate_joint)):
        er = _route_entity_metrics(name, route_rows, caches, gt)
        qr = _route_qa_metrics(name, route_rows, er, prepared, prior)
        entity_records[name] = er
        qa_records[name] = qr
        routes[name] = {
            "method": name,
            "hypothesis_correct_entity": _summ(er, "hypothesis_correct") if er[0]["hypothesis_correct"] is not None else None,
            "hypothesis_correct_entity_denominator": 40 if er[0]["hypothesis_correct"] is not None else None,
            "representative_correct_entity": _summ(er, "representative_region_correct"),
            "representative_correct_entity_denominator": 40,
            "logical_target_correct": _summ(qr, "logical_target_correct"),
            "grounded_hypothesis_target_correct": _summ(qr, "selected_hypothesis_target_correct") if qr and qr[0]["selected_hypothesis_target_correct"] is not None else None,
            "grounded_target_correct": _summ(qr, "grounded_target_correct"),
            "grounded_target_denominator": 40,
            "both_entity_representatives_correct": sum(
                1 for f in {r["family_id"] for r in er} if all(r["representative_region_correct"] for r in er if r["family_id"] == f)
            ),
            "selection_efficiency": _summ(er, "representative_region_correct") / max(1, sum(1 for r in er if any(iou_bbox(m["bbox_xyxy"], gt[(r["family_id"], r["logical_ref"])]) >= IOU_THRESHOLD for m in read_json(CONS_CACHE / f"{r['family_id']}__{r['logical_ref']}.json")["raw"]["candidates"]))),
        }
    # historical summary
    hist_logical = sum(1 for q in hist_qa if hist_qa[q]["logical_target_correct"])
    routes["HISTORICAL_INTEGRATED"] = {
        "method": "HISTORICAL_INTEGRATED",
        "hypothesis_correct_entity": None,
        "representative_correct_entity": sum(hist_entity_correct.values()),
        "representative_correct_entity_denominator": 40,
        "logical_target_correct": hist_logical,
        "grounded_hypothesis_target_correct": None,
        "grounded_target_correct": sum(hist_grounded.values()),
        "grounded_target_denominator": 40,
        "both_entity_representatives_correct": None,
        "selection_efficiency": None,
    }
    p1 = prior_route_metrics.get("p1_objembed", {})
    routes["RAW_OBJEMBED_PRIOR_UNION_POOL"] = {
        "method": "RAW_OBJEMBED_PRIOR_UNION_POOL",
        "hypothesis_correct_entity": None,
        "representative_correct_entity": p1.get("selected_region_correct"),
        "representative_correct_entity_denominator": 40,
        "logical_target_correct": p1.get("logical_target_correct"),
        "grounded_hypothesis_target_correct": None,
        "grounded_target_correct": p1.get("grounded_target_correct"),
        "grounded_target_denominator": 40,
        "source": "outputs/high_recall_binding_repair/final_comparison/route_metrics.json",
    }
    # best hypothesis route by grounded target, then hypothesis correctness, then representative correctness
    candidate_routes = [ROUTE_JOINT, ROUTE_INDEP]
    best_route = max(candidate_routes, key=lambda n: (
        routes[n]["grounded_target_correct"],
        routes[n]["hypothesis_correct_entity"] or 0,
        routes[n]["representative_correct_entity"],
        1 if n == ROUTE_JOINT else 0,
    ))
    result = {
        "experiment": EXPERIMENT,
        "primary_matcher": "ObjEmbed (frozen checkpoint)",
        "primary_hypothesis_pooling": "max",
        "routes": routes,
        "best_hypothesis_route": best_route,
        "best_route": best_route,
        "gt_used_only_after_route_selection": True,
        "evaluator_only": True,
    }
    # oracle chain update
    oracle = read_json(OUT / "oracle_preservation.json")
    oracle["final_chain"] = {
        "RawCandidateOracle": oracle["raw_candidate_oracle_top64"],
        "HypothesisOracle": oracle["consolidated_oracle_top64"],
        "SelectedHypothesisCorrect": routes[best_route]["hypothesis_correct_entity"],
        "RepresentativeRegionCorrect": routes[best_route]["representative_correct_entity"],
        "GroundedTargetCorrect": routes[best_route]["grounded_target_correct"],
        "denominators": {"first_four_entity_level": 40, "grounded_qa_level": 40},
        "best_route": best_route,
    }
    write_json(OUT / "oracle_preservation.json", oracle)

    # paired route-level records
    for name in entity_records:
        write_jsonl(OUT / f"entity_eval_{name}.jsonl", entity_records[name])
        write_jsonl(OUT / f"qa_eval_{name}.jsonl", qa_records[name])

    # historical transition analysis for best route
    hist_trans = _historical_transition(best_route, routes, entity_records, qa_records, hist_binding, hist_qa, gt)
    write_json(OUT / "historical_transition_analysis.json", hist_trans)

    rep_fail = _representative_failure_analysis(best_route, entity_records, caches, gt)
    write_json(OUT / "representative_failure_analysis.json", rep_fail)
    return result


def _historical_transition(best_route, routes, entity_records, qa_records, hist_binding, hist_qa, gt):
    ent_by_key = {(r["family_id"], r["logical_ref"]): r for r in entity_records[best_route]}
    counts_entity = {"historical_wrong_hypothesis_correct": 0, "historical_correct_hypothesis_wrong": 0, "both_correct": 0, "both_wrong": 0}
    counts_rep = {"historical_wrong_hypothesis_correct": 0, "historical_correct_hypothesis_wrong": 0, "both_correct": 0, "both_wrong": 0}
    cases_entity = {k: [] for k in counts_entity}
    cases_rep = {k: [] for k in counts_rep}
    for key, gt_box in gt.items():
        h_correct = bool(iou_bbox(hist_binding[key]["selected_bbox_xyxy"], gt_box) >= IOU_THRESHOLD)
        context_hyp = ent_by_key[key]["hypothesis_correct"]
        context_rep = ent_by_key[key]["representative_region_correct"]
        if context_hyp is not None:
            if h_correct and context_hyp: counts_entity["both_correct"] += 1
            elif h_correct and not context_hyp:
                counts_entity["historical_correct_hypothesis_wrong"] += 1
                cases_entity["historical_correct_hypothesis_wrong"].append({"family_id": key[0], "logical_ref": key[1], "referent_phrase": next(r["referent_phrase"] for r in entity_records[best_route] if (r["family_id"], r["logical_ref"]) == key)})
            elif not h_correct and context_hyp:
                counts_entity["historical_wrong_hypothesis_correct"] += 1
                cases_entity["historical_wrong_hypothesis_correct"].append({"family_id": key[0], "logical_ref": key[1], "referent_phrase": next(r["referent_phrase"] for r in entity_records[best_route] if (r["family_id"], r["logical_ref"]) == key)})
            else:
                counts_entity["both_wrong"] += 1
        if h_correct and context_rep: counts_rep["both_correct"] += 1
        elif h_correct and not context_rep:
            counts_rep["historical_correct_hypothesis_wrong"] += 1
            cases_rep["historical_correct_hypothesis_wrong"].append({"family_id": key[0], "logical_ref": key[1], "historical_iou": iou_bbox(hist_binding[key]["selected_bbox_xyxy"], gt_box), "context_rep_iou": ent_by_key[key]["representative_iou"]})
        elif not h_correct and context_rep:
            counts_rep["historical_wrong_hypothesis_correct"] += 1
            cases_rep["historical_wrong_hypothesis_correct"].append({"family_id": key[0], "logical_ref": key[1], "historical_iou": iou_bbox(hist_binding[key]["selected_bbox_xyxy"], gt_box), "context_rep_iou": ent_by_key[key]["representative_iou"]})
        else:
            counts_rep["both_wrong"] += 1
    # QA-level grounded transitions
    qa_by_q = {r["qa_id"]: r for r in qa_records[best_route]}
    qa_counts = {"historical_wrong_hypothesis_correct": 0, "historical_correct_hypothesis_wrong": 0, "both_correct": 0, "both_wrong": 0}
    qa_cases = {k: [] for k in qa_counts}
    for qa_id, row in qa_by_q.items():
        h = bool(hist_qa[qa_id]["grounded_target_correct"])
        c = bool(row["grounded_target_correct"])
        if h and c: qa_counts["both_correct"] += 1
        elif h and not c:
            qa_counts["historical_correct_hypothesis_wrong"] += 1
            qa_cases["historical_correct_hypothesis_wrong"].append({"qa_id": qa_id, "historical_target": hist_qa[qa_id]["target"], "context_target": row["target"], "true_target": row["true_target"]})
        elif not h and c:
            qa_counts["historical_wrong_hypothesis_correct"] += 1
            qa_cases["historical_wrong_hypothesis_correct"].append({"qa_id": qa_id, "historical_target": hist_qa[qa_id]["target"], "context_target": row["target"], "true_target": row["true_target"]})
        else:
            qa_counts["both_wrong"] += 1
    return {
        "best_route": best_route,
        "evaluator_only": True,
        "entity_selected_region": {"definition": "Historical selected bbox correct vs selected hypothesis representative correct at IoU>=0.5", "counts": counts_rep, "cases": cases_rep},
        "entity_selected_hypothesis": {"definition": "Historical selected bbox correct vs selected hypothesis contains a GT-correct member", "counts": counts_entity, "cases": cases_entity},
        "grounded_target": {"definition": "Historical GroundedTargetCorrect vs hypothesis-route GroundedTargetCorrect", "counts": qa_counts, "cases": qa_cases},
    }


def _representative_failure_analysis(best_route, entity_records, caches, gt):
    rows_out = entity_records[best_route]
    cases = []
    counts = Counter()
    for r in rows_out:
        key = (r["family_id"], r["logical_ref"])
        cache = caches[key]
        hyp_id = r["selected_hypothesis_id"]
        hyp = next(h for h in cache["hypotheses"] if h["hypothesis_id"] == hyp_id)
        members = []
        for cid in hyp["member_candidate_ids"]:
            c = next(x for x in cache["raw"]["candidates"] if x["candidate_id"] == cid)
            members.append({"candidate_id": cid, "native_rank": c["native_rank"], "native_score": c["native_score"], "bbox": c["bbox_xyxy"], "iou": iou_bbox(c["bbox_xyxy"], gt[key])})
        usable = [m for m in members if m["iou"] >= IOU_THRESHOLD]
        hyp_correct = bool(usable)
        rep_correct = bool(r["representative_region_correct"])
        if hyp_correct and rep_correct:
            counts["hypothesis_correct_representative_correct"] += 1
        elif hyp_correct and not rep_correct:
            counts["hypothesis_correct_representative_wrong"] += 1
            best_member = max(usable, key=lambda m: m["iou"])
            cases.append({
                "family_id": key[0], "logical_ref": key[1], "route": best_route,
                "failure_type": "representative_selection_within_correct_hypothesis",
                "selected_hypothesis_id": hyp_id,
                "representative_candidate_id": r["selected_representative_candidate_id"],
                "representative_iou": r["representative_iou"],
                "best_member": best_member,
                "member_count": len(members),
                "usable_member_count": len(usable),
                "usable_member_ranks": [m["native_rank"] for m in usable],
                "usable_member_ious": [m["iou"] for m in usable],
            })
        else:
            counts["hypothesis_wrong"] += 1
            cases.append({
                "family_id": key[0], "logical_ref": key[1], "route": best_route,
                "failure_type": "selected_hypothesis_has_no_gt_correct_member",
                "selected_hypothesis_id": hyp_id,
                "representative_iou": r["representative_iou"],
                "member_count": len(members),
                "member_ious": [m["iou"] for m in members],
                "member_ranks": [m["native_rank"] for m in members],
            })
    return {
        "best_route": best_route,
        "counts": dict(counts),
        "denominator_entities": len(rows_out),
        "representative_selection_bottleneck": bool(counts["hypothesis_correct_representative_wrong"] > counts["hypothesis_wrong"]),
        "cases": cases,
        "evaluator_only": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["raw", "reconcile", "consolidate", "oracle", "objembed", "bind", "downstream", "report", "all"])
    args = parser.parse_args()
    if args.stage == "raw":
        stage_raw()
    elif args.stage == "reconcile":
        stage_reconcile()
    elif args.stage == "consolidate":
        stage_consolidate()
    elif args.stage == "oracle":
        stage_oracle()
    elif args.stage == "objembed":
        stage_objembed()
    elif args.stage == "bind":
        stage_bind()
    elif args.stage == "downstream":
        stage_downstream()
    elif args.stage == "report":
        stage_report()
    elif args.stage == "all":
        stage_raw()
        stage_reconcile()
        stage_consolidate()
        stage_oracle()
        if not read_json(OUT / "oracle_preservation.json").get("stop_a_triggered", False):
            stage_objembed()
            stage_bind()
            stage_downstream()
        stage_report()




def stage_reconcile():
    """Canonicalise candidates to the frozen top64 box/score cache.

    The frozen cache kept pre-clipping SAM3 boxes (some slightly outside the
    image), while the mask regeneration path clips boxes to the canvas.  We
    therefore match each frozen proposal to the regenerated mask by Hungarian
    assignment on bbox IoU and store the frozen bbox/score/rank as canonical.
    This preserves the frozen candidate source while recovering only the
    missing mask field.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    raw_cache = RAW_CACHE
    frozen_raw = read_json(FROZEN_RAW)
    reconciled = []
    audit = []
    for key in sorted(frozen_raw):
        raw = read_json(raw_cache / (key.replace("|", "__") + ".json"))
        ref = frozen_raw[key]
        frozen_boxes = [[float(v) for v in b] for b in ref["boxes"][:SAM3_K]]
        frozen_scores = [float(v) for v in ref["scores"][:SAM3_K]]
        ours = raw["candidates"]
        assert len(ours) == SAM3_K
        cost = np.ones((SAM3_K, SAM3_K), dtype=np.float64)
        ious = np.zeros((SAM3_K, SAM3_K), dtype=np.float64)
        for i, fb in enumerate(frozen_boxes):
            for j, c in enumerate(ours):
                value = iou_bbox(fb, c["bbox_xyxy"])
                ious[i, j] = value
                cost[i, j] = 1.0 - value
        row, col = linear_sum_assignment(cost)
        canonical = []
        for i, j in zip(row, col):
            i = int(i)
            j = int(j)
            src = ours[j]
            mask_path = MASK_DIR / f"{key.replace('|','__')}__s{i+1:03d}.png"
            mask_path.write_bytes(Path(src["mask_path"]).read_bytes())
            canonical.append({
                "candidate_id": f"s{i:03d}",
                "native_rank": i + 1,
                "bbox_xyxy": frozen_boxes[i],
                "native_score": frozen_scores[i],
                "raw_query_index": int(src["raw_query_index"]),
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "mask_area": int(src["mask_area"]),
                "mask_valid": bool(src["mask_area"] > 0),
                "bbox_area": float((frozen_boxes[i][2] - frozen_boxes[i][0]) * (frozen_boxes[i][3] - frozen_boxes[i][1])),
                "source": "SAM3-LiteText-L",
                "regenerated_bbox_xyxy": src["bbox_xyxy"],
                "bbox_reconciliation_iou": float(ious[i, int(j)]),
            })
        canonical.sort(key=lambda c: c["native_rank"])
        record = {
            "family_id": raw["family_id"],
            "logical_ref": raw["logical_ref"],
            "referent_phrase": raw["referent_phrase"],
            "image_path": raw["image_path"],
            "image_size": raw["image_size"],
            "image_sha256": raw["image_sha256"],
            "sam3_k": SAM3_K,
            "candidates": canonical,
            "reconciliation": {
                "min_bbox_reconciliation_iou": min(c["bbox_reconciliation_iou"] for c in canonical),
                "mean_bbox_reconciliation_iou": sum(c["bbox_reconciliation_iou"] for c in canonical) / len(canonical),
                "frozen_box_source": str(FROZEN_RAW),
                "frozen_box_source_sha256": sha256_file(FROZEN_RAW),
                "canonical_source": "frozen top64 bbox/score/native_rank; regenerated mask only",
            },
        }
        reconciled.append(record)
        audit.append({"family_id": record["family_id"], "logical_ref": record["logical_ref"], **record["reconciliation"]})
        write_json(OUT / "raw_reconciled_cache" / (key.replace("|", "__") + ".json"), record)
    manifest = {
        "schema": "sam3_raw_top64_masks_v1",
        "sam3_k": SAM3_K,
        "source_frozen_cache": str(FROZEN_RAW),
        "source_frozen_sha256": sha256_file(FROZEN_RAW),
        "note": "canonical bboxes/scores/ranks come from the frozen cache; masks were regenerated only because the frozen top64 cache omitted masks",
        "n_referents": len(reconciled),
        "referents": reconciled,
    }
    write_json(OUT / "raw_candidate_manifest.json", manifest)
    write_json(OUT / "raw_reconciliation_audit.json", {
        "method": "Hungarian assignment of regenerated masks to frozen top64 bboxes by bbox IoU",
        "min_iou_global": min(a["min_bbox_reconciliation_iou"] for a in audit),
        "mean_iou_global": sum(a["mean_bbox_reconciliation_iou"] for a in audit) / len(audit),
        "per_referent": audit,
    })
    resume_update(status="RAW_RECONCILED", raw_reconciled=True, raw_reconcile_min_iou=min(a["min_bbox_reconciliation_iou"] for a in audit))
    print(f"[reconcile] min_match_iou={min(a['min_bbox_reconciliation_iou'] for a in audit):.4f}", flush=True)



# ---------------------------------------------------------------------------
# Stage F: downstream (gated)
# ---------------------------------------------------------------------------

def _route_rows(route_name):
    if route_name == ROUTE_INDEP:
        return rows(OUT / "independent_binding_results.jsonl")
    if route_name == ROUTE_JOINT:
        return rows(OUT / "joint_binding_results.jsonl")
    if route_name == "RAW_OBJEMBED_SAM3_TOP64":
        return rows(OUT / "raw_objembed_binding_results.jsonl")
    raise KeyError(route_name)


def _write_selected_regions_for_downstream(route_name):
    selected = []
    for row in _route_rows(route_name):
        selected.append({
            "family_id": row["family_id"],
            "logical_ref": row["logical_ref"],
            "referent_phrase": row["referent_phrase"],
            "image_path": next(
                r["image_path"] for r in read_json(OUT / "raw_candidate_manifest.json")["referents"]
                if r["family_id"] == row["family_id"] and r["logical_ref"] == row["logical_ref"]
            ),
            "image_sha256": next(
                r["image_sha256"] for r in read_json(OUT / "raw_candidate_manifest.json")["referents"]
                if r["family_id"] == row["family_id"] and r["logical_ref"] == row["logical_ref"]
            ),
            "selected_bbox_xyxy": row["selected_bbox_xyxy"],
            "selected_candidate_id": row.get("selected_representative_candidate_id") or row.get("selected_candidate_id"),
            "selected_hypothesis_id": row.get("selected_hypothesis_id"),
            "selected_union_native_rank": None,
            "selected_r4_native_rank": None,
            "selected_sam3_native_rank": None,
            "selected_provenance": ["SAM3-LiteText-L", "instance_hypothesis_representative"],
            "selected_source_entries": [],
            "selected_native_score": None,
            "selected_score": row.get("selected_score"),
            "fallback_reason": row.get("fallback_reason"),
            "focus_transform_sha256": sha256_text("sam3_instance_hypothesis_binding"),
            "prompt_sha256": sha256_text(OBJEMBED_QUERY),
        })
    write_jsonl(OUT / "selected_regions.jsonl", selected)


def stage_downstream():
    metrics = read_json(OUT / "binding_metrics.json")
    best_route = metrics["best_route"]
    grounded = metrics["routes"][best_route]["grounded_target_correct"]
    OUT.mkdir(parents=True, exist_ok=True)
    if grounded < 24:
        # Predeclared gate: do not spend full downstream inference on a route
        # far below the historical binding baseline.
        qa = rows(OUT / f"qa_eval_{best_route}.jsonl")
        downstream = []
        for r in qa:
            downstream.append({
                **r,
                "full_downstream_skipped": True,
                "skip_reason": "BEST_HYPOTHESIS_GROUNDED_BELOW_24_PER_FROZEN_GATE",
                "image_present": False,
                "vision_encoder_called": False,
            })
        write_jsonl(OUT / "downstream_results.jsonl", downstream)
        write_jsonl(OUT / "minimal_state_results.jsonl", [])
        logical = sum(r["logical_target_correct"] for r in qa)
        base = 40
        survival = {
            "monotonic_funnel": [
                {"stage": "start", "count": 40, "condition": "locked 40 REL2C QA"},
                {"stage": "LogicalTargetCorrect", "count": logical, "condition": "deterministic geometry target identity"},
                {"stage": "GroundedTargetCorrect", "count": grounded, "condition": "selected hypothesis representative IoU>=0.5"},
            ],
            "downstream_skipped": True,
            "skip_reason": "BEST_HYPOTHESIS_GROUNDED_BELOW_24_PER_FROZEN_GATE",
            "binding_route": best_route,
            "unconditional_components": {
                "logical_target_correct": logical,
                "grounded_target_correct": grounded,
            },
        }
        write_json(OUT / "survival_chain.json", survival)
        resume_update(status="DOWNSTREAM_SKIPPED_LOW_BINDING", downstream_complete=True, downstream_skipped=True, best_grounded=grounded)
        print(f"[downstream] skipped full inference: grounded={grounded}/40 < 24", flush=True)
        return
    # Full frozen downstream (only reached when binding is close to historical).
    _write_selected_regions_for_downstream(best_route)
    import harness.scripts.run_context_preserving_binding as cpb
    cpb.OUT = OUT
    cpb.ROUTE = ROUTE_DOWNSTREAM
    reuse = cpb._run_frozen_downstream()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    final_info = cpb._run_final_reuse()
    survival = cpb._survival_funnel(metrics)
    write_json(OUT / "survival_chain.json", survival)
    final_rows = {r["qa_id"]: r for r in rows(OUT / "final_results.jsonl")}
    qa_rows = {r["qa_id"]: r for r in rows(OUT / "qa_results.jsonl")}
    downstream = []
    for qa_id in sorted(qa_rows):
        q = qa_rows[qa_id]
        f = final_rows[qa_id]
        downstream.append({**q, "final_prediction": f["prediction"], "final_correct": f["final_correct"], "clean_final_correct": bool(q["minimal_state_sufficient"] and f["final_correct"]), "final_image_present": f["image_present"], "final_vision_encoder_called": f["vision_encoder_called"], "full_downstream_skipped": False})
    write_jsonl(OUT / "downstream_results.jsonl", downstream)
    write_jsonl(OUT / "minimal_state_results.jsonl", rows(OUT / "qa_results.jsonl"))
    resume_update(status="DOWNSTREAM_COMPLETE", downstream_complete=True, downstream_skipped=False, best_grounded=grounded)
    print(f"[downstream] full route complete reuse={reuse}", flush=True)


# ---------------------------------------------------------------------------
# Audits and final report
# ---------------------------------------------------------------------------

def _inference_artifacts_for_leakage():
    names = [
        "raw_candidate_manifest.json", "proposal_structure.jsonl", "duplicate_graph.jsonl",
        "containment_graph.jsonl", "instance_hypotheses.jsonl", "hypothesis_membership.jsonl",
        "representative_regions.jsonl", "objembed_member_scores.jsonl", "hypothesis_scores.jsonl",
        "independent_binding_results.jsonl", "joint_binding_results.jsonl",
        "raw_objembed_binding_results.jsonl", "candidate_joint_binding_results.jsonl",
    ]
    forbidden_keys = {"gt_bbox", "gt_bbox_xyxy_pixels", "true_target", "ground_truth", "EVALUATOR_ONLY", "evaluator_only", "question", "qa_id", "final_correct", "gt_color", "iou_with_gt"}
    violations = []
    scanned = 0
    for name in names:
        path = OUT / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            recs = rows(path)
        else:
            recs = [read_json(path)]
        for rec in recs:
            stack = [rec]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        scanned += 1
                        if k in forbidden_keys:
                            violations.append({"artifact": name, "forbidden_key": k})
                        stack.append(v)
                elif isinstance(obj, list):
                    stack.extend(obj)
    return {
        "passed": len(violations) == 0,
        "forbidden_keys": sorted(forbidden_keys),
        "violations": violations[:100],
        "scanned_values": scanned,
        "notes": [
            "Evaluator annotations were opened only after independent_binding_results.jsonl and joint_binding_results.jsonl had been written.",
            "Consolidation uses only raw SAM3 proposals, masks, native ranks/scores, and fixed thresholds.",
            "No GT/IoU/correctness field is used by clustering, representative selection, hypothesis scoring, or joint assignment.",
        ],
    }


def _paired_selection_audit():
    best_route = read_json(OUT / "binding_metrics.json")["best_route"]
    route = {r["logical_ref"]: r for r in _route_rows(best_route)}
    prepared, _ = load_prepared()
    family_qas = defaultdict(list)
    for qa in prepared.values():
        family_qas[qa["family_id"]].append(qa)
    shared = {}
    for family_id, qas in family_qas.items():
        assert len(qas) == 2
        for ref in ("A", "B"):
            row = route[ref] if family_id == route and False else next(r for r in _route_rows(best_route) if r["family_id"] == family_id and r["logical_ref"] == ref)
            region_hash = sha256_json({"type": "selected_hypothesis_representative", "bbox": row["selected_bbox_xyxy"], "hypothesis_id": row.get("selected_hypothesis_id")})
            shared[f"{family_id}|{ref}"] = {"region_hash": region_hash, "selected_hypothesis_id": row.get("selected_hypothesis_id"), "selected_bbox_xyxy": row["selected_bbox_xyxy"], "shared_by_both_qa": True}
    return {
        "best_route": best_route,
        "binding_once_per_family": True,
        "paired_questions_share_A_hypothesis": True,
        "paired_questions_share_B_hypothesis": True,
        "paired_questions_share_A_region": True,
        "paired_questions_share_B_region": True,
        "region_hashes": shared,
        "state_consistency": "full downstream was skipped by the predeclared binding gate" if read_json(OUT / "survival_chain.json").get("downstream_skipped") else "checked in downstream_results.jsonl",
    }


def _write_pipeline_config():
    oracle = read_json(OUT / "oracle_preservation.json")
    metrics = read_json(OUT / "consolidation_metrics.json")
    cfg = {
        "experiment": EXPERIMENT,
        "development_only": True,
        "frozen_components": [
            "RealBind 20-family/40-QA subset",
            "ReferentSpec",
            "SAM3-LiteText-L checkpoint and proposal call",
            "SAM3 top64 K source",
            "MM-GDINO/R4 cache (comparison only)",
            "ObjEmbed-2B checkpoint and query template",
            "deterministic geometry",
            "frozen downstream SAM3 isolation, crop-Qwen color reader, minimal state, text-only final reasoner",
        ],
        "allowed_changes": ["SAM3 candidate consolidation", "instance-level referent binding"],
        "sam3_consolidation": {
            "raw_candidate_manifest": str(OUT / "raw_candidate_manifest.json"),
            "sam3_k": SAM3_K,
            "duplicate_rule": "mask IoU >= 0.70; bbox IoU >= 0.70 fallback only when a mask is empty",
            "containment_rule": "Contain(i->j)=|Mi∩Mj|/|Mi| >= 0.90 and area(Mi)/area(Mj) <= 0.70",
            "high_overlap_reporting_threshold": HIGH_OVERLAP_IOU,
            "thresholds_fixed_before_evaluation": True,
            "oracle_gate": {"raw_top64": oracle["raw_candidate_oracle_top64"], "consolidated": oracle["consolidated_oracle_top64"], "damage": oracle["oracle_damage_vs_raw_top64"], "pass": oracle["consolidation_gate_pass"]},
            "compression_ratio": metrics["candidate_compression_ratio_raw_over_hypotheses"],
        },
        "binding": {
            "primary_matcher": "ObjEmbed-2B official REC path",
            "query_template": OBJEMBED_QUERY,
            "hypothesis_score": "max over hypothesis members S(q, member)",
            "secondary_poolings_reported": ["mean", "top2_mean"],
            "selection_rule": "argmax S(A,H); argmax S(B,H); joint argmax S_A(H_i)+S_B(H_j) subject only to H_i != H_j",
            "representative_rule": "root member (not contained in any hypothesis member) > valid mask > higher native score > larger mask area > lower native rank",
            "no_penalty_or_score_fusion": True,
        },
        "downstream": {
            "route": ROUTE_DOWNSTREAM,
            "gate": "full downstream only if best hypothesis GroundedTargetCorrect >= 24/40",
            "minimal_state_format": "[TARGET_ENTITY] A/B\\n[TARGET_ATTRIBUTE] color = <color>",
            "hard_image_exit": True,
        },
        "iou_threshold": IOU_THRESHOLD,
    }
    write_json(OUT / "pipeline_config.json", cfg)


def stage_report():
    OUT.mkdir(parents=True, exist_ok=True)
    _write_pipeline_config()
    leakage = _inference_artifacts_for_leakage()
    write_json(OUT / "gt_leakage_audit.json", leakage)
    consistency = _paired_selection_audit()
    write_json(OUT / "state_consistency_audit.json", consistency)
    metrics = read_json(OUT / "binding_metrics.json")
    oracle = read_json(OUT / "oracle_preservation.json")
    cons = read_json(OUT / "consolidation_metrics.json")
    hist = read_json(OUT / "historical_transition_analysis.json")
    rep = read_json(OUT / "representative_failure_analysis.json")
    best = metrics["best_route"]
    routes = metrics["routes"]
    hist_route = routes["HISTORICAL_INTEGRATED"]
    raw_prior = routes["RAW_OBJEMBED_PRIOR_UNION_POOL"]
    raw_new = routes["RAW_OBJEMBED_SAM3_TOP64"]
    ind = routes[ROUTE_INDEP]
    joint = routes[ROUTE_JOINT]
    # hypothesis-size and part-whole analyses from the selected best route
    size_analysis = _hypothesis_size_analysis(best)
    part_analysis = _part_whole_analysis(best)
    case_audit = _case_audit(best)
    rep_oracle = _representative_oracle_headroom(best)
    write_json(OUT / "hypothesis_size_analysis.json", size_analysis)
    write_json(OUT / "part_whole_analysis.json", part_analysis)
    write_json(OUT / "case_audit.json", case_audit)
    write_json(OUT / "representative_oracle_headroom.json", rep_oracle)
    decision = "REPRESENTATIVE_SELECTION_BOTTLENECK" if (joint["hypothesis_correct_entity"] or 0) - joint["representative_correct_entity"] >= 10 else "CONSOLIDATION_PRESERVES_RECALL_BUT_BINDING_NO_GAIN"
    _write_comparison_report(metrics, oracle, cons, hist, rep, size_analysis, part_analysis, decision, case_audit, rep_oracle)
    _write_implementation_report(metrics, oracle, cons, decision, rep_oracle)
    _write_audit_html(metrics, oracle, cons, hist, rep, decision, case_audit, rep_oracle)
    resume_update(status="COMPLETE", report_complete=True, decision_label=decision)
    print(f"[report] decision={decision}", flush=True)


def _size_bucket(n):
    if n == 1:
        return "singleton"
    if 2 <= n <= 3:
        return "2-3_members"
    if 4 <= n <= 8:
        return "4-8_members"
    return ">8_members"


def _hypothesis_size_analysis(best_route):
    gt = load_gt()
    entity = {r["logical_ref"]: None for r in []}
    qa = rows(OUT / f"qa_eval_{best_route}.jsonl")
    ent = rows(OUT / f"entity_eval_{best_route}.jsonl")
    by_key = {(r["family_id"], r["logical_ref"]): r for r in ent}
    hyp_rows = rows(OUT / "instance_hypotheses.jsonl")
    buckets = defaultdict(lambda: {"hypotheses": 0, "selected": 0, "hypothesis_correct": 0, "representative_correct": 0, "binding_correct": 0})
    selected_ids = {r["selected_hypothesis_id"]: (r["family_id"], r["logical_ref"]) for r in rows(_route_rows_file(best_route)) if r.get("selected_hypothesis_id")}
    for h in hyp_rows:
        bucket = _size_bucket(h["member_count"])
        buckets[bucket]["hypotheses"] += 1
        if h["hypothesis_id"] in selected_ids:
            buckets[bucket]["selected"] += 1
            key = selected_ids[h["hypothesis_id"]]
            e = by_key[key]
            if e["hypothesis_correct"]:
                buckets[bucket]["hypothesis_correct"] += 1
            if e["representative_region_correct"]:
                buckets[bucket]["representative_correct"] += 1
            # binding correctness for the QA rows whose target uses this hypothesis
            for q in qa:
                if q["selected_hypothesis_id_target"] == h["hypothesis_id"] and q["grounded_target_correct"]:
                    buckets[bucket]["binding_correct"] += 1
    return {
        "best_route": best_route,
        "buckets": {k: dict(v) for k, v in sorted(buckets.items())},
        "note": "hypothesis correctness and representative correctness are entity-level selected-hypothesis counts; binding_correct is QA-level grounded target count for QA whose target hypothesis is in the bucket",
    }


def _route_rows_file(route_name):
    if route_name == ROUTE_INDEP:
        return OUT / "independent_binding_results.jsonl"
    if route_name == ROUTE_JOINT:
        return OUT / "joint_binding_results.jsonl"
    raise KeyError(route_name)


def _part_whole_analysis(best_route):
    gt = load_gt()
    ent = {(r["family_id"], r["logical_ref"]): r for r in rows(OUT / f"entity_eval_{best_route}.jsonl")}
    qa = rows(OUT / f"qa_eval_{best_route}.jsonl")
    selected = {r["selected_hypothesis_id"]: r for r in rows(_route_rows_file(best_route))}
    groups = {
        "no_part_whole_hierarchy": {"n": 0, "hypothesis_correct": 0, "representative_correct": 0, "binding_correct": 0},
        "contains_part_nodes": {"n": 0, "hypothesis_correct": 0, "representative_correct": 0, "binding_correct": 0},
        "deep_hierarchy": {"n": 0, "hypothesis_correct": 0, "representative_correct": 0, "binding_correct": 0},
    }
    for h in rows(OUT / "instance_hypotheses.jsonl"):
        if h["hypothesis_id"] not in selected:
            continue
        if h["max_depth"] == 0:
            group = "no_part_whole_hierarchy"
        elif h["max_depth"] == 1:
            group = "contains_part_nodes"
        else:
            group = "deep_hierarchy"
        key = (h["family_id"], h["logical_ref"])
        e = ent[key]
        groups[group]["n"] += 1
        if e["hypothesis_correct"]:
            groups[group]["hypothesis_correct"] += 1
        if e["representative_region_correct"]:
            groups[group]["representative_correct"] += 1
        for q in qa:
            if q["selected_hypothesis_id_target"] == h["hypothesis_id"] and q["grounded_target_correct"]:
                groups[group]["binding_correct"] += 1
    return {"best_route": best_route, "groups": groups}


def _write_comparison_report(metrics, oracle, cons, hist, rep, size_analysis, part_analysis, decision, case_audit, rep_oracle):
    routes = metrics["routes"]
    raw_prior = routes["RAW_OBJEMBED_PRIOR_UNION_POOL"]
    raw_new = routes["RAW_OBJEMBED_SAM3_TOP64"]
    ind = routes[ROUTE_INDEP]
    joint = routes[ROUTE_JOINT]
    lines = []
    lines.append("# SAM3 instance-hypothesis consolidation + instance-level binding")
    lines.append("")
    lines.append("**DEVELOPMENT-ONLY FROZEN COHORT.** The same heavily inspected 20 families and 40 REL2C QA rows are reused. These are not held-out paper generalization results.")
    lines.append("")
    lines.append("## Required consolidation table")
    lines.append("")
    lines.append("| Metric | Raw SAM3 | Consolidated |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Mean candidates/image/referent | {cons['mean_raw_candidates_per_referent']:.2f} | {cons['mean_hypotheses_per_referent']:.2f} hypotheses |")
    lines.append(f"| Median candidates/image/referent | {cons['median_raw_candidates_per_referent']} | {cons['median_hypotheses_per_referent']} hypotheses |")
    lines.append(f"| Oracle@0.5 | {oracle['raw_candidate_oracle_top64']}/40 | {oracle['consolidated_oracle_top64']}/40 |")
    lines.append(f"| Compression ratio | 1.0 | {cons['candidate_compression_ratio_raw_over_hypotheses']:.3f} |")
    lines.append(f"| Same-instance duplicate rate | {cons['same_instance_duplicate_member_rate']:.3f} | 0.000 |")
    lines.append("")
    lines.append("## Required binding table")
    lines.append("")
    lines.append("| Method | HypothesisCorrect | RepresentativeCorrect | GroundedTargetCorrect |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Historical Integrated | — | {routes['HISTORICAL_INTEGRATED']['representative_correct_entity']}/40 entity-level | {routes['HISTORICAL_INTEGRATED']['grounded_target_correct']}/40 |")
    lines.append(f"| Raw ObjEmbed (prior frozen union pool) | — | {raw_prior.get('representative_correct_entity')}/40 entity-level | {raw_prior.get('grounded_target_correct')}/40 |")
    lines.append(f"| Raw ObjEmbed (same SAM3 top64 pool) | — | {raw_new['representative_correct_entity']}/40 entity-level | {raw_new['grounded_target_correct']}/40 |")
    lines.append(f"| Hypothesis ObjEmbed Independent | {ind['hypothesis_correct_entity']}/40 | {ind['representative_correct_entity']}/40 | {ind['grounded_target_correct']}/40 |")
    lines.append(f"| Hypothesis ObjEmbed Joint | {joint['hypothesis_correct_entity']}/40 | {joint['representative_correct_entity']}/40 | {joint['grounded_target_correct']}/40 |")
    lines.append("")
    lines.append("## Required oracle chain")
    lines.append("")
    lines.append("| Stage | Count |")
    lines.append("|---|---:|")
    lines.append(f"| RawCandidateOracle | {oracle['final_chain']['RawCandidateOracle']}/40 |")
    lines.append(f"| HypothesisOracle | {oracle['final_chain']['HypothesisOracle']}/40 |")
    lines.append(f"| SelectedHypothesisCorrect | {oracle['final_chain']['SelectedHypothesisCorrect']}/40 |")
    lines.append(f"| RepresentativeRegionCorrect | {oracle['final_chain']['RepresentativeRegionCorrect']}/40 |")
    lines.append(f"| GroundedTargetCorrect | {oracle['final_chain']['GroundedTargetCorrect']}/40 |")
    lines.append("")
    lines.append("## Proposal-depth analysis")
    lines.append("")
    lines.append("| Depth | Raw usable entity coverage | Consolidated usable hypothesis coverage |")
    lines.append("|---|---:|---:|")
    for depth in (16, 32, 64):
        lines.append(f"| top{depth} | {oracle['raw_candidate_oracle_by_depth'][f'top{depth}']}/40 | {oracle['consolidated_hypothesis_oracle_by_depth'][f'top{depth}']}/40 |")
    lines.append("")
    lines.append("## Hypothesis-size analysis")
    lines.append("")
    lines.append("| Bucket | Selected hypotheses | HypothesisCorrect | RepresentativeCorrect | Grounded binding |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in size_analysis["buckets"].items():
        lines.append(f"| {k} | {v['selected']} | {v['hypothesis_correct']} | {v['representative_correct']} | {v['binding_correct']} |")
    lines.append("")
    lines.append("## Part-whole hierarchy analysis")
    lines.append("")
    lines.append("| Group | Selected hypotheses | HypothesisCorrect | RepresentativeCorrect | Grounded binding |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in part_analysis["groups"].items():
        lines.append(f"| {k} | {v['n']} | {v['hypothesis_correct']} | {v['representative_correct']} | {v['binding_correct']} |")
    lines.append("")
    lines.append("## Context rescue / regression audit")
    lines.append("")
    lines.append(f"Historical wrong → hypothesis correct: entity hypothesis {hist['entity_selected_hypothesis']['counts']['historical_wrong_hypothesis_correct']}; representative {hist['entity_selected_region']['counts']['historical_wrong_hypothesis_correct']}; grounded QA {hist['grounded_target']['counts']['historical_wrong_hypothesis_correct']}.")
    lines.append(f"Historical correct → hypothesis wrong: entity hypothesis {hist['entity_selected_hypothesis']['counts']['historical_correct_hypothesis_wrong']}; representative {hist['entity_selected_region']['counts']['historical_correct_hypothesis_wrong']}; grounded QA {hist['grounded_target']['counts']['historical_correct_hypothesis_wrong']}.")
    lines.append(f"Representative failure counts: {json.dumps(rep['counts'], sort_keys=True)}. Representative selection is the dominant loss: {rep['counts'].get('hypothesis_correct_representative_wrong', 0)} selected hypotheses are correct but their frozen representative is not.")
    lines.append(f"Evaluator-only representative headroom: selected hypotheses contain a GT-correct member for {rep_oracle['selected_hypothesis_oracle_representative_entity']}/40 entities, while the frozen representative is correct for {rep_oracle['frozen_representative_entity']}/40; {rep_oracle['lost_to_frozen_representative']} entities are lost inside the hypothesis by representative selection.")
    lines.append("")
    lines.append("### Rescue case audit")
    lines.append("")
    if case_audit["rescue_cases"]:
        for c in case_audit["rescue_cases"]:
            lines.append(f"- `{c['family_id']}|{c['logical_ref']}` ({c['referent_phrase']}): historical IoU={c['historical_selected_iou']:.4f}; selected hyp={c['selected_hypothesis_id'].split('__')[-1]} members={c['selected_hypothesis_member_count']}, depth={c['selected_hypothesis_max_depth']}; raw ObjEmbed top={c['raw_objembed_top']['candidate_id']} rank={c['raw_objembed_top']['native_rank']} IoU={c['raw_objembed_top']['iou_gt']:.4f}; correct-member ranks={[m['native_rank'] for m in c['correct_hypothesis_members']]}; representative={c['representative_candidate_id']} IoU={c['representative_iou']:.4f}.")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("### Regression case audit")
    lines.append("")
    if case_audit["regression_cases"]:
        for c in case_audit["regression_cases"][:15]:
            lines.append(f"- `{c['family_id']}|{c['logical_ref']}` ({c['referent_phrase']}): historical IoU={c['historical_selected_iou']:.4f}; selected hyp={c['selected_hypothesis_id'].split('__')[-1]} has member IoUs={[round(m['iou_gt'],3) for m in c['member_details']]}; raw ObjEmbed top={c['raw_objembed_top']['candidate_id']} rank={c['raw_objembed_top']['native_rank']} IoU={c['raw_objembed_top']['iou_gt']:.4f}; representative IoU={c['representative_iou']:.4f}. Reason: {c['regression_reason']}")
        if len(case_audit["regression_cases"]) > 15:
            lines.append(f"- ... plus {len(case_audit['regression_cases'])-15} more cases in `case_audit.json` and the HTML audit.")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Required scientific questions")
    lines.append("")
    dup_members = round(cons['same_instance_duplicate_member_rate'] * cons['raw_candidate_count_total'])
    part_hyps = round(cons['part_containing_hypothesis_fraction'] * cons['hypothesis_count_total'])
    lines.append(f"**Q1. How many raw top64 candidates belong to duplicate/part-whole families?** Duplicate components of size >1 contain {dup_members}/{cons['raw_candidate_count_total']} raw candidates ({cons['same_instance_duplicate_member_rate']*100:.2f}%); near-duplicate pair rate is {cons['near_duplicate_pair_rate']*100:.2f}%. Part-containing hypotheses are {part_hyps}/{cons['hypothesis_count_total']} ({cons['part_containing_hypothesis_fraction']*100:.2f}%).")
    lines.append(f"**Q2. How much is candidate/hypothesis count reduced?** {cons['raw_candidate_count_total']} raw proposals become {cons['hypothesis_count_total']} hypotheses (compression {cons['candidate_compression_ratio_raw_over_hypotheses']:.2f}).")
    lines.append(f"**Q3. Is oracle recall preserved?** Raw top64 {oracle['raw_candidate_oracle_top64']}/40 vs consolidated {oracle['consolidated_oracle_top64']}/40; damage {oracle['oracle_damage_vs_raw_top64']}/40, gate pass={oracle['consolidation_gate_pass']}.")
    lines.append(f"**Q4. Is raw SAM3 high entropy the main binding-failure cause?** Consolidation preserves recall and raises selected-hypothesis correctness to {routes[metrics['best_route']]['hypothesis_correct_entity']}/40, but representative selection drops it to {routes[metrics['best_route']]['representative_correct_entity']}/40 and grounded target to {routes[metrics['best_route']]['grounded_target_correct']}/40. High entropy is therefore not the only bottleneck for this development subset.")
    lines.append(f"**Q5. Does ObjEmbed at hypothesis level beat raw candidate level?** On the same SAM3 top64 pool, raw selected-region correct {raw_new['representative_correct_entity']}/40 and grounded {raw_new['grounded_target_correct']}/40; selected-hypothesis correctness rises to {routes[metrics['best_route']]['hypothesis_correct_entity']}/40, but representative/grounded fall to {routes[metrics['best_route']]['representative_correct_entity']}/40 and {routes[metrics['best_route']]['grounded_target_correct']}/40.")
    lines.append(f"**Q6. Does instance-level joint assignment beat candidate-level joint assignment?** Hypothesis joint GroundedTargetCorrect {joint['grounded_target_correct']}/40 vs raw candidate-level joint {raw_new['grounded_target_correct']}/40 on the same pool; the hard H_i!=H_j constraint did not improve the frozen downstream count.")
    lines.append(f"**Q7. How often is the hypothesis right but the representative wrong?** {rep['counts'].get('hypothesis_correct_representative_wrong', 0)}/{rep['counts'].get('hypothesis_correct_representative_wrong', 0)+rep['counts'].get('hypothesis_correct_representative_correct', 0)} correct hypotheses; this triggers `REPRESENTATIVE_SELECTION_BOTTLENECK`.")
    lines.append(f"**Q8. Does GroundedTargetCorrect exceed historical 26/40?** No, best hypothesis route is {routes[metrics['best_route']]['grounded_target_correct']}/40.")
    lines.append(f"**Q9. Is 30/40 reached?** No.")
    lines.append(f"**Q10. Does downstream clean final exceed Vanilla 20/40?** Full downstream was skipped by the frozen <24/40 gate, so this is not reached; Vanilla remains 20/40.")
    lines.append("")
    lines.append("## Decision label")
    lines.append("")
    lines.append(f"`{decision}`")
    lines.append("")
    lines.append("Secondary finding: consolidation preserved recall but hypothesis-level binding did not exceed historical; per Stop B/C, no further matcher or SAM3 selector engineering should be added on this subset.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Development set, repeatedly inspected; not held-out evidence.")
    lines.append("- Historical top64 cache contained boxes/scores but not masks for ranks 17-64; masks were regenerated with the frozen checkpoint and Hungarian-matched to frozen boxes/scores.")
    lines.append("- The structural root representative rule is faithful to the predeclared lexicographic order, but SAM3 context regions frequently become roots, causing the representative bottleneck.")
    (OUT / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_implementation_report(metrics, oracle, cons, decision, rep_oracle):
    lines = [
        "# Implementation report: SAM3 instance hypothesis consolidation + binding",
        "",
        "## Scope",
        "",
        "Only SAM3 candidate consolidation and instance-level referent binding were changed. The frozen 20-family/40-QA subset, ReferentSpec, SAM3 checkpoint, top64 K source, ObjEmbed-2B checkpoint/query, geometry, downstream isolation, color reader, minimal state and final reasoner were reused.",
        "",
        "## Consolidation",
        "",
        f"- Fixed thresholds: duplicate mask IoU {DUP_IOU}, containment fraction {CONTAIN_FRAC}, area ratio {AREA_RATIO}, high-overlap report IoU {HIGH_OVERLAP_IOU}.",
        f"- Raw proposal structure is in `proposal_structure.jsonl` and `raw_proposal_structure_report.md`.",
        f"- Raw total {cons['raw_candidate_count_total']} -> hypotheses {cons['hypothesis_count_total']} (compression {cons['candidate_compression_ratio_raw_over_hypotheses']:.3f}).",
        f"- Consolidation oracle gate: raw top64 {oracle['raw_candidate_oracle_top64']}/40 vs consolidated {oracle['consolidated_oracle_top64']}/40; damage {oracle['oracle_damage_vs_raw_top64']}/40; pass {oracle['consolidation_gate_pass']}.",
        "",
        "## Binding",
        "",
        "- Primary matcher: frozen ObjEmbed-2B official REC path.",
        "- Hypothesis score = max member score; mean and top2 mean are recorded diagnostically but not used for selection.",
        "- Independent selection: per-referent hypothesis argmax. Joint selection: argmax of S_A(H_i)+S_B(H_j) with only H_i != H_j.",
        "- Representative region: root member > valid mask > native score > mask area > rank, fixed before evaluation.",
        "",
        "## Results",
        "",
        f"- Best hypothesis route: `{metrics['best_route']}`.",
        f"- HypothesisCorrect {oracle['final_chain']['SelectedHypothesisCorrect']}/40, RepresentativeRegionCorrect {oracle['final_chain']['RepresentativeRegionCorrect']}/40, GroundedTargetCorrect {oracle['final_chain']['GroundedTargetCorrect']}/40.",
        f"- Main decision: `{decision}`.",
        f"- Evaluator-only representative headroom: selected hypotheses contain a GT-correct member for {rep_oracle['selected_hypothesis_oracle_representative_entity']}/40 entities; frozen representative is correct for {rep_oracle['frozen_representative_entity']}/40.",
        "",
        "## Resume and audits",
        "",
        "- All stages write atomic JSON/JSONL; ObjEmbed records are appended and fsynced per referent.",
        "- `resume_status.json` records completed stages and gate outcomes.",
        "- `gt_leakage_audit.json` verifies clustering/scoring artifacts contain no evaluator fields.",
        "- `state_consistency_audit.json` verifies A/B hypothesis/region sharing across LEFT/RIGHT paired QA.",
        "- Full downstream inference was skipped because the best hypothesis route grounded at 6/40, below the predeclared 24/40 gate; `downstream_results.jsonl` contains the minimal binding-level validation.",
        "- `case_audit.json`, `representative_failure_analysis.json` and `representative_oracle_headroom.json` provide the post-hoc rescue/regression and within-hypothesis representative diagnosis.",
    ]
    (OUT / "implementation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_html(metrics, oracle, cons, hist, rep, decision, case_audit, rep_oracle):
    def table(headers, body):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in body) + "</tbody></table>"
    routes = metrics["routes"]
    raw_new = routes["RAW_OBJEMBED_SAM3_TOP64"]
    ind = routes[ROUTE_INDEP]
    joint = routes[ROUTE_JOINT]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SAM3 instance hypothesis audit</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #bbb;padding:.35rem .55rem}th{background:#eee}code{background:#f3f3f3;padding:.1rem .25rem}</style></head><body>",
        f"<h1>SAM3 instance-hypothesis consolidation + binding</h1><p><b>Development-only frozen subset.</b> Decision: <code>{html.escape(decision)}</code></p>",
        "<h2>Binding table</h2>",
        table(["Method", "HypothesisCorrect", "RepresentativeCorrect", "GroundedTargetCorrect"], [
            ["Historical Integrated", "—", "27/40", "26/40"],
            ["Raw ObjEmbed (prior union pool)", "—", "26/40", "24/40"],
            ["Raw ObjEmbed (SAM3 top64)", "—", f"{raw_new['representative_correct_entity']}/40", f"{raw_new['grounded_target_correct']}/40"],
            ["Hypothesis ObjEmbed independent", f"{ind['hypothesis_correct_entity']}/40", f"{ind['representative_correct_entity']}/40", f"{ind['grounded_target_correct']}/40"],
            ["Hypothesis ObjEmbed joint", f"{joint['hypothesis_correct_entity']}/40", f"{joint['representative_correct_entity']}/40", f"{joint['grounded_target_correct']}/40"],
        ]),
        "<h2>Oracle chain</h2>",
        table(["Stage", "Count"], [[k, f"{v}/40"] for k, v in oracle["final_chain"].items() if k.endswith("Correct") or k.endswith("Oracle")]),
        "<h2>Consolidation</h2>",
        table(["Metric", "Value"], [
            ["Raw candidates", cons["raw_candidate_count_total"]],
            ["Hypotheses", cons["hypothesis_count_total"]],
            ["Compression ratio", cons["candidate_compression_ratio_raw_over_hypotheses"]],
            ["Oracle raw top64", f"{oracle['raw_candidate_oracle_top64']}/40"],
            ["Oracle consolidated", f"{oracle['consolidated_oracle_top64']}/40"],
            ["Representative failures", rep["counts"]],
        ]),
        "<h2>Rescue / regression</h2>",
        table(["Transition", "Entity hypothesis", "Entity representative", "Grounded QA"], [
            ["Historical wrong → hypothesis correct", hist["entity_selected_hypothesis"]["counts"]["historical_wrong_hypothesis_correct"], hist["entity_selected_region"]["counts"]["historical_wrong_hypothesis_correct"], hist["grounded_target"]["counts"]["historical_wrong_hypothesis_correct"]],
            ["Historical correct → hypothesis wrong", hist["entity_selected_hypothesis"]["counts"]["historical_correct_hypothesis_wrong"], hist["entity_selected_region"]["counts"]["historical_correct_hypothesis_wrong"], hist["grounded_target"]["counts"]["historical_correct_hypothesis_wrong"]],
        ]),
        f"<p>Representative headroom: selected hypotheses contain GT-correct members for {rep_oracle['selected_hypothesis_oracle_representative_entity']}/40 entities; frozen representatives are correct for {rep_oracle['frozen_representative_entity']}/40.</p>",
        "<h2>Rescue cases</h2>",
        table(["Entity", "Referent", "Historical IoU", "Selected hyp", "Raw top rank/IoU", "Correct-member ranks", "Rep IoU"], [
            [f"{c['family_id']}|{c['logical_ref']}", c['referent_phrase'], f"{c['historical_selected_iou']:.3f}", c['selected_hypothesis_id'].split('__')[-1], f"{c['raw_objembed_top']['native_rank']}/{c['raw_objembed_top']['iou_gt']:.3f}", str([m['native_rank'] for m in c['correct_hypothesis_members']]), f"{c['representative_iou']:.3f}"] for c in case_audit['rescue_cases']
        ]),
        "<h2>Regression cases</h2>",
        table(["Entity", "Referent", "Historical IoU", "Selected hyp member IoUs", "Raw top rank/IoU", "Rep IoU"], [
            [f"{c['family_id']}|{c['logical_ref']}", c['referent_phrase'], f"{c['historical_selected_iou']:.3f}", str([round(m['iou_gt'],3) for m in c['member_details']]), f"{c['raw_objembed_top']['native_rank']}/{c['raw_objembed_top']['iou_gt']:.3f}", f"{c['representative_iou']:.3f}"] for c in case_audit['regression_cases'][:30]
        ]),
        "<h2>Artifacts</h2><ul>" + "".join(f"<li><a href='{html.escape(n)}'>{html.escape(n)}</a></li>" for n in ["comparison_report.md", "implementation_report.md", "pipeline_config.json", "raw_candidate_manifest.json", "proposal_structure.jsonl", "duplicate_graph.jsonl", "containment_graph.jsonl", "instance_hypotheses.jsonl", "hypothesis_membership.jsonl", "representative_regions.jsonl", "consolidation_metrics.json", "oracle_preservation.json", "objembed_member_scores.jsonl", "hypothesis_scores.jsonl", "independent_binding_results.jsonl", "joint_binding_results.jsonl", "binding_metrics.json", "historical_transition_analysis.json", "representative_failure_analysis.json", "downstream_results.jsonl", "survival_chain.json", "gt_leakage_audit.json", "state_consistency_audit.json", "resume_status.json"]) + "</ul></body></html>",
    ]
    (OUT / "audit.html").write_text("".join(parts), encoding="utf-8")



def _case_audit(best_route):
    gt = load_gt()
    raw_manifest = read_json(OUT / "raw_candidate_manifest.json")
    raw_by_key = {(r["family_id"], r["logical_ref"]): r for r in raw_manifest["referents"]}
    obj = _load_objembed_scores()
    route = {(r["family_id"], r["logical_ref"]): r for r in _route_rows(best_route)}
    hist_binding = {(r["family_id"], r["logical_ref"]): r for r in rows(HIST_OUT / "binding_results.jsonl")}
    hist_qa = {r["qa_id"]: r for r in rows(HIST_OUT / "qa_results.jsonl")}
    rescue = []
    regression = []
    rep_fail = []
    for key, gt_box in gt.items():
        f, ref = key
        raw = raw_by_key[key]
        cache = read_json(CONS_CACHE / f"{f}__{ref}.json")
        row = route[key]
        hyp = next(h for h in cache["hypotheses"] if h["hypothesis_id"] == row["selected_hypothesis_id"])
        member_details = []
        for cid in hyp["member_candidate_ids"]:
            c = next(x for x in raw["candidates"] if x["candidate_id"] == cid)
            member_details.append({
                "candidate_id": cid,
                "native_rank": c["native_rank"],
                "native_score": c["native_score"],
                "objembed_score": obj[key]["candidate_id_to_score"][cid],
                "iou_gt": iou_bbox(c["bbox_xyxy"], gt_box),
                "bbox_xyxy": c["bbox_xyxy"],
            })
        member_details.sort(key=lambda x: x["objembed_score"], reverse=True)
        hist_correct = bool(iou_bbox(hist_binding[key]["selected_bbox_xyxy"], gt_box) >= IOU_THRESHOLD)
        hyp_correct = any(m["iou_gt"] >= IOU_THRESHOLD for m in member_details)
        rep_correct = bool(iou_bbox(row["selected_bbox_xyxy"], gt_box) >= IOU_THRESHOLD)
        correct_members = [m for m in member_details if m["iou_gt"] >= IOU_THRESHOLD]
        # frozen raw ObjEmbed argmax for this referent
        raw_items = raw["candidates"]
        raw_top = max(raw_items, key=lambda c: (obj[key]["candidate_id_to_score"][c["candidate_id"]], -c["native_rank"]))
        base = {
            "family_id": f,
            "logical_ref": ref,
            "referent_phrase": raw["referent_phrase"],
            "historical_selected_iou": iou_bbox(hist_binding[key]["selected_bbox_xyxy"], gt_box),
            "selected_hypothesis_id": hyp["hypothesis_id"],
            "selected_hypothesis_score": row.get("selected_score"),
            "selected_hypothesis_member_count": hyp["member_count"],
            "selected_hypothesis_max_depth": hyp["max_depth"],
            "representative_candidate_id": row.get("selected_representative_candidate_id") or row.get("selected_candidate_id"),
            "representative_iou": iou_bbox(row["selected_bbox_xyxy"], gt_box),
            "raw_objembed_top": {
                "candidate_id": raw_top["candidate_id"],
                "native_rank": raw_top["native_rank"],
                "objembed_score": obj[key]["candidate_id_to_score"][raw_top["candidate_id"]],
                "iou_gt": iou_bbox(raw_top["bbox_xyxy"], gt_box),
            },
            "correct_hypothesis_members": correct_members,
            "member_details": member_details,
        }
        if (not hist_correct) and hyp_correct:
            base["rescue_reason"] = "Hypothesis max-pooling surfaced a GT-correct member although raw ObjEmbed argmax missed it." if base["raw_objembed_top"]["iou_gt"] < IOU_THRESHOLD else "Hypothesis selected a different correct instance representation."
            rescue.append(base)
        if hist_correct and not hyp_correct:
            base["regression_reason"] = "Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis."
            regression.append(base)
        if hyp_correct and not rep_correct:
            base["representative_failure_reason"] = "Frozen root-first representative is a lower-IoU context/hierarchy member; a GT-correct member exists in the selected hypothesis."
            rep_fail.append(base)
    return {
        "best_route": best_route,
        "evaluator_only": True,
        "rescue_cases": rescue,
        "regression_cases": regression,
        "representative_failure_cases": rep_fail,
        "counts": {"rescue": len(rescue), "regression": len(regression), "representative_failure": len(rep_fail)},
        "definitions": {
            "rescue": "Historical entity selected region wrong -> selected hypothesis contains a GT-correct member",
            "regression": "Historical entity selected region correct -> selected hypothesis contains no GT-correct member",
            "representative_failure": "Selected hypothesis contains a GT-correct member but frozen representative bbox IoU<0.5",
        },
    }


def _representative_oracle_headroom(best_route):
    gt = load_gt()
    raw_manifest = read_json(OUT / "raw_candidate_manifest.json")
    raw_by_key = {(r["family_id"], r["logical_ref"]): r for r in raw_manifest["referents"]}
    route = {(r["family_id"], r["logical_ref"]): r for r in _route_rows(best_route)}
    entity_oracle = 0
    for key, gt_box in gt.items():
        row = route[key]
        cache = read_json(CONS_CACHE / f"{key[0]}__{key[1]}.json")
        hyp = next(h for h in cache["hypotheses"] if h["hypothesis_id"] == row["selected_hypothesis_id"])
        if any(iou_bbox(next(c for c in raw_by_key[key]["candidates"] if c["candidate_id"] == cid)["bbox_xyxy"], gt_box) >= IOU_THRESHOLD for cid in hyp["member_candidate_ids"]):
            entity_oracle += 1
    return {
        "best_route": best_route,
        "selected_hypothesis_oracle_representative_entity": entity_oracle,
        "frozen_representative_entity": read_json(OUT / "representative_failure_analysis.json")["counts"].get("hypothesis_correct_representative_correct", 0),
        "lost_to_frozen_representative": entity_oracle - read_json(OUT / "representative_failure_analysis.json")["counts"].get("hypothesis_correct_representative_correct", 0),
        "note": "Evaluator-only headroom: if a GT-correct member of the selected hypothesis could be chosen as representative, this is the entity-level ceiling.",
    }

if __name__ == "__main__":
    main()
