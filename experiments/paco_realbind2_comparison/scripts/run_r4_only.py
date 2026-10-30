#!/usr/bin/env python3
"""R4-only repair on the frozen PACO/RealBind2-conditioned set.

Strict method:
  MM-GDINO-L candidate pool
  -> target-model numbered-grid selector
  -> target-model bbox-crop attribute reader (exact R4 COLOR_PROMPT)
  -> deterministic bbox-center geometry
  -> minimal two-line state
  -> target-model text-only final answer

No SAM3, no full image at final, no chain-of-thought, no label leakage.
The conditioned set is frozen by the primary selector before this script.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from common import (
    COLOR_PROMPT, atomic_json, code_identity, generation_call, iou_xyxy, jsonl,
    load_adapter, load_model_spec, load_paths, r4_protocol, scorer, select_candidates,
    sha256_file, stable_hash, write_jsonl_atomic,
)


def resolve_image_root(paths_cfg: dict, dataset_dir: Path) -> Path:
    return Path(paths_cfg["dataset_root"]).expanduser().resolve() if not dataset_dir.is_absolute() else dataset_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, default=None, help="defaults to paths.dataset_root")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit-families", type=int, default=None)
    ap.add_argument("--allow-primary-selection-drift", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    paths = load_paths(args.paths)
    model_cfg, runtime, model_path = load_model_spec(args.model_tag, paths)
    dataset_dir = args.dataset_dir.resolve()
    images_root = (args.images_root or Path(paths["dataset_root"])).expanduser().resolve()
    split_spec = json.loads((dataset_dir / "split_spec.json").read_text())
    attribute_prompt = split_spec.get("task_semantics", {}).get("attribute_prompt", COLOR_PROMPT)
    families = jsonl(dataset_dir / "families_conditioned.jsonl")
    benchmark = jsonl(dataset_dir / "benchmark.jsonl")
    if args.limit_families is not None:
        families = families[: args.limit_families]
    family_by_id = {f["family_id"]: f for f in families}
    questions_by_family = defaultdict(list)
    for q in benchmark:
        if q["family_id"] in family_by_id:
            questions_by_family[q["family_id"]].append(q)
    missing_questions = [fid for fid in family_by_id if not questions_by_family[fid]]
    if missing_questions:
        raise ValueError(f"conditioned families without questions: {missing_questions[:5]}")
    candidates = {r["task_id"]: r for r in jsonl(args.candidates)}
    missing_candidates = [f"{fid}|{ref}" for fid in family_by_id for ref in ("A", "B") if f"{fid}|{ref}" not in candidates]
    if missing_candidates:
        raise RuntimeError(f"candidate pools missing for {len(missing_candidates)} conditioned tasks, e.g. {missing_candidates[:5]}")

    adapter = load_adapter(model_cfg, model_path, runtime)
    score_prediction = scorer()
    protocol = r4_protocol()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "selector_grids").mkdir(exist_ok=True)
    (out / "attribute_crops").mkdir(exist_ok=True)
    result_path = out / "results.jsonl"
    existing_rows = jsonl(result_path) if result_path.is_file() else []
    existing = {r["qa_id"]: r for r in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("duplicate completed QA in existing R4 results")
    selection_path = out / "selected_targets.jsonl"
    selection_rows = {r["task_id"]: r for r in jsonl(selection_path)} if selection_path.is_file() else {}
    identity = {
        "schema": "paco-realbind2-r4-only-v1",
        "model_tag": args.model_tag,
        "model_id": model_cfg["model_id"],
        "model_config": model_cfg,
        "adapter_metadata": adapter.get_model_metadata(),
        "dataset_dir": str(dataset_dir),
        "images_root": str(images_root),
        "benchmark_sha256": sha256_file(dataset_dir / "benchmark.jsonl"),
        "families_conditioned_sha256": sha256_file(dataset_dir / "families_conditioned.jsonl"),
        "split_spec_sha256": sha256_file(dataset_dir / "split_spec.json"),
        "candidates_path": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates),
        "method": "r4_only",
        "mode": "live",
        "selector_prompt": "Which numbered crop best matches: {referent}? Answer only the number.",
        "color_prompt": attribute_prompt,
        "r4_protocol_version": protocol["VERSION"],
        "geometry_epsilon": protocol["EPSILON"],
        "final_text_only": True,
        "release_root": str(Path(paths["release_root"]).resolve()),
        "code_identity": code_identity(),
    }
    identity_path = out / "identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("resume identity mismatch; use a fresh output directory")
    else:
        atomic_json(identity_path, identity)
    if existing and any(r.get("model_tag") != args.model_tag or
                        r.get("run_benchmark_sha256") != identity["benchmark_sha256"] or
                        r.get("method") != "r4_only" for r in existing.values()):
        raise ValueError("existing result rows do not match current model/benchmark identity")

    primary_tag = paths["conditioning"]["primary_selector_tag"]
    if args.model_tag == primary_tag and not args.allow_primary_selection_drift:
        expected_selection = {f["family_id"]: (f["r4_selected_bbox_A_xyxy"], f["r4_selected_bbox_B_xyxy"])
                              for f in families if "r4_selected_bbox_A_xyxy" in f}

    started_all = time.monotonic()
    processed_families = 0
    for family in families:
        fid = family["family_id"]
        qs = questions_by_family[fid]
        if all(q["qa_id"] in existing for q in qs):
            continue
        image_path = images_root / family["image_relpath"]
        if not image_path.is_file() or sha256_file(image_path) != family["image_sha256"]:
            raise ValueError(f"image missing/hash mismatch: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        selected = {}
        family_selection = {}
        for ref in ("A", "B"):
            task_key = f"{fid}|{ref}"
            pool = candidates[task_key]["candidates"]
            selected[ref] = select_candidates(
                adapter, image, pool, family[f"referent_{ref}"],
                out / "selector_grids" / f"{fid}__{ref}.png",
            )
            family_selection[task_key] = {
                "task_id": task_key, "family_id": fid, "logical_ref": ref,
                "referent_phrase": family[f"referent_{ref}"],
                "candidate_count": selected[ref]["candidate_count"],
                "selected_index": selected[ref].get("selected_index"),
                "selected_bbox_xyxy": selected[ref].get("selected_bbox_xyxy"),
                "selected_native_score": selected[ref].get("selected_native_score"),
                "qwen_status": selected[ref]["selector_status"],
                "qwen_raw_output": selected[ref].get("selector_raw"),
                "qwen_generation_error": selected[ref].get("selector_generation_error"),
                "status": selected[ref]["status"],
                "selection_model_tag": args.model_tag,
                "selection_model_id": model_cfg["model_id"],
            }
            selection_rows[task_key] = family_selection[task_key]

        if args.model_tag == primary_tag and not args.allow_primary_selection_drift:
            got_a = selected["A"].get("selected_bbox_xyxy")
            got_b = selected["B"].get("selected_bbox_xyxy")
            exp_a, exp_b = expected_selection[fid]
            same = got_a is not None and got_b is not None and all(
                abs(float(x) - float(y)) <= 1e-6 for x, y in zip(got_a + got_b, exp_a + exp_b)
            )
            if not same:
                raise RuntimeError(
                    f"primary R4 selector drifted from frozen conditioning selection for {fid}: "
                    f"got A={got_a} B={got_b}, expected A={exp_a} B={exp_b}"
                )

        # R4-only uses a bbox crop, never SAM3 isolation.
        masks = {}
        for ref in ("A", "B"):
            sel = selected[ref]
            if sel["status"] != "ok" or not sel.get("selected_bbox_xyxy"):
                masks[ref] = {"status": "NO_SELECTED_TARGET", "crop_path": None,
                              "crop_hash": None, "attribute_raw": "", "attribute_normalized": None,
                              "attribute_valid": False, "attribute_source": "NONE"}
                continue
            crop_dir = out / "attribute_crops" / fid / ref
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / "bbox.png"
            crop = image.crop(protocol["crop_box"](image.size, sel["selected_bbox_xyxy"]))
            crop.save(crop_path, format="PNG")
            attr_call = generation_call(adapter, attribute_prompt, image_path=str(crop_path))
            normalized = protocol["parse_attribute"](attr_call["raw"])
            masks[ref] = {
                "status": "ok",
                "crop_path": str(crop_path),
                "crop_hash": sha256_file(crop_path),
                "attribute_raw": attr_call["raw"],
                "attribute_normalized": normalized,
                "attribute_valid": normalized is not None,
                "attribute_source": "R4_BBOX_CROP",
                "attribute_generation_error": attr_call["generation_error"],
                "attribute_call_cost": {"elapsed_seconds": attr_call["elapsed_seconds"], "model_calls": 1},
            }

        a, b = selected["A"], selected["B"]
        if a["status"] == "ok" and b["status"] == "ok":
            geometry = protocol["geometry"](a["selected_bbox_xyxy"], b["selected_bbox_xyxy"],
                                            int(family["width"]), protocol["EPSILON"])
        else:
            geometry = {"status": "GROUNDING_FAILED", "left": None, "right": None}

        left_is_a = family["bbox_A_xyxy"][0] + family["bbox_A_xyxy"][2] < family["bbox_B_xyxy"][0] + family["bbox_B_xyxy"][2]
        for q in qs:
            if q["qa_id"] in existing:
                continue
            target = geometry.get("left" if q["qa_type"].endswith("LEFT") else "right")
            selected_target = selected.get(target) if target in {"A", "B"} else None
            target_mask = masks.get(target) if target in {"A", "B"} else None
            target_attr = target_mask["attribute_normalized"] if target_mask else None
            state = protocol["minimal_state"](q, target, target_attr)
            final_call = generation_call(adapter, protocol["final_prompt"](state), image_path=None, text_only=True)
            score = score_prediction(final_call["raw"], q["answer"], "color")
            actual_target = "A" if left_is_a else "B"
            logical = target == actual_target
            actual_bbox = family[f"bbox_{actual_target}_xyxy"]
            grounded = bool(logical and selected_target and selected_target.get("selected_bbox_xyxy") and
                            iou_xyxy(selected_target["selected_bbox_xyxy"], actual_bbox) >= 0.5)
            attribute_correct = bool(target_attr is not None and target_attr == q["answer"])
            geometry_valid = geometry.get("status") == "VALID"
            sufficient = bool(logical and grounded and attribute_correct and geometry_valid)
            row = {
                "qa_id": q["qa_id"],
                "family_id": fid,
                "paco_split": q["paco_split"],
                "qa_type": q["qa_type"],
                "question": q["question"],
                "question_sha256": q["question_sha256"],
                "image_relpath": q["image_relpath"],
                "image_sha256": q["image_sha256"],
                "method": "r4_only",
                "mode": "live",
                "model_id": model_cfg["model_id"],
                "model_tag": args.model_tag,
                "prediction_raw": final_call["raw"],
                "prediction_normalized": score.normalized,
                "ground_truth": score.ground_truth,
                "correct": bool(score.correct),
                "valid": bool(score.valid),
                "scorer_error": score.error,
                "scorer_version": "realbind2-semantic-v1",
                "generation_error": final_call["generation_error"],
                "run_benchmark_sha256": identity["benchmark_sha256"],
                "target_role": target,
                "competitor_role": state["competitor_role"],
                "logical_target_correct": bool(logical),
                "r4_target_grounded": bool(grounded),
                "attribute_correct": bool(attribute_correct),
                "geometry_valid": bool(geometry_valid),
                "minimal_state_sufficient": bool(sufficient),
                "clean_correct": bool(sufficient and score.correct),
                "compensating_correct": bool((not sufficient) and score.correct),
                "stages": {
                    "selected": selected,
                    "selection": family_selection,
                    "geometry": geometry,
                    "attributes": masks,
                    "target_mask": target_mask,
                    "minimal": state,
                    "final_call": final_call,
                },
                "cost": {
                    "model_calls": int(2 if any(len(candidates[f"{fid}|{r}"]["candidates"]) > 1 for r in ("A", "B")) else 0)
                                  + sum(1 for r in ("A", "B") if masks[r]["status"] == "ok") + 1,
                    "elapsed_seconds": final_call["elapsed_seconds"] + sum(
                        x.get("attribute_call_cost", {}).get("elapsed_seconds", 0.0)
                        for x in masks.values()
                    ),
                },
            }
            existing[q["qa_id"]] = row
        processed_families += 1
        write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["qa_id"]))
        write_jsonl_atomic(selection_path, sorted(selection_rows.values(), key=lambda r: r["task_id"]))
        print(json.dumps({"family": fid, "processed_families": processed_families,
                          "completed_qa": len(existing), "elapsed_seconds": round(time.monotonic()-started_all, 1)}), flush=True)
    write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["qa_id"]))
    write_jsonl_atomic(selection_path, sorted(selection_rows.values(), key=lambda r: r["task_id"]))
    expected_qas = {q["qa_id"] for fid in family_by_id for q in questions_by_family[fid]}
    missing = expected_qas - set(existing)
    if missing:
        raise RuntimeError(f"R4 output incomplete: {len(missing)} missing QAs")
    atomic_json(out / "status.json", {"status": "COMPLETE", "n": len(existing), "families": len(family_by_id)})
    print(json.dumps({"status": "COMPLETE", "n": len(existing), "families": len(family_by_id)}))


if __name__ == "__main__":
    main()
