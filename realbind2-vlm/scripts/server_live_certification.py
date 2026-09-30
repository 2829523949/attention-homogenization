#!/usr/bin/env python3
"""Server-only live certification for Qwen, MM-GDINO, SAM3 and hooks.

This module is intentionally not called by the local artifact smoke.  Every
live stage requires CUDA, configured local checkpoints and a certified frozen
RealBind2 export.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.server_certification_common import (  # noqa: E402
    certification_root,
    certification_identity,
    require_live_receipt,
    cuda_begin,
    cuda_end,
    exact_crop_box,
    frozen_dataset_root,
    frozen_rows,
    image_path,
    load_server_config,
    profile_record,
    read_jsonl,
    resolve_smoke_families,
    sha256_file,
    smoke_manifest,
    write_json_atomic,
    write_jsonl_atomic,
)
from src.common.dataset import RealBind2Dataset  # noqa: E402
from src.common.logging_utils import JsonlRunLog  # noqa: E402
from src.common.scorer import COLORS, score_prediction  # noqa: E402
from src.models.qwen35 import Qwen35Adapter  # noqa: E402

SELECTOR_PROMPT = "Which numbered crop best matches: {referent}? Answer only the number."
COLOR_PROMPT = (
    "What color is the main object or region shown?\n"
    "Answer with only one color from this list: red, orange, yellow, green, "
    "blue, purple, pink, brown, black, white, gray."
)
MASK_MIN_BBOX_FRAC = 0.05
MASK_MAX_BBOX_FRAC = 1.25
MASK_MIN_BBOX_COVERAGE = 0.50
MASK_DILATION_RADIUS = 16


def qwen_adapter(config: dict[str, Any]) -> Qwen35Adapter:
    return Qwen35Adapter(
        config["models"]["qwen35_4b"],
        device="cuda",
        dtype="bfloat16",
        use_cache=False,
        max_new_tokens=16,
    )


def question_index(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    _, questions = frozen_rows(config)
    return {(str(row["family_id"]), str(row["qa_type"])): row for row in questions}


def check_paths(config: dict[str, Any]) -> None:
    manifest = json.loads((ROOT / "data" / "realbind2_manifest.json").read_text())
    if not manifest.get("frozen"):
        raise RuntimeError("DATASET_NOT_FROZEN: complete human acceptance/export and certify release manifest first")
    mandatory_models = ("qwen35_4b", "mm_gdino_l", "sam3")
    missing = [key for key in mandatory_models if not Path(config["models"][key]).expanduser().is_dir()]
    missing_python = [key for key, value in config["python"].items() if not Path(value).expanduser().is_file()]
    if missing or missing_python:
        raise FileNotFoundError({"missing_models": missing, "missing_python": missing_python})
    audit = RealBind2Dataset(frozen_dataset_root(config)).audit()
    if not audit.passed:
        raise RuntimeError(f"frozen RealBind2 integrity failed: {audit}")
    resolved = resolve_smoke_families(config)
    if len(resolved) != len(smoke_manifest()["families"]):
        raise RuntimeError("smoke family resolution mismatch")
    out = certification_root(config)
    out.mkdir(parents=True, exist_ok=True)
    probe = out / ".write_probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    print("PATH_CHECK_PASS")


def check_env(component: str) -> None:
    import torch
    import transformers

    expected = {"main_vlm": "5.8.1", "mm_gdino": "4.55.4", "sam3": "5.8.1"}[component]
    print(f"component={component} transformers={transformers.__version__} torch={torch.__version__}")
    if transformers.__version__ != expected:
        raise RuntimeError(f"{component} requires transformers=={expected}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{component} CUDA unavailable")
    print(f"ENV_CHECK_PASS component={component}")


def live_qwen(config: dict[str, Any]) -> None:
    families = resolve_smoke_families(config)
    questions = question_index(config)
    output = certification_root(config) / "qwen" / "results.jsonl"
    log = JsonlRunLog(output)
    adapter = qwen_adapter(config)
    started = cuda_begin()
    for family in families:
        qa = questions[(str(family["family_id"]), "PAIR_REL_A_TO_B")]
        case_id = str(qa["qa_id"])
        if log.is_complete(case_id):
            continue
        raw = adapter.generate(adapter.prepare_prompt(str(qa["question"]), image_path(config, family)))
        score = score_prediction(raw, str(qa["ground_truth"]), str(qa["answer_type"]))
        log.append({
            "case_id": case_id,
            "family_id": str(family["family_id"]),
            "candidate_family_id": str(family.get("candidate_family_id", "")),
            "image_id": str(family["image_id"]),
            "experiment": "server_live_qwen_smoke",
            "condition": "original_image",
            "prediction_raw": raw,
            "prediction_normalized": score.normalized,
            "ground_truth": score.ground_truth,
            "correct": score.correct,
            "valid": score.valid,
            "status": "ok",
        })
    runtime, peak = cuda_end(started)
    profile_record(config, "Qwen baseline", runtime, peak)
    before = output.read_bytes()
    resumed = JsonlRunLog(output)
    expected = {str(questions[(str(f["family_id"]), "PAIR_REL_A_TO_B")]["qa_id"]) for f in families}
    if resumed.completed != expected or output.read_bytes() != before:
        raise RuntimeError("Qwen logging/resume certification failed")
    print("LIVE_QWEN_SMOKE_PASS")


def _bounded_box(box: list[float], width: int, height: int) -> list[float]:
    bounded = [
        min(max(float(box[0]), 0.0), float(width)),
        min(max(float(box[1]), 0.0), float(height)),
        min(max(float(box[2]), 0.0), float(width)),
        min(max(float(box[3]), 0.0), float(height)),
    ]
    if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
        raise ValueError(f"degenerate candidate: {bounded}")
    return bounded


def live_mm_gdino(config: dict[str, Any]) -> None:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"transformers.__version__={transformers.__version__}")
    if transformers.__version__ != "4.55.4":
        raise RuntimeError("MM-GDINO certification requires transformers==4.55.4")
    family = resolve_smoke_families(config)[0]
    source = image_path(config, family)
    model_path = Path(config["models"]["mm_gdino_l"])
    output = certification_root(config) / "mm_gdino" / "candidates.jsonl"
    existing = {(row["family_id"], row["logical_ref"]): row for row in read_jsonl(output)} if output.exists() else {}
    started = cuda_begin()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32
    ).to("cuda").eval()
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    for ref in ("A", "B"):
        key = (str(family["family_id"]), ref)
        if key in existing:
            continue
        phrase = str(family[f"referent_{ref}"])
        inputs = processor(images=image, text=[[phrase]], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = list(result.get("text_labels") or result.get("labels") or [])
        candidates = []
        for raw_index, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
            confidence = float(score)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise RuntimeError("MM-GDINO returned invalid native confidence")
            candidates.append({
                "candidate_index": raw_index,
                "raw_candidate_index": raw_index,
                "bbox_xyxy": _bounded_box(box.tolist(), image.width, image.height),
                "native_score": confidence,
                "matched_phrase": str(labels[raw_index]) if raw_index < len(labels) else "",
                "backend_source": "mm_grounding_dino_large_all",
            })
        candidates.sort(key=lambda row: (-row["native_score"], row["raw_candidate_index"]))
        candidates = candidates[:16]
        if not candidates:
            raise RuntimeError(f"MM-GDINO produced no candidate for {ref}")
        existing[key] = {
            "case_id": f"{family['family_id']}|{ref}",
            "family_id": str(family["family_id"]),
            "candidate_family_id": str(family.get("candidate_family_id", "")),
            "logical_ref": ref,
            "referent_phrase": phrase,
            "image_path": str(source),
            "image_size": [image.width, image.height],
            "candidates": candidates,
            "candidate_retention_policy": "native_score_top16_ties_raw_index",
            "transformers_version": transformers.__version__,
            "status": "ok",
        }
    write_jsonl_atomic(output, [existing[key] for key in sorted(existing)])
    runtime, peak = cuda_end(started)
    profile_record(config, "MM-GDINO", runtime, peak)
    if len(existing) < 2:
        raise RuntimeError("MM-GDINO smoke did not cover both A and B")
    print("LIVE_MM_GDINO_SMOKE_PASS")


def _sam3_model(config: dict[str, Any]):
    import torch
    from transformers import AutoModel, AutoProcessor

    path = Path(config["models"]["sam3"])
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(
        path, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
    ).to("cuda").eval()
    return processor, model


def _mask_stats(mask, bbox: list[float], size: tuple[int, int]) -> dict[str, float | int]:
    import numpy as np

    width, height = size
    value = np.asarray(mask).astype(bool)
    area = int(value.sum())
    x0, y0, x1, y1 = exact_crop_box(size, bbox)
    support = np.zeros((height, width), dtype=bool)
    support[y0:y1, x0:x1] = True
    inside = int((value & support).sum())
    bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return {
        "mask_area": area,
        "bbox_area": int(bbox_area),
        "mask_area_over_bbox_area": area / bbox_area if bbox_area else 0.0,
        "mask_fraction_inside_bbox": inside / area if area else 0.0,
    }


def _sam3_box_mask(processor, model, image, phrase: str, bbox: list[float]):
    import torch

    image_inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        vision = model.get_vision_features(pixel_values=image_inputs["pixel_values"].to("cuda"))
    cx = (bbox[0] + bbox[2]) / 2.0 / image.width
    cy = (bbox[1] + bbox[3]) / 2.0 / image.height
    width = (bbox[2] - bbox[0]) / image.width
    height = (bbox[3] - bbox[1]) / image.height
    text = processor(text=[phrase], return_tensors="pt")
    inputs = {
        "vision_embeds": vision,
        "input_ids": text["input_ids"].to("cuda"),
        "attention_mask": text["attention_mask"].to("cuda"),
        "input_boxes": torch.tensor([[[cx, cy, width, height]]], dtype=torch.float32, device="cuda"),
        "input_boxes_labels": torch.tensor([[1]], dtype=torch.long, device="cuda"),
    }
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs, threshold=0.0, mask_threshold=0.5, target_sizes=[image.size[::-1]]
    )[0]
    if len(result["scores"]) == 0:
        raise RuntimeError("SAM3 returned no masks")
    order = sorted(range(len(result["scores"])), key=lambda index: (-float(result["scores"][index]), index))
    best = None
    best_value = -1e9
    for index in order:
        mask = result["masks"][index].to(torch.uint8).cpu().numpy() > 0
        stats = _mask_stats(mask, bbox, image.size)
        value = float(result["scores"][index]) + 0.2 * float(stats["mask_fraction_inside_bbox"])
        if value > best_value:
            best_value = value
            best = (mask, float(result["scores"][index]), stats)
    return best


def _mask_is_valid(stats: dict[str, float | int]) -> bool:
    fraction = float(stats["mask_area_over_bbox_area"])
    coverage = float(stats["mask_fraction_inside_bbox"])
    return bool(
        int(stats["mask_area"]) > 0
        and MASK_MIN_BBOX_FRAC <= fraction <= MASK_MAX_BBOX_FRAC
        and coverage >= MASK_MIN_BBOX_COVERAGE
    )


def live_sam3(config: dict[str, Any], *, strong: bool = False) -> None:
    import numpy as np
    from PIL import Image, ImageFilter

    family = resolve_smoke_families(config)[0]
    source = image_path(config, family)
    if strong:
        selected_path = certification_root(config) / "strong_integrated" / "selected_targets.jsonl"
        selected = read_jsonl(selected_path)
        input_rows = {row["logical_ref"]: row for row in selected}
        out_root = certification_root(config) / "strong_integrated"
        output = out_root / "mask_results.jsonl"
    else:
        candidates = read_jsonl(certification_root(config) / "mm_gdino" / "candidates.jsonl")
        input_rows = {
            row["logical_ref"]: {
                "logical_ref": row["logical_ref"],
                "referent_phrase": row["referent_phrase"],
                "selected_bbox_xyxy": row["candidates"][0]["bbox_xyxy"],
            }
            for row in candidates if row["family_id"] == str(family["family_id"])
        }
        out_root = certification_root(config) / "sam3"
        output = out_root / "results.jsonl"
    if set(input_rows) != {"A", "B"}:
        raise RuntimeError("SAM3 inputs do not contain persistent A/B regions")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(exist_ok=True)
    (out_root / "isolated").mkdir(exist_ok=True)
    (out_root / "crops").mkdir(exist_ok=True)
    processor, model = _sam3_model(config)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    started = cuda_begin()
    rows = []
    for ref in ("A", "B"):
        input_row = input_rows[ref]
        bbox = [float(value) for value in input_row["selected_bbox_xyxy"]]
        if strong:
            # Invoke the preserved historical support check; SAM3 never replaces
            # R4 as the primary candidate-grid selector.
            import torch
            sys.path.insert(0, str(ROOT / "legacy" / "vlm_repair_v2"))
            from src.repair.retrospective.strong_integrated_40case_historical import (
                _top_candidates_from_sam3, iou,
            )
            image_inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                vision = model.get_vision_features(pixel_values=image_inputs["pixel_values"].to("cuda"))
            support_pool, _, _ = _top_candidates_from_sam3(
                processor, model, vision, str(input_row["referent_phrase"]), image.size, 16
            )
            input_row["r4_selected_bbox_xyxy"] = bbox
            support = max((iou(bbox, c["bbox_xyxy"]) for c in support_pool), default=None)
            input_row["sam3_support_iou"] = support
            input_row["fallback_used"] = False
            if support is not None and support < 0.30 and input_row["top_native_score"] < 0.30:
                bbox = support_pool[0]["bbox_xyxy"]
                input_row["selected_bbox_xyxy"] = bbox
                input_row["selected_provider"] = "SAM3_LITETEXT_TOP1_FALLBACK"
                input_row["fallback_used"] = True
                input_row["fallback_reason"] = "R4_LOW_SAM3_SUPPORT_LOW_NATIVE_SCORE"
            del vision, support_pool
        result = _sam3_box_mask(processor, model, image, str(input_row["referent_phrase"]), bbox)
        if result is None:
            raise RuntimeError("SAM3 did not return a selectable mask")
        mask, score, stats = result
        if mask.shape != (image.height, image.width) or int(mask.sum()) == 0:
            raise RuntimeError("SAM3 mask dimensions/non-empty gate failed")
        if float(stats["mask_fraction_inside_bbox"]) < MASK_MIN_BBOX_COVERAGE:
            raise RuntimeError("SAM3 mask coordinate-alignment gate failed")
        mask_path = out_root / "masks" / f"{family['family_id']}__{ref}.png"
        Image.fromarray(mask.astype("uint8") * 255).save(mask_path, format="PNG")
        crop_box = exact_crop_box(image.size, bbox)
        crop_path = out_root / "crops" / f"{family['family_id']}__{ref}.png"
        image.crop(crop_box).save(crop_path, format="PNG")
        valid = _mask_is_valid(stats)
        isolated_path = None
        if valid:
            dilated = Image.fromarray(mask.astype("uint8") * 255).filter(
                ImageFilter.MaxFilter(2 * MASK_DILATION_RADIUS + 1)
            )
            mask_crop = dilated.crop(crop_box)
            crop = image.crop(crop_box)
            isolated = Image.composite(crop, Image.new("RGB", crop.size, (127, 127, 127)), mask_crop)
            isolated_path = out_root / "isolated" / f"{family['family_id']}__{ref}.png"
            isolated.save(isolated_path, format="PNG")
        rows.append({
            "case_id": f"{family['family_id']}|{ref}",
            "family_id": str(family["family_id"]),
            "logical_ref": ref,
            "referent_phrase": str(input_row["referent_phrase"]),
            "selected_bbox_xyxy": bbox,
            "selected_provider": input_row.get("selected_provider", "MM_GDINO_TOP1_SMOKE_ONLY"),
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "mask_score": score,
            "mask_stats": stats,
            "mask_valid": valid,
            "crop_path": str(crop_path),
            "isolated_path": str(isolated_path) if isolated_path else None,
            "isolation_used": valid,
            "status": "ok",
        })
    write_jsonl_atomic(output, rows)
    if strong:
        write_jsonl_atomic(selected_path, [input_rows[ref] for ref in ("A", "B")])
    runtime, peak = cuda_end(started)
    profile_record(config, "Strong Integrated SAM3" if strong else "SAM3", runtime, peak)
    print("STRONG_MASK_PASS" if strong else "LIVE_SAM3_SMOKE_PASS")


def _grid(image, candidates: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    tile, pad, columns = 224, 28, 4
    count = len(candidates)
    grid = Image.new("RGB", (columns * tile, math.ceil(count / columns) * (tile + pad)), "white")
    draw = ImageDraw.Draw(grid)
    for index, candidate in enumerate(candidates):
        crop = image.crop(exact_crop_box(image.size, candidate["bbox_xyxy"]))
        crop.thumbnail((tile - 12, tile - 12))
        x = (index % columns) * tile + (tile - crop.width) // 2
        y = (index // columns) * (tile + pad) + pad + (tile - crop.height) // 2
        grid.paste(crop, (x, y))
        draw.text(((index % columns) * tile + 8, (index // columns) * (tile + pad) + 5), str(index + 1), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path, format="PNG")


def strong_select(config: dict[str, Any]) -> None:
    from PIL import Image

    family = resolve_smoke_families(config)[0]
    source = image_path(config, family)
    candidates = [
        row for row in read_jsonl(certification_root(config) / "mm_gdino" / "candidates.jsonl")
        if row["family_id"] == str(family["family_id"])
    ]
    if {row["logical_ref"] for row in candidates} != {"A", "B"}:
        raise RuntimeError("Strong Integrated requires MM-GDINO A/B candidate pools")
    out = certification_root(config) / "strong_integrated"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(out / "candidates.jsonl", candidates)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    adapter = qwen_adapter(config)
    selected = []
    started = cuda_begin()
    for row in sorted(candidates, key=lambda value: value["logical_ref"]):
        pool = row["candidates"]
        if not pool:
            raise RuntimeError("R4 candidate pool empty")
        if len(pool) == 1:
            index, raw, status = 0, None, "SINGLETON_NO_QWEN_CALL"
        else:
            grid = out / "selector_grids" / f"{family['family_id']}__{row['logical_ref']}.png"
            _grid(image, pool, grid)
            raw = adapter.generate(
                adapter.prepare_prompt(SELECTOR_PROMPT.format(referent=row["referent_phrase"]), grid)
            )
            match = re.fullmatch(r"\s*(\d+)\s*", raw)
            if match and 1 <= int(match.group(1)) <= len(pool):
                index, status = int(match.group(1)) - 1, "VALID"
            else:
                index, status = 0, "QWEN_SELECTOR_INVALID_FALLBACK_NATIVE_TOP1"
        selected.append({
            "case_id": f"{family['family_id']}|{row['logical_ref']}",
            "family_id": str(family["family_id"]),
            "logical_ref": row["logical_ref"],
            "referent_phrase": row["referent_phrase"],
            "candidate_count": len(pool),
            "selected_index": index,
            "selected_bbox_xyxy": pool[index]["bbox_xyxy"],
            "selected_native_score": pool[index]["native_score"],
            "top_native_score": pool[0]["native_score"],
            "selected_provider": "R4_MM_GDINO_L_QWEN_SELECTOR",
            "qwen_status": status,
            "qwen_raw_output": raw,
            "persistent_entity_state": True,
            "question_provided": False,
            "target_side_provided": False,
            "ground_truth_provided": False,
            "status": "ok",
        })
    write_jsonl_atomic(out / "selected_targets.jsonl", selected)
    runtime, peak = cuda_end(started)
    profile_record(config, "Strong Integrated selector", runtime, peak)
    print("STRONG_SELECT_PASS")


def strong_finish(config: dict[str, Any]) -> None:
    family = resolve_smoke_families(config)[0]
    questions = question_index(config)
    out = certification_root(config) / "strong_integrated"
    selected = {row["logical_ref"]: row for row in read_jsonl(out / "selected_targets.jsonl")}
    masks = {row["logical_ref"]: row for row in read_jsonl(out / "mask_results.jsonl")}
    if set(selected) != {"A", "B"} or set(masks) != {"A", "B"}:
        raise RuntimeError("Strong Integrated persistent A/B state incomplete")
    adapter = qwen_adapter(config)
    started = cuda_begin()
    attributes = []
    for ref in ("A", "B"):
        # Preserve historical strict crop-color parsing and bbox fallback.
        crop_input = Path(masks[ref]["crop_path"])
        crop_raw = adapter.generate(adapter.prepare_prompt(COLOR_PROMPT, crop_input))
        crop_color = crop_raw.strip().casefold()
        crop_color = crop_color if crop_color in COLORS else None
        isolated_raw = None
        isolated_color = None
        if masks[ref]["isolated_path"]:
            isolated_raw = adapter.generate(adapter.prepare_prompt(COLOR_PROMPT, Path(masks[ref]["isolated_path"])))
            value = isolated_raw.strip().casefold()
            isolated_color = value if value in COLORS else None
        image_input = Path(masks[ref]["isolated_path"]) if isolated_color else crop_input
        raw = isolated_raw if isolated_color else crop_raw
        color = isolated_color or crop_color
        if color is None:
            raise RuntimeError(f"crop-Qwen returned invalid color for {ref}")
        attributes.append({
            "case_id": f"{family['family_id']}|{ref}",
            "family_id": str(family["family_id"]),
            "logical_ref": ref,
            "reader": "crop_qwen35",
            "input_kind": "sam3_isolated" if isolated_color else "bbox_crop_fallback",
            "input_path": str(image_input),
            "prediction_raw": raw,
            "bbox_crop_generation": crop_raw,
            "mask_isolated_generation": isolated_raw,
            "color": color,
            "composition_ground_truth_provided": False,
            "status": "ok",
        })
    write_jsonl_atomic(out / "attributes.jsonl", attributes)
    attribute_by_ref = {row["logical_ref"]: row for row in attributes}
    ax = (selected["A"]["selected_bbox_xyxy"][0] + selected["A"]["selected_bbox_xyxy"][2]) / 2.0
    bx = (selected["B"]["selected_bbox_xyxy"][0] + selected["B"]["selected_bbox_xyxy"][2]) / 2.0
    if abs(ax - bx) <= 1e-9:
        raise RuntimeError("deterministic geometry produced tied centers")
    left, right = ("A", "B") if ax < bx else ("B", "A")
    states = []
    audits = []
    for qa_type, target in (("REL2C_LEFT", left), ("REL2C_RIGHT", right)):
        qa = questions[(str(family["family_id"]), qa_type)]
        state_text = f"[TARGET_ENTITY] {target}\n[TARGET_ATTRIBUTE] color = {attribute_by_ref[target]['color']}"
        record = {
            "case_id": str(qa["qa_id"]),
            "qa_id": str(qa["qa_id"]),
            "family_id": str(family["family_id"]),
            "qa_type": qa_type,
            "logical_target": target,
            "minimal_state": state_text,
            "image_present": False,
            "state_source": "MM-GDINO-L -> Qwen selector -> persistent A/B -> SAM3 isolation -> crop-Qwen",
            "status": "ok",
        }
        states.append(record)
        final_prompt = str(qa["question"]) + "\n\nAcquired visual facts:\n" + state_text
        prepared = adapter.prepare_prompt(final_prompt, image_path=None)
        visual_keys = [key for key in prepared if any(token in key.casefold() for token in ("pixel", "image", "video"))]
        if visual_keys:
            raise RuntimeError(f"final reasoner received visual tensors: {visual_keys}")
        raw = adapter.generate(prepared)
        scored = score_prediction(raw, str(qa["ground_truth"]), str(qa["answer_type"]))
        audits.append({
            **record,
            "prediction_raw": raw,
            "prediction_normalized": scored.normalized,
            "ground_truth": scored.ground_truth,
            "correct": scored.correct,
            "valid": scored.valid,
            "final_reasoner_image_path": None,
            "final_reasoner_visual_input_keys": visual_keys,
            "final_reasoner_no_image": True,
        })
    write_jsonl_atomic(out / "minimal_states.jsonl", states)
    write_jsonl_atomic(out / "audit.jsonl", audits)
    runtime, peak = cuda_end(started)
    profile_record(config, "Strong Integrated Qwen", runtime, peak)
    profile_path = certification_root(config) / "runtime_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    components = profile["components"]
    names = ("Strong Integrated selector", "Strong Integrated SAM3", "Strong Integrated Qwen")
    profile_record(
        config,
        "Strong Integrated",
        sum(float(components[name]["runtime_sec"]) for name in names),
        max(float(components[name]["peak_gpu_memory_mb"]) for name in names),
    )
    print("LIVE_STRONG_INTEGRATED_PASS")


def _discover_layers(model):
    import torch.nn as nn

    config = getattr(model.config, "text_config", model.config)
    count = int(config.num_hidden_layers)
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) == count
    ]
    candidates.sort(key=lambda item: ("language_model" not in item[0], len(item[0])))
    if not candidates:
        raise RuntimeError("decoder ModuleList not found")
    name, layers = candidates[0]
    return name, list(layers)


def causal_selfpatch(config: dict[str, Any]) -> None:
    import torch

    family = resolve_smoke_families(config)[0]
    qa = question_index(config)[(str(family["family_id"]), "PAIR_REL_A_TO_B")]
    adapter = qwen_adapter(config)
    prepared = adapter.prepare_prompt(str(qa["question"]), image_path(config, family))
    model = adapter.model
    module_name, layers = _discover_layers(model)
    layer_index = len(layers) // 2
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().clone()
        return output

    def forward():
        try:
            return model(**prepared, return_dict=True, use_cache=False, logits_to_keep=1).logits.detach()
        except TypeError:
            return model(**prepared, return_dict=True, use_cache=False).logits[:, -1:, :].detach()

    started = cuda_begin()
    handle = layers[layer_index].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            clean = forward()
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("self-patch capture hook did not execute")

    def patch(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape != captured["hidden"].shape:
            raise RuntimeError("self-patch tensor shape mismatch")
        replacement = captured["hidden"].to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement

    handle = layers[layer_index].register_forward_hook(patch)
    try:
        with torch.inference_mode():
            patched = forward()
    finally:
        handle.remove()
    runtime, peak = cuda_end(started)
    exact = torch.equal(clean, patched)
    maximum = float((clean.float() - patched.float()).abs().max().item())
    if not exact or maximum != 0.0 or int(clean.argmax()) != int(patched.argmax()):
        raise RuntimeError("causal self-patch changed model behavior")
    report = {
        "passed": True,
        "case_id": str(qa["qa_id"]),
        "decoder_module_list": module_name,
        "num_layers": len(layers),
        "layer_index": layer_index,
        "layer_module_path": f"{module_name}.{layer_index}",
        "layer_class": layers[layer_index].__class__.__name__,
        "captured_shape": list(captured["hidden"].shape),
        "logits_shape": list(clean.shape),
        "max_abs_logit_diff": maximum,
        "logits_exact": exact,
        "top1_preserved": int(clean.argmax()) == int(patched.argmax()),
    }
    write_json_atomic(certification_root(config) / "causal" / "selfpatch.json", report)
    profile_record(config, "Qwen causal", runtime, peak)
    print("LIVE_CAUSAL_SELFPATCH_PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "check-paths", "check-env", "qwen", "mm-gdino", "sam3",
            "strong-select", "strong-mask", "strong-finish", "causal-selfpatch",
            "certify-receipt", "require-receipt",
        ),
    )
    parser.add_argument("--component", choices=("main_vlm", "mm_gdino", "sam3"))
    args = parser.parse_args()
    config = load_server_config(args.paths)
    if args.stage == "check-paths":
        check_paths(config)
    elif args.stage == "check-env":
        if not args.component:
            parser.error("--component is required for check-env")
        check_env(args.component)
    elif args.stage == "qwen":
        live_qwen(config)
    elif args.stage == "mm-gdino":
        live_mm_gdino(config)
    elif args.stage == "sam3":
        live_sam3(config)
    elif args.stage == "strong-select":
        strong_select(config)
    elif args.stage == "strong-mask":
        live_sam3(config, strong=True)
    elif args.stage == "strong-finish":
        strong_finish(config)
    elif args.stage == "causal-selfpatch":
        causal_selfpatch(config)
    elif args.stage == "certify-receipt":
        write_json_atomic(certification_root(config) / "live_certification.json", {
            "status": "LIVE_SMOKE_PASS", "identity": certification_identity(config),
        })
    elif args.stage == "require-receipt":
        require_live_receipt(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
