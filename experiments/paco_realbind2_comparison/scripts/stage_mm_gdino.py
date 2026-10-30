#!/usr/bin/env python3
"""Stage 1: MM-GDINO-L candidate pools for PACO A/B referents.

This copies the frozen RealBind2 R4 grounding semantics:
  - one image + one referent phrase per call;
  - threshold=.25, text_threshold=.25;
  - boxes bounded to the image; degenerate boxes dropped;
  - candidates ordered by native score then raw detector index;
  - top 16 retained.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

from common import (
    atomic_json, bounded_box, code_identity, ensure_release_on_path, jsonl,
    load_paths, load_yaml, sha256_file, stable_hash, write_jsonl_atomic,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--families", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--expected-transformers", default="4.55.4")
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if transformers.__version__ != args.expected_transformers:
        raise RuntimeError(
            f"MM-GDINO stage requires transformers=={args.expected_transformers}; "
            f"found {transformers.__version__}. Run this stage in the pinned MM-GDINO environment."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED for MM-GDINO stage")
    paths = load_paths(args.paths)
    release = ensure_release_on_path(Path(paths["release_root"]))
    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    model_path = Path(paths["models"]["mm_gdino_l"]).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"MM-GDINO model path missing: {model_path}")
    threshold = float(paths.get("runtime", {}).get("mm_threshold", 0.25))
    text_threshold = float(paths.get("runtime", {}).get("mm_text_threshold", 0.25))
    top_k = int(paths.get("runtime", {}).get("mm_top_k", 16))
    if (threshold, text_threshold, top_k) != (0.25, 0.25, 16):
        raise ValueError("This reproduction is frozen to threshold/text_threshold=.25 and top_k=16")

    families = jsonl(args.families)
    if args.limit is not None:
        families = families[: args.limit]
    tasks = [{"task_id": f"{f['family_id']}|{ref}", "family_id": f["family_id"], "logical_ref": ref,
              "referent_phrase": f[f"referent_{ref}"], "image_relpath": f["image_relpath"],
              "image_sha256": f["image_sha256"]} for f in families for ref in ("A", "B")]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "candidates.jsonl"
    existing = {}
    if result_path.is_file():
        for row in jsonl(result_path):
            key = row["task_id"]
            if key in existing:
                raise ValueError(f"duplicate candidate task in existing output: {key}")
            existing[key] = row
    identity = {
        "schema": "paco-realbind2-mm-gdino-v1",
        "stage": "mm_gdino_candidates",
        "model_path": str(model_path.resolve()),
        "model_config_sha256": sha256_file(model_path / "config.json") if (model_path / "config.json").is_file() else None,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": {"threshold": threshold, "text_threshold": text_threshold, "top_k": top_k,
                       "order": "native_score_then_raw_index"},
        "release_root": str(release),
        "code_identity": code_identity(),
        "families_sha256": sha256_file(args.families),
    }
    identity_path = out / "identity.json"
    if identity_path.is_file():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise ValueError("resume identity mismatch; use a fresh output directory")
    else:
        atomic_json(identity_path, identity)

    todo = [t for t in tasks if t["task_id"] not in existing]
    if not todo:
        write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["task_id"]))
        atomic_json(out / "status.json", {"status": "COMPLETE", "rows": len(existing), "new_rows": 0})
        print(json.dumps({"status": "COMPLETE", "rows": len(existing), "new_rows": 0}))
        return

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    started_all = time.monotonic()
    for index, task in enumerate(todo, 1):
        image_path = dataset_root / task["image_relpath"]
        if sha256_file(image_path) != task["image_sha256"]:
            raise ValueError(f"image hash mismatch: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        started = time.monotonic()
        inputs = processor(images=image, text=[[task["referent_phrase"]]], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        pred = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        candidates = []
        labels = pred.get("text_labels", pred.get("labels", []))
        for raw_index, (box, score) in enumerate(zip(pred["boxes"], pred["scores"])):
            confidence = float(score)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise RuntimeError("invalid MM-GDINO confidence")
            try:
                bounded = bounded_box(box.tolist(), image.width, image.height)
            except ValueError:
                continue
            candidates.append({
                "bbox_xyxy": bounded,
                "native_score": confidence,
                "raw_candidate_index": raw_index,
                "matched_phrase": str(labels[raw_index]) if raw_index < len(labels) else "",
            })
        candidates = sorted(candidates, key=lambda c: (-c["native_score"], c["raw_candidate_index"]))[:top_k]
        row = {
            **task,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "candidate_config": {"threshold": threshold, "text_threshold": text_threshold,
                                 "cap": top_k, "order": "native_score_then_raw_index"},
            "elapsed_seconds": time.monotonic() - started,
            "candidate_result_hash": stable_hash(candidates),
        }
        existing[task["task_id"]] = row
        if index % 50 == 0 or index == len(todo):
            write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["task_id"]))
            print(json.dumps({"progress": index, "total": len(todo), "elapsed_seconds": round(time.monotonic()-started_all, 1)}), flush=True)
    write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["task_id"]))
    atomic_json(out / "status.json", {"status": "COMPLETE", "rows": len(existing), "new_rows": len(todo)})
    print(json.dumps({"status": "COMPLETE", "rows": len(existing), "new_rows": len(todo)}))


if __name__ == "__main__":
    main()
