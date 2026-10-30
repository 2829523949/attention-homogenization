#!/usr/bin/env python3
"""Stage 2: R4 candidate-grid selector for PACO A/B referents.

Uses the exact frozen R4 selector prompt and numbered-grid layout:
  `Which numbered crop best matches: {referent}? Answer only the number.`
Singleton pools skip the model call. Invalid integers fall back to native top1.
For the conditioning set, run this stage with the primary Qwen3.5-4B config.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    atomic_json, code_identity, jsonl, load_adapter, load_model_spec, load_paths,
    select_candidates, sha256_file, stable_hash, write_jsonl_atomic,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--model-tag", default="qwen35_4b")
    ap.add_argument("--families", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from PIL import Image

    paths = load_paths(args.paths)
    model_cfg, runtime, model_path = load_model_spec(args.model_tag, paths)
    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    families = jsonl(args.families)
    if args.limit is not None:
        families = families[: args.limit]
    family_by_id = {f["family_id"]: f for f in families}
    tasks = [{"task_id": f"{f['family_id']}|{ref}", "family_id": f["family_id"],
              "logical_ref": ref, "referent_phrase": f[f"referent_{ref}"]}
             for f in families for ref in ("A", "B")]
    candidates_rows = {r["task_id"]: r for r in jsonl(args.candidates)}
    missing_candidates = [t["task_id"] for t in tasks if t["task_id"] not in candidates_rows]
    if missing_candidates:
        raise RuntimeError(f"candidate pool missing for {len(missing_candidates)} tasks, e.g. {missing_candidates[:5]}")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "selected_targets.jsonl"
    existing = {r["task_id"]: r for r in jsonl(result_path)} if result_path.is_file() else {}
    if len(existing) != len(set(existing)):
        raise ValueError("duplicate task rows in existing selected_targets.jsonl")

    adapter = load_adapter(model_cfg, model_path, runtime)
    metadata = adapter.get_model_metadata()
    identity = {
        "schema": "paco-realbind2-r4-selector-v1",
        "stage": "r4_selector",
        "model_tag": args.model_tag,
        "model_id": model_cfg["model_id"],
        "model_config": model_cfg,
        "adapter_metadata": metadata,
        "families_sha256": sha256_file(args.families),
        "candidates_sha256": sha256_file(args.candidates),
        "selector_prompt": "Which numbered crop best matches: {referent}? Answer only the number.",
        "fallback": "invalid integer -> candidate index 0 (native top1)",
        "release_root": str(Path(paths["release_root"]).resolve()),
        "code_identity": code_identity(),
    }
    identity_path = out / "identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("resume identity mismatch; use a fresh output directory")
    else:
        atomic_json(identity_path, identity)

    todo = [t for t in tasks if t["task_id"] not in existing]
    if not todo:
        write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["task_id"]))
        atomic_json(out / "status.json", {"status": "COMPLETE", "rows": len(existing), "new_rows": 0})
        print(json.dumps({"status": "COMPLETE", "rows": len(existing), "new_rows": 0}))
        return
    grids = out / "selector_grids"
    started_all = time.monotonic()
    for index, task in enumerate(todo, 1):
        family = family_by_id[task["family_id"]]
        image_path = dataset_root / family["image_relpath"]
        if sha256_file(image_path) != family["image_sha256"]:
            raise ValueError(f"image hash mismatch: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        pool = candidates_rows[task["task_id"]]["candidates"]
        grid_path = grids / f"{task['family_id']}__{task['logical_ref']}.png"
        selected = select_candidates(adapter, image, pool, task["referent_phrase"], grid_path)
        row = {
            **task,
            "selected_index": selected["selected_index"],
            "selected_bbox_xyxy": selected["selected_bbox_xyxy"],
            "selected_native_score": selected.get("selected_native_score"),
            "selected_provider": "R4_MM_GDINO_L_QWEN_SELECTOR",
            "qwen_status": selected["selector_status"],
            "qwen_raw_output": selected.get("selector_raw"),
            "qwen_generation_error": selected.get("selector_generation_error"),
            "candidate_count": selected["candidate_count"],
            "status": selected["status"],
            "selection_model_id": model_cfg["model_id"],
            "selection_model_tag": args.model_tag,
            "selected_result_hash": stable_hash(selected),
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
