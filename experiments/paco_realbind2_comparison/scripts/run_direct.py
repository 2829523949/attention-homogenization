#!/usr/bin/env python3
"""Baseline Direct on the frozen PACO/RealBind2-conditioned set.

Receives original image + frozen REL2C question. Uses the exact RealBind2
question template, semantic scorer, batch=1 greedy decoding, no thinking,
max_new_tokens=16, and the model-specific adapter config.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    atomic_json, code_identity, generation_call, jsonl, load_adapter, load_model_spec,
    load_paths, scorer, sha256_file, stable_hash, write_jsonl_atomic,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paths.yaml")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, default=None, help="defaults to paths.dataset_root")
    ap.add_argument("--benchmark", type=Path, default=None, help="defaults to <dataset-dir>/benchmark.jsonl")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = load_paths(args.paths)
    model_cfg, runtime, model_path = load_model_spec(args.model_tag, paths)
    dataset_dir = args.dataset_dir.resolve()
    images_root = (args.images_root or Path(paths["dataset_root"])).expanduser().resolve()
    benchmark_path = args.benchmark or (dataset_dir / "benchmark.jsonl")
    benchmark = jsonl(benchmark_path)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    if not benchmark:
        raise RuntimeError("empty benchmark")
    if len({q["qa_id"] for q in benchmark}) != len(benchmark):
        raise ValueError("duplicate QA IDs in benchmark")

    adapter = load_adapter(model_cfg, model_path, runtime)
    score_prediction = scorer()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    existing = {r["qa_id"]: r for r in jsonl(result_path)} if result_path.is_file() else {}
    if len(existing) != len(set(existing)):
        raise ValueError("duplicate completed QA in existing results")
    identity = {
        "schema": "paco-realbind2-direct-v1",
        "model_tag": args.model_tag,
        "model_id": model_cfg["model_id"],
        "model_config": model_cfg,
        "adapter_metadata": adapter.get_model_metadata(),
        "dataset_dir": str(dataset_dir),
        "images_root": str(images_root),
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": sha256_file(benchmark_path),
        "scorer": "realbind2-semantic-v1",
        "method": "direct",
        "mode": "live",
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
                        r.get("method") != "direct" for r in existing.values()):
        raise ValueError("existing result rows do not match current model/benchmark identity")

    todo = [q for q in benchmark if q["qa_id"] not in existing]
    started_all = time.monotonic()
    for index, q in enumerate(todo, 1):
        image_path = images_root / q["image_relpath"]
        if not image_path.is_file() or sha256_file(image_path) != q["image_sha256"]:
            raise ValueError(f"image missing/hash mismatch: {image_path}")
        call = generation_call(adapter, q["question"], image_path=str(image_path))
        score = score_prediction(call["raw"], q["answer"], "color")
        row = {
            "qa_id": q["qa_id"],
            "family_id": q["family_id"],
            "qa_type": q["qa_type"],
            "paco_split": q["paco_split"],
            "question": q["question"],
            "question_sha256": q["question_sha256"],
            "image_relpath": q["image_relpath"],
            "image_sha256": q["image_sha256"],
            "method": "direct",
            "mode": "live",
            "model_id": model_cfg["model_id"],
            "model_tag": args.model_tag,
            "prediction_raw": call["raw"],
            "prediction_normalized": score.normalized,
            "ground_truth": score.ground_truth,
            "correct": bool(score.correct),
            "valid": bool(score.valid),
            "scorer_error": score.error,
            "scorer_version": "realbind2-semantic-v1",
            "generation_error": call["generation_error"],
            "cost": {"elapsed_seconds": call["elapsed_seconds"], "model_calls": 1},
            "run_benchmark_sha256": identity["benchmark_sha256"],
        }
        existing[q["qa_id"]] = row
        if index % 20 == 0 or index == len(todo):
            write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["qa_id"]))
            print(json.dumps({"progress": index, "total": len(todo), "elapsed_seconds": round(time.monotonic()-started_all, 1)}), flush=True)
    write_jsonl_atomic(result_path, sorted(existing.values(), key=lambda r: r["qa_id"]))
    atomic_json(out / "status.json", {"status": "COMPLETE", "n": len(existing), "new_rows": len(todo)})
    print(json.dumps({"status": "COMPLETE", "n": len(existing), "new_rows": len(todo)}))


if __name__ == "__main__":
    main()
