#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.dataset import RealBind2Dataset
from src.common.logging_utils import JsonlRunLog
from src.common.manifests import stable_hash
from src.common.scorer import SCORER_VERSION, score_prediction
from src.models import ADAPTERS


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable RealBind2 baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-by", choices=("family", "qa"), default="family")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config, paths = load_yaml(args.config), load_yaml(args.paths)
    dataset_root = Path(paths["data"]["realbind2"])
    dataset = RealBind2Dataset(dataset_root)
    rows = (
        dataset.family_shard(args.num_shards, args.shard_index)
        if args.shard_by == "family"
        else dataset.shard(args.num_shards, args.shard_index)
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    model_key = str(config["adapter"])
    model_path = Path(paths["models"][config["path_key"]])
    adapter = ADAPTERS[model_key](model_path, **config.get("runtime", {}))
    families = {str(row["family_id"]): row for row in dataset.families()}
    log = JsonlRunLog(args.output)
    failed = JsonlRunLog(args.output.with_suffix(".failed.jsonl"))
    config_hash = stable_hash(config)
    dataset_hash = stable_hash(json.loads((dataset_root / "frozen" / "construction_stats.json").read_text()))

    for row in rows:
        case_id = str(row["qa_id"])
        if log.is_complete(case_id):
            continue
        family = families[str(row["family_id"])]
        image_path = dataset_root / "frozen" / family["image_path"]
        started = time.monotonic()
        try:
            raw = adapter.generate(adapter.prepare_prompt(str(row["question"]), image_path))
            scored = score_prediction(raw, str(row["ground_truth"]), str(row["answer_type"]))
            log.append({
                "case_id": case_id,
                "family_id": str(row["family_id"]),
                "image_id": str(family["image_id"]),
                "qa_id": case_id,
                "qa_type": str(row["qa_type"]),
                "model": str(config["model_id"]),
                "experiment": "baseline",
                "condition": "original_image",
                "prediction_raw": raw,
                "prediction_normalized": scored.normalized,
                "ground_truth": scored.ground_truth,
                "correct": scored.correct,
                "valid": scored.valid,
                "scorer_error": scored.error,
                "runtime_sec": time.monotonic() - started,
                "status": "ok",
                "config_hash": config_hash,
                "dataset_hash": dataset_hash,
                "scorer_version": SCORER_VERSION,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "shard_by": args.shard_by,
            })
        except Exception as exc:
            failed.append({
                "case_id": case_id,
                "family_id": str(row["family_id"]),
                "model": str(config["model_id"]),
                "experiment": "baseline",
                "condition": "original_image",
                "runtime_sec": time.monotonic() - started,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "config_hash": config_hash,
                "dataset_hash": dataset_hash,
            })
    incomplete = {str(row["qa_id"]) for row in rows} - log.completed
    print(f"BASELINE_COMPLETE completed={len(log.completed)} incomplete={len(incomplete)} failed_rows={failed.path}")
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
