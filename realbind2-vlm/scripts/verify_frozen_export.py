#!/usr/bin/env python3
"""Verify portable frozen release files and images without a construction DB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.manifests import sha256_file, stable_hash


def verify_export(dataset_root: Path, manifest: dict) -> None:
    if not manifest.get("frozen"):
        raise ValueError("DATASET_NOT_FROZEN")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if stable_hash(body) != manifest["manifest_sha256"]:
        raise ValueError("frozen manifest content hash mismatch")
    for relative, expected in manifest["export_file_hashes"].items():
        path = dataset_root / relative
        if not path.resolve().is_relative_to(dataset_root.resolve()):
            raise ValueError("export path escapes dataset root")
        if sha256_file(path) != expected:
            raise ValueError(f"release hash mismatch: {relative}")
    frozen = dataset_root / "frozen"
    families = [json.loads(x) for x in (frozen / "families.jsonl").read_text().splitlines() if x.strip()]
    images = {str(f["image_id"]): f["image_path"] for f in families}
    if set(images) != set(manifest["image_sha256"]):
        raise ValueError("release image IDs mismatch")
    for image_id, relative in images.items():
        path = frozen / relative
        if not path.resolve().is_relative_to(frozen.resolve()):
            raise ValueError("image path escapes frozen root")
        if sha256_file(path) != manifest["image_sha256"][image_id]:
            raise ValueError(f"image hash mismatch: {image_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--paths", type=Path)
    args = parser.parse_args()
    if args.dataset_root:
        root = args.dataset_root
    elif args.paths:
        import yaml
        root = Path(yaml.safe_load(args.paths.read_text())["data"]["realbind2"])
    else:
        parser.error("--dataset-root or --paths is required")
    manifest = json.loads((ROOT / "data/realbind2_manifest.json").read_text())
    verify_export(root.resolve(), manifest)
    print(f"FROZEN_EXPORT_HASH_PASS images={manifest['num_unique_images']} QA={manifest['num_qa']}")


if __name__ == "__main__":
    main()
