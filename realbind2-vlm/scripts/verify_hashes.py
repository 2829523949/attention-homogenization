#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.manifests import verify_hash_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.root / "data" / "HASHES.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = verify_hash_manifest(args.root, manifest)
    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        return 2
    print(f"HASH_PASS files={len(manifest.get('files', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
