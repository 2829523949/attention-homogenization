#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from harness.configs.paths import Paths, ensure_d_drive_environment


ASSETS = {"florence2": ("microsoft/Florence-2-base-ft", "florence2"), "siglip2": ("google/siglip2-base-patch16-224", "siglip2")}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("assets", nargs="*", choices=sorted(ASSETS), default=list(ASSETS)); args = parser.parse_args()
    p = Paths(); env = ensure_d_drive_environment(p)
    from huggingface_hub import snapshot_download
    records = []
    for name in args.assets:
        repo, attr = ASSETS[name]; destination = getattr(p, attr)
        complete = (destination / "config.json").is_file() and any((destination / x).is_file() for x in ("model.safetensors", "pytorch_model.bin"))
        if complete: records.append({"asset": name, "repo": repo, "path": str(destination), "status": "reused"}); continue
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=destination)
        records.append({"asset": name, "repo": repo, "path": str(destination), "status": "downloaded"})
    print(json.dumps({"environment": env, "assets": records}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
