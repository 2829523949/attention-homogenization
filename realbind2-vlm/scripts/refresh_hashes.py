#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.manifests import sha256_file

FILES = (
    "data/realbind2_manifest.json",
    "data/prompt_templates/realbind2.json",
    "data/prompt_templates/visual_interference.json",
    "src/common/prompts.py",
    "src/common/scorer.py",
    "configs/qwen35_4b.yaml",
    "configs/qwen35_9b.yaml",
    "configs/qwen3vl_4b.yaml",
    "configs/qwen3vl_8b.yaml",
    "configs/internvl35_8b.yaml",
    "configs/llava_next_7b.yaml",
    "src/repair/strong_integrated/pipeline.py",
    "src/failure/defocus_frozen.py",
    "data/DEFOCUS_PROTOCOL.sha256",
    "data/smoke_cases.json",
    "configs/server_paths.example.yaml",
    "scripts/server_live_certification.py",
    "scripts/build_realbind2_cohorts.py",
    "data/REALBIND2_FROZEN.sha256",
    "data_audit/human_acceptance_basis.json",
    "data_audit/human_acceptance_decisions.jsonl",
    "scripts/certify_frozen_realbind2.py",
    "scripts/verify_frozen_export.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    payload = {
        "algorithm": "sha256",
        "files": {relative: sha256_file(args.root / relative) for relative in FILES},
    }
    output = args.root / "data" / "HASHES.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"HASH_MANIFEST_WRITTEN files={len(FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
