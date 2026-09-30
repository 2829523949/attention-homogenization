#!/usr/bin/env python3
"""Create a truthful packaging manifest from the current construction DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.prompts import prompt_bundle_hash, template_hashes
from src.common.scorer import SCORER_VERSION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    db_path = args.construction_root / "state" / "construction_state.sqlite"
    db = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    rows = db.execute(
        "SELECT candidate_family_id,image_id,status,payload_json FROM families "
        "WHERE status IN ('AUTO_VALID','HUMAN_PENDING','HUMAN_ACCEPTED','FROZEN') "
        "ORDER BY candidate_family_id"
    ).fetchall()
    status_counts = dict(db.execute("SELECT status,count(*) FROM families GROUP BY status"))
    frozen = [row for row in rows if row[2] == "FROZEN"]
    human_accepted = [row for row in rows if row[2] == "HUMAN_ACCEPTED"]
    mine_path = args.construction_root / "manifests" / "mine_summary.json"
    mine = json.loads(mine_path.read_text(encoding="utf-8")) if mine_path.is_file() else {}
    scorer_path = Path(__file__).resolve().parents[1] / "src" / "common" / "scorer.py"
    manifest = {
        "dataset_version": "RealBind-v2.0",
        "benchmark_name": "RealBind2",
        "freeze_status": "FROZEN" if len(frozen) == 1400 else "HUMAN_REVIEW_PENDING",
        "formal_run_allowed": len(frozen) == 1400,
        "protocol_expected": {
            "images": 1400,
            "core_qa_per_image": 7,
            "qa": 9800,
            "composition_qa": 2800,
            "core_structure": ["2xI2C", "1xPAIR_REL", "2xREL2I", "2xREL2C"],
        },
        "construction_snapshot": {
            "family_status_counts": status_counts,
            "selected_family_count": len(rows),
            "unique_source_image_count": len({str(row[1]) for row in rows}),
            "human_accepted_count": len(human_accepted),
            "frozen_count": len(frozen),
            "caveat": "1400 AUTO_VALID families currently map to 973 unique VG source images; no frozen export exists.",
        },
        "family_ids": [str(row[0]) for row in rows],
        "image_ids": sorted({str(row[1]) for row in rows}),
        "split_information": {
            "status": "unassigned_until_frozen_export",
            "policy": "group_by_visual_genome_image_id; validation_20_percent; test_80_percent",
        },
        "prompt_template_hashes": template_hashes(),
        "prompt_bundle_sha256": prompt_bundle_hash(),
        "scorer": {
            "version": SCORER_VERSION,
            "sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
        },
        "source_annotation_hashes": mine.get("input_fingerprints", mine.get("inputs", {})),
        "source_snapshot": str(db_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MANIFEST_WRITTEN status={manifest['freeze_status']} families={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
