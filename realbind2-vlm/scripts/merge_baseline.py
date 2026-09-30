#!/usr/bin/env python3
"""Merge baseline shards and prove exact QA coverage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.dataset import RealBind2Dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {str(row["qa_id"]) for row in RealBind2Dataset(args.dataset_root).questions()}
    rows: dict[str, dict] = {}
    duplicates: list[str] = []
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                raise SystemExit(f"refusing incomplete/failed baseline row in {path}")
            qa_id = str(row["qa_id"])
            if qa_id in rows:
                duplicates.append(qa_id)
            rows[qa_id] = row
    actual = set(rows)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if duplicates or missing or extra:
        raise SystemExit(json.dumps({
            "duplicates": sorted(set(duplicates))[:20],
            "missing": missing[:20],
            "extra": extra[:20],
            "expected": len(expected),
            "actual": len(actual),
        }, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(rows[key], sort_keys=True, allow_nan=False) + "\n" for key in sorted(rows)),
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(f"MERGE_PASS rows={len(rows)} duplicates=0 missing=0 extra=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
