#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.dataset import EXPECTED_QA


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=EXPECTED_QA)
    args = parser.parse_args()
    rows: dict[str, dict] = {}
    duplicates: list[str] = []
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["case_id"])
            if key in rows:
                duplicates.append(key)
            rows[key] = row
    if duplicates:
        raise SystemExit(f"duplicate case IDs: {sorted(set(duplicates))[:20]}")
    if len(rows) != args.expected:
        raise SystemExit(f"row-count mismatch: expected {args.expected}, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(rows[key], sort_keys=True) + "\n" for key in sorted(rows)), encoding="utf-8")
    print(f"MERGE_PASS rows={len(rows)} duplicates=0 missing=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
