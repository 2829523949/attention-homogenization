#!/usr/bin/env python3
"""Merge shard JSONL files with duplicate-key protection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-failed", action="store_true", help="retain non-ok rows instead of rejecting them")
    args = parser.parse_args()

    merged: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in args.inputs:
        for row in read_jsonl(path):
            if row.get("status") != "ok" and not args.include_failed:
                errors.append(f"non_ok_row:{path}:{row.get('record_id')}:{row.get('status')}")
                continue
            record_id = str(row.get("record_id", ""))
            if not record_id:
                errors.append(f"missing_record_id:{path}")
                continue
            if record_id in merged:
                errors.append(f"duplicate_record_id:{record_id}")
                continue
            merged[record_id] = row
    if errors:
        raise ValueError("merge failed:\n" + "\n".join(errors[:100]))
    rows = sorted(merged.values(), key=lambda row: (str(row.get("case_id")), str(row.get("condition"))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"inputs": [str(path) for path in args.inputs], "output": str(args.output), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
