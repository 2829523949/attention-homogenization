#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(str(row["case_id"]) for row in rows)
    duplicates = sorted(key for key, n in counts.items() if n != 1)
    missing_fields = sum(any(field not in row for field in ("case_id", "status", "config_hash", "dataset_hash")) for row in rows)
    result = {"rows": len(rows), "expected": args.expected, "duplicates": duplicates, "missing_required_fields": missing_fields}
    result["passed"] = len(rows) == args.expected and not duplicates and not missing_fields
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

