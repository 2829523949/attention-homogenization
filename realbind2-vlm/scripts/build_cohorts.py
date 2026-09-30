#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.failure.build_cohorts import derive_cohorts, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.baseline.read_text(encoding="utf-8").splitlines() if line.strip()]
    f0, f1 = derive_cohorts(rows)
    write_jsonl(args.output_dir / "f0.jsonl", f0)
    write_jsonl(args.output_dir / "f1.jsonl", f1)
    print(f"COHORTS_PASS F0={len(f0)} F1={len(f1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
