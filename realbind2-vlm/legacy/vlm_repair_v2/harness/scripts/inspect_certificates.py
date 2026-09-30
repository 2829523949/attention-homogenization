#!/usr/bin/env python3
"""Inspect a smoke certificate trace without joining evaluator labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses = Counter(cert["status"] for row in rows for cert in row.get("binding_certificates", []))
    print(json.dumps({"trace": str(args.trace), "cases": len(rows), "certificate_statuses": dict(statuses), "raw_image_exit_checks": [row.get("architecture_status", {}).get("raw_image_exited_before_reasoning") for row in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

