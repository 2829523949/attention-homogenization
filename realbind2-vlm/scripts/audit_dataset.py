#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.dataset import RealBind2Dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit = RealBind2Dataset(args.data_root).audit()
    except FileNotFoundError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({**asdict(audit), "passed": audit.passed}, indent=2, sort_keys=True))
    return 0 if audit.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
