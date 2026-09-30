#!/usr/bin/env python3
"""Mechanically re-check the frozen RealBind referent audit without mutation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.configs.paths import Paths
from harness.dataset import read_jsonl
from harness.referents.referent_resolver import ReferentInterfaceStatus, resolve_question


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, default=Paths().qa)
    args = parser.parse_args()
    rows = read_jsonl(args.qa)
    statuses: Counter[str] = Counter()
    failures: list[str] = []
    for row in rows:
        question = str(row["question"])
        result = resolve_question(question, str(row["qa_type"]))
        statuses[result.status.value] += 1
        if row["qa_type"].startswith("REL2C") and result.status is not ReferentInterfaceStatus.RESOLVABLE_FROM_TEXT:
            failures.append(f"{row['qa_id']}: REL2C did not resolve from visible text")
        if re.search(r"\bA\b", question) or re.search(r"\bB\b", question):
            failures.append(f"{row['qa_id']}: literal A/B unexpectedly appears in frozen question")
        if row["qa_type"].startswith("REL2C") and len(result.specs) != 2:
            failures.append(f"{row['qa_id']}: expected two visible referents")
    report = {
        "schema_version": "realbind_referent_audit_check_v1",
        "qa_path": str(args.qa),
        "row_count": len(rows),
        "status_counts": dict(statuses),
        "failures": failures,
        "passed": not failures,
        "policy": "audit only; no dataset or benchmark artifact was written",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

