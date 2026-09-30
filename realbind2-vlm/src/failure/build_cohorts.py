"""Derive F0/F1 only from frozen baseline rows and declared dependencies."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DEPENDENCIES = {
    "REL2C_LEFT": ("I2C_A", "I2C_B", "PAIR_REL_A_TO_B", "REL2I_LEFT"),
    "REL2C_RIGHT": ("I2C_A", "I2C_B", "PAIR_REL_A_TO_B", "REL2I_RIGHT"),
}


def derive_cohorts(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_family: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_family[str(row["family_id"])][str(row["qa_type"])] = row
    f0: list[dict] = []
    f1: list[dict] = []
    for family_id in sorted(by_family):
        family = by_family[family_id]
        for composition_type, prerequisite_types in DEPENDENCIES.items():
            if composition_type not in family or any(name not in family for name in prerequisite_types):
                continue
            prerequisites = [family[name] for name in prerequisite_types]
            if not all(bool(row.get("correct")) for row in prerequisites):
                continue
            composition = family[composition_type]
            record = {
                "case_id": str(composition.get("case_id", composition["qa_id"])),
                "family_id": family_id,
                "qa_id": composition["qa_id"],
                "qa_type": composition_type,
                "prerequisite_qa_ids": [row["qa_id"] for row in prerequisites],
                "prerequisite_correct": True,
                "composition_correct": bool(composition.get("correct")),
            }
            (f0 if record["composition_correct"] else f1).append(record)
    return f0, f1


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

