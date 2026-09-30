from __future__ import annotations

from collections import defaultdict
from typing import Any


def classify(row: dict[str, Any]) -> str:
    if row.get("localization_ok") is False: return "F1_LOCALIZATION"
    if row.get("identity_mapping_ok") is False: return "F2_IDENTITY_BINDING"
    if row.get("attribute_ok") is False: return "F3_ATTRIBUTE"
    if row.get("geometry_ok") is False: return "F4_GEOMETRY"
    if row.get("compilation_ok") is False: return "F5_COMPILATION"
    if row.get("facts_only_correct") is True and row.get("image_plus_facts_correct") is False: return "F7_VISUAL_INTERFERENCE"
    if row.get("all_required_facts_ok") is True and row.get("final_correct") is False: return "F6_REASONING"
    return "UNKNOWN_AMBIGUOUS"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows: groups[classify(row)].append(str(row["case_id"]))
    n = len(rows)
    return {name: {"count": len(ids), "rate": len(ids)/n if n else None, "case_ids": ids, "representative_examples": ids[:5]} for name, ids in sorted(groups.items())}
