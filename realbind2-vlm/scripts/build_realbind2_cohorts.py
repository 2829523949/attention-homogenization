#!/usr/bin/env python3
"""Build all RealBind2 cohorts from immutable baseline outputs.

Stable F1 requires an independent deterministic baseline rerun.  A single run
is never silently treated as stability evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.dataset import RealBind2Dataset
from src.failure.build_cohorts import DEPENDENCIES, derive_cohorts, write_jsonl


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_unique(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"duplicate {field}: {key}")
        result[key] = row
    return result


def target_ref(family: dict[str, Any], qa_type: str) -> str:
    left = family.get("left_entity")
    if left is None:
        left = "A" if family["geometry_metrics"]["left_is_A"] else "B"
    return str(left) if qa_type.endswith("LEFT") else ("B" if str(left) == "A" else "A")


def bbox_xywh(family: dict[str, Any], ref: str) -> list[float]:
    if f"bbox_{ref}" in family:
        return [float(value) for value in family[f"bbox_{ref}"]]
    return [float(value) for value in family[ref]["bbox"]]


def metadata(case: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    ref = target_ref(family, str(case["qa_type"]))
    bbox = bbox_xywh(family, ref)
    width, height = float(family["width"] if "width" in family else family["image_width"]), float(
        family["height"] if "height" in family else family["image_height"]
    )
    fraction = bbox[2] * bbox[3] / (width * height)
    scale = "small" if fraction < 0.05 else "medium" if fraction < 0.20 else "large"
    relation = family.get("relation_A_to_B")
    if relation is None:
        relation = "LEFT" if family["geometry_metrics"]["left_is_A"] else "RIGHT"
    category = family.get(f"category_{ref}")
    color = family.get(f"color_{ref}")
    if category is None and ref in family:
        category = family[ref].get("object") or family[ref].get("category")
    if color is None and ref in family:
        color = family[ref].get("color")
    return {
        "relation_direction": str(relation),
        "target_role": ref,
        "target_side": "LEFT" if str(case["qa_type"]).endswith("LEFT") else "RIGHT",
        "target_attribute": str(color),
        "target_category": str(category),
        "target_bbox_scale": scale,
        "target_bbox_area_fraction": fraction,
    }


def stable_f1_rows(
    f1: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], repeat: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    stable: list[dict[str, Any]] = []
    for case in f1:
        required = list(case["prerequisite_qa_ids"]) + [str(case["qa_id"])]
        if any(qa_id not in baseline or qa_id not in repeat for qa_id in required):
            continue
        fields = ("prediction_normalized", "correct", "valid")
        if all(all(baseline[qa_id].get(field) == repeat[qa_id].get(field) for field in fields) for qa_id in required):
            stable.append({**case, "stability_pass": True, "stability_repeats": 2})
    return stable


def match_cost(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, ...]:
    fields = (
        "relation_direction", "target_role", "target_attribute",
        "target_category", "target_bbox_scale", "target_side",
    )
    mismatch = tuple(int(left[field] != right[field]) for field in fields)
    area_delta = int(abs(float(left["target_bbox_area_fraction"]) - float(right["target_bbox_area_fraction"])) * 1_000_000)
    return (sum(mismatch), *mismatch, area_delta)


def matched_f0(stable: list[dict[str, Any]], f0: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = {str(row["case_id"]): row for row in f0}
    matched: list[dict[str, Any]] = []
    for f1_case in sorted(stable, key=lambda row: str(row["case_id"])):
        if not available:
            raise RuntimeError("insufficient F0 rows for no-replacement matching")
        selected_id, selected = min(
            available.items(),
            key=lambda item: (match_cost(f1_case, item[1]), item[0]),
        )
        available.pop(selected_id)
        matched.append({
            **selected,
            "matched_f1_case_id": str(f1_case["case_id"]),
            "match_cost": list(match_cost(f1_case, selected)),
        })
    return matched


def build(
    dataset_root: Path,
    baseline_path: Path,
    repeat_path: Path,
    output_dir: Path,
    limit: int = 100,
) -> dict[str, Any]:
    dataset = RealBind2Dataset(dataset_root)
    if not 0 <= limit <= 100:
        raise ValueError("frozen cohort limit must be between 0 and 100")
    baseline_rows = read_rows(baseline_path)
    repeat_rows = read_rows(repeat_path)
    expected = {str(row["qa_id"]) for row in dataset.questions()}
    baseline = index_unique(baseline_rows, "qa_id")
    repeat = index_unique(repeat_rows, "qa_id")
    if set(baseline) != expected or set(repeat) != expected:
        raise ValueError("baseline and stability rerun must each cover exactly the frozen QA IDs")
    families = {str(row["family_id"]): row for row in dataset.families()}
    f0, f1 = derive_cohorts(baseline_rows)
    f0 = [{**row, **metadata(row, families[str(row["family_id"])])} for row in f0]
    f1 = [{**row, **metadata(row, families[str(row["family_id"])])} for row in f1]
    stable_all = stable_f1_rows(f1, baseline, repeat)
    selected = sorted(stable_all, key=lambda row: str(row["case_id"]))[:limit]
    matched = matched_f0(selected, f0)
    vir = [
        {
            "case_id": str(row["qa_id"]),
            "family_id": str(row["family_id"]),
            "qa_id": str(row["qa_id"]),
            "qa_type": str(row["qa_type"]),
        }
        for row in dataset.questions() if row["qa_type"] in ("REL2C_LEFT", "REL2C_RIGHT")
    ]
    causal = [
        *({**row, "cohort_role": "stable_F1"} for row in selected),
        *({**row, "cohort_role": "matched_F0"} for row in matched),
    ]
    retrospective = list(causal)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("f0.jsonl", f0),
        ("f1.jsonl", f1),
        ("stable_f1.jsonl", stable_all),
        ("vir_cohort.jsonl", vir),
        ("causal_cohort.jsonl", causal),
        ("repair_retrospective_cohort.jsonl", retrospective),
    ):
        write_jsonl(output_dir / name, rows)
    summary = {
        "F0": len(f0),
        "F1": len(f1),
        "stable_F1_all": len(stable_all),
        "stable_F1_selected": len(selected),
        "matched_F0": len(matched),
        "VIR": len(vir),
        "causal": len(causal),
        "repair_retrospective": len(retrospective),
        "stable_definition": "exact agreement of normalized prediction, correctness, and validity for composition plus all declared prerequisites across two complete deterministic baseline runs",
        "selection": "sorted stable F1 case IDs, first up to 100; no duplication",
        "matching": "deterministic no-replacement minimum mismatch over relation direction, target role, attribute, category, bbox scale, target side, then bbox-area delta",
    }
    (output_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=ROOT / "configs" / "server_paths.yaml")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--stability-baseline", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text(encoding="utf-8"))
    output_root = Path(config["output"]["root"])
    dataset_root = Path(config["data"]["realbind2"])
    baseline = args.baseline or output_root / "qwen35_4b" / "baseline" / "results.jsonl"
    repeat = args.stability_baseline or output_root / "qwen35_4b" / "baseline" / "stability_results.jsonl"
    output = args.output_dir or output_root / "qwen35_4b" / "cohorts"
    if not repeat.is_file():
        raise SystemExit(
            "stable F1 requires an independent complete deterministic baseline rerun at "
            f"{repeat}; no single-run stability inference is allowed"
        )
    print(json.dumps(build(dataset_root, baseline, repeat, output, args.limit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
