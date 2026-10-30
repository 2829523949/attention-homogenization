#!/usr/bin/env python3
"""Independent CPU verifier for the frozen PACO conditioning artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import jsonl, sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.dataset_dir.resolve()
    spec = json.loads((root / "split_spec.json").read_text())
    families = jsonl(root / "families_conditioned.jsonl")
    questions = jsonl(root / "benchmark.jsonl")
    questions2 = jsonl(root / "questions_conditioned.jsonl")
    qa_audit = jsonl(root / "selection_audit.jsonl")
    family_audit = jsonl(root / "family_conditioning_audit.jsonl")
    assert questions == questions2
    assert len({f["family_id"] for f in families}) == len(families)
    assert len({q["qa_id"] for q in questions}) == len(questions)
    assert {q["family_id"] for q in questions} == {f["family_id"] for f in families}
    counts = Counter(q["family_id"] for q in questions)
    assert set(counts.values()) == {2}
    assert {q["qa_type"] for q in questions} <= {"REL2C_LEFT", "REL2C_RIGHT"}
    for fid in counts:
        qtypes = {q["qa_type"] for q in questions if q["family_id"] == fid}
        assert qtypes == {"REL2C_LEFT", "REL2C_RIGHT"}
    family_audit_by_id = {a["family_id"]: a for a in family_audit}
    assert len(family_audit_by_id) == len(family_audit)
    keep = {a["family_id"] for a in family_audit if a["decision"] == "KEEP"}
    assert keep == {f["family_id"] for f in families}
    assert set(family_audit_by_id) >= keep
    for a in family_audit:
        expected = (not a["same_description"]) and float(a["ab_bbox_iou"]) < 0.1
        assert (a["decision"] == "KEEP") == expected, a
        if a["decision"] == "KEEP":
            assert float(a["ab_bbox_iou"]) < 0.1
            assert not a["same_description"]
    assert len(qa_audit) == len(questions)
    assert {a["qa_id"] for a in qa_audit} == {q["qa_id"] for q in questions}
    for a in qa_audit:
        assert a["decision"] == "KEEP"
        assert float(a["ab_bbox_iou"]) < 0.1
        assert not a["same_description"]
    for f in families:
        a = family_audit_by_id[f["family_id"]]
        assert float(f["r4_predicted_ab_iou"]) == float(a["ab_bbox_iou"])
        assert f["r4_selected_bbox_A_xyxy"] == a["selected_A_bbox_xyxy"]
        assert f["r4_selected_bbox_B_xyxy"] == a["selected_B_bbox_xyxy"]
    for q in questions:
        assert q["question_sha256"]
        assert q["answer"] in {"red","orange","yellow","green","blue","purple","pink","brown","black","white","gray"}
        assert q["image_sha256"]
    for name, expected in spec["files"].items():
        assert sha256_file(root / name) == expected, name
    summary = {
        "status": "PASS",
        "families": len(families),
        "questions": len(questions),
        "unique_images": len({q["image_sha256"] for q in questions}),
        "qa_type_counts": dict(sorted(Counter(q["qa_type"] for q in questions).items())),
        "split_counts": dict(sorted(Counter(q["paco_split"] for q in questions).items())),
        "all_keep_iou_strictly_less_than_0.1": True,
        "all_referents_distinct": True,
        "selection_audit_hashes_match_split_spec": True,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
