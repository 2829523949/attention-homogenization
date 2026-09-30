from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.common.dataset import EXPECTED_COMPOSITION_QA, EXPECTED_FAMILIES, EXPECTED_QA, RealBind2Dataset


def test_protocol_counts_are_frozen():
    assert EXPECTED_FAMILIES == 1400
    assert EXPECTED_QA == 9800
    assert EXPECTED_COMPOSITION_QA == 2800


def test_small_dataset_audit_detects_nonformal_counts(tmp_path: Path):
    frozen = tmp_path / "frozen"
    (frozen / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(frozen / "images" / "x.jpg")
    family = {"family_id": "F", "image_id": "x", "image_path": "images/x.jpg"}
    (frozen / "families.jsonl").write_text(json.dumps(family) + "\n")
    questions = [
        {"qa_id": f"F__{name}", "family_id": "F", "qa_type": name}
        for name in ("I2C_A", "I2C_B", "PAIR_REL_A_TO_B", "REL2I_LEFT", "REL2I_RIGHT", "REL2C_LEFT", "REL2C_RIGHT")
    ]
    (frozen / "questions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in questions))
    (frozen / "construction_stats.json").write_text("{}\n")
    audit = RealBind2Dataset(tmp_path).audit()
    assert audit.families == 1 and audit.qa == 7 and audit.composition_qa == 2
    assert not audit.passed

