from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.common.dataset import RealBind2Dataset


def _frozen(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen"
    (frozen / "images").mkdir(parents=True)
    families = [
        {"family_id": f"F{i}", "image_id": f"I{i}", "image_path": f"images/I{i}.jpg"}
        for i in range(4)
    ]
    questions = [
        {"family_id": family["family_id"], "qa_id": f"{family['family_id']}__Q{q}", "qa_type": f"Q{q}"}
        for family in families for q in range(3)
    ]
    (frozen / "families.jsonl").write_text("".join(json.dumps(x) + "\n" for x in families))
    (frozen / "questions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in questions))
    (frozen / "construction_stats.json").write_text("{}\n")
    return tmp_path


def test_family_shards_are_disjoint_and_complete(tmp_path: Path) -> None:
    root = _frozen(tmp_path)
    dataset = RealBind2Dataset(root)
    shard0 = dataset.family_shard(2, 0)
    shard1 = dataset.family_shard(2, 1)
    assert {row["family_id"] for row in shard0} == {"F0", "F2"}
    assert {row["family_id"] for row in shard1} == {"F1", "F3"}
    assert {row["qa_id"] for row in shard0}.isdisjoint(row["qa_id"] for row in shard1)
    assert len(shard0) + len(shard1) == 12


def test_merge_checks_exact_dataset_ids(tmp_path: Path) -> None:
    root = _frozen(tmp_path / "data")
    questions = RealBind2Dataset(root).questions()
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    for path, rows in ((one, questions[::2]), (two, questions[1::2])):
        path.write_text("".join(json.dumps({**row, "case_id": row["qa_id"], "status": "ok"}) + "\n" for row in rows))
    output = tmp_path / "merged.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts" / "merge_baseline.py"
    subprocess.run(
        [sys.executable, str(script), "--dataset-root", str(root), "--inputs", str(one), str(two), "--output", str(output)],
        check=True,
    )
    assert len(output.read_text().splitlines()) == 12
    saved = output.read_bytes()
    duplicate = subprocess.run(
        [sys.executable, str(script), "--dataset-root", str(root), "--inputs", str(one), str(one), "--output", str(output)],
        capture_output=True,
    )
    assert duplicate.returncode != 0
    assert output.read_bytes() == saved
