from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.server_certification_common import (
    certification_identity, require_live_receipt, write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_smoke_cases_cover_required_metadata() -> None:
    payload = json.loads((ROOT / "data" / "smoke_cases.json").read_text(encoding="utf-8"))
    rows = payload["families"]
    assert 4 <= len(rows) <= 8
    assert len({row["candidate_family_id"] for row in rows}) == len(rows)
    assert {row["left_entity"] for row in rows} == {"A", "B"}
    assert all(row["qa_types"] == ["REL2C_LEFT", "REL2C_RIGHT"] for row in rows)
    colors = {row[ref]["color"] for row in rows for ref in ("A", "B")}
    categories = {row[ref]["category"] for row in rows for ref in ("A", "B")}
    assert len(colors) >= 2
    assert len(categories) >= 2


def test_server_config_and_certification_stages_are_complete() -> None:
    config = yaml.safe_load((ROOT / "configs" / "server_paths.example.yaml").read_text())
    assert set(config) == {"python", "models", "data", "output"}
    assert {"qwen35_4b", "mm_gdino_l", "sam3"}.issubset(config["models"])
    expected = {
        "00_check_gpu.sh", "01_check_paths.sh", "02_live_qwen_smoke.sh",
        "03_live_mm_gdino_smoke.sh", "04_live_sam3_smoke.sh",
        "05_live_strong_integrated_smoke.sh", "06_live_causal_selfpatch.sh",
        "07_certify_all.sh",
    }
    root = ROOT / "experiments" / "server_certification"
    assert expected.issubset({path.name for path in root.iterdir()})
    aggregate = (root / "07_certify_all.sh").read_text(encoding="utf-8")
    assert "LIVE_SMOKE_FAIL" in aggregate
    assert aggregate.rstrip().endswith("echo LIVE_SMOKE_PASS")


def test_live_receipt_rejects_missing_and_changed_data(tmp_path: Path) -> None:
    frozen = tmp_path / "data" / "frozen"
    frozen.mkdir(parents=True)
    (frozen / "families.jsonl").write_text("{}\n")
    (frozen / "questions.jsonl").write_text("{}\n")
    config = {"data": {"realbind2": str(frozen.parent)}, "output": {"root": str(tmp_path / "outputs")}}
    with pytest.raises(RuntimeError, match="requires LIVE_SMOKE_PASS"):
        require_live_receipt(config)
    receipt = tmp_path / "outputs" / "server_certification" / "live_certification.json"
    write_json_atomic(receipt, {"status": "LIVE_SMOKE_PASS", "identity": certification_identity(config)})
    require_live_receipt(config)
    (frozen / "questions.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="stale"):
        require_live_receipt(config)


@pytest.mark.live_gpu
@pytest.mark.skip(reason="run experiments/server_certification/07_certify_all.sh on the 2xL20 server")
def test_live_gpu_certification_is_server_only() -> None:
    """Visible marker: local pytest must never impersonate the live certification."""
