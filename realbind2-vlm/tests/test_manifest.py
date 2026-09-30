from __future__ import annotations

import json
from pathlib import Path

from src.common.manifests import sha256_file, verify_hash_manifest


def test_hash_verification(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("frozen\n")
    manifest = {"files": {"a.txt": sha256_file(path)}}
    assert verify_hash_manifest(tmp_path, manifest) == []
    path.write_text("changed\n")
    assert verify_hash_manifest(tmp_path, manifest) == ["hash_mismatch:a.txt"]


def test_packaged_manifest_discloses_freeze_state():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/realbind2_manifest.json").read_text())
    assert manifest["protocol_expected"]["qa"] == 9800
    assert manifest["protocol_expected"]["composition_qa"] == 2800
    assert manifest["freeze_status"] in {"FROZEN", "HUMAN_REVIEW_PENDING"}

