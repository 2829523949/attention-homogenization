import json
from pathlib import Path

import pytest

from scripts.verify_frozen_export import verify_export
from src.common.manifests import sha256_file, stable_hash


def test_frozen_manifest_has_valid_content_hash_and_counts():
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root / "data/realbind2_manifest.json").read_text())
    assert m["frozen"] and m["formal_run_allowed"]
    assert m["manifest_sha256"] == stable_hash({k: v for k, v in m.items() if k != "manifest_sha256"})
    assert len(m["family_ids"]) == len(set(m["family_ids"])) == m["num_families"] == 1400
    assert len(m["image_sha256"]) == m["num_unique_images"] == 973
    assert sum(m["qa_types"].values()) == m["num_qa"] == 9800
    assert m["qa_types"]["REL2C_LEFT"] + m["qa_types"]["REL2C_RIGHT"] == m["num_composition"] == 2800


def test_portable_verification_rejects_image_tampering(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    image = frozen / "image.jpg"
    image.write_bytes(b"original fixture bytes")
    families = frozen / "families.jsonl"
    families.write_text(json.dumps({"image_id": "I", "image_path": "image.jpg"}) + "\n")
    m = {"frozen": True, "export_file_hashes": {"frozen/families.jsonl": sha256_file(families)}, "image_sha256": {"I": sha256_file(image)}}
    m["manifest_sha256"] = stable_hash(m)
    verify_export(tmp_path, m)
    image.write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError, match="image hash mismatch"):
        verify_export(tmp_path, m)
