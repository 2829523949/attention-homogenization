from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.failure.defocus_frozen import png_bytes, render_pair_defocus


FIXTURES = Path(__file__).parent / "fixtures" / "defocus"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_defocus_matches_five_historical_artifacts() -> None:
    manifest = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 5
    for case in manifest["cases"]:
        source = FIXTURES / case["input"]
        expected = FIXTURES / case["expected"]
        assert sha256(source) == case["input_sha256"]
        assert sha256(expected) == case["expected_sha256"]
        with Image.open(source) as opened:
            actual, _ = render_pair_defocus(opened, case["bbox_A"], case["bbox_B"])
        with Image.open(expected) as historical:
            assert np.array_equal(np.asarray(actual), np.asarray(historical.convert("RGB")))
        assert hashlib.sha256(png_bytes(actual)).hexdigest() == case["expected_sha256"]

