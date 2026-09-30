#!/usr/bin/env python3
"""Fast self-contained contract smoke plus historical artifact replay.

This validates the exact interfaces needed before a live model smoke.  A live
smoke additionally requires a human-frozen RealBind2 export and local weights;
the shell entrypoint enforces that gate when ``SMOKE_MODE=live``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging_utils import JsonlRunLog
from src.common.scorer import score_prediction
from src.mechanism.residual_patch import PatchSpec, assert_self_patch_preserved, self_patch
from src.repair.strong_integrated.cache import StageCache
from src.repair.strong_integrated.contracts import assert_final_handoff, assert_inference_safe

ROOT = Path(__file__).resolve().parents[1]


def contract_smoke() -> None:
    score = score_prediction("Final answer: blue", "blue", "color")
    assert score.correct and score.valid
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = JsonlRunLog(root / "resume.jsonl")
        log.append({"case_id": "smoke", "status": "ok"})
        resumed = JsonlRunLog(root / "resume.jsonl")
        assert resumed.is_complete("smoke")

        selected = StageCache(root / "selected_targets.jsonl", ("family_id", "logical_ref"))
        selected.append({
            "family_id": "smoke_family",
            "logical_ref": "A",
            "bbox_xyxy": [1, 1, 8, 8],
            "selected_provider": "R4_MM_GDINO_L_QWEN_SELECTOR",
            "sam3_mask_valid": True,
        })
        attribute = StageCache(root / "attributes.jsonl", ("family_id", "logical_ref"))
        attribute.append({
            "family_id": "smoke_family",
            "logical_ref": "A",
            "reader": "crop_qwen35",
            "color": "blue",
            "sam3_isolation_used": True,
        })
        for row in (*selected.rows.values(), *attribute.rows.values()):
            assert_inference_safe(row)
        handoff = {"minimal_state": "[TARGET_ENTITY] A\n[TARGET_ATTRIBUTE] color = blue", "image_present": False}
        assert_final_handoff(handoff)

    state = (1.0, 2.0, 3.0)
    spec = PatchSpec("smoke", "smoke", 0, (0,), "forward")
    patched = self_patch(state, spec)
    assert_self_patch_preserved(state, patched)

    strong = json.loads((ROOT / "legacy/evidence/strong_integrated/gt_leakage_audit.json").read_text())
    r4 = json.loads((ROOT / "legacy/evidence/r4_region/gt_leakage_audit.json").read_text())
    noop = json.loads((ROOT / "legacy/evidence/causal/noop_patch_audit.json").read_text())
    assert strong["pass"] is True
    assert r4.get("pass", r4.get("passed")) is True
    assert noop.get("passed", noop.get("pass")) is True


def live_preflight(paths_file: Path) -> None:
    import yaml
    paths = yaml.safe_load(paths_file.read_text(encoding="utf-8"))
    required = [
        Path(paths["data"]["realbind2"]) / "frozen" / "families.jsonl",
        Path(paths["models"]["qwen35_4b"]),
        Path(paths["models"]["mm_gdino_l"]),
        Path(paths["models"]["sam3"]),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("LIVE_SMOKE_BLOCKED missing=" + json.dumps(missing))
    raise SystemExit(
        "LIVE_SMOKE_BLOCKED: the unified one-case model executor is not certified against a frozen RealBind2 export; "
        "do not treat artifact replay as fresh model execution"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("artifact", "live"), default="artifact")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs" / "paths.yaml")
    args = parser.parse_args()
    contract_smoke()
    if args.mode == "live":
        live_preflight(args.paths)
    print("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
