#!/usr/bin/env python3
"""CPU-only contract test for PACO benchmark/conditioning plumbing."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "configs/paths.yaml"
DATA = Path("/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k")
BENCH_SCRIPT = ROOT / "scripts/build_benchmark.py"
FINAL_SCRIPT = ROOT / "scripts/finalize_conditioned_set.py"
VERIFY_SCRIPT = ROOT / "scripts/verify_conditioned_set.py"


def run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main():
    with tempfile.TemporaryDirectory(prefix="paco-cpu-contract-") as tmp:
        tmp = Path(tmp)
        bench = tmp / "benchmark"
        run([sys.executable, str(BENCH_SCRIPT),
             "--paths", str(PATHS), "--dataset-root", str(DATA),
             "--candidate-manifest", str(DATA / "manifests/candidate_families_test_val.jsonl"),
             "--image-checksums", str(DATA / "manifests/image_checksums.csv"),
             "--splits", "test", "--max-families", "24", "--output-dir", str(bench)])
        families = [json.loads(l) for l in (bench / "families_all.jsonl").read_text().splitlines()]
        questions = [json.loads(l) for l in (bench / "questions_all.jsonl").read_text().splitlines()]
        spec = json.loads((bench / "split_spec.json").read_text())
        assert len(families) == 24 and len(questions) == 48
        assert spec["task_semantics"]["question_color_semantics"] == "dominant"
        assert {q["qa_type"] for q in questions} == {"REL2C_LEFT", "REL2C_RIGHT"}
        for f in families:
            for ref in ("A", "B"):
                assert f[f"referent_{ref}"]
                assert f[f"referent_{ref}_info"]["basis"] in {"unique_category", "unique_noncolor_attribute"}
            assert f["referent_A"] != f["referent_B"]
            assert f["color_A"] in {"red","orange","yellow","green","blue","purple","pink","brown","black","white","gray"}
            assert f["color_B"] in {"red","orange","yellow","green","blue","purple","pink","brown","black","white","gray"}
            assert f["annotation_ab_iou"] < 0.1
            assert max(f["A_inside_B_fraction"], f["B_inside_A_fraction"]) < 0.5
            assert f["A_bbox_area_ratio"] >= 0.005 and f["B_bbox_area_ratio"] >= 0.005
        for q in questions:
            assert "dominant color" in q["question"]
        # Simulated selector boxes equal to annotation boxes: exercises join,
        # geometry diagnostics, strict IoU<0.1 decision, and question retention.
        selected_dir = tmp / "selector_primary"; selected_dir.mkdir(parents=True)
        rows = []
        for f in families:
            for ref in ("A", "B"):
                rows.append({
                    "task_id": f"{f['family_id']}|{ref}", "family_id": f["family_id"],
                    "logical_ref": ref, "referent_phrase": f[f"referent_{ref}"],
                    "candidate_count": 2, "selected_index": 0,
                    "selected_bbox_xyxy": f[f"bbox_{ref}_xyxy"],
                    "qwen_status": "VALID", "selector_raw": "1", "status": "ok",
                    "selection_model_id": "Qwen3.5-4B", "selection_model_tag": "qwen35_4b",
                })
        (selected_dir / "selected_targets.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        (selected_dir / "identity.json").write_text(json.dumps({"model_tag": "qwen35_4b"}))
        conditioned = tmp / "conditioned"
        run([sys.executable, str(FINAL_SCRIPT), "--paths", str(PATHS),
             "--families", str(bench / "families_all.jsonl"),
             "--questions", str(bench / "questions_all.jsonl"),
             "--selected-targets", str(selected_dir / "selected_targets.jsonl"),
             "--output-dir", str(conditioned)])
        audit = [json.loads(l) for l in (conditioned / "family_conditioning_audit.jsonl").read_text().splitlines()]
        for row in audit:
            expected = (not row["same_description"]) and (row["ab_bbox_iou"] < 0.1)
            assert (row["decision"] == "KEEP") == expected
        conditioned_questions = [json.loads(l) for l in (conditioned / "benchmark.jsonl").read_text().splitlines()]
        qa_audit = [json.loads(l) for l in (conditioned / "selection_audit.jsonl").read_text().splitlines()]
        kept = {row["family_id"] for row in audit if row["decision"] == "KEEP"}
        assert {q["family_id"] for q in conditioned_questions} == kept
        assert len(conditioned_questions) == 2 * len(kept)
        assert {a["qa_id"] for a in qa_audit} == {q["qa_id"] for q in conditioned_questions}
        run([sys.executable, str(VERIFY_SCRIPT), "--dataset-dir", str(conditioned)])
        print(json.dumps({"status": "PASS", "families_all": len(families),
                          "questions_all": len(questions), "families_conditioned": len(kept),
                          "questions_conditioned": len(conditioned_questions)}, indent=2))


if __name__ == "__main__":
    main()
