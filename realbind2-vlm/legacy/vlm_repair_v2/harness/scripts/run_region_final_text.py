#!/usr/bin/env python3
"""Frozen image-free final reasoner for the region-bottleneck pilot routes.

This script is deliberately isolated from visual inference: it reads the
already-compiled minimal state for each QA/route, sends only the question plus
that state to Qwen in text-only mode, and scores the response with the frozen
scorer.  It is run for all four routes once the infrastructure exists; the
predeclared 18/40 gate is reported separately.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/d/vlm_repair_v2")
sys.path.insert(0, str(ROOT))

from harness.configs.paths import Paths
from harness.dataset import read_jsonl
from harness.evaluators.end2end_metrics import score
from harness.processors.color_pixel import REALBIND_COLORS
from harness.reasoners.qwen35 import Qwen35Reasoner
from harness.scripts.run_minimal_decomposition_route_pilot import prompt

OUT = ROOT / "outputs/region_bottleneck_joint_repair"
ROUTES = ("R1", "R2", "R3", "R4")


def rows(path: Path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, row: dict):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: Path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    states = {(row["qa_id"], row["route"]): row for row in rows(OUT / "minimal_state_results.jsonl")}
    prepared = {row["qa_id"]: row for row in rows(ROOT / "outputs/minimal_decomposition_route_pilot/prepared.jsonl")}
    assert len(states) == 160 and len(prepared) == 40
    families = {row["family_id"]: row for row in read_jsonl(Paths().families)}
    vocab = {
        "color": list(REALBIND_COLORS),
        "position": ["left", "right"],
        "identity": sorted({family[key]["object"] for family in families.values() for key in ("A", "B")}),
    }
    result_path = OUT / "end_to_end_results.jsonl"
    done = {(row["qa_id"], row["route"]) for row in rows(result_path)} if result_path.exists() else set()
    reasoner = Qwen35Reasoner()
    total = len(states)
    for index, ((qa_id, route), state) in enumerate(sorted(states.items()), 1):
        if (qa_id, route) in done:
            continue
        qa = prepared[qa_id]
        query = prompt(qa["question"], state["state_text"])
        generation = reasoner.generate(query, None)
        assert generation["image_path"] is None
        assert not generation["vision_encoder_called"]
        assert not generation["visual_tokens_in_rendered_prompt"]
        assert not any("pixel" in key or "image" in key or "video" in key for key in generation["input_keys"])
        family = families[qa["family_id"]]
        scored = score(generation["response"], "color", state["gt_color"], family, vocab)
        append(result_path, {
            "qa_id": qa_id,
            "route": route,
            "state_sha256": state["state_sha256"],
            "state_sufficient": bool(state["minimal_state_sufficient"]),
            "target_correct": bool(state["target_correct"]),
            "target_region_correct": bool(state["target_region_correct"]),
            "target_color_correct": bool(state["target_color_correct"]),
            "question": qa["question"],
            "state_text": state["state_text"],
            "prompt_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "raw_response": generation["response"],
            "prediction": scored["normalized_prediction"],
            "invalid_prediction": bool(scored["invalid_prediction"]),
            "final_correct": bool(scored["correct"]),
            "image_path": None,
            "vision_encoder_called": False,
            "scoring_revision": scored["scoring_revision"],
        })
        print(f"final {index}/{total} {route} {qa_id} sufficient={state['minimal_state_sufficient']} pred={scored['normalized_prediction']} correct={scored['correct']}", flush=True)
    final = rows(result_path)
    metrics = {}
    for route in ROUTES:
        part = [row for row in final if row["route"] == route]
        metrics[route] = {
            "n": len(part),
            "final_correct": sum(row["final_correct"] for row in part),
            "final_accuracy": sum(row["final_correct"] for row in part) / len(part) if part else None,
            "state_sufficient": sum(row["state_sufficient"] for row in part),
            "sufficient_and_final_correct": sum(row["state_sufficient"] and row["final_correct"] for row in part),
            "insufficient_and_final_correct": sum((not row["state_sufficient"]) and row["final_correct"] for row in part),
            "insufficient_and_final_wrong": sum((not row["state_sufficient"]) and (not row["final_correct"]) for row in part),
            "vision_encoder_call_count": sum(bool(row["vision_encoder_called"]) for row in part),
        }
    write_json(OUT / "end_to_end_metrics.json", metrics)
    write_json(OUT / "end_to_end_gate.json", {
        "gate_threshold_state_sufficiency": 18,
        "state_sufficiency": json.loads((OUT / "route_comparison.json").read_text()),
        "run_all_four_routes": True,
        "reason": "Final text-only reasoner was run on all four routes to report measured end-to-end accuracy and compensating correctness, although no route reached the 18/40 state-sufficiency gate.",
        "vanilla_vlm_locked_accuracy": 20,
        "old_minimal_specialist_locked_accuracy": 15,
    })
    print("FINAL_COMPLETE", metrics, flush=True)


if __name__ == "__main__":
    main()
