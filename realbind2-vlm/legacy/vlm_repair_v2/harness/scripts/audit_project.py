#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from harness.configs.paths import Paths
from harness.dataset import load_frozen
from harness.io_utils import write_json_new, write_new


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = Paths(); p.validate()
    families, qa = load_frozen(p)
    type_counts = collections.Counter(row["qa_type"] for row in qa)
    composition = [row for row in qa if row["qa_type"] in {"REL2C_LEFT", "REL2C_RIGHT"}]
    if len(families) != 100 or len(qa) != 700 or len(composition) != 200: raise RuntimeError("frozen counts changed")
    audit_path = p.realbind / "audit/dryrun_100_pair_direct_v1_1_audit.json"
    scorer_text = p.scorer.read_text(encoding="utf-8")
    if 'SCORING_REVISION = "realbind_pair_direct_v1_1_semantic_vocab_v2"' not in scorer_text: raise RuntimeError("frozen scorer revision changed")
    historical_candidates = [
        p.realbind / "experiments/pair_direct_v1_1_qwen35_4b_baseline/inference_results.jsonl",
        p.realbind / "experiments/pair_direct_v1_1_qwen35_4b_baseline/summary.json",
        p.realbind / "experiments/qwen35_dah_oracle_v1/original_results.jsonl",
        p.realbind / "experiments/qwen35_dah_oracle_v1/full_dah_results.jsonl",
        p.realbind / "experiments/qwen35_dah_oracle_v1/full_dah_pair_rel2i_stage1.jsonl",
        p.realbind / "experiments/qwen35_dah_oracle_v1/summary_metrics.json",
    ]
    model_config = p.qwen / "config.json"
    manifest = {
        "schema_version": "realbind_visual_repair_v2_frozen_manifest_v1",
        "source_paths": {"project": str(p.project), "realbind": str(p.realbind), "families": str(p.families), "qa": str(p.qa), "dependency": str(p.dependency), "dataset_audit": str(audit_path), "scorer": str(p.scorer)},
        "sha256": {"families": sha256(p.families), "qa": sha256(p.qa), "dependency": sha256(p.dependency), "dataset_audit": sha256(audit_path), "scorer": sha256(p.scorer), "model_config": sha256(model_config)},
        "counts": {"families": len(families), "qa": len(qa), "composition": len(composition), "by_type": dict(sorted(type_counts.items()))},
        "family_ids": [str(row["family_id"]) for row in families],
        "composition_case_ids": [str(row["qa_id"]) for row in composition],
        "image_paths": sorted({str(row["image_path"]) for row in families}),
        "schema": {"A_B": "family.A/family.B objects with instance_id, object, canonical_color, bbox [x,y,w,h]", "prompts": "frozen pair-direct v1.1 question fields explicitly name A/B object categories", "left_right": "family.left_is_A plus QA target_side"},
        "scorer": {"version": "realbind_pair_direct_v1_1_semantic_vocab_v2", "path": str(p.scorer)},
        "model": {"identifier": "Qwen/Qwen3.5-4B", "path": str(p.qwen), "existing_oracle_manifest_hash": "ef8aca80440bcc9544ef1e423cf5f91f7d76f8aecfc36bde2c1544246c82f287"},
        "prompt": {"dataset_schema_version": "realbind_pair_direct_v1_1", "template_ids": sorted({str(row["template_id"]) for row in qa})},
        "generation": {"do_sample": False, "use_cache": False, "max_new_tokens": 16, "batch_size": 1, "dtype": "bfloat16", "enable_thinking": False},
        "historical_result_paths": [str(path) for path in historical_candidates if path.is_file()],
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json_new(p.manifests / "frozen_manifest.json", manifest)
    report = f"""# Environment and dataset audit

Status: **PASS**. This was a read-only audit of the frozen sources; no dataset row, prompt, score, or historical result was changed.

## Discovered paths

- Project: `{p.project}`
- Frozen families: `{p.families}`
- Frozen QA: `{p.qa}`
- Frozen scorer: `{p.scorer}`
- Qwen3.5-4B: `{p.qwen}`
- New D-drive root: `{p.root}`

## Frozen data

- Families: {len(families)}
- QA: {len(qa)}
- REL2C cases: {len(composition)} (`REL2C_LEFT=100`, `REL2C_RIGHT=100`)
- A/B are represented by `family.A` and `family.B`, each with object identity, unique instance ID, canonical color, and a GT `[x,y,w,h]` bbox.
- The actual v1.1 REL2C questions explicitly name both object categories; they do not use opaque A/B aliases.
- Scorer revision: `realbind_pair_direct_v1_1_semantic_vocab_v2`.
- Historical Original pooled REL2C accuracy: 0.490; historical Oracle Full-DaH: 0.620 (existing saved summaries).

## Protocol boundary

GT boxes are retained only in frozen source/evaluator records. The inference structures in the new harness reject GT-localization keys, and Auto-Defocus accepts only typed predicted regions.

See `{p.manifests / 'frozen_manifest.json'}` for hashes and all 200 composition case IDs.
"""
    write_new(p.outputs / "audits/environment_and_dataset_audit.md", report)
    print(p.manifests / "frozen_manifest.json")
    return 0


if __name__ == "__main__": raise SystemExit(main())
