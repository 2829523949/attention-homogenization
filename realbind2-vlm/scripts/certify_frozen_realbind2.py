#!/usr/bin/env python3
"""Read-only certification of a builder export with human-decision evidence."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legacy.realbind_v2_builder.question_generator import QA_ORDER, generate_questions
from src.common.manifests import sha256_file, stable_hash
from src.common.prompts import prompt_bundle_hash, template_hashes
from src.common.scorer import SCORER_VERSION, score_prediction


def certify(root: Path) -> dict:
    frozen = root / "frozen"
    families = [json.loads(x) for x in (frozen / "families.jsonl").read_text().splitlines() if x.strip()]
    questions = [json.loads(x) for x in (frozen / "questions.jsonl").read_text().splitlines() if x.strip()]
    stats = json.loads((frozen / "construction_stats.json").read_text())
    splits = json.loads((frozen / "splits.json").read_text())
    basis = json.loads((ROOT / "data_audit/human_acceptance_basis.json").read_text())
    decisions_path = ROOT / "data_audit/human_acceptance_decisions.jsonl"
    assert sha256_file(decisions_path) == basis["decision_file_sha256"]
    decisions = [json.loads(x) for x in decisions_path.read_text().splitlines() if x.strip()]
    accepted_ids = {d["candidate_family_id"] for d in decisions if d["decision"] == "HUMAN_ACCEPTED"}
    assert len(accepted_ids) == len(decisions) == len(families)
    assert {f["candidate_family_id"] for f in families} == accepted_ids
    assert len({f["family_id"] for f in families}) == len(families)
    assert len({q["qa_id"] for q in questions}) == len(questions)
    by_family = defaultdict(list)
    for q in questions:
        by_family[q["family_id"]].append(q)
        assert score_prediction(q["ground_truth"], q["ground_truth"], q["answer_type"]).correct
    assert set(by_family) == {f["family_id"] for f in families}
    image_splits = {}
    image_hashes = {}
    pairs = set()
    for f in families:
        qs = by_family[f["family_id"]]
        assert tuple(q["qa_type"] for q in qs) == QA_ORDER
        assert qs == generate_questions(f), f["family_id"]
        pair = (str(f["image_id"]), *sorted((str(f["object_A_id"]), str(f["object_B_id"]))))
        assert pair not in pairs
        pairs.add(pair)
        image_id = str(f["image_id"])
        assert image_splits.setdefault(image_id, f["split"]) == f["split"]
        path = frozen / f["image_path"]
        assert path.resolve().is_relative_to(frozen.resolve())
        if image_id not in image_hashes:
            with Image.open(path) as image:
                image.load()
                assert image.size == (f["width"], f["height"])
            image_hashes[image_id] = sha256_file(path)
    assert not set(splits["validation"]) & set(splits["test"])
    assert set(splits["validation"]) | set(splits["test"]) == set(image_hashes)
    for image_id, split in image_splits.items():
        assert image_id in splits[split]
    db_path = root / "state/construction_state.sqlite"
    db = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    state = {r["candidate_family_id"]: r for r in db.execute(
        "SELECT candidate_family_id,status,annotator,decision_at FROM families WHERE status='FROZEN'"
    )}
    assert set(state) == accepted_ids
    assert all(r["annotator"] and r["decision_at"] for r in state.values())
    accepted_events = {r[0] for r in db.execute("SELECT candidate_family_id FROM events WHERE event='HUMAN_ACCEPTED'")}
    assert accepted_ids <= accepted_events
    events = list(db.execute("SELECT details_json FROM events WHERE event='DATASET_FROZEN'"))
    assert events and json.loads(events[-1][0]) == stats
    hashes = stats["hashes"]
    assert sha256_file(frozen / "families.jsonl") == hashes["family_manifest_sha256"]
    assert sha256_file(frozen / "questions.jsonl") == hashes["question_manifest_sha256"]
    assert sha256_file(frozen / "dependencies.json") == hashes["dependency_graph_sha256"]
    assert sha256_file(root / "realbind_v2_protocol.yaml") == hashes["protocol_config_sha256"]
    # The preserved builder's image digest includes one entry per family.
    import hashlib
    image_digest = hashlib.sha256("\n".join(sorted(
        f"{f['image_id']}:{image_hashes[str(f['image_id'])]}" for f in families
    )).encode()).hexdigest()
    assert image_digest == hashes["image_manifest_sha256"]
    db.close()
    counts = Counter(q["qa_type"] for q in questions)
    assert stats["families"] == len(families) and stats["questions"] == len(questions)
    assert stats["unique_images"] == len(image_hashes)
    source = json.loads((root / "manifests/mine_summary.json").read_text())
    manifest = {
        "dataset_name": "RealBind2", "benchmark_name": "RealBind2",
        "dataset_version": "RealBind-v2.0", "source_version": "RealBind-v2.0-construction-pre1",
        "dataset_unit": "family", "frozen": True, "freeze_status": "FROZEN",
        "formal_run_allowed": True, "freeze_basis": basis,
        "num_families": len(families), "num_unique_images": len(image_hashes),
        "num_qa": len(questions), "num_composition": counts["REL2C_LEFT"] + counts["REL2C_RIGHT"],
        "qa_types": dict(sorted(counts.items())),
        "family_ids": [f["family_id"] for f in families],
        "candidate_family_ids": [f["candidate_family_id"] for f in families],
        "image_ids": sorted(image_hashes), "image_sha256": dict(sorted(image_hashes.items())),
        "split_information": splits,
        "protocol_expected": {"families": 1400, "qa": 9800, "composition_qa": 2800, "qa_per_family": 7},
        "export_file_hashes": {str(p.relative_to(root)): sha256_file(p) for p in sorted(frozen.iterdir()) if p.is_file()},
        "source_annotation_hashes": {str(Path(r["path"]).relative_to(Path(r["path"]).parents[2])): r["sha256"] for r in source["source_fingerprints"]},
        "source_snapshot": "state/construction_state.sqlite", "source_snapshot_sha256": sha256_file(db_path),
        "prompt_hash": prompt_bundle_hash(), "prompt_bundle_sha256": prompt_bundle_hash(),
        "prompt_template_hashes": template_hashes(), "scorer_hash": sha256_file(ROOT / "src/common/scorer.py"),
        "scorer": {"version": SCORER_VERSION, "sha256": sha256_file(ROOT / "src/common/scorer.py")},
        "manifest_hash_definition": "SHA256 of canonical JSON excluding manifest_sha256; exact file SHA256 in REALBIND2_FROZEN.sha256",
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = certify(args.dataset_root.resolve())
    path = ROOT / "data/realbind2_manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and json.loads(path.read_text()).get("frozen") and path.read_text() != payload:
        raise RuntimeError("existing frozen manifest differs; do not overwrite")
    path.write_text(payload)
    (ROOT / "data/REALBIND2_FROZEN.sha256").write_text(f"{sha256_file(path)}  data/realbind2_manifest.json\n")
    (ROOT / "data_audit/frozen_export_certification.json").write_text(json.dumps({
        "passed": True, "num_families": manifest["num_families"],
        "num_unique_images": manifest["num_unique_images"], "num_qa": manifest["num_qa"],
        "num_composition": manifest["num_composition"], "qa_types": manifest["qa_types"],
        "human_decision_events_verified": True, "dataset_frozen_event_verified": True,
        "image_hashes_verified": len(manifest["image_ids"]), "split_leakage": False,
        "qa_regeneration_exact": True, "manifest_file_sha256": sha256_file(path),
    }, indent=2, sort_keys=True) + "\n")
    print(f"REALBIND2_DATASET_CERTIFIED families={manifest['num_families']} unique_images={manifest['num_unique_images']} QA={manifest['num_qa']} composition={manifest['num_composition']}")


if __name__ == "__main__":
    main()
