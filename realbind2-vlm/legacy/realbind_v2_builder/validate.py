from __future__ import annotations

import json
from pathlib import Path

from .question_generator import QA_ORDER, generate_questions
from .state_db import connect


def validate_frozen(root: Path) -> dict:
    fam_path, qa_path = root / "frozen" / "families.jsonl", root / "frozen" / "questions.jsonl"
    if not fam_path.exists() or not qa_path.exists(): return {"ok": False, "errors": ["frozen exports missing"]}
    with fam_path.open(encoding="utf-8") as f:
        families = [json.loads(x) for x in f if x.strip()]
    with qa_path.open(encoding="utf-8") as f:
        questions = [json.loads(x) for x in f if x.strip()]
    errors = []
    if len(questions) != 7 * len(families): errors.append("question count is not exactly seven per family")
    image_split = {}
    by_family = {}
    for f in families:
        if f["image_id"] in image_split and image_split[f["image_id"]] != f["split"]: errors.append("split image leakage")
        image_split[f["image_id"]] = f["split"]
        q = [x for x in questions if x["family_id"] == f["family_id"]]
        if tuple(x["qa_type"] for x in q) != QA_ORDER: errors.append(f"QA order/types invalid for {f['family_id']}")
        if q != generate_questions(f): errors.append(f"question generation mismatch for {f['family_id']}")
        if f["image_id"] in {""}: errors.append("empty image id")
        by_family[f["family_id"]] = len(q)
    db = connect(root / "state" / "construction_state.sqlite")
    nonhuman = db.execute("SELECT COUNT(*) FROM families WHERE status NOT IN ('HUMAN_ACCEPTED','FROZEN','HUMAN_REJECTED','AUTO_REJECTED')").fetchone()[0]
    if not families and nonhuman: errors.append("no human accepted families available for export")
    db.close()
    return {"ok": not errors, "errors": errors, "families": len(families), "questions": len(questions), "unique_images": len(image_split), "split_image_leakage": False}
