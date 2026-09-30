from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .question_generator import DEPENDENCIES, QA_ORDER, generate_questions
from .state_db import connect, event

REJECT_REASONS = ("INVALID_BBOX", "OBJECT_TOO_SMALL", "ATTRIBUTE_MISSING", "ATTRIBUTE_CONFLICT", "SAME_TARGET_ATTRIBUTE", "AMBIGUOUS_GEOMETRY", "HIGH_BBOX_OVERLAP", "REFERENT_NOT_UNIQUE", "REFERENT_LEAKS_COLOR", "REFERENT_LEAKS_RELATION", "IMAGE_CORRUPTED", "DIMENSION_MISMATCH", "DUPLICATE_IMAGE", "DUPLICATE_FAMILY", "HUMAN_REJECT", "NO_VALID_PAIR")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def build_audit_html(root: Path) -> dict:
    db = connect(root / "state" / "construction_state.sqlite")
    out = root / "audit" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = db.execute("SELECT f.candidate_family_id,f.payload_json,i.accepted_path FROM families f JOIN images i USING(image_id) WHERE f.status IN ('AUTO_VALID','HUMAN_PENDING','HUMAN_ACCEPTED','HUMAN_REJECTED') ORDER BY f.image_id,f.candidate_family_id").fetchall()
    cards = []
    for row in rows:
        fam = json.loads(row["payload_json"])
        fam["family_id"] = fam["candidate_family_id"]
        questions = generate_questions(fam)
        x, y, w, h = fam["bbox_A"]
        xa, ya, wa, ha = fam["bbox_B"]
        img = html.escape(os.path.relpath(row["accepted_path"], out.parent))
        qa = "".join(f"<li><b>{html.escape(q['qa_type'])}</b>: {html.escape(q['question'])} <em>→ {html.escape(q['ground_truth'])}</em></li>" for q in questions)
        cards.append(f'''<article data-family="{html.escape(row["candidate_family_id"])}">
<h2>{html.escape(row["candidate_family_id"])} · VG {html.escape(fam["image_id"])}</h2>
<div class="image"><img src="{img}"><svg viewBox="0 0 {fam["width"]} {fam["height"]}" preserveAspectRatio="none"><rect x="{x}" y="{y}" width="{w}" height="{h}"/><rect class="b" x="{xa}" y="{ya}" width="{wa}" height="{ha}"/></svg></div>
<p><b>A:</b> {html.escape(fam["referent_A"])} · color={html.escape(fam["color_A"])} · box={fam["bbox_A"]}<br><b>B:</b> {html.escape(fam["referent_B"])} · color={html.escape(fam["color_B"])} · box={fam["bbox_B"]}<br><b>Relation:</b> A is {"LEFT" if fam["geometry_metrics"]["left_is_A"] else "RIGHT"} of B · same category={fam["same_category"]} · IoU={fam["geometry_metrics"]["iou"]:.3f}</p>
<ol>{qa}</ol><label>Annotator <input class="annotator" placeholder="alias"></label>
<label>Decision <select class="decision"><option value="">Pending</option><option>HUMAN_ACCEPTED</option><option>HUMAN_REJECTED</option></select></label>
<label>Reject reason <select class="reason"><option value="">—</option>{''.join(f'<option>{r}</option>' for r in REJECT_REASONS)}</select></label><label>Note <input class="note"></label></article>''')
    page = '''<!doctype html><meta charset="utf-8"><title>RealBind-v2 human audit</title><style>body{font:16px system-ui;max-width:1300px;margin:auto}article{border:1px solid #aaa;padding:16px;margin:20px 0}.image{position:relative;width:100%;max-height:75vh}.image img{display:block;max-width:100%;max-height:75vh}.image svg{position:absolute;inset:0;width:100%;height:100%;stroke:#ff2d2d;stroke-width:4;fill:#ff2d2d22}.image svg .b{stroke:#1675ff;fill:#1675ff22}label{margin-right:12px}#export{position:sticky;top:0;padding:12px;background:#fff}</style>
<div id="export"><b>Human visual audit</b> — save decisions to JSONL, then import with <code>build_realbind_v2.py audit --import-decisions FILE</code>.<button onclick="download()">Download decisions.jsonl</button></div>
''' + "\n".join(cards) + '''<script>function download(){let out=[];document.querySelectorAll('article').forEach(a=>{let d=a.querySelector('.decision').value;if(!d)return;out.push({candidate_family_id:a.dataset.family,decision:d,annotator:a.querySelector('.annotator').value||'single-review',reject_reason:a.querySelector('.reason').value||null,note:a.querySelector('.note').value||''})});let b=new Blob([out.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{type:'application/x-ndjson'});let u=URL.createObjectURL(b),x=document.createElement('a');x.href=u;x.download='decisions.jsonl';x.click();URL.revokeObjectURL(u)}</script>'''
    out.write_text(page, encoding="utf-8")
    db.close()
    return {"family_cards": len(rows), "html": str(out)}


def import_decisions(root: Path, path: Path) -> dict:
    db = connect(root / "state" / "construction_state.sqlite")
    seen, bad = 0, []
    with db, path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip(): continue
            item = json.loads(line)
            fid, decision = item["candidate_family_id"], item["decision"]
            if decision not in ("HUMAN_ACCEPTED", "HUMAN_REJECTED"):
                bad.append({"line": line_no, "error": "invalid decision"}); continue
            if decision == "HUMAN_REJECTED" and item.get("reject_reason") not in REJECT_REASONS:
                bad.append({"line": line_no, "error": "missing/invalid standardized reject reason"}); continue
            row = db.execute("SELECT image_id,status FROM families WHERE candidate_family_id=?", (fid,)).fetchone()
            if not row or row["status"] not in ("AUTO_VALID", "HUMAN_PENDING"):
                bad.append({"line": line_no, "error": "family not available for human review"}); continue
            stamp = datetime.now(timezone.utc).isoformat()
            reasons = [] if decision == "HUMAN_ACCEPTED" else [item["reject_reason"]]
            db.execute("UPDATE families SET status=?,reject_reasons=?,annotator=?,decision_at=?,note=? WHERE candidate_family_id=?",
                       (decision, json.dumps(reasons), item.get("annotator", "single-review"), stamp, item.get("note", ""), fid))
            event(db, row["image_id"], fid, decision, {"annotator": item.get("annotator", "single-review"), "timestamp": stamp, "reject_reason": reasons, "note": item.get("note", "")})
            db.execute("UPDATE images SET status='HUMAN_PENDING' WHERE image_id=? AND status='AUTO_ACCEPTED'", (row["image_id"],))
            seen += 1
    db.close()
    return {"decisions_imported": seen, "errors": bad}


def export_frozen(root: Path) -> dict:
    db = connect(root / "state" / "construction_state.sqlite")
    out = root / "frozen"; (out / "images").mkdir(parents=True, exist_ok=True)
    accepted = db.execute("SELECT f.candidate_family_id,f.image_id,f.payload_json,i.accepted_path,i.sha256 FROM families f JOIN images i USING(image_id) WHERE f.status='HUMAN_ACCEPTED' ORDER BY f.image_id,f.candidate_family_id").fetchall()
    images = sorted({r["image_id"] for r in accepted})
    val_ids = {i for i in images if int(hashlib.sha256(("realbind-v2-split:" + i).encode()).hexdigest()[:8], 16) % 5 == 0}
    family_rows, questions = [], []
    for i, row in enumerate(accepted, 1):
        fam = json.loads(row["payload_json"])
        fid = f"RB2_{i:06d}"
        fam.update({"family_id": fid, "image_path": f"images/{row['image_id']}.jpg", "source": "VAW+Visual Genome",
                    "relation_A_to_B": "LEFT" if fam["geometry_metrics"]["left_is_A"] else "RIGHT",
                    "left_entity": "A" if fam["geometry_metrics"]["left_is_A"] else "B",
                    "right_entity": "B" if fam["geometry_metrics"]["left_is_A"] else "A",
                    "protocol_version": "RealBind-v2.0", "split": "validation" if row["image_id"] in val_ids else "test"})
        family_rows.append(fam); questions.extend(generate_questions(fam))
        src, dest = Path(row["accepted_path"]), out / "images" / f"{row['image_id']}.jpg"
        if not dest.exists():
            try: os.link(src, dest)
            except OSError: shutil.copy2(src, dest)
    def dump(path, rows):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    dump(out / "families.jsonl", family_rows); dump(out / "questions.jsonl", questions)
    split = {"validation": sorted(val_ids), "test": sorted(set(images) - val_ids), "group_key": "Visual Genome image_id", "v1_1_dev_images_excluded": True}
    (out / "splits.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (out / "dependencies.json").write_text(json.dumps(DEPENDENCIES, indent=2, sort_keys=True) + "\n")
    protocol = {"dataset_version": "RealBind-v2.0", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_version": "RealBind-v2.0-construction-pre1", "construction_is_model_agnostic": True,
                "qa_order": list(QA_ORDER), "color_vocabulary": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"],
                "atomic_ready": "I2C_A_correct AND I2C_B_correct AND PAIR_REL_A_TO_B_correct",
                "PCFR_atomic": "P(REL2C_LEFT_or_RIGHT_fail | AtomicReady), report LEFT, RIGHT, pooled",
                "PCFR_binding_LEFT": "P(REL2C_LEFT_fail | AtomicReady AND REL2I_LEFT_correct)",
                "PCFR_binding_RIGHT": "P(REL2C_RIGHT_fail | AtomicReady AND REL2I_RIGHT_correct)",
                "family_success": "all seven QA correct"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (out / "dataset_card.md").write_text("""# RealBind-v2.0

RealBind is a family-structured, prerequisite-controlled benchmark for visual compositional binding. Each compositional query is paired with the exact perceptual, relational, and referential prerequisite probes required to answer it, allowing direct measurement of composition failures that persist despite successful prerequisite judgments.

The benchmark uses seven deterministic questions per family and splits at the Visual Genome image ID level. Construction uses source annotations, fixed geometry gates, deterministic referent leakage checks, and human visual review. No VLM task performance was used during construction. RealBind-v1.1 images are excluded from the frozen release.
""", encoding="utf-8")
    family_path, qa_path = out / "families.jsonl", out / "questions.jsonl"
    image_digest = hashlib.sha256("\n".join(sorted(f"{r['image_id']}:{r['sha256']}" for r in accepted)).encode()).hexdigest()
    protocol_config = root / "realbind_v2_protocol.yaml"
    hashes = {"family_manifest_sha256": _sha(family_path), "question_manifest_sha256": _sha(qa_path), "dependency_graph_sha256": _sha(out / "dependencies.json"), "image_manifest_sha256": image_digest,
              "protocol_config_sha256": _sha(protocol_config if protocol_config.exists() else out / "protocol.json")}
    construction_stats = {"families": len(family_rows), "questions": len(questions), "unique_images": len(images), "same_category": sum(bool(r["same_category"]) for r in family_rows), "cross_category": sum(not r["same_category"] for r in family_rows), "hashes": hashes, "no_vlm_performance_used": True}
    (out / "construction_stats.json").write_text(json.dumps(construction_stats, indent=2, sort_keys=True) + "\n")
    with db:
        for r in accepted:
            db.execute("UPDATE families SET status='FROZEN',split=? WHERE candidate_family_id=?", ("validation" if r["image_id"] in val_ids else "test", r["candidate_family_id"]))
            db.execute("UPDATE images SET status='FROZEN' WHERE image_id=? AND EXISTS(SELECT 1 FROM families WHERE image_id=? AND status='FROZEN') AND NOT EXISTS(SELECT 1 FROM families WHERE image_id=? AND status NOT IN ('FROZEN','HUMAN_REJECTED','AUTO_REJECTED'))", (r["image_id"], r["image_id"], r["image_id"]))
        event(db, None, None, "DATASET_FROZEN", construction_stats)
    db.close()
    return construction_stats
