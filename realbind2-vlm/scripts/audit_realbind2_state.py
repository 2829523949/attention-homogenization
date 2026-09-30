#!/usr/bin/env python3
"""Audit the current RealBind2 construction state without changing it.

This script deliberately reads SQLite in immutable mode.  AUTO_VALID records
are audited as a construction snapshot; they are never promoted to accepted or
frozen records here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy.realbind_v2_builder.question_generator import (  # noqa: E402
    DEPENDENCIES,
    QA_ORDER,
    generate_questions,
)

COLOR_VOCABULARY = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray",
}
AUDITED_STATES = ("AUTO_VALID", "HUMAN_PENDING", "HUMAN_ACCEPTED", "FROZEN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data_audit")
    args = parser.parse_args()

    source_root = args.construction_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    db_path = source_root / "state" / "construction_state.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    db = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in AUDITED_STATES)
    rows = db.execute(
        f"SELECT f.candidate_family_id,f.image_id,f.status,f.payload_json,"  # noqa: S608
        "i.accepted_path,i.sha256 AS image_sha256,i.width AS db_width,i.height AS db_height "
        "FROM families f JOIN images i USING(image_id) "
        f"WHERE f.status IN ({placeholders}) ORDER BY f.image_id,f.candidate_family_id",
        AUDITED_STATES,
    ).fetchall()
    family_status_counts = dict(db.execute("SELECT status,count(*) FROM families GROUP BY status"))
    image_status_counts = dict(db.execute("SELECT status,count(*) FROM images GROUP BY status"))
    freeze_events = db.execute(
        "SELECT COUNT(*) FROM events WHERE event='DATASET_FROZEN'"
    ).fetchone()[0]
    db.close()

    family_id_counts = Counter(str(row["candidate_family_id"]) for row in rows)
    families_by_image: dict[str, list[str]] = defaultdict(list)
    ordered_pair_counts: Counter[str] = Counter()
    unordered_pair_counts: Counter[str] = Counter()
    qa_id_counts: Counter[str] = Counter()
    qa_type_counts: Counter[str] = Counter()
    mapping_rows: list[dict] = []
    integrity_rows: list[dict] = []

    for row in rows:
        family = json.loads(row["payload_json"])
        family_id = str(row["candidate_family_id"])
        image_id = str(row["image_id"])
        family["family_id"] = family_id
        questions = generate_questions(family)
        qa_ids = [str(question["qa_id"]) for question in questions]
        qa_types = [str(question["qa_type"]) for question in questions]
        qa_id_counts.update(qa_ids)
        qa_type_counts.update(qa_types)
        families_by_image[image_id].append(family_id)

        object_a = str(family.get("object_A_id", ""))
        object_b = str(family.get("object_B_id", ""))
        ordered_pair_id = f"{image_id}:{object_a}:{object_b}"
        unordered_pair_id = f"{image_id}:{':'.join(sorted((object_a, object_b)))}"
        ordered_pair_counts[ordered_pair_id] += 1
        unordered_pair_counts[unordered_pair_id] += 1

        accepted_path = Path(str(row["accepted_path"])) if row["accepted_path"] else None
        image_exists = bool(accepted_path and accepted_path.is_file())
        image_decodable = False
        dimension_match = False
        actual_size = None
        image_error = None
        if image_exists and accepted_path is not None:
            try:
                with Image.open(accepted_path) as image:
                    image.load()
                    actual_size = [int(image.width), int(image.height)]
                    image_decodable = True
                    dimension_match = actual_size == [int(family["width"]), int(family["height"])]
            except Exception as exc:  # report source corruption without masking it
                image_error = f"{type(exc).__name__}: {exc}"

        expected_answers = {
            "I2C_A": COLOR_VOCABULARY,
            "I2C_B": COLOR_VOCABULARY,
            "PAIR_REL_A_TO_B": {"LEFT", "RIGHT"},
            "REL2I_LEFT": {"A", "B"},
            "REL2I_RIGHT": {"A", "B"},
            "REL2C_LEFT": COLOR_VOCABULARY,
            "REL2C_RIGHT": COLOR_VOCABULARY,
        }
        gt_valid = all(
            str(question["ground_truth"]) in expected_answers[str(question["qa_type"])]
            for question in questions
        )
        dependencies_complete = (
            list(QA_ORDER) == qa_types
            and set(DEPENDENCIES["atomic_prerequisites"]).issubset(qa_types)
            and all(
                set(prerequisites).issubset(qa_types)
                for prerequisites in DEPENDENCIES["binding_prerequisites"].values()
            )
        )
        referent_review_required = any(
            family.get(key) == "HUMAN_UNIQUENESS_CHECK_REQUIRED"
            for key in ("referent_status_A", "referent_status_B")
        )
        complete = (
            len(questions) == 7
            and len(set(qa_ids)) == 7
            and qa_types == list(QA_ORDER)
            and image_exists
            and image_decodable
            and dimension_match
            and gt_valid
            and dependencies_complete
        )
        integrity_rows.append({
            "family_id": family_id,
            "image_id": image_id,
            "status": str(row["status"]),
            "qa_count": len(questions),
            "qa_types_complete_and_ordered": qa_types == list(QA_ORDER),
            "qa_ids_unique_within_family": len(set(qa_ids)) == len(qa_ids),
            "ground_truth_valid": gt_valid,
            "dependencies_complete": dependencies_complete,
            "image_exists": image_exists,
            "image_decodable": image_decodable,
            "image_dimension_match": dimension_match,
            "actual_image_size": actual_size,
            "image_error": image_error,
            "referent_human_uniqueness_review_required": referent_review_required,
            "integrity_pass": complete,
        })
        mapping_rows.append({
            "image_id": image_id,
            "family_id": family_id,
            "family_status": str(row["status"]),
            "object_A_id": object_a,
            "object_B_id": object_b,
            "ordered_entity_pair_id": ordered_pair_id,
            "unordered_entity_pair_id": unordered_pair_id,
            "referent_A": family.get("referent_A"),
            "referent_B": family.get("referent_B"),
            "category_A": family.get("category_A"),
            "category_B": family.get("category_B"),
            "color_A": family.get("color_A"),
            "color_B": family.get("color_B"),
            "accepted_path": (
                str(accepted_path.relative_to(source_root))
                if accepted_path and accepted_path.is_relative_to(source_root)
                else ""
            ),
        })

    mapping_path = output_root / "family_image_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(mapping_rows[0]) if mapping_rows else [], lineterminator="\n"
        )
        if mapping_rows:
            writer.writeheader()
            writer.writerows(mapping_rows)

    integrity_path = output_root / "family_integrity.jsonl"
    with integrity_path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in integrity_rows:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

    families_per_image = Counter(len(family_ids) for family_ids in families_by_image.values())
    duplicate_family_ids = sorted(key for key, count in family_id_counts.items() if count != 1)
    duplicate_ordered_pairs = sorted(key for key, count in ordered_pair_counts.items() if count != 1)
    duplicate_unordered_pairs = sorted(key for key, count in unordered_pair_counts.items() if count != 1)
    duplicate_qa_ids = sorted(key for key, count in qa_id_counts.items() if count != 1)
    summary = {
        "audit_scope": list(AUDITED_STATES),
        "construction_database": str(db_path.relative_to(source_root)),
        "construction_database_sha256": sha256_file(db_path),
        "family_status_counts_all": family_status_counts,
        "image_status_counts_all": image_status_counts,
        "num_families": len(rows),
        "num_unique_images": len(families_by_image),
        "families_per_image_distribution": {
            str(key): families_per_image[key] for key in sorted(families_per_image)
        },
        "max_families_per_image": max(families_per_image, default=0),
        "duplicate_family_ids": duplicate_family_ids,
        "duplicate_ordered_entity_pair_ids": duplicate_ordered_pairs,
        "duplicate_unordered_entity_pair_ids": duplicate_unordered_pairs,
        "duplicate_qa_ids": duplicate_qa_ids,
        "num_integrity_pass": sum(bool(row["integrity_pass"]) for row in integrity_rows),
        "num_integrity_fail": sum(not bool(row["integrity_pass"]) for row in integrity_rows),
        "num_referent_human_review_required": sum(
            bool(row["referent_human_uniqueness_review_required"]) for row in integrity_rows
        ),
        "dataset_freeze_events": int(freeze_events),
        "realbind2_unit": "family",
    }
    write_json(output_root / "family_count_summary.json", summary)

    qa_summary = {
        "num_families": len(rows),
        "num_unique_images": len(families_by_image),
        "num_QA_total": sum(qa_type_counts.values()),
        "num_I2C_A": qa_type_counts["I2C_A"],
        "num_I2C_B": qa_type_counts["I2C_B"],
        "num_PAIR_REL": qa_type_counts["PAIR_REL_A_TO_B"],
        "num_REL2I_LEFT": qa_type_counts["REL2I_LEFT"],
        "num_REL2I_RIGHT": qa_type_counts["REL2I_RIGHT"],
        "num_REL2C_LEFT": qa_type_counts["REL2C_LEFT"],
        "num_REL2C_RIGHT": qa_type_counts["REL2C_RIGHT"],
        "num_composition": qa_type_counts["REL2C_LEFT"] + qa_type_counts["REL2C_RIGHT"],
        "all_families_exactly_seven_qa": all(row["qa_count"] == 7 for row in integrity_rows),
        "all_family_integrity_pass": all(row["integrity_pass"] for row in integrity_rows),
        "qa_id_uniqueness_pass": not duplicate_qa_ids,
        "source": "deterministically regenerated from current construction family payloads",
        "certification_scope": "AUTO_VALID_CONSTRUCTION_SNAPSHOT_NOT_FROZEN_RELEASE",
    }
    write_json(output_root / "qa_type_counts.json", qa_summary)
    print(json.dumps({"family_summary": summary, "qa_summary": qa_summary}, indent=2, sort_keys=True))
    return 0 if qa_summary["all_family_integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
