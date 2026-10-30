#!/usr/bin/env python3
"""Audit the 500-case cohort/package integrity and paired-prompt contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=500)
    parser.add_argument("--package-root", type=Path, default=None)
    args = parser.parse_args()

    cases_path = args.cases.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    rows = read_jsonl(cases_path)
    errors: list[str] = []
    checked_images = 0
    for row in rows:
        cid = str(row.get("case_id", ""))
        image_file = str(row.get("image_file", ""))
        if not cid:
            errors.append("empty_case_id")
        if not image_file.startswith("images/") or ".." in Path(image_file).parts:
            errors.append(f"unsafe_image_file:{cid}:{image_file}")
            continue
        image_path = images_root.parent / image_file if images_root.name == "images" else images_root / image_file
        if not image_path.is_file():
            errors.append(f"missing_image:{cid}:{image_path}")
            continue
        if sha256_file(image_path) != str(row.get("image_sha256", "")):
            errors.append(f"image_sha_mismatch:{cid}")
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                if image.size != (int(row["width"]), int(row["height"])):
                    errors.append(f"image_dimensions:{cid}:{image.size}!={(row['width'], row['height'])}")
                image.verify()
        except Exception as exc:
            errors.append(f"image_decode:{cid}:{type(exc).__name__}:{exc}")
        checked_images += 1

        facts = row.get("facts")
        if not isinstance(facts, list) or len(facts) != 3:
            errors.append(f"fact_count:{cid}")
            continue
        joined = "\n".join(str(fact) for fact in facts)
        if any(token in joined.casefold() for token in ("therefore", "the answer is", "the left one is", "the right one is")):
            errors.append(f"derived_answer_leak:{cid}")
        reconstructed = joined + "\n\n" + str(row.get("question", ""))
        if reconstructed != str(row.get("text_prompt", "")):
            errors.append(f"prompt_reconstruction:{cid}")
        actual_hash = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
        if actual_hash != str(row.get("text_prompt_sha256", "")):
            errors.append(f"text_prompt_hash:{cid}")
        contract = row.get("condition_contract") or {}
        facts_contract = contract.get("facts_only") or {}
        image_contract = contract.get("image_plus_facts") or {}
        if facts_contract.get("text_sha256") != actual_hash or image_contract.get("text_sha256") != actual_hash:
            errors.append(f"condition_text_hash:{cid}")
        if bool(facts_contract.get("image_present", True)) or not bool(image_contract.get("image_present", False)):
            errors.append(f"condition_image_flag:{cid}")
        if str(row.get("target_color", "")) != str(row.get("ground_truth", "")):
            errors.append(f"target_color_mismatch:{cid}")
        if str(row.get("target_identity", "")) not in {str(row.get("referent_A")), str(row.get("referent_B"))}:
            errors.append(f"target_identity:{cid}")
        if " ".join(str(row.get("referent_A", "")).casefold().split()) == " ".join(str(row.get("referent_B", "")).casefold().split()):
            errors.append(f"same_referent:{cid}")
        expected_relation = "left" if bool(row.get("left_is_A")) else "right"
        if str(row.get("relation_A_to_B")) != expected_relation:
            errors.append(f"relation:{cid}")

    case_ids = [str(row.get("case_id")) for row in rows]
    image_ids = [str(row.get("image_id")) for row in rows]
    if len(rows) != args.expected:
        errors.append(f"case_count:{len(rows)}!={args.expected}")
    if len(set(case_ids)) != len(case_ids):
        errors.append("duplicate_case_ids")
    if len(set(image_ids)) != len(image_ids):
        errors.append("duplicate_image_ids")
    side_counts = Counter(str(row.get("qa_type")) for row in rows)
    half = args.expected // 2
    if side_counts != {"REL2C_LEFT": half, "REL2C_RIGHT": args.expected - half}:
        errors.append(f"side_balance:{dict(side_counts)}")

    package_checks: dict[str, bool] = {}
    if args.package_root is not None:
        root = args.package_root.expanduser().resolve()
        required = [
            "README.md", "PROTOCOL.md", "cases.jsonl", "manifest.json", "selection_audit.json",
            "scripts/run_facts_image.py", "scripts/summarize_facts_image.py", "scripts/audit_package.py", "scripts/merge_results.py",
            "configs/qwen35_4b.yaml", "configs/qwen3vl_4b.yaml", "configs/internvl35_8b.yaml",
            "configs/llava_next_7b.yaml", "src/models/__init__.py", "src/common/scorer.py",
            "run_qwen35_4b.sh", "run_qwen3vl_4b.sh", "run_internvl35_8b.sh", "run_llava_next_7b.sh",
        ]
        for relative in required:
            package_checks[relative] = (root / relative).is_file()
        for relative, present in package_checks.items():
            if not present:
                errors.append(f"package_missing:{relative}")

    audit = {
        "schema_version": "realbind2_facts_image_500_package_audit_v1",
        "passed": not errors,
        "cases": len(rows),
        "unique_images": len(set(image_ids)),
        "checked_images": checked_images,
        "side_counts": dict(sorted(side_counts.items())),
        "split_counts": dict(sorted(Counter(str(row.get("split")) for row in rows).items())),
        "same_category_count": sum(bool(row.get("same_category")) for row in rows),
        "cases_sha256": sha256_file(cases_path),
        "package_checks": package_checks,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
