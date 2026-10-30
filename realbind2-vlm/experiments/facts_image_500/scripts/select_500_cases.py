#!/usr/bin/env python3
"""Select a deterministic 500-image facts-only vs image-plus-facts cohort.

The selection is made from the frozen RealBind2 release.  By default it uses
the test split, excludes families whose two referents are textually identical,
selects at most one family per source image, and balances the two REL2C target
sides (250 LEFT / 250 RIGHT).  Model predictions are never used.

Outputs are self-contained: each case carries the exact fact text and question
needed by the runner, so downstream VLM jobs do not need the frozen dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("/mnt/d/datasets/RealBind_v2")
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "facts_image_500" / "output" / "cohort_v1"
SELECTION_SEED = "realbind2-facts-image-500-v1-20261030"
CASE_SCHEMA = "realbind2_facts_image_500_cases_v1"
MANIFEST_SCHEMA = "realbind2_facts_image_500_manifest_v1"
EXPECTED_DEFAULT_CASES = 500
COMPOSITION_TYPES = ("REL2C_LEFT", "REL2C_RIGHT")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_phrase(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def strip_leading_article(value: Any) -> str:
    text = " ".join(str(value).strip().split())
    if text.casefold().startswith("the "):
        text = text[4:]
    if not text:
        raise ValueError("empty referent after stripping article")
    return text


def hash_rank(seed: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def target_fields(family: Mapping[str, Any], qa: Mapping[str, Any]) -> dict[str, Any]:
    side = "left" if str(qa["qa_type"]) == "REL2C_LEFT" else "right"
    left_is_a = bool(family["geometry_metrics"]["left_is_A"])
    target_label = ("A" if left_is_a else "B") if side == "left" else ("B" if left_is_a else "A")
    other_label = "B" if target_label == "A" else "A"
    return {
        "target_side": side,
        "target_identity_label": target_label,
        "target_identity": str(family[f"referent_{target_label}"]),
        "target_color": str(family[f"color_{target_label}"]),
        "competitor_identity_label": other_label,
        "competitor_identity": str(family[f"referent_{other_label}"]),
        "competitor_color": str(family[f"color_{other_label}"]),
    }


def build_case(
    *,
    rank: int,
    family: Mapping[str, Any],
    qa: Mapping[str, Any],
    image_path: Path,
    selection_seed: str,
) -> dict[str, Any]:
    phrase_a = strip_leading_article(family["referent_A"])
    phrase_b = strip_leading_article(family["referent_B"])
    left_is_a = bool(family["geometry_metrics"]["left_is_A"])
    relation = "left" if left_is_a else "right"
    facts = [
        f"The {phrase_a} is {family['color_A']}.",
        f"The {phrase_b} is {family['color_B']}.",
        f"The {phrase_a} is to the {relation} of the {phrase_b}.",
    ]
    text_prompt = "\n".join(facts) + "\n\n" + str(qa["question"])
    text_hash = sha256_bytes(text_prompt.encode("utf-8"))
    target = target_fields(family, qa)
    if target["target_color"] != str(qa["ground_truth"]):
        raise ValueError(
            f"target color mismatch for {qa['qa_id']}: family={target['target_color']} qa={qa['ground_truth']}"
        )
    forbidden = ("therefore", "the answer is", "the left one is", "the right one is", "the left object is", "the right object is")
    if any(token in "\n".join(facts).casefold() for token in forbidden):
        raise ValueError(f"derived answer leaked into facts for {qa['qa_id']}")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_sha = sha256_file(image_path)
    return {
        "schema_version": CASE_SCHEMA,
        "case_id": f"RB2_FI500_{rank:04d}",
        "selection_rank": rank,
        "selection_seed": selection_seed,
        "family_id": str(family["family_id"]),
        "image_id": str(family["image_id"]),
        "image_file": f"images/{family['image_id']}.jpg",
        "image_sha256": image_sha,
        "source_image_path": str(image_path),
        "split": str(family["split"]),
        "qa_id": str(qa["qa_id"]),
        "qa_type": str(qa["qa_type"]),
        "question": str(qa["question"]),
        "answer_type": str(qa["answer_type"]),
        "ground_truth": str(qa["ground_truth"]),
        **target,
        "referent_A": str(family["referent_A"]),
        "referent_B": str(family["referent_B"]),
        "object_phrase_A": phrase_a,
        "object_phrase_B": phrase_b,
        "color_A": str(family["color_A"]),
        "color_B": str(family["color_B"]),
        "left_is_A": left_is_a,
        "relation_A_to_B": relation,
        "same_category": bool(family["same_category"]),
        "geometry_metrics": dict(family["geometry_metrics"]),
        "bbox_A": list(family["bbox_A"]),
        "bbox_B": list(family["bbox_B"]),
        "width": int(family["width"]),
        "height": int(family["height"]),
        "facts": facts,
        "text_prompt": text_prompt,
        "text_prompt_sha256": text_hash,
        "condition_contract": {
            "facts_only": {"image_present": False, "text_sha256": text_hash},
            "image_plus_facts": {"image_present": True, "text_sha256": text_hash},
            "only_difference": "presence of the original full image content block",
        },
    }


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=EXPECTED_DEFAULT_CASES)
    parser.add_argument("--split", choices=("test", "validation", "all"), default="test")
    parser.add_argument("--seed", default=SELECTION_SEED)
    parser.add_argument("--keep-same-referent", action="store_true", help="disable the default exclusion of families whose two referent strings are identical")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    frozen = data_root / "frozen"
    families_path = frozen / "families.jsonl"
    questions_path = frozen / "questions.jsonl"
    splits_path = frozen / "splits.json"
    stats_path = frozen / "construction_stats.json"
    for path in (families_path, questions_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    families = read_jsonl(families_path)
    questions = read_jsonl(questions_path)
    by_family_q = defaultdict(dict)
    for qa in questions:
        by_family_q[str(qa["family_id"])][str(qa["qa_type"])] = qa

    eligible: list[dict[str, Any]] = []
    excluded_same_referent = 0
    excluded_split = 0
    excluded_no_composition = 0
    for family in families:
        if args.split != "all" and str(family["split"]) != args.split:
            excluded_split += 1
            continue
        if not args.keep_same_referent and normalized_phrase(family["referent_A"]) == normalized_phrase(family["referent_B"]):
            excluded_same_referent += 1
            continue
        typed = by_family_q.get(str(family["family_id"]), {})
        if not all(qa_type in typed for qa_type in COMPOSITION_TYPES):
            excluded_no_composition += 1
            continue
        eligible.append(family)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in eligible:
        by_image[str(family["image_id"])].append(family)
    image_ids = sorted(by_image, key=lambda image_id: (hash_rank(args.seed, "image", image_id), int(image_id)))
    if len(image_ids) < args.count:
        raise ValueError(f"only {len(image_ids)} eligible images, requested {args.count}")
    selected_images = image_ids[: args.count]

    selected_cases: list[dict[str, Any]] = []
    for index, image_id in enumerate(selected_images):
        # Deterministically choose one family for images that have two frozen families.
        family = min(
            by_image[image_id],
            key=lambda row: (hash_rank(args.seed, "family", str(row["family_id"])), str(row["family_id"])),
        )
        # Exact 50/50 side balance independent of model behaviour.
        qa_type = "REL2C_LEFT" if index % 2 == 0 else "REL2C_RIGHT"
        qa = by_family_q[str(family["family_id"])][qa_type]
        image_path = frozen / str(family["image_path"])
        selected_cases.append(
            build_case(
                rank=index + 1,
                family=family,
                qa=qa,
                image_path=image_path,
                selection_seed=args.seed,
            )
        )

    selected_cases.sort(key=lambda row: int(row["selection_rank"]))
    image_ids_selected = [str(case["image_id"]) for case in selected_cases]
    if len(selected_cases) != args.count or len(set(image_ids_selected)) != args.count:
        raise AssertionError("selection is not one case per unique image")
    if Counter(case["qa_type"] for case in selected_cases) != {"REL2C_LEFT": args.count // 2, "REL2C_RIGHT": args.count - args.count // 2}:
        raise AssertionError("side balance failed")
    if any(not (frozen / str(case.get("source_image_path", ""))).exists() for case in []):
        pass
    if any(not Path(str(case["source_image_path"])).is_file() for case in selected_cases):
        raise AssertionError("selected image missing")

    source_hashes = {path.name: sha256_file(path) for path in (families_path, questions_path, stats_path)}
    if splits_path.is_file():
        source_hashes[splits_path.name] = sha256_file(splits_path)

    # Previous v1.1 pilot is a different frozen benchmark; record its overlap explicitly.
    previous_pilot = Path("/mnt/d/datasets/RealBind/realbind/experiments/qwen35_visual_interference_pilot_v1/selected_cases.jsonl")
    previous_images: set[str] = set()
    if previous_pilot.is_file():
        previous_images = {str(row["image_id"]) for row in read_jsonl(previous_pilot)}
    overlap_previous = sorted(set(image_ids_selected) & previous_images, key=int)

    audit = {
        "schema_version": "realbind2_facts_image_500_selection_audit_v1",
        "passed": True,
        "selection_seed": args.seed,
        "requested_cases": args.count,
        "selected_cases": len(selected_cases),
        "selected_unique_images": len(set(image_ids_selected)),
        "dataset": {
            "data_root": str(data_root),
            "families": len(families),
            "questions": len(questions),
            "eligible_families": len(eligible),
            "eligible_unique_images": len(image_ids),
            "source_hashes": source_hashes,
        },
        "exclusions": {
            "split_mismatch": excluded_split,
            "same_referent": excluded_same_referent,
            "missing_composition_qa": excluded_no_composition,
            "rule": "exclude families whose normalized referent_A and referent_B strings are identical unless --keep-same-referent is passed",
        },
        "counts": {
            "qa_type": distribution(selected_cases, "qa_type"),
            "split": distribution(selected_cases, "split"),
            "left_is_A": {str(key): value for key, value in sorted(Counter(bool(case["left_is_A"]) for case in selected_cases).items())},
            "same_category": {str(key): value for key, value in sorted(Counter(bool(case["same_category"]) for case in selected_cases).items())},
            "target_color": distribution(selected_cases, "target_color"),
            "families_on_source_image": {str(key): value for key, value in sorted(Counter(len(by_image[str(case["image_id"])]) for case in selected_cases).items())},
            "qa_type_x_left_is_A": {
                f"{qa_type}|left_is_A={left_is_a}": value
                for (qa_type, left_is_a), value in sorted(Counter((case["qa_type"], bool(case["left_is_A"])) for case in selected_cases).items())
            },
        },
        "previous_v1_1_pilot_overlap_images": overlap_previous,
        "previous_v1_1_pilot_overlap_count": len(overlap_previous),
        "checks": {
            "one_case_per_image": len(set(image_ids_selected)) == len(selected_cases),
            "exact_side_balance": Counter(case["qa_type"] for case in selected_cases) == {"REL2C_LEFT": args.count // 2, "REL2C_RIGHT": args.count - args.count // 2},
            "all_selected_images_exist": all(Path(str(case["source_image_path"])).is_file() for case in selected_cases),
            "no_same_referent_cases": all(normalized_phrase(case["referent_A"]) != normalized_phrase(case["referent_B"]) for case in selected_cases),
            "all_target_colors_match_qa_ground_truth": all(case["target_color"] == case["ground_truth"] for case in selected_cases),
            "all_text_prompt_hashes_recorded": all(len(case["text_prompt_sha256"]) == 64 for case in selected_cases),
        },
    }
    if not all(audit["checks"].values()):
        audit["passed"] = False
        raise AssertionError(f"selection audit failed: {audit['checks']}")
    if overlap_previous:
        audit["warning"] = "Selected images overlap the prior v1.1 pilot; this should not occur for a frozen RealBind2 release."

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol": "realbind2_facts_only_vs_image_plus_facts",
        "dataset": "RealBind2",
        "dataset_version": "RealBind-v2.0",
        "selection_seed": args.seed,
        "split_policy": args.split,
        "case_count": len(selected_cases),
        "unique_image_count": len(set(image_ids_selected)),
        "conditions": ["facts_only", "image_plus_facts"],
        "paired_text_contract": "facts_only.text_prompt_sha256 == image_plus_facts.text_prompt_sha256; only the image content block differs",
        "facts_only_image_present": False,
        "image_plus_facts_image_present": True,
        "image_plus_facts_source": "original full image, no crop, bbox, mark, arrow, or label",
        "model_ids": ["Qwen3.5-4B", "Qwen3-VL-4B", "Qwen3-VL-8B", "InternVL3.5-8B", "LLaVA-NeXT-7B"],
        "source_hashes": source_hashes,
        "cases_sha256": sha256_file(out / "cases.jsonl") if (out / "cases.jsonl").is_file() else None,
        "selection_audit_sha256": None,
    }

    # Write JSONL first, then hash it, then write audit and final manifest.
    out.mkdir(parents=True, exist_ok=True)
    cases_path = out / "cases.jsonl"
    write_jsonl(cases_path, selected_cases)
    audit["cases_sha256"] = sha256_file(cases_path)
    write_json(out / "selection_audit.json", audit)
    manifest["cases_sha256"] = sha256_file(cases_path)
    manifest["selection_audit_sha256"] = sha256_file(out / "selection_audit.json")

    csv_fields = [
        "case_id", "family_id", "image_id", "image_file", "split", "qa_id", "qa_type", "target_side",
        "target_identity", "target_color", "competitor_identity", "competitor_color", "left_is_A",
        "same_category", "text_prompt_sha256",
    ]
    write_csv(out / "selected_images.csv", selected_cases, csv_fields)
    write_json(out / "manifest.json", manifest)

    report = [
        "# RealBind2 facts-only vs image-plus-facts 500-image cohort",
        "",
        f"- Selected cases/images: **{len(selected_cases)} / {len(set(image_ids_selected))}**",
        f"- Dataset split: `{args.split}`",
        f"- Selection seed: `{args.seed}`",
        f"- Target-side balance: `{dict(Counter(case['qa_type'] for case in selected_cases))}`",
        f"- Left-is-A balance: `{dict(Counter(bool(case['left_is_A']) for case in selected_cases))}`",
        f"- Same-category cases: `{sum(bool(case['same_category']) for case in selected_cases)}`",
        f"- Previous v1.1 pilot image overlap: `{len(overlap_previous)}`",
        "",
        "## Selection rule",
        "",
        f"1. Start from the frozen RealBind2 release and use split `{args.split}`.",
        "2. Exclude families whose two referent strings are identical after whitespace/case normalization.",
        "3. Group eligible families by source image and choose at most one family per image.",
        "4. Rank images by SHA256 of `seed:image:<image_id>` and take the first 500.",
        "5. Balance `REL2C_LEFT` / `REL2C_RIGHT` as 250 / 250 by selection rank.",
        "6. No model prediction is used in selection.",
        "",
        "## Condition contract",
        "",
        "Both conditions receive byte-identical fact text and question text. `facts_only` has no image;",
        "`image_plus_facts` prepends the original full image as the only additional content block.",
        "No crop, bbox, mark, arrow, or object label is added.",
        "",
        "## Files",
        "",
        "- `cases.jsonl`: one self-contained case per image (facts, prompt, hash, metadata).",
        "- `manifest.json`: dataset/protocol hashes and condition contract.",
        "- `selection_audit.json`: exclusion, balance, and integrity audit.",
        "- `selected_images.csv`: compact case/image index.",
        "",
    ]
    (out / "SELECTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "cases": len(selected_cases),
        "unique_images": len(set(image_ids_selected)),
        "output_dir": str(out),
        "cases_sha256": manifest["cases_sha256"],
        "qa_type_counts": dict(Counter(case["qa_type"] for case in selected_cases)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
