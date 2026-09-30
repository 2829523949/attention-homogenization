#!/usr/bin/env python3
"""Unified training-free Qwen3.5-4B RealBind DaH oracle-bbox evaluator.

This runner is intentionally a single evaluator for the five preregistered
variants:

    Original                  full image -> frozen REL2C question
    Blur-only                 pair-defocus image -> frozen REL2C question
    Decompose-only            full image -> REL2I -> fresh full image + I2C
    Handoff+TargetLock        full image -> REL2I -> fresh target-lock + I2C
    Full DaH                  pair-defocus -> REL2I -> fresh target-lock + I2C

The frozen RealBind pair-direct v1.1 files, the project Qwen loader/renderer,
and the project's deterministic semantic scorer are read-only dependencies.
No model state, attention, prompt, training, grounding, or test-time search is
introduced here.  Family GT bboxes are used only to construct the three
deterministic diagnostic image variants; the target-lock image is selected
from the Stage-1 A/B prediction.

The script is deliberately conservative about resumability.  Raw generation
checkpoints are written atomically after every successful call.  Existing
complete raw files are never replaced; incompatible or duplicate checkpoints
raise ``EXPERIMENT_INVALID``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "experiments" / "realbind_phase2" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CANONICAL_COLORS, canonical_json, read_json, read_json_records, sha256_file  # noqa: E402
from generate_pair_direct_qa_v1_1 import dependency_fields  # noqa: E402
from experiments.realbind_phase2 import pair_direct_v1_1_baseline as frozen_scorer  # noqa: E402


REALBIND_ROOT = Path(
    os.environ.get("REALBIND_V1_ROOT", ROOT / "external_data" / "RealBind" / "realbind")
)
DEFAULT_FAMILIES = REALBIND_ROOT / "families" / "dryrun_100_pair_direct_v1_1_frozen.jsonl"
DEFAULT_QA = REALBIND_ROOT / "qa" / "dryrun_100_pair_direct_v1_1.jsonl"
DEFAULT_DEPENDENCY = REALBIND_ROOT / "qa" / "pair_direct_v1_1_dependency_spec.json"
DEFAULT_HISTORICAL = REALBIND_ROOT / "experiments" / "pair_direct_v1_1_qwen35_4b_baseline"
DEFAULT_VISUAL_PILOT = REALBIND_ROOT / "experiments" / "qwen35_visual_interference_pilot_v1"
DEFAULT_OUTPUT = REALBIND_ROOT / "experiments" / "qwen35_dah_oracle_v1"
MODEL_DIR = ROOT / "models" / "Qwen3.5-4B"
MODEL_IDENTIFIER = "Qwen/Qwen3.5-4B"
SCORER_FILE = ROOT / "experiments" / "realbind_phase2" / "pair_direct_v1_1_baseline.py"

EXPECTED_FAMILIES = 100
EXPECTED_CASES = 200
EXPECTED_PER_COMPOSITION = 100
COMPOSITION_TYPES = ("REL2C_LEFT", "REL2C_RIGHT")
REL2I_TYPES = ("REL2I_LEFT", "REL2I_RIGHT")
REQUIRED_QA_TYPES = (
    "I2C_A",
    "I2C_B",
    "PAIR_REL_A_TO_B",
    "REL2I_LEFT",
    "REL2I_RIGHT",
    "REL2C_LEFT",
    "REL2C_RIGHT",
)
VARIANTS = (
    "original",
    "blur_only",
    "decompose_only",
    "handoff_targetlock",
    "full_dah",
)
VARIANT_LABELS = {
    "original": "Original",
    "blur_only": "Blur-only",
    "decompose_only": "Decompose-only",
    "handoff_targetlock": "Handoff+TargetLock",
    "full_dah": "Full DaH",
}
GENERATION_CONFIG = {
    "dtype": "bfloat16",
    "batch_size": 1,
    "do_sample": False,
    "use_cache": False,
    "max_new_tokens": 16,
    "direct": True,
    "enable_thinking": False,
}
TRANSFORM_CONFIG = {
    "bbox_expand_frac": 0.10,
    "blur_radius_fraction_of_min_dimension": 0.025,
    "feather_radius_fraction_of_min_dimension": 0.010,
    "blur_radius_unit": "pixels after reading original image resolution",
    "mask": "soft grayscale mask; union by ImageChops.lighter; Gaussian feather",
    "output_format": "PNG",
    "resize": "none in transform; normal processor behavior only",
    "crop": False,
    "draw_annotations": False,
}
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
SCORING_REVISION = "realbind_pair_direct_v1_1_semantic_vocab_v2"
EXPERIMENT_SCHEMA = "realbind_qwen35_dah_oracle_v1"


class ExperimentInvalid(RuntimeError):
    """A frozen-protocol or implementation gate failed."""

    def __init__(self, message: str, failures: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.failures = list(failures or [])


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=json_default)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ExperimentInvalid(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise ExperimentInvalid(f"refusing to replace existing raw output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False, "sha256": None, "size_bytes": None}
    return {"path": str(path.resolve()), "exists": True, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def git_state() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=30, check=False)
            return result.stdout.strip()
        except Exception as exc:  # pragma: no cover
            return repr(exc)

    return {"commit": run(["git", "rev-parse", "HEAD"]), "status_short": run(["git", "status", "--short"])}


def output_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "provenance": root / "provenance.json",
        "config": root / "config.json",
        "data_alignment": root / "data_alignment_audit.json",
        "bbox_audit": root / "bbox_audit.json",
        "transform_audit": root / "image_transform_audit.json",
        "visual_html": root / "visual_audit.html",
        "cache": root / "call_cache.jsonl",
        "original": root / "original_results.jsonl",
        "blur_only": root / "blur_only_results.jsonl",
        "shared_original_rel2i": root / "shared_original_rel2i_stage1.jsonl",
        "full_dah_rel2i": root / "full_dah_pair_rel2i_stage1.jsonl",
        "decompose_only": root / "decompose_only_results.jsonl",
        "handoff_targetlock": root / "handoff_targetlock_results.jsonl",
        "full_dah": root / "full_dah_results.jsonl",
        "comparison": root / "all_case_comparison.jsonl",
        "summary_json": root / "summary_metrics.json",
        "summary_md": root / "summary_metrics.md",
        "paired_stats": root / "paired_statistics.json",
        "failure_breakdown": root / "pipeline_failure_breakdown.csv",
        "strong_path": root / "strong_path_repair.csv",
        "visual_repair": root / "visual_interference_repair.csv",
        "invalid": root / "EXPERIMENT_INVALID.json",
        "model_manifest": root / "model_hash_manifest.json",
    }


def ensure_dirs(paths: Mapping[str, Path]) -> None:
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pair_defocus", "target_lock_A", "target_lock_B"):
        (root / "derived_images" / name).mkdir(parents=True, exist_ok=True)


def required_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "families": args.families.expanduser().resolve(),
        "qa": args.qa.expanduser().resolve(),
        "dependency": args.dependency.expanduser().resolve(),
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExperimentInvalid(f"missing required file: {path}")
    return read_json_records(path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def norm(value: Any) -> str:
    return " ".join(str(value).strip().split())


def family_maps(families: Sequence[Mapping[str, Any]], qa_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    families_by_id = {str(row["family_id"]): dict(row) for row in families}
    typed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in qa_rows:
        typed[str(row["family_id"])][str(row["qa_type"])] = dict(row)
    return families_by_id, dict(typed)


def bbox_from_family(family: Mapping[str, Any], side: str) -> list[float]:
    target = family.get(side)
    if not isinstance(target, Mapping) or not isinstance(target.get("bbox"), list):
        raise ExperimentInvalid(f"{family.get('family_id')}: {side}.bbox is missing")
    return [float(value) for value in target["bbox"]]


def validate_frozen_schema(
    paths: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    families = read_records(paths["families"])
    qa_rows = read_records(paths["qa"])
    dependency = read_json(paths["dependency"])
    failures: list[str] = []
    family_ids = [str(row.get("family_id", "")) for row in families]
    if len(families) != EXPECTED_FAMILIES or len(set(family_ids)) != EXPECTED_FAMILIES:
        failures.append(f"families={len(families)} unique={len(set(family_ids))}; expected {EXPECTED_FAMILIES}")
    qa_ids = [str(row.get("qa_id", "")) for row in qa_rows]
    if len(qa_rows) != EXPECTED_CASES * 3 + EXPECTED_FAMILIES:
        failures.append(f"QA count={len(qa_rows)}; expected 700")
    if len(set(qa_ids)) != len(qa_ids):
        failures.append("QA IDs are not unique")
    by_family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    type_counts = Counter()
    for row in qa_rows:
        fid = str(row.get("family_id", ""))
        qa_type = str(row.get("qa_type", ""))
        by_family[fid][qa_type] = dict(row)
        type_counts[qa_type] += 1
        if row.get("schema_version") != "realbind_pair_direct_v1_1":
            failures.append(f"{row.get('qa_id')}: schema version mismatch")
        if not str(row.get("question", "")).strip():
            failures.append(f"{row.get('qa_id')}: empty question")
    for qa_type in REQUIRED_QA_TYPES:
        if type_counts[qa_type] != EXPECTED_FAMILIES:
            failures.append(f"{qa_type} count={type_counts[qa_type]}; expected {EXPECTED_FAMILIES}")
    unknown = sorted(set(type_counts) - set(REQUIRED_QA_TYPES))
    if unknown:
        failures.append(f"unknown QA types: {unknown}")

    actual_image_sizes: dict[str, list[int]] = {}
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ExperimentInvalid(f"Pillow is required for bbox/image audit: {exc}") from exc
    for family in families:
        fid = str(family.get("family_id", ""))
        typed = by_family.get(fid, {})
        if set(typed) != set(REQUIRED_QA_TYPES):
            failures.append(f"{fid}: required QA type set mismatch: {sorted(typed)}")
        image_path = Path(str(family.get("image_path", ""))).expanduser()
        if not image_path.is_file():
            failures.append(f"{fid}: missing image {image_path}")
            continue
        try:
            with Image.open(image_path) as image:
                actual_size = [int(image.width), int(image.height)]
        except Exception as exc:
            failures.append(f"{fid}: image cannot be opened: {exc}")
            continue
        actual_image_sizes[fid] = actual_size
        if actual_size != [int(family.get("image_width", -1)), int(family.get("image_height", -1))]:
            failures.append(f"{fid}: family image dimensions {family.get('image_width')}x{family.get('image_height')} != actual {actual_size}")
        for side in ("A", "B"):
            target = family.get(side)
            if not isinstance(target, Mapping) or not str(target.get("object", "")).strip() or not str(target.get("instance_id", "")).strip():
                failures.append(f"{fid}: {side} identity/instance_id missing")
                continue
            try:
                bbox = bbox_from_family(family, side)
            except ExperimentInvalid as exc:
                failures.append(str(exc))
                continue
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < 0 or bbox[1] < 0:
                failures.append(f"{fid}: {side}.bbox invalid {bbox}")
            if bbox[0] + bbox[2] > actual_size[0] + 1e-6 or bbox[1] + bbox[3] > actual_size[1] + 1e-6:
                failures.append(f"{fid}: {side}.bbox exceeds image {bbox} vs {actual_size}")
        if str(family.get("A", {}).get("instance_id")) == str(family.get("B", {}).get("instance_id")):
            failures.append(f"{fid}: A/B instance IDs collide")
        if not isinstance(family.get("left_is_A"), bool):
            failures.append(f"{fid}: left_is_A is not bool")
        for qa_type, qa in typed.items():
            if str(qa.get("image_path")) != str(family.get("image_path")) or str(qa.get("image_id")) != str(family.get("image_id")):
                failures.append(f"{fid}/{qa_type}: QA image provenance mismatch")
        for comp_type in COMPOSITION_TYPES:
            qa = typed.get(comp_type)
            if qa is None:
                continue
            expected = dependency_fields(family, comp_type)
            for field in ("minimum_required_qa_ids", "minimum_required_qa_types", "strict_required_qa_ids", "strict_required_qa_types"):
                if qa.get(field) != expected[field]:
                    failures.append(f"{fid}/{comp_type}: dependency field {field} differs")
            target_side = "A" if comp_type == "REL2C_LEFT" and bool(family.get("left_is_A")) else None
            if comp_type == "REL2C_LEFT":
                target_side = "A" if bool(family.get("left_is_A")) else "B"
            else:
                target_side = "B" if bool(family.get("left_is_A")) else "A"
            if qa.get("target_side") != ("left" if comp_type == "REL2C_LEFT" else "right"):
                failures.append(f"{fid}/{comp_type}: frozen target_side field mismatch")
            if qa.get("ground_truth") != family[target_side].get("canonical_color"):
                failures.append(f"{fid}/{comp_type}: composition ground truth mismatch")
        for side, i2c_type in (("A", "I2C_A"), ("B", "I2C_B")):
            qa = typed.get(i2c_type)
            if qa is not None and qa.get("ground_truth") != family[side].get("canonical_color"):
                failures.append(f"{fid}/{i2c_type}: I2C ground truth mismatch")

    if dependency.get("schema_version") != "realbind_pair_direct_v1_1_dependency_v1":
        failures.append(f"dependency schema mismatch: {dependency.get('schema_version')}")
    if dependency.get("protocol_schema_version") != "realbind_pair_direct_v1_1":
        failures.append(f"dependency protocol mismatch: {dependency.get('protocol_schema_version')}")
    dependency_rows = dependency.get("rules_by_family", [])
    if not isinstance(dependency_rows, list) or len(dependency_rows) != EXPECTED_FAMILIES:
        failures.append("dependency rules_by_family count mismatch")
    dep_by_id = {str(row.get("family_id")): row for row in dependency_rows if isinstance(row, Mapping)}
    if set(dep_by_id) != set(family_ids):
        failures.append("dependency family IDs differ from frozen family IDs")
    for fid, family in {str(f["family_id"]): f for f in families}.items():
        dep = dep_by_id.get(fid, {})
        for comp_type in COMPOSITION_TYPES:
            expected = dependency_fields(family, comp_type)
            observed = dep.get("composition", {}).get(comp_type, {}) if isinstance(dep, Mapping) else {}
            if observed != expected:
                failures.append(f"dependency {fid}/{comp_type} differs from frozen generator fields")
    if failures:
        raise ExperimentInvalid("FROZEN_SCHEMA_AUDIT_FAILED", failures)
    input_hashes = {name: file_record(paths[name]) for name in ("families", "qa", "dependency")}
    audit = {
        "schema_version": f"{EXPERIMENT_SCHEMA}_data_alignment_v1",
        "passed": True,
        "families": len(families),
        "qa": len(qa_rows),
        "composition_cases": EXPECTED_CASES,
        "composition_by_type": {qa_type: type_counts[qa_type] for qa_type in COMPOSITION_TYPES},
        "qa_type_counts": dict(sorted(type_counts.items())),
        "required_frozen_files": input_hashes,
        "family_schema": {
            "image_path": "family.image_path",
            "image_dimensions": ["family.image_width", "family.image_height"],
            "identity": "family.A.object / family.B.object",
            "instance_id": "family.A.instance_id / family.B.instance_id",
            "bbox": "family.A.bbox / family.B.bbox",
            "bbox_convention": "[x, y, width, height]",
            "left_is_A": "family.left_is_A",
        },
        "actual_image_sizes": actual_image_sizes,
        "dependency_links": {
            "REL2C_LEFT": "frozen REL2I_LEFT is used for Stage 1; target side is family left object",
            "REL2C_RIGHT": "frozen REL2I_RIGHT is used for Stage 1; target side is family right object",
            "Stage2": "selected Stage-1 A/B maps only to the same-family frozen I2C_A/I2C_B question",
        },
        "gt_usage_gate": {
            "allowed": ["construct pair-defocus from A∪B bbox", "construct target_lock_A and target_lock_B image cache", "score against frozen GT offline"],
            "forbidden": ["choose target identity", "correct Stage-1 prediction", "select target lock at runtime"],
            "runtime_target_lock_selector": "Stage-1 normalized prediction mapped to A or B; no GT fallback",
        },
    }
    return families, qa_rows, dependency, audit


def build_semantic_vocabulary(families: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    identities = sorted({str(family[side]["object"]).strip() for family in families for side in ("A", "B")})
    return {"color": list(CANONICAL_COLORS), "position": ["left", "right"], "identity": identities}


def build_cases(families: Sequence[Mapping[str, Any]], qa_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    families_by_id, typed = family_maps(families, qa_rows)
    cases: list[dict[str, Any]] = []
    for family in families:
        fid = str(family["family_id"])
        for comp_type in COMPOSITION_TYPES:
            composition = typed[fid][comp_type]
            target_side = "A" if (comp_type == "REL2C_LEFT" and bool(family["left_is_A"])) else "B"
            if comp_type == "REL2C_RIGHT":
                target_side = "B" if bool(family["left_is_A"]) else "A"
            rel2i_type = "REL2I_LEFT" if comp_type == "REL2C_LEFT" else "REL2I_RIGHT"
            rel2i = typed[fid][rel2i_type]
            cases.append(
                {
                    "case_id": str(composition["qa_id"]),
                    "family_id": fid,
                    "image_id": str(family["image_id"]),
                    "image_path": str(family["image_path"]),
                    "composition_qa_type": comp_type,
                    "composition_qa_id": str(composition["qa_id"]),
                    "composition_question": str(composition["question"]),
                    "composition_ground_truth": str(composition["ground_truth"]),
                    "target_side": target_side,
                    "target_identity": str(family[target_side]["object"]),
                    "target_color": str(family[target_side]["canonical_color"]),
                    "rel2i_type": rel2i_type,
                    "rel2i_qa_id": str(rel2i["qa_id"]),
                    "rel2i_question": str(rel2i["question"]),
                    "rel2i_ground_truth": str(rel2i["ground_truth"]),
                    "i2c_A_qa_id": str(typed[fid]["I2C_A"]["qa_id"]),
                    "i2c_A_question": str(typed[fid]["I2C_A"]["question"]),
                    "i2c_B_qa_id": str(typed[fid]["I2C_B"]["qa_id"]),
                    "i2c_B_question": str(typed[fid]["I2C_B"]["question"]),
                    "object_A": str(family["A"]["object"]),
                    "object_B": str(family["B"]["object"]),
                    "color_A": str(family["A"]["canonical_color"]),
                    "color_B": str(family["B"]["canonical_color"]),
                    "left_is_A": bool(family["left_is_A"]),
                }
            )
    if len(cases) != EXPECTED_CASES or Counter(str(row["composition_qa_type"]) for row in cases) != Counter({"REL2C_LEFT": 100, "REL2C_RIGHT": 100}):
        raise ExperimentInvalid("composition case construction count/type gate failed")
    return cases


def model_file_records(model_dir: Path, cache_path: Path | None = None) -> tuple[list[dict[str, Any]], bool]:
    if not model_dir.is_dir():
        raise ExperimentInvalid(f"model directory is missing: {model_dir}")
    paths = sorted(path for path in model_dir.iterdir() if path.is_file())
    if not paths:
        raise ExperimentInvalid(f"model directory is empty: {model_dir}")
    if cache_path is not None and cache_path.is_file():
        try:
            old = read_json(cache_path)
            records = old.get("files", [])
            if old.get("model_path") == str(model_dir.resolve()) and isinstance(records, list):
                current_stat = {path.name: (int(path.stat().st_size), int(path.stat().st_mtime_ns)) for path in paths}
                old_stat = {str(row.get("relative_path")): (int(row.get("size_bytes", -1)), int(row.get("mtime_ns", -2))) for row in records if isinstance(row, Mapping)}
                if current_stat == old_stat and all(row.get("sha256") for row in records):
                    return [dict(row) for row in records], True
        except Exception:
            pass
    records: list[dict[str, Any]] = []
    for path in paths:
        records.append({"relative_path": path.name, "size_bytes": int(path.stat().st_size), "mtime_ns": int(path.stat().st_mtime_ns), "sha256": sha256_file(path)})
    return records, False


def model_fingerprint(model_dir: Path, manifest_path: Path) -> tuple[str, list[dict[str, Any]], bool]:
    records, cache_used = model_file_records(model_dir, manifest_path)
    fingerprint = stable_hash({"model_path": str(model_dir.resolve()), "files": [{"relative_path": r["relative_path"], "size_bytes": r["size_bytes"], "sha256": r["sha256"]} for r in records]})
    write_json(manifest_path, {"schema_version": f"{EXPERIMENT_SCHEMA}_model_manifest_v1", "model_path": str(model_dir.resolve()), "model_hash": fingerprint, "hash_cache_used": cache_used, "files": records})
    return fingerprint, records, cache_used


def config_payload(paths: Mapping[str, Path], input_hashes: Mapping[str, Any], model_hash: str, scorer_hash: str, model_dir: Path) -> dict[str, Any]:
    generation_hash = stable_hash(GENERATION_CONFIG)
    return {
        "schema_version": f"{EXPERIMENT_SCHEMA}_config_v1",
        "experiment": "REALBIND DaH ORACLE PILOT",
        "protocol": "frozen RealBind pair-direct v1.1; oracle-bbox pilot",
        "data": {"families": str(paths["families"]), "qa": str(paths["qa"]), "dependency": str(paths["dependency"]), "family_count": EXPECTED_FAMILIES, "composition_case_count": EXPECTED_CASES, "composition_types": list(COMPOSITION_TYPES), "modify_frozen_inputs": False},
        "model": {"identifier": MODEL_IDENTIFIER, "path": str(model_dir.resolve()), "model_hash": model_hash, "loader": "experiments.realbind_phase2.atomic_composition_baseline.load_frozen_model -> qwen35_stage2d_behavior_audit.load_behavior_model", "renderer": "experiments.qwen35_kv_gdn_factorial_pilot.render_chat + encode_single"},
        "generation": {**GENERATION_CONFIG, "generation_config_hash": generation_hash, "independent_request": True},
        "scoring": {"revision": SCORING_REVISION, "scorer_file": str(SCORER_FILE.resolve()), "scorer_sha256": scorer_hash, "semantic_vocabulary": "frozen project vocabulary: all canonical colors and all canonical identities in frozen families", "llm_judge": False},
        "transforms": dict(TRANSFORM_CONFIG),
        "randomness": {"family_bootstrap_seed": BOOTSTRAP_SEED, "family_bootstrap_replicates": BOOTSTRAP_REPLICATES, "visual_audit_seed": 42, "visual_audit_families": 20},
        "input_hashes": dict(input_hashes),
        "variants": {name: VARIANT_LABELS[name] for name in VARIANTS},
        "forbidden": ["training", "attention intervention", "AB/BA consistency gate", "automatic grounding", "DeepSeek", "CoT", "thinking", "prompt search", "test-time optimization", "answer history in Stage 2"],
    }


def validate_saved_config(path: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(current)
    old = read_json(path)
    for key in ("schema_version", "data", "model", "generation", "scoring", "transforms", "input_hashes"):
        if stable_json(old.get(key)) != stable_json(current.get(key)):
            raise ExperimentInvalid(f"saved config differs at {key}; refusing protocol mixing")
    return old


def expanded_bbox(bbox: Sequence[float], width: int, height: int, expand_frac: float) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    dx = w * float(expand_frac)
    dy = h * float(expand_frac)
    x0 = max(0.0, x - dx)
    y0 = max(0.0, y - dy)
    x1 = min(float(width), x + w + dx)
    y1 = min(float(height), y + h + dy)
    return [x0, y0, x1, y1]


def bbox_mask(size: tuple[int, int], bbox: Sequence[float], config: Mapping[str, Any]) -> Any:
    from PIL import Image, ImageDraw, ImageFilter

    width, height = size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(tuple(float(v) for v in bbox), fill=255)
    feather = float(config["feather_radius_fraction_of_min_dimension"]) * min(width, height)
    return mask.filter(ImageFilter.GaussianBlur(radius=feather))


def blend_clear_with_blur(image: Any, mask: Any, config: Mapping[str, Any]) -> Any:
    from PIL import Image, ImageFilter

    blur_radius = float(config["blur_radius_fraction_of_min_dimension"]) * min(image.width, image.height)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.composite(image, blurred, mask).convert("RGB")


def make_pair_defocus(image: Any, bbox_A: Sequence[float], bbox_B: Sequence[float], config: Mapping[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Keep the expanded A∪B union clear and Gaussian-blur its complement."""

    from PIL import ImageChops

    cfg = dict(TRANSFORM_CONFIG if config is None else config)
    exp_a = expanded_bbox(bbox_A, image.width, image.height, float(cfg["bbox_expand_frac"]))
    exp_b = expanded_bbox(bbox_B, image.width, image.height, float(cfg["bbox_expand_frac"]))
    mask_a = bbox_mask((image.width, image.height), exp_a, cfg)
    mask_b = bbox_mask((image.width, image.height), exp_b, cfg)
    union = ImageChops.lighter(mask_a, mask_b)
    out = blend_clear_with_blur(image.convert("RGB"), union, cfg)
    return out, {"mask": "A_union_B", "bbox_A": list(map(float, bbox_A)), "bbox_B": list(map(float, bbox_B)), "expanded_bbox_A_xyxy": exp_a, "expanded_bbox_B_xyxy": exp_b, "blur_radius": float(cfg["blur_radius_fraction_of_min_dimension"]) * min(image.width, image.height), "feather_radius": float(cfg["feather_radius_fraction_of_min_dimension"]) * min(image.width, image.height), "size": [int(image.width), int(image.height)]}


def make_target_lock(image: Any, target_bbox: Sequence[float], target_side: str, config: Mapping[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Keep one caller-selected expanded bbox clear; blur every other region."""

    cfg = dict(TRANSFORM_CONFIG if config is None else config)
    if target_side not in {"A", "B"}:
        raise ValueError(f"target_lock target_side must be A or B, got {target_side}")
    exp = expanded_bbox(target_bbox, image.width, image.height, float(cfg["bbox_expand_frac"]))
    mask = bbox_mask((image.width, image.height), exp, cfg)
    out = blend_clear_with_blur(image.convert("RGB"), mask, cfg)
    return out, {"mask": f"target_{target_side}", "target_side": target_side, "bbox": list(map(float, target_bbox)), "expanded_bbox_xyxy": exp, "blur_radius": float(cfg["blur_radius_fraction_of_min_dimension"]) * min(image.width, image.height), "feather_radius": float(cfg["feather_radius_fraction_of_min_dimension"]) * min(image.width, image.height), "size": [int(image.width), int(image.height)]}


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def save_png_no_overwrite(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    payload = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != payload:
            raise ExperimentInvalid(f"existing derived PNG differs; refusing overwrite: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_images(paths: Mapping[str, Path], families: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    from PIL import Image

    root = paths["root"] / "derived_images"
    rows: list[dict[str, Any]] = []
    for family in families:
        fid = str(family["family_id"])
        source = Path(str(family["image_path"])).expanduser()
        with Image.open(source) as opened:
            original = opened.convert("RGB")
            pair, pair_meta = make_pair_defocus(original, bbox_from_family(family, "A"), bbox_from_family(family, "B"), config["transforms"])
            target_a, target_a_meta = make_target_lock(original, bbox_from_family(family, "A"), "A", config["transforms"])
            target_b, target_b_meta = make_target_lock(original, bbox_from_family(family, "B"), "B", config["transforms"])
            targets = {
                "pair_defocus": (root / "pair_defocus" / f"{safe_name(fid)}.png", pair, pair_meta),
                "target_lock_A": (root / "target_lock_A" / f"{safe_name(fid)}.png", target_a, target_a_meta),
                "target_lock_B": (root / "target_lock_B" / f"{safe_name(fid)}.png", target_b, target_b_meta),
            }
            for variant, (destination, transformed, metadata) in targets.items():
                save_png_no_overwrite(destination, transformed)
                transformed.close()
            original.close()
        original_hash = sha256_file(source)
        for variant, (destination, _transformed, metadata) in targets.items():
            rows.append({"family_id": fid, "variant": variant, "original_image_path": str(source.resolve()), "derived_image_path": str(destination.resolve()), "original_image_sha256": original_hash, "transformed_image_sha256": sha256_file(destination), **metadata, "transform_config": dict(config["transforms"])})
    audit = {"schema_version": f"{EXPERIMENT_SCHEMA}_image_transform_audit_v1", "passed": True, "family_count": len(families), "derived_image_count": len(rows), "expected_derived_image_count": EXPECTED_FAMILIES * 3, "rows": rows, "deterministic": True, "resolution_preserved": all(row["size"] == [int(f["image_width"]), int(f["image_height"])] for row in rows for f in families if str(f["family_id"]) == row["family_id"]), "gt_bbox_used_only_for_visual_filtering": True, "target_lock_runtime_selection": "selected Stage-1 A/B label chooses precomputed target_lock_A/B; no GT correction", "no_annotations_in_pixels": True}
    write_json(paths["transform_audit"], audit)
    return audit


def bbox_audit(paths: Mapping[str, Path], families: Sequence[Mapping[str, Any]], transform_audit_payload: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family in families:
        rows.append({"family_id": str(family["family_id"]), "image_path": str(family["image_path"]), "image_width": int(family["image_width"]), "image_height": int(family["image_height"]), "bbox_A": bbox_from_family(family, "A"), "bbox_B": bbox_from_family(family, "B"), "object_A": str(family["A"]["object"]), "object_B": str(family["B"]["object"]), "instance_id_A": str(family["A"]["instance_id"]), "instance_id_B": str(family["B"]["instance_id"])})
    audit = {"schema_version": f"{EXPERIMENT_SCHEMA}_bbox_audit_v1", "passed": True, "family_count": len(rows), "schema_source": "frozen family JSONL, audited before transformation", "bbox_field": "family.A.bbox / family.B.bbox", "bbox_convention": "[x, y, width, height]", "identity_fields": {"A": "family.A.object", "B": "family.B.object"}, "target_selection_gate": "GT bbox is never used to choose predicted target; Stage-1 prediction supplies A/B", "rows": rows, "transform_audit_sha256": sha256_text(stable_json(transform_audit_payload))}
    write_json(paths["bbox_audit"], audit)
    return audit


def as_file_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except ValueError:
        return "file://" + str(path.resolve())


def build_visual_audit(paths: Mapping[str, Path], families: Sequence[Mapping[str, Any]]) -> None:
    rng = random.Random(42)
    selected = rng.sample(list(families), min(20, len(families)))
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>RealBind DaH oracle visual audit</title>", "<style>body{font-family:system-ui,sans-serif} .family{margin:1.5em 0;border-bottom:1px solid #ccc;padding-bottom:1em} .grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:8px} img{max-width:100%;height:auto} figcaption{font-size:.85em;overflow-wrap:anywhere}</style></head><body>", "<h1>RealBind DaH oracle pilot visual audit</h1><p>Random seed 42; 20 families. Audit only; no boxes, labels, arrows, or text are drawn on images.</p>"]
    for family in selected:
        fid = str(family["family_id"])
        safe = safe_name(fid)
        original = Path(str(family["image_path"])).resolve()
        parts.append(f"<section class='family'><h2>{html.escape(fid)}</h2><div class='grid'>")
        entries = [("Original", original), ("PAIR_DEFOCUS", paths["root"] / "derived_images" / "pair_defocus" / f"{safe}.png"), ("TARGET_LOCK_A", paths["root"] / "derived_images" / "target_lock_A" / f"{safe}.png"), ("TARGET_LOCK_B", paths["root"] / "derived_images" / "target_lock_B" / f"{safe}.png")]
        for label, image_path in entries:
            parts.append(f"<figure><img src='{html.escape(as_file_uri(image_path))}' alt='{html.escape(label)} {html.escape(fid)}'><figcaption>{html.escape(label)}</figcaption></figure>")
        parts.append("</div></section>")
    parts.append("</body></html>\n")
    paths["visual_html"].write_text("".join(parts), encoding="utf-8")


def prepare_protocol(args: argparse.Namespace, paths: Mapping[str, Path], input_paths: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if args.model_dir.expanduser().resolve() != MODEL_DIR.resolve():
        raise ExperimentInvalid(f"the formal RealBind loader is frozen to {MODEL_DIR}; refusing --model-dir={args.model_dir}")
    families, qa_rows, _dependency, data_audit = validate_frozen_schema(input_paths)
    model_hash, _model_files, _cache_used = model_fingerprint(args.model_dir.expanduser().resolve(), paths["model_manifest"])
    scorer_hash = sha256_file(SCORER_FILE)
    input_hashes = {name: file_record(input_paths[name]) for name in input_paths}
    config = config_payload(input_paths, input_hashes, model_hash, scorer_hash, args.model_dir.expanduser().resolve())
    config = validate_saved_config(paths["config"], config)
    write_json(paths["config"], config)
    write_json(paths["data_alignment"], data_audit)
    transform_audit_payload = prepare_images(paths, families, config)
    bbox_payload = bbox_audit(paths, families, transform_audit_payload)
    build_visual_audit(paths, families)
    cases = build_cases(families, qa_rows)
    provenance = {
        "schema_version": f"{EXPERIMENT_SCHEMA}_provenance_v1",
        "experiment": "REALBIND DaH ORACLE PILOT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "config_sha256": sha256_text(stable_json(config)),
        "config": config,
        "data_alignment_audit": data_audit,
        "bbox_audit": bbox_payload,
        "image_transform_audit": {key: value for key, value in transform_audit_payload.items() if key != "rows"},
        "case_count": len(cases),
        "frozen_inputs_read_only": True,
    }
    write_json(paths["provenance"], provenance)
    return families, qa_rows, cases, config, data_audit


def check_current_frozen_hashes(config: Mapping[str, Any], input_paths: Mapping[str, Path]) -> None:
    for name, expected in config.get("input_hashes", {}).items():
        path = input_paths.get(name)
        if path is None or not path.is_file() or sha256_file(path) != expected.get("sha256"):
            raise ExperimentInvalid(f"frozen {name} hash changed or file disappeared")
    if not SCORER_FILE.is_file() or sha256_file(SCORER_FILE) != config.get("scoring", {}).get("scorer_sha256"):
        raise ExperimentInvalid("scorer hash changed since protocol preparation")


def transform_index(paths: Mapping[str, Path]) -> dict[tuple[str, str], dict[str, Any]]:
    audit = read_json(paths["transform_audit"])
    rows = audit.get("rows", [])
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["family_id"]), str(row["variant"]))
        if key in index:
            raise ExperimentInvalid(f"duplicate transform audit row {key}")
        if not Path(str(row["derived_image_path"])).is_file() or sha256_file(Path(str(row["derived_image_path"]))) != row.get("transformed_image_sha256"):
            raise ExperimentInvalid(f"derived image audit mismatch {key}")
        index[key] = dict(row)
    if len(index) != EXPECTED_FAMILIES * 3:
        raise ExperimentInvalid(f"derived image count={len(index)}; expected {EXPECTED_FAMILIES * 3}")
    return index


def cache_key(model_hash: str, image_hash: str, prompt: str, generation_hash: str) -> tuple[str, dict[str, str]]:
    fields = {"model_hash": model_hash, "image_sha256": image_hash, "prompt_sha256": sha256_text(prompt), "generation_config_hash": generation_hash}
    return stable_hash(fields), fields


def load_cache(path: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    rows = read_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    expected_model = str(config["model"]["model_hash"])
    expected_generation = str(config["generation"]["generation_config_hash"])
    for row in rows:
        key = str(row.get("cache_key", ""))
        if not key or key in index:
            raise ExperimentInvalid(f"cache key missing or duplicate: {key}")
        if row.get("model_hash") != expected_model or row.get("generation_config_hash") != expected_generation:
            raise ExperimentInvalid("call cache uses a different model or generation configuration")
        index[key] = dict(row)
    return index


def save_cache(path: Path, index: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [index[key] for key in sorted(index)]
    write_jsonl_atomic(path, rows)


def tokenizer_from_processor(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def call_record_from_raw(
    *,
    case_id: str,
    stage: str,
    image_path: Path,
    prompt: str,
    answer_type: str,
    ground_truth: str,
    family: Mapping[str, Any],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    raw_response_with_special: str,
    decoded_response: str,
    token_ids: Sequence[int],
    processor: Any,
    cache_key_value: str,
    cache_fields: Mapping[str, str],
    config: Mapping[str, Any],
    cache_reused: bool,
    cache_source: str,
) -> dict[str, Any]:
    parsed = frozen_scorer.parse_semantic_answer(decoded_response, answer_type, ground_truth, family, semantic_vocabulary)
    token_count = len(token_ids)
    eos = frozen_scorer.eos_observed(token_ids, processor)
    semantic_trunc = frozen_scorer.semantic_truncation(parsed, decoded_response, token_count, eos)
    return {
        "case_id": case_id,
        "stage": stage,
        "image_path": str(image_path.resolve()),
        "image_sha256": cache_fields["image_sha256"],
        "prompt": prompt,
        "prompt_sha256": cache_fields["prompt_sha256"],
        "answer_type": answer_type,
        "ground_truth": ground_truth,
        "raw_response": decoded_response,
        "raw_response_with_special_tokens": raw_response_with_special,
        "raw_response_repr": repr(decoded_response),
        "generated_token_ids": [int(value) for value in token_ids],
        "generated_token_count": token_count,
        "eos_observed": bool(eos),
        "hit_max_new_tokens": bool(token_count >= int(config["generation"]["max_new_tokens"])),
        "semantic_truncation": bool(semantic_trunc),
        "normalized_prediction": parsed.get("normalized_prediction"),
        "invalid_prediction": bool(parsed.get("invalid_prediction")),
        "format_violation": bool(parsed.get("format_violation")),
        "parser_reason": str(parsed.get("parser_reason")),
        "correct": bool(not parsed.get("invalid_prediction") and parsed.get("normalized_prediction") == ground_truth),
        "scoring_revision": SCORING_REVISION,
        "model_identifier": MODEL_IDENTIFIER,
        "model_hash": config["model"]["model_hash"],
        "generation_config": dict(GENERATION_CONFIG),
        "generation_config_hash": config["generation"]["generation_config_hash"],
        "cache_key": cache_key_value,
        "cache_key_fields": dict(cache_fields),
        "cache_reused": bool(cache_reused),
        "cache_source": cache_source,
        "fresh_context_request": True,
        "enable_thinking": False,
    }


def load_model_once() -> tuple[Any, Any, Any]:
    """Use the existing audited baseline loader; no second loader is defined."""

    return frozen_scorer.frozen_baseline.load_frozen_model()


def run_single_vlm_call(
    processor: Any,
    model: Any,
    device: Any,
    *,
    case_id: str,
    stage: str,
    image_path: Path,
    prompt: str,
    answer_type: str,
    ground_truth: str,
    family: Mapping[str, Any],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any]:
    import torch
    # Import lazily: the frozen renderer module also contains unrelated
    # pandas-backed analysis helpers, while prepare/analyze should remain
    # usable as offline audits in the project environment.
    from experiments import qwen35_kv_gdn_factorial_pilot as frozen_runtime

    image_hash = sha256_file(image_path)
    key, fields = cache_key(str(config["model"]["model_hash"]), image_hash, prompt, str(config["generation"]["generation_config_hash"]))
    if key in cache:
        cached = dict(cache[key])
        if cached.get("image_sha256") != image_hash or cached.get("prompt_sha256") != fields["prompt_sha256"] or cached.get("cache_key_fields") != fields:
            raise ExperimentInvalid(f"cache key payload mismatch for {key}")
        cached.update({"case_id": case_id, "stage": stage, "cache_reused": True, "cache_source": str(cache_path.resolve())})
        return cached
    rendered = frozen_runtime.render_chat(processor, str(image_path), prompt)
    inputs = frozen_runtime.encode_single(processor, rendered, str(image_path), device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(config["generation"]["max_new_tokens"]), do_sample=False, use_cache=False)
    input_length = int(inputs["input_ids"].shape[-1])
    continuation = generated[0, input_length:]
    raw_with_special, decoded, token_ids = frozen_scorer.frozen_baseline.decode_generation(processor, continuation)
    row = call_record_from_raw(
        case_id=case_id,
        stage=stage,
        image_path=image_path,
        prompt=prompt,
        answer_type=answer_type,
        ground_truth=ground_truth,
        family=family,
        semantic_vocabulary=semantic_vocabulary,
        raw_response_with_special=raw_with_special,
        decoded_response=decoded,
        token_ids=token_ids,
        processor=processor,
        cache_key_value=key,
        cache_fields=fields,
        config=config,
        cache_reused=False,
        cache_source="fresh_model_generate",
    )
    cache[key] = dict(row)
    save_cache(cache_path, cache)
    return row


def checkpoint_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.stem + ".partial.jsonl")


def load_checkpoint(path: Path, expected_case_ids: set[str], config: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in expected_case_ids or case_id in result:
            raise ExperimentInvalid(f"{label} checkpoint has invalid/duplicate case_id={case_id}")
        if row.get("model_hash") != config["model"]["model_hash"] or row.get("generation_config_hash") != config["generation"]["generation_config_hash"]:
            raise ExperimentInvalid(f"{label} checkpoint has incompatible model/generation hash")
        result[case_id] = dict(row)
    return result


def load_existing_rows(final_path: Path, expected_case_ids: set[str], config: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    """Prefer a completed raw file, otherwise use its atomic partial file."""

    if final_path.is_file():
        result = load_checkpoint(final_path, expected_case_ids, config, label)
        if len(result) != len(expected_case_ids):
            raise ExperimentInvalid(f"{label} final raw output exists but is incomplete; refusing to modify it")
        return result
    return load_checkpoint(checkpoint_path(final_path), expected_case_ids, config, label)


def promote_complete_raw(final_path: Path, rows: Sequence[Mapping[str, Any]], expected_case_ids: set[str], label: str) -> None:
    if len(rows) != len(expected_case_ids) or {str(row.get("case_id")) for row in rows} != expected_case_ids:
        raise ExperimentInvalid(f"{label} incomplete: {len(rows)}/{len(expected_case_ids)}")
    if final_path.exists():
        existing = load_checkpoint(final_path, expected_case_ids, {"model": {"model_hash": rows[0]["model_hash"]}, "generation": {"generation_config_hash": rows[0]["generation_config_hash"]}}, label) if rows else {}
        if stable_hash(existing) != stable_hash({str(row["case_id"]): dict(row) for row in rows}):
            raise ExperimentInvalid(f"{label} complete output exists with different raw rows; refusing overwrite")
        return
    write_jsonl_atomic(final_path, rows, refuse_existing=True)


def run_stage1(
    cases: Sequence[Mapping[str, Any]],
    families_by_id: Mapping[str, Mapping[str, Any]],
    processor: Any,
    model: Any,
    device: Any,
    config: Mapping[str, Any],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    transform_index_payload: Mapping[tuple[str, str], Mapping[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    final_path: Path,
    image_variant: str,
    stage_label: str,
) -> list[dict[str, Any]]:
    expected = {str(case["case_id"]) for case in cases}
    partial = load_existing_rows(final_path, expected, config, stage_label)
    rows_by_id = dict(partial)
    ordered: list[dict[str, Any]] = []
    for case in cases:
        cid = str(case["case_id"])
        if cid in rows_by_id:
            ordered.append(rows_by_id[cid])
            continue
        family = families_by_id[str(case["family_id"])]
        image_path = Path(str(case["image_path"]))
        image_audit: dict[str, Any] | None = None
        if image_variant != "original":
            image_audit = dict(transform_index_payload[(str(case["family_id"]), image_variant)])
            image_path = Path(str(image_audit["derived_image_path"]))
        call = run_single_vlm_call(processor, model, device, case_id=cid, stage=stage_label, image_path=image_path, prompt=str(case["rel2i_question"]), answer_type="identity", ground_truth=str(case["rel2i_ground_truth"]), family=family, semantic_vocabulary=semantic_vocabulary, config=config, cache=cache, cache_path=cache_path)
        call.update({"family_id": str(case["family_id"]), "qa_id": str(case["rel2i_qa_id"]), "qa_type": str(case["rel2i_type"]), "target_side": str(case["target_side"]), "image_variant": image_variant, "image_transform": image_audit, "frozen_question_reused": True})
        rows_by_id[cid] = call
        write_jsonl_atomic(checkpoint_path(final_path), [rows_by_id[str(item["case_id"])] for item in cases if str(item["case_id"]) in rows_by_id])
        print(f"{stage_label}: {len(rows_by_id)}/{len(cases)}", flush=True)
        ordered.append(call)
    complete = [rows_by_id[str(case["case_id"])] for case in cases]
    promote_complete_raw(final_path, complete, expected, stage_label)
    return complete


def run_direct_variant(
    cases: Sequence[Mapping[str, Any]],
    families_by_id: Mapping[str, Mapping[str, Any]],
    processor: Any,
    model: Any,
    device: Any,
    config: Mapping[str, Any],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    transform_index_payload: Mapping[tuple[str, str], Mapping[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    final_path: Path,
    variant: str,
) -> list[dict[str, Any]]:
    expected = {str(case["case_id"]) for case in cases}
    partial = load_existing_rows(final_path, expected, config, variant)
    rows_by_id = dict(partial)
    for case in cases:
        cid = str(case["case_id"])
        if cid in rows_by_id:
            continue
        family = families_by_id[str(case["family_id"])]
        image_variant = "original" if variant == "original" else "pair_defocus"
        image_audit = None if image_variant == "original" else dict(transform_index_payload[(str(case["family_id"]), image_variant)])
        image_path = Path(str(case["image_path"])) if image_audit is None else Path(str(image_audit["derived_image_path"]))
        call = run_single_vlm_call(processor, model, device, case_id=cid, stage="composition", image_path=image_path, prompt=str(case["composition_question"]), answer_type="color", ground_truth=str(case["composition_ground_truth"]), family=family, semantic_vocabulary=semantic_vocabulary, config=config, cache=cache, cache_path=cache_path)
        row = {"case_id": cid, "variant": variant, "variant_label": VARIANT_LABELS[variant], "family_id": str(case["family_id"]), "image_id": str(case["image_id"]), "composition_qa_id": str(case["composition_qa_id"]), "composition_qa_type": str(case["composition_qa_type"]), "composition_question": str(case["composition_question"]), "composition_ground_truth": str(case["composition_ground_truth"]), "image_variant": image_variant, "image_transform": image_audit, "raw_response": call["raw_response"], "raw_response_with_special_tokens": call["raw_response_with_special_tokens"], "normalized_prediction": call["normalized_prediction"], "final_prediction": call["normalized_prediction"], "final_correct": bool(call["correct"]), "final_invalid": bool(call["invalid_prediction"]), "final_format_violation": bool(call["format_violation"]), "final_semantic_truncation": bool(call["semantic_truncation"]), "final_generated_token_count": int(call["generated_token_count"]), "final_parser_reason": call["parser_reason"], "invalid_prediction": bool(call["invalid_prediction"]), "format_violation": bool(call["format_violation"]), "semantic_truncation": bool(call["semantic_truncation"]), "hit_max_new_tokens": bool(call["hit_max_new_tokens"]), "call_key": call["cache_key"], "model_hash": call["model_hash"], "generation_config_hash": call["generation_config_hash"], "scoring_revision": call["scoring_revision"], "fresh_context_request": True, "stage1_raw": None, "stage1_normalized": None, "stage1_selected_identity": None, "stage1_selected_identity_label": None, "stage1_correct": None, "stage1_valid": None, "stage2_raw": None, "stage2_prediction": None, "stage2_correct_for_selected_identity": None, "stage2_correct_for_composition_GT": None}
        rows_by_id[cid] = row
        write_jsonl_atomic(checkpoint_path(final_path), [rows_by_id[str(item["case_id"])] for item in cases if str(item["case_id"]) in rows_by_id])
        print(f"{variant}: {len(rows_by_id)}/{len(cases)}", flush=True)
    complete = [rows_by_id[str(case["case_id"])] for case in cases]
    promote_complete_raw(final_path, complete, expected, variant)
    return complete


def stage1_selection(binding: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    normalized = binding.get("normalized_prediction")
    a = str(case["object_A"])
    b = str(case["object_B"])
    selected_label: str | None = None
    if normalized is not None and str(normalized).casefold() == a.casefold():
        selected_label = "A"
    elif normalized is not None and str(normalized).casefold() == b.casefold():
        selected_label = "B"
    valid = bool(not binding.get("invalid_prediction") and selected_label in {"A", "B"})
    selected_identity = a if selected_label == "A" else b if selected_label == "B" else None
    return {"stage1_raw": binding.get("raw_response"), "stage1_normalized": normalized, "stage1_selected_identity": selected_identity, "stage1_selected_identity_label": selected_label, "stage1_correct": bool(valid and selected_label == ("A" if case["target_side"] == "A" else "B")), "stage1_valid": valid, "stage1_invalid": not valid, "stage1_scorer_invalid": bool(binding.get("invalid_prediction")), "stage1_format_violation": bool(binding.get("format_violation")) or not valid, "stage1_semantic_truncation": bool(binding.get("semantic_truncation")), "stage1_generated_token_count": int(binding.get("generated_token_count", 0)), "stage1_hit_max_new_tokens": bool(binding.get("hit_max_new_tokens")), "stage1_parser_reason": binding.get("parser_reason"), "stage1_call_key": binding.get("cache_key"), "stage1_image_path": binding.get("image_path"), "stage1_image_sha256": binding.get("image_sha256"), "stage1_question": binding.get("prompt"), "stage1_qa_id": binding.get("qa_id"), "stage1_qa_type": binding.get("qa_type")}


def run_decomposed_variant(
    cases: Sequence[Mapping[str, Any]],
    families_by_id: Mapping[str, Mapping[str, Any]],
    binding_rows: Mapping[str, Mapping[str, Any]],
    processor: Any,
    model: Any,
    device: Any,
    config: Mapping[str, Any],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    transform_index_payload: Mapping[tuple[str, str], Mapping[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    final_path: Path,
    variant: str,
    stage1_variant: str,
    use_target_lock: bool,
) -> list[dict[str, Any]]:
    expected = {str(case["case_id"]) for case in cases}
    partial = load_existing_rows(final_path, expected, config, variant)
    rows_by_id = dict(partial)
    for case in cases:
        cid = str(case["case_id"])
        if cid in rows_by_id:
            continue
        family = families_by_id[str(case["family_id"])]
        binding = binding_rows[cid]
        bind = stage1_selection(binding, case)
        selected_label = bind["stage1_selected_identity_label"]
        stage2: dict[str, Any] | None = None
        if bind["stage1_valid"]:
            i2c_type = "I2C_A" if selected_label == "A" else "I2C_B"
            prompt = str(case["i2c_A_question"] if selected_label == "A" else case["i2c_B_question"])
            gt_color = str(case["color_A"] if selected_label == "A" else case["color_B"])
            image_variant = "original"
            image_audit = None
            if use_target_lock:
                image_variant = "target_lock_A" if selected_label == "A" else "target_lock_B"
                image_audit = dict(transform_index_payload[(str(case["family_id"]), image_variant)])
            image_path = Path(str(case["image_path"])) if image_audit is None else Path(str(image_audit["derived_image_path"]))
            stage2 = run_single_vlm_call(processor, model, device, case_id=cid, stage="attribute", image_path=image_path, prompt=prompt, answer_type="color", ground_truth=gt_color, family=family, semantic_vocabulary=semantic_vocabulary, config=config, cache=cache, cache_path=cache_path)
            stage2.update({"family_id": str(case["family_id"]), "qa_id": str(case["i2c_A_qa_id"] if selected_label == "A" else case["i2c_B_qa_id"]), "qa_type": i2c_type, "selected_identity_label": selected_label, "selected_identity": bind["stage1_selected_identity"], "selected_identity_color": gt_color, "image_variant": image_variant, "image_transform": image_audit, "frozen_question_reused": True})
        stage2_prediction = stage2.get("normalized_prediction") if stage2 else None
        selected_color = str(case["color_A"] if selected_label == "A" else case["color_B"]) if selected_label in {"A", "B"} else None
        attr_correct_selected = bool(stage2 is not None and not stage2.get("invalid_prediction") and stage2_prediction == selected_color)
        attr_correct_comp = bool(stage2 is not None and not stage2.get("invalid_prediction") and stage2_prediction == case["composition_ground_truth"])
        final_prediction = stage2_prediction if stage2 is not None else None
        final_invalid = bool(stage2 is None or stage2.get("invalid_prediction"))
        final_format = bool(bind["stage1_format_violation"] or (stage2 is not None and stage2.get("format_violation")))
        final_trunc = bool(bind["stage1_semantic_truncation"] or (stage2 is not None and stage2.get("semantic_truncation")))
        row: dict[str, Any] = {"case_id": cid, "variant": variant, "variant_label": VARIANT_LABELS[variant], "family_id": str(case["family_id"]), "image_id": str(case["image_id"]), "composition_qa_id": str(case["composition_qa_id"]), "composition_qa_type": str(case["composition_qa_type"]), "composition_question": str(case["composition_question"]), "composition_ground_truth": str(case["composition_ground_truth"]), "target_identity": str(case["target_identity"]), "target_color": str(case["target_color"]), "stage1_source_variant": stage1_variant, **bind, "stage2_raw": stage2.get("raw_response") if stage2 else None, "stage2_prediction": stage2_prediction, "stage2_normalized_prediction": stage2_prediction, "stage2_correct_for_selected_identity": attr_correct_selected if stage2 is not None else None, "stage2_correct_for_composition_GT": attr_correct_comp if stage2 is not None else None, "stage2_invalid": bool(stage2.get("invalid_prediction")) if stage2 else None, "stage2_format_violation": bool(stage2.get("format_violation")) if stage2 else None, "stage2_semantic_truncation": bool(stage2.get("semantic_truncation")) if stage2 else None, "stage2_hit_max_new_tokens": bool(stage2.get("hit_max_new_tokens")) if stage2 else None, "stage2_generated_token_count": int(stage2.get("generated_token_count", 0)) if stage2 else None, "stage2_call_key": stage2.get("cache_key") if stage2 else None, "stage2_qa_id": stage2.get("qa_id") if stage2 else None, "stage2_qa_type": stage2.get("qa_type") if stage2 else None, "stage2_question": stage2.get("prompt") if stage2 else None, "stage2_image_path": stage2.get("image_path") if stage2 else None, "stage2_image_sha256": stage2.get("image_sha256") if stage2 else None, "stage2_image_variant": stage2.get("image_variant") if stage2 else None, "stage2_selected_identity_color": selected_color, "final_prediction": final_prediction, "final_correct": bool(not final_invalid and final_prediction == case["composition_ground_truth"]), "final_invalid": final_invalid, "final_format_violation": final_format, "final_semantic_truncation": final_trunc, "final_generated_token_count": int(stage2.get("generated_token_count", 0)) if stage2 is not None else int(binding.get("generated_token_count", 0)), "final_hit_max_new_tokens": bool((stage2 or binding).get("hit_max_new_tokens")), "final_parser_reason": stage2.get("parser_reason") if stage2 else "stage1_invalid_no_stage2", "wrong_binding_same_color_accidentally_correct": bool(not bind["stage1_correct"] and bind["stage1_valid"] and attr_correct_comp and selected_color == case["target_color"]), "model_hash": config["model"]["model_hash"], "generation_config_hash": config["generation"]["generation_config_hash"], "scoring_revision": SCORING_REVISION, "fresh_stage2_context": True, "stage2_contains_stage1_history": False, "gt_bbox_used_for_target_selection": False}
        rows_by_id[cid] = row
        write_jsonl_atomic(checkpoint_path(final_path), [rows_by_id[str(item["case_id"])] for item in cases if str(item["case_id"]) in rows_by_id])
        print(f"{variant}: {len(rows_by_id)}/{len(cases)}", flush=True)
    complete = [rows_by_id[str(case["case_id"])] for case in cases]
    promote_complete_raw(final_path, complete, expected, variant)
    return complete


def metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(as_bool(row.get("final_correct")) for row in rows)
    return {"N": n, "correct": correct, "accuracy": correct / n if n else None, "invalid_count": sum(as_bool(row.get("final_invalid")) for row in rows), "format_violation_count": sum(as_bool(row.get("final_format_violation")) for row in rows), "truncation_count": sum(as_bool(row.get("final_semantic_truncation")) for row in rows), "hit_max_new_tokens_count": sum(as_bool(row.get("final_hit_max_new_tokens")) for row in rows)}


def side_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {qa_type: metric([row for row in rows if row["composition_qa_type"] == qa_type]) for qa_type in COMPOSITION_TYPES}


def decomposed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    valid = [row for row in rows if as_bool(row.get("stage1_valid"))]
    bind_correct = [row for row in rows if as_bool(row.get("stage1_correct"))]
    bind_wrong_valid = [row for row in rows if as_bool(row.get("stage1_valid")) and not as_bool(row.get("stage1_correct"))]
    def rate(values: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
        denominator = len(values)
        numerator = sum(as_bool(row.get(field)) for row in values)
        return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}
    categories = Counter(str(row.get("failure_category")) for row in rows)
    wrong_binding_final_correct = [row for row in rows if as_bool(row.get("stage1_valid")) and not as_bool(row.get("stage1_correct")) and as_bool(row.get("final_correct"))]
    same_color_accidental = sum(as_bool(row.get("wrong_binding_same_color_accidentally_correct")) for row in rows)
    return {"N": n, "stage1_binding_accuracy": rate(rows, "stage1_correct"), "stage1_valid_rate": rate(rows, "stage1_valid"), "stage2_attribute_accuracy_given_binding_correct": rate(bind_correct, "stage2_correct_for_selected_identity"), "stage2_attribute_accuracy_given_binding_wrong": rate(bind_wrong_valid, "stage2_correct_for_selected_identity"), "final_composition_accuracy": rate(rows, "final_correct"), "final": metric(rows), "pipeline_failure_categories": dict(sorted(categories.items())), "wrong_binding_final_correct_count": len(wrong_binding_final_correct), "wrong_binding_final_correct_different_color_count": len(wrong_binding_final_correct) - same_color_accidental, "wrong_binding_same_color_accidentally_correct_count": same_color_accidental}


def classify_failure(row: Mapping[str, Any]) -> str:
    if not as_bool(row.get("stage1_valid")):
        return "BIND_INVALID"
    if as_bool(row.get("stage1_correct")):
        return "BIND_CORRECT_ATTR_CORRECT" if as_bool(row.get("stage2_correct_for_selected_identity")) else "BIND_CORRECT_ATTR_WRONG"
    return "BIND_WRONG_ATTR_CORRECT_FOR_WRONG_ENTITY" if as_bool(row.get("stage2_correct_for_selected_identity")) else "BIND_WRONG_ATTR_OTHER"


def add_failure_categories(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if row.get("stage1_valid") is not None:
            updated["failure_category"] = classify_failure(updated)
        output.append(updated)
    return output


def exact_mcnemar(original: Sequence[bool], repaired: Sequence[bool]) -> dict[str, Any]:
    b = sum(bool(a) and not bool(r) for a, r in zip(original, repaired))
    c = sum(not bool(a) and bool(r) for a, r in zip(original, repaired))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2.0**n)
        p = min(1.0, 2.0 * tail)
    return {"b_original_correct_repair_wrong": b, "c_original_wrong_repair_correct": c, "discordant": n, "p_exact_two_sided": p}


def family_bootstrap_delta(rows: Sequence[Mapping[str, Any]], original_by_case: Mapping[str, Mapping[str, Any]], value_field: str = "final_correct") -> dict[str, Any]:
    family_order = sorted({str(row["family_id"]) for row in rows})
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(family_order, size=len(family_order), replace=True)
        repaired_values: list[bool] = []
        original_values: list[bool] = []
        for fid in sampled:
            for row in by_family[str(fid)]:
                repaired_values.append(as_bool(row.get(value_field)))
                original_values.append(as_bool(original_by_case[str(row["case_id"])].get(value_field)))
        values.append(float(np.mean(repaired_values) - np.mean(original_values)))
    observed = float(np.mean([as_bool(row.get(value_field)) for row in rows]) - np.mean([as_bool(original_by_case[str(row["case_id"])].get(value_field)) for row in rows]))
    return {"observed_delta_accuracy": observed, "confidence_interval_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "unit": "family_id"}


def family_bootstrap_ratio(rows: Sequence[Mapping[str, Any]], numerator_fn: Any, denominator_fn: Any) -> dict[str, Any]:
    family_order = sorted({str(row["family_id"]) for row in rows})
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    zero_denominator = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(family_order, size=len(family_order), replace=True)
        sample = [row for fid in sampled for row in by_family[str(fid)]]
        denominator = int(denominator_fn(sample))
        if denominator == 0:
            zero_denominator += 1
        else:
            values.append(float(numerator_fn(sample) / denominator))
    denominator = int(denominator_fn(rows))
    numerator = int(numerator_fn(rows))
    observed = numerator / denominator if denominator else None
    return {"numerator": numerator, "denominator": denominator, "observed_rate": observed, "confidence_interval_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "valid_bootstrap_draws": len(values), "zero_denominator_draws": zero_denominator, "unit": "family_id"}


def load_historical_baseline(historical_root: Path, input_paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol_path = historical_root / "protocol.json"
    results_path = historical_root / "inference_results.jsonl"
    if not protocol_path.is_file() or not results_path.is_file():
        raise ExperimentInvalid(f"historical Qwen baseline is missing: {historical_root}")
    protocol = read_json(protocol_path)
    failures: list[str] = []
    if protocol.get("scoring", {}).get("scoring_revision") != SCORING_REVISION:
        failures.append("historical scorer revision mismatch")
    if protocol.get("model", {}).get("identifier") != MODEL_IDENTIFIER:
        failures.append("historical model identifier mismatch")
    historical_generation = protocol.get("generation", {})
    required_generation = {"batch_size": 1, "do_sample": False, "use_cache": False, "max_new_tokens": 16}
    if any(historical_generation.get(key) != value for key, value in required_generation.items()):
        failures.append("historical generation config mismatch")
    if historical_generation.get("independent_query") is not True or historical_generation.get("other_answers_in_prompt") is not False or historical_generation.get("prompt_suffix") != "":
        failures.append("historical independent-query/prompt-suffix config mismatch")
    rows = read_jsonl(results_path)
    if len(rows) != 700 or len({str(row.get("qa_id")) for row in rows}) != 700:
        failures.append("historical Qwen baseline is not complete 700-row output")
    by_qa = {str(row["qa_id"]): dict(row) for row in rows}
    by_family: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_family[str(row["family_id"])][str(row["qa_type"])] = dict(row)
    for name, path in input_paths.items():
        old_hash = protocol.get("input_hashes", {}).get(name, {}).get("sha256")
        if old_hash and old_hash != sha256_file(path):
            failures.append(f"historical baseline input hash mismatch: {name}")
    if failures:
        raise ExperimentInvalid("HISTORICAL_BASELINE_AUDIT_FAILED", failures)
    return {"protocol_path": str(protocol_path.resolve()), "results_path": str(results_path.resolve()), "protocol_sha256": sha256_file(protocol_path), "results_sha256": sha256_file(results_path), "protocol": protocol}, by_qa, by_family


def historical_audit(cases: Sequence[Mapping[str, Any]], original_rows: Sequence[Mapping[str, Any]], historical_by_qa: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in original_rows:
        old = historical_by_qa.get(str(row["composition_qa_id"]))
        if old is None:
            raise ExperimentInvalid(f"historical baseline missing composition QA {row['composition_qa_id']}")
        rows.append({"case_id": row["case_id"], "historical_raw_response": old.get("raw_response"), "rerun_raw_response": row.get("raw_response"), "raw_response_agreement": old.get("raw_response") == row.get("raw_response"), "historical_normalized_prediction": old.get("normalized_prediction"), "rerun_normalized_prediction": row.get("normalized_prediction"), "normalized_prediction_agreement": old.get("normalized_prediction") == row.get("normalized_prediction"), "historical_correct": as_bool(old.get("correct")), "rerun_correct": as_bool(row.get("final_correct")), "correctness_agreement": as_bool(old.get("correct")) == as_bool(row.get("final_correct"))})
    return {"N": len(rows), "raw_response_agreement_count": sum(row["raw_response_agreement"] for row in rows), "raw_response_agreement_rate": sum(row["raw_response_agreement"] for row in rows) / len(rows), "normalized_prediction_agreement_count": sum(row["normalized_prediction_agreement"] for row in rows), "normalized_prediction_agreement_rate": sum(row["normalized_prediction_agreement"] for row in rows) / len(rows), "correctness_agreement_count": sum(row["correctness_agreement"] for row in rows), "correctness_agreement_rate": sum(row["correctness_agreement"] for row in rows) / len(rows), "rows": rows}


def original_strong_path(cases: Sequence[Mapping[str, Any]], historical_by_qa: Mapping[str, Mapping[str, Any]], historical_by_family: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subset: list[dict[str, Any]] = []
    for case in cases:
        family_rows = historical_by_family[str(case["family_id"])]
        relation = family_rows["PAIR_REL_A_TO_B"]
        rel2i = family_rows[str(case["rel2i_type"])]
        target_i2c = family_rows["I2C_A" if case["target_side"] == "A" else "I2C_B"]
        comp = historical_by_qa[str(case["composition_qa_id"])]
        ready = as_bool(relation.get("correct")) and as_bool(rel2i.get("correct")) and as_bool(target_i2c.get("correct"))
        subset.append({"case_id": case["case_id"], "family_id": case["family_id"], "composition_qa_type": case["composition_qa_type"], "original_correct": as_bool(comp.get("correct")), "strong_path_ready": ready, "original_strong_failure": bool(ready and not as_bool(comp.get("correct"))), "relation_correct": as_bool(relation.get("correct")), "rel2i_correct": as_bool(rel2i.get("correct")), "target_i2c_correct": as_bool(target_i2c.get("correct"))})
    counts = {"N": len(subset), "strong_path_ready": sum(row["strong_path_ready"] for row in subset), "original_strong_failures": sum(row["original_strong_failure"] for row in subset), "original_composition_correct": sum(row["original_correct"] for row in subset), "original_composition_failures": sum(not row["original_correct"] for row in subset)}
    if counts["strong_path_ready"] != 88 or counts["original_strong_failures"] != 28:
        counts["expected_88_28_gate"] = False
    else:
        counts["expected_88_28_gate"] = True
    return subset, counts


def strong_path_metrics(subset: Sequence[Mapping[str, Any]], rows_by_variant: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        joined = [{**dict(mask), "repair_correct": as_bool(rows_by_variant[variant][str(mask["case_id"])].get("final_correct"))} for mask in subset]
        ready = [row for row in joined if row["strong_path_ready"]]
        failures = [row for row in joined if row["original_strong_failure"]]
        successes = [row for row in joined if row["original_correct"]]
        rescue_bootstrap = family_bootstrap_ratio(
            joined,
            lambda sample: sum(row["original_strong_failure"] and row["repair_correct"] for row in sample),
            lambda sample: sum(row["original_strong_failure"] for row in sample),
        )
        row = {"variant": variant, "variant_label": VARIANT_LABELS[variant], "strong_path_ready_N": len(ready), "strong_path_ready_final_failures": sum(not row["repair_correct"] for row in ready), "strong_path_ready_final_failure_rate": (sum(not row["repair_correct"] for row in ready) / len(ready) if ready else None), "original_strong_failures_N": len(failures), "rescued_original_strong_failures": sum(row["repair_correct"] for row in failures), "strong_failure_rescue_rate": (sum(row["repair_correct"] for row in failures) / len(failures) if failures else None), "original_successes_N": len(successes), "preserved_original_successes": sum(row["repair_correct"] for row in successes), "original_success_preservation_rate": (sum(row["repair_correct"] for row in successes) / len(successes) if successes else None)}
        row["strong_failure_rescue_bootstrap"] = rescue_bootstrap
        if rescue_bootstrap["confidence_interval_95"] is not None:
            row["strong_failure_rescue_ci_low"] = rescue_bootstrap["confidence_interval_95"][0]
            row["strong_failure_rescue_ci_high"] = rescue_bootstrap["confidence_interval_95"][1]
        else:
            row["strong_failure_rescue_ci_low"] = None
            row["strong_failure_rescue_ci_high"] = None
        output.append(row)
        summary[variant] = row
    return output, summary


def write_failure_breakdown(path: Path, rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for variant, rows_for_variant in rows_by_variant.items():
        if variant in {"original", "blur_only"}:
            continue
        counts = Counter(str(row.get("failure_category")) for row in rows_for_variant)
        n = len(rows_for_variant)
        pipeline_failures = sum(value for key, value in counts.items() if key != "BIND_CORRECT_ATTR_CORRECT")
        for category in ("BIND_CORRECT_ATTR_CORRECT", "BIND_CORRECT_ATTR_WRONG", "BIND_WRONG_ATTR_CORRECT_FOR_WRONG_ENTITY", "BIND_WRONG_ATTR_OTHER", "BIND_INVALID"):
            count = counts[category]
            rows.append({"variant": variant, "variant_label": VARIANT_LABELS[variant], "category": category, "count": count, "N": n, "percent_of_all_cases": count / n if n else None, "percent_of_pipeline_failures": (count / pipeline_failures if pipeline_failures else None) if category != "BIND_CORRECT_ATTR_CORRECT" else None})
    write_csv(path, rows, ["variant", "variant_label", "category", "count", "N", "percent_of_all_cases", "percent_of_pipeline_failures"])


def load_visual_interference_cases(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_path = root / "selected_cases.jsonl"
    cases_path = root / "visual_interference_cases.jsonl"
    if not selected_path.is_file() or not cases_path.is_file():
        raise ExperimentInvalid(f"visual-interference pilot files are missing under {root}")
    selected = read_jsonl(selected_path)
    interference = read_jsonl(cases_path)
    if len(selected) != 56 or len(interference) != 11:
        raise ExperimentInvalid(f"visual-interference pilot counts are {len(selected)} selected / {len(interference)} interference; expected 56/11")
    wrong_target = [row for row in interference if str(row.get("vision_facts_prediction")) == str(row.get("color_A")) and str(row.get("color_A")) != str(row.get("target_color"))]
    reversals = [row for row in interference if as_bool(row.get("failure_rescue_reversed_by_vision"))]
    if len(wrong_target) != 9 or len(reversals) != 6:
        raise ExperimentInvalid(f"visual-interference category counts are wrong-target={len(wrong_target)}, reversal={len(reversals)}; expected 9/6")
    return interference, {"selected_56_count": len(selected), "visual_interference_count": len(interference), "wrong_target_color_count": len(wrong_target), "rescue_reversal_count": len(reversals), "source_selected_sha256": sha256_file(selected_path), "source_interference_sha256": sha256_file(cases_path)}


def write_visual_interference_repair(path: Path, visual_cases: Sequence[Mapping[str, Any]], rows_by_variant: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in visual_cases:
        row: dict[str, Any] = {"case_id": item["case_id"], "family_id": item["family_id"], "qa_type": item["qa_type"], "target_identity": item["target_identity"], "target_color": item["target_color"], "color_A": item["color_A"], "color_B": item["color_B"], "original_subset": item["original_subset"], "visual_interference": True, "wrong_target_color_case": str(item.get("vision_facts_prediction")) == str(item.get("color_A")) and str(item.get("color_A")) != str(item.get("target_color")), "rescue_reversal_case": as_bool(item.get("failure_rescue_reversed_by_vision")), "historical_original_prediction": item.get("original_prediction"), "historical_original_correct": as_bool(item.get("original_correct"))}
        for variant in VARIANTS:
            result = rows_by_variant[variant][str(item["case_id"])]
            prefix = variant
            row[f"{prefix}_prediction"] = result.get("final_prediction")
            row[f"{prefix}_correct"] = as_bool(result.get("final_correct"))
            row[f"{prefix}_invalid"] = as_bool(result.get("final_invalid"))
            row[f"{prefix}_stage1_selected_identity_label"] = result.get("stage1_selected_identity_label")
            row[f"{prefix}_stage1_correct"] = result.get("stage1_correct")
        rows.append(row)
    fields = list(rows[0].keys()) if rows else []
    write_csv(path, rows, fields)
    summary: dict[str, Any] = {"N": len(rows), "wrong_target_color_N": sum(row["wrong_target_color_case"] for row in rows), "rescue_reversal_N": sum(row["rescue_reversal_case"] for row in rows), "by_variant": {}}
    for variant in VARIANTS:
        summary["by_variant"][variant] = {"all_visual_interference": {"N": len(rows), "correct": sum(row[f"{variant}_correct"] for row in rows), "accuracy": sum(row[f"{variant}_correct"] for row in rows) / len(rows) if rows else None}, "wrong_target_color": {"N": sum(row["wrong_target_color_case"] for row in rows), "correct": sum(row[f"{variant}_correct"] and row["wrong_target_color_case"] for row in rows), "rescue_count": sum(row[f"{variant}_correct"] and row["wrong_target_color_case"] for row in rows)}, "rescue_reversal": {"N": sum(row["rescue_reversal_case"] for row in rows), "correct": sum(row[f"{variant}_correct"] and row["rescue_reversal_case"] for row in rows)}}
    return summary


def compare_rows(rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]], cases: Sequence[Mapping[str, Any]], historical_by_qa: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexes = {variant: {str(row["case_id"]): row for row in rows} for variant, rows in rows_by_variant.items()}
    comparison: list[dict[str, Any]] = []
    for case in cases:
        cid = str(case["case_id"])
        row = {"case_id": cid, "family_id": case["family_id"], "composition_qa_type": case["composition_qa_type"], "target_side": case["target_side"], "target_identity": case["target_identity"], "target_color": case["target_color"], "object_A": case["object_A"], "object_B": case["object_B"], "color_A": case["color_A"], "color_B": case["color_B"]}
        old = historical_by_qa[cid]
        row.update({"historical_original_prediction": old.get("normalized_prediction"), "historical_original_correct": as_bool(old.get("correct"))})
        for variant in VARIANTS:
            result = indexes[variant][cid]
            prefix = variant
            row[f"{prefix}_prediction"] = result.get("final_prediction")
            row[f"{prefix}_correct"] = as_bool(result.get("final_correct"))
            row[f"{prefix}_invalid"] = as_bool(result.get("final_invalid"))
            row[f"{prefix}_format_violation"] = as_bool(result.get("final_format_violation"))
            row[f"{prefix}_truncation"] = as_bool(result.get("final_semantic_truncation"))
            if variant not in {"original", "blur_only"}:
                for field in ("stage1_raw", "stage1_normalized", "stage1_selected_identity", "stage1_selected_identity_label", "stage1_correct", "stage1_valid", "stage2_raw", "stage2_prediction", "stage2_correct_for_selected_identity", "stage2_correct_for_composition_GT", "failure_category", "wrong_binding_same_color_accidentally_correct"):
                    row[f"{prefix}_{field}"] = result.get(field)
        comparison.append(row)
    return comparison


def signal_status(summary: Mapping[str, Any], paired: Mapping[str, Any], strong: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    full = summary["variants"]["full_dah"]["pooled"]["accuracy"]
    original = summary["variants"]["original"]["pooled"]["accuracy"]
    delta = paired["full_dah"]["bootstrap"]["confidence_interval_95"]
    rescue = strong["full_dah"]["strong_failure_rescue_rate"]
    # This fixed descriptive label is applied only after all rows are frozen;
    # it cannot alter selection, data, inference, or tuning.
    if full is not None and original is not None and full > original and delta[0] > 0 and rescue is not None and rescue >= 0.5:
        return "DAH_STRONG_SIGNAL", {"rule": "Full DaH observed accuracy exceeds Original, family-bootstrap delta CI lower bound > 0, and strong-failure rescue >= .5"}
    if full is not None and original is not None and full > original and (delta[0] > 0 or (rescue is not None and rescue > 0)):
        return "DAH_MODERATE_SIGNAL", {"rule": "Full DaH observed accuracy exceeds Original and has either positive bootstrap lower bound or nonzero strong-failure rescue"}
    return "DAH_WEAK_SIGNAL", {"rule": "The fixed strong/moderate criteria are not met"}


def make_summary(
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    cases: Sequence[Mapping[str, Any]],
    historical: Mapping[str, Any],
    historical_audit_payload: Mapping[str, Any],
    strong_masks: Sequence[Mapping[str, Any]],
    strong_counts: Mapping[str, Any],
    visual_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = {variant: {str(row["case_id"]): row for row in rows} for variant, rows in rows_by_variant.items()}
    original_by_case = rows_by_id["original"]
    variants_summary: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        pooled = metric(rows)
        variants_summary[variant] = {"label": VARIANT_LABELS[variant], "pooled": pooled, "by_composition_type": side_metrics(rows), "decomposed": decomposed_metrics(rows) if variant not in {"original", "blur_only"} else None}
        if variant != "original":
            repaired = [as_bool(row.get("final_correct")) for row in rows]
            original = [as_bool(original_by_case[str(row["case_id"])].get("final_correct")) for row in rows]
            paired[variant] = {"mcnemar_exact": exact_mcnemar(original, repaired), "bootstrap": family_bootstrap_delta(rows, original_by_case)}
    strong_rows, strong_summary = strong_path_metrics(strong_masks, rows_by_id)
    status, status_rule = signal_status({"variants": variants_summary}, paired, strong_summary)
    ablation_pairs = (
        ("original_vs_blur_only", "original", "blur_only", "background/non-candidate defocus"),
        ("original_vs_decompose_only", "original", "decompose_only", "two-stage decomposition and explicit handoff"),
        ("decompose_only_vs_handoff_targetlock", "decompose_only", "handoff_targetlock", "target visual restriction"),
        ("handoff_targetlock_vs_full_dah", "handoff_targetlock", "full_dah", "pair-defocus Stage 1"),
        ("blur_only_vs_full_dah", "blur_only", "full_dah", "full method versus pair blur"),
    )
    ablations: dict[str, Any] = {}
    for key, baseline_variant, comparison_variant, question in ablation_pairs:
        baseline_accuracy = variants_summary[baseline_variant]["pooled"]["accuracy"]
        comparison_accuracy = variants_summary[comparison_variant]["pooled"]["accuracy"]
        ablations[key] = {"baseline_variant": baseline_variant, "comparison_variant": comparison_variant, "question": question, "baseline_accuracy": baseline_accuracy, "comparison_accuracy": comparison_accuracy, "delta_comparison_minus_baseline": comparison_accuracy - baseline_accuracy if baseline_accuracy is not None and comparison_accuracy is not None else None}
    summary = {"schema_version": f"{EXPERIMENT_SCHEMA}_summary_v1", "experiment": "REALBIND DaH ORACLE PILOT", "status": status, "status_rule": status_rule, "data": {"families": EXPECTED_FAMILIES, "composition_cases": EXPECTED_CASES, "left_cases": EXPECTED_PER_COMPOSITION, "right_cases": EXPECTED_PER_COMPOSITION}, "variants": variants_summary, "decomposed": {variant: variants_summary[variant]["decomposed"] for variant in ("decompose_only", "handoff_targetlock", "full_dah")}, "ablation_comparisons": ablations, "historical_baseline": historical, "historical_vs_rerun_audit": historical_audit_payload, "original_strong_path": strong_counts, "strong_path_repair": strong_summary, "visual_interference_repair": visual_summary, "error_source_full_dah": {}, "paired_statistics": paired, "sanity": {"same_model_and_generation_config": True, "stage2_fresh_context": True, "stage2_no_stage1_history": True, "gt_bbox_never_used_to_select_target": True, "frozen_rel2i_questions_reused": True, "frozen_i2c_questions_reused": True, "no_thinking": True, "no_training": True, "no_attention_intervention": True}}
    full_rows = rows_by_variant["full_dah"]
    full_failures = [row for row in full_rows if not as_bool(row.get("final_correct"))]
    binding_failures = [row for row in full_failures if not as_bool(row.get("stage1_correct"))]
    attr_failures = [row for row in full_failures if as_bool(row.get("stage1_correct"))]
    denominator = len(full_failures)
    summary["error_source_full_dah"] = {"full_dah_failures": denominator, "failure_due_to_binding_count": len(binding_failures), "failure_due_to_binding_percent_of_full_dah_failures": len(binding_failures) / denominator if denominator else None, "failure_due_to_attribute_given_correct_binding_count": len(attr_failures), "failure_due_to_attribute_given_correct_binding_percent_of_full_dah_failures": len(attr_failures) / denominator if denominator else None, "failure_due_to_binding_percent_of_all_cases": len(binding_failures) / EXPECTED_CASES, "failure_due_to_attribute_given_correct_binding_percent_of_all_cases": len(attr_failures) / EXPECTED_CASES}
    failure_rows = []
    for variant in ("decompose_only", "handoff_targetlock", "full_dah"):
        failure_rows.extend(add_failure_categories(rows_by_variant[variant]))
    return summary, paired, strong_rows, failure_rows


def format_rate(numerator: Any, denominator: Any) -> str:
    if denominator in (None, 0):
        return "NA"
    return f"{float(numerator) / float(denominator):.6f}"


def format_optional_float(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def build_summary_md(summary: Mapping[str, Any], strong_rows: Sequence[Mapping[str, Any]]) -> str:
    original_acc = summary["variants"]["original"]["pooled"]["accuracy"]
    lines = ["# RealBind DaH oracle pilot", "", f"Status: **{summary['status']}**", "", "## Primary comparison", "", "| Variant | Left | Right | Composition | Δ vs Original | McNemar p |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        item = summary["variants"][variant]
        left = item["by_composition_type"]["REL2C_LEFT"]["accuracy"]
        right = item["by_composition_type"]["REL2C_RIGHT"]["accuracy"]
        pooled = item["pooled"]["accuracy"]
        delta = None if pooled is None or original_acc is None else pooled - original_acc
        p = None if variant == "original" else summary["paired_statistics"][variant]["mcnemar_exact"]["p_exact_two_sided"]
        lines.append(f"| {item['label']} | {left:.6f} | {right:.6f} | {pooled:.6f} | {'NA' if delta is None else f'{delta:+.6f}'} | {'NA' if p is None else f'{p:.6g}'} |")
    lines.extend(["", "Primary generation-quality counts (pooled):", "", "| Variant | Invalid | Format violation | Truncation | Hit max tokens |", "| --- | ---: | ---: | ---: | ---: |"])
    for variant in VARIANTS:
        m = summary["variants"][variant]["pooled"]
        lines.append(f"| {summary['variants'][variant]['label']} | {m['invalid_count']} | {m['format_violation_count']} | {m['truncation_count']} | {m['hit_max_new_tokens_count']} |")
    lines.extend(["", "## Ablation comparisons", "", "Positive Δ means the comparison variant is more accurate on the same 200 cases.", "", "| Comparison | Baseline variant | Comparison variant | Baseline | Comparison | Δ accuracy |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for key in ("original_vs_blur_only", "original_vs_decompose_only", "decompose_only_vs_handoff_targetlock", "handoff_targetlock_vs_full_dah", "blur_only_vs_full_dah"):
        item = summary["ablation_comparisons"][key]
        lines.append(f"| {item['question']} | {VARIANT_LABELS[item['baseline_variant']]} | {VARIANT_LABELS[item['comparison_variant']]} | {item['baseline_accuracy']:.6f} | {item['comparison_accuracy']:.6f} | {item['delta_comparison_minus_baseline']:+.6f} |")
    lines.extend(["", "## Decomposed pipelines", "", "| Variant | Bind Acc | Attr Acc given correct bind | Final Acc |", "| --- | ---: | ---: | ---: |"])
    for variant in ("decompose_only", "handoff_targetlock", "full_dah"):
        d = summary["variants"][variant]["decomposed"]
        attr_rate = d["stage2_attribute_accuracy_given_binding_correct"]["rate"]
        attr_text = "NA" if attr_rate is None else f"{attr_rate:.6f}"
        lines.append(f"| {summary['variants'][variant]['label']} | {d['stage1_binding_accuracy']['rate']:.6f} | {attr_text} | {d['final_composition_accuracy']['rate']:.6f} |")
    lines.extend(["", "Wrong-binding final-correct diagnostics (not counted as binding success):", "", "| Variant | Wrong bind but final correct | Of which same-color accidental |", "| --- | ---: | ---: |"])
    for variant in ("decompose_only", "handoff_targetlock", "full_dah"):
        d = summary["variants"][variant]["decomposed"]
        lines.append(f"| {summary['variants'][variant]['label']} | {d['wrong_binding_final_correct_count']} | {d['wrong_binding_same_color_accidentally_correct_count']} |")
    lines.extend(["", "## Strong-path repair", "", "| Variant | Strong Failure Rescue | Original Success Preservation | Failure on original strong-ready |", "| --- | ---: | ---: | ---: |"])
    for row in strong_rows:
        rescue_ci = row["strong_failure_rescue_bootstrap"]["confidence_interval_95"]
        rescue_text = f"{format_rate(row['rescued_original_strong_failures'], row['original_strong_failures_N'])} ({row['rescued_original_strong_failures']}/{row['original_strong_failures_N']}; CI [{rescue_ci[0]:.6f},{rescue_ci[1]:.6f}])" if rescue_ci is not None else "NA"
        lines.append(f"| {row['variant_label']} | {rescue_text} | {format_rate(row['preserved_original_successes'], row['original_successes_N'])} ({row['preserved_original_successes']}/{row['original_successes_N']}) | {format_rate(row['strong_path_ready_final_failures'], row['strong_path_ready_N'])} ({row['strong_path_ready_final_failures']}/{row['strong_path_ready_N']}) |")
    error = summary["error_source_full_dah"]
    lines.extend(["", "## Full DaH failure source", "", f"- Full DaH failures: {error['full_dah_failures']}/{EXPECTED_CASES}.", f"- Failure due to binding: {error['failure_due_to_binding_count']} ({format_optional_float(error['failure_due_to_binding_percent_of_full_dah_failures'])} of Full DaH failures).", f"- Failure due to attribute after correct binding: {error['failure_due_to_attribute_given_correct_binding_count']} ({format_optional_float(error['failure_due_to_attribute_given_correct_binding_percent_of_full_dah_failures'])} of Full DaH failures).", "", "## Visual-interference subset", "", f"- Source selected set: {summary['visual_interference_repair']['selected_56_count']} cases; visual-interference rows: {summary['visual_interference_repair']['N']}; wrong-target-color rows: {summary['visual_interference_repair']['wrong_target_color_N']}; rescue-reversal rows: {summary['visual_interference_repair']['rescue_reversal_N']}."])
    for variant in VARIANTS:
        item = summary["visual_interference_repair"]["by_variant"][variant]
        if variant == "full_dah":
            lines.append(f"- Full DaH rescued among 9 wrong-target-color cases: {item['wrong_target_color']['rescue_count']}/{item['wrong_target_color']['N']}.")
    lines.extend(["", "## Historical baseline audit", "", f"- Raw output agreement: {summary['historical_vs_rerun_audit']['raw_response_agreement_count']}/{summary['historical_vs_rerun_audit']['N']}.", f"- Normalized prediction agreement: {summary['historical_vs_rerun_audit']['normalized_prediction_agreement_count']}/{summary['historical_vs_rerun_audit']['N']}.", f"- Correctness agreement: {summary['historical_vs_rerun_audit']['correctness_agreement_count']}/{summary['historical_vs_rerun_audit']['N']}.", "", "## Paired bootstrap", "", "All repair-vs-Original deltas use the same 200 cases, family-cluster bootstrap, seed 42, and 10,000 replicates. McNemar is exact two-sided."])
    for variant in ("blur_only", "decompose_only", "handoff_targetlock", "full_dah"):
        boot = summary["paired_statistics"][variant]["bootstrap"]
        lines.append(f"- {summary['variants'][variant]['label']}: Δ={boot['observed_delta_accuracy']:+.6f}, 95% CI=[{boot['confidence_interval_95'][0]:+.6f}, {boot['confidence_interval_95'][1]:+.6f}].")
    lines.extend(["", "## Fixed interpretation rule", "", summary["status_rule"]["rule"], "", "The label is descriptive and was applied after inference; it did not alter data, prompts, target selection, scoring, or tuning.", ""])
    return "\n".join(lines)


def write_terminal_summary(summary: Mapping[str, Any]) -> None:
    print("REALBIND DaH ORACLE PILOT")
    print("                LEFT    RIGHT    FINAL")
    for variant in VARIANTS:
        item = summary["variants"][variant]
        left = item["by_composition_type"]["REL2C_LEFT"]["accuracy"]
        right = item["by_composition_type"]["REL2C_RIGHT"]["accuracy"]
        final = item["pooled"]["accuracy"]
        print(f"{item['label']:<24}{left:>7.6f} {right:>8.6f} {final:>8.6f}")
    print("\nDECOMPOSED PIPELINES:")
    print("                      Bind Acc    Attr|CorrectBind    Final")
    for variant in ("decompose_only", "handoff_targetlock", "full_dah"):
        d = summary["variants"][variant]["decomposed"]
        attr = d["stage2_attribute_accuracy_given_binding_correct"]["rate"]
        print(f"{summary['variants'][variant]['label']:<24}{d['stage1_binding_accuracy']['rate']:>8.6f} {('NA' if attr is None else f'{attr:.6f}'):>18} {d['final_composition_accuracy']['rate']:>8.6f}")
    print("\nSTRONG FAILURE REPAIR:")
    print("                      Rescue    Preservation")
    for variant in ("blur_only", "decompose_only", "handoff_targetlock", "full_dah"):
        row = summary["strong_path_repair"][variant]
        print(f"{summary['variants'][variant]['label']:<24}{format_rate(row['rescued_original_strong_failures'], row['original_strong_failures_N']):>8} {format_rate(row['preserved_original_successes'], row['original_successes_N']):>14}")
    error = summary["error_source_full_dah"]
    print("\nFULL DaH ERROR SOURCE:")
    print(f"Binding failures = {error['failure_due_to_binding_count']}/{error['full_dah_failures']} ({error['failure_due_to_binding_percent_of_full_dah_failures']})")
    print(f"Attribute failures after correct binding = {error['failure_due_to_attribute_given_correct_binding_count']}/{error['full_dah_failures']} ({error['failure_due_to_attribute_given_correct_binding_percent_of_full_dah_failures']})")
    visual = summary["visual_interference_repair"]["by_variant"]["full_dah"]["wrong_target_color"]
    print("\nVISUAL-INTERFERENCE CASES:")
    print(f"9 wrong-target-color cases\nFull DaH rescued = {visual['rescue_count']}/{visual['N']}")
    print(f"\n{summary['status']}")
    print("FULL RAW METRICS:")
    print(json.dumps({"variants": summary["variants"], "strong_path_repair": summary["strong_path_repair"], "error_source_full_dah": summary["error_source_full_dah"], "historical_vs_rerun_audit": summary["historical_vs_rerun_audit"]}, ensure_ascii=False, indent=2, sort_keys=True))


def analyze(args: argparse.Namespace, paths: Mapping[str, Path], input_paths: Mapping[str, Path]) -> dict[str, Any]:
    families, qa_rows, _dependency, _audit = validate_frozen_schema(input_paths)
    if not paths["config"].is_file():
        raise ExperimentInvalid("config.json is missing; run --prepare-images first")
    config = read_json(paths["config"])
    check_current_frozen_hashes(config, input_paths)
    cases = build_cases(families, qa_rows)
    historical, historical_by_qa, historical_by_family = load_historical_baseline(args.historical_root.expanduser().resolve(), input_paths)
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        final_path = paths[variant]
        if not final_path.is_file():
            raise ExperimentInvalid(f"missing completed variant result: {final_path}")
        rows = load_checkpoint(final_path, {str(case["case_id"]) for case in cases}, config, variant)
        if len(rows) != EXPECTED_CASES:
            raise ExperimentInvalid(f"{variant} result count={len(rows)}")
        ordered = [rows[str(case["case_id"])] for case in cases]
        rows_by_variant[variant] = add_failure_categories(ordered)
    original_audit = historical_audit(cases, rows_by_variant["original"], historical_by_qa)
    strong_masks, strong_counts = original_strong_path(cases, historical_by_qa, historical_by_family)
    visual_cases, visual_source_summary = load_visual_interference_cases(args.visual_pilot_root.expanduser().resolve())
    rows_indexes = {variant: {str(row["case_id"]): row for row in rows} for variant, rows in rows_by_variant.items()}
    visual_summary = write_visual_interference_repair(paths["visual_repair"], visual_cases, rows_indexes)
    visual_summary.update(visual_source_summary)
    summary, paired, strong_rows, failure_rows = make_summary(rows_by_variant, cases, historical, original_audit, strong_masks, strong_counts, visual_summary)
    comparison = compare_rows(rows_by_variant, cases, historical_by_qa)
    write_jsonl_atomic(paths["comparison"], comparison)
    write_failure_breakdown(paths["failure_breakdown"], rows_by_variant)
    write_csv(paths["strong_path"], strong_rows, list(strong_rows[0].keys()) if strong_rows else [])
    write_json(paths["paired_stats"], paired)
    write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(build_summary_md(summary, strong_rows), encoding="utf-8")
    if paths["provenance"].is_file():
        provenance = read_json(paths["provenance"])
    else:
        provenance = {}
    provenance.update({"analysis_completed_utc": datetime.now(timezone.utc).isoformat(), "historical_baseline": historical, "historical_vs_rerun_audit": original_audit, "original_strong_path": strong_counts, "final_status": summary["status"], "raw_outputs": {variant: str(paths[variant].resolve()) for variant in VARIANTS}})
    write_json(paths["provenance"], provenance)
    write_terminal_summary(summary)
    return summary


def run_all(args: argparse.Namespace, paths: Mapping[str, Path], input_paths: Mapping[str, Path]) -> None:
    families, qa_rows, cases, config, _audit = prepare_protocol(args, paths, input_paths)
    check_current_frozen_hashes(config, input_paths)
    transform_payload = transform_index(paths)
    semantic_vocabulary = build_semantic_vocabulary(families)
    families_by_id, _typed = family_maps(families, qa_rows)
    cache = load_cache(paths["cache"], config)
    processor, model, device = load_model_once()
    try:
        requested = list(VARIANTS) if args.variant == "all" else [args.variant]
        if any(value in requested for value in ("decompose_only", "handoff_targetlock")):
            requested_stage1_original = True
        else:
            requested_stage1_original = False
        if "original" in requested:
            run_direct_variant(cases, families_by_id, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["original"], "original")
        if "blur_only" in requested:
            run_direct_variant(cases, families_by_id, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["blur_only"], "blur_only")
        shared_original: list[dict[str, Any]] | None = None
        if requested_stage1_original:
            shared_original = run_stage1(cases, families_by_id, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["shared_original_rel2i"], "original", "shared_original_rel2i_stage1")
        if "handoff_targetlock" in requested:
            if shared_original is None:
                shared_original = load_jsonl(paths["shared_original_rel2i"])
            binding = {str(row["case_id"]): row for row in shared_original}
            rows = run_decomposed_variant(cases, families_by_id, binding, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["handoff_targetlock"], "handoff_targetlock", "shared_original_rel2i_stage1", True)
            # Keep the completed rows available for all-mode analysis but do not
            # overwrite a raw output with a transformed copy.
            _ = rows
        if "decompose_only" in requested:
            if shared_original is None:
                shared_original = load_jsonl(paths["shared_original_rel2i"])
            binding = {str(row["case_id"]): row for row in shared_original}
            run_decomposed_variant(cases, families_by_id, binding, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["decompose_only"], "decompose_only", "shared_original_rel2i_stage1", False)
        if "full_dah" in requested:
            pair_binding = run_stage1(cases, families_by_id, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["full_dah_rel2i"], "pair_defocus", "full_dah_pair_rel2i_stage1")
            binding = {str(row["case_id"]): row for row in pair_binding}
            run_decomposed_variant(cases, families_by_id, binding, processor, model, device, config, semantic_vocabulary, transform_payload, cache, paths["cache"], paths["full_dah"], "full_dah", "full_dah_pair_rel2i_stage1", True)
    finally:
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExperimentInvalid(f"missing checkpoint: {path}")
    return read_jsonl(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-images", action="store_true", help="audit frozen inputs and build deterministic PNG caches/HTML")
    parser.add_argument("--run", action="store_true", help="run requested variants; existing checkpoints are resumed safely")
    parser.add_argument("--resume", action="store_true", help="resume pending raw calls from atomic checkpoints")
    parser.add_argument("--analyze", action="store_true", help="run offline scoring/statistics after all five outputs complete")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--dependency", type=Path, default=DEFAULT_DEPENDENCY)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--visual-pilot-root", type=Path, default=DEFAULT_VISUAL_PILOT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not any((args.prepare_images, args.run, args.resume, args.analyze)):
        parser.error("select at least one of --prepare-images, --run, --resume, --analyze")
    if args.analyze and args.variant != "all":
        parser.error("--analyze requires --variant all so the five-way table is complete")
    return args


def main() -> int:
    args = parse_args()
    paths = output_paths(args.output_root.expanduser().resolve())
    ensure_dirs(paths)
    input_paths = required_input_paths(args)
    try:
        # Preparation is idempotent and verifies existing derived PNG bytes;
        # --run also performs it so a durable resume cannot use an un-audited
        # image cache.
        if args.prepare_images:
            # Preparation is an offline audit/image-cache operation and must
            # not load the VLM.
            prepare_protocol(args, paths, input_paths)
        if args.run or args.resume:
            run_all(args, paths, input_paths)
        if args.analyze:
            analyze(args, paths, input_paths)
        return 0
    except ExperimentInvalid as exc:
        payload = {"schema_version": f"{EXPERIMENT_SCHEMA}_invalid_v1", "status": "EXPERIMENT_INVALID", "message": str(exc), "failures": exc.failures, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git": git_state()}
        try:
            write_json(paths["invalid"], payload)
        except Exception:
            pass
        print("EXPERIMENT_INVALID", file=sys.stderr)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
