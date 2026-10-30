#!/usr/bin/env python3
"""Run the paired RealBind2 facts-only vs image-plus-facts experiment.

This runner is model-agnostic: it uses the adapter registry in ``src.models``
(``qwen35``, ``qwen3vl``, ``internvl35``, ``llava_next``) or a user-provided
plugin class.  Each case produces exactly two generations sharing byte-identical
prompt text; only the image content block differs.

Typical use (from the packaged cohort root)::

    python scripts/run_facts_image.py \
      --config configs/qwen35_4b.yaml \
      --paths configs/paths.yaml \
      --output outputs/qwen35_4b/results.jsonl

Resume is default.  Re-running the same command skips complete
(case_id, condition) records and retries failed/missing pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_framework_root() -> Path:
    """Locate a directory containing the ``src`` package.

    Works both in the source repository (``realbind2-vlm``) and in the packaged
    cohort (which ships a minimal copy of ``src``).
    """
    candidates = [
        SCRIPT_DIR.parent,
        SCRIPT_DIR.parent.parent,
        SCRIPT_DIR.parent.parent.parent,
        SCRIPT_DIR.parent.parent.parent.parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "src" / "models" / "__init__.py").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "cannot locate the RealBind2 framework root containing src/models; "
        "run from the package root or set PYTHONPATH to the realbind2-vlm repository"
    )


FRAMEWORK_ROOT = _find_framework_root()
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
# Allow custom adapters placed next to the runner to be referenced by module name.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.common.scorer import SCORER_VERSION, score_prediction  # noqa: E402
from src.models import ADAPTERS  # noqa: E402

CONDITIONS = ("facts_only", "image_plus_facts")
FORBIDDEN_FACT_TOKENS = (
    "therefore",
    "the answer is",
    "the left one is",
    "the right one is",
    "the left object is",
    "the right object is",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def resolve_adapter(config: Mapping[str, Any], paths: Mapping[str, Any]) -> Any:
    runtime = dict(config.get("runtime") or {})
    adapter_key = str(config.get("adapter", "")).strip()
    if not adapter_key:
        raise ValueError("model config is missing 'adapter'")

    if adapter_key == "custom":
        entrypoint = str(config.get("adapter_class", "")).strip()
        if "." not in entrypoint:
            raise ValueError("custom adapter requires config.adapter_class='module.ClassName'")
        module_name, class_name = entrypoint.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    else:
        try:
            cls = ADAPTERS[adapter_key]
        except KeyError as exc:
            raise ValueError(f"unknown adapter {adapter_key!r}; available={sorted(ADAPTERS)}") from exc

    path_key = str(config.get("path_key", "")).strip()
    if not path_key:
        raise ValueError("model config is missing 'path_key'")
    model_path = Path(str((paths.get("models") or {})[path_key])).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist for {path_key}: {model_path}")
    return cls(model_path, **runtime)


def build_record(
    *,
    case: Mapping[str, Any],
    condition: str,
    raw: str,
    scored: Any,
    config: Mapping[str, Any],
    config_hash: str,
    cases_sha256: str,
    runtime_sec: float,
) -> dict[str, Any]:
    competitor_color = str(case["competitor_color"]).casefold()
    normalized = scored.normalized
    if scored.correct:
        error_type = "correct"
    elif normalized is None:
        error_type = "invalid_or_ambiguous"
    elif str(normalized).casefold() == competitor_color:
        error_type = "other_object_color"
    else:
        error_type = "third_color"
    return {
        "schema_version": "realbind2_facts_image_500_result_v1",
        "record_id": f"{case['case_id']}::{condition}",
        "case_id": str(case["case_id"]),
        "family_id": str(case["family_id"]),
        "image_id": str(case["image_id"]),
        "qa_id": str(case["qa_id"]),
        "qa_type": str(case["qa_type"]),
        "target_side": str(case["target_side"]),
        "split": str(case["split"]),
        "condition": condition,
        "image_present": condition == "image_plus_facts",
        "image_file": str(case["image_file"]),
        "image_sha256_expected": str(case["image_sha256"]),
        "prompt_text": str(case["text_prompt"]),
        "prompt_sha256": str(case["text_prompt_sha256"]),
        "prediction_raw": str(raw),
        "prediction_normalized": normalized,
        "ground_truth": str(scored.ground_truth),
        "target_color": str(case["target_color"]),
        "competitor_color": str(case["competitor_color"]),
        "correct": bool(scored.correct),
        "valid": bool(scored.valid),
        "scorer_error": scored.error,
        "error_type": error_type,
        "model_id": str(config.get("model_id", "")),
        "adapter": str(config.get("adapter", "")),
        "runtime": dict(config.get("runtime") or {}),
        "model_metadata": None,
        "config_hash": config_hash,
        "cases_sha256": cases_sha256,
        "scorer_version": SCORER_VERSION,
        "runtime_sec": float(runtime_sec),
        "status": "ok",
    }


def resolve_image_path(images_root: Path, image_file: str) -> Path:
    """Accept either the package root or its images/ directory."""
    relative = Path(str(image_file))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image_file: {image_file}")
    if images_root.name == "images":
        return images_root / relative.name
    return images_root / relative


def preflight(cases: list[dict[str, Any]], images_root: Path, *, require_images: bool) -> dict[str, Any]:
    errors: list[str] = []
    case_ids: set[str] = set()
    image_ids: set[str] = set()
    for row in cases:
        cid = str(row.get("case_id", ""))
        if not cid or cid in case_ids:
            errors.append(f"duplicate_or_empty_case_id:{cid}")
        case_ids.add(cid)
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in image_ids:
            errors.append(f"duplicate_or_empty_image_id:{image_id}")
        image_ids.add(image_id)
        facts = row.get("facts")
        text = str(row.get("text_prompt", ""))
        if not isinstance(facts, list) or len(facts) != 3:
            errors.append(f"fact_count:{cid}")
        else:
            joined = "\n".join(str(fact) for fact in facts).casefold()
            if any(token in joined for token in FORBIDDEN_FACT_TOKENS):
                errors.append(f"derived_answer_leak:{cid}")
            reconstructed = "\n".join(str(fact) for fact in facts) + "\n\n" + str(row.get("question", ""))
            if reconstructed != text:
                errors.append(f"prompt_reconstruction:{cid}")
        actual_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_text_hash != str(row.get("text_prompt_sha256", "")):
            errors.append(f"text_hash:{cid}")
        if str(row.get("target_color", "")) != str(row.get("ground_truth", "")):
            errors.append(f"target_color_mismatch:{cid}")
        contract = row.get("condition_contract") or {}
        f_contract = contract.get("facts_only") or {}
        i_contract = contract.get("image_plus_facts") or {}
        if f_contract.get("text_sha256") != actual_text_hash or i_contract.get("text_sha256") != actual_text_hash:
            errors.append(f"condition_contract_hash:{cid}")
        if bool(f_contract.get("image_present", True)) or not bool(i_contract.get("image_present", False)):
            errors.append(f"condition_image_contract:{cid}")
        image_path = resolve_image_path(images_root, str(row.get("image_file", "")))
        if not image_path.is_file():
            if require_images:
                errors.append(f"missing_image:{cid}:{image_path}")
        elif "image_sha256" in row and sha256_file(image_path) != str(row["image_sha256"]):
            errors.append(f"image_hash:{cid}")
    if errors:
        raise ValueError("preflight failed:\n" + "\n".join(errors[:100]))
    return {
        "passed": True,
        "cases": len(cases),
        "unique_case_ids": len(case_ids),
        "unique_image_ids": len(image_ids),
        "images_checked": require_images,
    }


def select_cases(cases: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard arguments")
    ordered = sorted(cases, key=lambda row: str(row["case_id"]))
    return [row for index, row in enumerate(ordered) if index % num_shards == shard_index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="audit cases and print the planned contract without loading a model")
    parser.add_argument("--no-resume", action="store_true", help="refuse to reuse an existing output log")
    parser.add_argument("--ignore-image-hash", action="store_true", help="skip image sha256 checks")
    args = parser.parse_args()

    cases_path = args.cases
    if cases_path is None:
        candidates = [
            SCRIPT_DIR.parent / "cases.jsonl",
            FRAMEWORK_ROOT / "experiments" / "facts_image_500" / "output" / "cohort_v1" / "cases.jsonl",
        ]
        cases_path = next((path for path in candidates if path.is_file()), None)
        if cases_path is None:
            raise FileNotFoundError("could not infer --cases; pass it explicitly")
    cases_path = cases_path.expanduser().resolve()
    images_root = (args.images_root.expanduser().resolve() if args.images_root else cases_path.parent / "images")
    cases = read_jsonl(cases_path)
    audit = preflight(cases, images_root, require_images=not args.dry_run)
    if args.limit is not None:
        cases = cases[: args.limit]
    cases = select_cases(cases, args.num_shards, args.shard_index)

    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    config_hash = stable_hash({"config": config, "paths": paths, "cases_sha256": sha256_file(cases_path)})
    cases_sha256 = sha256_file(cases_path)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and args.no_resume:
        raise FileExistsError(f"--no-resume set but output exists: {output}")

    completed: set[str] = set()
    if output.exists():
        for row in read_jsonl(output):
            if row.get("status") == "ok":
                completed.add(str(row.get("record_id")))

    planned = [(str(case["case_id"]), condition) for case in cases for condition in args.conditions]
    missing = [(cid, condition) for cid, condition in planned if f"{cid}::{condition}" not in completed]
    meta = {
        "schema_version": "realbind2_facts_image_500_run_meta_v1",
        "config_path": str(args.config.expanduser().resolve()),
        "paths_path": str(args.paths.expanduser().resolve()),
        "cases_path": str(cases_path),
        "cases_sha256": cases_sha256,
        "images_root": str(images_root),
        "config_hash": config_hash,
        "model_id": config.get("model_id"),
        "adapter": config.get("adapter"),
        "runtime": config.get("runtime"),
        "conditions": list(args.conditions),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "case_count_total": len(read_jsonl(cases_path)),
        "case_count_shard": len(cases),
        "planned_generations": len(planned),
        "already_complete": len(planned) - len(missing),
        "missing_generations": len(missing),
        "preflight": audit,
    }
    (output.parent / f"{output.stem}.run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("DRY_RUN_PASS")
        return 0
    if not missing:
        print(f"FACTS_IMAGE_COMPLETE completed={len(completed)} planned={len(planned)}")
        return 0

    adapter = resolve_adapter(config, paths)
    failed_log = output.with_suffix(".failed.jsonl")
    completed_now = 0
    for case in cases:
        image_path = resolve_image_path(images_root, str(case["image_file"]))
        for condition in args.conditions:
            record_id = f"{case['case_id']}::{condition}"
            if record_id in completed:
                continue
            started = time.monotonic()
            try:
                if condition == "facts_only":
                    prepared = adapter.prepare_prompt(str(case["text_prompt"]), None)
                else:
                    prepared = adapter.prepare_prompt(str(case["text_prompt"]), image_path)
                raw = adapter.generate(prepared)
                scored = score_prediction(str(raw), str(case["ground_truth"]), str(case["answer_type"]))
                record = build_record(
                    case=case,
                    condition=condition,
                    raw=str(raw),
                    scored=scored,
                    config=config,
                    config_hash=config_hash,
                    cases_sha256=cases_sha256,
                    runtime_sec=time.monotonic() - started,
                )
                try:
                    record["model_metadata"] = adapter.get_model_metadata()
                except Exception as exc:  # metadata is diagnostic, not part of the condition
                    record["model_metadata"] = {"error": f"{type(exc).__name__}: {exc}"}
                append_jsonl(output, record)
                completed.add(record_id)
                completed_now += 1
                print(f"{condition} {case['case_id']} :: {scored.normalized!r} :: {'OK' if scored.correct else 'WRONG'}", flush=True)
            except Exception as exc:
                append_jsonl(failed_log, {
                    "schema_version": "realbind2_facts_image_500_failure_v1",
                    "record_id": record_id,
                    "case_id": str(case["case_id"]),
                    "condition": condition,
                    "model_id": config.get("model_id"),
                    "adapter": config.get("adapter"),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "runtime_sec": time.monotonic() - started,
                    "config_hash": config_hash,
                    "cases_sha256": cases_sha256,
                })
                print(f"{condition} {case['case_id']} :: FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    total_expected = len(planned)
    total_complete = len(completed & {f"{cid}::{condition}" for cid, condition in planned})
    incomplete = total_expected - total_complete
    print(f"FACTS_IMAGE_RUN_STOPPED completed={total_complete}/{total_expected} new={completed_now} incomplete={incomplete} failed_log={failed_log}")
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
