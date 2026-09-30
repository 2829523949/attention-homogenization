"""Shared helpers for the server-only live certification commands."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_server_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    for section in ("python", "models", "data", "output"):
        if section not in config:
            raise KeyError(f"missing server config section: {section}")
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_dataset_root(config: dict[str, Any]) -> Path:
    return Path(config["data"]["realbind2"]).expanduser().resolve()


def certification_root(config: dict[str, Any]) -> Path:
    return Path(config["output"]["root"]).expanduser().resolve() / "server_certification"


def smoke_manifest() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "data" / "smoke_cases.json").read_text(encoding="utf-8"))


def frozen_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = frozen_dataset_root(config)
    family_path = root / "frozen" / "families.jsonl"
    question_path = root / "frozen" / "questions.jsonl"
    if not family_path.is_file() or not question_path.is_file():
        raise FileNotFoundError("certified frozen RealBind2 export is required")
    return read_jsonl(family_path), read_jsonl(question_path)


def certification_identity(config: dict[str, Any]) -> dict[str, str]:
    root = frozen_dataset_root(config) / "frozen"
    return {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "families_sha256": sha256_file(root / "families.jsonl"),
        "questions_sha256": sha256_file(root / "questions.jsonl"),
        "repository_hashes_sha256": sha256_file(REPO_ROOT / "data" / "HASHES.json"),
    }


def require_live_receipt(config: dict[str, Any]) -> None:
    path = certification_root(config) / "live_certification.json"
    if not path.is_file():
        raise RuntimeError("formal baseline requires LIVE_SMOKE_PASS receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "LIVE_SMOKE_PASS" or receipt.get("identity") != certification_identity(config):
        raise RuntimeError("live certification receipt is stale or failed; rerun certification")


def resolve_smoke_families(config: dict[str, Any]) -> list[dict[str, Any]]:
    families, _ = frozen_rows(config)
    indexed: dict[str, dict[str, Any]] = {}
    for family in families:
        for key in (family.get("candidate_family_id"), family.get("family_id")):
            if key is not None:
                indexed[str(key)] = family
    requested = smoke_manifest()["families"]
    missing = [row["candidate_family_id"] for row in requested if row["candidate_family_id"] not in indexed]
    if missing:
        raise RuntimeError(
            "fixed smoke families are absent from the frozen export; do not substitute cases: "
            + ", ".join(missing)
        )
    return [indexed[row["candidate_family_id"]] for row in requested]


def image_path(config: dict[str, Any], family: dict[str, Any]) -> Path:
    path = frozen_dataset_root(config) / "frozen" / str(family["image_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def profile_record(
    config: dict[str, Any], name: str, runtime_sec: float, peak_gpu_memory_mb: float
) -> None:
    path = certification_root(config) / "runtime_profile.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema_version": "realbind2_server_runtime_profile_v1",
        "components": {},
    }
    payload["components"][name] = {
        "runtime_sec": float(runtime_sec),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
    }
    write_json_atomic(path, payload)


def cuda_begin() -> float:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live certification")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return __import__("time").monotonic()


def cuda_end(started: float) -> tuple[float, float]:
    import torch

    torch.cuda.synchronize()
    return (
        __import__("time").monotonic() - started,
        torch.cuda.max_memory_allocated() / (1024 * 1024),
    )


def exact_crop_box(image_size: tuple[int, int], bbox: list[float]) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = (int(round(float(value))) for value in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty bbox after clamping: {bbox}")
    return x0, y0, x1, y1
