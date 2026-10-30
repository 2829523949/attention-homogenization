#!/usr/bin/env python3
"""Shared helpers for the PACO RealBind2-protocol reproduction.

The R4 method code is intentionally imported from the frozen RealBind2
release rather than reimplemented where possible:
  - question templates: src/common/prompts.py
  - scorer: src/common/scorer.py
  - geometry/minimal state/final prompt/attribute parser:
    src/repair/strong_integrated/protocol.py
  - R4 constants and exact crop/grid behavior: release scripts and protocol
  - paired statistics: src/repair/strong_integrated/statistics.py

PACO-specific code only builds benchmark manifests, orchestrates model calls,
and adds the selector-predicted A/B IoU<0.1 conditioning step.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_RELEASE_ROOT = Path("/root/Qwen3-VL/repair-github/realbind2-final-delivery")
OTHER_COLOR_ID = 29
BASE_COLORS = ("black", "white", "gray", "brown", "red", "blue", "green", "yellow", "orange", "pink", "purple")
SELECTOR_PROMPT = "Which numbered crop best matches: {referent}? Answer only the number."
COLOR_PROMPT = (
    "What color is the main object or region shown?\n"
    "Answer with only one color from this list: red, orange, yellow, green, "
    "blue, purple, pink, brown, black, white, gray."
)


def release_root() -> Path:
    value = os.environ.get("REALBIND2_RELEASE_ROOT")
    return Path(value).expanduser().resolve() if value else DEFAULT_RELEASE_ROOT.resolve()


def ensure_release_on_path(root: Path | None = None) -> Path:
    root = (root or release_root()).resolve()
    if not (root / "src" / "common" / "scorer.py").is_file():
        raise FileNotFoundError(
            f"RealBind2 release root not found/correct: {root}. "
            "Set REALBIND2_RELEASE_ROOT or --release-root."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_paths(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    for key in ("release_root", "dataset_root", "output_root", "models"):
        if key not in cfg:
            raise KeyError(f"paths config missing key: {key}")
    return cfg


def model_config_path(tag: str, paths_cfg: dict[str, Any]) -> Path:
    root = Path(paths_cfg["release_root"])
    p = root / "configs" / f"{tag}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"model config not found for tag {tag}: {p}")
    return p


def load_model_spec(tag: str, paths_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Return (model_config, merged_runtime_config, model_path)."""
    model_cfg = load_yaml(model_config_path(tag, paths_cfg))
    key = model_cfg.get("path_key")
    if not key or key not in paths_cfg["models"]:
        raise KeyError(f"path key {key!r} for model tag {tag} missing from paths.models")
    model_path = Path(paths_cfg["models"][key]).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path for {tag} is missing: {model_path}")
    runtime = dict(model_cfg.get("runtime") or {})
    # The frozen RealBind2 adapter contract is batch=1 / greedy / non-thinking.
    # Do not allow callers to silently change these in the YAML.
    if runtime.get("max_new_tokens") not in (None, 16):
        raise ValueError(f"{tag}: RealBind2 protocol requires max_new_tokens=16")
    if runtime.get("use_cache") not in (None, False, True):
        raise ValueError(f"{tag}: invalid use_cache")
    return model_cfg, runtime, model_path


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def code_identity() -> dict[str, Any]:
    scripts = Path(__file__).resolve().parent
    files = {str(p.relative_to(scripts.parent)): sha256_file(p) for p in sorted(scripts.glob("*.py"))}
    return {"files": files, "hash": stable_hash(files)}


def canonical_color(name: str) -> str:
    name = str(name).strip().lower()
    if name in {"grey", "gray"}:
        return "gray"
    for base in ("blue", "brown", "green", "orange", "pink", "purple", "red", "yellow"):
        if name == base or name.endswith("_" + base):
            return base
    if name.endswith("_grey") or name.endswith("_gray"):
        return "gray"
    if name in {"black", "white"}:
        return name
    return name


def display_category(name: str) -> str:
    name = str(name).replace("_", " ")
    replacements = {
        "car (automobile)": "car",
        "glass (drink container)": "glass",
        "mouse (computer equipment)": "computer mouse",
        "pan (for cooking)": "pan",
        "tape (sticky cloth or paper)": "tape",
        "slipper (footwear)": "slipper",
        "drum (musical instrument)": "drum",
        "cellular telephone": "cell phone",
    }
    return replacements.get(name, name)


def referent_phrase(category: str) -> str:
    return "the " + display_category(category)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 1.0


def bounded_box(box: list[float], width: int, height: int) -> list[float]:
    out = [
        min(max(float(box[0]), 0.0), float(width)),
        min(max(float(box[1]), 0.0), float(height)),
        min(max(float(box[2]), 0.0), float(width)),
        min(max(float(box[3]), 0.0), float(height)),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        raise ValueError(f"degenerate candidate: {out}")
    return out


def exact_crop_box(image_size: tuple[int, int], bbox: list[float]) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty bbox after clamping: {bbox}")
    return x0, y0, x1, y1


def make_selector_grid(image, candidates: list[dict[str, Any]], path: Path) -> None:
    """Verbatim numbered-grid construction from RealBind2 server_live_certification._grid."""
    from PIL import Image, ImageDraw
    tile, pad, columns = 224, 28, 4
    count = len(candidates)
    grid = Image.new("RGB", (columns * tile, math.ceil(count / columns) * (tile + pad)), "white")
    draw = ImageDraw.Draw(grid)
    for index, candidate in enumerate(candidates):
        crop = image.crop(exact_crop_box(image.size, candidate["bbox_xyxy"]))
        crop.thumbnail((tile - 12, tile - 12))
        x = (index % columns) * tile + (tile - crop.width) // 2
        y = (index // columns) * (tile + pad) + pad + (tile - crop.height) // 2
        grid.paste(crop, (x, y))
        draw.text(((index % columns) * tile + 8, (index // columns) * (tile + pad) + 5), str(index + 1), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path, format="PNG")


def select_candidates(adapter, image, candidates: list[dict[str, Any]], referent: str,
                      grid_path: Path) -> dict[str, Any]:
    """Exact R4 selector semantics: singleton/no-call, exact integer, native-top1 fallback."""
    if not candidates:
        return {"status": "NO_CANDIDATE", "selected_index": None, "selected_bbox_xyxy": None,
                "selector_raw": None, "selector_status": "NO_CANDIDATE", "candidate_count": 0}
    if len(candidates) == 1:
        return {"status": "ok", "selected_index": 0,
                "selected_bbox_xyxy": candidates[0]["bbox_xyxy"],
                "selected_native_score": candidates[0].get("native_score"),
                "selector_raw": None, "selector_status": "SINGLETON_NO_QWEN_CALL", "candidate_count": 1}
    make_selector_grid(image, candidates, grid_path)
    generation_error = None
    try:
        raw = adapter.generate(adapter.prepare_prompt(SELECTOR_PROMPT.format(referent=referent), image_path=str(grid_path)))
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if any(token in message for token in ("cuda", "out of memory", "driver", "alloc", "device")):
            raise
        raw = ""
        generation_error = f"{type(exc).__name__}: {exc}"
    match = re.fullmatch(r"\s*(\d+)\s*", str(raw))
    if match and 1 <= int(match.group(1)) <= len(candidates):
        idx, status = int(match.group(1)) - 1, "VALID"
    else:
        idx, status = 0, "QWEN_SELECTOR_INVALID_FALLBACK_NATIVE_TOP1"
    return {"status": "ok", "selected_index": idx,
            "selected_bbox_xyxy": candidates[idx]["bbox_xyxy"],
            "selected_native_score": candidates[idx].get("native_score"),
            "selector_raw": raw, "selector_status": status, "candidate_count": len(candidates),
            "selector_generation_error": generation_error}


def load_adapter(model_cfg: dict[str, Any], model_path: Path, runtime: dict[str, Any]):
    ensure_release_on_path()
    from src.models import ADAPTERS
    adapter_key = model_cfg["adapter"]
    if adapter_key not in ADAPTERS:
        raise ValueError(f"unsupported adapter: {adapter_key}")
    return ADAPTERS[adapter_key](str(model_path), **runtime)


def scorer():
    ensure_release_on_path()
    from src.common.scorer import score_prediction
    return score_prediction


def render_question(qa_type: str, referent_a: str, referent_b: str) -> str:
    ensure_release_on_path()
    from src.common.prompts import render_prompt
    return render_prompt(qa_type, referent_a, referent_b)


def r4_protocol():
    ensure_release_on_path()
    from src.repair.strong_integrated.protocol import (
        geometry, crop_box, parse_attribute, minimal_state, final_prompt, EPSILON, VERSION
    )
    return {
        "geometry": geometry, "crop_box": crop_box, "parse_attribute": parse_attribute,
        "minimal_state": minimal_state, "final_prompt": final_prompt,
        "EPSILON": EPSILON, "VERSION": VERSION,
    }


def ensure_non_empty_text(raw: Any) -> str:
    return "" if raw is None else str(raw)


def generation_call(adapter, prompt: str, image_path: str | None = None,
                    *, text_only: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    generation_error = None
    try:
        prepared = adapter.prepare_prompt(prompt, image_path=image_path)
        raw = adapter.generate(prepared)
    except Exception as exc:  # resource failures must abort; content failures stay invalid
        message = str(exc).lower()
        if any(token in message for token in ("cuda", "out of memory", "driver", "alloc", "device")):
            raise
        return {"raw": "", "generation_error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started, "prompt": prompt,
                "image_path": image_path, "text_only": text_only}
    return {"raw": str(raw), "generation_error": generation_error,
            "elapsed_seconds": time.monotonic() - started, "prompt": prompt,
            "image_path": image_path, "text_only": text_only}
