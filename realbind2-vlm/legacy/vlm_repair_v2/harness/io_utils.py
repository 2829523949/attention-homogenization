from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def write_new(path: Path, text: str) -> None:
    if path.exists(): raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json_new(path: Path, value: Any) -> None:
    write_new(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
