"""Append-only, fsync-backed per-case result logging with resume support."""

from __future__ import annotations

import json
import os
from pathlib import Path


class JsonlRunLog:
    def __init__(self, path: Path, *, id_field: str = "case_id"):
        self.path = Path(path)
        self.id_field = id_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: set[str] = set()
        if self.path.exists():
            for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row[id_field])
                if key in self.completed:
                    raise ValueError(f"duplicate {id_field}={key} at {self.path}:{line_no}")
                if row.get("status") == "ok":
                    self.completed.add(key)

    def is_complete(self, case_id: str) -> bool:
        return str(case_id) in self.completed

    def append(self, row: dict) -> None:
        key = str(row[self.id_field])
        if key in self.completed:
            raise ValueError(f"refusing to overwrite completed {self.id_field}={key}")
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if row.get("status") == "ok":
            self.completed.add(key)

