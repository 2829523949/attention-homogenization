"""Reusable stage cache; each model consumes the same frozen minimal state."""

from __future__ import annotations

import json
import os
from pathlib import Path


class StageCache:
    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = Path(path)
        self.key_fields = key_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: dict[tuple[str, ...], dict] = {}
        if self.path.exists():
            for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = self.key(row)
                if key in self.rows:
                    raise ValueError(f"duplicate cache key at {self.path}:{line_no}: {key}")
                self.rows[key] = row

    def key(self, row: dict) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in self.key_fields)

    def append(self, row: dict) -> None:
        key = self.key(row)
        if key in self.rows:
            if self.rows[key] != row:
                raise ValueError(f"refusing to overwrite frozen cache row {key}")
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = row

