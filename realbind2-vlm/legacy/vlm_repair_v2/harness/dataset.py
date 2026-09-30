from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.configs.paths import Paths


@dataclass(frozen=True)
class InferenceCase:
    case_id: str
    family_id: str
    qa_type: str
    image_path: str
    question: str
    object_A: str
    object_B: str

    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_frozen(paths: Paths | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p = paths or Paths()
    return read_jsonl(p.families), read_jsonl(p.qa)


def composition_cases(paths: Paths | None = None) -> list[InferenceCase]:
    families, qa = load_frozen(paths)
    by_family = {str(row["family_id"]): row for row in families}
    cases = []
    for row in qa:
        if row["qa_type"] not in {"REL2C_LEFT", "REL2C_RIGHT"}: continue
        family = by_family[str(row["family_id"])]
        cases.append(InferenceCase(str(row["qa_id"]), str(row["family_id"]), str(row["qa_type"]), str(row["image_path"]), str(row["question"]), str(family["A"]["object"]), str(family["B"]["object"])))
    if len(cases) != 200: raise RuntimeError(f"frozen composition case count changed: {len(cases)}")
    return cases


def evaluator_records(paths: Paths | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    families, qa = load_frozen(paths)
    return ({str(row["family_id"]): row for row in families}, {str(row["qa_id"]): row for row in qa})
