"""Frozen RealBind2 JSONL loader and integrity gates."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

EXPECTED_FAMILIES = 1400
EXPECTED_QA = 9800
EXPECTED_COMPOSITION_QA = 2800
COMPOSITION_TYPES = {"REL2C_LEFT", "REL2C_RIGHT"}


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


@dataclass(frozen=True)
class DatasetAudit:
    families: int
    unique_source_images: int
    qa: int
    composition_qa: int
    duplicate_qa_ids: tuple[str, ...]
    missing_image_paths: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.families == EXPECTED_FAMILIES
            and self.qa == EXPECTED_QA
            and self.composition_qa == EXPECTED_COMPOSITION_QA
            and not self.duplicate_qa_ids
            and not self.missing_image_paths
        )


class RealBind2Dataset:
    def __init__(self, root: Path, *, require_frozen: bool = True):
        self.root = Path(root).expanduser().resolve()
        self.families_path = self.root / "frozen" / "families.jsonl"
        self.questions_path = self.root / "frozen" / "questions.jsonl"
        self.stats_path = self.root / "frozen" / "construction_stats.json"
        if require_frozen and not all(p.is_file() for p in (self.families_path, self.questions_path, self.stats_path)):
            raise FileNotFoundError(
                "RealBind2 frozen export is absent. Complete human review and run the builder export; "
                f"expected {self.families_path} and {self.questions_path}."
            )

    def families(self) -> list[dict]:
        return list(read_jsonl(self.families_path))

    def questions(self) -> list[dict]:
        return list(read_jsonl(self.questions_path))

    def audit(self, *, check_images: bool = True) -> DatasetAudit:
        families, questions = self.families(), self.questions()
        qa_counts = Counter(str(row["qa_id"]) for row in questions)
        missing: list[str] = []
        if check_images:
            for family in families:
                image = self.root / "frozen" / family["image_path"]
                if not image.is_file():
                    missing.append(str(image))
        return DatasetAudit(
            families=len(families),
            unique_source_images=len({str(row["image_id"]) for row in families}),
            qa=len(questions),
            composition_qa=sum(row.get("qa_type") in COMPOSITION_TYPES for row in questions),
            duplicate_qa_ids=tuple(sorted(key for key, n in qa_counts.items() if n != 1)),
            missing_image_paths=tuple(missing),
        )

    def shard(self, num_shards: int, shard_index: int) -> list[dict]:
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError("invalid deterministic shard")
        return [row for i, row in enumerate(self.questions()) if i % num_shards == shard_index]

    def family_shard(self, num_shards: int, shard_index: int) -> list[dict]:
        """Assign every QA from one family to ``family_index % num_shards``."""

        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError("invalid deterministic shard")
        family_ids = [str(row["family_id"]) for row in self.families()]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("duplicate family IDs prevent deterministic family sharding")
        selected = {
            family_id
            for family_index, family_id in enumerate(family_ids)
            if family_index % num_shards == shard_index
        }
        return [row for row in self.questions() if str(row["family_id"]) in selected]
