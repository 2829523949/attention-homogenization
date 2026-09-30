"""Frozen deterministic semantic normalizer for RealBind2.

The scorer accepts exactly one canonical candidate mention.  Conflicting or
unmapped output remains invalid even when an explanatory prefix looks useful.
It never uses the expected answer to choose a parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

SCORER_VERSION = "realbind2-semantic-v1"
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray")
ALIASES = {"grey": "gray"}
CANDIDATES = {
    "color": COLORS,
    "identity": ("A", "B"),
    "position": ("LEFT", "RIGHT"),
}


@dataclass(frozen=True)
class Score:
    raw: str
    normalized: str | None
    ground_truth: str
    correct: bool
    valid: bool
    error: str | None


def _mentions(text: str, candidates: Iterable[str]) -> list[str]:
    found: list[str] = []
    for candidate in candidates:
        pattern = rf"(?<![A-Za-z]){re.escape(candidate)}(?![A-Za-z])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(candidate)
    return found


def normalize_output(raw: str, answer_type: str) -> tuple[str | None, str | None]:
    if answer_type not in CANDIDATES:
        raise ValueError(f"unsupported answer_type: {answer_type}")
    text = str(raw).strip()
    for alias, canonical in ALIASES.items():
        text = re.sub(rf"(?<![A-Za-z]){alias}(?![A-Za-z])", canonical, text, flags=re.IGNORECASE)
    mentions = _mentions(text, CANDIDATES[answer_type])
    if len(mentions) != 1:
        return None, "zero_or_multiple_candidate_mentions"
    value = mentions[0]
    return value.lower() if answer_type == "color" else value.upper(), None


def score_prediction(raw: str, ground_truth: str, answer_type: str) -> Score:
    normalized, error = normalize_output(raw, answer_type)
    expected = ground_truth.lower() if answer_type == "color" else ground_truth.upper()
    return Score(
        raw=str(raw),
        normalized=normalized,
        ground_truth=expected,
        correct=normalized == expected,
        valid=normalized is not None,
        error=error,
    )

