from __future__ import annotations

import re
from collections import Counter

from .gates import COLORS

RELATION_LEAK_TOKENS = {"left", "right", "leftmost", "rightmost", "east", "west", "leftward", "rightward"}


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def leakage_reasons(referent: str, answer: str | None = None) -> list[str]:
    tokens = set(normalize_words(referent))
    reasons = []
    if tokens.intersection(COLORS):
        reasons.append("REFERENT_LEAKS_COLOR")
    if tokens.intersection(RELATION_LEAK_TOKENS):
        reasons.append("REFERENT_LEAKS_RELATION")
    if answer and answer.lower() in tokens:
        reasons.append("REFERENT_LEAKS_RELATION" if answer.lower() in RELATION_LEAK_TOKENS else "REFERENT_LEAKS_COLOR")
    return reasons


def construct_referent(category: str, attrs: list[str], same_category_count: int) -> tuple[str, str]:
    """Only emits an annotation-template descriptor; uniqueness stays human-gated."""
    noun = re.sub(r"\s+", " ", category.strip().lower())
    safe = []
    for attr in attrs:
        value = re.sub(r"\s+", " ", str(attr).strip().lower())
        if not value or set(normalize_words(value)).intersection(set(COLORS) | RELATION_LEAK_TOKENS):
            continue
        safe.append(value)
    # The smallest deterministic disambiguator is selected by lexical order,
    # then audited visually. Metadata cannot prove visibility or uniqueness.
    suffix = f" with {sorted(set(safe))[0]}" if safe else ""
    phrase = f"the {noun}{suffix}"
    status = "HUMAN_UNIQUENESS_CHECK_REQUIRED"
    return phrase, status
