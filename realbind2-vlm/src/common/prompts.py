"""Frozen RealBind2 prompt templates.

These strings are copied from ``realbind_v2/question_generator.py``.  Changes
are protocol changes and therefore require a new dataset version.
"""

from __future__ import annotations

import hashlib
import json

QA_ORDER = (
    "I2C_A",
    "I2C_B",
    "PAIR_REL_A_TO_B",
    "REL2I_LEFT",
    "REL2I_RIGHT",
    "REL2C_LEFT",
    "REL2C_RIGHT",
)

TEMPLATES = {
    "I2C_A": "What color is {A}? Answer with one color.",
    "I2C_B": "What color is {B}? Answer with one color.",
    "PAIR_REL_A_TO_B": "Is {A} to the LEFT or RIGHT of {B}? Answer LEFT or RIGHT.",
    "REL2I_LEFT": "Between {A} and {B}, which one is on the LEFT? Answer A or B.",
    "REL2I_RIGHT": "Between {A} and {B}, which one is on the RIGHT? Answer A or B.",
    "REL2C_LEFT": "Between {A} and {B}, what color is the one on the LEFT? Answer with one color.",
    "REL2C_RIGHT": "Between {A} and {B}, what color is the one on the RIGHT? Answer with one color.",
}


def template_hashes() -> dict[str, str]:
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sorted(TEMPLATES.items())
    }


def prompt_bundle_hash() -> str:
    payload = json.dumps(TEMPLATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_prompt(qa_type: str, referent_a: str, referent_b: str) -> str:
    try:
        template = TEMPLATES[qa_type]
    except KeyError as exc:
        raise ValueError(f"unknown RealBind2 qa_type: {qa_type}") from exc
    return template.format(A=referent_a, B=referent_b)

