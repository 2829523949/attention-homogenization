"""Frozen fact injection and condition contracts for VIR.

The image transformation is supplied by a separately hashed defocus backend;
this module never selects or tunes one.
"""

from __future__ import annotations

import hashlib


def build_correct_facts(family: dict, question: dict) -> dict:
    a = str(family["referent_A"])
    b = str(family["referent_B"])
    color_a = str(family["color_A"])
    color_b = str(family["color_B"])
    relation = "left" if bool(family["geometry_metrics"]["left_is_A"]) else "right"
    lines = (
        f"The {a} is {color_a}.",
        f"The {b} is {color_b}.",
        f"The {a} is to the {relation} of the {b}.",
    )
    text = "\n".join(lines) + "\n\n" + str(question["question"])
    forbidden = ("therefore", "the answer is", "the left one is", "the right one is")
    if any(token in "\n".join(lines).casefold() for token in forbidden):
        raise AssertionError("derived answer leaked into fact injection")
    return {"facts": list(lines), "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def condition_contract(family: dict, question: dict, defocus_image_path: str) -> list[dict]:
    facts = build_correct_facts(family, question)
    return [
        {"condition": "facts_only", "image_path": None, **facts},
        {"condition": "image_correct_facts", "image_path": family["image_path"], **facts},
        {"condition": "defocus_correct_facts", "image_path": defocus_image_path, **facts},
    ]

