"""Parser for the audited frozen RealBind prompt interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .referent_spec import GroundingQuery, Provenance, Provenanced, ReferentSpec


class ReferentInterfaceStatus(str, Enum):
    RESOLVABLE_FROM_TEXT = "RESOLVABLE_FROM_TEXT"
    RESOLVABLE_WITH_PROTOCOL_ROLE = "RESOLVABLE_WITH_PROTOCOL_ROLE"
    REFERENT_INTERFACE_UNDERSPECIFIED = "REFERENT_INTERFACE_UNDERSPECIFIED"


@dataclass(frozen=True)
class ResolvedReferents:
    status: ReferentInterfaceStatus
    task_type: str | None
    specs: tuple[ReferentSpec, ...]
    selection_side: str | None
    notes: tuple[str, ...] = ()

    def queries(self) -> tuple[GroundingQuery, ...]:
        return tuple(GroundingQuery.from_spec(spec) for spec in self.specs)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "task_type": self.task_type,
            "specs": [spec.to_dict() for spec in self.specs],
            "selection_side": self.selection_side,
            "notes": list(self.notes),
        }


def _spec(logical_ref: str, noun: str, *, role_detail: str | None = None) -> ReferentSpec:
    text = Provenanced(noun, Provenance.QUESTION_TEXT, f"frozen question category noun: {noun!r}")
    category = Provenanced(noun, Provenance.QUESTION_TEXT, f"frozen question category noun: {noun!r}")
    if role_detail is not None:
        relation = Provenanced(logical_ref, Provenance.PROTOCOL_INTERFACE, role_detail)
        return ReferentSpec(logical_ref, text, category, relation_constraints=(relation,))
    return ReferentSpec(logical_ref, text, category)


def resolve_question(question: str, task_type: str | None = None) -> ResolvedReferents:
    """Resolve only visible nouns and an explicit legal protocol role.

    This function deliberately has no family/QA-row argument. Hidden A/B
    metadata therefore cannot be smuggled into a GroundingQuery.
    """

    text = str(question)
    both = re.fullmatch(
        r"Between the (.+?) and the (.+?),\n(?:(?:what color is the one on the (left|right))|(?:which one is on the (left|right)))\?\nAnswer with only (?:the color|the object name)\.",
        text,
        flags=re.IGNORECASE,
    )
    if both:
        noun_a, noun_b = both.group(1).strip(), both.group(2).strip()
        side = (both.group(3) or both.group(4)).casefold()
        if not noun_a or not noun_b or noun_a.casefold() == noun_b.casefold():
            return ResolvedReferents(ReferentInterfaceStatus.REFERENT_INTERFACE_UNDERSPECIFIED, task_type, (), side, ("missing or duplicate visible category noun",))
        return ResolvedReferents(
            ReferentInterfaceStatus.RESOLVABLE_FROM_TEXT,
            task_type,
            (_spec("A", noun_a), _spec("B", noun_b)),
            side,
            ("A/B are logical roles assigned by visible template order; they are not detector labels",),
        )

    pair = re.fullmatch(
        r"Is the (.+?) to the left or right of the (.+?)\?\nAnswer with only left or right\.",
        text,
        flags=re.IGNORECASE,
    )
    if pair:
        noun_a, noun_b = pair.group(1).strip(), pair.group(2).strip()
        return ResolvedReferents(ReferentInterfaceStatus.RESOLVABLE_FROM_TEXT, task_type, (_spec("A", noun_a), _spec("B", noun_b)), None, ("ordered nouns are visible in the question",))

    single = re.fullmatch(r"What color is the (.+?)\?\nAnswer with only the color\.", text, flags=re.IGNORECASE)
    if single:
        noun = single.group(1).strip()
        if task_type in {"I2C_A", "I2C_B"}:
            return ResolvedReferents(
                ReferentInterfaceStatus.RESOLVABLE_WITH_PROTOCOL_ROLE,
                task_type,
                (_spec(task_type[-1], noun, role_detail=f"explicit protocol operation {task_type}"),),
                None,
                ("the noun is visible; the A/B role comes from the explicit protocol task type",),
            )
        return ResolvedReferents(
            ReferentInterfaceStatus.RESOLVABLE_WITH_PROTOCOL_ROLE,
            task_type,
            (_spec("TARGET", noun),),
            None,
            ("raw-question-only interface does not assign the hidden A/B role",),
        )

    return ResolvedReferents(
        ReferentInterfaceStatus.REFERENT_INTERFACE_UNDERSPECIFIED,
        task_type,
        (),
        None,
        ("question does not match an audited legal RealBind template",),
    )

