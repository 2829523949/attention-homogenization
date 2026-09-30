"""Intervention contracts shared by the frozen causal runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSpec:
    recipient_case_id: str
    donor_case_id: str
    layer: int
    token_indices: tuple[int, ...]
    direction: str


def self_patch(hidden, spec: PatchSpec):
    """Return the original tensor for the mandatory self-patch no-op control."""
    if spec.recipient_case_id != spec.donor_case_id:
        raise ValueError("self-patch requires identical donor and recipient")
    return hidden


def assert_self_patch_preserved(before, after) -> None:
    try:
        import torch
        equal = torch.equal(before, after) if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor) else before == after
    except ImportError:
        equal = before == after
    if not equal:
        raise AssertionError("self-patch changed the recipient state")
