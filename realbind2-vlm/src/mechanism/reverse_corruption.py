from __future__ import annotations

from .residual_patch import PatchSpec


def reverse_spec(clean_case_id: str, failure_case_id: str, layer: int, token_indices: tuple[int, ...]) -> PatchSpec:
    return PatchSpec(
        recipient_case_id=clean_case_id,
        donor_case_id=failure_case_id,
        layer=layer,
        token_indices=token_indices,
        direction="reverse_corruption",
    )

