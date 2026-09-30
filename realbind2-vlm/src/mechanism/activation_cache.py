from __future__ import annotations

import hashlib
from pathlib import Path


def immutable_cache_path(root: Path, case_id: str, trace_kind: str) -> Path:
    digest = hashlib.sha256(f"{case_id}\0{trace_kind}".encode()).hexdigest()[:16]
    return Path(root) / trace_kind / f"{case_id}__{digest}.pt"


def save_once(path: Path, payload) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite activation cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

