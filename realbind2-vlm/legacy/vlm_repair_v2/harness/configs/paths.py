from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/d/vlm_repair_v2")


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    project: Path = Path("/root/Qwen3-VL")
    realbind: Path = Path("/mnt/d/datasets/RealBind/realbind")
    qwen: Path = Path("/root/Qwen3-VL/models/Qwen3.5-4B")

    @property
    def harness(self) -> Path: return self.root / "harness"
    @property
    def cache(self) -> Path: return self.root / "cache"
    @property
    def outputs(self) -> Path: return self.root / "outputs"
    @property
    def manifests(self) -> Path: return self.root / "manifests"
    @property
    def florence2(self) -> Path: return self.root / "models/florence2/Florence-2-base-ft"
    @property
    def siglip2(self) -> Path: return self.root / "models/siglip2/siglip2-base-patch16-224"
    @property
    def sam2(self) -> Path: return self.root / "models/sam2"
    @property
    def grounding_dino(self) -> Path: return self.root / "models/grounding_dino/grounding-dino-base"
    @property
    def families(self) -> Path: return self.realbind / "families/dryrun_100_pair_direct_v1_1_frozen.jsonl"
    @property
    def qa(self) -> Path: return self.realbind / "qa/dryrun_100_pair_direct_v1_1.jsonl"
    @property
    def dependency(self) -> Path: return self.realbind / "qa/pair_direct_v1_1_dependency_spec.json"
    @property
    def scorer(self) -> Path: return self.project / "experiments/realbind_phase2/pair_direct_v1_1_baseline.py"

    def validate(self) -> None:
        if self.root != ROOT or not str(self.root).startswith("/mnt/d/"):
            raise RuntimeError(f"new artifact root must be on D drive: {self.root}")
        for required in (self.project, self.realbind, self.qwen, self.families, self.qa, self.scorer):
            if not required.exists():
                raise FileNotFoundError(required)


def ensure_d_drive_environment(paths: Paths | None = None) -> dict[str, str]:
    p = paths or Paths()
    values = {
        "HF_HOME": str(p.root / "hf_cache"),
        "HUGGINGFACE_HUB_CACHE": str(p.root / "hf_cache/hub"),
        "TRANSFORMERS_CACHE": str(p.root / "hf_cache/transformers"),
        "TORCH_HOME": str(p.root / "torch_cache"),
        "PIP_CACHE_DIR": str(p.root / "pip_cache"),
        "XDG_CACHE_HOME": str(p.root / "cache/xdg"),
        "TMPDIR": str(p.root / "tmp"),
    }
    for key, value in values.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    return values
