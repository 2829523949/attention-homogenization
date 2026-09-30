"""Subprocess wrapper for the bundled Florence-2 checkpoint.

The repository ships Florence-2 custom code that requires its isolated
``transformers`` dependency tree under ``external/florence_pydeps``.  The main
harness process uses a newer Transformers version for SigLIP-2.  This adapter
keeps the two dependency trees isolated by running Florence in a persistent
line-oriented worker process.

Only acquisition is exposed.  Florence-2 does not expose a referent-region
confidence score, so ``score`` is deliberately unsupported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from harness.certification.evidence import EvidenceScore
from harness.processors.base import RegionPrediction
from harness.referents.referent_spec import GroundingQuery


ROOT = Path(__file__).resolve().parents[2]


class FlorenceSubprocessGrounding:
    name = "florence2_query_grounding"
    version = "florence2_open_vocabulary_query_subprocess_v2"
    can_acquire = True
    supports_certification_score = False
    supports_same_claim_rescoring = False
    comparable_across_counterfactual_views = False
    score_semantics = (
        "CERTIFICATION_SCORE_UNSUPPORTED: Florence-2 open-vocabulary postprocessing "
        "returns bboxes/labels but no numeric referent-region confidence."
    )
    score_normalization = "none"

    def __init__(self, model_path: Path, device: str = "cpu", *, worker_script: Path | None = None):
        if not Path(model_path).is_dir():
            raise FileNotFoundError(model_path)
        self.model_path = Path(model_path)
        self.device = device
        self.worker_script = Path(worker_script) if worker_script else ROOT / "harness" / "scripts" / "florence_worker.py"
        self.process: subprocess.Popen[str] | None = None
        # Per-request, logging-only metadata from the worker.  Selection never
        # reads this field.
        self.last_candidate_trace: dict[str, Any] | None = None

    def _worker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        paths = [
            str(ROOT),
            str(ROOT / "external" / "florence_pydeps"),
            str(ROOT / "external" / "siglip_pydeps"),
        ]
        existing = env.get("PYTHONPATH")
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("HF_HOME", str(ROOT / "hf_cache"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / "hf_cache" / "hub"))
        env.setdefault("TRANSFORMERS_CACHE", str(ROOT / "hf_cache" / "transformers"))
        env.setdefault("TORCH_HOME", str(ROOT / "torch_cache"))
        env.setdefault("XDG_CACHE_HOME", str(ROOT / "cache" / "xdg"))
        env.setdefault("TMPDIR", str(ROOT / "tmp"))
        return env

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self.process is not None and self.process.poll() is None:
            return self.process
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(self.worker_script),
                "--model",
                str(self.model_path),
                "--device",
                self.device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        handshake = self._read_response()
        if handshake.get("ok") is not True:
            raise RuntimeError(f"Florence worker failed to start: {handshake}")
        self.version = str(handshake.get("version", self.version))
        return self.process

    def _read_response(self) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Florence worker terminated unexpectedly")
        return json.loads(line)

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_worker()
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response = self._read_response()
        if "error" in response:
            raise RuntimeError(f"Florence worker error: {response['error']}")
        return response

    def ground(self, image_path: Path, query: GroundingQuery) -> list[RegionPrediction]:
        if not isinstance(query, GroundingQuery):
            raise TypeError("grounding requires GroundingQuery")
        response = self._call({"op": "ground", "image_path": str(image_path), "query": query.semantic_text})
        self.last_candidate_trace = response.get("candidate_pipeline_trace")
        result: list[RegionPrediction] = []
        for item in response.get("candidates", []):
            result.append(
                RegionPrediction.from_pixels(
                    item["bbox_xyxy_pixels"],
                    int(item["image_width"]),
                    int(item["image_height"]),
                    score=None if item.get("score") is None else float(item["score"]),
                    label=str(item.get("label", "")),
                    source=str(item.get("source", "florence2:open_vocabulary_detection:subprocess")),
                )
            )
        return result

    # Alias used by a few diagnostics/documentation examples.
    predict_query = ground

    def score(self, image_path: Path, query: GroundingQuery, region: Any, *, view_id: str = "RAW") -> EvidenceScore:
        return EvidenceScore.unsupported(
            self.name,
            reason=(
                "CERTIFICATION_SCORE_UNSUPPORTED: Florence-2 output contains no referent-region "
                "confidence; subprocess acquisition cannot certify a binding score."
            ),
            claimed_value=region,
            diagnostics={"certification_capability": "CERTIFICATION_SCORE_UNSUPPORTED"},
            view_id=view_id,
        )

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        finally:
            self.process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
