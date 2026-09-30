#!/usr/bin/env python3
"""Persistent JSON-line worker for the isolated Florence-2 dependency tree.

Run with ``PYTHONPATH`` containing the repository root and
``external/florence_pydeps`` (the parent adapter sets this automatically).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from harness.processors.florence2 import Florence2Processor

    processor = Florence2Processor(Path(args.model), device=args.device)
    print(json.dumps({"ok": True, "version": processor.version}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "ping":
                response = {"ok": True, "version": processor.version}
            elif operation == "ground":
                image_path = Path(request["image_path"])
                query = str(request["query"])
                regions = processor.predict_query(image_path, query)
                from PIL import Image

                with Image.open(image_path) as opened:
                    width, height = opened.size
                response = {
                    "ok": True,
                    "candidate_pipeline_trace": processor.last_candidate_trace,
                    "candidates": [
                        {
                            "bbox_xyxy_pixels": list(region.bbox_xyxy_pixels),
                            "bbox_xyxy_normalized": list(region.bbox_xyxy_normalized),
                            "score": region.score,
                            "label": region.label,
                            "source": region.source,
                            "image_width": width,
                            "image_height": height,
                        }
                        for region in regions
                    ],
                }
            else:
                response = {"error": f"unknown operation: {operation!r}"}
        except Exception as exc:  # pragma: no cover - worker diagnostics
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
