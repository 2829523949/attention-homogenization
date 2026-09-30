#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.cache import CachedColorReader,CachedRegionProcessor,JsonCache,cache_key
from harness.dataset import composition_cases
from harness.io_utils import write_jsonl_new
from harness.pipeline import acquire_visual_state
from harness.planners.deterministic import DeterministicPlanner
from harness.processors.florence2 import Florence2Processor
from harness.processors.color_pixel import PixelColorReader


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke-count", type=int, default=0); ap.add_argument("--case-id", action="append"); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--with-siglip", action="store_true"); ap.add_argument("--allow-full-benchmark", action="store_true"); args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    cases = composition_cases(); selected = [c for c in cases if not args.case_id or c.case_id in set(args.case_id)]
    if args.smoke_count:
        # Exercise different images/families rather than LEFT/RIGHT twins.
        distinct=[]; seen=set()
        for case in selected:
            if case.family_id in seen: continue
            distinct.append(case); seen.add(case.family_id)
            if len(distinct) == args.smoke_count: break
        selected = distinct
    if not selected: raise ValueError("select cases with --smoke-count or --case-id")
    if len(selected) > 5 and not args.allow_full_benchmark: raise RuntimeError("safety gate: >5 cases requires --allow-full-benchmark")
    p = Paths(); ensure_d_drive_environment(p)
    localizer = CachedRegionProcessor(Florence2Processor(p.florence2),JsonCache(p.cache/"regions/florence2"),"microsoft_Florence-2-base-ft_local")
    pixel = CachedColorReader(PixelColorReader(),JsonCache(p.cache/"colors/pixel"),"deterministic_v1")
    siglip = None
    if args.with_siglip:
        from harness.processors.color_siglip import SiglipColorReader
        siglip = SiglipColorReader(p.siglip2)
    planner = DeterministicPlanner(); rows=[]
    for case in selected:
        payload=case.to_dict(); payload["plan"] = planner.plan({"qa_id": case.case_id, "qa_type": case.qa_type, "image_path": case.image_path}).to_dict()
        state=acquire_visual_state(payload, localizer, pixel_reader=pixel, siglip_reader=siglip)
        state_key,state_fields=cache_key(image_path=Path(case.image_path),processor="visual_state",version="v1",model_revision=localizer.version,config={"case_id":case.case_id,"object_A":case.object_A,"object_B":case.object_B,"siglip":bool(siglip)})
        JsonCache(p.cache/"visual_states").put(state_key,{"key_fields":state_fields,"visual_state":state.to_dict()})
        rows.append({"case": payload, "visual_state": state.to_dict()})
        print(json.dumps({"case_id": case.case_id, "missing": state.uncertainty["missing_entities"]}))
    write_jsonl_new(args.output, rows); return 0


if __name__ == "__main__": raise SystemExit(main())
