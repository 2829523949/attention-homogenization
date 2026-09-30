#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from harness.cache import JsonCache,cache_key
from harness.configs.paths import Paths,ensure_d_drive_environment
from harness.io_utils import write_jsonl_new
from harness.processors.base import VisualState
from harness.processors.color_siglip import SiglipColorReader


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--states",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--allow-full-benchmark",action="store_true");args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    items=[json.loads(x) for x in args.states.read_text().splitlines() if x.strip()]
    if len(items)>5 and not args.allow_full_benchmark:raise RuntimeError("safety gate: >5 cases requires --allow-full-benchmark")
    p=Paths();ensure_d_drive_environment(p);reader=SiglipColorReader(p.siglip2);cache=JsonCache(p.cache/"colors/siglip2");rows=[]
    for item in items:
        state=VisualState.from_dict(item["visual_state"]);image=Path(item["case"]["image_path"]);predictions={}
        for entity in state.entities:
            if entity.predicted_region is None:predictions[entity.reference_id]=None;continue
            key,fields=cache_key(image_path=image,processor=reader.name,version=reader.version,model_revision="local_download",config={"vocabulary":list(reader.vocabulary),"bbox":list(entity.predicted_region.bbox_xyxy_pixels)},query=entity.reference_id)
            cached=cache.get(key)
            if cached is None:
                result=reader.read(image,entity.predicted_region);cached={"key_fields":fields,"prediction":result.__dict__};cache.put(key,cached)
            predictions[entity.reference_id]={"cache_key":key,**cached}
        row={"case_id":state.case_id,"predictions":predictions};rows.append(row);print(json.dumps({"case_id":state.case_id,"colors":{k:(v or {}).get("prediction",{}).get("color") for k,v in predictions.items()}}))
    write_jsonl_new(args.output,rows);return 0
if __name__=="__main__":raise SystemExit(main())
