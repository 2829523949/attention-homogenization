#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from harness.configs.paths import Paths
from harness.dataset import evaluator_records
from harness.evaluators.processor_metrics import GroundTruthRecord, aggregate, evaluate_state
from harness.io_utils import write_json_new, write_jsonl_new
from harness.processors.base import VisualState
from harness.processors.base import RegionPrediction
from harness.processors.color_pixel import PixelColorReader


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--states",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    families,_=evaluator_records(Paths()); rows=[]; oracle_rows=[]; automatic_rows=[]; reader=PixelColorReader()
    for line in args.states.read_text().splitlines():
        item=json.loads(line); state=VisualState.from_dict(item["visual_state"]); family=families[item["case"]["family_id"]]
        gt=GroundTruthRecord(state.case_id,tuple(family["A"]["bbox"]),tuple(family["B"]["bbox"]),family["A"]["canonical_color"],family["B"]["canonical_color"],bool(family["left_is_A"]))
        row=evaluate_state(state,gt); image=Path(item["case"]["image_path"]); width,height=int(family["image_width"]),int(family["image_height"])
        oracle_correct=[]
        for ref in ("A","B"):
            x,y,w,h=family[ref]["bbox"]
            region=RegionPrediction.from_pixels([x,y,x+w,y+h],width,height,score=1.,label="evaluator_region",source="oracle_region_diagnostic")
            prediction=reader.read(image,region); oracle_correct.append(prediction.color==family[ref]["canonical_color"])
            row["entities"][ref]["color_gt_region_prediction"]={"color":prediction.color,"confidence":prediction.confidence,"diagnostics":prediction.diagnostics}
        row["color_accuracy_gt_region"]=sum(oracle_correct)/2
        oracle_rows.append({"case_id":state.case_id,"mode":"A_ORACLE_REGION_DIAGNOSTIC","region_input":"evaluator_gt_bbox","inference_use":False,"entities":{ref:row["entities"][ref]["color_gt_region_prediction"] for ref in ("A","B")},"color_accuracy_gt_region":row["color_accuracy_gt_region"]})
        automatic={key:value for key,value in row.items() if key!="color_accuracy_gt_region"};automatic["mode"]="B_AUTOMATIC_ACQUISITION";automatic["gt_enters_inference"]=False
        automatic_rows.append(automatic);rows.append(row)
    automatic_summary=aggregate(rows);automatic_summary.pop("color_accuracy_gt_region",None)
    summary={"mode_a_oracle_region_diagnostic":{"N":len(oracle_rows),"color_accuracy_gt_region":sum(r["color_accuracy_gt_region"] for r in oracle_rows)/len(oracle_rows) if oracle_rows else None},"mode_b_automatic_acquisition":automatic_summary}
    args.output_dir.mkdir(parents=True);write_jsonl_new(args.output_dir/"oracle_region_diagnostics.jsonl",oracle_rows);write_jsonl_new(args.output_dir/"automatic_acquisition.jsonl",automatic_rows);write_jsonl_new(args.output_dir/"per_case.jsonl",rows);write_json_new(args.output_dir/"summary.json",summary);print(json.dumps(summary,indent=2));return 0
if __name__=="__main__": raise SystemExit(main())
