"""Visual acquisition diagnostics. Inference entry accepts prompt-only inputs."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from harness.cache import sha256_file
from harness.configs.paths import Paths
from harness.io_utils import write_json_new
from harness.processors.base import RegionPrediction
from harness.processors.florence2 import Florence2Processor, map_generic_regions

P = Paths()
OUT = P.outputs / 'phase1a_visual_diagnosis'
MANIFEST = P.manifests / 'phase1a_20family_diagnostic.json'
THRESHOLDS = (.3, .5, .7)

def load(path):
    return json.loads(Path(path).read_text())

def prompt_references(question):
    match = re.fullmatch(r'Between the (.+?) and the (.+?),\nwhat color is the one on the (left|right)\?\nAnswer with only the color\.', question)
    if not match:
        raise ValueError('Frozen composition prompt does not match audited template; no metadata fallback')
    return {'A': match[1], 'B': match[2]}

def verify_sources():
    frozen = load(P.manifests / 'frozen_manifest.json')
    for key, path in [('families', P.families), ('qa', P.qa), ('scorer', P.scorer), ('dependency', P.dependency)]:
        if sha256_file(path) != frozen['sha256'][key]:
            raise RuntimeError(f'frozen source changed: {key}')
    return frozen

def prepare():
    from harness.dataset import read_jsonl
    frozen = verify_sources()
    families = frozen['family_ids'][:20]
    qa = read_jsonl(P.qa)
    cases = [r for fid in families for typ in ('REL2C_LEFT', 'REL2C_RIGHT') for r in qa if r['family_id'] == fid and r['qa_type'] == typ]
    assert len(cases) == 40 and len(set(r['qa_id'] for r in cases)) == 40
    inputs = []
    for fid in families:
        pair = [r for r in cases if r['family_id'] == fid]
        assert prompt_references(pair[0]['question']) == prompt_references(pair[1]['question'])
        inputs.append({'family_id': fid, 'image_path': pair[0]['image_path'], 'question': pair[0]['question'], 'case_ids': [r['qa_id'] for r in pair]})
    manifest = {'schema_version': 'phase1a_v1', 'selection_rule': 'First 20 family IDs in frozen_manifest.json order; both LEFT and RIGHT retained; no outcome/label-based selection', 'family_ids': families, 'case_ids': [r['qa_id'] for r in cases], 'source_manifest': str(P.manifests/'frozen_manifest.json'), 'source_manifest_sha256': sha256_file(P.manifests/'frozen_manifest.json'), 'source_hashes': frozen['sha256'], 'family_count': 20, 'case_count': 40, 'inference_inputs': inputs, 'ranking': 'All valid proposals in acquisition order, then model decode order; no GT sorting or synthetic score ranking. Report all-pool recall separately.', 'qwen_enabled': False}
    if MANIFEST.exists():
        assert load(MANIFEST) == manifest, 'existing phase manifest differs'
    else: write_json_new(MANIFEST, manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'prepared_families':families, 'cases':len(cases)}), flush=True)

class RecordingFlorence(Florence2Processor):
    """Record every parsed proposal before the frozen adapter filters/selects."""
    def start(self):
        self.calls = []
    def _run_task(self, image, task, max_new_tokens, *, task_token=None):
        result = super()._run_task(image, task, max_new_tokens, task_token=task_token)
        self.calls.append({'task': task_token or task, 'prompt': task, 'path': 'fallback_query' if task_token else ('object_detection' if task == '<OD>' else 'generic_proposal' if task == '<REGION_PROPOSAL>' else 'other'), 'max_new_tokens':max_new_tokens, 'parsed':result})
        return result

def acquire_one(request, localizer):
    allowed = {'family_id','image_path','question','case_ids'}
    if set(request) != allowed: raise ValueError('Inference input contains hidden/unapproved fields')
    refs = prompt_references(request['question'])
    localizer.start()
    image = Path(request['image_path'])
    regions = localizer.predict(image, task='<OD>', fallback_region_proposals=True)
    mapping, trace = map_generic_regions(regions, refs)
    for ref in ('A','B'):
        if mapping[ref] is None:
            candidates = localizer.predict_query(image, refs[ref])
            mapping[ref] = max(candidates, key=lambda r:((r.score if r.score is not None else float("-inf")),-r.bbox_xyxy_pixels[0])) if candidates else None
    from PIL import Image
    with Image.open(image) as im: width,height=im.size
    proposals=[]
    for call_index, call in enumerate(localizer.calls):
        payload=call['parsed'].get(call['task'],{})
        boxes=payload.get('bboxes',[]) if isinstance(payload,dict) else []
        labels=payload.get('labels',[]) if isinstance(payload,dict) else []
        scores=payload.get('scores',[]) if isinstance(payload,dict) else []
        for index,box in enumerate(boxes):
            valid=len(box)==4 and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height
            proposals.append({'proposal_index':len(proposals), 'call_index':call_index, 'decode_index':index, 'bbox':box, 'bbox_normalized':[box[0]/width,box[1]/height,box[2]/width,box[3]/height] if len(box)==4 else None, 'score':scores[index] if index<len(scores) else None, 'label':labels[index] if index<len(labels) else None, 'task':call['task'], 'path':call['path'], 'query':call['prompt'][len(call['task']):] or None, 'valid':bool(valid)})
    selected={}
    for ref,region in mapping.items():
        indices=[p['proposal_index'] for p in proposals if region and p['valid'] and list(region.bbox_xyxy_pixels)==p['bbox']]
        if region and not indices: raise RuntimeError('selected region missing from raw proposals')
        selected[ref]={'region':dataclasses.asdict(region) if region else None,'proposal_index':indices[0] if indices else None}
    return {'family_id':request['family_id'],'case_ids':request['case_ids'],'image_path':request['image_path'],'image_sha256':sha256_file(image),'references_from_prompt':refs,'question':request['question'],'raw_calls':localizer.calls,'proposals':proposals,'selected':selected,'selection_trace':trace,'inference_has_gt':False}

def acquire():
    verify_sources()
    localizer=RecordingFlorence(P.florence2)
    for request in load(MANIFEST)['inference_inputs']:
        path=OUT/'acquisition'/f"{request['family_id']}.json"
        if path.exists():
            saved=load(path)
            assert saved['image_sha256']==sha256_file(Path(request['image_path'])) and saved['question']==request['question']
            continue
        result=acquire_one(request,localizer)
        write_json_new(path,result)
        print(json.dumps({'family':request['family_id'],'proposals':len(result['proposals']),'selected':{r:v['proposal_index'] for r,v in result['selected'].items()}}),flush=True)

def acquisitions():
    return [load(OUT/'acquisition'/f'{fid}.json') for fid in load(MANIFEST)['family_ids']]

def region(box,width,height,source='diagnostic'):
    return RegionPrediction.from_pixels(box,width,height,score=1.,label='',source=source)

def evaluate_regions():
    from harness.dataset import read_jsonl
    from harness.evaluators.processor_metrics import iou, xywh_to_xyxy
    families={r['family_id']:r for r in read_jsonl(P.families)}
    rows=[]
    for acq in acquisitions():
        f=families[acq['family_id']]; proposals=[p for p in acq['proposals'] if p['valid']]
        gt={ref:xywh_to_xyxy(tuple(f[ref]['bbox'])) for ref in ('A','B')}
        for ref in ('A','B'):
            overlaps=[(iou(tuple(p['bbox']),gt[ref]),p) for p in proposals]
            best=max(overlaps,key=lambda x:(x[0],-x[1]['proposal_index'])) if overlaps else None
            selected=acq['selected'][ref]['region']; box=selected['bbox_xyxy_pixels'] if selected else None
            selected_iou=iou(tuple(box),gt[ref]) if box else 0.
            other='B' if ref=='A' else 'A'
            top={str(k):max((v for v,p in overlaps[:k]),default=0.) for k in (1,3,5,10)}
            rows.append({'entity_id':f"{f['family_id']}__{ref}",'family_id':f['family_id'],'case_ids':acq['case_ids'],'entity':ref,'image_path':acq['image_path'],'width':f['image_width'],'height':f['image_height'],'gt_color':f[ref]['canonical_color'],'gt_bbox':gt[ref],'gt_relation_A_to_B':'left' if f['left_is_A'] else 'right','best_iou':best[0] if best else 0.,'best_proposal_index':best[1]['proposal_index'] if best else None,'best_bbox':best[1]['bbox'] if best else None,'selected_bbox':box,'selected_proposal_index':acq['selected'][ref]['proposal_index'],'selected_iou':selected_iou,'selected_other_iou':iou(tuple(box),gt[other]) if box else 0.,'top_k_best_iou':top,'proposal_available':bool(best and best[0]>=.5),'selected_correct':selected_iou>=.5,'gt_pair_iou':iou(gt['A'],gt['B']),'multiple_valid_proposals':sum(v>=.5 for v,p in overlaps),'best_tie_count':sum(abs(v-best[0])<1e-9 for v,p in overlaps) if best else 0,'proposal_count':len(proposals)})
    write_json_new(OUT/'evaluator_region_records.json',rows)

def colors(reader_name):
    import numpy as np
    from harness.processors.color_pixel import PixelColorReader, REALBIND_COLORS
    from PIL import Image
    rows=load(OUT/'evaluator_region_records.json')
    if reader_name=='pixel': reader=PixelColorReader()
    else:
        import torch
        from harness.processors.color_siglip import SiglipColorReader
        torch.manual_seed(0);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        reader=SiglipColorReader(P.siglip2)
        # Verify processing analytically using a solid RGB input, independently
        # of all diagnostic labels, and verify the declared tokenizer pairing.
        ip=reader.processor.image_processor
        sample=Image.new('RGB',(32,24),(255,0,0))
        pv=ip(images=sample,return_tensors='np')['pixel_values']
        assert np.allclose(pv[0,:,0,0],[1.,-1.,-1.],atol=1e-5), 'RGB/normalization validation failed'
        assert tuple(reader.vocabulary)==REALBIND_COLORS
        assert reader.templates==tuple(f'a {c} object' for c in REALBIND_COLORS)
        audit={'processor_class':type(reader.processor).__name__,'model_class':type(reader.model).__name__,'tokenizer_class':type(reader.processor.tokenizer).__name__,'model_path':str(P.siglip2),'model_type':reader.model.config.model_type,'text_vocabulary_size':reader.model.config.text_config.vocab_size,'vocabulary':list(reader.vocabulary),'templates':list(reader.templates),'normalization':ip.to_dict(),'solid_red_processed_channels':pv[0,:,0,0].tolist(),'probabilities':'softmax across fixed colors; not calibrated probabilities','deterministic_algorithms':True,'model_training':reader.model.training,'gt_label_input':False}
        assert not reader.model.training and 'Gemma' in audit['tokenizer_class'] and audit['text_vocabulary_size']==256000
        r=rows[0]; box=region(r['gt_bbox'],r['width'],r['height'],'evaluator_only')
        first=reader.read(Path(r['image_path']),box);second=reader.read(Path(r['image_path']),box)
        audit['repeat_max_logit_difference']=float(np.max(np.abs(np.array(first.diagnostics['logits'])-second.diagnostics['logits'])))
        assert audit['repeat_max_logit_difference']==0
        write_json_new(OUT/'siglip_validation.json',audit)
    for r in rows:
        path=OUT/'colors'/reader_name/f"{r['entity_id']}.json"
        if path.exists(): continue
        predictions={}
        for mode,key in [('gt_region','gt_bbox'),('best_proposal','best_bbox'),('automatic','selected_bbox')]:
            box=r[key]
            if box is None:
                predictions[mode]={'color':None,'confidence':0.,'diagnostics':{'reason':'missing_region'}};continue
            typed=region(box,r['width'],r['height'],'evaluator_only' if mode!='automatic' else 'predicted')
            prediction=reader.read(Path(r['image_path']),typed)
            with Image.open(r['image_path']) as opened:
                crop=opened.convert('RGB').crop(tuple(int(round(x)) for x in box))
                crop_hash=hashlib.sha256(crop.tobytes()).hexdigest()
            predictions[mode]={**dataclasses.asdict(prediction),'region_mode':mode,'EVALUATOR_ONLY':mode!='automatic','bbox':box,'crop_rgb_sha256':crop_hash,'integer_crop_xyxy':[int(round(x)) for x in box],'crop_size':list(crop.size)}
        write_json_new(path,predictions)
        print(json.dumps({'reader':reader_name,'entity':r['entity_id'],'predictions':{k:v['color'] for k,v in predictions.items()}}),flush=True)
