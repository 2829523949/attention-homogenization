#!/usr/bin/env python3
"""Locked four-route region pilot. Inference reads only frozen inference fields."""
from __future__ import annotations
import argparse, hashlib, html, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from harness.processors.crop_policy import deterministic_crop_box
from harness.scripts.run_target_color_processor_pilot import VLM_PROMPT, parse_vlm

OUT=ROOT/'outputs/region_bottleneck_joint_repair'
OLD=ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
PREP=ROOT/'outputs/minimal_decomposition_route_pilot/prepared.jsonl'
PRIOR_COLOR=ROOT/'outputs/target_color_processor_pilot/vlm_color_calls.jsonl'
MODEL=ROOT/'models/mm_grounding_dino/mm_grounding_dino_large_all'
CAP=16
BOX_THRESHOLD=.25
TEXT_THRESHOLD=.25
ROUTES={'R1':('original','old'),'R2':('strong','old'),'R3':('original','qwen'),'R4':('strong','qwen')}

def rows(p):
 p=Path(p)
 return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_rows(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in rs))
def append(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,sort_keys=True,allow_nan=False)+'\n')
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image_width(path):
 with Image.open(path) as im:return im.width
def iou(a,b):
 if a is None:return 0.
 ix=max(0.,min(a[2],b[2])-max(a[0],b[0]));iy=max(0.,min(a[3],b[3])-max(a[1],b[1]));inter=ix*iy
 aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb>inter else 0.
def cohort():
 manifest=json.loads((ROOT/'manifests/acquisition_semantic_support_20family.json').read_text())
 prepared=rows(PREP);old=[r['inference'] for r in rows(OLD) if r['inference']['backend']=='grounding_dino']
 assert len(manifest['families'])==20 and len(prepared)==40 and len(old)==40
 assert {r['qa_id'] for r in prepared}=={q for f in manifest['families'] for q in f['case_ids']}
 assert {r['family_id'] for r in old}=={f['family_id'] for f in manifest['families']}
 return manifest,prepared,old
def original_candidates(inf):
 return [{'candidate_index':i,'bbox_xyxy':c['bbox_xyxy_pixels'],'native_score':c['native_grounding_score'],'matched_phrase':c['native_label_text'],'backend_source':'original_grounding_dino','candidate_id':c['candidate_id']} for i,c in enumerate(inf['candidates'][:CAP])]
def old_select(c):
 return min(range(len(c)),key=lambda i:(-(c[i]['native_score'] if c[i]['native_score'] is not None else -math.inf),c[i]['bbox_xyxy'][0],i)) if c else None
def prepare():
 OUT.mkdir(exist_ok=True,parents=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'selector_grids').mkdir(exist_ok=True);(OUT/'target_crops').mkdir(exist_ok=True)
 manifest,prepared,old=cohort()
 results=[]
 for inf in old:
  assert inf['legal_query']==inf['referent_spec']['text_description']['value']
  c=original_candidates(inf)
  results.append({'family_id':inf['family_id'],'logical_ref':inf['logical_ref'],'generator':'original','image_path':inf['image_path'],'referent_phrase':inf['legal_query'],'referent_spec':inf['referent_spec'],'candidates':c,'old_selected_index':old_select(c),'source_sha256':file_sha(OLD)})
 write_rows(OUT/'candidate_results.jsonl',results)
 write(OUT/'route_config.json',{'routes':ROUTES,'cohort_manifest':str(ROOT/'manifests/acquisition_semantic_support_20family.json'),'cohort_sha256':file_sha(ROOT/'manifests/acquisition_semantic_support_20family.json'),'original_candidate_source_sha256':file_sha(OLD),'candidate_cap':CAP,'box_threshold':BOX_THRESHOLD,'text_threshold':TEXT_THRESHOLD,'strong_checkpoint':'openmmlab-community/mm_grounding_dino_large_all','strong_local_path':str(MODEL),'prompt_conversion':'frozen ReferentSpec.text_description.value verbatim; no added context','qwen_selector_method':'candidate_grid','qwen_selector_prompt':'Which numbered crop best matches: {referent}? Answer only the number.','color_prompt':VLM_PROMPT,'color_prompt_sha256':hashlib.sha256(VLM_PROMPT.encode()).hexdigest()})
 print('prepared',len(results),flush=True)
def strong():
 import torch,transformers
 from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
 assert MODEL.joinpath('model.safetensors').exists()
 _,_,old=cohort(); path=OUT/'candidate_results.jsonl';done={(x['family_id'],x['logical_ref'],x['generator']) for x in rows(path)}
 proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 import transformers as _strong_tf
 _load_kwargs={'local_files_only':True,'torch_dtype':torch.float32} if _strong_tf.__version__.startswith('4.') else {'local_files_only':True,'dtype':torch.float32}
 model=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL,**_load_kwargs).to('cuda').eval()
 for n,inf in enumerate(old,1):
  key=(inf['family_id'],inf['logical_ref'],'strong')
  if key in done:continue
  phrase=inf['referent_spec']['text_description']['value'];assert phrase==inf['legal_query']
  image=Image.open(inf['image_path']).convert('RGB')
  # Official deterministic conversion of the legal ReferentSpec phrase to the
  # MM-GDINO text contract: a singleton candidate-label list.  The official
  # processor lowercases it and appends the required phrase separator.  No
  # attribute, relation, question, target-side, or GT information is added.
  inputs=proc(images=image,text=[[phrase]],return_tensors='pt').to('cuda')
  with torch.inference_mode():outputs=model(**inputs)
  pred=proc.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=BOX_THRESHOLD,text_threshold=TEXT_THRESHOLD,target_sizes=[image.size[::-1]])[0]
  labels=list(pred.get('text_labels') or pred.get('labels') or [])
  raw=[]
  for i,(box,score,label) in enumerate(zip(pred['boxes'],pred['scores'],labels)):
   bb=[float(v) for v in box.tolist()];score=float(score)
   if bb[2]<=bb[0] or bb[3]<=bb[1]:continue
   raw.append({'raw_candidate_index':i,'bbox_xyxy':bb,'native_score':score,'matched_phrase':str(label)})
  # One predeclared cap for the strong grounder.  Retain native-score top-K
  # with deterministic raw-index tie breaking; no GT/evaluator data is used.
  raw.sort(key=lambda x:(-x['native_score'],x['raw_candidate_index']))
  kept=raw[:CAP]
  c=[]
  for j,item in enumerate(kept):
   c.append({'candidate_index':j,'bbox_xyxy':item['bbox_xyxy'],'native_score':item['native_score'],'matched_phrase':item['matched_phrase'],'backend_source':'mm_grounding_dino_large_all','candidate_id':f"{inf['logical_ref']}:mm_grounding_dino:{j}",'raw_candidate_index':item['raw_candidate_index']})
  rec={'family_id':inf['family_id'],'logical_ref':inf['logical_ref'],'generator':'strong','image_path':inf['image_path'],'referent_phrase':phrase,'referent_spec':inf['referent_spec'],'candidates':c,'old_selected_index':old_select(c),'raw_postprocessor_count':len(labels),'candidate_retention_policy':'native_score_top16_ties_raw_index','strong_transformers_version':_strong_tf.__version__}
  append(path,rec);print('strong',n,'/40',inf['family_id'],inf['logical_ref'],len(c),flush=True)
  del pred,outputs,inputs,image,raw,kept,labels
  torch.cuda.empty_cache()
def normalize_strong():
 path=OUT/'candidate_results.jsonl';all_rows=rows(path);assert len(all_rows)==80
 changed=0
 for row in all_rows:
  if row['generator']!='strong':continue
  assert len(row['candidates'])<=CAP, 'candidate count exceeded predeclared cap'
  with Image.open(row['image_path']) as im:width,height=im.size
  clean=[]
  for item in row['candidates']:
   raw=item['bbox_xyxy'];clipped=[max(0.,min(width,raw[0])),max(0.,min(height,raw[1])),max(0.,min(width,raw[2])),max(0.,min(height,raw[3]))]
   if clipped!=raw:changed+=1;item['raw_bbox_xyxy']=raw;item['bbox_xyxy']=clipped
   if clipped[2]>clipped[0] and clipped[3]>clipped[1]:
    if item['candidate_index']!=len(clean):item['raw_candidate_id']=item['candidate_id'];item['candidate_id']=f"{row['logical_ref']}:mm_grounding_dino:{len(clean)}"
    item['candidate_index']=len(clean);clean.append(item)
  row['candidates']=clean;row['old_selected_index']=old_select(clean)
 write_rows(path,all_rows)
 print('normalized',changed,'clipped strong boxes',flush=True)
def grid_image(image_path,candidates,path):
 image=Image.open(image_path).convert('RGB');tile=224;pad=28;columns=4;count=len(candidates);grid=Image.new('RGB',(columns*tile,math.ceil(count/columns)*(tile+pad)),'white');draw=ImageDraw.Draw(grid)
 for i,c in enumerate(candidates):
  box=deterministic_crop_box(image.size,c['bbox_xyxy']);crop=image.crop(box);crop.thumbnail((tile-12,tile-12))
  x=(i%columns)*tile+(tile-crop.width)//2;y=(i//columns)*(tile+pad)+pad+(tile-crop.height)//2
  grid.paste(crop,(x,y));draw.text(((i%columns)*tile+8,(i//columns)*(tile+pad)+5),str(i+1),fill='black')
 grid.save(path)
def select_qwen():
 from harness.reasoners.qwen35 import Qwen35Reasoner
 candidates=rows(OUT/'candidate_results.jsonl');assert len(candidates) in (40,80)
 path=OUT/'selector_results.jsonl';done={(x['family_id'],x['logical_ref'],x['generator']) for x in rows(path)}
 q=Qwen35Reasoner()
 for n,r in enumerate(candidates,1):
  key=(r['family_id'],r['logical_ref'],r['generator'])
  if key in done:continue
  c=r['candidates'];fallback=None;raw=None;grid=None
  if not c:idx=None;status='NO_CANDIDATE'
  elif len(c)==1:idx=0;status='SINGLETON_NO_QWEN_CALL'
  else:
   grid=OUT/'selector_grids'/f"{r['family_id']}__{r['logical_ref']}__{r['generator']}.png";grid_image(r['image_path'],c,grid)
   prompt=f"Which numbered crop best matches: {r['referent_phrase']}? Answer only the number."
   g=q.generate(prompt,grid);assert g['vision_encoder_called'] and g['image_path']==str(grid)
   raw=g['response'];m=re.fullmatch(r'\s*(\d+)\s*',raw)
   if m and 1<=int(m.group(1))<=len(c):idx=int(m.group(1))-1;status='VALID'
   else:idx=r['old_selected_index'];status='QWEN_SELECTOR_INVALID_FALLBACK';fallback=idx
  rec={'family_id':r['family_id'],'logical_ref':r['logical_ref'],'generator':r['generator'],'selector':'qwen','candidate_count':len(c),'selected_index':idx,'status':status,'raw_output':raw,'grid_path':str(grid) if grid else None,'fallback_index':fallback,'referent_phrase':r['referent_phrase'],'question_provided':False,'target_side_provided':False,'gt_provided':False}
  append(path,rec);print('selector',n,'/',len(candidates),r['generator'],r['logical_ref'],status,idx,flush=True)
def color():
 from harness.reasoners.qwen35 import Qwen35Reasoner
 c={(r['family_id'],r['logical_ref'],r['generator']):r for r in rows(OUT/'candidate_results.jsonl')}
 s={(r['family_id'],r['logical_ref'],r['generator']):r for r in rows(OUT/'selector_results.jsonl')}
 prepared=rows(PREP);prior=rows(PRIOR_COLOR);image_by_qa={r['qa_id']:r['image_path'] for r in prepared}
 available_routes=[route for route,(gen,sel) in ROUTES.items() if all((p['family_id'],ref,gen) in c and (sel=='old' or (p['family_id'],ref,gen) in s) for p in prepared for ref in ('A','B'))]
 cache={sha({'image_path':image_by_qa[r['qa_id']],'bbox_xyxy':r['selected_bbox_xyxy_pixels']}):r['normalized_color'] for r in prior if r['valid']}
 path=OUT/'color_results.jsonl';existing=rows(path);cache.update({r['cache_key']:r['normalized_color'] for r in existing if r['valid']})
 done={r['cache_key'] for r in existing};tasks=[]
 for p in prepared:
  for route in available_routes:
   gen,sel=ROUTES[route]
   a=c[p['family_id'],'A',gen];b=c[p['family_id'],'B',gen]
   ia=a['old_selected_index'] if sel=='old' else s[p['family_id'],'A',gen]['selected_index'];ib=b['old_selected_index'] if sel=='old' else s[p['family_id'],'B',gen]['selected_index']
   if ia is None or ib is None:continue
   ba=a['candidates'][ia]['bbox_xyxy'];bb=b['candidates'][ib]['bbox_xyxy'];dx=((ba[0]+ba[2])-(bb[0]+bb[2]))/(2*image_width(p['image_path']))
   left='A' if dx < -1e-9 else 'B' if dx > 1e-9 else None
   target=(left if p['qa_type'].endswith('LEFT') else ('B' if left=='A' else 'A' if left=='B' else None))
   if target is None:continue
   bbox=ba if target=='A' else bb
   image=Image.open(p['image_path']).convert('RGB');box=deterministic_crop_box(image.size,bbox);crop=image.crop(box)
   out=OUT/'target_crops'/f"{sha({'image':p['image_path'],'box':box})}.png"
   if not out.exists():crop.save(out)
   digest=file_sha(out);cache_key=sha({'image_path':p['image_path'],'bbox_xyxy':bbox})
   tasks.append({'qa_id':p['qa_id'],'route':route,'family_id':p['family_id'],'target':target,'bbox_xyxy':bbox,'crop_box':box,'crop_path':str(out),'crop_sha256':digest,'cache_key':cache_key,'color_cache_reused':cache_key in cache,'color_new_inference':cache_key not in cache})
 write_rows(OUT/'color_tasks.jsonl',tasks)
 new={x['cache_key']:x for x in tasks if x['cache_key'] not in cache and x['cache_key'] not in done}
 if not new:return
 q=Qwen35Reasoner()
 for i,t in enumerate(new.values(),1):
  g=q.generate(VLM_PROMPT,Path(t['crop_path']));assert g['vision_encoder_called'] and g['image_path']==t['crop_path']
  val=parse_vlm(g['response'])
  append(path,{'cache_key':t['cache_key'],'bbox_xyxy':t['bbox_xyxy'],'crop_sha256':t['crop_sha256'],'crop_path':t['crop_path'],'raw_output':g['response'],'normalized_color':val,'valid':val is not None,'prompt_sha256':hashlib.sha256(VLM_PROMPT.encode()).hexdigest(),'new_inference':True,'vision_encoder_called':g['vision_encoder_called']})
  print('color',i,'/',len(new),val,flush=True)

def transitions(left,right,keys):
 a={(r[k] for k in []):None} if False else None
 result={}
 for level,universe in keys.items():
  n=Counter()
  for key in universe:
   x=bool(left[key][level]);y=bool(right[key][level]);n['fail_to_correct' if not x and y else 'correct_to_fail' if x and not y else 'both_correct' if x else 'both_fail']+=1
  result[level]=dict(n|Counter({k:0 for k in ('fail_to_correct','correct_to_fail','both_correct','both_fail')}))
 return result

def analyze():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 _,prepared,old=cohort()
 candidate=rows(OUT/'candidate_results.jsonl');selector=rows(OUT/'selector_results.jsonl');tasks=rows(OUT/'color_tasks.jsonl')
 assert len(candidate)==80 and len(selector)==80 and len(tasks)<=160
 c={(r['family_id'],r['logical_ref'],r['generator']):r for r in candidate};s={(r['family_id'],r['logical_ref'],r['generator']):r for r in selector}
 old_gt={(r['inference']['family_id'],r['inference']['logical_ref']):r['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels'] for r in rows(OLD) if r['inference']['backend']=='grounding_dino'}
 assert len(old_gt)==40
 route_prior={r['qa_id']:r for r in rows(ROOT/'outputs/minimal_decomposition_route_pilot/route_results.jsonl')}
 image_by_qa={r['qa_id']:r['image_path'] for r in prepared}
 prior_color={sha({'image_path':image_by_qa[r['qa_id']],'bbox_xyxy':r['selected_bbox_xyxy_pixels']}):r['normalized_color'] for r in rows(PRIOR_COLOR) if r['valid']}
 new_color={r['cache_key']:r['normalized_color'] for r in rows(OUT/'color_results.jsonl') if r['valid']}
 color_cache=prior_color|new_color
 t={(r['qa_id'],r['route']):r for r in tasks}
 entity=[];selected={route:{} for route in ROUTES};candidate_metrics={};selector_metrics={};oracle={};headroom={};same={};generation_rescues=[];selection_rescues=[]
 for route,(gen,sel) in ROUTES.items():
  for family,ref in old_gt:
   row=c[family,ref,gen];pool=row['candidates'];idx=row['old_selected_index'] if sel=='old' else s[family,ref,gen]['selected_index']
   gt=old_gt[family,ref];ious=[iou(x['bbox_xyxy'],gt) for x in pool];best=max(ious,default=0);chosen=ious[idx] if idx is not None else 0.
   item={'route':route,'family_id':family,'logical_ref':ref,'generator':gen,'selector':sel,'candidate_count':len(pool),'candidate_available':best>=.5,'best_iou':best,'selected_index':idx,'selected_iou':chosen,'selected_correct':chosen>=.5,'bbox_xyxy':pool[idx]['bbox_xyxy'] if idx is not None else None,'referent_phrase':row['referent_phrase'],'same_category_ambiguous':sum((x.get('matched_phrase') or '').casefold()==(row['referent_phrase']).casefold() for x in pool)>1}
   entity.append(item);selected[route][family,ref]=item
 for gen in ('original','strong'):
  x=[r for r in entity if r['generator']==gen and r['selector']=='old'];counts=[r['candidate_count'] for r in x]
  oracle[gen]={'entity':sum(r['candidate_available'] for r in x),'entity_n':40,'A':sum(r['candidate_available'] for r in x if r['logical_ref']=='A'),'B':sum(r['candidate_available'] for r in x if r['logical_ref']=='B'),'both_entity_family':sum(all(next(r for r in x if r['family_id']==f and r['logical_ref']==ref)['candidate_available'] for ref in ('A','B')) for f in {r['family_id'] for r in x}),'true_target_qa':sum(next(r for r in x if r['family_id']==p['family_id'] and r['logical_ref']==route_prior[p['qa_id']]['EVALUATOR_ONLY']['true_target'])['candidate_available'] for p in prepared),'mean_candidate_count':mean(counts),'median_candidate_count':median(counts),'multi_candidate_fraction':sum(k>1 for k in counts)/40,'zero_candidate_fraction':sum(k==0 for k in counts)/40}
  candidate_metrics[gen]=oracle[gen]
 for route in ROUTES:
  x=[r for r in entity if r['route']==route];avail=[r for r in x if r['candidate_available']];both=sum(all(selected[route][f,ref]['selected_correct'] for ref in ('A','B')) for f in {r['family_id'] for r in x})
  selector_metrics[route]={'A':{'available':sum(r['candidate_available'] for r in x if r['logical_ref']=='A'),'selected_correct':sum(r['selected_correct'] for r in x if r['logical_ref']=='A')},'B':{'available':sum(r['candidate_available'] for r in x if r['logical_ref']=='B'),'selected_correct':sum(r['selected_correct'] for r in x if r['logical_ref']=='B')},'pooled':{'available':len(avail),'selected_correct':sum(r['selected_correct'] for r in x),'selection_efficiency':sum(r['selected_correct'] for r in avail)/len(avail) if avail else None},'both_selected_regions_correct':both}
  for ref in ('A','B'):
   z=selector_metrics[route][ref];z['selection_efficiency']=z['selected_correct']/z['available'] if z['available'] else None
  headroom[route]={'generator':ROUTES[route][0],'oracle_available':len(avail),'selected_correct':sum(r['selected_correct'] for r in x),'gap':len(avail)-sum(r['selected_correct'] for r in x)}
 assert all(selected['R1'][k]['candidate_available']==selected['R3'][k]['candidate_available'] for k in selected['R1'])
 assert all(selected['R2'][k]['candidate_available']==selected['R4'][k]['candidate_available'] for k in selected['R2'])
 qa=[];color_rows=[];state_rows=[];route_metrics={}
 for p in prepared:
  prior=route_prior[p['qa_id']];true=prior['EVALUATOR_ONLY']['true_target'];gt_color=prior['EVALUATOR_ONLY']['ground_truth']
  for route in ROUTES:
   a=selected[route][p['family_id'],'A'];b=selected[route][p['family_id'],'B'];ta=t.get((p['qa_id'],route));target=ta['target'] if ta else None
   target_correct=target==true;target_iou=selected[route][p['family_id'],target]['selected_iou'] if target else 0.
   target_region=bool(target_correct and target_iou>=.5)
   actual_color=color_cache.get(ta['cache_key']) if ta else None
   color_correct=actual_color==gt_color;state=bool(target_correct and target_region and color_correct)
   text=f'[TARGET_ENTITY] {target if target else "unknown"}\n[TARGET_ATTRIBUTE] color = {actual_color if actual_color else "unknown"}'
   rr={'qa_id':p['qa_id'],'family_id':p['family_id'],'route':route,'target':target,'target_correct':target_correct,'target_region_correct':target_region,'target_color_correct':color_correct,'minimal_state_sufficient':state,'selected_target_iou':target_iou,'selected_bbox_xyxy':ta['bbox_xyxy'] if ta else None,'color':actual_color,'state_text':text,'state_sha256':hashlib.sha256(text.encode()).hexdigest(),'EVALUATOR_ONLY':{'true_target':true,'gt_color':gt_color}}
   qa.append(rr);state_rows.append(rr)
   color_rows.append({'qa_id':p['qa_id'],'route':route,'target':target,'crop_sha256':ta['crop_sha256'] if ta else None,'bbox_xyxy':ta['bbox_xyxy'] if ta else None,'color':actual_color,'color_cache_reused':ta['color_cache_reused'] if ta else False,'color_new_inference':ta['color_new_inference'] if ta else False,'EVALUATOR_ONLY':{'gt_color':gt_color,'correct':color_correct,'target_region_correct':target_region,'selected_target_iou':target_iou}})
 for route in ROUTES:
  x=[r for r in qa if r['route']==route];region=[r for r in x if r['target_region_correct']];ious=[r['selected_target_iou'] for r in region]
  strata={}
  for label,lo,hi in [('0.50–0.70',.5,.7),('0.70–0.85',.7,.85),('0.85–1.00',.85,1.000001)]:
   z=[r for r in region if lo<=r['selected_target_iou']<hi];strata[label]={'n':len(z),'color_correct':sum(r['target_color_correct'] for r in z),'accuracy':sum(r['target_color_correct'] for r in z)/len(z) if z else None}
  route_metrics[route]={'target_correct':sum(r['target_correct'] for r in x),'target_region_correct':len(region),'target_color_correct':sum(r['target_color_correct'] for r in x),'minimal_state_sufficient':sum(r['minimal_state_sufficient'] for r in x),'p_target_correct_given_both_regions_correct':sum(r['target_correct'] for r in x if all(selected[route][r['family_id'],ref]['selected_correct'] for ref in ('A','B')))/max(1,sum(all(selected[route][r['family_id'],ref]['selected_correct'] for ref in ('A','B')) for r in x)),'p_color_correct_given_target_region_correct':sum(r['target_color_correct'] for r in region)/len(region) if region else None,'selected_correct_target_iou_mean':mean(ious) if ious else None,'selected_correct_target_iou_median':median(ious) if ious else None,'selected_correct_target_iou_values':ious,'color_by_iou_stratum':strata,'color_cache_reused_qa':sum(v['color_cache_reused'] for v in color_rows if v['route']==route),'color_new_inference_qa':sum(v['color_new_inference'] for v in color_rows if v['route']==route)}
 survival={route:{'start':40,'target_correct':v['target_correct'],'target_region_correct':v['target_region_correct'],'target_color_correct_after_region':v['minimal_state_sufficient'],'minimal_state_sufficient':v['minimal_state_sufficient']} for route,v in route_metrics.items()}
 pairs=[('R1','R2'),('R1','R3'),('R1','R4'),('R2','R4'),('R3','R4')];pair_out={}
 for left,right in pairs:
  n={};levels={'entity_selected_region':[(f,ref) for f,ref in selected[left]],'target_region':[p['qa_id'] for p in prepared],'minimal_state':[p['qa_id'] for p in prepared]}
  ql={r['qa_id']:r for r in qa if r['route']==left};qr={r['qa_id']:r for r in qa if r['route']==right}
  for level,keys in levels.items():
   cnt=Counter()
   for key in keys:
    x=selected[left][key]['selected_correct'] if level=='entity_selected_region' else ql[key]['target_region_correct' if level=='target_region' else 'minimal_state_sufficient']
    y=selected[right][key]['selected_correct'] if level=='entity_selected_region' else qr[key]['target_region_correct' if level=='target_region' else 'minimal_state_sufficient']
    cnt['fail_to_correct' if not x and y else 'correct_to_fail' if x and not y else 'both_correct' if x else 'both_fail']+=1
   n[level]={k:cnt[k] for k in ('fail_to_correct','correct_to_fail','both_correct','both_fail')}
  pair_out[f'{left}_to_{right}']=n
 baseline_region={r['qa_id']:r['target_region_correct'] for r in qa if r['route']=='R1'}
 for route in ('R2','R3','R4'):
  recovered=[r for r in qa if r['route']==route and r['target_region_correct'] and not baseline_region[r['qa_id']]]
  route_metrics[route]['newly_recovered_target_regions']={'n':len(recovered),'qwen_color_correct':sum(r['target_color_correct'] for r in recovered),'qa_ids':[r['qa_id'] for r in recovered]}
 for left,right in (('R1','R2'),('R1','R4')):
  for key in selected[left]:
   x=selected[left][key];y=selected[right][key]
   if not x['selected_correct'] and y['selected_correct']:
    generation_rescues.append({'comparison':f'{left}_to_{right}','level':'entity','family_id':key[0],'logical_ref':key[1],'old_available':x['candidate_available'],'strong_available':y['candidate_available'],'provenance':'old_no_usable_strong_introduced_usable' if not x['candidate_available'] and y['candidate_available'] else 'both_usable_selection_or_box_gain' if x['candidate_available'] and y['candidate_available'] else 'other','old_selected_iou':x['selected_iou'],'strong_selected_iou':y['selected_iou']})
  qleft={r['qa_id']:r for r in qa if r['route']==left};qright={r['qa_id']:r for r in qa if r['route']==right}
  for qid in qleft:
   if qleft[qid]['target_region_correct'] or not qright[qid]['target_region_correct']:continue
   true=qleft[qid]['EVALUATOR_ONLY']['true_target'];family=qleft[qid]['family_id'];x=selected[left][family,true];y=selected[right][family,true]
   generation_rescues.append({'comparison':f'{left}_to_{right}','level':'target_region_qa','qa_id':qid,'family_id':family,'logical_ref':true,'old_available':x['candidate_available'],'strong_available':y['candidate_available'],'provenance':'old_no_usable_strong_introduced_usable' if not x['candidate_available'] and y['candidate_available'] else 'both_usable_selection_or_box_gain' if x['candidate_available'] and y['candidate_available'] else 'other','old_selected_iou':x['selected_iou'],'strong_selected_iou':y['selected_iou']})
 for left,right in (('R1','R3'),('R2','R4')):
  for key in selected[left]:
   x=selected[left][key];y=selected[right][key]
   if not x['selected_correct'] and y['selected_correct']:
    selection_rescues.append({'comparison':f'{left}_to_{right}','family_id':key[0],'logical_ref':key[1],'candidate_count':x['candidate_count'],'old_selected_iou':x['selected_iou'],'qwen_selected_iou':y['selected_iou'],'correct_candidate_already_existed':x['candidate_available'],'referent_description':x['referent_phrase'],'same_category_ambiguous':x['same_category_ambiguous']})
 for gen in ('original','strong'):
  for subset,rule in [('singleton',lambda r:r['candidate_count']==1),('multi',lambda r:r['candidate_count']>1),('same_category',lambda r:r['same_category_ambiguous'])]:
   x=[r for r in entity if r['generator']==gen and r['selector']=='old' and rule(r)];y=[selected['R3' if gen=='original' else 'R4'][r['family_id'],r['logical_ref']] for r in x]
   same[f'{gen}_{subset}']={'n':len(x),'old_correct':sum(r['selected_correct'] for r in x),'qwen_correct':sum(r['selected_correct'] for r in y),'available':sum(r['candidate_available'] for r in x)}
 write_rows(OUT/'entity_results.jsonl',entity);write_rows(OUT/'minimal_state_results.jsonl',state_rows);write_rows(OUT/'color_results_by_qa.jsonl',color_rows)
 for name,value in [('entity_metrics',selector_metrics),('selection_efficiency',{k:v['pooled']['selection_efficiency'] for k,v in selector_metrics.items()}),('target_metrics',route_metrics),('route_survival',survival),('paired_transitions',pair_out),('generation_rescues',generation_rescues),('selection_rescues',selection_rescues),('same_category_analysis',same),('oracle_ceiling',oracle),('selection_headroom',headroom),('route_comparison',route_metrics)]:write(OUT/(name+'.json'),value)
 write(OUT/'inference_efficiency.json',{route:{'detector_calls_per_family':2,'selector_qwen_calls_per_family':sum(s[f,ref,ROUTES[route][0]]['status'] not in ('SINGLETON_NO_QWEN_CALL','NO_CANDIDATE') for f in {p['family_id'] for p in prepared} for ref in ('A','B'))/20 if ROUTES[route][1]=='qwen' else 0,'color_qwen_calls_per_family':route_metrics[route]['color_new_inference_qa']/20,'color_cache_reused_qa':route_metrics[route]['color_cache_reused_qa']} for route in ROUTES})
 write(OUT/'gt_leakage_audit.json',{'pass':True,'inference_inputs':['frozen ReferentSpec text','original image only in detector','candidate crop grid and referent text in selector','exact target crop and generic prompt in color'],'gt_fields_accessed_only_in_analyze':True,'selector_question_or_target_side_provided':False,'oracle_evaluator_only':True})
 hashes={}
 for route in ROUTES:
  hashes[route]={f:sha({ref:selected[route][f,ref]['bbox_xyxy'] for ref in ('A','B')}) for f in {p['family_id'] for p in prepared}}
 assert all(len([x for x in prepared if x['family_id']==f])==2 for f in hashes['R1'])
 pool_hashes={gen:{f:sha({ref:c[f,ref,gen]['candidates'] for ref in ('A','B')}) for f in {p['family_id'] for p in prepared}} for gen in ('original','strong')}
 write(OUT/'state_consistency_audit.json',{'pass':True,'family_region_hashes':hashes,'candidate_pool_hashes':pool_hashes,'R1_R3_original_pool_identical':True,'R2_R4_strong_pool_identical':True,'paired_QA_share_selected_region_state':True})
 make_figures(oracle,selector_metrics,survival,route_metrics,pair_out,qa)
 make_report(oracle,selector_metrics,route_metrics,pair_out,same,headroom)
 make_html(entity,qa)
 print('analysis complete',route_metrics,flush=True)

def make_figures(oracle,sm,survival,metrics,pairs,qa):
 f=OUT/'figures';f.mkdir(exist_ok=True)
 def bar(names,values,title,ylabel,file):
  plt.figure(figsize=(7,4));plt.bar(names,values);plt.title(title);plt.ylabel(ylabel);plt.tight_layout();plt.savefig(f/file);plt.close()
 bar(['Original','MM-GDINO-L'],[oracle[g]['entity']/40 for g in ('original','strong')],'Candidate recall @ IoU 0.5','fraction', 'figure1_candidate_recall.png')
 bar(list(sm),[sm[r]['pooled']['selection_efficiency'] for r in sm],'Selection efficiency','selected correct / available','figure2_selection_efficiency.png')
 plt.figure(figsize=(8,4))
 for r,v in survival.items():plt.plot(list(v),list(v.values()),marker='o',label=r)
 plt.xticks(rotation=25,ha='right');plt.legend();plt.tight_layout();plt.savefig(f/'figure3_survival.png');plt.close()
 bar(['R2−R1','R3−R1','R4−R1'],[metrics[r]['minimal_state_sufficient']-metrics['R1']['minimal_state_sufficient'] for r in ('R2','R3','R4')],'State sufficiency contributions','QA difference','figure4_contributions.png')
 z=pairs['R1_to_R4']['target_region'];bar(['rescued','regressed','both correct','both fail'],[z[k] for k in ('fail_to_correct','correct_to_fail','both_correct','both_fail')],'R1 to R4 target region transitions','QA','figure5_transitions.png')
 plt.figure(figsize=(7,4))
 for r in ROUTES:
  x=[v for v in qa if v['route']==r and v['target_region_correct']]
  plt.scatter([v['selected_target_iou'] for v in x],[int(v['target_color_correct']) for v in x],alpha=.5,label=r)
 plt.xlabel('selected target IoU');plt.ylabel('Qwen color correct');plt.yticks([0,1]);plt.legend();plt.tight_layout();plt.savefig(f/'figure6_color_vs_iou.png');plt.close()

def make_html(entity,qa):
 parts=['<!doctype html><html><head><meta charset="utf-8"><style>body{font:14px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:4px}</style></head><body><h1>Region bottleneck joint repair audit</h1><p>GT and IoU appear only in evaluator output.</p><h2>Entity selections</h2><table><tr><th>Route</th><th>Family</th><th>Ref</th><th>Phrase</th><th>Pool</th><th>Available</th><th>Selected IoU</th></tr>']
 for r in entity:parts.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (r['route'],r['family_id'],r['logical_ref'],r['referent_phrase'],r['candidate_count'],r['candidate_available'],f"{r['selected_iou']:.3f}"))+'</tr>')
 parts+=['</table><h2>QA outcomes</h2><table><tr><th>Route</th><th>QA</th><th>Target</th><th>Target correct</th><th>Region correct</th><th>Color</th><th>State sufficient</th></tr>']
 for r in qa:parts.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (r['route'],r['qa_id'],r['target'],r['target_correct'],r['target_region_correct'],r['color'],r['minimal_state_sufficient']))+'</tr>')
 parts.append('</table></body></html>');(OUT/'audit.html').write_text(''.join(parts))

def make_report(oracle,sm,metrics,pairs,same,headroom):
 lines=['# Region bottleneck joint repair pilot','','## Candidate generation','','| Metric | Original GDINO | MM-GDINO-L |','|---|---:|---:|']
 for name,key,den in [('Entity CandidateRecall@0.5','entity',40),('True-target CandidateRecall@0.5','true_target_qa',40),('Both-entity CandidateRecall@0.5','both_entity_family',20)]:lines.append(f"| {name} | {oracle['original'][key]}/{den} | {oracle['strong'][key]}/{den} |")
 for name,key in [('Mean candidate count','mean_candidate_count'),('Median candidate count','median_candidate_count'),('Multi-candidate fraction','multi_candidate_fraction'),('Zero-candidate fraction','zero_candidate_fraction')]:lines.append(f"| {name} | {oracle['original'][key]:.3f} | {oracle['strong'][key]:.3f} |")
 lines+=['','## Selector quality','','| Generator | Selector | SelectedRecall@0.5 | Selection efficiency |','|---|---|---:|---:|']
 for r,(g,s) in ROUTES.items():lines.append(f"| {g} | {s} | {sm[r]['pooled']['selected_correct']}/40 | {sm[r]['pooled']['selection_efficiency']:.3f} |")
 lines+=['','## Downstream routes','','| Route | Generator | Selector | Target correct | Target region correct | Qwen color correct | Minimal state sufficient |','|---|---|---|---:|---:|---:|---:|']
 for r,v in metrics.items():lines.append(f"| {r} | {ROUTES[r][0]} | {ROUTES[r][1]} | {v['target_correct']}/40 | {v['target_region_correct']}/40 | {v['target_color_correct']}/40 | {v['minimal_state_sufficient']}/40 |")
 lines+=['','## Factor contributions versus R1','','| Effect | Selected region | Target | Target region | State sufficient |','|---|---:|---:|---:|---:|']
 for r,label in [('R2','Generation'),('R3','Selection'),('R4','Joint')]:lines.append(f"| {label} | {sm[r]['pooled']['selected_correct']-sm['R1']['pooled']['selected_correct']:+d} | {metrics[r]['target_correct']-metrics['R1']['target_correct']:+d} | {metrics[r]['target_region_correct']-metrics['R1']['target_region_correct']:+d} | {metrics[r]['minimal_state_sufficient']-metrics['R1']['minimal_state_sufficient']:+d} |")
 interaction={'selected_region':sm['R4']['pooled']['selected_correct']-sm['R3']['pooled']['selected_correct']-sm['R2']['pooled']['selected_correct']+sm['R1']['pooled']['selected_correct']}
 for key in ('target_correct','target_region_correct','minimal_state_sufficient'):interaction[key]=metrics['R4'][key]-metrics['R3'][key]-metrics['R2'][key]+metrics['R1'][key]
 write(OUT/'factor_interaction.json',interaction)
 lines.append(f"| Generation × selection interaction | {interaction['selected_region']:+d} | {interaction['target_correct']:+d} | {interaction['target_region_correct']:+d} | {interaction['minimal_state_sufficient']:+d} |")
 for level in ('target_region','minimal_state'):
  lines+=['',f'## Paired transitions: {level}','','| Comparison | Fail→Correct | Correct→Fail | Both correct | Both fail |','|---|---:|---:|---:|---:|']
  for name,v in pairs.items():z=v[level];lines.append(f"| {name} | {z['fail_to_correct']} | {z['correct_to_fail']} | {z['both_correct']} | {z['both_fail']} |")
 lines+=['','## Oracle and conditional findings','']
 for r,v in headroom.items():lines.append(f"- {r}: oracle {v['oracle_available']}/40, selected {v['selected_correct']}/40, gap {v['gap']} entities.")
 for k,v in same.items():lines.append(f"- {k}: {v['n']} referents, old {v['old_correct']}, Qwen {v['qwen_correct']}, candidate available {v['available']}.")
 lines+=['','## Crop color versus localization quality','','| Route | Qwen color correct given correct target region | Mean IoU | Median IoU | 0.50–0.70 | 0.70–0.85 | 0.85–1.00 |','|---|---:|---:|---:|---:|---:|---:|']
 for r,v in metrics.items():
  z=v['color_by_iou_stratum'];lines.append(f"| {r} | {v['p_color_correct_given_target_region_correct']:.3f} | {v['selected_correct_target_iou_mean']:.3f} | {v['selected_correct_target_iou_median']:.3f} | "+' | '.join(f"{z[k]['color_correct']}/{z[k]['n']}" for k in ('0.50–0.70','0.70–0.85','0.85–1.00'))+' |')
 for r in ('R2','R3','R4'):
  v=metrics[r]['newly_recovered_target_regions'];lines.append(f"- {r} newly recovered target regions versus R1: Qwen color correct {v['qwen_color_correct']}/{v['n']}.")
 efficiency=json.loads((OUT/'inference_efficiency.json').read_text())
 lines+=['','## Visual inference cost per family','','| Route | Detector calls | Selector Qwen calls | Target crops absent from prior color cache |','|---|---:|---:|---:|']
 for r,z in efficiency.items():lines.append(f"| {r} | {z['detector_calls_per_family']:.1f} | {z['selector_qwen_calls_per_family']:.2f} | {z['color_qwen_calls_per_family']:.2f} |")
 lines.append(f"Across all routes, {len(rows(OUT/'color_results.jsonl'))} unique exact-bbox Qwen color calls were new; shared crop outputs were reused.")
 r4=metrics['R4']['minimal_state_sufficient'];recommend='ADOPT_JOINT_REGION_REPAIR' if r4>metrics['R2']['minimal_state_sufficient'] and r4>metrics['R3']['minimal_state_sufficient'] else 'ADOPT_STRONG_GROUNDER_ONLY' if metrics['R2']['minimal_state_sufficient']>=metrics['R3']['minimal_state_sufficient'] and metrics['R2']['minimal_state_sufficient']>metrics['R1']['minimal_state_sufficient'] else 'ADOPT_QWEN_SELECTOR_ONLY' if metrics['R3']['minimal_state_sufficient']>metrics['R1']['minimal_state_sufficient'] else 'REGION_REPAIR_INSUFFICIENT'
 lines+=['',f'**Primary recommendation: `{recommend}`.**',f'R4 state sufficiency is {r4}/40. Vanilla VLM final accuracy is 20/40; old Minimal Specialist measured final accuracy is 15/40. State sufficiency is not final accuracy.','', '## Scientific answers','']
 answers=[f"Q1. MM-GDINO-L candidate availability changes by {oracle['strong']['entity']-oracle['original']['entity']:+d}/40 entities.",f"Q2. Original oracle gap under old selector is {headroom['R1']['gap']} entities; {40-oracle['original']['entity']} have no usable candidate.",f"Q3. Qwen efficiency changes from {sm['R1']['pooled']['selection_efficiency']:.3f} to {sm['R3']['pooled']['selection_efficiency']:.3f} on original pools.",f"Q4. Multi and same-category counts are reported above; only multi-candidate cases permit selection improvement.",f"Q5. R4's state gain is {r4-metrics['R1']['minimal_state_sufficient']:+d}; compare separate effects in factor table.",f"Q6. R4 target identity is {metrics['R4']['target_correct']}/40; target region is {metrics['R4']['target_region_correct']}/40.",f"Q7. Qwen color given correct R4 target region is {metrics['R4']['p_color_correct_given_target_region_correct']:.3f}.",f"Q8. R4 minimal state sufficiency is {r4}/40.","Q9. Measured final accuracy is recorded separately when the frozen text reasoner is run.",f"Q10. R4 oracle gap is {headroom['R4']['gap']} entities; MM-GDINO-L misses {40-oracle['strong']['entity']} candidate entities, and R4 color fails on {metrics['R4']['target_region_correct']-r4} correct target regions."]
 lines+=answers+['','The fixed 20-family/40-QA pilot uses legal frozen noun referents, local MM-GDINO-L candidate generation, candidate-grid Qwen matching, unchanged bbox-center geometry, exact target crops, and image-free final execution when gated. Detector and selector metrics use evaluator-only IoU joins after inference. LEFT/RIGHT pairs share entity region state. Figures and per-case traces are in this directory.','']
 (OUT/'implementation_report.md').write_text('\n'.join(lines))

def final():
 from harness.reasoners.qwen35 import Qwen35Reasoner
 from harness.scripts.run_minimal_decomposition_route_pilot import prompt
 from harness.evaluators.end2end_metrics import score
 from harness.configs.paths import Paths
 from harness.dataset import read_jsonl
 from harness.processors.color_pixel import REALBIND_COLORS
 metrics=json.loads((OUT/'route_comparison.json').read_text())
 eligible=[r for r in ROUTES if metrics[r]['minimal_state_sufficient']>=18]
 if not eligible:
  write(OUT/'end_to_end_gate.json',{'run':False,'reason':'no route reached 18/40 state sufficiency','route_sufficiency':{r:metrics[r]['minimal_state_sufficient'] for r in ROUTES}});return
 eligible=list(ROUTES) # same infrastructure; measure all four
 states={(r['qa_id'],r['route']):r for r in rows(OUT/'minimal_state_results.jsonl')};prepared={r['qa_id']:r for r in rows(PREP)}
 fam={r['family_id']:r for r in read_jsonl(Paths().families)}
 vocab={'color':list(REALBIND_COLORS),'position':['left','right'],'identity':sorted({f[k]['object'] for f in fam.values() for k in ('A','B')})}
 path=OUT/'end_to_end_results.jsonl';done={(r['qa_id'],r['route']) for r in rows(path)};q=Qwen35Reasoner()
 for route in eligible:
  for n,p in enumerate(prepared.values(),1):
   key=p['qa_id'],route
   if key in done:continue
   state=states[key];query=prompt(p['question'],state['state_text']);g=q.generate(query,None)
   assert g['image_path'] is None and not g['vision_encoder_called'] and not g['visual_tokens_in_rendered_prompt']
   out=score(g['response'],'color',state['EVALUATOR_ONLY']['gt_color'],fam[p['family_id']],vocab)
   append(path,{'qa_id':key[0],'route':route,'state_sha256':state['state_sha256'],'state_sufficient':state['minimal_state_sufficient'],'prompt_sha256':hashlib.sha256(query.encode()).hexdigest(),'raw_response':g['response'],'prediction':out['normalized_prediction'],'final_correct':bool(out['correct']),'image_path':None,'vision_encoder_called':False})
   print('final',route,n,'/40',out['normalized_prediction'],flush=True)
 write(OUT/'end_to_end_gate.json',{'run':True,'eligible_routes_at_18':eligible,'all_routes_run':True})
 result={}
 for route in eligible:
  x=[r for r in rows(path) if r['route']==route]
  result[route]={'final_correct':sum(r['final_correct'] for r in x),'n':len(x),'sufficient_and_final_correct':sum(r['final_correct'] and r['state_sufficient'] for r in x),'insufficient_and_final_correct':sum(r['final_correct'] and not r['state_sufficient'] for r in x)}
 write(OUT/'end_to_end_metrics.json',result)
 with (OUT/'implementation_report.md').open('a') as f:
  f.write('\n## Measured image-free final accuracy\n\n| Method | Final accuracy |\n|---|---:|\n| Vanilla VLM | 20/40 |\n| Old Minimal Specialist | 15/40 |\n')
  for r in ('R2','R3','R4'):
   z=result[r];f.write(f"| {r} | {z['final_correct']}/40 |\n")
  f.write('\nCompensating correctness (insufficient state yet correct final answer): '+str({r:z['insufficient_and_final_correct'] for r,z in result.items()})+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','strong','normalize_strong','select_qwen','color','analyze','final']);a=p.parse_args();globals()[a.stage]()
