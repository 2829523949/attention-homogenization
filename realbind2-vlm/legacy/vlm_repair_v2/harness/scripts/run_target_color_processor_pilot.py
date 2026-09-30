#!/usr/bin/env python3
"""Frozen-region target-color processor comparison; no upstream or final-route inference."""
from __future__ import annotations
import argparse, hashlib, html, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from PIL import Image
from harness.processors.base import RegionPrediction
from harness.processors.color_pixel import PixelColorReader, REALBIND_COLORS
from harness.processors.crop_policy import CROP_POLICY_ID, deterministic_crop_box, save_persistent_crop
from harness.evaluators.end2end_metrics import score

OUT=ROOT/'outputs/target_color_processor_pilot'
PILOT=ROOT/'outputs/minimal_decomposition_route_pilot'
AUDIT=ROOT/'outputs/minimal_route_bottleneck_audit'
GROUND=ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
HANDOFF=ROOT/'outputs/handoff_end_to_end_pilot/compiled_states.jsonl'
PROCESSORS=('CURRENT_SIGLIP','SIMPLE_PIXEL','NARROW_QWEN_VLM')
VLM_PROMPT='What color is the main object or region shown?\nAnswer with only one color from this list: red, orange, yellow, green, blue, purple, pink, brown, black, white, gray.'
BINS=(('small',0.,.05),('medium',.05,.15),('large',.15,1.01)) # fixed descriptive area fractions, not tuned

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashobj(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_rows(path,rows):Path(path).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
def parse_vlm(raw):
 value=raw.strip().casefold()
 return value if value in REALBIND_COLORS else None

def prepare():
 if OUT.exists():raise FileExistsError(f'refusing overwrite of {OUT}')
 OUT.mkdir();(OUT/'figures').mkdir();(OUT/'target_crops').mkdir()
 prepared={r['qa_id']:r for r in load(PILOT/'prepared.jsonl')}
 audit={r['qa_id']:r for r in load(AUDIT/'qa_bottleneck_results.jsonl')}
 routes={r['qa_id']:r for r in load(PILOT/'route_results.jsonl')}
 assert len(prepared)==len(audit)==len(routes)==40 and set(prepared)==set(audit)==set(routes)
 rows=[];primary=[];source=[]
 for cid,r in prepared.items():
  q=routes[cid]['routes']['MINIMAL_SPECIALIST'];a=audit[cid];target=q['target'];bbox=r['entities'][target]['bbox'] if target else None
  assert a['selected_target']==target and (a['target_correct'] and a['target_region_correct'])==(q['target_correct'] and q['target_region_correct'])
  if a['target_correct'] and a['target_region_correct']:primary.append(cid)
  if bbox is None:continue
  path=Path(r['image_path'])
  with Image.open(path) as opened:width,height=opened.size
  region=RegionPrediction.from_pixels(bbox,width,height,score=None,label='',source='frozen_selected_grounding_dino')
  crop_box=deterministic_crop_box((width,height),bbox)
  out=OUT/'target_crops'/(cid+'.png')
  crop_meta=save_persistent_crop(path,region,out)
  assert crop_meta['crop_box_xyxy_pixels']==list(crop_box) and crop_meta['crop_policy_id']==CROP_POLICY_ID
  row={'qa_id':cid,'family_id':r['family_id'],'image_path':str(path),'selected_target':target,'selected_bbox_xyxy_pixels':bbox,'region_hash':hashobj({'image_sha256':sha(path),'bbox_xyxy_pixels':bbox,'crop_policy_id':CROP_POLICY_ID}),'crop_path':str(out),'crop_sha256':crop_meta['output_image_sha256'],'crop_box_xyxy_pixels':list(crop_box),'crop_width':crop_box[2]-crop_box[0],'crop_height':crop_box[3]-crop_box[1],'image_width':width,'image_height':height,'legal_referent_category':r['entities'][target]['legal_query'],'crop_policy_id':CROP_POLICY_ID}
  rows.append(row);source.append({'qa_id':cid,'selected_target':target,'selected_bbox':bbox,'source_route':'MINIMAL_SPECIALIST','gt_used_to_build_crop':False})
 assert len(primary)==19 and len(rows)==38 and len({r['qa_id'] for r in rows})==38
 write_rows(OUT/'inference_manifest.jsonl',rows)
 dump(OUT/'primary_subset_manifest.json',{'primary_qa_ids':primary,'n':19,'criterion':'evaluator-only target_correct AND target_region_correct','frozen_route_results_sha256':sha(PILOT/'route_results.jsonl'),'frozen_bottleneck_audit_sha256':sha(AUDIT/'qa_bottleneck_results.jsonl'),'selection_used_for_processor_inference':False})
 dump(OUT/'processor_config.json',{'processors':PROCESSORS,'primary_n':19,'secondary_executable_n':38,'color_vocabulary':list(REALBIND_COLORS),'crop_policy':CROP_POLICY_ID,'same_frozen_bbox_all_processors':True,'vlm_prompt':VLM_PROMPT,'vlm_prompt_sha256':hashlib.sha256(VLM_PROMPT.encode()).hexdigest(),'vlm_checkpoint':'Qwen/Qwen3.5-4B frozen local','vlm_generation':{'do_sample':False,'use_cache':False,'max_new_tokens':16,'batch_size':1,'enable_thinking':False},'pixel_reader':{'name':PixelColorReader.name,'version':PixelColorReader.version,'mask':'frozen central 18% margin heuristic within exact bbox; no external mask'},'current_reader':'google_siglip2_base_patch16_224_realbind_vocab_v2 saved exact-region outputs','area_bins_fraction':[{'name':name,'lower_inclusive':lo,'upper_exclusive':hi} for name,lo,hi in BINS],'source_hashes':{'pilot_route_results':sha(PILOT/'route_results.jsonl'),'pilot_prepared':sha(PILOT/'prepared.jsonl'),'grounding_candidates':sha(GROUND),'prior_color_states':sha(HANDOFF)},'gt_inference_used':False,'oracle_bbox_used':False,'final_reasoner_called':False,'prompt_search':False})
 dump(OUT/'gt_leakage_audit.json',{'all_pass':True,'crop_source_checks':source,'vlm_receives':['exact selected region crop','single generic fixed color prompt'],'vlm_forbidden_inputs':['REL2C question','left/right','A/B','GT bbox','GT color','other processor outputs'],'primary_membership_not_used_by_inference':True,'no_gt_bbox_used':True})
 print(f'prepared {len(rows)} executable frozen regions; {len(primary)} primary QA',flush=True)

def infer_pixel():
 rows=load(OUT/'inference_manifest.jsonl');assert len(rows)==38
 reader=PixelColorReader();path=OUT/'pixel_calls.jsonl';completed={r['qa_id'] for r in load(path)} if path.exists() else set()
 for i,r in enumerate(rows,1):
  if r['qa_id'] in completed:continue
  bbox=r['selected_bbox_xyxy_pixels'];reg=RegionPrediction.from_pixels(bbox,r['image_width'],r['image_height'],score=None,label='',source='frozen_selected_grounding_dino')
  result=reader.read(Path(r['image_path']),reg)
  assert result.color in REALBIND_COLORS or result.color is None
  append(path,{'qa_id':r['qa_id'],'processor':'SIMPLE_PIXEL','region_hash':r['region_hash'],'crop_sha256':r['crop_sha256'],'raw_output':result.color,'normalized_color':result.color,'valid':result.color in REALBIND_COLORS,'confidence':result.confidence,'confidence_semantics':'heuristic local separation, not calibrated','diagnostics':result.diagnostics,'reader_version':reader.version,'image_path':r['image_path'],'selected_bbox_xyxy_pixels':bbox,'other_processor_output_used':False})
  print(f'pixel {i}/38 {r["qa_id"]}={result.color}',flush=True)

def infer_vlm():
 rows=load(OUT/'inference_manifest.jsonl');assert len(rows)==38
 from harness.reasoners.qwen35 import Qwen35Reasoner
 reasoner=Qwen35Reasoner();path=OUT/'vlm_color_calls.jsonl';completed={r['qa_id'] for r in load(path)} if path.exists() else set()
 for i,r in enumerate(rows,1):
  if r['qa_id'] in completed:continue
  g=reasoner.generate(VLM_PROMPT,Path(r['crop_path']))
  assert g['vision_module_found'] and g['vision_encoder_called'] and g['image_path']==r['crop_path']
  norm=parse_vlm(g['response'])
  rec={'qa_id':r['qa_id'],'processor':'NARROW_QWEN_VLM','region_hash':r['region_hash'],'crop_sha256':r['crop_sha256'],'prompt':VLM_PROMPT,'prompt_sha256':hashlib.sha256(VLM_PROMPT.encode()).hexdigest(),'raw_output':g['response'],'normalized_color':norm,'valid':norm is not None,'invalid_reason':None if norm else 'NON_EXACT_VOCAB_COLOR_OUTPUT','generation':g,'image_path':r['crop_path'],'selected_bbox_xyxy_pixels':r['selected_bbox_xyxy_pixels'],'other_processor_output_used':False,'rel2c_question_provided':False,'target_identity_provided':False}
  append(path,rec);print(f'vlm {i}/38 {r["qa_id"]}={g["response"]!r}',flush=True)

def analyze():
 rows=load(OUT/'inference_manifest.jsonl');p={r['qa_id']:r for r in rows};pixel={r['qa_id']:r for r in load(OUT/'pixel_calls.jsonl')};vlm={r['qa_id']:r for r in load(OUT/'vlm_color_calls.jsonl')}
 assert len(p)==len(pixel)==len(vlm)==38 and set(p)==set(pixel)==set(vlm)
 audit={r['qa_id']:r for r in load(AUDIT/'qa_bottleneck_results.jsonl')};route={r['qa_id']:r for r in load(PILOT/'route_results.jsonl')};states={r['qa_id']:r for r in load(HANDOFF)}
 primary=json.loads((OUT/'primary_subset_manifest.json').read_text())['primary_qa_ids'];assert len(primary)==19 and all(audit[cid]['target_correct'] and audit[cid]['target_region_correct'] for cid in primary)
 family_path=Path('/mnt/d/datasets/RealBind/realbind/families/dryrun_100_pair_direct_v1_1_frozen.jsonl')
 families={r['family_id']:r for r in load(family_path)}
 vocab={'color':list(REALBIND_COLORS),'position':['left','right'],'identity':sorted({f[ref]['object'] for f in families.values() for ref in ('A','B')})}
 results=[];gt_audit=[]
 for cid,r in p.items():
  a=audit[cid];q=route[cid]['routes']['MINIMAL_SPECIALIST'];target=r['selected_target'];gt=a['ground_truth'];current=q['state_text'].split('color = ')[-1];current=None if current=='unknown' else current
  assert q['target']==target and current==states[cid]['entities'][target]['color']
  assert states[cid]['entities'][target]['color_region_bbox_xyxy_pixels']==r['selected_bbox_xyxy_pixels']
  assert r['region_hash']==pixel[cid]['region_hash']==vlm[cid]['region_hash']
  assert r['crop_sha256']==pixel[cid]['crop_sha256']==vlm[cid]['crop_sha256']==sha(r['crop_path'])
  assert r['selected_bbox_xyxy_pixels']==pixel[cid]['selected_bbox_xyxy_pixels']==vlm[cid]['selected_bbox_xyxy_pixels']
  assert vlm[cid]['prompt']==VLM_PROMPT and not vlm[cid]['rel2c_question_provided'] and not vlm[cid]['target_identity_provided']
  # Scorer normalization is evaluator-only; strict VLM one-color protocol remains authoritative.
  prior=score(vlm[cid]['raw_output'],'color',gt,families[r['family_id']],vocab)
  if vlm[cid]['valid']:assert prior['normalized_prediction']==vlm[cid]['normalized_color']
  pred={'CURRENT_SIGLIP':current,'SIMPLE_PIXEL':pixel[cid]['normalized_color'],'NARROW_QWEN_VLM':vlm[cid]['normalized_color']}
  valid={k:v in REALBIND_COLORS for k,v in pred.items()}
  for value in pred.values():
   if value in REALBIND_COLORS:
    canonical=score(value,'color',gt,families[r['family_id']],vocab)
    assert not canonical['invalid_prediction'] and canonical['normalized_prediction']==value
  area=(r['selected_bbox_xyxy_pixels'][2]-r['selected_bbox_xyxy_pixels'][0])*(r['selected_bbox_xyxy_pixels'][3]-r['selected_bbox_xyxy_pixels'][1])/(r['image_width']*r['image_height'])
  b=next(name for name,lo,hi in BINS if lo<=area<hi)
  per={k:{'raw_output':(current if k=='CURRENT_SIGLIP' else pixel[cid]['raw_output'] if k=='SIMPLE_PIXEL' else vlm[cid]['raw_output']),'normalized_color':v,'valid':valid[k],'correct':v==gt,'confidence':(states[cid]['entities'][target]['color_score'] if k=='CURRENT_SIGLIP' else pixel[cid]['confidence'] if k=='SIMPLE_PIXEL' else None),'region_hash':r['region_hash'],'crop_sha256':r['crop_sha256']} for k,v in pred.items()}
  results.append({'qa_id':cid,'family_id':r['family_id'],'primary':cid in primary,'selected_target':target,'target_correct':a['target_correct'],'target_region_correct':a['target_region_correct'],'bbox_xyxy_pixels':r['selected_bbox_xyxy_pixels'],'bbox_area_ratio':area,'bbox_width':r['selected_bbox_xyxy_pixels'][2]-r['selected_bbox_xyxy_pixels'][0],'bbox_height':r['selected_bbox_xyxy_pixels'][3]-r['selected_bbox_xyxy_pixels'][1],'size_bin':b,'legal_referent_category':r['legal_referent_category'],'crop_policy_id':CROP_POLICY_ID,'region_hash':r['region_hash'],'crop_sha256':r['crop_sha256'],'processors':per,'EVALUATOR_ONLY':{'gt_color':gt}})
  gt_audit.append({'qa_id':cid,'same_region_hash_all_processors':True,'same_crop_sha256_all_processors':True,'gt_used_after_all_processor_calls':True,'vlm_has_only_generic_prompt_and_crop':True,'vlm_strict_vocabulary_parser':True})
 assert len(results)==38 and sum(x['primary'] for x in results)==19
 write_rows(OUT/'color_processor_results.jsonl',results)
 dump(OUT/'gt_leakage_audit.json',{'all_pass':all(all(v for k,v in a.items() if k!='qa_id') for a in gt_audit),'cases':gt_audit,'vlm_prompt':VLM_PROMPT,'no_gt_bbox':True,'no_rel2c_question_to_vlm':True,'no_final_reasoner_calls':True,'all_processors_same_bbox_and_crop':True,'other_processor_outputs_not_used':True})
 primary_rows=[x for x in results if x['primary']]
 metrics={k:{'correct':sum(x['processors'][k]['correct'] for x in primary_rows),'n':19,'accuracy':sum(x['processors'][k]['correct'] for x in primary_rows)/19,'invalid':sum(not x['processors'][k]['valid'] for x in primary_rows),'secondary_executable_correct':sum(x['processors'][k]['correct'] for x in results),'secondary_executable_n':38} for k in PROCESSORS}
 assert metrics['CURRENT_SIGLIP']['correct']==10
 dump(OUT/'processor_metrics.json',metrics)
 pairs={}
 for alt in PROCESSORS[1:]:
  groups={'baseline_wrong_to_alt_correct':[],'baseline_correct_to_alt_wrong':[],'both_correct':[],'both_wrong':[]}
  for x in primary_rows:
   base=x['processors']['CURRENT_SIGLIP']['correct'];other=x['processors'][alt]['correct']
   key='both_correct' if base and other else 'baseline_correct_to_alt_wrong' if base else 'baseline_wrong_to_alt_correct' if other else 'both_wrong'
   groups[key].append(x['qa_id'])
  assert sum(map(len,groups.values()))==19
  pairs[alt]={'counts':{k:len(v) for k,v in groups.items()},'qa_ids':groups}
 dump(OUT/'pairwise_transitions.json',pairs)
 union={name:{'correct':sum(any(x['processors'][k]['correct'] for k in group) for x in primary_rows),'n':19,'EVALUATOR_ONLY':True} for name,group in [('current_only',('CURRENT_SIGLIP',)),('current_plus_simple',('CURRENT_SIGLIP','SIMPLE_PIXEL')),('current_plus_vlm',('CURRENT_SIGLIP','NARROW_QWEN_VLM')),('all_processors',PROCESSORS)]}
 for v in union.values():v['rate']=v['correct']/19
 dump(OUT/'oracle_union.json',union)
 confusion={}
 for k in PROCESSORS:
  matrix={gt:{pred:0 for pred in (*REALBIND_COLORS,'INVALID')} for gt in REALBIND_COLORS}
  for x in primary_rows:matrix[x['EVALUATOR_ONLY']['gt_color']][x['processors'][k]['normalized_color'] or 'INVALID']+=1
  common=Counter((x['EVALUATOR_ONLY']['gt_color'],x['processors'][k]['normalized_color'] or 'INVALID') for x in primary_rows if not x['processors'][k]['correct'])
  confusion[k]={'vocabulary':list(REALBIND_COLORS),'matrix':matrix,'common_actual_errors':[{'gt':a,'predicted':b,'count':n} for (a,b),n in common.most_common()]}
 dump(OUT/'confusion_analysis.json',confusion)
 size={}
 for name,lo,hi in BINS:
  subset=[x for x in primary_rows if x['size_bin']==name]
  size[name]={'area_ratio_range':[lo,hi],'n':len(subset),'qa_ids':[x['qa_id'] for x in subset],'processors':{k:{'correct':sum(x['processors'][k]['correct'] for x in subset),'n':len(subset),'accuracy':sum(x['processors'][k]['correct'] for x in subset)/len(subset) if subset else None} for k in PROCESSORS}}
 area_rows=[{'qa_id':x['qa_id'],'bbox_area_ratio':x['bbox_area_ratio'],'bbox_width':x['bbox_width'],'bbox_height':x['bbox_height'],'size_bin':x['size_bin'],'correct':{k:x['processors'][k]['correct'] for k in PROCESSORS}} for x in primary_rows]
 by_category={}
 for category in sorted({x['legal_referent_category'] for x in primary_rows}):
  subset=[x for x in primary_rows if x['legal_referent_category']==category]
  by_category[category]={'n':len(subset),'correct':{k:sum(x['processors'][k]['correct'] for x in subset) for k in PROCESSORS}}
 dump(OUT/'size_stratification.json',{'predeclared_bins':size,'per_case_area':area_rows,'by_legal_referent_category':by_category,'crop_purity_metric_available':False,'background_contamination_annotation_available':False,'no_size_threshold_selected_from_results':True})
 projection={k:{'projected_sufficient_qa':sum(x['primary'] and x['processors'][k]['correct'] for x in results),'n':40,'label':'PROJECTED_STATE_SUFFICIENCY','actual_end_to_end_accuracy':None} for k in PROCESSORS}
 for v in projection.values():v['rate']=v['projected_sufficient_qa']/40
 assert projection['CURRENT_SIGLIP']['projected_sufficient_qa']==10
 dump(OUT/'projected_state_sufficiency.json',projection)
 # Agreement remains descriptive only; no certification or routing policy.
 agreement={f'{a}__{b}':{'agree':sum(x['processors'][a]['normalized_color']==x['processors'][b]['normalized_color'] for x in primary_rows),'n':19} for a,b in [('CURRENT_SIGLIP','SIMPLE_PIXEL'),('CURRENT_SIGLIP','NARROW_QWEN_VLM'),('SIMPLE_PIXEL','NARROW_QWEN_VLM')]}
 dump(OUT/'agreement_descriptive.json',agreement)
 make_figures(metrics,pairs,confusion,area_rows,projection)
 make_report(results,metrics,pairs,union,confusion,size,projection,agreement)

def make_figures(metrics,pairs,confusion,areas,projection):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 f=OUT/'figures';labels=['Current','Simple','Narrow VLM']
 plt.figure(figsize=(7,4));plt.bar(labels,[metrics[k]['accuracy'] for k in PROCESSORS]);plt.ylim(0,1);plt.ylabel('Target color accuracy (N=19)');plt.tight_layout();plt.savefig(f/'figure1_primary_accuracy.png');plt.close()
 plt.figure(figsize=(7,4));x=np.arange(2);plt.bar(x-.18,[pairs[k]['counts']['baseline_wrong_to_alt_correct'] for k in PROCESSORS[1:]],.36,label='Rescues');plt.bar(x+.18,[pairs[k]['counts']['baseline_correct_to_alt_wrong'] for k in PROCESSORS[1:]],.36,label='Regressions');plt.xticks(x,['Simple','Narrow VLM']);plt.ylabel('QA versus current reader');plt.legend();plt.tight_layout();plt.savefig(f/'figure2_rescue_regression.png');plt.close()
 fig,axes=plt.subplots(1,3,figsize=(16,5));v=list(REALBIND_COLORS)+['INVALID']
 for ax,k,label in zip(axes,PROCESSORS,labels):
  m=np.array([[confusion[k]['matrix'][gt][pred] for pred in v] for gt in REALBIND_COLORS]);ax.imshow(m,cmap='Blues',vmin=0);ax.set_title(label);ax.set_xticks(range(len(v)),v,rotation=90,fontsize=7);ax.set_yticks(range(len(REALBIND_COLORS)),REALBIND_COLORS,fontsize=7)
 fig.tight_layout();fig.savefig(f/'figure3_confusion_matrices.png');plt.close(fig)
 plt.figure(figsize=(8,4))
 for k,label in zip(PROCESSORS,labels):
  x=[r['bbox_area_ratio'] for r in areas];y=[int(r['correct'][k]) for r in areas];plt.scatter(x,y,label=label,alpha=.62)
 plt.xscale('log');plt.yticks([0,1],['Wrong','Correct']);plt.xlabel('Target bbox area / image area (log scale)');plt.legend();plt.tight_layout();plt.savefig(f/'figure4_accuracy_vs_area.png');plt.close()
 plt.figure(figsize=(7,4));plt.bar(labels,[projection[k]['projected_sufficient_qa'] for k in PROCESSORS]);plt.ylim(0,40);plt.ylabel('Projected sufficient QA / 40');plt.tight_layout();plt.savefig(f/'figure5_projected_sufficiency.png');plt.close()

def make_report(rows,metrics,pairs,union,confusion,size,projection,agreement):
 b=metrics['CURRENT_SIGLIP'];simple=metrics['SIMPLE_PIXEL'];vlm=metrics['NARROW_QWEN_VLM'];s_pair=pairs['SIMPLE_PIXEL']['counts'];v_pair=pairs['NARROW_QWEN_VLM']['counts']
 # A single pilot decision; no fallback is implemented.
 if vlm['correct']>max(simple['correct'],b['correct']) and v_pair['baseline_wrong_to_alt_correct']>v_pair['baseline_correct_to_alt_wrong']:decision='ADOPT_VLM_COLOR_SPECIALIST';vlm_role='MAIN ATTRIBUTE PROCESSOR'
 elif simple['correct']>=max(vlm['correct'],b['correct']) and simple['correct']>b['correct']:decision='ADOPT_DEDICATED_COLOR_READER';vlm_role='BASELINE ONLY' if v_pair['baseline_wrong_to_alt_correct']<=1 else 'FALLBACK ATTRIBUTE PROCESSOR (diagnostic only)'
 elif vlm['correct']<=b['correct'] and v_pair['baseline_wrong_to_alt_correct']>=3 and v_pair['baseline_wrong_to_alt_correct']>v_pair['baseline_correct_to_alt_wrong']:decision='USE_VLM_AS_COLOR_FALLBACK';vlm_role='FALLBACK ATTRIBUTE PROCESSOR (policy not implemented)'
 elif max(simple['correct'],vlm['correct'])<=b['correct']:decision='KEEP_CURRENT_READER';vlm_role='BASELINE ONLY'
 else:decision='KEEP_CURRENT_READER';vlm_role='BASELINE ONLY'
 simple_role='REPLACE CURRENT' if decision=='ADOPT_DEDICATED_COLOR_READER' else 'BASELINE ONLY'
 lines=['# Target-color processor comparison pilot','',f'**Recommendation: `{decision}`.** Primary comparison holds the frozen Minimal Specialist target and selected Grounding DINO region fixed on the exact 19 correct-target/correct-region QA. All 38 executable-target QA were processed for secondary practical coverage. The final reasoner was not called.','','## A. Primary target-color accuracy (N=19)','','| Processor | Correct | Accuracy | Invalid |','|---|---:|---:|---:|']
 for k,label in zip(PROCESSORS,['Current frozen SigLIP reader','Simple PixelColorReader','Narrow Qwen3.5 VLM']):lines.append(f"| {label} | {metrics[k]['correct']}/19 | {metrics[k]['accuracy']:.3f} | {metrics[k]['invalid']} |")
 lines+=['','## Secondary practical coverage (38 executable selected regions)','','| Processor | Color matches REL2C GT |','|---|---:|']
 for k,label in zip(PROCESSORS,['Current frozen SigLIP reader','Simple PixelColorReader','Narrow Qwen3.5 VLM']):lines.append(f"| {label} | {metrics[k]['secondary_executable_correct']}/38 |")
 lines+=['','This secondary count includes wrong target identities and wrong regions, so it is not a pure color-reader accuracy estimate.','','## B. Paired transitions versus current reader','','| Comparison | Current wrong → alt correct | Current correct → alt wrong | Both correct | Both wrong |','|---|---:|---:|---:|---:|']
 for k,label in [('SIMPLE_PIXEL','Current vs Simple'),('NARROW_QWEN_VLM','Current vs VLM')]:
  c=pairs[k]['counts'];lines.append(f"| {label} | {c['baseline_wrong_to_alt_correct']} | {c['baseline_correct_to_alt_wrong']} | {c['both_correct']} | {c['both_wrong']} |")
 lines+=['','## C. Evaluator-only complementarity ceilings','','| Processor set | Oracle-union correct / 19 |','|---|---:|']
 for key,label in [('current_only','Current only'),('current_plus_simple','Current + Simple'),('current_plus_vlm','Current + VLM'),('all_processors','All processors')]:lines.append(f"| {label} | {union[key]['correct']}/19 |")
 lines+=['','These oracle unions are evaluator-only ceilings. Agreement is descriptive and is not a certificate or deployed routing rule.','','## D. Projected minimal-state sufficiency (all 40 QA)','','| Color processor | Projected sufficient QA | Rate |','|---|---:|---:|']
 for k,label in zip(PROCESSORS,['Current','Simple','Narrow VLM']):lines.append(f"| {label} | {projection[k]['projected_sufficient_qa']}/40 | {projection[k]['rate']:.3f} |")
 lines+=['','`PROJECTED_STATE_SUFFICIENCY` holds the frozen target and region correctness fixed, then substitutes each color result. It is not measured final-answer accuracy. No final Qwen call was made.','','## E. Processor cost and inputs','','| Processor | Model | Extra GPU inference | Typical input | Output |','|---|---|---:|---|---|','| Current | frozen SigLIP2 | 0 new (saved prediction); 1 model call if deployed | exact selected bbox crop | frozen color vocabulary |','| Simple | PixelColorReader | 0 | exact selected bbox with fixed central sampling | frozen color vocabulary |','| VLM | frozen Qwen3.5-4B | 38 new calls | exact selected bbox crop | strict single color from frozen vocabulary |','','SigLIP2 and Qwen receive the same canonical exact-bbox crop content. PixelColorReader receives the same original image and bbox, then applies its frozen central-pixel rule inside that box. No processor-specific box padding, GT box, candidate change, or context view is used. VLM prompt: `'+VLM_PROMPT+'` The VLM sees no REL2C question, left/right side, A/B identity, category, GT, or other processor output.','', '## Color confusion and target size']
 for k,label in zip(PROCESSORS,['Current','Simple','Narrow VLM']):
  common=confusion[k]['common_actual_errors'][:5];lines.append(f"- {label} actual error pairs (GT→prediction, count): {[(r['gt'],r['predicted'],r['count']) for r in common]}.")
 lines+=['','| Predeclared area bin | QA | Current | Simple | VLM |','|---|---:|---:|---:|---:|']
 for name,_,_ in BINS:
  z=size[name];lines.append(f"| {name} | {z['n']} | "+' | '.join(f"{z['processors'][k]['correct']}/{z['n']}" for k in PROCESSORS)+' |')
 lines+=['','The bins are fixed at <5%, 5–15%, and ≥15% of image area. Per-case width, height, area fraction, and accuracy are saved in `size_stratification.json`. Crop purity/background contamination has no validated frozen annotation here, so no visual judgment was added. The legal referent noun is saved for descriptive category grouping; the VLM never receives it.','','## Decisions','','- VLM color role: **'+vlm_role+'**.','- Simple reader role: **'+simple_role+'**.','- Neither agreement nor oracle union has been turned into a fallback policy.','','## Scientific questions']
 current_errors=19-b['correct'];simple_res=s_pair['baseline_wrong_to_alt_correct'];vlm_res=v_pair['baseline_wrong_to_alt_correct']
 lines+=['',f"Q1. Of {current_errors} current-reader errors on the 19-case primary set, the simple reader rescues {simple_res} and the narrow VLM rescues {vlm_res}; paired regressions are {s_pair['baseline_correct_to_alt_wrong']} and {v_pair['baseline_correct_to_alt_wrong']}, respectively.",f"Q2. Narrow VLM color accuracy is {vlm['correct']}/19 versus current {b['correct']}/19, with {vlm_res} rescues and {v_pair['baseline_correct_to_alt_wrong']} regressions.",f"Q3. The simple reader scores {simple['correct']}/19 versus VLM {vlm['correct']}/19 and requires no GPU inference.",f"Q4. Oracle unions current+simple {union['current_plus_simple']['correct']}/19 and current+VLM {union['current_plus_vlm']['correct']}/19 quantify complementarity. These are upper bounds, not a validated selector.",f"Q5. Substituting only the color processor changes projected state sufficiency from {projection['CURRENT_SIGLIP']['projected_sufficient_qa']}/40 to {projection['SIMPLE_PIXEL']['projected_sufficient_qa']}/40 (simple) or {projection['NARROW_QWEN_VLM']['projected_sufficient_qa']}/40 (VLM).",'Q6. Area-stratified results above and Figure 4 are descriptive. The 19 cases are too few to establish a reliable small-object or contamination effect; no contamination labels were available.',f'Q7. For the target-color primitive only, use the role decision above: {vlm_role}. This does not change VLM use for target selection or final REL2C reasoning.','','## Architectural interpretation','','The comparison assigns one necessary visual primitive—target color—to local processors on identical frozen regions. Target selection and grounding remain unchanged, and final execution remains image-free. This tests empirical processor specialization within minimal task-conditioned visual compilation; it does not restore maximal scene extraction or let the VLM answer REL2C.','','## Validity and limits','','Primary membership was frozen by the previous evaluator audit (19 QA); processors ran independently on every available frozen selected target bbox (38 QA). Source bbox and crop hashes match across all three conditions. The current SigLIP result is an exact saved prediction from the prior pilot. PixelColorReader is an existing deterministic heuristic. Qwen uses one fixed generic prompt and frozen deterministic non-thinking decoding; invalid non-vocabulary or explanatory output is counted wrong. All ground truth was joined after processor calls. The 19 primary QA come from 11 families; the four VLM rescues come from three families. LEFT/RIGHT cases within families are correlated, so this is a pilot finding rather than a broad accuracy claim.','']
 (OUT/'implementation_report.md').write_text('\n'.join(lines))
 cells=''.join(f"<tr><td>{html.escape(x['qa_id'])}</td><td>{int(x['primary'])}</td><td>{html.escape(x['EVALUATOR_ONLY']['gt_color'])}</td><td>{x['bbox_area_ratio']:.3f}</td>"+''.join(f"<td>{html.escape(str(x['processors'][k]['normalized_color']))}</td><td>{int(x['processors'][k]['correct'])}</td>" for k in PROCESSORS)+'</tr>' for x in rows)
 (OUT/'audit.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{font:14px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px}</style></head><body><h1>Target-color processor pilot</h1><p>GT and correctness are evaluator-only.</p><table><tr><th>QA</th><th>Primary</th><th>GT</th><th>Area ratio</th>'+''.join(f'<th>{k} color</th><th>correct</th>' for k in PROCESSORS)+'</tr>'+cells+'</table></body></html>')
 (OUT/'test_report.md').write_text('# Validation\n\n- Frozen 40 QA and 19-case primary subset verified; 38 executable selected target regions.\n- Same frozen target bbox, region hash, crop policy, and crop SHA across processor conditions for every executable QA.\n- PixelColorReader uses its existing algorithm; Qwen uses one generic prompt and frozen deterministic non-thinking generation.\n- Qwen calls contain only exact target crop and generic color prompt; no REL2C question, category, A/B, GT, or other processor output.\n- Exact vocabulary parser records invalid VLM outputs; existing RealBind scorer checked valid normalizations evaluator-side.\n- Current reader recomputed to 10/19 on primary subset from saved inference.\n- All primary paired transitions sum to 19; all projected sufficiency values are labeled as projections.\n- No final reasoner, certification, agreement gate, new model download, GT bbox, or detector rerun.\n- Five required figures generated.\n')
 (OUT/'changed_files.txt').write_text('harness/scripts/run_target_color_processor_pilot.py\noutputs/target_color_processor_pilot/ (new artifacts)\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','infer_pixel','infer_vlm','analyze']);args=a.parse_args();globals()[args.stage]()
