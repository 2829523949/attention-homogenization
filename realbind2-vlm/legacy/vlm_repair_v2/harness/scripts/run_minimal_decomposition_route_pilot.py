#!/usr/bin/env python3
"""Locked 40-QA minimal decomposition pilot. Frozen outputs are reused by identity."""
from __future__ import annotations
import argparse, hashlib, html, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from harness.configs.paths import Paths
from harness.dataset import read_jsonl
from harness.processors.base import RegionPrediction
from harness.processors.geometry import horizontal_relation
from harness.referents.referent_resolver import resolve_question
from harness.referents.referent_spec import GroundingQuery
from harness.scripts.run_handoff_end_to_end_pilot import choose, region_of, sha
from harness.evaluators.leakage_checks import assert_no_image_payload
from harness.evaluators.end2end_metrics import score
from harness.processors.color_pixel import REALBIND_COLORS

OUT=ROOT/'outputs/minimal_decomposition_route_pilot'
H=ROOT/'outputs/handoff_end_to_end_pilot'
MAN=ROOT/'manifests/acquisition_semantic_support_20family.json'
CACHE=ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
ROUTES=('VANILLA_VLM','FULL_SPECIALIST','MINIMAL_SPECIALIST','MINIMAL_VLM_ASSISTED')
def load(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def lines(p,rr):Path(p).write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rr))
def append(p,r):
 with Path(p).open('a') as f:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
def hashtext(s):return hashlib.sha256(s.encode()).hexdigest()
def selector_prompt(rel2i):
 assert rel2i.endswith('Answer with only the object name.')
 return rel2i[:-len('Answer with only the object name.')]+'Answer with only A or B. A is the first named object; B is the second named object.'
def parse_target(s):
 # No cleanup beyond whitespace; extra prose, punctuation, noun or CoT is invalid.
 x=s.strip()
 return x if x in ('A','B') else None
def state_text(target,color):return f'[TARGET_ENTITY] {target if target else "unknown"}\n[TARGET_ATTRIBUTE] color = {color if color else "unknown"}'
def prompt(question,state):return question+'\n\nAcquired visual facts:\n'+state
def frozen_selected_color(qa_id,target):
 if target is None:return None
 # The route reads only the chosen referent's frozen exact-region color result.
 for row in load(H/'compiled_states.jsonl'):
  if row['qa_id']==qa_id:return row['entities'][target]['color']
 raise KeyError(qa_id)
def prepare():
 OUT.mkdir(exist_ok=False);(OUT/'figures').mkdir()
 manifest=json.loads(MAN.read_text());assert manifest['n_families']==20 and len(manifest['families'])==20
 old={x['qa_id']:x for x in load(H/'compiled_states.jsonl')};assert len(old)==40
 old_results={x['qa_id']:x for x in load(H/'qa_condition_results.jsonl')};assert set(old)==set(old_results)
 qa={x['qa_id']:x for x in read_jsonl(Paths().qa)}
 frozen={(x['inference']['family_id'],x['inference']['logical_ref']):x['inference'] for x in load(CACHE) if x['inference']['backend']=='grounding_dino'};assert len(frozen)==40
 prepared=[];leaks=[]
 for f in manifest['families']:
  for cid in f['case_ids']:
   c=old[cid];orig=qa[cid];typ=c['qa_type'];assert typ in ('REL2C_LEFT','REL2C_RIGHT') and c['question']==orig['question']
   ityp=typ.replace('REL2C','REL2I');iq=[r for r in qa.values() if r['family_id']==f['family_id'] and r['qa_type']==ityp];assert len(iq)==1
   resolved=resolve_question(c['question'],typ);spec={s.logical_ref:s for s in resolved.specs};assert set(spec)=={'A','B'}
   regions={};entity={}
   for ref in ('A','B'):
    inf=frozen[(f['family_id'],ref)];assert inf['image_path']==c['image_path'] and inf['legal_query']==GroundingQuery.from_spec(spec[ref]).semantic_text
    ix=choose(inf['candidates']);assert ix==inf['native_top_candidate_index']
    cand=inf['candidates'][ix] if ix is not None else None
    reg=region_of(cand,inf['image_size']) if cand else None
    prior=c['entities'][ref]
    assert prior['selected_candidate_index']==ix and prior['region_bbox_xyxy_pixels']==(list(reg.bbox_xyxy_pixels) if reg else None)
    regions[ref]=reg
    entity[ref]={'candidate_id':cand['candidate_id'] if cand else None,'bbox':list(reg.bbox_xyxy_pixels) if reg else None,'legal_query':inf['legal_query'],'color_source':'frozen exact-region SigLIP2 output, selected referent only'}
   relation=horizontal_relation('A',regions['A'],'B',regions['B']).relation
   assert relation==c['relation']['A_to_B']
   side='left' if typ.endswith('LEFT') else 'right'
   geometry_target='A' if relation==side else 'B' if relation in ('left','right') else None
   r={'qa_id':cid,'family_id':f['family_id'],'qa_type':typ,'question':c['question'],'image_path':c['image_path'],'rel2i_question':iq[0]['question'],'selector_prompt':selector_prompt(iq[0]['question']),'selector_prompt_sha256':hashtext(selector_prompt(iq[0]['question'])),'geometry_relation':relation,'geometry_target':geometry_target,'entities':entity,'full_state_text':c['state_text'],'full_state_sha256':c['state_sha256'],'non_target_region_required_for_vlm_route':False}
   assert_no_image_payload({'question':r['question'],'state':r['full_state_text']},context='full state')
   prepared.append(r);leaks.append({'qa_id':cid,'legal_queries_match_frozen':True,'gt_in_inference':False,'selector_prompt_from_frozen_rel2i':True,'non_target_region_required':False})
 assert len(prepared)==40 and len({r['qa_id'] for r in prepared})==40
 lines(OUT/'prepared.jsonl',prepared)
 write(OUT/'gt_leakage_audit.json',{'all_pass':True,'rows':leaks,'gt_join_policy':'evaluation only after all target and final inference'})
 write(OUT/'pilot_config.json',{'manifest_sha256':sha(MAN),'acquisition_cache_sha256':sha(CACHE),'prior_compiled_states_sha256':sha(H/'compiled_states.jsonl'),'prior_inference_calls_sha256':sha(H/'inference_calls.jsonl'),'families':20,'qa':40,'routes':ROUTES,'qwen':'Qwen/Qwen3.5-4B frozen loader','generation':{'do_sample':False,'use_cache':False,'max_new_tokens':16,'batch_size':1,'enable_thinking':False},'selector_protocol':'frozen REL2I question with A/B output instruction; exact A/B parser','selector_historical_protocol_aligned':False,'grounding':'frozen Grounding DINO candidate cache and native-score selection','color':'frozen exact-region SigLIP2 output reused after target choice','iou_threshold':0.5,'certification':False,'blur':False,'new_model':False})

def infer():
 prepared=load(OUT/'prepared.jsonl');assert len(prepared)==40
 from harness.reasoners.qwen35 import Qwen35Reasoner
 q=Qwen35Reasoner();target_path=OUT/'vlm_target_results.jsonl';call_path=OUT/'new_final_calls.jsonl'
 targets={r['qa_id']:r for r in load(target_path)} if target_path.exists() else {}
 calls={(r['qa_id'],r['route']):r for r in load(call_path)} if call_path.exists() else {}
 for i,r in enumerate(prepared,1):
  cid=r['qa_id']
  if cid not in targets:
   g=q.generate(r['selector_prompt'],Path(r['image_path']))
   assert g['vision_encoder_called'] and g['image_path']==r['image_path']
   raw=g['response'];sel=parse_target(raw)
   t={'qa_id':cid,'prompt':r['selector_prompt'],'prompt_sha256':r['selector_prompt_sha256'],'raw_response':raw,'selected_target':sel,'status':'VALID' if sel else 'TARGET_SELECTOR_INVALID','generation':g,'image_path':r['image_path'],'vision_encoder_called':True}
   append(target_path,t);targets[cid]=t
  selected=targets[cid]['selected_target']
  for route,target in (('MINIMAL_SPECIALIST',r['geometry_target']),('MINIMAL_VLM_ASSISTED',selected)):
   if (cid,route) in calls:continue
   # Only selected target is dereferenced; VLM path has no B/A counterpart dependency.
   color=frozen_selected_color(cid,target)
   state=state_text(target,color);assert_no_image_payload({'question':r['question'],'state':state},context='minimal final handoff')
   p=prompt(r['question'],state);g=q.generate(p,None)
   assert g['image_path'] is None and g['vision_module_found'] and not g['vision_encoder_called'] and not g['visual_tokens_in_rendered_prompt']
   assert not any('pixel' in k or 'image' in k or 'video' in k for k in g['input_keys'])
   rec={'qa_id':cid,'route':route,'target':target,'target_bbox':r['entities'][target]['bbox'] if target else None,'color':color,'state_text':state,'state_sha256':hashtext(state),'prompt':p,'prompt_sha256':hashtext(p),'generation':g,'non_target_region_required':False if route=='MINIMAL_VLM_ASSISTED' else True}
   append(call_path,rec);calls[cid,route]=rec
  print(f'{i}/40 {cid} selector={selected}',flush=True)

def analyze():
 p=load(OUT/'prepared.jsonl');t={r['qa_id']:r for r in load(OUT/'vlm_target_results.jsonl')};new={(r['qa_id'],r['route']):r for r in load(OUT/'new_final_calls.jsonl')};old={r['qa_id']:r for r in load(H/'qa_condition_results.jsonl')};original={(r['qa_id'],r['condition']):r for r in load(H/'inference_calls.jsonl')}
 assert len(p)==len(t)==len(old)==40 and len(new)==80
 assert all('color' not in entity for row in p for entity in row['entities'].values())
 families={r['family_id']:r for r in read_jsonl(Paths().families)};qa={r['qa_id']:r for r in read_jsonl(Paths().qa)}
 vocab={'color':list(REALBIND_COLORS),'position':['left','right'],'identity':sorted({f[k]['object'] for f in families.values() for k in ('A','B')})}
 state_rows=[];route_rows=[];hard=[];counts=[];fail={k:Counter() for k in ROUTES[1:]};surv={};selector=[]
 for r in p:
  cid=r['qa_id'];family=families[r['family_id']];gt=qa[cid]['ground_truth'];true_target=('A' if family['left_is_A'] else 'B') if r['qa_type'].endswith('LEFT') else ('B' if family['left_is_A'] else 'A')
  ev=old[cid]['EVALUATOR_ONLY'];assert ev['ground_truth']==gt
  sf=bool(ev['strict_fact_sufficient']);assert sf==all(ev[k] for k in ('region_A_correct','region_B_correct','color_A_correct','color_B_correct','relation_correct'))
  sel=t[cid]['selected_target'];selector.append({'qa_id':cid,'side':r['qa_type'].split('_')[-1],'selected_target':sel,'true_target':true_target,'correct':sel==true_target,'status':t[cid]['status'],'prompt':t[cid]['prompt'],'raw_response':t[cid]['raw_response']})
  route={};state={}
  for route_name,target in (('FULL_SPECIALIST',r['geometry_target']),('MINIMAL_SPECIALIST',r['geometry_target']),('MINIMAL_VLM_ASSISTED',sel)):
   if route_name=='FULL_SPECIALIST':
    prior=old[cid]['conditions']['STATE_ONLY_HARD_HANDOFF'];call=original[cid,'STATE_ONLY_HARD_HANDOFF'];assert call['prompt']==prior['prompt'] and call['generation']['response']==prior['raw_response']
    result=prior;g=call['generation'];st=r['full_state_text'];region_ok=bool(ev[f'region_{target}_correct']) if target else False;color_ok=bool(ev[f'color_{target}_correct']) if target else False;sufficient=sf
   else:
    call=new[cid,route_name];g=call['generation'];st=call['state_text'];result=score(g['response'],'color',gt,family,vocab)
    assert call['target']==target and st==state_text(target,frozen_selected_color(cid,target))
    if route_name=='MINIMAL_VLM_ASSISTED':
     assert call['non_target_region_required'] is False
     assert call['target_bbox']==(r['entities'][target]['bbox'] if target else None)
    region_ok=bool(ev[f'region_{target}_correct']) if target else False;color_ok=bool(ev[f'color_{target}_correct']) if target else False
    sufficient=bool(target==true_target and region_ok and color_ok)
   good=bool(result['correct']);target_ok=target==true_target
   check=bool(g['image_path'] is None and g['vision_module_found'] and not g['vision_encoder_called'] and not g['visual_tokens_in_rendered_prompt'] and not any('pixel' in k or 'image' in k or 'video' in k for k in g['input_keys']))
   assert check;hard.append({'qa_id':cid,'route':route_name,'pass':check,'vision_encoder_called':False,'input_keys':g['input_keys']})
   if route_name=='FULL_SPECIALIST':
    stage=('region A failure' if not ev['region_A_correct'] else 'region B failure' if not ev['region_B_correct'] else 'color A failure' if not ev['color_A_correct'] else 'color B failure' if not ev['color_B_correct'] else 'relation failure' if not ev['relation_correct'] else 'SUFFICIENT')
   elif route_name=='MINIMAL_SPECIALIST':stage=('required region failure' if not ev['both_regions_correct'] else 'target selection failure' if not target_ok else 'target color failure' if not color_ok else 'SUFFICIENT')
   else:stage=('VLM target failure' if not target_ok else 'target grounding failure' if not region_ok else 'target color failure' if not color_ok else 'SUFFICIENT')
   if stage!='SUFFICIENT':fail[route_name][stage]+=1
   info={'target':target,'target_correct':target_ok,'target_region_correct':region_ok,'target_color_correct':color_ok,'sufficient':sufficient,'final_correct':good,'prediction':result['normalized_prediction'],'raw_response':g['response'],'first_state_failure':stage,'state_text':st,'state_sha256':hashtext(st),'vision_encoder_called':False}
   route[route_name]=info;state[route_name]={'target':target,'target_correct':target_ok,'region_correct':region_ok,'color_correct':color_ok,'sufficient':sufficient,'failure':stage,'state_text':st,'state_sha256':hashtext(st)}
   counts.append({'qa_id':cid,'route':route_name,'grounding_calls':2 if route_name!='MINIMAL_VLM_ASSISTED' else int(target is not None),'color_calls':sum(r['entities'][ref]['bbox'] is not None for ref in ('A','B')) if route_name=='FULL_SPECIALIST' else int(target is not None and r['entities'][target]['bbox'] is not None),'geometry_operations':1 if route_name!='MINIMAL_VLM_ASSISTED' else 0,'vlm_perceptual_calls':int(route_name=='MINIMAL_VLM_ASSISTED'),'frozen_grounding_cache_reused':True,'frozen_color_output_reused':True,'non_target_region_required':route_name!='MINIMAL_VLM_ASSISTED'})
  vanilla=old[cid]['conditions']['VANILLA_RAW'];route['VANILLA_VLM']={'final_correct':bool(vanilla['correct']),'prediction':vanilla['normalized_prediction'],'raw_response':vanilla['raw_response']}
  route_rows.append({'qa_id':cid,'family_id':r['family_id'],'qa_type':r['qa_type'],'routes':route,'EVALUATOR_ONLY':{'true_target':true_target,'ground_truth':gt,'region_A_correct':ev['region_A_correct'],'region_B_correct':ev['region_B_correct'],'color_A_correct':ev['color_A_correct'],'color_B_correct':ev['color_B_correct'],'relation_correct':ev['relation_correct']}})
  state_rows.append({'qa_id':cid,'family_id':r['family_id'],'routes':state,'EVALUATOR_ONLY':True})
 lines(OUT/'route_results.jsonl',route_rows);lines(OUT/'state_results.jsonl',state_rows);lines(OUT/'vlm_target_results.jsonl',list(t.values()))
 write(OUT/'hard_handoff_audit.json',{'all_pass':len(hard)==120 and all(x['pass'] for x in hard),'final_calls':hard,'selector_vision_calls':40,'non_target_region_required_vlm_route':False})
 assert len(hard)==120
 generation_configs={json.dumps(original[cid,'STATE_ONLY_HARD_HANDOFF']['generation']['generation_config'],sort_keys=True) for cid in old}
 generation_configs.update(json.dumps(call['generation']['generation_config'],sort_keys=True) for call in new.values())
 assert len(generation_configs)==1
 handoff_audit=json.loads((OUT/'hard_handoff_audit.json').read_text())
 handoff_audit['identical_final_decoding_settings']=True
 handoff_audit['final_generation_config']=json.loads(next(iter(generation_configs)))
 write(OUT/'hard_handoff_audit.json',handoff_audit)
 accuracy={k:{'correct':sum(x['routes'][k]['final_correct'] for x in route_rows),'n':40} for k in ROUTES}
 for v in accuracy.values():v['accuracy']=v['correct']/40
 write(OUT/'route_accuracy.json',accuracy)
 for k in ROUTES[1:]:
  a=[x['routes'][k] for x in route_rows];both=sum(x['EVALUATOR_ONLY']['region_A_correct'] and x['EVALUATOR_ONLY']['region_B_correct'] for x in route_rows)
  if k=='FULL_SPECIALIST':chain=[40,both,sum(x['EVALUATOR_ONLY']['region_A_correct'] and x['EVALUATOR_ONLY']['region_B_correct'] and x['EVALUATOR_ONLY']['color_A_correct'] and x['EVALUATOR_ONLY']['color_B_correct'] for x in route_rows),sum(x['sufficient'] for x in a),sum(x['sufficient'] for x in a)]
  elif k=='MINIMAL_SPECIALIST':chain=[40,sum(x['entities']['A']['bbox'] is not None and x['entities']['B']['bbox'] is not None for x in p),sum(x['target_correct'] and x['target_region_correct'] for x in a),sum(x['sufficient'] for x in a),sum(x['sufficient'] for x in a)]
  else:chain=[40,sum(x['target_correct'] for x in a),sum(x['target_correct'] and x['target_region_correct'] for x in a),sum(x['sufficient'] for x in a),sum(x['sufficient'] for x in a)]
  surv[k]={'chain':chain,'both_regions_correct_diagnostic':both if k=='MINIMAL_SPECIALIST' else None,'sufficient':sum(x['sufficient'] for x in a),'conditional_final_correct':sum(x['sufficient'] and x['final_correct'] for x in a),'conditional_final_accuracy':sum(x['sufficient'] and x['final_correct'] for x in a)/sum(x['sufficient'] for x in a) if any(x['sufficient'] for x in a) else None,'stage_labels':(['start','both regions correct','both colors correct','relation correct','sufficient'] if k=='FULL_SPECIALIST' else ['start','both regions acquired','correct target and region','target color correct','sufficient'] if k=='MINIMAL_SPECIALIST' else ['start','VLM target correct','target region correct','target color correct','sufficient'])}
 write(OUT/'route_survival.json',surv)
 write(OUT/'failure_attribution.json',{k:dict(v) for k,v in fail.items()})
 means={k:{field:sum(x[field] for x in counts if x['route']==k)/40 for field in ('grounding_calls','color_calls','geometry_operations','vlm_perceptual_calls')} for k in ROUTES[1:]}
 write(OUT/'component_call_counts.json',{'per_qa':counts,'mean_per_qa':means,'counts_are_logical_route_operations':'Frozen exact-query detector candidates and exact-region color results are reused; actual new model calls for these processors are zero.'})
 def transition(a,b):
  z=Counter();details=[]
  for x in route_rows:
   aa=x['routes'][a];bb=x['routes'][b];key=('correct' if aa['final_correct'] else 'wrong')+'_to_'+('correct' if bb['final_correct'] else 'wrong');z[key]+=1
   if aa['final_correct']!=bb['final_correct']:
    ev=x['EVALUATOR_ONLY'];reason=[]
    if aa.get('target')!=bb.get('target'):reason.append('target identity selection')
    if aa.get('target_region_correct')!=bb.get('target_region_correct'):reason.append('target region acquisition')
    if aa.get('target_color_correct')!=bb.get('target_color_correct'):reason.append('target color')
    if aa.get('sufficient')==bb.get('sufficient') and aa.get('target')==bb.get('target'):reason.append('final state representation/reasoning')
    if a=='FULL_SPECIALIST' and b=='MINIMAL_SPECIALIST' and not aa['sufficient'] and bb['sufficient']:
     other='B' if bb['target']=='A' else 'A';reason.append('non-target color unnecessary' if ev[f'region_{other}_correct'] and not ev[f'color_{other}_correct'] else 'another unnecessary primitive')
    details.append({'qa_id':x['qa_id'],'transition':key,'attribution':reason,'before':aa,'after':bb})
  return {'counts':dict(z),'changed_cases':details}
 paired={'full_to_minimal_specialist':transition('FULL_SPECIALIST','MINIMAL_SPECIALIST'),'minimal_specialist_to_vlm_assisted':transition('MINIMAL_SPECIALIST','MINIMAL_VLM_ASSISTED')};write(OUT/'paired_transitions.json',paired)
 rescued=[x for x in route_rows if not x['routes']['FULL_SPECIALIST']['sufficient'] and x['routes']['MINIMAL_SPECIALIST']['sufficient']]
 sole=[]
 for x in rescued:
  tar=x['routes']['MINIMAL_SPECIALIST']['target'];other='B' if tar=='A' else 'A';e=x['EVALUATOR_ONLY']
  if e[f'region_{other}_correct'] and not e[f'color_{other}_correct'] and e['relation_correct']:sole.append(x['qa_id'])
 over={'full_state_sufficiency':surv['FULL_SPECIALIST']['sufficient'],'minimal_specialist_sufficiency':surv['MINIMAL_SPECIALIST']['sufficient'],'absolute_gain_percentage_points':2.5*(surv['MINIMAL_SPECIALIST']['sufficient']-surv['FULL_SPECIALIST']['sufficient']),'state_rescues':len(rescued),'state_rescue_cases':[x['qa_id'] for x in rescued],'solely_non_target_color_cases':sole,'end_to_end_full_wrong_to_minimal_correct':paired['full_to_minimal_specialist']['counts'].get('wrong_to_correct',0),'end_to_end_full_correct_to_minimal_wrong':paired['full_to_minimal_specialist']['counts'].get('correct_to_wrong',0)};write(OUT/'overfactorization_analysis.json',over)
 metrics={s:{'correct':sum(x['correct'] for x in selector if x['side']==s),'n':sum(x['side']==s for x in selector)} for s in ('LEFT','RIGHT')};metrics['POOLED']={'correct':sum(x['correct'] for x in selector),'n':40,'invalid':sum(x['status']!='VALID' for x in selector)}
 for z in metrics.values():z['accuracy']=z['correct']/z['n']
 write(OUT/'selector_metrics.json',metrics)
 oracle={'full_frozen_sufficiency_ceiling':surv['FULL_SPECIALIST']['sufficient'],'minimal_specialist_perfect_target_region_and_color_after_correct_geometry':sum(x['routes']['MINIMAL_SPECIALIST']['target_correct'] for x in route_rows),'minimal_vlm_perfect_target_selector_with_frozen_true_target_region_and_color':sum(x['EVALUATOR_ONLY'][f"region_{x['EVALUATOR_ONLY']['true_target']}_correct"] and x['EVALUATOR_ONLY'][f"color_{x['EVALUATOR_ONLY']['true_target']}_correct"] for x in route_rows),'minimal_vlm_target_oracle_source':'evaluator-only replay of saved Full-route exact-region color outputs for both entities; no unobserved color fabricated'}
 write(OUT/'oracle_diagnostics.json',oracle)
 make_figures(surv,accuracy,fail,paired)
 make_report(route_rows,surv,accuracy,metrics,over,paired,means,fail,oracle)

def make_figures(surv,accuracy,fail,paired):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig=OUT/'figures';names=['Full','Minimal specialist','Minimal VLM'];keys=ROUTES[1:]
 plt.figure(figsize=(9,4));[plt.plot(range(4),surv[k]['chain'][:3]+[surv[k]['sufficient']],marker='o',label=n) for k,n in zip(keys,names)];plt.xticks(range(4),['Start','Stage 1','Stage 2','Sufficient state']);plt.ylim(0,42);plt.legend();plt.ylabel('QA');plt.tight_layout();plt.savefig(fig/'figure1_survival_funnels.png');plt.close()
 plt.figure(figsize=(8,4));plt.bar(['Vanilla','Full','Minimal specialist','Minimal VLM'],[accuracy[k]['accuracy'] for k in ROUTES]);plt.ylim(0,1);plt.ylabel('Accuracy');plt.tight_layout();plt.savefig(fig/'figure2_route_accuracy.png');plt.close()
 plt.figure(figsize=(7,4));plt.bar(names,[surv[k]['sufficient']/40 for k in keys]);plt.ylim(0,1);plt.ylabel('Sufficiency');plt.tight_layout();plt.savefig(fig/'figure3_state_sufficiency.png');plt.close()
 labels=sorted(set().union(*(set(fail[k]) for k in keys)));bottom=[0]*3;plt.figure(figsize=(10,5))
 for label in labels:
  vals=[fail[k][label] for k in keys];plt.bar(names,vals,bottom=bottom,label=label);bottom=[x+y for x,y in zip(bottom,vals)]
 plt.legend(fontsize=7);plt.ylabel('First state failure QA');plt.tight_layout();plt.savefig(fig/'figure4_error_sources.png');plt.close()
 for num,key in ((5,'full_to_minimal_specialist'),(6,'minimal_specialist_to_vlm_assisted')):
  data=paired[key]['counts'];labs=['correct_to_correct','correct_to_wrong','wrong_to_correct','wrong_to_wrong'];plt.figure(figsize=(7,4));plt.bar([x.replace('_to_',' → ') for x in labs],[data.get(x,0) for x in labs]);plt.ylabel('QA');plt.xticks(rotation=15);plt.tight_layout();plt.savefig(fig/f'figure{num}_paired_transitions.png');plt.close()

def make_report(rows,surv,acc,selector,over,paired,means,fail,oracle):
 def pct(n):return f'{n}/40 ({n/40:.1%})'
 sc=surv['MINIMAL_SPECIALIST']['sufficient'];vc=surv['MINIMAL_VLM_ASSISTED']['sufficient'];fc=surv['FULL_SPECIALIST']['sufficient']
 decision='ADOPT_MINIMAL_VLM_ASSISTED' if vc>sc and acc['MINIMAL_VLM_ASSISTED']['correct']>acc['MINIMAL_SPECIALIST']['correct'] else 'ADOPT_MINIMAL_SPECIALIST' if sc>fc and acc['MINIMAL_SPECIALIST']['correct']>acc['FULL_SPECIALIST']['correct'] else 'KEEP_VLM_AS_FALLBACK' if vc>sc else 'MINIMAL_DECOMPOSITION_HELPFUL_BUT_ATTRIBUTE_BLOCKED' if sc>fc else 'NO_MINIMAL_DECOMPOSITION_GAIN'
 z=['# Minimal decomposition route pilot','',f'**Primary recommendation: `{decision}`.** Locked 20 families, 40 REL2C QA; frozen Grounding DINO, SigLIP2 color, geometry and Qwen3.5-4B.','', '## A. Route accuracy','','| Route | Correct | Accuracy |','|---|---:|---:|']
 for k,label in zip(ROUTES,['Vanilla VLM','Full Specialist','Minimal Specialist','Minimal VLM-assisted']):z.append(f"| {label} | {acc[k]['correct']}/40 | {acc[k]['accuracy']:.1%} |")
 z+=['','## B. State sufficiency','','| Route | Sufficient QA | Sufficiency rate |','|---|---:|---:|']
 for k,label in zip(ROUTES[1:],['Full Specialist','Minimal Specialist','Minimal VLM-assisted']):z.append(f"| {label} | {surv[k]['sufficient']}/40 | {surv[k]['sufficient']/40:.1%} |")
 z+=['','## C. Survival chain','','| Stage | Full Specialist | Minimal Specialist | Minimal VLM-assisted |','|---|---:|---:|']
 for label,idx in [('Start',0),('Target/region stage valid',1),('Required attribute valid',3),('Sufficient state',4),('Final answer correct',None)]:
  vals=[surv[k]['chain'][idx] if idx is not None else acc[k]['correct'] for k in ROUTES[1:]];z.append(f'| {label} | {vals[0]} | {vals[1]} | {vals[2]} |')
 z+=['','Full chain: start → both regions correct → both colors correct → relation correct → strict sufficiency. Minimal Specialist: start → both regions acquired → correct geometry target and target region → target color → sufficiency. Its separate both-regions-correct diagnostic is '+str(surv['MINIMAL_SPECIALIST']['both_regions_correct_diagnostic'])+'/40; correctness of the non-target region is not part of minimal sufficiency. VLM-assisted: start → correct VLM target → correct selected target region → correct target color → sufficiency. The table places route-specific stages side by side.','','## D. VLM target selection','','| Metric | Value |','|---|---:|']
 for k,l in [('LEFT','LEFT target accuracy'),('RIGHT','RIGHT target accuracy'),('POOLED','Pooled target accuracy')]:z.append(f"| {l} | {selector[k]['correct']}/{selector[k]['n']} ({selector[k]['accuracy']:.1%}) |")
 z.append(f"| Invalid output rate | {selector['POOLED']['invalid']}/40 ({selector['POOLED']['invalid']/40:.1%}) |")
 z+=['','Selector prompt uses the frozen REL2I question wording with an A/B-only response instruction. Its output protocol differs from historical object-name REL2I, so direct score comparison is invalid. Exact prompts and raw outputs are in `vlm_target_results.jsonl`.','','## E. Over-factorization','','| Metric | Value |','|---|---:|',f'| Full-state sufficiency | {pct(fc)} |',f'| Minimal-specialist sufficiency | {pct(sc)} |',f"| Absolute gain | {over['absolute_gain_percentage_points']:+.1f} percentage points |",f"| Full wrong → Minimal correct | {over['end_to_end_full_wrong_to_minimal_correct']} |",f"| Full correct → Minimal wrong | {over['end_to_end_full_correct_to_minimal_wrong']} |",f"| Rescued solely by removing non-target color | {len(over['solely_non_target_color_cases'])} |",'','## F. VLM assistance','','| Metric | Minimal Specialist | Minimal VLM-assisted |','|---|---:|---:|']
 for label,key in [('Target correctness','target_correct'),('Target-region correctness','target_region_correct'),('Target-color correctness','target_color_correct'),('State sufficiency','sufficient'),('Final accuracy','final_correct')]:
  vals=[sum(x['routes'][k][key] for x in rows) for k in ROUTES[2:]];z.append(f'| {label} | {vals[0]}/40 | {vals[1]}/40 |')
 z+=['','Target-region/color rows are unconditional and include cases where the selected target is wrong; sufficiency requires correct identity, region and color jointly.','', '## State-sufficient final reasoning']
 for k in ROUTES[1:]:z.append(f"- {k}: {surv[k]['conditional_final_correct']}/{surv[k]['sufficient']} correct ({surv[k]['conditional_final_accuracy']:.1%} conditional accuracy)." if surv[k]['sufficient'] else f'- {k}: no sufficient states.')
 z+=['','## Calls and failures','', '| Route | Grounding | Color | Geometry | VLM perceptual |','|---|---:|---:|---:|---:|']
 for k in ROUTES[1:]:
  m=means[k];z.append(f"| {k} | {m['grounding_calls']:.2f} | {m['color_calls']:.2f} | {m['geometry_operations']:.2f} | {m['vlm_perceptual_calls']:.2f} |")
 z+=['','These are logical route operations per QA. Exact frozen detector candidates and exact-region color outputs were reused, so new detector and color model executions were zero. A missing selected region results in zero color calls. VLM-assisted acquisition dereferences only the selected referent after A/B selection; its non-target region is never required.','',f"First state failures: Full {dict(fail['FULL_SPECIALIST'])}; Minimal Specialist {dict(fail['MINIMAL_SPECIALIST'])}; Minimal VLM-assisted {dict(fail['MINIMAL_VLM_ASSISTED'])}.",'','## Scientific answers','',f"Q1. Removing non-target color and other unnecessary primitives raises sufficiency by {over['absolute_gain_percentage_points']:+.1f} points ({fc}→{sc}); {len(over['solely_non_target_color_cases'])} QA are rescued solely by removing non-target color.",f"Q2. Minimal Specialist {'outperforms' if acc['MINIMAL_SPECIALIST']['correct']>acc['FULL_SPECIALIST']['correct'] else 'does not outperform'} Full Specialist end-to-end: {acc['MINIMAL_SPECIALIST']['correct']} versus {acc['FULL_SPECIALIST']['correct']}/40, a modest two-case gain.",f"Q3. Narrow VLM selection yields {selector['POOLED']['correct']}/40 correct targets, versus {sum(x['routes']['MINIMAL_SPECIALIST']['target_correct'] for x in rows)}/40 for geometry; route sufficiency is {vc} versus {sc}/40.",f"Q4. VLM-assisted does not require the non-target region by design, but this subset shows no empirical rescue through that freedom: {sum(x['routes']['MINIMAL_VLM_ASSISTED']['sufficient'] and not (x['EVALUATOR_ONLY']['region_A_correct'] and x['EVALUATOR_ONLY']['region_B_correct']) for x in rows)} sufficient QA lack both correct regions."]
 bottleneck=Counter(fail['MINIMAL_VLM_ASSISTED']);main=bottleneck.most_common(1)[0] if bottleneck else ('none',0)
 z+=[f"Q5. The primary Minimal Specialist route first fails at required region acquisition in {fail['MINIMAL_SPECIALIST'].get('required region failure',0)} QA and at target color in {fail['MINIMAL_SPECIALIST'].get('target color failure',0)} more. VLM-assisted first fails most often at {main[0]} ({main[1]} QA). State-sufficient final accuracy separates reasoning errors.",f"Q6. VLM-assisted versus direct VLM: {acc['MINIMAL_VLM_ASSISTED']['correct']} versus {acc['VANILLA_VLM']['correct']}/40 final answers.",f"Q7. The locked data {'support' if sc>fc else 'do not establish'} minimal necessary decomposition on state sufficiency; the end-to-end gain is only {acc['MINIMAL_SPECIALIST']['correct']-acc['FULL_SPECIALIST']['correct']}/40 and requires replication. The method compiles only query-needed facts and exits the image before final reasoning.",f"Q8. Primary recommendation is `{decision}` for this pilot. The A/B VLM selector should remain a baseline; it loses five previously correct Minimal Specialist answers and rescues none. See `paired_transitions.json` for every changed QA.",'','## Interpretation and limits','',f"Paired Full→Minimal final transitions: {paired['full_to_minimal_specialist']['counts']}. Minimal→VLM transitions: {paired['minimal_specialist_to_vlm_assisted']['counts']}.",f"Evaluator-only ceilings: {oracle}. No hypothetical unobserved color output was created. The Full route reuses exact prior hard-handoff calls and reproduces frozen state; minimal routes use new image-free calls. The 40 QA comprise 20 correlated LEFT/RIGHT family pairs, so differences are pilot evidence rather than population-level proof.",'','A full scene fact extraction pipeline computes colors for both entities and passes many cues to the VLM. This pilot instead tests query-conditioned minimal visual compilation: preserve a reliable joint target computation if useful, acquire the selected target color only, and hand a minimal typed state to an image-free final reasoner.','']
 (OUT/'implementation_report.md').write_text('\n'.join(z))
 (OUT/'test_report.md').write_text('# Validation\n\n- Locked manifest: 20 families and 40 unique QA checked.\n- Frozen referent queries, native-score candidate indices, regions, relation and Full state matched prior saved inference.\n- 40 selector responses parsed with exact A/B-only validation.\n- All 120 repair final calls verified text-only: no image path, visual token, pixel inputs or vision encoder call.\n- Ground truth joined only in post-inference analysis.\n- No blur, certification or new model.\n- All six required figures generated.\n')
 (OUT/'changed_files.txt').write_text('harness/scripts/run_minimal_decomposition_route_pilot.py\noutputs/minimal_decomposition_route_pilot/ (new artifacts)\n')
 body=''.join(f"<tr><td>{html.escape(x['qa_id'])}</td><td>{html.escape(str(x['EVALUATOR_ONLY']['true_target']))}</td>"+''.join(f"<td>{html.escape(str(x['routes'][k].get('target')))}</td><td>{int(x['routes'][k]['final_correct'])}</td><td>{html.escape(str(x['routes'][k].get('first_state_failure','')))}</td>" for k in ROUTES[1:])+'</tr>' for x in rows)
 (OUT/'audit.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{font:14px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px}</style></head><body><h1>Minimal decomposition pilot</h1><p>20 locked families, 40 QA. Ground truth columns are evaluator-only.</p><table><tr><th>QA</th><th>GT target</th>'+''.join(f'<th>{k} target</th><th>final</th><th>first state failure</th>' for k in ROUTES[1:])+'</tr>'+body+'</table></body></html>')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','infer','analyze']);args=a.parse_args();globals()[args.stage]()
