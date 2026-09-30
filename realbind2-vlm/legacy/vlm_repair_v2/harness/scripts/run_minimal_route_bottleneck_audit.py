#!/usr/bin/env python3
"""Evaluator-only audit of frozen Minimal Specialist predictions; never calls inference."""
from __future__ import annotations
import hashlib, html, json, re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'outputs/minimal_decomposition_route_pilot'
STATE=ROOT/'outputs/state_bottleneck_audit'
GROUND=ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
HANDOFF=ROOT/'outputs/handoff_end_to_end_pilot/compiled_states.jsonl'
MANIFEST=ROOT/'manifests/acquisition_semantic_support_20family.json'
OUT=ROOT/'outputs/minimal_route_bottleneck_audit'
THRESHOLD=.5
PATTERN=re.compile(r'\A\[TARGET_ENTITY\] (A|B|unknown)\n\[TARGET_ATTRIBUTE\] color = ([a-z]+|unknown)\Z')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_rows(path,rows):Path(path).write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in rows))
def metric(n,d):return {'count':n,'denominator':d,'rate':n/d if d else None}
def count(rows,fn):return sum(bool(fn(x)) for x in rows)
def implied(state):
 match=PATTERN.fullmatch(state)
 if match is None:raise AssertionError('unrecognized frozen minimal state')
 ref,color=match.groups()
 return {'target':None if ref=='unknown' else ref,'color':None if color=='unknown' else color}

def main():
 if OUT.exists():raise FileExistsError(f'refusing to overwrite {OUT}')
 config=json.loads((PILOT/'pilot_config.json').read_text())
 manifest=json.loads(MANIFEST.read_text())
 assert config['families']==manifest['n_families']==20 and config['qa']==40 and sha(MANIFEST)==config['manifest_sha256']
 assert sha(GROUND)==config['acquisition_cache_sha256'] and sha(HANDOFF)==config['prior_compiled_states_sha256']
 pilot=read_rows(PILOT/'route_results.jsonl');assert len(pilot)==40
 expected={cid for f in manifest['families'] for cid in f['case_ids']}
 assert len(expected)==40 and {r['qa_id'] for r in pilot}==expected
 states={r['qa_id']:r for r in read_rows(STATE/'state_component_results.jsonl')};assert set(states)==expected
 prepared={r['qa_id']:r for r in read_rows(PILOT/'prepared.jsonl')};assert set(prepared)==expected
 final={r['qa_id']:r for r in read_rows(PILOT/'new_final_calls.jsonl') if r['route']=='MINIMAL_SPECIALIST'};assert set(final)==expected
 full={r['qa_id']:r for r in read_rows(HANDOFF)};assert set(full)==expected
 ground={(r['inference']['family_id'],r['inference']['logical_ref']):r for r in read_rows(GROUND) if r['inference']['backend']=='grounding_dino'};assert len(ground)==40
 oldjoint=json.loads((ROOT/'outputs/grounding_dino_acquisition_pilot/joint_candidate_metrics.json').read_text())['grounding_dino'];assert oldjoint['families_with_joint_candidate_at_0_5']==12
 oldpaired=json.loads((PILOT/'paired_transitions.json').read_text())['full_to_minimal_specialist']
 OUT.mkdir();(OUT/'figures').mkdir()
 results=[];sel_errors=[];region_errors=[];color_errors=[];accidents=[];reasoner_cases=[];candidate_cases=[]
 for row in pilot:
  cid=row['qa_id'];r=row['routes']['MINIMAL_SPECIALIST'];e=row['EVALUATOR_ONLY'];target=r['target'];truth=e['true_target'];gt=e['ground_truth'];s=implied(r['state_text']);call=final[cid];pr=prepared[cid]
  assert s['target']==target and r['state_text']==call['state_text'] and r['prediction'] is not None
  assert r['target_correct']==(target==truth)
  assert r['target_region_correct']==(bool(e[f'region_{target}_correct']) if target else False)
  assert r['target_color_correct']==(bool(e[f'color_{target}_correct']) if target else False)
  assert r['sufficient']==(r['target_correct'] and r['target_region_correct'] and r['target_color_correct'])
  assert r['final_correct']==(r['prediction']==gt)
  assert s['color']==call['color']
  assert call['generation']['image_path'] is None and not call['generation']['vision_encoder_called']
  assert all('color' not in ent for ent in pr['entities'].values())
  selected_ground=ground.get((row['family_id'],target)) if target else None
  true_ground=ground[row['family_id'],truth]
  selected_prior=states[cid]['entities'][target] if target else None
  true_prior=states[cid]['entities'][truth]
  true_best=float(true_ground['EVALUATOR_ONLY']['best_candidate_iou']);true_available=true_best>=THRESHOLD
  candidate_cases.append({'qa_id':cid,'family_id':row['family_id'],'true_target':truth,'target_candidate_available':true_available,'target_best_candidate_iou':true_best,'target_correct':r['target_correct'],'selected_true_target_region_correct':r['target_correct'] and r['target_region_correct'],'selected_region_correct_for_chosen_entity':r['target_region_correct']})
  if not r['target_correct']:
   both=e['region_A_correct'] and e['region_B_correct'];label='GEOMETRY_TARGET_ERROR_DESPITE_CORRECT_REGIONS' if both else 'GEOMETRY_TARGET_ERROR_FROM_WRONG_REGION'
   sel_errors.append({'qa_id':cid,'family_id':row['family_id'],'selected_target':target,'true_target':truth,'classification':label,'region_A_correct':e['region_A_correct'],'region_B_correct':e['region_B_correct'],'geometry_relation':pr['geometry_relation'],'qa_type':row['qa_type'],'bbox_A':pr['entities']['A']['bbox'],'bbox_B':pr['entities']['B']['bbox']})
  if r['target_correct'] and not r['target_region_correct']:
   raw=selected_prior['region_failure_class'];available=selected_prior['candidate_available']
   label='USABLE_CANDIDATE_EXISTS_BUT_NOT_SELECTED' if available else raw if raw in ('NO_USABLE_CANDIDATE','PARTIAL_OVERLAP','OVERLARGE_REGION','SEMANTIC_DISTRACTOR','WRONG_SAME_CATEGORY_INSTANCE') else 'UNKNOWN'
   assert label in ('NO_USABLE_CANDIDATE','USABLE_CANDIDATE_EXISTS_BUT_NOT_SELECTED','WRONG_SAME_CATEGORY_INSTANCE','PARTIAL_OVERLAP','OVERLARGE_REGION','SEMANTIC_DISTRACTOR','UNKNOWN')
   region_errors.append({'qa_id':cid,'family_id':row['family_id'],'target':target,'classification':label,'prior_region_failure_class':raw,'candidate_available':available,'best_candidate_iou':selected_prior['best_candidate_iou'],'selected_iou':selected_prior['selected_iou'],'selected_candidate_index':selected_prior['selected_candidate_index'],'same_query_zero_overlap_distractor_proxy':selected_prior['same_query_zero_overlap_distractor_proxy'],'note':'proxy is not a verified same-category-instance or semantic-distractor label'})
  if r['target_correct'] and r['target_region_correct'] and not r['target_color_correct']:
   bbox=pr['entities'][target]['bbox'];size=selected_ground['inference']['image_size'];area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(size[0]*size[1]) if bbox else None
   prior=full[cid]['entities'][target]
   color_errors.append({'qa_id':cid,'family_id':row['family_id'],'target':target,'classification':'COLOR_FAILURE_AFTER_CORRECT_TARGET_REGION','gt_target_color':gt,'predicted_color':s['color'],'target_bbox_xyxy_pixels':bbox,'target_bbox_area_pixels':(bbox[2]-bbox[0])*(bbox[3]-bbox[1]) if bbox else None,'target_area_ratio':area,'candidate_source':selected_ground['inference']['candidates'][selected_ground['inference']['native_top_candidate_index']]['source_backend'] if selected_ground['inference']['native_top_candidate_index'] is not None else None,'existing_color_reader_score':prior['color_score'],'color_reader_score_semantics':prior['color_score_semantics'],'background_contamination_annotation':None,'background_contamination_status':'not annotated in frozen outputs'})
  sufficient=r['sufficient'];final_ok=r['final_correct'];implied_ok=s['color']==gt
  if final_ok and not sufficient:
   if not r['target_correct'] and s['color']==gt:kind='WRONG_TARGET_BUT_SAME_FINAL_COLOR'
   elif r['target_correct'] and not r['target_region_correct'] and s['color']==gt:kind='WRONG_REGION_BUT_SAME_COLOR'
   elif not r['target_color_correct'] and s['color']!=gt and r['prediction']==gt:kind='TARGET_COLOR_WRONG_FINAL_REASONER_ACCIDENTALLY_CORRECT'
   elif s['color']!=gt and r['prediction']==gt:kind='COMPENSATING_ERRORS'
   else:kind='UNKNOWN'
   accidents.append({'qa_id':cid,'classification':kind,'selected_target':target,'true_target':truth,'target_region_correct':r['target_region_correct'],'target_color_correct':r['target_color_correct'],'state_implied_answer':s['color'],'final_prediction':r['prediction'],'ground_truth':gt})
  if not r['target_correct']:first='TARGET_SELECTION_FAILURE'
  elif not r['target_region_correct']:first='TARGET_REGION_FAILURE'
  elif not r['target_color_correct']:first='TARGET_COLOR_FAILURE'
  elif not final_ok:first='FINAL_REASONING_FAILURE'
  else:first='SUCCESS'
  reasoner_cases.append({'qa_id':cid,'state_implied_answer':s['color'],'state_implied_correct':implied_ok,'final_prediction':r['prediction'],'final_correct':final_ok,'faithful_to_state':s['color'] is not None and s['color']==r['prediction'],'sufficient':sufficient,'correct_implied_changed_to_wrong':implied_ok and not final_ok,'wrong_implied_changed_to_correct':not implied_ok and final_ok,'has_executable_implied_answer':s['color'] is not None})
  results.append({'qa_id':cid,'family_id':row['family_id'],'qa_type':row['qa_type'],'selected_target':target,'true_target':truth,'target_correct':r['target_correct'],'target_region_correct':r['target_region_correct'],'target_color_correct':r['target_color_correct'],'minimal_state_sufficient':sufficient,'final_answer_correct':final_ok,'state_implied_answer':s['color'],'final_prediction':r['prediction'],'ground_truth':gt,'first_failure_stage':first,'true_target_candidate_available_at_0_5':true_available,'selected_target_region_iou':selected_prior['selected_iou'] if selected_prior else 0.,'selected_target_gt_color':selected_prior['evaluator_color'] if selected_prior else None,'selected_target_predicted_color':s['color'],'EVALUATOR_ONLY':True})
 assert len(results)==40
 assert count(results,lambda x:x['target_correct'])==30 and count(results,lambda x:x['minimal_state_sufficient'])==10 and count(results,lambda x:x['final_answer_correct'])==15
 write_rows(OUT/'qa_bottleneck_results.jsonl',results)
 survival=[]
 stages=[('Start',lambda x:True),('Target correct',lambda x:x['target_correct']),('Target + target region correct',lambda x:x['target_correct'] and x['target_region_correct']),('Target + region + color correct',lambda x:x['minimal_state_sufficient']),('Sufficient state + final correct',lambda x:x['minimal_state_sufficient'] and x['final_answer_correct'])]
 previous=40
 for name,fn in stages:
  n=count(results,fn);survival.append({'stage':name,'count':n,'survival_from_previous':n/previous if previous else None,'survival_from_all':n/40});previous=n
 write(OUT/'survival_chain.json',{'stages':survival,'frozen_qa':40})
 t=count(results,lambda x:x['target_correct']);tr=count(results,lambda x:x['target_region_correct']);ttr=count(results,lambda x:x['target_correct'] and x['target_region_correct']);c=count(results,lambda x:x['target_color_correct']);tc=count(results,lambda x:x['target_correct'] and x['target_color_correct']);suff=count(results,lambda x:x['minimal_state_sufficient']);finals=count(results,lambda x:x['final_answer_correct']);final_suff=count(results,lambda x:x['minimal_state_sufficient'] and x['final_answer_correct'])
 metrics={'P_target_correct':metric(t,40),'P_target_region_correct':metric(tr,40),'P_target_region_correct_given_target_correct':metric(ttr,t),'P_target_color_correct':metric(c,40),'P_target_color_correct_given_target_correct':metric(tc,t),'P_target_color_correct_given_target_and_region_correct':metric(suff,ttr),'P_minimal_state_sufficient':metric(suff,40),'P_final_correct_given_sufficient':metric(final_suff,suff),'P_final_correct_given_insufficient':metric(finals-final_suff,40-suff)}
 write(OUT/'conditional_metrics.json',metrics)
 write(OUT/'target_selection_analysis.json',{'total_errors':len(sel_errors),'classification_counts':dict(Counter(x['classification'] for x in sel_errors)),'errors':sel_errors,'correct_target_despite_wrong_selected_region':count(results,lambda x:x['target_correct'] and not x['target_region_correct']),'protocol_note':'geometry target versus evaluator true A/B; no inference changed'})
 write(OUT/'target_region_analysis.json',{'target_correct_region_wrong_qa':len(region_errors),'classification_counts':dict(Counter(x['classification'] for x in region_errors)),'candidate_unavailable_cases':sum(not x['candidate_available'] for x in region_errors),'candidate_available_but_not_selected_cases':sum(x['candidate_available'] for x in region_errors),'errors':region_errors})
 write(OUT/'target_color_analysis.json',{'pure_color_failure_qa':len(color_errors),'label':'COLOR_FAILURE_AFTER_CORRECT_TARGET_REGION','errors':color_errors,'background_contamination_judgment_added':False})
 write(OUT/'accidental_correctness.json',{'final_correct_and_sufficient':final_suff,'final_correct_and_insufficient':len(accidents),'classification_counts':dict(Counter(x['classification'] for x in accidents)),'cases':accidents,'not_counted_as_state_success':True})
 faithful=sum(x['faithful_to_state'] for x in reasoner_cases);executable=sum(x['has_executable_implied_answer'] for x in reasoner_cases);implied_correct=sum(x['state_implied_correct'] for x in reasoner_cases)
 write(OUT/'reasoner_faithfulness.json',{'state_implied_answer_accuracy':metric(implied_correct,40),'final_reasoner_accuracy':metric(finals,40),'reasoner_faithfulness_given_executable_state':metric(faithful,executable),'executable_state_coverage':metric(executable,40),'correct_implied_changed_to_wrong':sum(x['correct_implied_changed_to_wrong'] for x in reasoner_cases),'wrong_implied_changed_to_correct':sum(x['wrong_implied_changed_to_correct'] for x in reasoner_cases),'sufficient_state_reasoner_failures':[x for x in reasoner_cases if x['sufficient'] and not x['final_correct']],'cases':reasoner_cases})
 failures=Counter(x['first_failure_stage'] for x in results);assert sum(failures.values())==40
 failure_order=['TARGET_SELECTION_FAILURE','TARGET_REGION_FAILURE','TARGET_COLOR_FAILURE','FINAL_REASONING_FAILURE','SUCCESS']
 write(OUT/'first_failure_attribution.json',{'counts':{k:failures[k] for k in failure_order},'rates':{k:failures[k]/40 for k in failure_order},'hierarchy':failure_order,'mutually_exclusive':True,'qa':[{'qa_id':x['qa_id'],'first_failure_stage':x['first_failure_stage']} for x in results]})
 available=sum(x['target_candidate_available'] for x in candidate_cases);denom=sum(x['target_candidate_available'] and x['target_correct'] for x in candidate_cases);selected=sum(x['selected_true_target_region_correct'] for x in candidate_cases);both_available_qa=2*oldjoint['families_with_joint_candidate_at_0_5']
 candidate={'true_target_candidate_available_at_0_5':metric(available,40),'target_candidate_oracle_at_0_5':metric(available,40),'selected_true_target_region_correct':metric(selected,40),'selected_chosen_entity_region_correct':metric(tr,40),'selection_success_given_candidate_available_and_target_correct':metric(selected,denom),'old_joint_candidate_available_family_level':metric(oldjoint['families_with_joint_candidate_at_0_5'],20),'old_joint_candidate_available_qa_equivalent':metric(both_available_qa,40),'target_only_availability_gain_percentage_points':100*(available-both_available_qa)/40,'target_only_additional_candidate_covered_qa':available-both_available_qa,'per_qa':candidate_cases}
 write(OUT/'target_candidate_ceiling.json',candidate)
 # Target-only oracle uses true-entity region/color outputs actually saved in the Full route.
 target_oracle_extra=sum(not x['target_correct'] and x['true_target_candidate_available_at_0_5'] for x in results) # diagnostic candidate coverage, not sufficiency
 selector_replay=[]
 for x in results:
  if x['target_correct']:continue
  e=next(r['EVALUATOR_ONLY'] for r in pilot if r['qa_id']==x['qa_id']);truth=x['true_target'];region=e[f'region_{truth}_correct'];color=e[f'color_{truth}_correct'];selector_replay.append({'qa_id':x['qa_id'],'saved_full_true_target_region_correct':region,'saved_full_true_target_color_correct':color,'would_be_sufficient':region and color})
 selector_unlock=sum(x['would_be_sufficient'] for x in selector_replay)
 perfect_color_extra=ttr-suff
 recover={'perfect_target_selection_only':{'additional_sufficient_qa_from_saved_full_route_replay':selector_unlock,'resulting_sufficient_qa':suff+selector_unlock,'replay_cases':selector_replay,'basis':'Full route processed and saved both entity regions/colors; only true-target frozen outputs reused evaluator-side','new_inference':False},'perfect_target_region_only':{'qa_blocked_after_correct_target':t-ttr,'additional_sufficient_qa_lower_bound':0,'additional_sufficient_qa_upper_bound':t-ttr,'resulting_sufficient_qa_bounds':[suff,suff+t-ttr],'reason':'Color reader outputs on hypothetically corrected regions were never observed; saved wrong-region color cannot be transferred'},'perfect_target_color_only':{'additional_sufficient_qa':perfect_color_extra,'resulting_sufficient_qa':suff+perfect_color_extra,'basis':'target and selected target region already correct'},'qa_blocked_at_target_selection':40-t,'qa_blocked_at_target_region_after_correct_target':t-ttr,'qa_blocked_at_target_color_after_correct_target_and_region':ttr-suff,'candidate_covered_wrong_target_qa':target_oracle_extra}
 write(OUT/'marginal_recoverability.json',recover)
 full_to_min=oldpaired['changed_cases'];rescues=[x for x in full_to_min if x['transition']=='wrong_to_correct'];rescue_mechanisms=Counter()
 for x in rescues:
  attrs=x['attribution'];rescue_mechanisms['NON_TARGET_COLOR_NOT_REQUIRED' if 'non-target color unnecessary' in attrs else 'FINAL_STATE_REPRESENTATION_RESPONSE_DIFFERENCE' if 'final state representation/reasoning' in attrs else 'OTHER']+=1
 state_rescues=json.loads((PILOT/'overfactorization_analysis.json').read_text());assert len(state_rescues['solely_non_target_color_cases'])==4 and state_rescues['state_rescues']==4
 comparison={'full_wrong_to_minimal_correct':len(rescues),'full_correct_to_minimal_wrong':oldpaired['counts'].get('correct_to_wrong',0),'rescue_mechanism_counts':dict(rescue_mechanisms),'rescue_cases':rescues,'additional_sufficient_states_from_removing_non_target_color':4,'additional_sufficient_state_case_ids':state_rescues['solely_non_target_color_cases']}
 write(OUT/'full_vs_minimal_analysis.json',comparison)
 make_figures(survival,failures,candidate,metrics,results)
 make_report(results,survival,metrics,failures,candidate,recover,comparison,accidents,faithful,executable,implied_correct,sel_errors,region_errors,color_errors)
 write(OUT/'source_manifest.json',{'read_only_sources':{str(p):sha(p) for p in (PILOT/'route_results.jsonl',PILOT/'prepared.jsonl',PILOT/'new_final_calls.jsonl',PILOT/'paired_transitions.json',PILOT/'overfactorization_analysis.json',STATE/'state_component_results.jsonl',GROUND,HANDOFF,MANIFEST)},'inference_calls_made':0,'model_loads':0,'gt_role':'evaluator only'})

def make_figures(survival,fail,candidate,metrics,rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 f=OUT/'figures';names=['Start','Target','Target+region','Target+region+color','Final on sufficient'];vals=[x['count'] for x in survival]
 plt.figure(figsize=(9,4));plt.plot(names,vals,marker='o');plt.ylim(0,42);plt.ylabel('Surviving QA');plt.xticks(rotation=15,ha='right');plt.tight_layout();plt.savefig(f/'figure1_minimal_survival.png');plt.close()
 order=['TARGET_SELECTION_FAILURE','TARGET_REGION_FAILURE','TARGET_COLOR_FAILURE','FINAL_REASONING_FAILURE','SUCCESS'];plt.figure(figsize=(9,4));plt.bar(['Target','Region','Color','Reasoning','Success'],[fail[x] for x in order]);plt.ylabel('QA');plt.tight_layout();plt.savefig(f/'figure2_first_failure.png');plt.close()
 plt.figure(figsize=(7,4));plt.bar(['True target\ncandidate exists','Selected true\ntarget region correct'],[candidate['true_target_candidate_available_at_0_5']['count'],candidate['selected_true_target_region_correct']['count']]);plt.ylim(0,40);plt.ylabel('QA');plt.tight_layout();plt.savefig(f/'figure3_candidate_vs_selection.png');plt.close()
 good=metrics['P_target_color_correct_given_target_and_region_correct'];plt.figure(figsize=(6,4));plt.bar(['Correct color','Wrong color'],[good['count'],good['denominator']-good['count']]);plt.ylabel('QA with correct target and region');plt.tight_layout();plt.savefig(f/'figure4_color_after_region.png');plt.close()
 suff=[x for x in rows if x['minimal_state_sufficient']];other=[x for x in rows if not x['minimal_state_sufficient']];plt.figure(figsize=(7,4));plt.bar(['Sufficient','Insufficient'],[sum(x['final_answer_correct'] for x in suff),sum(x['final_answer_correct'] for x in other)],label='Final correct');plt.bar(['Sufficient','Insufficient'],[sum(not x['final_answer_correct'] for x in suff),sum(not x['final_answer_correct'] for x in other)],bottom=[sum(x['final_answer_correct'] for x in suff),sum(x['final_answer_correct'] for x in other)],label='Final wrong');plt.ylabel('QA');plt.legend();plt.tight_layout();plt.savefig(f/'figure5_final_by_sufficiency.png');plt.close()

def make_report(rows,survival,m,fail,candidate,recover,comparison,accidents,faithful,executable,implied_correct,sel_errors,region_errors,color_errors):
 def frac(x):return f"{x['count']}/{x['denominator']} ({x['rate']:.3f})"
 decision='MULTI_STAGE_MINIMAL_ROUTE_FAILURE'
 lines=['# Minimal route bottleneck audit','',f'**Recommendation: `{decision}`.** Evaluator-only analysis of the unchanged 20-family/40-QA Minimal Specialist route. No model, prompt, selection policy, state, or final inference was rerun.','','## A. Minimal route survival','','| Stage | Surviving QA | From previous | From all 40 |','|---|---:|---:|---:|']
 for x in survival:lines.append(f"| {x['stage']} | {x['count']} | {x['survival_from_previous']:.3f} | {x['survival_from_all']:.3f} |")
 lines+=['','## B. Conditional bottlenecks','','| Metric | Value |','|---|---:|']
 for label,key in [('P(target correct)','P_target_correct'),('P(target region correct)','P_target_region_correct'),('P(target region correct \| target correct)','P_target_region_correct_given_target_correct'),('P(target color correct)','P_target_color_correct'),('P(target color correct \| target correct)','P_target_color_correct_given_target_correct'),('P(target color correct \| target + region correct)','P_target_color_correct_given_target_and_region_correct'),('P(minimal state sufficient)','P_minimal_state_sufficient'),('P(final correct \| sufficient)','P_final_correct_given_sufficient'),('P(final correct \| insufficient)','P_final_correct_given_insufficient')]:lines.append(f'| {label} | {frac(m[key])} |')
 lines+=['','The unconditional region/color metrics score the selected logical entity, even when target identity is wrong. The survival chain requires correct target identity at every later stage.','','## C. First failure stage','','| Stage | Count | Rate |','|---|---:|---:|']
 for key,label in [('TARGET_SELECTION_FAILURE','Target selection failure'),('TARGET_REGION_FAILURE','Target-region failure'),('TARGET_COLOR_FAILURE','Target-color failure'),('FINAL_REASONING_FAILURE','Final reasoning failure'),('SUCCESS','Success')]:lines.append(f'| {label} | {fail[key]} | {fail[key]/40:.3f} |')
 lines+=['','These mutually exclusive counts are close: 10 target, 11 region, 9 color. The route has no single dominant failure stage. All 10 geometry target errors occur with at least one wrong entity region; none occur despite both regions being GT-compatible.','','## D. Target-only acquisition ceiling','','| Metric | Value |','|---|---:|',f"| True target candidate available@0.5 | {frac(candidate['true_target_candidate_available_at_0_5'])} |",f"| Selected true target region correct | {frac(candidate['selected_true_target_region_correct'])} |",f"| Selection success given candidate available and target correct | {frac(candidate['selection_success_given_candidate_available_and_target_correct'])} |",f"| Old both-entity candidate requirement | {frac(candidate['old_joint_candidate_available_family_level'])} families = {frac(candidate['old_joint_candidate_available_qa_equivalent'])} QA-equivalent |",f"| Target-only availability gain | +{candidate['target_only_additional_candidate_covered_qa']}/40 (+{candidate['target_only_availability_gain_percentage_points']:.1f} percentage points) |",'','The old joint measure is family-level; each locked family contributes two QA, so its QA-equivalent is 24/40. Target-only availability is 29/40. This is an evaluator-only candidate ceiling, not an observed route gain.','','## E. Accidental correctness','','| Type | Count |','|---|---:|',f"| Final correct + sufficient state | {m['P_final_correct_given_sufficient']['count']} |",f'| Final correct + insufficient state | {len(accidents)} |',f"| Wrong target but answer accidentally correct | {sum(x['classification']=='WRONG_TARGET_BUT_SAME_FINAL_COLOR' for x in accidents)} |",f"| Wrong region but same color | {sum(x['classification']=='WRONG_REGION_BUT_SAME_COLOR' for x in accidents)} |",f"| Other compensating error | {sum(x['classification'] not in ('WRONG_TARGET_BUT_SAME_FINAL_COLOR','WRONG_REGION_BUT_SAME_COLOR') for x in accidents)} |",'','The five insufficient-state correct answers are listed in `accidental_correctness.json`; they are not successful state construction.','','## F. Reasoner necessity','','| Metric | Value |','|---|---:|',f'| State-implied-answer accuracy | {implied_correct}/40 ({implied_correct/40:.3f}) |',f"| Final reasoner accuracy | {m['P_final_correct_given_sufficient']['count']+m['P_final_correct_given_insufficient']['count']}/40 (0.375) |",f'| Reasoner faithfulness to executable state | {faithful}/{executable} ({faithful/executable:.3f}) |','| Correct implied answer changed to wrong | 0 |','| Wrong implied answer changed to correct | 0 |','','The two nonexecutable states serialize unknown target/color. On all 38 states with a color, the final LLM returned the state-implied color. Deterministic execution would match the observed 15/40 final accuracy; this is diagnostic and does not alter production inference.','','## Target, region and color cases','',f"Target errors: {len(sel_errors)} wrong-region-induced, 0 despite both regions correct. Correct-target region errors: {len(region_errors)}, with {sum(x['candidate_available'] for x in region_errors)} usable-candidate selection failures and {sum(not x['candidate_available'] for x in region_errors)} without a usable candidate. Pure color errors after correct target and region: {len(color_errors)}. The frozen outputs contain no background-contamination annotations, so no visual judgment was added.",'','## Marginal repair bounds','',f"Perfect target selection alone unlocks {recover['perfect_target_selection_only']['additional_sufficient_qa_from_saved_full_route_replay']} additional sufficient QA when replaying true-target region/color outputs actually saved by the Full route. Perfect target region alone has an honest additional-state bound of 0–{recover['perfect_target_region_only']['additional_sufficient_qa_upper_bound']}; corrected-region color outputs were never observed. Perfect target color alone unlocks {recover['perfect_target_color_only']['additional_sufficient_qa']} additional sufficient QA, raising sufficiency from 10 to {recover['perfect_target_color_only']['resulting_sufficient_qa']}/40.",'','## Full versus Minimal Specialist','',f"The frozen Full route is final-wrong while Minimal Specialist is final-correct in {comparison['full_wrong_to_minimal_correct']} QA: {comparison['rescue_mechanism_counts']}. One QA reverses. Four additional state-sufficient QA come exactly from removing non-target color from the required state; only two of the three final-answer rescues are from that mechanism. The third is a final prompt/state response difference despite both routes having sufficient state.",'','## Scientific answers','',"Q1. No single stage dominates: first failures are target 10, region 11, color 9, final reasoning 0. Region is the largest count by one QA, with target and color close behind.",f"Q2. With target identity correct, selected target region is correct in {frac(m['P_target_region_correct_given_target_correct'])}.",f"Q3. With correct target and region, color is correct in {frac(m['P_target_color_correct_given_target_and_region_correct'])}.",f"Q4. The candidate ceiling rises from 24/40 QA-equivalent for both-entity availability to 29/40 for true-target-only availability, +5 QA or +12.5 points.",f"Q5. {len(accidents)}/15 final-correct QA have insufficient states; 2 involve wrong target and 3 involve wrong target-region color coincidence.","Q6. The final LLM adds no observed value on this subset: it follows all 38 executable minimal states, changes no correct implied answer to wrong, and repairs no wrong implied answer. All 10 sufficient states are answered correctly.","Q7. If one component must be investigated next, choose target color because perfect color has a directly measurable +9 sufficient-state unlock. Region acquisition/selection is a comparably important parallel problem (11 first failures), with an unmeasurable corrected-region color outcome. The recommendation remains multi-stage rather than a single-module fix.","Q8. Perception remains the bottleneck; final reasoning has zero first failures and 10/10 accuracy on sufficient states.",'','## Minimal-decomposition interpretation','',"The minimum path is target identity → selected target region → its color → task-sufficient state. This path fails at several perceptual boundaries; extracting more non-target facts would not address them. Target-only candidate availability improves the ceiling, while current region selection and color reading still prevent that ceiling from becoming sufficient states. The next method refinement should improve target-specific perception without restoring full scene fact extraction.",'','## Scope and reproducibility','',"All 40 locked QA were joined by ID against the previous pilot, the state forensic audit, and the frozen Grounding DINO candidate cache. Ground truth was used only to label saved predictions. The broader historical REL2I ≈0.79 is context only: the present geometry 30/40 and pilot A/B VLM 26/40 use different target-selection protocols, and this 20-family subset cannot establish unusual difficulty relative to the broader set. No inference was rerun.",'']
 (OUT/'implementation_report.md').write_text('\n'.join(lines))
 table=''.join(f"<tr><td>{html.escape(x['qa_id'])}</td><td>{html.escape(str(x['selected_target']))}</td><td>{html.escape(str(x['true_target']))}</td><td>{int(x['target_region_correct'])}</td><td>{int(x['target_color_correct'])}</td><td>{html.escape(x['first_failure_stage'])}</td><td>{html.escape(str(x['state_implied_answer']))}</td><td>{html.escape(str(x['final_prediction']))}</td><td>{html.escape(x['ground_truth'])}</td></tr>" for x in rows)
 (OUT/'audit.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{font:14px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:5px}</style></head><body><h1>Minimal route bottleneck audit</h1><p>40 frozen QA; all correctness columns are evaluator-only.</p><table><tr><th>QA</th><th>Selected</th><th>True target</th><th>Region</th><th>Color</th><th>First failure</th><th>State answer</th><th>LLM answer</th><th>GT</th></tr>'+table+'</table></body></html>')
 (OUT/'test_report.md').write_text('# Validation\n\n- No inference/model imports, loads, or calls.\n- Locked manifest and frozen source hashes checked; exact 20 families and 40 QA.\n- Every saved target, region, color, sufficiency, and final result recomputed and checked.\n- First-failure categories are mutually exclusive and sum to 40.\n- Candidate availability uses frozen Grounding DINO evaluator-only best IoU and threshold 0.5.\n- Strict serialized-state parser used for deterministic implied answer.\n- Saved final calls verified image-free; no new visual judgment added.\n- Five requested figures generated.\n')
 (OUT/'changed_files.txt').write_text('harness/scripts/run_minimal_route_bottleneck_audit.py\noutputs/minimal_route_bottleneck_audit/ (new artifacts)\n')

if __name__=='__main__':main()
