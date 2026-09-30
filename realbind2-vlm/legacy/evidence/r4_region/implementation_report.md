# Region bottleneck joint repair pilot

## Frozen subset and architecture

Locked 20 families / 40 REL2C QA.  Four routes share one frozen legal ReferentSpec per entity, fixed bbox-center geometry, and the same exact-crop Qwen color reader.  A/B region state is acquired once per family and reused for paired LEFT/RIGHT QA.

Route map: R1 = Original GDINO + old selector; R2 = MM-GDINO-L + old selector; R3 = Original GDINO + Qwen local selector; R4 = MM-GDINO-L + Qwen local selector.

## Table A - Candidate generation (evaluator-only oracle)

| Metric | Original GDINO | MM-GDINO-L |
|---|---:|---:|
| Entity CandidateRecall@0.5 | 29/40 | 29/40 |
| True-target CandidateRecall@0.5 | 29/40 | 29/40 |
| Both-entity CandidateRecall@0.5 | 12/20 | 11/20 |
| Mean candidate count | 1.850 | 1.800 |
| Median candidate count | 2.000 | 1.000 |
| Multi-candidate fraction | 0.525 | 0.450 |
| Zero-candidate fraction | 0.025 | 0.000 |

## Table B - Selector quality

| Generator | Selector | SelectedRecall@0.5 | Selection efficiency |
|---|---|---:|---:|
| original | old | 22/40 | 0.759 |
| original | qwen | 25/40 | 0.862 |
| strong | old | 23/40 | 0.793 |
| strong | qwen | 26/40 | 0.897 |

Candidate availability is identical within generator pairs (R1/R3 and R2/R4); assertions are recorded in `state_consistency_audit.json`.

## Table B2 - Both-entity region correctness

| Route | BothSelectedRegionsCorrect | Fraction |
|---|---:|---:|
| R1 original+old | 8/20 | 0.400 |
| R2 strong+old | 6/20 | 0.300 |
| R3 original+qwen | 8/20 | 0.400 |
| R4 strong+qwen | 10/20 | 0.500 |

## Table C - Downstream four-route outcome

| Route | Target correct | Target region correct | Qwen color correct | Minimal state sufficient |
|---|---:|---:|---:|---:|
| R1 original+old | 30/40 | 19/40 | 16/40 | 14/40 |
| R2 strong+old | 28/40 | 19/40 | 17/40 | 13/40 |
| R3 original+qwen | 34/40 | 22/40 | 17/40 | 13/40 |
| R4 strong+qwen | 34/40 | 25/40 | 21/40 | 16/40 |

## Table D - Factor contributions versus R1

| Effect | Selected region | Both-entity families | Target correct | Target region correct | Minimal state sufficient |
|---|---:|---:|---:|---:|---:|
| Generation gain (R2-R1) | +1 | -2 | -2 | +0 | -1 |
| Selection gain (R3-R1) | +3 | +0 | +4 | +3 | -1 |
| Joint gain (R4-R1) | +4 | +2 | +4 | +6 | +2 |
| Generation x selection interaction | +0 | +4 | +2 | +3 | +4 |

## Table E - Paired transitions at target region level

| Comparison | Fail→Correct | Correct→Fail | Both correct | Both fail |
|---|---:|---:|---:|---:|
| R1→R2 | 4 | 4 | 15 | 17 |
| R1→R3 | 5 | 2 | 17 | 16 |
| R1→R4 | 10 | 4 | 15 | 11 |
| R2→R4 | 7 | 1 | 18 | 14 |
| R3→R4 | 8 | 5 | 17 | 10 |

## Table E - Paired transitions at minimal state level

| Comparison | Fail→Correct | Correct→Fail | Both correct | Both fail |
|---|---:|---:|---:|---:|
| R1→R2 | 2 | 3 | 11 | 24 |
| R1→R3 | 2 | 3 | 11 | 24 |
| R1→R4 | 6 | 4 | 10 | 20 |
| R2→R4 | 4 | 1 | 12 | 23 |
| R3→R4 | 6 | 3 | 10 | 21 |

## Survival funnels

| Route | Start | Target correct | Target region correct | Target color correct after region | Minimal state sufficient |
|---|---:|---:|---:|---:|---:|
| R1 | 40 | 30 | 19 | 14 | 14 |
| R2 | 40 | 28 | 19 | 13 | 13 |
| R3 | 40 | 34 | 22 | 13 | 13 |
| R4 | 40 | 34 | 25 | 16 | 16 |

## Oracle ceiling, selection headroom, and failure location

- Evaluator-only true-target candidate oracle: original 29/40; MM-GDINO-L 29/40.
- R4 selector headroom among usable candidates: 3/40 entities.
- R4 target-region failures: 15/40; the generator itself has no usable true-target candidate for 11/40 QA, and among the 25 QA with a correct target region, 9 fail at target color.
- R4 color accuracy conditional on a correct target region: 0.640; R1 baseline is 0.737.
- R4 newly recovered target regions versus R1: 10; Qwen color correct on those regions: 6.

## Table I - Selected-target IoU quality and color stratification

| Route | Mean IoU | Median IoU | 0.50-0.70 color | 0.70-0.85 color | 0.85-1.00 color |
|---|---:|---:|---:|---:|---:|
| R1 | 0.871 | 0.934 | 3/3 | 1/2 | 10/14 |
| R2 | 0.854 | 0.924 | 1/3 | 2/4 | 10/12 |
| R3 | 0.822 | 0.889 | 4/6 | 2/3 | 7/13 |
| R4 | 0.841 | 0.924 | 4/5 | 1/4 | 11/16 |

## Same-category / singleton selection analysis

| Subset | n | Available | Old correct | Qwen correct |
|---|---:|---:|---:|---:|
| original_singleton | 18 | 11 | 11 | 11 |
| original_multi_candidate | 21 | 18 | 11 | 14 |
| original_same_category_ambiguous | 21 | 18 | 11 | 14 |
| strong_singleton | 22 | 14 | 14 | 14 |
| strong_multi_candidate | 18 | 15 | 9 | 12 |
| strong_same_category_ambiguous | 18 | 15 | 9 | 12 |

Qwen selector invalid-fallback rate: R3 0.000, R4 0.000; invalid outputs fall back to the old selector and are recorded explicitly.

## Rescue provenance

- R1_to_R2 entity: 5 fail→correct cases; provenance `{'old_no_usable_candidate_strong_introduced_usable': 4, 'both_pools_had_usable_candidate_strong_box_or_ranking_gain': 1}`.
- R1_to_R2 target_region_qa: 4 fail→correct cases; provenance `{'old_no_usable_candidate_strong_introduced_usable': 3, 'both_pools_had_usable_candidate_strong_box_or_ranking_gain': 1}`.
- R1_to_R4 entity: 8 fail→correct cases; provenance `{'old_no_usable_candidate_strong_introduced_usable': 5, 'both_pools_had_usable_candidate_strong_box_or_ranking_gain': 3}`.
- R1_to_R4 target_region_qa: 10 fail→correct cases; provenance `{'old_no_usable_candidate_strong_introduced_usable': 4, 'both_pools_had_usable_candidate_strong_box_or_ranking_gain': 6}`.
- R1_to_R3 selection rescues: 5 entity fail→correct cases; correct candidate already existed in 5.
- R2_to_R4 selection rescues: 4 entity fail→correct cases; correct candidate already existed in 4.

Every rescue row in `generation_rescues.json` / `selection_rescues.json` records the old/available flags, selected IoUs, referent phrase, candidate count, and same-category status; GT remains evaluator-only.

## Efficiency

| Route | Detector calls/family (new in pilot) | Selector Qwen calls/family | Color Qwen calls/family | Color cache reused QA |
|---|---:|---:|---:|---:|
| R1 | 0.0 | 0.00 | 0.00 | 38 |
| R2 | 2.0 | 0.00 | 2.00 | 0 |
| R3 | 0.0 | 1.05 | 0.00 | 38 |
| R4 | 2.0 | 0.90 | 2.00 | 0 |

Across all four routes, 51 unique exact-bbox target crops required new Qwen color inference; the remaining route/QA crop requests were satisfied by the exact canonical bbox cache.  R2 and R4 share many of those 51 calls because they use the same strong candidate pools, so route-level per-family counts above should be read as conditional-isolation counts, not as additive wall-clock costs.

The original detector pools are the frozen prior outputs; MM-GDINO-L was run once for 40 referents and shared by R2/R4.  Exact-bbox color outputs from the earlier color-specialist pilot were reused only when the canonical image+bbox key matched exactly; every other target crop was read by the fixed crop-only Qwen color prompt.  The corrected MM-GDINO-L run used Transformers 4.55.4; the initial Transformers 5.8.1 run was invalid, archived under `invalid_5_8_1_run/`, and excluded from all results.

## Scientific answers

**Q1.** MM-GDINO-L candidate availability changes by +0/40 entities (original 29/40, strong 29/40); both-entity family recall changes by -1/20.
**Q2.** Missing candidates versus wrong selection: under R1 the evaluator-only oracle leaves 29/40 true-target entities available and the old selector leaves 7/40 usable entities unselected.  Under R4, 29/40 true-target entities are available and 3/40 usable entities remain unselected; candidate generation is still the larger unresolved region loss.
**Q3.** Qwen local referent matching improves selection efficiency on original frozen pools from 0.759 to 0.862; on MM-GDINO-L pools it improves from 0.793 to 0.897.
**Q4.** Qwen selection helps multi-candidate/same-category cases: original multi-candidate subset old 11→Qwen 14; strong multi-candidate subset old 9→Qwen 12.  Singleton cases are unchanged because they bypass the selector.
**Q5.** Gains are complementary but not additive: R4 is the strongest route (state 16/40); the generation×selection interaction is +4 QA and +3 target-region QA.  Generation alone (R2) is not a state-sufficiency gain in this run.
**Q6.** R4 target correctness is 34/40 and target-region correctness is 25/40, versus R1 30/40 and 19/40.
**Q7.** Newly recovered regions remain partially usable by the crop-only Qwen color reader: among R4 target regions newly recovered versus R1, color is correct on 6/10; overall R4 has 21/40 target-color correct answers.
**Q8.** R4 minimal-state sufficiency is 16/40.
**Q9.** The frozen image-free final reasoner was run on all four routes. R4 reaches 21/40 measured final accuracy versus the locked Vanilla VLM 20/40, but only 16/40 comes from state-sufficient cases; 5/40 are compensating errors with an insufficient state. This is a measured final-accuracy pass but not a clean state-sufficiency pass.
**Q10.** After this repair the dominant remaining bottleneck is still **candidate generation**: MM-GDINO-L and the original grounder both leave 11/40 true-target entities without a usable candidate.  Selection is much less limiting (3/40 headroom), and target color is the next local failure (9/40 correct-region QA lose the state at color).

## Decision logic

Primary recommendation: **`GENERATION_REMAINS_BOTTLENECK`**.

The joint repair works partially: R4 is the strongest route for target-region acquisition (25/40) and state sufficiency (16/40), the Qwen selector materially improves conditional selection, and the measured image-free final accuracy is 21/40 versus Vanilla 20/40.  However, the 18/40 state-sufficiency gate is not met, and the remaining dominant loss is still candidate generation: MM-GDINO-L leaves 11/40 true-target entities without a usable candidate, while selector headroom is only 3/40.  The 21/40 final accuracy includes 5 compensating-error cases whose state was insufficient.  Per the predeclared gate, do not keep tuning selector tricks; the next work item is an alternative region formulation, richer local visual representation, or soft handoff aimed at the generation/color-interface losses.

## Table F - Measured image-free final accuracy (if run)

| Method | Final accuracy |
|---|---:|
| Vanilla VLM (locked baseline) | 20/40 |
| Old Minimal Specialist (locked) | 15/40 |
| R1 original+old | 16/40 |
| R2 strong+old | 17/40 |
| R3 original+qwen | 17/40 |
| R4 strong+qwen | 21/40 |

Measured state sufficiency and final accuracy are separated below:

| Route | State sufficient | Sufficient & final correct | Insufficient but final correct | Final correct |
|---|---:|---:|---:|---:|
| R1 | 14/40 | 14/40 | 2/40 | 16/40 |
| R2 | 13/40 | 13/40 | 4/40 | 17/40 |
| R3 | 13/40 | 13/40 | 4/40 | 17/40 |
| R4 | 16/40 | 16/40 | 5/40 | 21/40 |

## Acceptance audit

- Same frozen 20 families / 40 QA: yes.
- MM-GDINO-L local, no API / SAM / ensemble / fusion / blur / certification: yes.
- Legal referent information only in inference: yes; GT joined only in this analysis.
- Four routes evaluated: yes.
- Original candidate pools identical R1/R3 and strong pools identical R2/R4: asserted.
- Qwen selector sees only numbered candidate crops + legal referent phrase: yes; no question/side/answer/GT.
- Geometry unchanged: yes; bbox-center relation.
- Crop-Qwen color prompt unchanged: yes; fixed prompt SHA recorded.
- Non-target color never required: yes.
- Paired rescues/regressions and oracle ceilings reported: yes.
- State sufficiency and final accuracy separated: yes.

## Novelty interpretation

This is query-conditioned minimal visual compilation with primitive-specific processors: a local high-recall grounder proposes hypotheses, a narrow local referent matcher resolves entity binding, deterministic geometry transforms the relation, a local VLM reads the target attribute, and only a minimal executable state is handed to an image-free reasoner.  The pilot shows the architecture is scientifically useful even when the grounder swap is not: it localizes the remaining errors to candidate generation and crop-level attribute reading rather than final reasoning.
