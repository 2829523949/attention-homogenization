# SAM3 instance-hypothesis consolidation + instance-level binding

**DEVELOPMENT-ONLY FROZEN COHORT.** The same heavily inspected 20 families and 40 REL2C QA rows are reused. These are not held-out paper generalization results.

## Required consolidation table

| Metric | Raw SAM3 | Consolidated |
|---|---:|---:|
| Mean candidates/image/referent | 64.00 | 23.93 hypotheses |
| Median candidates/image/referent | 64 | 24 hypotheses |
| Oracle@0.5 | 35/40 | 35/40 |
| Compression ratio | 1.0 | 2.675 |
| Same-instance duplicate rate | 0.153 | 0.000 |

## Required binding table

| Method | HypothesisCorrect | RepresentativeCorrect | GroundedTargetCorrect |
|---|---:|---:|---:|
| Historical Integrated | — | 27/40 entity-level | 26/40 |
| Raw ObjEmbed (prior frozen union pool) | — | 26/40 entity-level | 24/40 |
| Raw ObjEmbed (same SAM3 top64 pool) | — | 17/40 entity-level | 14/40 |
| Hypothesis ObjEmbed Independent | 22/40 | 7/40 | 6/40 |
| Hypothesis ObjEmbed Joint | 22/40 | 7/40 | 6/40 |

## Required oracle chain

| Stage | Count |
|---|---:|
| RawCandidateOracle | 35/40 |
| HypothesisOracle | 35/40 |
| SelectedHypothesisCorrect | 22/40 |
| RepresentativeRegionCorrect | 7/40 |
| GroundedTargetCorrect | 6/40 |

## Proposal-depth analysis

| Depth | Raw usable entity coverage | Consolidated usable hypothesis coverage |
|---|---:|---:|
| top16 | 31/40 | 31/40 |
| top32 | 33/40 | 33/40 |
| top64 | 35/40 | 35/40 |

## Hypothesis-size analysis

| Bucket | Selected hypotheses | HypothesisCorrect | RepresentativeCorrect | Grounded binding |
|---|---:|---:|---:|---:|
| 2-3_members | 6 | 0 | 0 | 0 |
| 4-8_members | 5 | 2 | 1 | 1 |
| >8_members | 24 | 19 | 5 | 4 |
| singleton | 5 | 1 | 1 | 1 |

## Part-whole hierarchy analysis

| Group | Selected hypotheses | HypothesisCorrect | RepresentativeCorrect | Grounded binding |
|---|---:|---:|---:|---:|
| no_part_whole_hierarchy | 5 | 1 | 1 | 1 |
| contains_part_nodes | 6 | 1 | 1 | 1 |
| deep_hierarchy | 29 | 20 | 5 | 4 |

## Context rescue / regression audit

Historical wrong → hypothesis correct: entity hypothesis 6; representative 1; grounded QA 0.
Historical correct → hypothesis wrong: entity hypothesis 11; representative 21; grounded QA 20.
Representative failure counts: {"hypothesis_correct_representative_correct": 7, "hypothesis_correct_representative_wrong": 15, "hypothesis_wrong": 18}. Representative selection is the dominant loss: 15 selected hypotheses are correct but their frozen representative is not.
Evaluator-only representative headroom: selected hypotheses contain a GT-correct member for 22/40 entities, while the frozen representative is correct for 7/40; 15 entities are lost inside the hypothesis by representative selection.

### Rescue case audit

- `realbind_test_2397782_2397782000_2397782007|A` (flower): historical IoU=0.1590; selected hyp=H006 members=19, depth=3; raw ObjEmbed top=s021 rank=22 IoU=0.8184; correct-member ranks=[22, 29]; representative=s021 IoU=0.8184.
- `realbind_val_2369344_2369344002_2369344001|B` (hat): historical IoU=0.0422; selected hyp=H002 members=23, depth=3; raw ObjEmbed top=s001 rank=2 IoU=0.4330; correct-member ranks=[1, 10, 39]; representative=s008 IoU=0.0473.
- `realbind_test_2347354_2347354005_2347354023|A` (banana): historical IoU=0.0000; selected hyp=H000 members=55, depth=6; raw ObjEmbed top=s012 rank=13 IoU=0.0000; correct-member ranks=[5]; representative=s001 IoU=0.0085.
- `realbind_test_2347354_2347354005_2347354023|B` (sauce): historical IoU=0.4979; selected hyp=H000 members=44, depth=5; raw ObjEmbed top=s007 rank=8 IoU=0.5949; correct-member ranks=[8, 5]; representative=s000 IoU=0.1336.
- `realbind_test_2362197_2362197010_2362197013|B` (jacket): historical IoU=0.3940; selected hyp=H000 members=21, depth=5; raw ObjEmbed top=s001 rank=2 IoU=0.6331; correct-member ranks=[2, 25, 5]; representative=s000 IoU=0.4304.
- `realbind_test_2374119_2374119000_2374119004|A` (blanket): historical IoU=0.1761; selected hyp=H000 members=60, depth=5; raw ObjEmbed top=s046 rank=47 IoU=0.0632; correct-member ranks=[60]; representative=s001 IoU=0.0457.

### Regression case audit

- `realbind_test_2368070_2368070003_2368070008|A` (racket): historical IoU=0.7950; selected hyp=H000 has member IoUs=[0.0, 0.0, 0.0, 0.0]; raw ObjEmbed top=s048 rank=49 IoU=0.0000; representative IoU=0.0000. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2380660_2380660001_2380660004|A` (candle): historical IoU=0.9754; selected hyp=H002 has member IoUs=[0.0, 0.0]; raw ObjEmbed top=s007 rank=8 IoU=0.0000; representative IoU=0.0000. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2380660_2380660001_2380660004|B` (rug): historical IoU=0.9257; selected hyp=H000 has member IoUs=[0.078, 0.073, 0.041, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; raw ObjEmbed top=s019 rank=20 IoU=0.0778; representative IoU=0.0730. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2327340_2327340003_2327340002|B` (tree): historical IoU=0.8608; selected hyp=H024 has member IoUs=[0.012]; raw ObjEmbed top=s061 rank=62 IoU=0.0120; representative IoU=0.0120. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2374119_2374119000_2374119004|B` (couch): historical IoU=0.7719; selected hyp=H001 has member IoUs=[0.084, 0.101, 0.092, 0.272, 0.251, 0.077, 0.064, 0.1, 0.0, 0.0, 0.046, 0.0, 0.18, 0.178, 0.0, 0.12, 0.0, 0.101, 0.005, 0.081, 0.0, 0.0, 0.0, 0.0, 0.0, 0.053, 0.0, 0.0]; raw ObjEmbed top=s003 rank=4 IoU=0.0838; representative IoU=0.0915. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2383540_2383540002_2383540001|A` (shirt): historical IoU=0.5093; selected hyp=H000 has member IoUs=[0.193, 0.066, 0.087, 0.153, 0.014, 0.272, 0.207, 0.024, 0.0, 0.038, 0.015, 0.012, 0.034, 0.03]; raw ObjEmbed top=s001 rank=2 IoU=0.1934; representative IoU=0.1528. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2383540_2383540002_2383540001|B` (uniform): historical IoU=0.5660; selected hyp=H003 has member IoUs=[0.0, 0.0, 0.0]; raw ObjEmbed top=s006 rank=7 IoU=0.0000; representative IoU=0.0000. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2319488_2319488001_2319488002|A` (apple): historical IoU=0.9258; selected hyp=H013 has member IoUs=[0.31]; raw ObjEmbed top=s063 rank=64 IoU=0.3096; representative IoU=0.3096. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2384340_2384340003_2384340005|A` (floor): historical IoU=0.8495; selected hyp=H014 has member IoUs=[0.249, 0.101, 0.339]; raw ObjEmbed top=s003 rank=4 IoU=0.2486; representative IoU=0.3389. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2386641_2386641002_2386641000|B` (street sign): historical IoU=0.5004; selected hyp=H000 has member IoUs=[0.354, 0.026, 0.427, 0.341, 0.409, 0.39, 0.355, 0.338, 0.064, 0.276, 0.186, 0.0, 0.293, 0.167, 0.282, 0.339, 0.0, 0.269, 0.155, 0.01, 0.0, 0.006, 0.254, 0.266, 0.023, 0.251, 0.024, 0.2, 0.0, 0.159, 0.0, 0.053, 0.0, 0.0, 0.243, 0.0, 0.132, 0.169, 0.0, 0.063, 0.054, 0.0, 0.0, 0.0, 0.034, 0.048, 0.007, 0.0, 0.208, 0.091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; raw ObjEmbed top=s062 rank=63 IoU=0.3535; representative IoU=0.2763. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.
- `realbind_test_2413686_2413686004_2413686003|A` (floor): historical IoU=0.6488; selected hyp=H030 has member IoUs=[0.025]; raw ObjEmbed top=s052 rank=53 IoU=0.0247; representative IoU=0.0247. Reason: Selected hypothesis contains no GT-correct member; semantic binding selected another instance/context hypothesis.

## Required scientific questions

**Q1. How many raw top64 candidates belong to duplicate/part-whole families?** Duplicate components of size >1 contain 392/2560 raw candidates (15.31%); near-duplicate pair rate is 0.38%. Part-containing hypotheses are 277/957 (28.94%).
**Q2. How much is candidate/hypothesis count reduced?** 2560 raw proposals become 957 hypotheses (compression 2.68).
**Q3. Is oracle recall preserved?** Raw top64 35/40 vs consolidated 35/40; damage 0/40, gate pass=True.
**Q4. Is raw SAM3 high entropy the main binding-failure cause?** Consolidation preserves recall and raises selected-hypothesis correctness to 22/40, but representative selection drops it to 7/40 and grounded target to 6/40. High entropy is therefore not the only bottleneck for this development subset.
**Q5. Does ObjEmbed at hypothesis level beat raw candidate level?** On the same SAM3 top64 pool, raw selected-region correct 17/40 and grounded 14/40; selected-hypothesis correctness rises to 22/40, but representative/grounded fall to 7/40 and 6/40.
**Q6. Does instance-level joint assignment beat candidate-level joint assignment?** Hypothesis joint GroundedTargetCorrect 6/40 vs raw candidate-level joint 14/40 on the same pool; the hard H_i!=H_j constraint did not improve the frozen downstream count.
**Q7. How often is the hypothesis right but the representative wrong?** 15/22 correct hypotheses; this triggers `REPRESENTATIVE_SELECTION_BOTTLENECK`.
**Q8. Does GroundedTargetCorrect exceed historical 26/40?** No, best hypothesis route is 6/40.
**Q9. Is 30/40 reached?** No.
**Q10. Does downstream clean final exceed Vanilla 20/40?** Full downstream was skipped by the frozen <24/40 gate, so this is not reached; Vanilla remains 20/40.

## Decision label

`REPRESENTATIVE_SELECTION_BOTTLENECK`

Secondary finding: consolidation preserved recall but hypothesis-level binding did not exceed historical; per Stop B/C, no further matcher or SAM3 selector engineering should be added on this subset.

## Limitations

- Development set, repeatedly inspected; not held-out evidence.
- Historical top64 cache contained boxes/scores but not masks for ranks 17-64; masks were regenerated with the frozen checkpoint and Hungarian-matched to frozen boxes/scores.
- The structural root representative rule is faithful to the predeclared lexicographic order, but SAM3 context regions frequently become roots, causing the representative bottleneck.
