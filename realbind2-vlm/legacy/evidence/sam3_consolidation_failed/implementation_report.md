# Implementation report: SAM3 instance hypothesis consolidation + binding

## Scope

Only SAM3 candidate consolidation and instance-level referent binding were changed. The frozen 20-family/40-QA subset, ReferentSpec, SAM3 checkpoint, top64 K source, ObjEmbed-2B checkpoint/query, geometry, downstream isolation, color reader, minimal state and final reasoner were reused.

## Consolidation

- Fixed thresholds: duplicate mask IoU 0.7, containment fraction 0.9, area ratio 0.7, high-overlap report IoU 0.3.
- Raw proposal structure is in `proposal_structure.jsonl` and `raw_proposal_structure_report.md`.
- Raw total 2560 -> hypotheses 957 (compression 2.675).
- Consolidation oracle gate: raw top64 35/40 vs consolidated 35/40; damage 0/40; pass True.

## Binding

- Primary matcher: frozen ObjEmbed-2B official REC path.
- Hypothesis score = max member score; mean and top2 mean are recorded diagnostically but not used for selection.
- Independent selection: per-referent hypothesis argmax. Joint selection: argmax of S_A(H_i)+S_B(H_j) with only H_i != H_j.
- Representative region: root member > valid mask > native score > mask area > rank, fixed before evaluation.

## Results

- Best hypothesis route: `HYPOTHESIS_OBJEMBED_JOINT`.
- HypothesisCorrect 22/40, RepresentativeRegionCorrect 7/40, GroundedTargetCorrect 6/40.
- Main decision: `REPRESENTATIVE_SELECTION_BOTTLENECK`.
- Evaluator-only representative headroom: selected hypotheses contain a GT-correct member for 22/40 entities; frozen representative is correct for 7/40.

## Resume and audits

- All stages write atomic JSON/JSONL; ObjEmbed records are appended and fsynced per referent.
- `resume_status.json` records completed stages and gate outcomes.
- `gt_leakage_audit.json` verifies clustering/scoring artifacts contain no evaluator fields.
- `state_consistency_audit.json` verifies A/B hypothesis/region sharing across LEFT/RIGHT paired QA.
- Full downstream inference was skipped because the best hypothesis route grounded at 6/40, below the predeclared 24/40 gate; `downstream_results.jsonl` contains the minimal binding-level validation.
- `case_audit.json`, `representative_failure_analysis.json` and `representative_oracle_headroom.json` provide the post-hoc rescue/regression and within-hypothesis representative diagnosis.
