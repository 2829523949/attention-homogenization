# RealBind referent-interface audit

Status: **AUDITED — legal semantic resolution exists for composition and relation prompts; atomic A/B role is protocol-dependent.**

Date: 2026-09-23

This audit was performed before adding the corrected grounding branch. It reads the existing frozen v1.1 family/QA release, the v1/v1.1 QA generators, the current D-drive harness, and the historical Stage-1/Stage-2 runners. It does not regenerate, rewrite, or score the frozen release.

## Source inventory

| Artifact | Path | Observed role |
|---|---|---|
| Frozen families | `/mnt/d/datasets/RealBind/realbind/families/dryrun_100_pair_direct_v1_1_frozen.jsonl` | 100 selected families; contains evaluator annotations for A/B |
| Frozen QA | `/mnt/d/datasets/RealBind/realbind/qa/dryrun_100_pair_direct_v1_1.jsonl` | 700 rows, seven task types, fixed prompts |
| Historical v1 QA | `/mnt/d/datasets/RealBind/realbind/qa/dryrun_100_qa.jsonl` | 1,000-row historical release; preserved, not used by the corrected main path |
| Dependency specification | `/mnt/d/datasets/RealBind/realbind/qa/pair_direct_v1_1_dependency_spec.json` | evaluator/protocol dependency graph |
| v1.1 generator | `/root/Qwen3-VL/experiments/realbind_phase2/scripts/generate_pair_direct_qa_v1_1.py` | confirms fixed templates and referential identity audit |
| Existing harness | `/mnt/d/vlm_repair_v2/harness` | legacy generic proposal path plus partial processor code |
| Historical Stage-1/Stage-2 | `/root/Qwen3-VL/experiments/realbind_phase2/qwen35_dah_oracle_eval.py`, `qwen35_stage2_role_handoff_eval.py`, `qwen35_stage2_binding_guided_routing_v1.py` | historical evidence only; not modified |

Frozen v1.1 hashes recorded by the existing manifest are:

- families: `b8d366c296403646a8c9a71c09cf8dd8ba007eea2d1d7a3c930cd95edd1816ee`
- QA: `433ccc24f7a7070ae56f756e8936f10c181105e6fa31b3fa9bf206fcf06c3c87`
- dependency: `9238308038aa552f307b8cbb25058a373e0d979dd408656dacc82ace7e57cdf9`
- scorer: `841f6b0dfbbd1710e9c1e74ed583fc609a5113740ad83cf9090ad71bc237f18e`

The v1.1 release contains 100 families, 700 QA rows, 100 rows of each task type, and 200 REL2C composition cases. All 700 rows have `referential_identity_valid=true`, `template_generic_noun_valid=true`, and `template_generic_object_word_violation=false`. All released rows have `object_A_unique=true`, `object_B_unique=true`, and `annotation_identity_unique=true`. The split is 623 test and 77 val rows.

## What the user-visible prompts contain

The released v1.1 templates are:

```text
I2C_A:            What color is the {object_A}?\nAnswer with only the color.
I2C_B:            What color is the {object_B}?\nAnswer with only the color.
PAIR_REL_A_TO_B:  Is the {object_A} to the left or right of the {object_B}?\nAnswer with only left or right.
REL2I_LEFT:       Between the {object_A} and the {object_B},\nwhich one is on the left?\nAnswer with only the object name.
REL2I_RIGHT:      Between the {object_A} and the {object_B},\nwhich one is on the right?\nAnswer with only the object name.
REL2C_LEFT:       Between the {object_A} and the {object_B},\nwhat color is the one on the left?\nAnswer with only the color.
REL2C_RIGHT:      Between the {object_A} and the {object_B},\nwhat color is the one on the right?\nAnswer with only the color.
```

The nouns are actual category strings such as `racket` and `shirt`. None of the 700 questions contains literal standalone `A` or `B`. No question contains a color descriptor identifying a referent. No visual marker or referring expression beyond the category nouns and the left/right template language is present.

## Task-by-task legal interface

| Task type | Visible question content | Hidden/evaluator-only fields | Legal ReferentSpec construction | Automatic status |
|---|---|---|---|---|
| `REL2C_LEFT` | Both category nouns in order; target side `left`; asks for color | `family.A/B.bbox`, `instance_id`, `canonical_color`, `left_is_A`, `target_color`, `ground_truth` | `A` = first noun and `B` = second noun, each `category_constraint` from `QUESTION_TEXT`; target selection is a predicate obligation, not a hidden identity label | `RESOLVABLE_FROM_TEXT` |
| `REL2C_RIGHT` | Both category nouns in order; target side `right`; asks for color | Same as above | Same as above | `RESOLVABLE_FROM_TEXT` |
| `REL2I_LEFT` | Both category nouns in order; target side `left`; asks for identity | Same as above | Same as above; relation/selection node points to the left entity | `RESOLVABLE_FROM_TEXT` |
| `REL2I_RIGHT` | Both category nouns in order; target side `right`; asks for identity | Same as above | Same as above; relation/selection node points to the right entity | `RESOLVABLE_FROM_TEXT` |
| `PAIR_REL_A_TO_B` | Both category nouns in order; asks whether first is left/right of second | `left_is_A`, bboxes, instance IDs, relation answer, `ground_truth` | `A` = first noun and `B` = second noun; relation is acquired/derived from regions | `RESOLVABLE_FROM_TEXT` |
| `I2C_A` | One category noun, the noun rendered from A; asks for color | A/B annotation, bbox, color, instance ID, `ground_truth` | If the protocol task label `I2C_A` is supplied as a legal task operation, create `A` from the visible noun. If only the raw question is supplied, create `TARGET` and do not infer A/B from the sentence. | `RESOLVABLE_WITH_PROTOCOL_ROLE` |
| `I2C_B` | One category noun, the noun rendered from B; asks for color | A/B annotation, bbox, color, instance ID, `ground_truth` | Same rule with protocol role `B`; raw-question-only interface remains `TARGET`. | `RESOLVABLE_WITH_PROTOCOL_ROLE` |

The corrected smoke test therefore uses REL2C cases: both logical references are text-grounded without relying on a family record. The automatic pipeline never receives `family.A`, `family.B`, `bbox_A`, `bbox_B`, `left_is_A`, colors, instance IDs, or any target label.

## Existing interface findings

The old D-drive `harness.dataset.composition_cases()` joins QA rows to frozen family rows and copies `family["A"]["object"]` and `family["B"]["object"]` into an `InferenceCase`. That duplicates visible nouns for v1.1, but it is still the wrong boundary and would become leakage if used as a general referent mechanism. The refactor removes that shortcut from the corrected path and parses the nouns from the frozen question.

The old `harness.pipeline.acquire_visual_state()` calls Florence `<OD>`/`<REGION_PROPOSAL>`, then maps generic labels to A/B and optionally calls an open-vocabulary query. This is tagged `LEGACY_GENERIC_PROPOSAL_BASELINE`; it is not selected by the main configuration.

Historical Stage-1/Stage-2 runners use family/QA metadata to validate outputs and map a generated identity back to A/B. Those artifacts remain historical evidence and evaluator-side analysis. They are not used to construct automatic ReferentSpecs in the corrected main path.

## Audit conclusion

The frozen v1.1 composition interface is **not underspecified for REL2C/REL2I/PAIR_REL**: category nouns and the selection relation are visible. It is not legal to treat `A` or `B` as a detector query; they are logical labels assigned from the visible template order. Atomic I2C rows require an explicit protocol role if the pipeline must preserve the A/B label; otherwise the safe raw-question interface is a single `TARGET` referent.

No `REFERENT_INTERFACE_BLOCKER.md` is required for the REL2C smoke branch. Any future task family whose prompt contains only opaque A/B symbols, or whose noun is absent/non-unique, must be marked `REFERENT_INTERFACE_UNDERSPECIFIED` and abstained from automatic grounding rather than repaired with hidden metadata.
