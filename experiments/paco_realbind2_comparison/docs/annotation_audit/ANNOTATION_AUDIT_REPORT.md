# PACO Annotation Audit for the RealBind2-Protocol Runner

Source data:

- PACO-LVIS v1 annotations, test + val
- Candidate pool: `data/paco_lvis_candidates_test_val_4k`
- Audit output: `outputs/paco_realbind2_comparison/annotation_audit`

This audit asks whether the annotation fields used by the current runner are
valid and whether a bare-category referent produces a meaningful answer.

## Verdict

The raw candidate pool is **not safe to use as-is** for the main PACO comparison.
The main problems are referent ambiguity and multi-color ambiguity, not missing
annotation fields. The annotation fields themselves are structurally usable.

The benchmark builder has therefore been changed to a curated default and the
curated test benchmark now has:

- **2,138 families**
- **4,276 side-specific QAs**
- **1,335 unique images**

The val audit/curation gives 571 families / 1,142 QAs / 346 images.

## Raw test-candidate findings

| Check | Count | Share of 3,184 |
|---|---:|---:|
| At least one A/B category repeated in image | 2,079 | 65.3% |
| At least one object has multiple color attributes | 2,056 | 64.6% |
| Both objects have exactly one color attribute | 1,128 | 35.4% |
| Annotation A/B IoU ≥ 0.1 | 290 | 9.1% |
| Annotation A/B IoU ≥ 0.3 | 90 | 2.8% |
| Annotation A/B IoU ≥ 0.5 | 33 | 1.0% |
| Max containment fraction ≥ 0.5 | 405 | 12.7% |
| Either bbox touches image border | 1,529 | 48.0% |
| Either bbox area < 0.5% of image | 516 | 16.2% |

Additional checks:

- `dom_color_ids` was exactly one for all candidate objects; no empty/unknown
  dominant-color labels were found in this pool.
- `unknown_color` was zero for all candidate objects.
- No candidate pair had the same display referent after category mapping.
- Some repeated-category cases still have a unique non-color attribute
  (material/pattern/transparency). The curated builder uses that attribute to
  disambiguate; only 92 test families had no color-free unique referent and were
  excluded.

## High-overlap examples

Examples with very high annotation A/B IoU:

| family | A/B categories | annotation IoU |
|---|---|---:|
| PACOLVIS_TEST_F000883 | tray / pan | 0.992 |
| PACOLVIS_TEST_F002989 | trash_can / plastic_bag | 0.987 |
| PACOLVIS_TEST_F002556 | cup / mug | 0.968 |
| PACOLVIS_TEST_F000791 | mug / cup | 0.965 |
| PACOLVIS_TEST_F003184 | table / plate | 0.941 |
| PACOLVIS_TEST_F000513 | plate / table | 0.937 |
| PACOLVIS_TEST_F001378 | tray / plate | 0.897 |
| PACOLVIS_TEST_F001523 | plate / table | 0.886 |

These are not all annotation errors: small objects can legitimately sit inside a
table or tray. However, for the REL2C question they produce weak or ambiguous
left/right evidence and crop contamination. They should not be admitted by the
default benchmark construction.

A separate contact-sheet folder is provided:

```text
previews/top_annotation_overlap_16.jpg
previews/top_annotation_containment_16.jpg
previews/curated_referent_preview_16.jpg
```

Blue = A annotation box; red = B annotation box.

## Meaning-ambiguity findings

### 1. Bare category referents are often ambiguous

Example: an image with 5 cups and 5 mugs. `Between the cup and the mug ...` does
not identify the intended instance. A bare-category benchmark would therefore
test an ill-posed question, not compositional evidence binding.

The curated builder now uses:

- bare category if that category occurs once in the image;
- otherwise `the <unique non-color attribute> <category>`, e.g.
  `the metal wrench`, `the ceramic plate`, `the plain shoe`;
- exclude if no color-free unique descriptor exists.

This avoids leaking the queried color while making the referent unambiguous.

### 2. “Color” is not always one unique attribute

PACO stores an object's dominant color in `dom_color_ids`, but
`attribute_ids` can contain several color attributes for the same object. Using
the exact RealBind2 prompt `What color is ...?` would make valid alternative
colors look wrong.

The curated PACO benchmark therefore says **dominant color** in the REL2C
question and in the R4 crop attribute prompt:

```text
Between A and B, what is the dominant color of the one on the LEFT?
What is the dominant color of the main object or region shown?
```

This is a task-semantics adaptation to PACO's annotation, not a change to the
baseline/R4 method mechanics. Use the exact RealBind2 wording only with the
diagnostic `--curation-mode raw --color-semantics color` variant.

### 3. Border-touching boxes are not automatically invalid

1,529 raw test candidates have at least one box touching the image border. A
truncated object can still be a valid referent and a valid color question. The
curated builder does not exclude border-touching boxes; it records geometry and
area so they can be audited separately.

## Frozen curated rule used by `build_benchmark.py`

Quality mode:

```text
annotation A/B IoU < 0.1
max containment fraction < 0.5
bbox area ratio >= 0.005 for both objects
referent unique without using color
dominant-color question/readout semantics
```

Exclusion reasons on the test pool:

- annotation IoU ≥ 0.1: 290
- containment ≥ 0.5: 177
- bbox area < 0.5%: 487
- no unambiguous color-free referent: 92

These are sequential first-failure counts, so they are not additive after the
first matching rejection.

## Output files

```text
family_annotation_audit.jsonl   # every raw candidate with flags/metrics
family_annotation_audit.csv
annotation_audit_summary.json
ANNOTATION_AUDIT_REPORT.md
previews/
```

Curated benchmark:

```text
/root/Qwen3-VL/outputs/paco_realbind2_comparison/benchmark/
  families_all.jsonl
  questions_all.jsonl
  benchmark_all.jsonl
  curation_audit.jsonl
  split_spec.json
```

## Recommendation

Use the curated test benchmark for the Direct vs R4-only PACO comparison. Do not
revert to raw bare-category referents for the headline result. If a strict
single-color/single-template replication is required, treat it as a secondary
diagnostic; it has substantially fewer usable families and is not the main
external validation.
