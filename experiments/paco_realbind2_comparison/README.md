# PACO reproduction: RealBind2 Direct baseline vs R4-only

This directory is a self-contained runner for the **same comparison performed in
RealBind2**:

- **baseline:** Direct VLM call on the original full image with the PACO-adapted
  RealBind2-style REL2C question (dominant-color wording by default);
- **R4-only:** MM-GDINO-L candidate pool → target-model numbered-grid selector
  → selected bbox crop attribute read → deterministic bbox-center geometry →
  minimal two-line state → target-model text-only final answer;
- **single shared conditioned population:** `ref_A != ref_B` and
  **R4-predicted A/B bbox IoU strictly < 0.1**, selected with the primary
  Qwen3.5-4B selector before any multi-model run;
- paired model table: Direct and R4-only use the same `qa_id` set, same images,
  same questions, same semantic scorer, and family-cluster paired statistics.

No SAM3 isolation, no chain-of-thought, no answer hint, no target-side hint, and
no evaluator color or annotation box is passed to inference.

---

## 1. Files

```text
configs/paths.yaml                  # edit model paths before GPU run
configs/reference_hashes.json       # hashes of frozen RealBind2 implementation files
configs/paths.example.yaml

scripts/build_benchmark.py          # annotation-audited PACO candidate families/questions
scripts/audit_paco_annotations.py   # standalone annotation-quality audit and flags
scripts/stage_mm_gdino.py           # MM-GDINO-L candidate pools (pinned env)
scripts/stage_selector.py           # primary R4 selector for conditioning
scripts/finalize_conditioned_set.py # ref_A!=ref_B + predicted A/B IoU < 0.1
scripts/verify_conditioned_set.py    # independent CPU audit of the frozen condition set
scripts/run_direct.py               # Direct baseline for one model tag
scripts/run_r4_only.py              # R4-only for one model tag
scripts/aggregate_results.py        # paired multi-model Direct vs R4-only table
scripts/common.py                   # protocol imports/shared helpers

run_conditioning_only.sh            # build benchmark + selector + frozen set
run_paco_matrix.sh                  # full pipeline: conditioning + all models + table
```

The benchmark source is:

```text
/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k
```

The default final evaluation split is **PACO-LVIS test**. PACO-LVIS val is built
separately and must be used only for prompt/pipeline tuning unless you explicitly
pre-register a test+val external evaluation.

---

## 1.1 Annotation audit and benchmark curation

The raw PACO candidate pool is not used directly. The standalone audit
(`scripts/audit_paco_annotations.py`) found three material annotation risks in
the raw test candidates:

- many images contain repeated categories, so a bare referent such as “the chair”
  is ambiguous;
- many PACO objects carry multiple color attributes; `dom_color_ids` is the
  annotated dominant color, not the only visible color;
- a small but non-trivial subset has high annotation-box overlap or containment,
  including near-duplicate category pairs such as cup/mug and tray/plate.

`build_benchmark.py` therefore defaults to a frozen quality curation:

- annotation A/B IoU strictly below 0.1;
- maximum bbox containment fraction strictly below 0.5;
- each bbox area at least 0.5% of the image;
- A/B referents must be unique without using the queried color:
  bare category when unique, otherwise category plus a unique non-color
  attribute (material/pattern/transparency); ambiguous families are excluded;
- the question and R4 color-reader prompt use **dominant color** semantics,
  matching PACO's `dom_color_ids`. Use `--curation-mode raw --color-semantics color`
  only for a diagnostic exact-template variant, not for the main comparison.

Current benchmark counts after curation:

```text
test:      2,138 families / 4,276 side-specific QAs / 1,335 images
val:         571 families / 1,142 side-specific QAs /   346 images
raw test:  3,184 families / 6,368 side-specific QAs / 1,825 images
```

The raw candidate manifest remains unchanged; the curated manifest is written
by `build_benchmark.py`, and exclusions are recorded in
`curation_audit.jsonl`.

The full audit and contact sheets are generated at:

```text
/root/Qwen3-VL/outputs/paco_realbind2_comparison/annotation_audit/
  ANNOTATION_AUDIT_REPORT.md
  family_annotation_audit.jsonl
  annotation_audit_summary.json
  previews/top_annotation_overlap_16.jpg
  previews/top_annotation_containment_16.jpg
  previews/curated_referent_preview_16.jpg
```

A static copy of the audit report, summary, previews, and current benchmark
specs is included in this repository under `docs/annotation_audit/` and
`docs/benchmark_specs/`.

## 2. Conditioned set: exact rule

The conditioning stage does **not** use annotation boxes and does not use
answers. It joins the primary R4 selector outputs and retains a family iff:

```text
referent_A != referent_B
AND
IoU(R4_selected_bbox_A, R4_selected_bbox_B) < 0.1
```

The predicted boxes are the output of:

1. MM-GDINO-L per referent with `threshold=0.25`,
   `text_threshold=0.25`, native-score/raw-index order, top 16;
2. the target model numbered-crop selector with prompt:

```text
Which numbered crop best matches: {referent}? Answer only the number.
```

3. an invalid selector response falls back to candidate index 0 (native top1).

The annotation boxes are kept only in the family manifest for evaluation and
diagnostics. They are never inserted into a Direct or R4 prompt.

The conditioned set is frozen once and then reused for every model. Do not
re-condition separately using different target models; that would destroy the
paired comparison.

---

## 3. Model matrix

The script supports the same model tags/configs as the RealBind2 release:

| Tag | Model | Expected role |
|---|---|---|
| `qwen35_4b` | Qwen3.5-4B | primary + conditioning selector |
| `qwen35_9b` | Qwen3.5-9B | size / same family |
| `qwen3vl_4b` | Qwen3-VL-4B | Qwen3-VL family |
| `qwen3vl_8b` | Qwen3-VL-8B | Qwen3-VL scale |
| `internvl35_8b` | InternVL3.5-8B | different family |
| `llava_next_7b` | LLaVA-NeXT-7B | different family (optional) |

Model configs are read from the frozen RealBind2 release
`configs/*.yaml`; model paths are in your `configs/paths.yaml`. Do not inherit
historical numbers: every Direct/R4 row for this PACO benchmark must be newly
produced under the same frozen benchmark hash.

Minimum sensible multi-model matrix for a paper:

```text
qwen35_4b,qwen3vl_4b,internvl35_8b
```

Recommended:

```text
qwen35_4b,qwen3vl_4b,qwen3vl_8b,internvl35_8b
```

Add `qwen35_9b` and/or `llava_next_7b` only when their exact local checkpoints,
processors, and prompt runtimes are available.

---

## 4. GPU environment setup

Two different Python environments are intentional:

- `MM_PYTHON`: pinned MM-GDINO runtime, `transformers==4.55.4`;
- `VLM_PYTHON`: the model-adapter environment with the exact local model
  checkpoints and required packages.

Edit `configs/paths.yaml`, then:

```bash
export VLM_PYTHON=/path/to/vlm/env/bin/python
export MM_PYTHON=/path/to/mm_gdino/env/bin/python
export CPU_PYTHON=python3   # needs numpy/scipy; used for manifests, conditioning merge, aggregation
export MODEL_TAGS="qwen35_4b qwen3vl_4b qwen3vl_8b internvl35_8b"
```

The model adapters are imported from:

```text
/root/Qwen3-VL/repair-github/realbind2-final-delivery/src/models/
```

This is intentional: the PACO runner does not reimplement tokenizer rendering,
thinking control, text-only guards, generation settings, R4 protocol, or the
semantic scorer.

---

## 5. Run

### A. Conditioning only

```bash
cd /root/Qwen3-VL/experiments/paco_realbind2_comparison
VLM_PYTHON=/path/to/vlm/python \
MM_PYTHON=/path/to/mm/python \
bash run_conditioning_only.sh
```

Outputs under:

```text
/root/Qwen3-VL/outputs/paco_realbind2_comparison/
  benchmark/                         # curated test set (2,138 families / 4,276 QAs)
  conditioning/mm_gdino/             # candidate pools
  conditioning/selector_primary/     # primary R4 selected A/B boxes
  conditioned/                       # frozen IoU<0.1 population; selection_audit.jsonl is per QA, family_conditioning_audit.jsonl includes all candidate families and exclusion reasons
```

### B. Full model matrix

```bash
VLM_PYTHON=/path/to/vlm/python \
MM_PYTHON=/path/to/mm/python \
MODEL_TAGS="qwen35_4b qwen3vl_4b qwen3vl_8b internvl35_8b" \
bash run_paco_matrix.sh
```

### C. Aggregate only

```bash
python3 scripts/aggregate_results.py \
  --paths configs/paths.yaml \
  --dataset-dir /root/Qwen3-VL/outputs/paco_realbind2_comparison/conditioned \
  --model-tags qwen35_4b,qwen3vl_4b,qwen3vl_8b,internvl35_8b \
  --runs-root /root/Qwen3-VL/outputs/paco_realbind2_comparison/runs \
  --output-dir /root/Qwen3-VL/outputs/paco_realbind2_comparison/analysis
```

Main outputs:

```text
analysis/analysis.json
analysis/paired_diagnostics.csv
analysis/paper_table.md
```

The paired accuracy gain is **R4-only minus Direct**. The 95% CI resamples
families with a fixed seed, preserving the two side-specific QAs per family.
QA-level exact McNemar is reported only as descriptive.

---

## 6. Exact RealBind2 compatibility points

The following are imported or copied verbatim from the frozen RealBind2
release, not independently redesigned:


PACO-specific adaptation: because `dom_color_ids` is the PACO dominant-color
annotation and many objects have multiple color attributes, the default PACO
question and crop-attribute prompt say **dominant color**. The R4 structure and
all model/selector/geometry/scoring mechanics are unchanged. The exact
RealBind2 “what color” wording is available only as the diagnostic
`--curation-mode raw --color-semantics color` variant.

- question templates:
  `src/common/prompts.py`;
- semantic color scorer:
  `src/common/scorer.py` (`realbind2-semantic-v1`);
- R4 crop, geometry, minimal-state, final-prompt, and attribute parser:
  `src/repair/strong_integrated/protocol.py`;
- selector/grid fallback semantics:
  `scripts/server_live_certification.py`,
  `scripts/repair_component_worker.py`;
- paired family-cluster statistics:
  `src/repair/strong_integrated/statistics.py`.

R4-only semantics are the same as RealBind2:

```text
A/B bbox crops
-> COLOR_PROMPT on each crop
-> normalized center dx = (cx_A - cx_B) / image_width
-> absolute dx <= 1e-9 => geometry ambiguous
-> [TARGET_ENTITY] A/B/unknown
   [TARGET_ATTRIBUTE] color = <color>/unknown
-> question + "\n\nAcquired visual facts:\n" + state
-> text-only final call
```

Invalid answers remain in the denominator and count as wrong. No failed row is
silently dropped. A selector or attribute failure yields an unknown state but
still runs the final call, matching RealBind2.

For the primary model `qwen35_4b`, `run_r4_only.py` checks by default that the
fresh selector output exactly reproduces the boxes used to freeze the
conditioned set. If the determinism/identity check drifts, the run aborts rather
than silently evaluating a different population. Use
`--allow-primary-selection-drift` only for an explicitly documented debugging
run.

---

## 7. Output schema highlights

Direct rows contain:

```text
qa_id, family_id, qa_type, question, image_sha256,
model_id, method=direct, prediction_raw, prediction_normalized,
ground_truth, correct, valid, scorer_version, run_benchmark_sha256
```

R4-only rows additionally contain:

```text
selected A/B bboxes and selector statuses
bbox crop paths and hashes
attribute raw and normalized colors
geometry status and normalized center difference
minimal state and final text-only call
logical_target_correct, r4_target_grounded, attribute_correct,
geometry_valid, minimal_state_sufficient, clean_correct, compensating_correct
```

All inference stages write `identity.json`, `status.json`, and resumable
`results.jsonl`. Identity includes the frozen benchmark hash, candidate-pool
hash, model config/metadata, condition split spec hash, and code hash.

---

## 8. Current execution status

CPU-side construction and mock contract checks have been run; no GPU model or
MM-GDINO inference has been executed in this workspace because the local VLM
environment currently lacks Torch and several model paths in
`configs/paths.yaml` are placeholders. The scripts fail closed on missing paths,
identity mismatch, benchmark mismatch, or output coverage mismatch.

---

## 9. CPU contract check

The following check does not load a model. It builds a 24-family PACO slice,
synthesizes selector boxes from annotation boxes only to exercise the join, and
verifies that every `KEEP` decision is exactly equivalent to
`ref_A != ref_B and predicted IoU < 0.1`:

```bash
cd /root/Qwen3-VL/experiments/paco_realbind2_comparison
python3 tests/test_cpu_contracts.py
```

This test is plumbing-only and is **not** a result; the real conditioned set
must be produced by running `stage_mm_gdino.py` and `stage_selector.py` with the
configured models.
