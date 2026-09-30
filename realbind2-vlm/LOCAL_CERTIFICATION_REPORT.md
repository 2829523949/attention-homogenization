# Local certification report

Date: 2026-09-30 CST.

## Data

- Current construction snapshot: 1,400 `AUTO_VALID` families on 973 unique
  source images.
- QA reconstruction: 9,800 total; 2,800 composition; every family has exactly
  seven distinct, dependency-complete QA rows.
- Duplicate family, entity-pair, and QA IDs: zero.
- Human accepted: zero. Frozen: zero. Freeze events: zero.
- Decision: `REALBIND2_DATASET_BLOCKED` pending human audit/export.

## DEFOCUS

- Historical implementation and transform audit preserved under
  `legacy/defocus_history/`.
- Frozen implementation: `src/failure/defocus_frozen.py`.
- Five historical fixtures reproduce exact pixel arrays and exact PNG bytes.
- Decision: `DEFOCUS_CERTIFIED`.

## Server bundle

- Fixed six-family metadata-selected smoke manifest prepared.
- Qwen, MM-GDINO, SAM3, Strong Integrated, and causal self-patch live stages
  prepared with explicit success markers.
- Final aggregator emits `LIVE_SMOKE_PASS` only after all component passes.
- Deterministic two-GPU baseline, exact merge checks, independent stability
  run, and automatic cohort construction are prepared.
- GPU inference was not run on this local machine by design.

## Local checks

Local certification uses the existing `next3d_qwen` Python environment, since
the default base Python has no pytest installation. No model was loaded.

- `pytest`: 21 passed, 1 skipped (`live_gpu`)
- `python -m compileall`: pass
- shell syntax: pass
- repository hash verification: PASS, 18 protected files
- dataset schema audit: pass, 1,400/1,400
- DEFOCUS regression: pass, 5/5 exact pixels and bytes
- artifact-backed smoke: `SMOKE_PASS`

## Remaining limits

The live bundle is statically prepared, and all live model/API behavior remains
pending server execution. Dataset-scale VIR, causal, and retrospective runners
retain their fail-closed gates. No formal release hash is created while human
acceptance/export is incomplete. Runtime profiles will be generated only by
actual server execution.
