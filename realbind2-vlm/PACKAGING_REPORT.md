# Packaging report

Generated: 2026-09-30 CST.

## Repository

- Parent workspace: `/root/Qwen3-VL`
- Packaged project root: `/root/Qwen3-VL/realbind2-vlm`
- Git branch: `realbind2-packaging`
- Packaging commit: `9c8474c` (parent branch; this report is included)
- Standalone publication branch: `realbind2-release`
- Existing parent remote: `git@github.com:2829523949/attention-homogenization.git` (unrelated; not used for publishing)
- GitHub private remote: not created; `gh` is not installed in this environment
- Publication status: `GITHUB_PUSH_BLOCKED_AUTH`

## Data integrity

- Protocol target: 1,400 benchmark image items, 9,800 QA, 2,800 composition QA.
- Construction snapshot: 1,400 `AUTO_VALID` families over 973 unique source
  image IDs; 0 `HUMAN_ACCEPTED`, 0 `FROZEN`.
- Frozen dataset files: absent.
- Formal run allowed: no.
- Prompt/scorer/config integrity: `data/HASHES.json`, 12 protected files,
  verification PASS.
- Current family and source image IDs: recorded in
  `data/realbind2_manifest.json`.

## Experiment entrypoints

Qwen3.5 main:

```bash
bash experiments/qwen35_main/00_smoke.sh
bash experiments/qwen35_main/01_baseline.sh
bash experiments/qwen35_main/02_build_cohorts.sh
bash experiments/qwen35_main/03_vir.sh
bash experiments/qwen35_main/04_causal.sh
bash experiments/qwen35_main/05_repair_retrospective.sh
bash experiments/qwen35_main/06_strong_integrated.sh
```

Cross-model:

```bash
bash experiments/cross_model/qwen35_9b.sh
bash experiments/cross_model/qwen3vl_4b.sh
bash experiments/cross_model/qwen3vl_8b.sh
bash experiments/cross_model/internvl35_8b.sh
bash experiments/cross_model/llava_next_7b.sh
```

The VIR, causal, and retrospective entries currently fail closed at their
uncertified prerequisites rather than silently changing protocol semantics.

## Repair

`FINAL_REPAIR = Strong Integrated`.

Frozen route: minimal dependency decomposition → persistent A/B grounding →
R4 MM-GDINO-L candidate pool + Qwen selector → deterministic bbox-center
target resolution → SAM3 target isolation with frozen fallback → crop-Qwen
attribute acquisition → minimal state → image exit → text-only final reasoner.

`SAM3 consolidation != final repair`. The direct SAM3 candidate-selection and
instance-hypothesis consolidation code is labeled
`HISTORICAL_FAILED_BRANCH` under `src/repair/retrospective/`.

## Tests

- Default pytest: 13 passed.
- Preserved historical harness: 73 passed in its original workspace.
- Artifact-backed `bash experiments/qwen35_main/00_smoke.sh`: `SMOKE_PASS`.
- Live smoke: blocked because frozen RealBind2 and configured local model paths
  are absent; artifact replay is not described as fresh model execution.
- Hash verification: PASS, 12 files.
- Leakage tests: PASS (selector GT, attribute composition GT, and final-image
  rejection).
- Self-patch test: PASS (strict preservation).
- Shell syntax and Python compile checks: PASS.

## Historical preservation

No historical source or output was deleted or overwritten. Copied source maps:

- `/mnt/d/vlm_repair_v2/harness/` → `legacy/vlm_repair_v2/harness/`
- `/mnt/d/vlm_repair_v2/docs/` → `legacy/vlm_repair_v2/docs/`
- `/mnt/d/vlm_repair_v2/manifests/` → `legacy/vlm_repair_v2/manifests/`
- RealBind2 builder → `legacy/realbind_v2_builder/`
- DaH oracle evaluator → `src/repair/dah/qwen35_oracle_retrospective.py`
- R4/Strong Integrated 40-case runners → `src/repair/retrospective/`
- failed SAM3 consolidation → `src/repair/retrospective/sam3_consolidation_failed.py`
- compact byte copies of key reports → `legacy/evidence/`

Large output roots remain at their original paths and are excluded from Git.
Older mechanism branches that cannot be safely classified are retained in the
parent workspace as `UNKNOWN_LEGACY`.

## Known limitations

1. Human review/frozen RealBind2 export is incomplete; actual 9,800-row
   integrity and formal splits cannot yet be verified.
2. The current 1,400 family snapshot uses 973 unique source images, which does
   not match a literal 1,400 unique-image claim.
3. The requested RealBind2 VIR defocus condition has no located frozen
   implementation hash. The entrypoint is blocked to prevent an invented
   semantic change.
4. Dataset-scale causal and retrospective execution awaits frozen baseline
   F0/F1 cohorts.
5. The fresh unified live smoke across Qwen, MM-GDINO-L, SAM3, and self-patch
   is not certified; only historical artifact replay and component contracts
   pass.
6. Historical source snapshots retain original absolute paths and are not
   official runnable entrypoints.
