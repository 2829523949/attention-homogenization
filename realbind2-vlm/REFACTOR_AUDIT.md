# Refactor audit

> Update, 2026-09-30: the owner explicitly confirmed acceptance of the audited
> 1,400-family set. The original builder imported those decisions and froze
> 1,400 families / 973 images / 9,800 QA / 2,800 composition. Exact export and
> image-hash certification passed. See FREEZE_REPORT.md. Prior unfrozen
> findings below are retained as historical evidence and are superseded.


Audit date: 2026-09-30 CST. Source workspace paths are described by role so
local mount and account names are not published.

## Workspace state

The parent repository is a 45GB long-running Qwen-VL research tree with an
existing Git history and a dirty worktree. Its `origin` points to an unrelated
public repository. No existing output was edited or deleted. The integrated
repair development workspace and RealBind2 construction root were separate
local trees.

Relevant current structure:

- `realbind_v2/`, `build_realbind_v2.py`: metadata-first RealBind2 builder.
- `experiments/realbind_phase2/`: RealBind v1.1 baseline, VIR, causal, DaH, and
  routing studies.
- `experiments/qwen35_*` and matching `outputs/qwen35_*`: mechanism history.
- repair workspace `harness/`: repair development harness.
- repair workspace `outputs/region_bottleneck_joint_repair/`: corrected R4.
- repair workspace `outputs/strong_integrated_repair/`: completed 40-case
  Strong Integrated development audit.
- repair workspace `outputs/sam3_instance_hypothesis_binding/`: failed
  SAM3 direct-selection/consolidation branch.

## Entrypoints and active use

Official packaged entrypoints are the shell scripts in
`experiments/qwen35_main`, `experiments/cross_model`, and
`experiments/server_2xl20`. `scripts/run_baseline.py`, `merge_shards.py`,
`build_cohorts.py`, `audit_dataset.py`, `audit_outputs.py`,
`summarize_results.py`, and `verify_hashes.py` are shared infrastructure.

The final repair code is only `src/repair/strong_integrated/`. Historical
runners are snapshots and are not called by the official pipeline.

## RealBind2 integrity finding

The construction database has 1,400 `AUTO_VALID` families, 11,902 remaining
`METADATA_VALID` candidates, 973 unique accepted source images, zero
`HUMAN_ACCEPTED`, and zero `FROZEN` families. No `frozen/families.jsonl` or
`frozen/questions.jsonl` exists. The 1,400-family snapshot is locally certified
at 9,800 QA; release membership and execution remain blocked. The packaged
manifest records the expected protocol and all current family/image IDs, marks
`formal_run_allowed=false`, and does not fabricate split IDs or QA rows.

## Duplicated implementations

- Dataset question/scorer logic exists in the builder and RealBind v1.1
  baselines. The unified layer freezes the RealBind2 seven templates and a
  deterministic semantic scorer while retaining originals.
- Qwen loading/generation is repeated across historical scripts. The new model
  adapter removes repetition for baseline/final runs; causal hooks remain
  model-specific.
- Region acquisition and color reading have multiple research variants in the
  repair harness. They remain historical; only R4 + Qwen selector, SAM3
  isolation, and crop-Qwen belong to the final route.
- JSONL resume logic was repeated across runners. New runs use one fsync-backed
  logger and duplicate guard.

## Hard-coded paths

Historical provenance used machine-specific paths. Packaged runnable copies
use configuration, explicit CLI paths, or environment variables. Original
snapshots under `legacy/` are not official entrypoints. `paths.yaml` is
ignored by Git.

## Required dependency boundaries

- Main VLM: Python 3.10, torch 2.7.0+cu128, Transformers 5.8.1.
- Corrected MM-GDINO-L: isolated Transformers 4.55.4. A 5.8.1 run produced
  invalid boxes and must remain excluded.
- SAM3: isolated environment/checkpoint with recorded hash.
- Qwen3.5 uses `use_cache=false` because historical cached and uncached
  generation diverged under the validated runtime.

## Byte-preserved evidence

No source output directory was touched. Compact copied evidence is under
`legacy/evidence/`; its source remains canonical. The following source roots
must stay byte-for-byte unchanged: the complete RealBind v1.1 baseline,
visual-interference pilot/mechanism, DaH oracle results, corrected R4 outputs,
Strong Integrated outputs, SAM3 consolidation outputs, and all mechanism
history referenced by the parent `AGENTS.md`.

## Unsafe or unknown classification

The old certification/proof-state harness and earlier automatic routes are not
equivalent to Strong Integrated. Position-ID, index, recurrent/GDN, KV,
KEY-refine, K-rescuable, LBI/OCBW, and numerous synthetic binding experiments
remain `UNKNOWN_LEGACY` for packaging purposes. They were not deleted or
algorithmically modified.

## Blocking gaps found

1. Human review and frozen RealBind2 export are incomplete.
2. Historical VIR/DaH DEFOCUS semantics are recovered and regression
   certified; formal VIR remains blocked only by the missing frozen dataset.
3. The dataset-scale causal and retrospective runners cannot be certified
   before F0/F1 exist; their shell entrypoints fail closed.
4. A fresh unified live smoke spanning Qwen, MM-GDINO-L, SAM3, and self-patch
   is prepared for the server but has not been run locally by design.
