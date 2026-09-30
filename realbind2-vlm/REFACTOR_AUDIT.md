# Refactor audit

Audit date: 2026-09-30 CST. Source workspace: `/root/Qwen3-VL`.

## Workspace state

The parent repository is a 45GB long-running Qwen-VL research tree with an
existing Git history and a dirty worktree. Its `origin` points to the unrelated
public `attention-homogenization` repository. No existing output was edited or
deleted. The integrated repair development workspace was found separately at
`/mnt/d/vlm_repair_v2`; the RealBind2 construction root is
`/mnt/d/datasets/RealBind_v2`.

Relevant current structure:

- `realbind_v2/`, `build_realbind_v2.py`: metadata-first RealBind2 builder.
- `experiments/realbind_phase2/`: RealBind v1.1 baseline, VIR, causal, DaH, and
  routing studies.
- `experiments/qwen35_*` and matching `outputs/qwen35_*`: mechanism history.
- `/mnt/d/vlm_repair_v2/harness/`: repair development harness.
- `/mnt/d/vlm_repair_v2/outputs/region_bottleneck_joint_repair/`: corrected R4.
- `/mnt/d/vlm_repair_v2/outputs/strong_integrated_repair/`: completed 40-case
  Strong Integrated development audit.
- `/mnt/d/vlm_repair_v2/outputs/sam3_instance_hypothesis_binding/`: failed
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
`frozen/questions.jsonl` exists. Thus the claimed formal 1,400-image / 9,800-QA
benchmark cannot yet be independently verified or executed. The packaged
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

Historical code contains `/root/Qwen3-VL`, `/mnt/d/vlm_repair_v2`, and
`/mnt/d/datasets/RealBind`. Those copies are preserved under `legacy/` or
`src/repair/retrospective/` and are not official entrypoints. New code reads
`configs/paths.yaml` or explicit CLI paths. `paths.yaml` is ignored by Git.

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
2. RealBind2 VIR defocus semantics have no located frozen implementation/hash;
   the formal VIR entrypoint fails closed rather than inventing one.
3. The dataset-scale causal and retrospective runners cannot be certified
   before F0/F1 exist; their shell entrypoints fail closed.
4. A fresh unified live smoke spanning Qwen, MM-GDINO-L, SAM3, and self-patch
   has not been implemented against frozen RealBind2. Artifact-backed smoke is
   available and clearly labeled.

