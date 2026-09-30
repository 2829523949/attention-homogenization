# Runbook

All commands start at the repository root. Formal runs use two independent
L20 jobs; do not enable tensor parallelism. Never continue past a failed gate.

## 1. Environment

Create three environments. Keep MM-GDINO isolated because its corrected run
requires Transformers 4.55.4.

```bash
conda env create -f environment.yml
conda create -n realbind2-gdino python=3.10 -y
conda run -n realbind2-gdino pip install -r envs/mm_gdino/requirements.txt
conda create -n realbind2-sam3 python=3.10 -y
conda run -n realbind2-sam3 pip install -r envs/sam3/requirements.txt
cp configs/paths.example.yaml configs/paths.yaml
```

Expected artifact: local `configs/paths.yaml` with all data/model paths. PASS:
each referenced path exists and `nvidia-smi` shows the intended devices.
Failure: correct paths or environments; do not edit model code or preprocessing.

## 2. Verify hashes and dataset freeze

```bash
python scripts/verify_hashes.py
python scripts/audit_dataset.py --data-root /path/from/config/to/RealBind_v2
```

Expected: `HASH_PASS`; dataset audit with families/images=1,400, QA=9,800,
composition=2,800, no duplicate QA IDs, and no missing images. The current
snapshot fails here because human review/export is incomplete. Complete the
existing builder audit/import/export flow, then regenerate
`data/realbind2_manifest.json` and `data/HASHES.json`. Do not export
`AUTO_VALID` rows directly.

## 3. Smoke

Packaging/artifact replay:

```bash
bash experiments/qwen35_main/00_smoke.sh
```

Formal live gate:

```bash
SMOKE_MODE=live PATHS_CONFIG=configs/paths.yaml \
  bash experiments/qwen35_main/00_smoke.sh
```

Expected artifact: no benchmark rows; terminal line `SMOKE_PASS`. PASS for a
formal run requires live mode to exercise one dataset/image/prompt/scorer/log
case, resume, one Strong Integrated case, one SAM3 isolation, one MM-GDINO-L
case, and one causal self-patch. Artifact mode validates completed historical
evidence and contracts only. At packaging time the live executor is blocked
and intentionally exits nonzero.

## 4. Qwen3.5-4B baseline

Single GPU:

```bash
PATHS_CONFIG=configs/paths.yaml bash experiments/qwen35_main/01_baseline.sh
```

Two deterministic shards:

```bash
bash experiments/server_2xl20/phase1_baseline_gpu0.sh
bash experiments/server_2xl20/phase1_baseline_gpu1.sh
```

Expected: 4,900 successful rows per shard; failed rows in sibling
`*.failed.jsonl`. PASS: total successful+failed coverage is 9,800 and failed is
zero before merge. Resume: rerun the same command; successful case IDs are
skipped.

## 5. Merge and audit

```bash
python scripts/merge_shards.py \
  --inputs outputs/qwen35_4b/baseline/shard_0.jsonl \
           outputs/qwen35_4b/baseline/shard_1.jsonl \
  --output outputs/qwen35_4b/baseline/results.jsonl
python scripts/audit_outputs.py outputs/qwen35_4b/baseline/results.jsonl --expected 9800
python scripts/summarize_results.py --kind baseline \
  --input outputs/qwen35_4b/baseline/results.jsonl \
  --output outputs/qwen35_4b/baseline/metrics.json
```

PASS: exactly 9,800 unique rows, no missing required fields, and metrics include
atomic, relation, composition, per-type accuracy, and invalid rates. Merge is
atomic at the file level; delete no shard while auditing.

## 6. Build F0/F1

```bash
bash experiments/qwen35_main/02_build_cohorts.sh
```

Expected: `outputs/qwen35_4b/cohorts/f0.jsonl` and `f1.jsonl`. F0 requires all
declared prerequisites correct and composition correct; F1 requires the same
prerequisites and composition wrong. PASS: no duplicate QA, every F1 row has
all four side-specific prerequisites correct. Resume: deterministic rebuild is
safe after verifying the baseline hash.

## 7. VIR

```bash
bash experiments/qwen35_main/03_vir.sh
```

Expected after certification: one condition-level JSONL row for every one of
2,800 composition cases under facts-only, image+correct-facts, and
defocus+correct-facts; original comes from baseline and is never regenerated.
PASS: paired injected-text hashes match, 8,400 new rows are unique, and summary
reports VIR, fact rescue, image-induced regression, competitor-attribute,
third-value, and invalid/error rates. Historical DEFOCUS is now recovered,
hashed, and exact-regression certified; see `DEFOCUS_CERTIFICATION.md`.
This formal step remains blocked by the missing human-accepted frozen dataset
and the uncertified dataset-scale VIR execution path.

## 8. Qwen3.5 causal mechanism

```bash
bash experiments/qwen35_main/04_causal.sh
```

Expected after cohort availability: at most 100 stable F1 without replacement,
matched F0 where required, clean donor/recipient traces, forward restoration,
reverse corruption, full layer sweep, self-patch controls, per-case logs, and
family-bootstrap CIs. If stable F1 is below 100, use all eligible rows and log
actual N. PASS: self-patch preservation is exact and all registered cases have
complete forward/reverse/layer keys. Resume from per-case checkpoints.

## 9. Historical repair retrospective

```bash
bash experiments/qwen35_main/05_repair_retrospective.sh
```

Expected: frozen subset of up to 100 stable F1 plus equal matched F0. Headline
methods are Original, Blur/Defocus, Decompose, Handoff+TargetLock, Full DaH,
R4, and Strong Integrated. Context binder, P0/P1/P2/P3, and SAM3 consolidation
use only a fixed 40–50 case diagnostic subset and are labeled development
branches. PASS: no GT bbox enters non-oracle methods and SAM3 consolidation is
never listed as final repair. Resume by method/case key.

## 10. Strong Integrated

Build and freeze entity caches once using the corrected MM-GDINO environment,
Qwen selector, SAM3 isolation environment, and crop-Qwen reader. Required
files under `outputs/strong_integrated/` are `candidates.jsonl`,
`selected_targets.jsonl`, `masks/`, `attributes.jsonl`,
`minimal_states.jsonl`, and `audit.jsonl`.

```bash
REALBIND2_ROOT=/path/to/RealBind_v2 \
STRONG_OUTPUT=outputs/strong_integrated STRONG_STAGE=audit-cache \
  bash experiments/qwen35_main/06_strong_integrated.sh

python -m src.repair.strong_integrated.pipeline --stage assemble \
  --dataset-root /path/to/RealBind_v2 \
  --output-root outputs/strong_integrated

python -m src.repair.strong_integrated.pipeline --stage final \
  --dataset-root /path/to/RealBind_v2 \
  --output-root outputs/strong_integrated \
  --model-key qwen35 --model-path /path/to/Qwen3.5-4B
```

PASS: every selected provider is frozen R4, every attribute reader is
crop-Qwen, SAM3 is only isolation/fallback, no evaluator key appears, and the
final reasoner records `image_present=false`. All later final VLMs reuse the
same `minimal_states.jsonl`; do not rerun MM-GDINO-L or SAM3 per VLM.

## 11. Summary and audit

```bash
python scripts/audit_outputs.py outputs/strong_integrated/final_qwen35.jsonl --expected 2800
python scripts/verify_hashes.py
pytest
```

PASS: all output keys are complete, hashes still match, and tests pass.

## 12. Cross-model evaluation

Run only after Qwen3.5 certification is complete.

```bash
bash experiments/cross_model/qwen35_9b.sh
bash experiments/cross_model/qwen3vl_4b.sh
bash experiments/cross_model/qwen3vl_8b.sh
bash experiments/cross_model/internvl35_8b.sh
bash experiments/cross_model/llava_next_7b.sh
```

Use the two phase-3 server scripts to split models across L20s. Qwen3-VL-4B
and LLaVA reduced causal replication uses only configured normalized depths;
never copy the 32-layer Qwen3.5 sweep. Cross-model runs may consume the frozen
minimal state but cannot change grounding, isolation, reader, prompt, scorer,
or repair policy.
