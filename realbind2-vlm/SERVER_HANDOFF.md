# Server handoff

This is the primary operator guide for the 2 × NVIDIA L20 server. The current
dataset is not frozen; complete the human audit/export before step 6. Do not
substitute the current `AUTO_VALID` database for a frozen export.

## 1. Clone

```bash
git clone <private-repository-url> realbind2-vlm
cd realbind2-vlm
```

## 2. Create environments

Create separate environments from `envs/`. The required Transformers pins are:

- main VLM: `transformers==5.8.1`
- MM-GDINO-L: `transformers==4.55.4`
- SAM3: `transformers==5.8.1`

Do not upgrade the MM-GDINO environment to Transformers 5.8.1. Put the three
Python interpreter paths in the server path configuration.

```bash
python3.10 -m venv .venvs/main_vlm
python3.10 -m venv .venvs/mm_gdino
python3.10 -m venv .venvs/sam3
.venvs/main_vlm/bin/pip install -r envs/main_vlm/requirements.txt
.venvs/mm_gdino/bin/pip install -r envs/mm_gdino/requirements.txt
.venvs/sam3/bin/pip install -r envs/sam3/requirements.txt
```

## 3. Place models and data

Install local model checkpoints and the builder-produced frozen RealBind2
export outside the repository. The RealBind2 root must contain:

```text
frozen/families.jsonl
frozen/questions.jsonl
frozen/construction_stats.json
frozen/<paths named by family image_path>
```

The release manifest must be certified and committed with `frozen=true` before
live certification. This checkout deliberately has `frozen=false`; changing
that flag alone is insufficient. Follow `DATASET_FREEZE_BLOCKERS.md`.

## 4. Configure paths

```bash
cp configs/server_paths.example.yaml configs/server_paths.yaml
$EDITOR configs/server_paths.yaml
```

`configs/server_paths.yaml` is ignored by Git.

## 5. Verify repository hashes

```bash
python scripts/verify_hashes.py --root .
```

Expected: `HASH_PASS`.

## 6. Run live certification

```bash
set -o pipefail
PATHS_CONFIG=configs/server_paths.yaml BOOTSTRAP_PYTHON=.venvs/main_vlm/bin/python \
  bash experiments/server_certification/07_certify_all.sh \
  2>&1 | tee server_certification.log
```

This runs fixed metadata-selected cases and records runtime and peak allocated
GPU memory in `outputs/server_certification/runtime_profile.json`.

## 7. Stop unless certification passes

The final line must be:

```text
LIVE_SMOKE_PASS
```

Any stage failure prints `LIVE_SMOKE_FAIL` and exits nonzero. Do not start a
formal run after failure.

## 8. Run the first Qwen3.5-4B baseline

Run these in separate terminals:

```bash
PATHS_CONFIG=configs/server_paths.yaml bash experiments/server_2xl20/gpu0_baseline_shard0.sh
PATHS_CONFIG=configs/server_paths.yaml bash experiments/server_2xl20/gpu1_baseline_shard1.sh
```

Each family is assigned by `family_index % 2`; all seven QA rows from a family
stay on the same shard. Both scripts are resumable.

## 9. Merge and verify exact coverage

```bash
PATHS_CONFIG=configs/server_paths.yaml bash experiments/server_2xl20/merge_baseline.sh
```

Expected: `MERGE_PASS rows=9800 duplicates=0 missing=0 extra=0` if the frozen
release retains 1,400 complete families. The merge derives its expected IDs
from the export and fails on any duplicate, missing, or extra row.

## 10. Run the independent stability baseline and build cohorts

Run both stability shards, then merge:

```bash
PATHS_CONFIG=configs/server_paths.yaml bash experiments/server_2xl20/gpu0_baseline_stability_shard0.sh
PATHS_CONFIG=configs/server_paths.yaml bash experiments/server_2xl20/gpu1_baseline_stability_shard1.sh
PATHS_CONFIG=configs/server_paths.yaml bash experiments/server_2xl20/merge_stability_baseline.sh
python scripts/build_realbind2_cohorts.py --paths configs/server_paths.yaml
```

This writes F0, F1, stable F1, VIR, causal, and retrospective cohorts. Stable
F1 requires exact agreement across both complete runs and all prerequisites.
Matching is deterministic and without replacement. Do not edit cohort JSONL.

## 11. Run Qwen3.5 formal experiments

Follow `RUNBOOK.md` after the cohort summary has passed. The formal order is
baseline, F0/F1, VIR, causal mechanism, repair retrospective, then frozen
Strong Integrated evaluation. The repository intentionally blocks formal VIR
while the dataset is not frozen. Dataset-scale VIR, causal, and retrospective
execution paths still require certification; their existing fail-closed
entrypoints are listed in `RUNBOOK.md`. They must not be bypassed.

## 12. Run cross-model evaluation

Start `experiments/cross_model/` only after Qwen3.5 certification and formal
main-model outputs are complete. Reuse the frozen Strong Integrated states;
do not rerun MM-GDINO or SAM3 for each final VLM.
