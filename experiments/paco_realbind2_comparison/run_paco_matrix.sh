#!/usr/bin/env bash
# End-to-end PACO RealBind2-protocol reproduction.
# Two Python environments are intentionally configurable:
#   VLM_PYTHON: model adapters / Direct / R4-only
#   MM_PYTHON: pinned transformers==4.55.4 environment for MM-GDINO-L
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
PATHS="${PATHS:-$HERE/configs/paths.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k}"
OUT_ROOT="${OUT_ROOT:-/root/Qwen3-VL/outputs/paco_realbind2_comparison}"
VLM_PYTHON="${VLM_PYTHON:-python3}"
CPU_PYTHON="${CPU_PYTHON:-python3}"
MM_PYTHON="${MM_PYTHON:-python3}"
MODEL_TAGS="${MODEL_TAGS:-qwen35_4b qwen3vl_4b qwen3vl_8b internvl35_8b}"
SPLITS="${SPLITS:-test}"
BENCH_DIR="$OUT_ROOT/benchmark"
CONDITIONING_DIR="$OUT_ROOT/conditioning"
RUNS_ROOT="$OUT_ROOT/runs"
CANDIDATES="$CONDITIONING_DIR/mm_gdino/candidates.jsonl"
SELECTED="$CONDITIONING_DIR/selector_primary/selected_targets.jsonl"
CONDITIONED="$OUT_ROOT/conditioned"

mkdir -p "$BENCH_DIR" "$CONDITIONING_DIR" "$RUNS_ROOT" "$CONDITIONED"

echo "[1/6] Build candidate benchmark"
"$CPU_PYTHON" "$HERE/scripts/build_benchmark.py" \
  --paths "$PATHS" --dataset-root "$DATASET_ROOT" \
  --candidate-manifest "$DATASET_ROOT/manifests/candidate_families_test_val.jsonl" \
  --image-checksums "$DATASET_ROOT/manifests/image_checksums.csv" \
  --splits "$SPLITS" --curation-mode quality --color-semantics dominant --output-dir "$BENCH_DIR"

echo "[2/6] MM-GDINO-L candidate pools (pinned MM environment)"
"$MM_PYTHON" "$HERE/scripts/stage_mm_gdino.py" \
  --paths "$PATHS" --families "$BENCH_DIR/families_all.jsonl" \
  --output-dir "$CONDITIONING_DIR/mm_gdino"

echo "[3/6] Primary R4 selector for conditioning"
"$VLM_PYTHON" "$HERE/scripts/stage_selector.py" \
  --paths "$PATHS" --model-tag qwen35_4b \
  --families "$BENCH_DIR/families_all.jsonl" \
  --candidates "$CANDIDATES" \
  --output-dir "$CONDITIONING_DIR/selector_primary"

echo "[4/6] Freeze ref_A!=ref_B and R4-predicted A/B IoU<0.1"
"$CPU_PYTHON" "$HERE/scripts/finalize_conditioned_set.py" \
  --paths "$PATHS" \
  --families "$BENCH_DIR/families_all.jsonl" \
  --questions "$BENCH_DIR/questions_all.jsonl" \
  --selected-targets "$SELECTED" \
  --output-dir "$CONDITIONED"
"$CPU_PYTHON" "$HERE/scripts/verify_conditioned_set.py" --dataset-dir "$CONDITIONED"

echo "[5/6] Direct and R4-only for each model on the same frozen set"
for tag in $MODEL_TAGS; do
  echo "  -> $tag Direct"
  "$VLM_PYTHON" "$HERE/scripts/run_direct.py" \
    --paths "$PATHS" --model-tag "$tag" --dataset-dir "$CONDITIONED" --images-root "$DATASET_ROOT" \
    --output-dir "$RUNS_ROOT/$tag/direct"
  echo "  -> $tag R4-only"
  "$VLM_PYTHON" "$HERE/scripts/run_r4_only.py" \
    --paths "$PATHS" --model-tag "$tag" --dataset-dir "$CONDITIONED" --images-root "$DATASET_ROOT" \
    --candidates "$CANDIDATES" \
    --output-dir "$RUNS_ROOT/$tag/r4_only"
done

echo "[6/6] Paired aggregate"
TAGS_CSV="$(echo $MODEL_TAGS | tr ' ' ',')"
"$CPU_PYTHON" "$HERE/scripts/aggregate_results.py" \
  --paths "$PATHS" --dataset-dir "$CONDITIONED" \
  --model-tags "$TAGS_CSV" --runs-root "$RUNS_ROOT" \
  --output-dir "$OUT_ROOT/analysis"
echo "Done: $OUT_ROOT/analysis/paper_table.md"
