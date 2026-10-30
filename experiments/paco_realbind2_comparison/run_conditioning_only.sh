#!/usr/bin/env bash
# Conditioning-only entry point, useful for checking IoU attrition before GPU
# Direct/R4 model runs. Fill MM_PYTHON with the pinned MM-GDINO environment.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
PATHS="${PATHS:-$HERE/configs/paths.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/Qwen3-VL/data/paco_lvis_candidates_test_val_4k}"
OUT_ROOT="${OUT_ROOT:-/root/Qwen3-VL/outputs/paco_realbind2_comparison}"
VLM_PYTHON="${VLM_PYTHON:-python3}"
CPU_PYTHON="${CPU_PYTHON:-python3}"
MM_PYTHON="${MM_PYTHON:-python3}"
SPLITS="${SPLITS:-test}"
BENCH="$OUT_ROOT/benchmark"
COND="$OUT_ROOT/conditioning"
mkdir -p "$BENCH" "$COND"
"$CPU_PYTHON" "$HERE/scripts/build_benchmark.py" --paths "$PATHS" --dataset-root "$DATASET_ROOT" --candidate-manifest "$DATASET_ROOT/manifests/candidate_families_test_val.jsonl" --image-checksums "$DATASET_ROOT/manifests/image_checksums.csv" --splits "$SPLITS" --curation-mode quality --color-semantics dominant --output-dir "$BENCH"
"$MM_PYTHON" "$HERE/scripts/stage_mm_gdino.py" --paths "$PATHS" --families "$BENCH/families_all.jsonl" --output-dir "$COND/mm_gdino"
"$VLM_PYTHON" "$HERE/scripts/stage_selector.py" --paths "$PATHS" --model-tag qwen35_4b --families "$BENCH/families_all.jsonl" --candidates "$COND/mm_gdino/candidates.jsonl" --output-dir "$COND/selector_primary"
"$CPU_PYTHON" "$HERE/scripts/finalize_conditioned_set.py" --paths "$PATHS" --families "$BENCH/families_all.jsonl" --questions "$BENCH/questions_all.jsonl" --selected-targets "$COND/selector_primary/selected_targets.jsonl" --output-dir "$OUT_ROOT/conditioned"
"$CPU_PYTHON" "$HERE/scripts/verify_conditioned_set.py" --dataset-dir "$OUT_ROOT/conditioned"
