#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES="${CASES:-$ROOT/cases.jsonl}"
IMAGES="${IMAGES:-$ROOT/images}"
if [[ ! -f "$CASES" ]]; then CASES="$ROOT/output/cohort_v1/cases.jsonl"; fi
if [[ ! -d "$IMAGES" ]]; then IMAGES="$ROOT/output/cohort_v1/images"; fi
python "$ROOT/scripts/run_facts_image.py" \
  --config "${CONFIG:-$ROOT/configs/qwen35_4b.yaml}" \
  --paths "${PATHS:-$ROOT/configs/paths.example.yaml}" \
  --cases "$CASES" --images-root "$IMAGES" \
  --output "${OUTPUT:-$ROOT/outputs/dry_run/results.jsonl}" --dry-run
