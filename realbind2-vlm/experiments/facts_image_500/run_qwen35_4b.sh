#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES="${CASES:-$ROOT/cases.jsonl}"
IMAGES="${IMAGES:-$ROOT/images}"
if [[ ! -f "$CASES" ]]; then CASES="$ROOT/output/cohort_v1/cases.jsonl"; fi
if [[ ! -d "$IMAGES" ]]; then IMAGES="$ROOT/output/cohort_v1/images"; fi
PATHS="${PATHS:-$ROOT/configs/paths.yaml}"
CONFIG="${CONFIG:-$ROOT/configs/qwen35_4b.yaml}"
OUTPUT="${OUTPUT:-$ROOT/outputs/qwen35_4b/facts_image.jsonl}"
[[ -f "$PATHS" ]] || { echo "Missing $PATHS; copy configs/paths.example.yaml to configs/paths.yaml and fill model paths." >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
exec "${PYTHON_BIN:-python}" "$ROOT/scripts/run_facts_image.py" --config "$CONFIG" --paths "$PATHS" --cases "$CASES" --images-root "$IMAGES" --output "$OUTPUT" "$@"
