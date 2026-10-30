#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="${1:?usage: bash summarize_one_model.sh <model_key> [results.jsonl]}"
CASES="${CASES:-$ROOT/cases.jsonl}"
if [[ ! -f "$CASES" ]]; then CASES="$ROOT/output/cohort_v1/cases.jsonl"; fi
RESULTS="${2:-$ROOT/outputs/$MODEL_KEY/facts_image.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/$MODEL_KEY}"
[[ -f "$RESULTS" ]] || { echo "Missing results: $RESULTS" >&2; exit 2; }
python "$ROOT/scripts/summarize_facts_image.py" --results "$RESULTS" --cases "$CASES" --output-dir "$OUT_DIR" "${@:3}"
