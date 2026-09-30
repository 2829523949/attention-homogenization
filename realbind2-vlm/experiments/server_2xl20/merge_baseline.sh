#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../server_certification/_common.sh"
cd "$ROOT"
REALBIND2_ROOT="$(yaml_value data realbind2)"
"$MAIN_PYTHON" scripts/merge_baseline.py \
  --dataset-root "$REALBIND2_ROOT" \
  --inputs \
    "$OUTPUT_ROOT/qwen35_4b/baseline/shard_0.jsonl" \
    "$OUTPUT_ROOT/qwen35_4b/baseline/shard_1.jsonl" \
  --output "$OUTPUT_ROOT/qwen35_4b/baseline/results.jsonl"

