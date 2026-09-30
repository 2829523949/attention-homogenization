#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../server_certification/_common.sh"
cd "$ROOT"
"$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage require-receipt
mkdir -p "$OUTPUT_ROOT/qwen35_4b/baseline/stability_shards"
CUDA_VISIBLE_DEVICES=1 "$MAIN_PYTHON" scripts/run_baseline.py \
  --config configs/qwen35_4b.yaml \
  --paths "$PATHS_CONFIG" \
  --output "$OUTPUT_ROOT/qwen35_4b/baseline/stability_shards/shard_1.jsonl" \
  --num-shards 2 --shard-index 1 --shard-by family
