#!/usr/bin/env bash
set -euo pipefail
CONFIG="$1"
MODEL_TAG="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python scripts/run_baseline.py \
  --config "$CONFIG" --paths "${PATHS_CONFIG:-configs/paths.yaml}" \
  --output "outputs/${MODEL_TAG}/baseline/shard_${SHARD_INDEX:-0}.jsonl" \
  --num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}"

