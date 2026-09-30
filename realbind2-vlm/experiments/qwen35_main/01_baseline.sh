#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
test -f "${PATHS_CONFIG:-configs/paths.yaml}" || { echo "missing paths config" >&2; exit 2; }
python scripts/run_baseline.py \
  --config configs/qwen35_4b.yaml \
  --paths "${PATHS_CONFIG:-configs/paths.yaml}" \
  --output "${BASELINE_OUTPUT:-outputs/qwen35_4b/baseline/shard_${SHARD_INDEX:-0}.jsonl}" \
  --num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}"

