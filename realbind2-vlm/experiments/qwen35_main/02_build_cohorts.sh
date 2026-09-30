#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python scripts/build_cohorts.py \
  --baseline "${BASELINE_MERGED:-outputs/qwen35_4b/baseline/results.jsonl}" \
  --output-dir outputs/qwen35_4b/cohorts

