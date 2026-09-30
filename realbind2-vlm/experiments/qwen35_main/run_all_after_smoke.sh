#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SMOKE_MODE=live bash experiments/qwen35_main/00_smoke.sh
bash experiments/qwen35_main/01_baseline.sh
echo "Merge shards, audit exactly 9800 rows, then continue with 02_build_cohorts.sh per RUNBOOK.md."

