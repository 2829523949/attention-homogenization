#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
test -f "${PATHS_CONFIG:-configs/paths.yaml}" || { echo "missing paths config" >&2; exit 2; }
python -m src.repair.strong_integrated.pipeline \
  --stage "${STRONG_STAGE:-audit-cache}" \
  --dataset-root "${REALBIND2_ROOT:?set REALBIND2_ROOT}" \
  --output-root "${STRONG_OUTPUT:-outputs/strong_integrated}"

