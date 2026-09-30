#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python scripts/verify_hashes.py --root "$ROOT"
python scripts/smoke.py --mode "${SMOKE_MODE:-artifact}" --paths "${PATHS_CONFIG:-configs/paths.yaml}"

