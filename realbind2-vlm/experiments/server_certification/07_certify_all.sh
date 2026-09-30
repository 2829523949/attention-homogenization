#!/usr/bin/env bash
set -euo pipefail
trap 'echo LIVE_SMOKE_FAIL >&2' ERR
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
bash experiments/server_certification/00_check_gpu.sh
bash experiments/server_certification/01_check_paths.sh
bash experiments/server_certification/02_live_qwen_smoke.sh
bash experiments/server_certification/03_live_mm_gdino_smoke.sh
bash experiments/server_certification/04_live_sam3_smoke.sh
bash experiments/server_certification/05_live_strong_integrated_smoke.sh
bash experiments/server_certification/06_live_causal_selfpatch.sh
source experiments/server_certification/_common.sh
"$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage certify-receipt
echo LIVE_SMOKE_PASS
