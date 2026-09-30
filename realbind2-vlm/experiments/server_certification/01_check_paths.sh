#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
"$MAIN_PYTHON" scripts/verify_hashes.py --root "$ROOT"
"$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage check-paths
"$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage check-env --component main_vlm
"$MM_GDINO_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage check-env --component mm_gdino
"$SAM3_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage check-env --component sam3

