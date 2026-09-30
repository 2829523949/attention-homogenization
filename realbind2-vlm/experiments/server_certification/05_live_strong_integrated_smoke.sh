#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
gpu="${STRONG_SMOKE_GPU:-0}"
CUDA_VISIBLE_DEVICES="$gpu" "$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage strong-select
CUDA_VISIBLE_DEVICES="$gpu" "$SAM3_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage strong-mask
CUDA_VISIBLE_DEVICES="$gpu" "$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage strong-finish

