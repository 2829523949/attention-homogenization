#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${SAM3_SMOKE_GPU:-0}" "$SAM3_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage sam3

