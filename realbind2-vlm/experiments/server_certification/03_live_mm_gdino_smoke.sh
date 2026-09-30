#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${MM_GDINO_SMOKE_GPU:-0}" "$MM_GDINO_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage mm-gdino

