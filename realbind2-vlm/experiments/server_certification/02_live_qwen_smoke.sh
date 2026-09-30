#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${QWEN_SMOKE_GPU:-0}" "$MAIN_PYTHON" scripts/server_live_certification.py --paths "$PATHS_CONFIG" --stage qwen

