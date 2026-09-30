#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT/experiments/cross_model/_run_baseline.sh" "$ROOT/configs/qwen3vl_8b.yaml" qwen3vl_8b

