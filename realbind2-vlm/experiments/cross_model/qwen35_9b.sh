#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT/experiments/cross_model/_run_baseline.sh" "$ROOT/configs/qwen35_9b.yaml" qwen35_9b

