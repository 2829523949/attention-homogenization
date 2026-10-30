#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for wrapper in run_qwen35_4b.sh run_qwen35_9b.sh run_qwen3vl_4b.sh run_qwen3vl_8b.sh run_internvl35_8b.sh run_llava_next_7b.sh; do
  echo "=== $wrapper ==="
  bash "$ROOT/$wrapper"
done
