#!/usr/bin/env bash
set -euo pipefail

command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing" >&2; exit 2; }
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
test "${#GPU_ROWS[@]}" -ge 2 || { echo "expected at least two GPUs" >&2; exit 2; }
for row in "${GPU_ROWS[@]:0:2}"; do
  name="${row%,*}"
  memory="${row##*, }"
  [[ "$name" == *L20* ]] || { echo "expected L20, found $name" >&2; exit 2; }
  test "$memory" -ge 45000 || { echo "expected approximately 48GB, found ${memory}MiB" >&2; exit 2; }
done
printf '%s\n' "${GPU_ROWS[@]}"
echo GPU_CHECK_PASS

