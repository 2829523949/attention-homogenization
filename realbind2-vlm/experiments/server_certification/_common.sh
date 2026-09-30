#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATHS_CONFIG="${PATHS_CONFIG:-$ROOT/configs/server_paths.yaml}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python}"

test -f "$PATHS_CONFIG" || { echo "missing $PATHS_CONFIG" >&2; exit 2; }

yaml_value() {
  "$BOOTSTRAP_PYTHON" - "$PATHS_CONFIG" "$1" "$2" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg[sys.argv[2]][sys.argv[3]])
PY
}

MAIN_PYTHON="${MAIN_PYTHON:-$(yaml_value python main_vlm)}"
MM_GDINO_PYTHON="${MM_GDINO_PYTHON:-$(yaml_value python mm_gdino)}"
SAM3_PYTHON="${SAM3_PYTHON:-$(yaml_value python sam3)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(yaml_value output root)}"
export ROOT PATHS_CONFIG MAIN_PYTHON MM_GDINO_PYTHON SAM3_PYTHON OUTPUT_ROOT

