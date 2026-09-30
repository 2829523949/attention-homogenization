#!/usr/bin/env bash
set -euo pipefail
python scripts/verify_hashes.py --root .
echo "VIR_BLOCKED_DATASET_NOT_FROZEN: historical DEFOCUS is certified, but formal VIR requires a human-accepted frozen RealBind2 export." >&2
exit 2
