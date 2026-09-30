#!/usr/bin/env bash
set -euo pipefail
python scripts/verify_hashes.py --root .
echo "VIR_BLOCKED_EXECUTOR_UNCERTIFIED: dataset and historical DEFOCUS are certified; the dataset-scale VIR executor still requires certification." >&2
exit 2
