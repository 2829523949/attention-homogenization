#!/usr/bin/env python3
"""Historical smoke entrypoint.

The repaired smoke runner writes to ``outputs/certification_repair`` and must
not overwrite the frozen ``outputs/refactor`` artifacts.  Use that runner for
all new main-method smoke tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.scripts.run_certification_repair_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
