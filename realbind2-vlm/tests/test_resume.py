from pathlib import Path

import pytest

from src.common.logging_utils import JsonlRunLog


def test_resume_skips_completed_and_rejects_overwrite(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = JsonlRunLog(path)
    log.append({"case_id": "a", "status": "ok"})
    resumed = JsonlRunLog(path)
    assert resumed.is_complete("a")
    with pytest.raises(ValueError):
        resumed.append({"case_id": "a", "status": "ok"})

