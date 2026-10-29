from pathlib import Path

from research.f1_zero_shot_runtime_integrity_v1 import validate


ROOT = Path(__file__).resolve().parents[1]


def test_repository_zero_shot_runtime_integrity_passes():
    result = validate(ROOT)

    assert result["status"] == "PASS"
    assert result["zero_shot_default"] is True
    assert result["blockers"] == []
    assert all(result["checks"].values())

