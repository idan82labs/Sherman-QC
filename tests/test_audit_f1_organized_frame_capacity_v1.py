from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_f1_organized_frame_capacity_v1.py"
SPEC = importlib.util.spec_from_file_location("organized_capacity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_absent_capability_and_protected_paths_are_fail_closed(tmp_path):
    (tmp_path / "scan.ply").write_bytes(b"ply")
    protected = tmp_path / "evaluator_only"
    protected.mkdir()
    (protected / "secret.npy").write_bytes(b"secret")
    result = MODULE.audit(tmp_path)
    assert result["status"] == "CAPABILITY_NOT_PRESENT"
    assert result["fused_ply_count"] == 1
    assert result["raw_sensor_candidate_count"] == 0
    assert result["protected_paths_skipped"] == 1


def test_requires_both_manifest_and_raw_observation(tmp_path):
    (tmp_path / "organized_frame_bundle_v1.json").write_text("{}", encoding="utf-8")
    assert MODULE.audit(tmp_path)["status"] == "CAPABILITY_NOT_PRESENT"
    (tmp_path / "depth.npy").write_bytes(b"raw")
    result = MODULE.audit(tmp_path)
    assert result["status"] == "READY_FOR_ORGANIZED_FRAME_VALIDATION"
    assert result["raw_sensor_candidate_count"] == 1

