from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from research.f1_ciip_v1 import execution_lock
from research.f1_ciip_v1 import native_worker


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_f1_ciip_v1_smoke.py"
SPEC = importlib.util.spec_from_file_location("ciip_correction_v2_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

V1_PATHS = {
    Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_20260809.json"),
    Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_EXECUTION_CLAIM_20260809.json"),
    Path("output/research/ciip-digital-occupied-shell-h0-v1/native-lock-v1"),
    Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-v1"),
    Path("tmp/ciip-digital-occupied-shell-h0-v1-smoke-v1"),
}
V2_PATHS = {
    Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_CORRECTION_V2_20260809.json"),
    Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_EXECUTION_CLAIM_CORRECTION_V2_20260809.json"),
    Path("output/research/ciip-digital-occupied-shell-h0-v1/native-lock-correction-v2"),
    Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v2"),
    Path("tmp/ciip-digital-occupied-shell-h0-v1-smoke-correction-v2"),
}
V3_PATHS = {
    Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_CORRECTION_V3_20260809.json"),
    Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_EXECUTION_CLAIM_CORRECTION_V3_20260809.json"),
    Path("output/research/ciip-digital-occupied-shell-h0-v1/native-lock-correction-v3"),
    Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v3"),
    Path("tmp/ciip-digital-occupied-shell-h0-v1-smoke-correction-v3"),
}


def test_correction_v4_attempt_namespace_is_exact_and_disjoint_from_history() -> None:
    current = {
        execution_lock.LOCK_RELATIVE,
        execution_lock.CLAIM_RELATIVE,
        execution_lock.NATIVE_BUILD_RELATIVE,
        execution_lock.OUTPUT_RELATIVE,
        execution_lock.SCRATCH_RELATIVE,
    }
    assert len(current) == 5
    assert current.isdisjoint(V1_PATHS | V2_PATHS | V3_PATHS)
    assert execution_lock.ATTEMPT_ID == "CORRECTION_V4"
    assert runner.LOCK_PATH == ROOT / execution_lock.LOCK_RELATIVE
    assert runner.CLAIM_PATH == ROOT / execution_lock.CLAIM_RELATIVE
    assert runner.NATIVE_BUILD_DIR == ROOT / execution_lock.NATIVE_BUILD_RELATIVE
    assert "FRESH_INDEPENDENT_SOURCE_SCHEMA_CORRECTED" in execution_lock.AUTHORIZATION_RECEIPT
    assert "PRESERVE_PRIOR_V1_AND_CORRECTION_V2_AND_CORRECTION_V3_CONSUMED_CLAIMS_LOCKS_AND_TERMINAL_STOPS" in execution_lock.AUTHORIZATION_SCOPE


def test_historical_v1_v2_v3_and_consumed_v4_evidence_remain_present() -> None:
    assert (ROOT / next(path for path in V1_PATHS if "PREEXECUTION_LOCK_20260809" in path.name)).is_file()
    assert (ROOT / next(path for path in V1_PATHS if "EXECUTION_CLAIM_20260809" in path.name)).is_file()
    assert (ROOT / Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-v1/result.json")).is_file()
    assert (ROOT / Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_CORRECTION_V2_20260809.json")).is_file()
    assert (ROOT / Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_EXECUTION_CLAIM_CORRECTION_V2_20260809.json")).is_file()
    assert (ROOT / Path("output/research/ciip-digital-occupied-shell-h0-v1/native-lock-correction-v2")).is_dir()
    assert (ROOT / Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v2/result.json")).is_file()
    assert (ROOT / next(path for path in V3_PATHS if "PREEXECUTION_LOCK_CORRECTION_V3" in path.name)).is_file()
    assert (ROOT / next(path for path in V3_PATHS if "EXECUTION_CLAIM_CORRECTION_V3" in path.name)).is_file()
    assert (ROOT / Path("output/research/ciip-digital-occupied-shell-h0-v1/native-lock-correction-v3")).is_dir()
    assert (ROOT / Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v3/result.json")).is_file()
    assert (ROOT / execution_lock.LOCK_RELATIVE).is_file()
    assert (ROOT / execution_lock.CLAIM_RELATIVE).is_file()
    assert (ROOT / execution_lock.NATIVE_BUILD_RELATIVE).is_dir()
    assert (ROOT / execution_lock.OUTPUT_RELATIVE / "result.json").is_file()
    assert not (ROOT / execution_lock.SCRATCH_RELATIVE).exists()


def test_prepare_lock_rejects_preexisting_v2_native_namespace_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "lock.json")
    monkeypatch.setattr(runner, "CLAIM_PATH", tmp_path / "claim.json")
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE", Path("output-v2"))
    monkeypatch.setattr(runner, "SCRATCH_RELATIVE", Path("scratch-v2"))
    native = tmp_path / "native-v2"
    native.mkdir()
    monkeypatch.setattr(runner, "NATIVE_BUILD_DIR", native)
    monkeypatch.setattr(
        native_worker,
        "build_native_worker",
        lambda _path: (_ for _ in ()).throw(AssertionError("compiler must not run")),
    )
    with pytest.raises(RuntimeError, match="native build path already exists"):
        runner.prepare_lock()


def test_resource_patch_rejects_non_supervisor_key_set_before_file_read(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.bin"
    with pytest.raises(RuntimeError, match="exact supervisor-owned set"):
        runner._patch_resource_metadata(missing, {"serialized_bytes": 1})
