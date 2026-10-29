from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_f1_ciip_v1_smoke.py"
SPEC = importlib.util.spec_from_file_location("ciip_smoke_runner_routing_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_baseline_container_is_routed_only_to_independent_validator() -> None:
    baseline = Path("/source-free/BASE_A/container.bin")

    assert runner._baseline_container_for_role("BASE_A", None) is None
    assert runner._baseline_container_for_role("BASE_B", baseline) is None
    assert runner._baseline_container_for_role("INPUT_ORDER_PERMUTED", baseline) is None
    assert runner._baseline_container_for_role("INDEPENDENT_VALIDATOR", baseline) == baseline


def test_independent_validator_fails_closed_without_baseline_container() -> None:
    with pytest.raises(RuntimeError, match="independent validator has no baseline container"):
        runner._baseline_container_for_role("INDEPENDENT_VALIDATOR", None)
