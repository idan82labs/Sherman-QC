from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_f1_ciip_v1_smoke.py"
SPEC = importlib.util.spec_from_file_location("ciip_smoke_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_independent_worker_process_import_graph_is_semantically_clean() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--probe-independent-import-graph"],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "forbidden_imports": [],
        "status": "INDEPENDENT_IMPORT_GRAPH_CLEAN",
    }


def test_independent_import_guard_stops_preexisting_or_midflight_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = "research.f1_ciip_v1.native_worker"
    monkeypatch.setitem(sys.modules, forbidden, object())
    with pytest.raises(RuntimeError, match="independent import graph contaminated"):
        runner._assert_independent_import_graph_clean("regression-probe")


def test_execution_claim_is_one_shot_and_capability_is_role_path_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = tmp_path / "claim.json"
    monkeypatch.setattr(runner, "CLAIM_PATH", claim)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SCRATCH_RELATIVE", Path("scratch"))
    role_dir = tmp_path / "scratch" / "BASE_A"
    role_dir.mkdir(parents=True)
    lock = {"execution_lock_sha256": "ab" * 32}
    capability = runner._create_execution_claim(lock)
    payload = json.loads(claim.read_text(encoding="ascii"))
    assert payload["claim_state"] == "CONSUMED_BEFORE_FIRST_WORKER_LAUNCH"
    with pytest.raises(FileExistsError):
        runner._create_execution_claim(lock)

    monkeypatch.setenv("CIIP_EXECUTION_CAPABILITY", capability)
    runner._validate_worker_capability(lock, "BASE_A", role_dir)
    assert "CIIP_EXECUTION_CAPABILITY" not in os.environ
    monkeypatch.setenv("CIIP_EXECUTION_CAPABILITY", capability)
    with pytest.raises(RuntimeError, match="escaped exact one-shot scratch"):
        runner._validate_worker_capability(lock, "BASE_A", tmp_path / "elsewhere")


def test_process_tree_sample_includes_group_members_and_detects_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        stdout = "\n".join((
            "100 1 100 10",
            "101 100 100 20",
            "102 101 999 30",
            "103 1 100 40",
        ))

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())
    rss, live, escaped, root_seen = runner._process_tree_sample(100, 100)
    assert live == (100, 101, 102, 103)
    assert rss == (10 + 20 + 30 + 40) * 1024
    assert escaped is True
    assert root_seen is True


def test_supervisor_pregeometry_gates_precede_live_source_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lock = {"execution_lock_sha256": "ab" * 32}
    monkeypatch.setattr(runner, "load_execution_lock", lambda _path: lock)
    monkeypatch.setattr(
        runner,
        "_validate_pregeometry_gates",
        lambda value: events.append("gates") if value is lock else None,
    )

    class StopAfterLiveCheck(RuntimeError):
        pass

    def live(value: object) -> object:
        assert value is lock
        events.append("live_source_hash")
        raise StopAfterLiveCheck

    monkeypatch.setattr(runner, "_validate_live_lock", live)
    with pytest.raises(StopAfterLiveCheck):
        runner.execute_once()
    assert events == ["gates", "live_source_hash"]


@pytest.mark.skipif(runner.sys.platform != "darwin", reason="Darwin renamex_np contract")
def test_final_publication_is_exclusive(tmp_path: Path) -> None:
    source = tmp_path / "scratch"
    destination = tmp_path / "published"
    source.mkdir()
    (source / "evidence").write_bytes(b"locked")
    runner._rename_exclusive(source, destination)
    assert (destination / "evidence").read_bytes() == b"locked"

    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(OSError):
        runner._rename_exclusive(second, destination)
    assert second.is_dir()
