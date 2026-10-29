from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/exp_cad_0035_process_monitor.py"
SPEC = importlib.util.spec_from_file_location("exp35_process_monitor", PATH)
MONITOR = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MONITOR)


def test_frozen_resource_caps_and_source_independent_surface() -> None:
    assert MONITOR.SOURCE_WALL_LIMIT_SECONDS == 180.0
    assert MONITOR.RENDERER_WALL_LIMIT_SECONDS == 60.0
    assert MONITOR.RSS_LIMIT_BYTES == 4 * 1024**3
    source = PATH.read_text(encoding="utf-8")
    for forbidden in (
        "N009",
        "N010",
        "read_point_cloud",
        "evaluator",
        "gold",
        "expected_counts",
        "quality",
    ):
        assert forbidden not in source


def test_process_tree_rss_includes_all_descendants_once(monkeypatch) -> None:
    monkeypatch.setattr(
        MONITOR,
        "_process_table",
        lambda: {
            10: (1, 100),
            11: (10, 200),
            12: (11, 300),
            13: (1, 400),
            14: (12, 500),
        },
    )
    rss, pids = MONITOR.process_tree_snapshot(10)
    assert rss == 100 + 200 + 300 + 500
    assert pids == frozenset({10, 11, 12, 14})
    assert MONITOR.process_tree_rss_bytes(10) == rss


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  123456 maximum resident set size\n", 123456),
        ("Maximum resident set size (kbytes): 123\n", 123 * 1024),
        ("no supported field", None),
    ],
)
def test_parse_time_l_max_rss(text: str, expected: int | None) -> None:
    assert MONITOR.parse_time_l_max_rss(text) == expected


def test_canonical_publication_nulls_nonfinite_and_is_compact(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    MONITOR.canonical_json_write(
        path,
        {"z": [math.nan, math.inf, -math.inf], "a": "שלום"},
    )
    assert path.read_bytes() == (
        '{"a":"שלום","z":[null,null,null]}\n'.encode("utf-8")
    )
    assert json.loads(path.read_text(encoding="utf-8"))["z"] == [None, None, None]


def test_successful_fresh_child_publishes_external_measurement(tmp_path: Path) -> None:
    root = tmp_path / "measurement"
    result = MONITOR.run_measured_command(
        [sys.executable, "-c", "print('synthetic-only')"],
        measurement_root=root,
        wall_limit_seconds=10.0,
        rss_limit_bytes=512 * 1024**2,
    )
    assert result["status"] == "PASS"
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["rss_limit_exceeded"] is False
    assert result["wall_seconds"] <= 10.0
    assert result["peak_process_tree_rss_bytes"] <= 512 * 1024**2
    assert result["observed_process_tree_pids"] == sorted(
        set(result["observed_process_tree_pids"])
    )
    assert result["root_pid"] in result["observed_process_tree_pids"]
    assert all(pid > 0 for pid in result["observed_process_tree_pids"])
    assert (root / "PROCESS_MEASUREMENT.json").is_file()
    assert (root / "stdout.txt").read_text().strip() == "synthetic-only"


def test_measurement_records_observed_descendant_pid_union(tmp_path: Path) -> None:
    code = (
        "import subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.25)']);"
        "time.sleep(0.25);p.wait()"
    )
    result = MONITOR.run_measured_command(
        [sys.executable, "-c", code],
        measurement_root=tmp_path / "pid-tree",
        wall_limit_seconds=5.0,
        rss_limit_bytes=512 * 1024**2,
    )
    assert result["status"] == "PASS"
    observed = result["observed_process_tree_pids"]
    assert observed == sorted(set(observed))
    assert result["root_pid"] in observed
    assert len(observed) >= 3


def test_wall_breach_kills_process_group_and_is_terminal(tmp_path: Path) -> None:
    result = MONITOR.run_measured_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        measurement_root=tmp_path / "timeout",
        wall_limit_seconds=0.08,
        rss_limit_bytes=512 * 1024**2,
    )
    assert result["status"] == "FAIL_TERMINAL"
    assert result["timed_out"] is True
    assert result["resource_kill_reason"] == "WALL_LIMIT"


def test_rss_breach_kills_process_group_and_is_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MONITOR,
        "process_tree_snapshot",
        lambda pid: (MONITOR.RSS_LIMIT_BYTES + 1, frozenset({pid})),
    )
    result = MONITOR.run_measured_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        measurement_root=tmp_path / "rss",
        wall_limit_seconds=5.0,
        rss_limit_bytes=MONITOR.RSS_LIMIT_BYTES,
    )
    assert result["status"] == "FAIL_TERMINAL"
    assert result["rss_limit_exceeded"] is True
    assert result["resource_kill_reason"] == "RSS_LIMIT"


def test_sampler_failure_kills_child_and_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MONITOR,
        "process_tree_snapshot",
        lambda pid: (_ for _ in ()).throw(RuntimeError("sampler unavailable")),
    )
    result = MONITOR.run_measured_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        measurement_root=tmp_path / "sampler",
        wall_limit_seconds=5.0,
    )
    assert result["status"] == "FAIL_TERMINAL"
    assert result["sampler_failed"] is True
    assert result["resource_kill_reason"] == "RSS_SAMPLER_FAILURE"


def test_nonempty_measurement_root_is_rejected_before_spawn(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "measurement"
    root.mkdir()
    (root / "existing").write_bytes(b"x")

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("nonempty-root gate must precede child spawn")

    monkeypatch.setattr(MONITOR.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="absent or empty"):
        MONITOR.run_measured_command(
            [sys.executable, "-c", "pass"],
            measurement_root=root,
            wall_limit_seconds=1.0,
        )


@pytest.mark.parametrize("wall", [0.0, -1.0, math.inf, math.nan])
def test_invalid_wall_limit_rejected_before_spawn(
    monkeypatch, tmp_path: Path, wall: float
) -> None:
    monkeypatch.setattr(
        MONITOR.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid limit must precede child spawn")
        ),
    )
    with pytest.raises(ValueError, match="finite and positive"):
        MONITOR.run_measured_command(
            [sys.executable, "-c", "pass"],
            measurement_root=tmp_path / "never",
            wall_limit_seconds=wall,
        )
