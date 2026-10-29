#!/usr/bin/env python3
"""Strict post-precommit gate runner for V29 CRCO_V1.

This file never opens historical or real scan data.  It may only generate and
execute the immutable source-free suite after a passing static precommit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v29_crco_v1_lock.json"
AUDIT_PATH = ROOT / "output" / "research" / "f1-v29-crco-v1" / "precommit_manifest.json"
FIXTURE_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_fixtures"
RESULT_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_results"
REPORT_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "gate_reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_calibrated_range_continuity_ownership import select_unified_v29  # noqa: E402


TERMINAL = "V29_CRCO_TERMINAL_RETIRE"
METAMORPHIC_TERMINAL = "V29_CRCO_METAMORPHIC_FAILURE_RETIRE"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, payload: Any) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / name
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def _require_precommit() -> dict[str, Any]:
    audit = _load(AUDIT_PATH)
    if audit.get("status") != "ELIGIBLE_FOR_GATE_1" or audit.get("passed") is not True:
        raise SystemExit("V29 static precommit is not passing")
    for row in audit["source_hashes"]:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit("V29 frozen source hash drift")
    return audit


def _worker(fixture_id: str, run_tag: str) -> None:
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "run_f1_v29_crco_source_free.py"), "--fixture-id", fixture_id, "--run-tag", run_tag],
        cwd=ROOT,
        check=True,
    )


def _decision_signature(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for decision in result["decisions"]:
        candidate = decision["candidate"]
        rows.append(
            {
                "legacy_rank_bytes": candidate["legacy_rank_bytes"],
                "selected_legacy": candidate["selected_legacy"],
                "action": decision["action"],
                "continuous": decision["continuous_station_count"],
                "jump": decision["jump_station_count"],
                "occluded": decision["occluded_station_count"],
                "unknown": decision["unknown_station_count"],
                "continuous_run": decision["longest_continuous_run"],
                "jump_run": decision["longest_jump_run"],
                "states": decision["cross_view_states"],
            }
        )
    return sorted(rows, key=lambda row: (row["legacy_rank_bytes"], row["selected_legacy"], row["action"]))


def _behavior_signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "decisions": _decision_signature(result),
        "final_sources": sorted((row["source_lane"], row["v29_action"]) for row in result["final_lines"]),
        "final_line_count": len(result["final_lines"]),
        "queried_pixel_count": result["queried_pixel_count"],
    }


def gate1() -> int:
    _require_precommit()
    lock = _load(LOCK_PATH)
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "build_f1_v29_crco_source_free_fixtures.py")],
        cwd=ROOT,
        check=True,
    )
    rows = []
    failures = []
    for fixture_id in lock["source_free_fixture_ids"]:
        _worker(fixture_id, "run_a")
        result = _load(RESULT_ROOT / "run_a" / fixture_id / "result.json")
        observed = {
            "rescues": result["rescue_count"],
            "vetoes": result["veto_count"],
            "final_lines": len(result["final_lines"]),
        }
        expected = lock["hidden_expected_outputs"][fixture_id]
        passed = observed == expected
        rows.append({"fixture_id": fixture_id, "observed": observed, "expected": expected, "passed": passed})
        if not passed:
            failures.append(fixture_id)
    # Hidden semantic status must not change inference behavior.
    first, second = lock["semantic_twin_pair"]
    twin_equal = _behavior_signature(_load(RESULT_ROOT / "run_a" / first / "result.json")) == _behavior_signature(_load(RESULT_ROOT / "run_a" / second / "result.json"))
    if not twin_equal:
        failures.append("semantic_twin")
    report = {"gate": 1, "passed": not failures, "terminal_token": None if not failures else TERMINAL, "fixtures": rows, "semantic_twin_equal": twin_equal, "failures": failures}
    _write("gate1_source_free.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 2


def _reorder_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reorder_dict(value[key]) for key in reversed(tuple(value))}
    if isinstance(value, list):
        return [_reorder_dict(item) for item in value]
    return value


def _pixel_reverse(frame: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(frame)
    result["depth_q"] = [list(reversed(row)) for row in reversed(result["depth_q"])]
    result["valid"] = [list(reversed(row)) for row in reversed(result["valid"])]
    result["calibration"]["cx"] = float(result["width"] - 1) - float(result["calibration"]["cx"])
    result["calibration"]["cy"] = float(result["height"] - 1) - float(result["calibration"]["cy"])
    pose = np.asarray(result["sensor_to_scan"], dtype=np.float64)
    pose[:3, 0] *= -1.0
    pose[:3, 1] *= -1.0
    result["sensor_to_scan"] = pose.tolist()
    return result


def _rigid(axis: tuple[float, float, float], angle_deg: float, translation: tuple[float, float, float]) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray(
        [
            [c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
            [y*x*(1-c)+z*s, c + y*y*(1-c), y*z*(1-c)-x*s],
            [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c + z*z*(1-c)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _transform_fixture(fixture: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    result = copy.deepcopy(fixture)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    for candidate in result["candidate_records"]:
        for field in ("line_start", "line_end"):
            candidate[field] = (rotation @ np.asarray(candidate[field], dtype=np.float64) + translation).tolist()
        candidate["flank_tangents"] = [(rotation @ np.asarray(value, dtype=np.float64)).tolist() for value in candidate["flank_tangents"]]
    for frame in result["range_frames"]:
        frame["sensor_to_scan"] = (transform @ np.asarray(frame["sensor_to_scan"], dtype=np.float64)).tolist()
    return result


def gate2() -> int:
    _require_precommit()
    lock = _load(LOCK_PATH)
    if not _load(REPORT_ROOT / "gate1_source_free.json")["passed"]:
        raise SystemExit("Gate 1 did not pass")
    failures = []
    rows = []
    for fixture_id in lock["source_free_fixture_ids"]:
        fixture = _load(FIXTURE_ROOT / f"{fixture_id}.json")
        baseline = _behavior_signature(select_unified_v29(fixture["candidate_records"], fixture["range_frames"], fixture["s_mm"]))
        variants: dict[str, dict[str, Any]] = {}
        value = copy.deepcopy(fixture); value["candidate_records"].reverse(); variants["point_record_reversal"] = value
        value = copy.deepcopy(fixture); value["range_frames"].reverse(); variants["frame_record_reversal"] = value
        value = copy.deepcopy(fixture); value["range_frames"] = [_pixel_reverse(frame) for frame in value["range_frames"]]; variants["pixel_record_reversal"] = value
        variants["content_key_permutation"] = _reorder_dict(copy.deepcopy(fixture))
        value = copy.deepcopy(fixture)
        for candidate in value["candidate_records"]:
            candidate["line_start"], candidate["line_end"] = candidate["line_end"], candidate["line_start"]
        variants["candidate_endpoint_reversal"] = value
        value = copy.deepcopy(fixture)
        for candidate in value["candidate_records"]:
            candidate["flank_tangents"].reverse()
        variants["flank_vector_swap"] = value
        value = copy.deepcopy(fixture)
        for candidate in value["candidate_records"]:
            candidate["line_start"], candidate["line_end"] = candidate["line_end"], candidate["line_start"]
            candidate["flank_tangents"].reverse()
        variants["endpoint_reversal_plus_flank_swap"] = value
        for name, transformed in variants.items():
            observed = _behavior_signature(select_unified_v29(transformed["candidate_records"], transformed["range_frames"], transformed["s_mm"]))
            passed = observed == baseline
            rows.append({"fixture_id": fixture_id, "variant": name, "passed": passed})
            if not passed:
                failures.append(f"{fixture_id}:{name}")
        rigid_cases = {
            "identity": np.eye(4),
            "cyclic_axis_rotation_translation_13_-7_5_s": _rigid((1.0, 1.0, 1.0), 120.0, (13.0*fixture["s_mm"], -7.0*fixture["s_mm"], 5.0*fixture["s_mm"])),
            "axis_1_2_3_rotation_37deg_nonaxis_translation": _rigid((1.0, 2.0, 3.0), 37.0, (3.1, -4.7, 8.3)),
            "axis_-2_1_4_rotation_121deg_nonaxis_translation": _rigid((-2.0, 1.0, 4.0), 121.0, (-6.2, 2.9, 4.4)),
        }
        for name, transform in rigid_cases.items():
            transformed = _transform_fixture(fixture, transform)
            observed = _behavior_signature(select_unified_v29(transformed["candidate_records"], transformed["range_frames"], transformed["s_mm"]))
            passed = observed == baseline
            rows.append({"fixture_id": fixture_id, "variant": name, "passed": passed})
            if not passed:
                failures.append(f"{fixture_id}:{name}")
    report = {"gate": 2, "passed": not failures, "terminal_token": None if not failures else METAMORPHIC_TERMINAL, "checks": rows, "failures": failures}
    _write("gate2_metamorphic.json", report)
    print(json.dumps({"gate": 2, "passed": report["passed"], "check_count": len(rows), "failures": failures}, sort_keys=True))
    return 0 if not failures else 2


def gate3() -> int:
    _require_precommit()
    if not _load(REPORT_ROOT / "gate2_metamorphic.json")["passed"]:
        raise SystemExit("Gate 2 did not pass")
    lock = _load(LOCK_PATH)
    rows, failures = [], []
    for fixture_id in lock["source_free_fixture_ids"]:
        _worker(fixture_id, "run_b")
        first = _load(RESULT_ROOT / "run_a" / fixture_id / "result.json")
        second = _load(RESULT_ROOT / "run_b" / fixture_id / "result.json")
        passed = _behavior_signature(first) == _behavior_signature(second) and first["cache"] == second["cache"]
        rows.append({"fixture_id": fixture_id, "passed": passed})
        if not passed:
            failures.append(fixture_id)
    report = {"gate": 3, "passed": not failures, "terminal_token": None if not failures else TERMINAL, "fixtures": rows, "failures": failures}
    _write("gate3_clean_replay.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 2


def gate4() -> int:
    _require_precommit()
    if not _load(REPORT_ROOT / "gate3_clean_replay.json")["passed"]:
        raise SystemExit("Gate 3 did not pass")
    lock = _load(LOCK_PATH)
    elapsed, rows, failures = [], [], []
    for fixture_id in lock["source_free_fixture_ids"]:
        result = _load(RESULT_ROOT / "run_b" / fixture_id / "result.json")
        elapsed.append(float(result["elapsed_seconds"]))
        rss = int(result["process_maxrss_native_units"])
        passed = result["queried_pixel_count"] <= lock["constants"]["maximum_queried_pixels"] and result["cache_bytes"] <= lock["constants"]["maximum_cache_bytes"] and rss < lock["constants"]["maximum_process_tree_rss_bytes"]
        rows.append({"fixture_id": fixture_id, "elapsed_seconds": result["elapsed_seconds"], "rss_bytes": rss, "queried_pixels": result["queried_pixel_count"], "cache_bytes": result["cache_bytes"], "passed": passed})
        if not passed:
            failures.append(fixture_id)
    p95 = sorted(elapsed)[max(0, math.ceil(0.95 * len(elapsed)) - 1)]
    if p95 > lock["constants"]["maximum_incremental_p95_seconds"]:
        failures.append("incremental_p95")
    report = {"gate": 4, "passed": not failures, "terminal_token": None if not failures else TERMINAL, "incremental_p95_seconds": p95, "fixtures": rows, "failures": failures}
    _write("gate4_resources.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("gate1", "gate2", "gate3", "gate4"))
    args = parser.parse_args()
    return {"gate1": gate1, "gate2": gate2, "gate3": gate3, "gate4": gate4}[args.gate]()


if __name__ == "__main__":
    raise SystemExit(main())
