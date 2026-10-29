#!/usr/bin/env python3
"""Strict one-shot post-lock gate controller for V30 AMFC_V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v30_amfc_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v30-amfc-v1"
PRECOMMIT_PATH = OUTPUT_ROOT / "precommit_manifest.json"
HARNESS_PATH = OUTPUT_ROOT / "gate1_harness_manifest.json"
RAW_VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "gate1_raw_visual_manifest.json"
RESULT_ROOT = OUTPUT_ROOT / "source_free_results"
REPORT_ROOT = OUTPUT_ROOT / "gate_reports"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
EXEC_VISUAL_MANIFEST_PATH = OUTPUT_ROOT / "VISUAL_MANIFEST.json"
WORKER = ROOT / "scripts" / "run_f1_v30_amfc_source_free.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_v30_amfc_source_free import VARIANTS, transform  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, value: Any) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / name
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock() -> dict[str, Any]:
    return _load(LOCK_PATH)


def _require_precommit() -> dict[str, Any]:
    manifest = _load(PRECOMMIT_PATH)
    if manifest.get("passed") is not True or manifest.get("token") != _lock()["tokens"]["mechanism_locked"]:
        raise SystemExit("V30 Gate 0 is not locked")
    for row in manifest["source_hashes"]:
        if _sha(ROOT / row["path"]) != row["sha256"]:
            raise SystemExit(f"V30 source drift: {row['path']}")
    return manifest


def _invocation_count() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "AMFC_INVOKE")


def gate1_verify() -> int:
    _require_precommit()
    harness = _load(HARNESS_PATH)
    visuals = _load(RAW_VISUAL_MANIFEST_PATH)
    rows = visuals.get("rows", [])
    hash_valid = all(_sha(ROOT / row["path"]) == row["sha256"] for row in rows)
    signed = bool(rows) and all(row.get("signed") is True and row.get("decision") == "PHYSICALLY_CORRECT" for row in rows)
    passed = (
        harness.get("mechanism_invocation_count") == 0
        and harness.get("passed_analytic_checks") is True
        and harness.get("raw_visuals_signed") is True
        and visuals.get("all_signed") is True
        and signed
        and hash_valid
        and _invocation_count() == 0
    )
    report = {
        "gate": 1,
        "passed": passed,
        "fixture_count": len(harness.get("fixture_rows", [])),
        "visual_count": len(rows),
        "visual_hashes_valid": hash_valid,
        "all_raw_visuals_signed": signed,
        "invocation_count": _invocation_count(),
        "terminal_token": None if passed else _lock()["tokens"]["harness_invalid_preinvoke"],
    }
    _write("gate1_fixture_validity.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


def _require_gate(name: str) -> dict[str, Any]:
    value = _load(REPORT_ROOT / name)
    if value.get("passed") is not True:
        raise SystemExit(f"prior V30 gate did not pass: {name}")
    return value


def _worker(fixture_id: str, variant: str, run_tag: str, worker_mode: str, *, render: bool) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(WORKER),
        "--fixture-id",
        fixture_id,
        "--variant",
        variant,
        "--run-tag",
        run_tag,
        "--worker-mode",
        worker_mode,
    ]
    if render:
        command.append("--render")
    subprocess.run(command, cwd=ROOT, check=True)


def _result(run_tag: str, fixture_id: str, variant: str = "base") -> dict[str, Any]:
    return _load(RESULT_ROOT / run_tag / fixture_id / variant / "result.json")


def _metrics(run_tag: str, fixture_id: str, variant: str = "base") -> dict[str, Any]:
    return _load(RESULT_ROOT / run_tag / fixture_id / variant / "metrics.json")


def _single_decision(result: dict[str, Any]) -> dict[str, Any]:
    if len(result["decisions"]) != 1:
        raise RuntimeError("fixture must contain exactly one candidate")
    return result["decisions"][0]


def gate2_source_free() -> int:
    _require_gate("gate1_fixture_validity.json")
    if _invocation_count() != 0:
        raise SystemExit("V30 source-free execution was already consumed")
    lock = _lock()
    rows = []
    failures = []
    for fixture_id in lock["fixture_ids"][:9]:
        try:
            _worker(fixture_id, "base", "base", "one", render=True)
            result = _result("base", fixture_id)
            decision = _single_decision(result)
            expected = lock["expected_results"][fixture_id]
            observed = {"rescues": int(result["rescue_count"]), "reason": decision.get("reason")}
            passed = observed == expected
        except Exception as exc:
            observed = {"exception": f"{type(exc).__name__}: {exc}"}
            expected = lock["expected_results"][fixture_id]
            passed = False
        rows.append({"fixture_id": fixture_id, "expected": expected, "observed": observed, "passed": passed})
        if not passed:
            failures.append(fixture_id)
            break
    report = {
        "gate": 2,
        "passed": not failures,
        "rows": rows,
        "failures": failures,
        "invocation_count": _invocation_count(),
        "terminal_token": None if not failures else lock["tokens"]["source_free_retire"],
    }
    _write("gate2_source_free_base.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not failures else 2


def _classification_signature(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(row["candidate_key"]),
            str(row["action"]),
            row.get("reason"),
            int(row["valid_station_count"]),
            int(row["valid_path_count"]),
            int(row["median_turn_error_q"]),
        )
        for row in result["decisions"]
    )


def _coarse_signature(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted((str(row["candidate_key"]), str(row["action"]), row.get("reason")) for row in result["decisions"])


def _line_error(expected: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    direct = max(float(np.linalg.norm(expected[0] - observed[0])), float(np.linalg.norm(expected[1] - observed[1])))
    reverse = max(float(np.linalg.norm(expected[0] - observed[1])), float(np.linalg.norm(expected[1] - observed[0])))
    position = min(direct, reverse)
    first = expected[1] - expected[0]
    second = observed[1] - observed[0]
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    angle = math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))
    return position, angle


def gate3_metamorphic() -> int:
    _require_gate("gate2_source_free_base.json")
    lock = _lock()
    exact_variants = {
        "candidate_axis_reversal",
        "parent_plane_swap",
        "point_or_vertex_permutation",
        "face_permutation_and_winding_reversal",
    }
    rows = []
    failures = []
    for fixture_id in lock["fixture_ids"][:9]:
        baseline = _result("base", fixture_id)
        fixture_payload = _load(OUTPUT_ROOT / "source_free_fixtures" / f"{fixture_id}.json")
        for variant in VARIANTS[1:]:
            try:
                _worker(fixture_id, variant, "metamorphic", "one", render=variant.startswith("composite_"))
                observed = _result("metamorphic", fixture_id, variant)
                coarse_equal = _coarse_signature(observed) == _coarse_signature(baseline)
                exact_equal = True
                if variant in exact_variants:
                    exact_equal = _classification_signature(observed) == _classification_signature(baseline)
                turn_ok = True
                if variant in {"rigid_euler_37_-23_61_translation", "uniform_scale_0_5", "uniform_scale_2_0", "composite_parent_swap_axis_reversal_permutation_rigid_scale_2"}:
                    base_by_key = {row["candidate_key"]: row for row in baseline["decisions"]}
                    turn_ok = all(abs(int(row["median_turn_error_q"]) - int(base_by_key[row["candidate_key"]]["median_turn_error_q"])) <= 1 for row in observed["decisions"])
                transformed = transform(fixture_payload, variant)
                expected_lines = {row["candidate_key"]: np.asarray([row["line_start"], row["line_end"]], dtype=np.float64) for row in transformed["candidate_records"]}
                line_errors = [_line_error(expected_lines[row["candidate_key"]], np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)) for row in observed["final_lines"]]
                line_ok = all(position <= 1.0e-6 * float(transformed["s_mm"]) and angle <= 1.0e-7 for position, angle in line_errors)
                passed = coarse_equal and exact_equal and turn_ok and line_ok and observed["ordinary_nms_call_count"] == 1
                evidence = {"coarse_equal": coarse_equal, "exact_equal": exact_equal, "turn_ok": turn_ok, "line_ok": line_ok}
            except Exception as exc:
                passed = False
                evidence = {"exception": f"{type(exc).__name__}: {exc}"}
            rows.append({"fixture_id": fixture_id, "variant": variant, "passed": passed, "evidence": evidence})
            if not passed:
                failures.append(f"{fixture_id}:{variant}")
                break
        if failures:
            break
    report = {
        "gate": 3,
        "passed": not failures,
        "checks": rows,
        "failures": failures,
        "invocation_count": _invocation_count(),
        "terminal_token": None if not failures else lock["tokens"]["metamorphic_or_replay_retire"],
    }
    _write("gate3_metamorphic.json", report)
    print(json.dumps({"gate": 3, "passed": report["passed"], "check_count": len(rows), "failures": failures}, sort_keys=True))
    return 0 if not failures else 2


def gate4_replay() -> int:
    _require_gate("gate3_metamorphic.json")
    lock = _lock()
    failures = []
    rows = []
    for fixture_id in lock["fixture_ids"][:9]:
        _worker(fixture_id, "base", "single_replay_2", "one", render=True)
        _worker(fixture_id, "base", "single_replay_3", "one", render=True)
        for tag in ("max_replay_1", "max_replay_2", "max_replay_3"):
            _worker(fixture_id, "base", tag, "max", render=True)
        tags = ("base", "single_replay_2", "single_replay_3", "max_replay_1", "max_replay_2", "max_replay_3")
        result_hashes = [_metrics(tag, fixture_id)["result_sha256"] for tag in tags]
        cache_hashes = [_metrics(tag, fixture_id)["cache_sha256"] for tag in tags]
        visual_hashes = [_metrics(tag, fixture_id)["visual_sha256"] for tag in tags]
        passed = len(set(result_hashes)) == len(set(cache_hashes)) == len(set(visual_hashes)) == 1
        rows.append({"fixture_id": fixture_id, "passed": passed, "result_hashes": result_hashes, "cache_hashes": cache_hashes, "visual_hashes": visual_hashes})
        if not passed:
            failures.append(fixture_id)
            break
    report = {
        "gate": 4,
        "passed": not failures,
        "rows": rows,
        "failures": failures,
        "invocation_count": _invocation_count(),
        "terminal_token": None if not failures else lock["tokens"]["metamorphic_or_replay_retire"],
    }
    _write("gate4_replay.json", report)
    print(json.dumps({"gate": 4, "passed": report["passed"], "failures": failures}, sort_keys=True))
    return 0 if not failures else 2


def gate5_resource() -> int:
    _require_gate("gate4_replay.json")
    lock = _lock()
    fixture_id = "SF10_RESOURCE_STRESS"
    rows = []
    failures = []
    for index in range(1, 6):
        tag = f"stress_{index}"
        try:
            _worker(fixture_id, "base", tag, "max", render=True)
            result = _result(tag, fixture_id)
            metrics = _metrics(tag, fixture_id)
            reasons = {row.get("reason") for row in result["decisions"]}
            passed = result["rescue_count"] == 0 and reasons == {"AXIAL_COVERAGE"} and result["cache_bytes"] <= lock["resource_contract"]["stress_summary_cache_bytes"] and int(metrics["process_tree_peak_rss_bytes"]) < lock["resource_contract"]["stress_peak_process_tree_rss_bytes"]
            evidence = {"elapsed_seconds": metrics["elapsed_seconds"], "rss_bytes": metrics["process_tree_peak_rss_bytes"], "cache_bytes": result["cache_bytes"], "reason_set": sorted(str(value) for value in reasons), "visual_sha256": metrics["visual_sha256"]}
        except Exception as exc:
            passed = False
            evidence = {"exception": f"{type(exc).__name__}: {exc}"}
        rows.append({"run": index, "passed": passed, "evidence": evidence})
        if not passed:
            failures.append(f"stress_{index}")
            break
    if not failures:
        elapsed = sorted(float(row["evidence"]["elapsed_seconds"]) for row in rows)
        p95 = elapsed[max(0, math.ceil(0.95 * len(elapsed)) - 1)]
        if p95 > lock["resource_contract"]["stress_incremental_p95_seconds"]:
            failures.append("stress_p95")
    else:
        p95 = None
    report = {
        "gate": 5,
        "passed": not failures,
        "rows": rows,
        "incremental_p95_seconds": p95,
        "failures": failures,
        "invocation_count": _invocation_count(),
        "terminal_token": None if not failures else lock["tokens"]["resource_retire"],
    }
    _write("gate5_resource.json", report)
    if not failures:
        visual_rows = []
        for fixture_id_base in lock["fixture_ids"][:9]:
            for tag, variant, kind in (("base", "base", "base"), ("metamorphic", "composite_parent_swap_axis_reversal_permutation_rigid_scale_2", "composite")):
                path = RESULT_ROOT / tag / fixture_id_base / variant / "diagnostic.png"
                visual_rows.append({"fixture_id": fixture_id_base, "kind": kind, "path": str(path.relative_to(ROOT)), "sha256": _sha(path), "width": 4096, "height": 4096, "signed": False, "decision": None})
        stress_path = RESULT_ROOT / "stress_1" / fixture_id / "base" / "diagnostic.png"
        visual_rows.append({"fixture_id": fixture_id, "kind": "stress", "path": str(stress_path.relative_to(ROOT)), "sha256": _sha(stress_path), "width": 4096, "height": 4096, "signed": False, "decision": None})
        EXEC_VISUAL_MANIFEST_PATH.write_text(json.dumps({"kind": "v30_amfc_execution_visual_manifest", "all_signed": False, "rows": visual_rows}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"gate": 5, "passed": report["passed"], "p95": p95, "failures": failures}, sort_keys=True))
    return 0 if not failures else 2


def gate6_visual() -> int:
    _require_gate("gate5_resource.json")
    manifest = _load(EXEC_VISUAL_MANIFEST_PATH)
    rows = manifest.get("rows", [])
    hashes_valid = all(_sha(ROOT / row["path"]) == row["sha256"] for row in rows)
    signed = bool(rows) and all(row.get("signed") is True and row.get("decision") == "PHYSICALLY_CORRECT" for row in rows)
    passed = manifest.get("all_signed") is True and hashes_valid and signed
    report = {
        "gate": 6,
        "passed": passed,
        "visual_count": len(rows),
        "hashes_valid": hashes_valid,
        "all_signed": signed,
        "eligible_for_prospective": passed,
        "terminal_token": None if passed else _lock()["tokens"]["visual_retire"],
    }
    _write("gate6_visual.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("gate1_verify", "gate2_source_free", "gate3_metamorphic", "gate4_replay", "gate5_resource", "gate6_visual"))
    args = parser.parse_args()
    return {
        "gate1_verify": gate1_verify,
        "gate2_source_free": gate2_source_free,
        "gate3_metamorphic": gate3_metamorphic,
        "gate4_replay": gate4_replay,
        "gate5_resource": gate5_resource,
        "gate6_visual": gate6_visual,
    }[args.gate]()


if __name__ == "__main__":
    raise SystemExit(main())
