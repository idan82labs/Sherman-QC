#!/usr/bin/env python3
"""Strict ordered, visually gated, terminal one-shot V32 driver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v32_avhfm_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v32-avhfm-v1"
STATE_PATH = OUTPUT_ROOT / "source_free_gate_result.json"
POST_VISUAL_MANIFEST = OUTPUT_ROOT / "postinvoke_visual_manifest.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_v32_avhfm_source_free import VARIANTS, run  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": lock["profile"],
        "terminal": False,
        "passed": False,
        "token": "V32_PHYSICAL_GATES_NOT_COMPLETE",
        "awaiting_manual_visual_fixture": None,
        "physical": [{"fixture_id": fixture_id, "status": "NOT_RUN", "visual_signed": False} for fixture_id in lock["physical_fixture_ids"]],
        "metamorphics": "NOT_RUN",
        "replay_one": "NOT_RUN",
        "replay_max": "NOT_RUN",
        "stress": "NOT_RUN",
    }


def _state(lock: dict[str, Any]) -> dict[str, Any]:
    return _load(STATE_PATH) if STATE_PATH.exists() else _initial(lock)


def _decision_result(result: dict[str, Any]) -> tuple[int, int, str | None, int]:
    decisions = result["decisions"]
    accepted = [row for row in decisions if row["action"] == "AVHFM_PASS"]
    reason = decisions[0].get("reason") if decisions else None
    suppressed = sum(row.get("nms_reason") == "ORDINARY_NMS_SUPPRESSED" for row in decisions)
    return int(result["pre_nms_rescue_count"]), int(result["rescue_count"]), reason, suppressed


def _check_expected(fixture_id: str, row: dict[str, Any], expected: dict[str, Any], phase_geometry: Any) -> tuple[bool, dict[str, Any]]:
    result = row["result"]
    pre, final, reason, suppressed = _decision_result(result)
    passed = pre == int(expected["pre_nms"]) and final == int(expected["final"]) and reason == expected["reason"]
    if "nms_suppressed" in expected:
        passed = passed and suppressed == int(expected["nms_suppressed"])
    observed: dict[str, Any] = {"pre_nms": pre, "final": final, "reason": reason, "nms_suppressed": suppressed}
    if expected.get("phase_gate"):
        decision = result["decisions"][0]
        measure = int(decision.get("accepted_length_q21", 0)) / (1 << 21)
        gap = int(decision.get("maximum_gap_q21", 0)) / (1 << 21)
        geometry = result["final_lines"][0] if result["final_lines"] else None
        passed = passed and measure >= 22.5 and gap <= 0.5 and geometry is not None
        if phase_geometry is not None:
            passed = passed and geometry["line_start"] == phase_geometry["line_start"] and geometry["line_end"] == phase_geometry["line_end"]
        observed.update({"accepted_measure_s": measure, "maximum_gap_s": gap, "canonical_geometry": geometry})
    return passed, observed


def _terminal(state: dict[str, Any], stage: str, detail: Any, token: str = "V32_AVHFM_TERMINAL_RETIRE") -> int:
    state.update({"terminal": True, "passed": False, "token": token, "terminal_stage": stage, "terminal_detail": detail, "awaiting_manual_visual_fixture": None})
    for row in state["physical"]:
        if row["status"] == "NOT_RUN":
            row["status"] = "NOT_RUN_AFTER_TERMINAL_FAILURE"
    for key in ("metamorphics", "replay_one", "replay_max", "stress"):
        if state[key] == "NOT_RUN":
            state[key] = "NOT_RUN_AFTER_TERMINAL_FAILURE"
    _write(STATE_PATH, state)
    print(json.dumps({"terminal": True, "stage": stage, "detail": detail, "token": token}, sort_keys=True))
    return 2


def run_next_physical() -> int:
    lock = _load(LOCK_PATH)
    state = _state(lock)
    if state.get("terminal"):
        raise SystemExit("V32 is terminal and may not be rerun")
    if state.get("awaiting_manual_visual_fixture"):
        raise SystemExit(f"manual native-resolution review required for {state['awaiting_manual_visual_fixture']}")
    pending = next((row for row in state["physical"] if row["status"] == "NOT_RUN"), None)
    if pending is None:
        print(json.dumps({"physical_complete": True, "next": "continue-nonvisual"}, sort_keys=True))
        return 0
    fixture_id = pending["fixture_id"]
    ordinal = state["physical"].index(pending) + 1
    row = run(fixture_id, "base", f"physical_{ordinal:02d}", "one", True)
    phase_geometry = next((value.get("observed", {}).get("canonical_geometry") for value in state["physical"] if value["fixture_id"] == "SF01_FOLD60_MESH_C2"), None)
    passed, observed = _check_expected(fixture_id, row, lock["expected_results"][fixture_id], phase_geometry)
    pending.update({"status": "NUMERIC_PASS_AWAITING_VISUAL" if passed else "NUMERIC_MISMATCH", "observed": observed, "result_path": row["result_path"], "result_sha256": row["result_sha256"], "cache_sha256": row["cache_sha256"], "semantic_result_sha256": row["semantic_result_sha256"]})
    visual_dir = OUTPUT_ROOT / "postinvoke_visuals" / fixture_id
    manifest = _load(POST_VISUAL_MANIFEST) if POST_VISUAL_MANIFEST.exists() else {"kind": "v32_avhfm_postinvoke_visual_manifest", "all_signed": False, "rows": []}
    for name in ("harmonic_current_multiview.png", "axial_measure_audit.png", "final_overlay.png"):
        path = visual_dir / name
        manifest["rows"].append({"fixture_id": fixture_id, "kind": name.removesuffix(".png"), "path": str(path.relative_to(ROOT)), "sha256": _sha(path), "width": 4096, "height": 4096, "signed": False, "decision": None})
    _write(POST_VISUAL_MANIFEST, manifest)
    if not passed:
        return _terminal(state, f"physical:{fixture_id}", observed, lock["tokens"]["phase_stop"] if ordinal <= 5 else lock["tokens"]["terminal_retire"])
    state["awaiting_manual_visual_fixture"] = fixture_id
    _write(STATE_PATH, state)
    print(json.dumps({"fixture_id": fixture_id, "numeric_pass": True, "awaiting_manual_native_visual_review": True, "observed": observed}, sort_keys=True))
    return 0


def sign_current(decision: str) -> int:
    lock = _load(LOCK_PATH)
    state = _state(lock)
    fixture_id = state.get("awaiting_manual_visual_fixture")
    if not fixture_id:
        raise SystemExit("no V32 physical result is awaiting visual review")
    if decision != "accept":
        return _terminal(state, f"visual:{fixture_id}", "native imagery contradicts the causal contract")
    row = next(value for value in state["physical"] if value["fixture_id"] == fixture_id)
    row["status"] = "PASS"
    row["visual_signed"] = True
    state["awaiting_manual_visual_fixture"] = None
    manifest = _load(POST_VISUAL_MANIFEST)
    for visual in manifest["rows"]:
        if visual["fixture_id"] == fixture_id:
            visual["signed"] = True
            visual["decision"] = "ACCEPT_CAUSAL_GEOMETRY_AND_NUMERIC_OUTCOME"
    manifest["all_signed"] = len(manifest["rows"]) == 81 and all(value["signed"] for value in manifest["rows"])
    _write(POST_VISUAL_MANIFEST, manifest)
    _write(STATE_PATH, state)
    print(json.dumps({"fixture_id": fixture_id, "visual_signed": True, "next": "next-physical"}, sort_keys=True))
    return 0


def continue_nonvisual() -> int:
    lock = _load(LOCK_PATH)
    state = _state(lock)
    if state.get("terminal"):
        raise SystemExit("V32 is terminal")
    if not all(row["status"] == "PASS" and row["visual_signed"] for row in state["physical"]):
        raise SystemExit("all 27 physical results must pass native visual review first")
    base_semantic = {row["fixture_id"]: row["semantic_result_sha256"] for row in state["physical"]}
    if state["metamorphics"] == "NOT_RUN":
        for fixture_id in lock["physical_fixture_ids"]:
            for variant in VARIANTS[1:]:
                row = run(fixture_id, variant, f"metamorphic_{fixture_id}", "max", False)
                if row["semantic_result_sha256"] != base_semantic[fixture_id]:
                    return _terminal(state, f"metamorphic:{fixture_id}:{variant}", {"base": base_semantic[fixture_id], "observed": row["semantic_result_sha256"]})
        state["metamorphics"] = "PASS"
        _write(STATE_PATH, state)
    for mode, state_key in (("one", "replay_one"), ("max", "replay_max")):
        if state[state_key] == "NOT_RUN":
            for fixture_id in lock["physical_fixture_ids"]:
                row = run(fixture_id, "base", f"{state_key}_{fixture_id}", mode, False)
                if row["semantic_result_sha256"] != base_semantic[fixture_id]:
                    return _terminal(state, f"{state_key}:{fixture_id}", {"base": base_semantic[fixture_id], "observed": row["semantic_result_sha256"]})
            state[state_key] = "PASS"
            _write(STATE_PATH, state)
    if state["stress"] == "NOT_RUN":
        started = time.monotonic()
        one = run("STRESS_200K_128", "base", "stress_one", "one", False)
        maximum = run("STRESS_200K_128", "base", "stress_max", "max", False)
        elapsed = time.monotonic() - started
        result = one["result"]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform == "darwin" else 1)
        passed = int(result["pre_nms_rescue_count"]) == 0 and int(result["rescue_count"]) == 0 and int(result["cache_bytes"]) == 131_328 and one["semantic_result_sha256"] == maximum["semantic_result_sha256"] and one["cache_sha256"] == maximum["cache_sha256"] and elapsed <= 300.0 and rss < 16 * 1024**3
        if not passed:
            return _terminal(state, "stress", {"pre_nms": result["pre_nms_rescue_count"], "final": result["rescue_count"], "cache_bytes": result["cache_bytes"], "elapsed_seconds": elapsed, "rss_bytes": rss})
        state["stress"] = {"status": "PASS", "elapsed_seconds": elapsed, "rss_bytes": rss, "cache_bytes": result["cache_bytes"]}
        _write(STATE_PATH, state)
    state.update({"passed": True, "terminal": False, "token": lock["tokens"]["source_free_pass"]})
    _write(STATE_PATH, state)
    print(json.dumps({"passed": True, "token": state["token"], "stress": state["stress"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--next-physical", action="store_true")
    actions.add_argument("--sign-current", choices=("accept", "reject"))
    actions.add_argument("--continue-nonvisual", action="store_true")
    args = parser.parse_args()
    if args.next_physical:
        return run_next_physical()
    if args.sign_current:
        return sign_current(args.sign_current)
    return continue_nonvisual()


if __name__ == "__main__":
    raise SystemExit(main())
