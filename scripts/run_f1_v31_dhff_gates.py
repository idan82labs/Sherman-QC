#!/usr/bin/env python3
"""Terminal one-shot V31 DHFF source-free gate driver."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v31_dhff_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v31-dhff-v1"
GATE_PATH = OUTPUT_ROOT / "source_free_gate_result.json"
POST_VISUAL_MANIFEST = OUTPUT_ROOT / "postinvoke_visual_manifest.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_v31_dhff_source_free import VARIANTS, run  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_result(row: dict[str, Any]) -> tuple[int, int, int, str | None, int]:
    result = row["result"]
    first = result["decisions"][0]
    return int(result["pre_nms_rescue_count"]), int(result["rescue_count"]), int(first["station_mask"]), first["reason"], sum(1 for value in result["decisions"] if value.get("nms_reason") == "ORDINARY_NMS_SUPPRESSED")


def _check_expected(fixture_id: str, row: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    observed = _decision_result(row)
    passed = observed[0] == int(expected["pre_nms"]) and observed[1] == int(expected["final"])
    if "mask" in expected:
        passed = passed and observed[2] == int(expected["mask"])
    if "reason" in expected:
        passed = passed and observed[3] == expected["reason"]
    if "nms_suppressed" in expected:
        passed = passed and observed[4] == int(expected["nms_suppressed"])
    if fixture_id.startswith("SF0") and "FRAGMENTED" in fixture_id:
        reasons = row["result"]["decisions"][0]["station_reasons"]
        passed = passed and reasons[1:4] == ["NO_MATERIAL_CONNECTION"] * 3
    return passed, {"pre_nms": observed[0], "final": observed[1], "mask": observed[2], "reason": observed[3], "nms_suppressed": observed[4]}


def _terminal(lock: dict[str, Any], stage: str, detail: Any, started: float) -> int:
    result = {"profile": lock["profile"], "passed": False, "terminal": True, "stage": stage, "detail": detail, "elapsed_seconds": time.monotonic() - started, "token": lock["tokens"]["source_free_retire"]}
    _write(GATE_PATH, result)
    print(json.dumps(result, sort_keys=True))
    return 2


def main() -> int:
    if GATE_PATH.exists():
        prior = _load(GATE_PATH)
        if prior.get("terminal") is True:
            raise SystemExit("V31 source-free execution is terminal and cannot be rerun")
    lock = _load(LOCK_PATH)
    started = time.monotonic()
    base_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for fixture_id in lock["fixture_ids"]:
        row = run(fixture_id, "base", "lexical_base", "one", True)
        passed, observed = _check_expected(fixture_id, row, lock["expected_results"][fixture_id])
        base_rows.append({"fixture_id": fixture_id, "passed": passed, "observed": observed, "result_sha256": row["result_sha256"], "cache_sha256": row["cache_sha256"]})
        visual_dir = OUTPUT_ROOT / "postinvoke_visuals" / fixture_id
        for name in ("station_flux_4096.png", "phi_theta_certificate_4096.png", "final_native_overlay_4096.png"):
            path = visual_dir / name
            visual_rows.append({"fixture_id": fixture_id, "kind": name.removesuffix(".png"), "path": str(path.relative_to(ROOT)), "sha256": _sha(path), "width": 4096, "height": 4096, "signed": False, "decision": None})
        if not passed:
            _write(POST_VISUAL_MANIFEST, {"kind": "v31_dhff_postinvoke_visual_manifest", "all_signed": False, "rows": visual_rows})
            return _terminal(lock, f"base:{fixture_id}", observed, started)

    metamorphic_rows: list[dict[str, Any]] = []
    for fixture_id, base in zip(lock["fixture_ids"], base_rows):
        for variant in VARIANTS[1:]:
            row = run(fixture_id, variant, "metamorphic", "one", False)
            same = row["result_sha256"] == base["result_sha256"] and row["cache_sha256"] == base["cache_sha256"]
            metamorphic_rows.append({"fixture_id": fixture_id, "variant": variant, "passed": same, "result_sha256": row["result_sha256"], "cache_sha256": row["cache_sha256"]})
            if not same:
                _write(POST_VISUAL_MANIFEST, {"kind": "v31_dhff_postinvoke_visual_manifest", "all_signed": False, "rows": visual_rows})
                return _terminal(lock, f"metamorphic:{fixture_id}:{variant}", metamorphic_rows[-1], started)

    replay_rows: list[dict[str, Any]] = []
    for fixture_id, base in zip(lock["fixture_ids"], base_rows):
        for mode in ("one", "max"):
            for replay in range(1, 4):
                row = run(fixture_id, "base", f"{mode}_replay_{replay}", mode, False)
                same = row["result_sha256"] == base["result_sha256"] and row["cache_sha256"] == base["cache_sha256"]
                replay_rows.append({"fixture_id": fixture_id, "workers": mode, "replay": replay, "passed": same})
                if not same:
                    _write(POST_VISUAL_MANIFEST, {"kind": "v31_dhff_postinvoke_visual_manifest", "all_signed": False, "rows": visual_rows})
                    return _terminal(lock, f"replay:{fixture_id}:{mode}:{replay}", replay_rows[-1], started)

    stress_started = time.monotonic()
    stress_one = run("STRESS_200K_128", "base", "stress_one", "one", False)
    stress_max = run("STRESS_200K_128", "base", "stress_max", "max", False)
    stress_elapsed = time.monotonic() - stress_started
    stress_result = stress_one["result"]
    stress_passed = (
        int(stress_result["rescue_count"]) == 8
        and int(stress_result["pre_nms_rescue_count"]) == 8
        and int(stress_result["cache_bytes"]) <= 24_832
        and stress_one["result_sha256"] == stress_max["result_sha256"]
        and stress_one["cache_sha256"] == stress_max["cache_sha256"]
        and stress_elapsed <= float(lock["constants"]["wall_seconds"])
        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform == "darwin" else 1) < int(lock["constants"]["process_tree_rss_bytes"])
    )
    if not stress_passed:
        _write(POST_VISUAL_MANIFEST, {"kind": "v31_dhff_postinvoke_visual_manifest", "all_signed": False, "rows": visual_rows})
        return _terminal(lock, "stress", {"one": stress_one["result_sha256"], "max": stress_max["result_sha256"], "rescue_count": stress_result["rescue_count"], "cache_bytes": stress_result["cache_bytes"], "elapsed_seconds": stress_elapsed}, started)

    manifest = {"kind": "v31_dhff_postinvoke_visual_manifest", "all_signed": False, "rows": visual_rows}
    _write(POST_VISUAL_MANIFEST, manifest)
    result = {
        "profile": lock["profile"],
        "passed": True,
        "terminal": False,
        "awaiting_manual_postinvoke_visual_signoff": True,
        "base_rows": base_rows,
        "metamorphic_count": len(metamorphic_rows),
        "replay_count": len(replay_rows),
        "stress": {"passed": True, "rescue_count": 8, "cache_bytes": stress_result["cache_bytes"], "elapsed_seconds": stress_elapsed},
        "elapsed_seconds": time.monotonic() - started,
        "token": "V31_DHFF_SOURCE_FREE_NUMERIC_PASS_AWAITING_VISUAL",
    }
    _write(GATE_PATH, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
