#!/usr/bin/env python3
"""Fail-closed source-free preflight for the frozen V51 blind tranche."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES


TOKEN = "V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED"
LOCK_PATH = ROOT / "research/f1_v51_new_untouched_corpus_lock_v2.json"
EVENTS_PATH = ROOT / "output/research/f1-v51-new-untouched-corpus-20260814/invocation_events.jsonl"
SANDBOX_PATH = ROOT / "research/f1_v51_new_untouched_corpus_detector.sb"

PACKET_TESTS = (
    "tests/test_build_professor_f1_v51_post_prospective_package_20260814.py",
    "tests/test_f1_elongated_lane_consolidation_v46.py",
    "tests/test_f1_route_independent_elongated_lane_v47.py",
    "tests/test_f1_nested_partial_duplicate_nms_v48.py",
    "tests/test_f1_short_span_cross_source_duplicate_nms_v49.py",
    "tests/test_f1_longest_incumbent_ownership_v50.py",
    "tests/test_f1_single_axis_parallel_return_recovery_v51.py",
)
FOCUSED_INHERITED_TESTS = (
    "tests/test_f1_elongated_lane_consolidation_v46.py",
    "tests/test_f1_route_independent_elongated_lane_v47.py",
    "tests/test_f1_nested_partial_duplicate_nms_v48.py",
    "tests/test_f1_short_span_cross_source_duplicate_nms_v49.py",
    "tests/test_f1_longest_incumbent_ownership_v50.py",
    "tests/test_f1_single_axis_parallel_return_recovery_v51.py",
    "tests/test_run_isolated_eyal_f1_baseline.py",
    "tests/test_run_isolated_f1_blind_ply_benchmark.py",
    "tests/test_f1_fragment_tolerant_parent_bridge_v41.py",
    "tests/test_run_f1_v41_v16_real_scan_compatibility.py",
    "tests/test_f1_single_shell_parent_bridge_v42.py",
    "tests/test_f1_v43_fragment_stitch_v44.py",
    "tests/test_f1_deterministic_paired_return_completion_v45.py",
    "tests/test_select_f1_mesh_route_v16.py::test_native_full_resolution_mesh_never_enters_bounded_v15_lane",
)
PROTOCOL_TESTS = (
    "tests/test_run_f1_v51_new_untouched_corpus_blind_once.py",
    "tests/test_score_f1_v51_new_untouched_corpus.py",
    "tests/test_finalize_f1_v51_new_corpus_gate.py",
    "tests/test_render_f1_v51_axis_normal_stack.py",
)

DENIAL_PROBES = (
    "data/bend_corpus/annotations/ground_truth_registry.json",
    "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json",
    "output/research/f1-v51-new-untouched-corpus-20260814/truth/DENIAL_PROBE.txt",
    "output/research/f1-v51-new-untouched-corpus-20260814/family_map/DENIAL_PROBE.txt",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _pytest(paths: tuple[str, ...], expected_count: int) -> dict[str, Any]:
    command = [str(ROOT / ".venv/bin/pytest"), "-q", *paths]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "LC_ALL": "C",
            "TZ": "UTC",
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    passed = completed.returncode == 0 and f"{expected_count} passed" in output
    return {
        "passed": passed,
        "expected_count": expected_count,
        "returncode": completed.returncode,
        "command": command,
        "output_tail": output[-2000:],
    }


def _sandbox_denied(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(SANDBOX_PATH), "/bin/cat", str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=15,
    )
    return {
        "path": str(path),
        "exists": path.exists(),
        "denied": completed.returncode != 0,
        "returncode": completed.returncode,
        "stdout_bytes": len(completed.stdout),
    }


def audit(*, run_tests: bool) -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    checks.append(_check("token_exact", lock.get("token") == TOKEN, lock.get("token")))
    observed_bindings = []
    for row in lock.get("source_bindings") or []:
        path = ROOT / str(row["path"])
        observed = _sha(path) if path.is_file() else None
        match = observed == row["sha256"]
        observed_bindings.append({**row, "observed_sha256": observed, "matches": match})
    checks.append(
        _check(
            "all_source_model_config_test_hashes_match",
            bool(observed_bindings) and all(row["matches"] for row in observed_bindings),
            observed_bindings,
        )
    )

    packet_path = ROOT / lock["pre_exposure_packet"]["path"]
    checks.append(
        _check(
            "pre_exposure_packet_matches",
            _sha(packet_path) == lock["pre_exposure_packet"]["sha256"],
            {"path": str(packet_path), "observed_sha256": _sha(packet_path)},
        )
    )
    packet_manifest = ROOT / lock["pre_exposure_manifest"]["path"]
    checks.append(
        _check(
            "pre_exposure_manifest_matches",
            _sha(packet_manifest) == lock["pre_exposure_manifest"]["sha256"],
            {"path": str(packet_manifest), "observed_sha256": _sha(packet_manifest)},
        )
    )

    event_rows = [json.loads(line) for line in EVENTS_PATH.read_text().splitlines() if line.strip()]
    detector_count = sum(int(row.get("detector_invocation_count") or 0) for row in event_rows)
    event_names = [str(row.get("event")) for row in event_rows]
    checks.append(
        _check(
            "append_only_precommit_events_and_zero_invocations",
            event_names == [
                "PRECOMMIT_LOCK_CREATED",
                "PRECOMMIT_BINDINGS_SEALED",
                "PRECOMMIT_LOCK_V1_INVALIDATED_NO_CORPUS",
                "PRECOMMIT_BINDINGS_V2_SEALED",
            ]
            and all(row.get("token") == TOKEN for row in event_rows)
            and detector_count == 0,
            {
                "event_count": len(event_rows),
                "events": event_names,
                "detector_invocation_count": detector_count,
            },
        )
    )

    builder_text = (ROOT / "scripts/build_f1_scan_only_candidate_pool.py").read_text(encoding="utf-8")
    checks.append(
        _check(
            "candidate_pool_contract_static",
            "GLOBAL_POINT_BUDGET = 200_000" in builder_text
            and '"source_point_serialization": False' in builder_text
            and '"support_atom_serialization": "bounded_identifiers_only"' in builder_text,
            {
                "working_point_budget": 200000,
                "source_point_serialization": False,
                "support_atom_serialization": "bounded_identifiers_only",
            },
        )
    )

    v9_path = ROOT / lock["model_bundles"]["v9_path"]
    v4_path = ROOT / lock["model_bundles"]["v4_path"]
    v9 = joblib.load(v9_path)
    v4 = joblib.load(v4_path)
    checks.append(
        _check(
            "model_profiles_and_feature_order_match",
            v9.get("profile") == "F1_BOUNDED_MESH_COMPLEXITY_ROUTER_V9"
            and v4.get("profile") == "F1_TOPOLOGY_ADJACENCY_RECALL_V4"
            and tuple(v9.get("feature_names") or ()) == FEATURE_NAMES
            and tuple(v4.get("feature_names") or ()) == FEATURE_NAMES,
            {"feature_count": len(FEATURE_NAMES), "v9_profile": v9.get("profile"), "v4_profile": v4.get("profile")},
        )
    )

    denial = [_sandbox_denied(ROOT / relative) for relative in DENIAL_PROBES]
    checks.append(
        _check(
            "detector_sandbox_denies_truth_and_gold",
            all(row["exists"] and row["denied"] and row["stdout_bytes"] == 0 for row in denial),
            denial,
        )
    )

    if run_tests:
        packet = _pytest(PACKET_TESTS, 25)
        broader = _pytest(FOCUSED_INHERITED_TESTS, 45)
        protocol = _pytest(PROTOCOL_TESTS, 7)
        checks.append(_check("frozen_packet_selector_25_of_25", packet["passed"], packet))
        checks.append(_check("frozen_focused_inherited_45_of_45", broader["passed"], broader))
        checks.append(_check("frozen_new_corpus_protocol_7_of_7", protocol["passed"], protocol))
    else:
        checks.append(_check("frozen_tests_not_skipped", False, "rerun with --run-tests"))

    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_new_untouched_corpus_source_free_preflight",
        "token": TOKEN,
        "research_only": True,
        "real_scans_opened": False,
        "truth_opened": False,
        "detector_invoked": False,
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "checks": checks,
        "passed": all(row["passed"] for row in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(run_tests=args.run_tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"passed": report["passed"], "check_count": len(report["checks"])}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
