#!/usr/bin/env python3
"""Run every signed CIBEC_V1 source-free prerequisite gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cibec_v1.contract import (
    CibecValidationError,
    canonical_json_bytes,
    canonicalize_scan_truth,
    validate_cad_correspondence,
    validate_charter,
)
from scripts.build_cibec_v1_authority_candidate import candidate_outputs


DEFAULT_INPUT = ROOT / "output/research/cibec-v1-signed-materialization"
DEFAULT_OUTPUT = ROOT / "output/research/cibec-v1-source-free-qualification/qualification_report.json"
GEOMETRY = ROOT / "output/research/cibec-v1-neutral-authority-pack-v3/geometry"
RENDERER = ROOT / "scripts/render_f1_source_only_prediction_audit.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(fixture_id: str, row: dict[str, Any], ontology_sha: str) -> dict[str, Any]:
    events = []
    for event in row["events"]:
        events.append({
            "class_code": event["class_code"],
            "line_start_mm": event["line_start_mm"],
            "line_end_mm": event["line_end_mm"],
            "visibility": event["visibility"],
            "render_admissible": True,
            "verified_view_ids": ["V1", "V2"],
            "adjudication_rule_id": event["rule_id"],
        })
    return {
        "schema": "SCAN_BEND_EVENTS_V1",
        "ontology_sha256": ontology_sha,
        "renderer_sha256": _sha(RENDERER),
        "case_id": hashlib.sha256(f"CIBEC_SOURCE_FREE::{fixture_id}".encode()).hexdigest()[:32],
        "scan_sha256": _sha(GEOMETRY / f"{fixture_id}.ply"),
        "units": "mm",
        "coordinate_frame": "RAW_SCAN",
        "events": events,
        "case_status": row["case_status"],
    }


def _must_reject(payload: dict[str, Any], *, cad: bool = False) -> bool:
    try:
        validate_cad_correspondence(payload) if cad else canonicalize_scan_truth(payload)
    except CibecValidationError:
        return True
    return False


def run(input_dir: Path, output: Path) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    charter_path = input_dir / "CIBEC_V1_CHARTER.yaml"
    fixture_path = input_dir / "source_free_fixture_manifest.json"
    report: dict[str, Any] = {
        "schema": "CIBEC_V1_SOURCE_FREE_QUALIFICATION",
        "physical_assets_opened": 0,
        "prediction_assets_opened": 0,
        "fixture_count": 0,
        "gates": {},
    }
    try:
        charter = json.loads(charter_path.read_text())
        fixtures = json.loads(fixture_path.read_text())
        validate_charter(charter)
    except (OSError, json.JSONDecodeError, CibecValidationError) as exc:
        report.update(status="REFUSED_UNSIGNED_OR_INVALID_INPUT", blockers=[str(exc)])
        return 2, report
    expected = fixtures.get("expected_outputs")
    if fixtures.get("status") != "SIGNED_AND_FROZEN" or not isinstance(expected, dict):
        report.update(status="REFUSED_UNSIGNED_OR_INVALID_INPUT", blockers=["signed fixture expectations required"])
        return 2, report
    observed = candidate_outputs()
    fixture_ids = [f"S{index:02d}" for index in range(1, 15)]
    report["fixture_count"] = len(expected)
    exact = [fixture_id for fixture_id in fixture_ids if expected.get(fixture_id) == observed.get(fixture_id)]
    report["gates"]["semantic_exact"] = {"passed": len(exact) == 14, "exact": len(exact), "required": 14, "failed_ids": [item for item in fixture_ids if item not in exact]}
    ontology_sha = _sha(charter_path)
    replay_passed = []
    canonical = {}
    for fixture_id in fixture_ids:
        payload = _payload(fixture_id, observed[fixture_id], ontology_sha)
        rows = [canonicalize_scan_truth(copy.deepcopy(payload)) for _ in range(10)]
        blobs = [canonical_json_bytes(row) for row in rows]
        if len(set(blobs)) == 1:
            replay_passed.append(fixture_id)
        canonical[fixture_id] = rows[0]
    report["gates"]["ten_run_replay"] = {"passed": len(replay_passed) == 14, "exact": len(replay_passed), "required": 14}
    s02_equivalent = copy.deepcopy(observed["S02"])
    for event in s02_equivalent["events"]:
        event["line_start_mm"][1] *= -1
        event["line_end_mm"][1] *= -1
    report["gates"]["s02_rigid_equivalence"] = {"passed": s02_equivalent == observed["S01"]}
    base = _payload("S01", observed["S01"], ontology_sha)
    malformed = {}
    for name, mutation in {
        "nan": lambda row: row["events"][0]["line_start_mm"].__setitem__(0, float("nan")),
        "infinity": lambda row: row["events"][0]["line_end_mm"].__setitem__(0, float("inf")),
        "zero_length": lambda row: row["events"][0].__setitem__("line_end_mm", list(row["events"][0]["line_start_mm"])),
        "non_mm_units": lambda row: row.__setitem__("units", "inch"),
    }.items():
        value = copy.deepcopy(base); mutation(value); malformed[name] = _must_reject(value)
    bad_transform = {"schema": "CAD_EVENT_CORRESPONDENCE_V1", "primary_truth_sha256": "0" * 64, "step_sha256": "1" * 64, "overlay_transform_sha256": "bad", "overlay_status": "TRUSTED", "events": [], "cad_only_unobserved": []}
    malformed["malformed_transform"] = _must_reject(bad_transform, cad=True)
    report["gates"]["malformed_rejection"] = {"passed": all(malformed.values()), "checks": malformed}
    s03_many_to_one = {"schema": "CAD_EVENT_CORRESPONDENCE_V1", "primary_truth_sha256": "0" * 64, "step_sha256": "1" * 64, "overlay_transform_sha256": "2" * 64, "overlay_status": "TRUSTED", "events": [{"event_id": "E0001", "step_face_indices": [1, 2], "v9_primitive_ids": ["inner", "outer"], "relation": "MANY_TO_ONE", "mapping_status": "RESOLVED"}], "cad_only_unobserved": []}
    try:
        validate_cad_correspondence(s03_many_to_one); many_to_one = True
    except CibecValidationError:
        many_to_one = False
    duplicate = copy.deepcopy(s03_many_to_one)
    duplicate["events"].append({"event_id": "E0002", "step_face_indices": [2, 3], "v9_primitive_ids": ["duplicate"], "relation": "ONE_TO_ONE", "mapping_status": "RESOLVED"})
    report["gates"]["cad_face_ownership"] = {"passed": many_to_one and _must_reject(duplicate, cad=True), "many_faces_one_event_accepted": many_to_one, "duplicate_ownership_rejected": _must_reject(duplicate, cad=True)}
    abstain = {"schema": "CAD_EVENT_CORRESPONDENCE_V1", "primary_truth_sha256": "0" * 64, "step_sha256": "1" * 64, "overlay_transform_sha256": "2" * 64, "overlay_status": "ABSTAIN", "reason": "OVERLAY_UNAVAILABLE_OR_UNTRUSTED", "events": [], "cad_only_unobserved": []}
    try:
        validate_cad_correspondence(abstain); abstain_ok = True
    except CibecValidationError:
        abstain_ok = False
    report["gates"]["s14_cad_abstention"] = {"passed": abstain_ok and len(canonical["S14"]["events"]) == 1, "primary_events_retained": len(canonical["S14"]["events"]), "cad_abstained": abstain_ok}
    all_passed = all(gate["passed"] for gate in report["gates"].values())
    report["status"] = "PASS" if all_passed else "FAIL"
    report["blockers"] = [] if all_passed else [name for name, gate in report["gates"].items() if not gate["passed"]]
    report["runtime_seconds"] = round(time.perf_counter() - started, 6)
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 3)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return (0 if all_passed else 1), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    code, report = run(args.input.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
