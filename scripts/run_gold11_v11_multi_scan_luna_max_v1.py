#!/usr/bin/env python3
"""Run the versioned V11 healing candidate over the frozen consumed Gold11 set.

This is a new development evaluation, not a retry of the terminal V11 attempt.
It uses the exact hash-bound case list from the September input review and the
retained V11 control registration evidence.  When an accepted context is absent,
the V2 seam may select a rejected registration candidate only through its
label-free finite-distance/coverage gate.  Cases with no gated pose remain
explicit abstentions.

Gold reference lines are loaded only after every inference attempt has returned.
The unchanged one-to-one matcher scores the resulting predictions.  This
candidate is research-only: retained poses are not a calibrated datum and
native axes are conditional scan-supported contact-line proposals, not verified
manufacturing events.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REVIEW_PATH = ROOT / "scripts/review_gold11_inputs_v1.py"
CATALOG_PATH = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
PROTOCOL_PATH = ROOT / "docs/research/2026-09-15_gold11_v11_multi_scan_luna_max_v1_protocol.md"
ATTEMPT_ROOT = ROOT / "docs/research/gold11_mesh_development_v11_attempt/cases"
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_multi_scan_luna_max_v1"

REVIEW_SHA256 = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
CATALOG_SHA256 = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
SELECTED_REFERENCE_SHA256 = "d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc"
V11_REVIEW_RESULT_SHA256 = "1570128992d82c21cccc69a8d196e28b3ae5a7ac4de5406ee74bf4f5a57fd595"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_cases() -> list[dict[str, Any]]:
    blob = REVIEW_PATH.read_bytes()
    if hashlib.sha256(blob).hexdigest() != REVIEW_SHA256:
        raise ValueError("GOLD11_REVIEW_SOURCE_HASH_MISMATCH")
    source = blob.decode("utf-8")
    assignment = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CASES"
                for target in node.targets)
    )
    value = assignment.value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name) and value.func.value.id == "json"
            and value.func.attr == "loads" and len(value.args) == 1 and not value.keywords):
        raise ValueError("GOLD11_CASE_CONSTANT_SCHEMA")
    cases = json.loads(ast.literal_eval(value.args[0]))
    if (len(cases) != 11 or len({row["case_token"] for row in cases}) != 11
            or len({row["scan_sha256"] for row in cases}) != 11
            or len({row["cad_sha256"] for row in cases}) != 6):
        raise ValueError("GOLD11_CASE_SCOPE_MISMATCH")
    return cases


def _selected_references(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    blob = CATALOG_PATH.read_bytes()
    if hashlib.sha256(blob).hexdigest() != CATALOG_SHA256:
        raise ValueError("GOLD11_CATALOG_HASH_MISMATCH")
    payload = json.loads(blob)
    by_scan = {row["scan_sha256"]: row for row in cases}
    selected: dict[str, dict[str, Any]] = {}
    for row in payload.get("cases", []):
        case = by_scan.get(row.get("scan_sha256"))
        if case is None:
            continue
        token = case["case_token"]
        if token in selected or row.get("case_id") != case["reference_id"]:
            raise ValueError("GOLD11_REFERENCE_IDENTITY_OR_DUPLICATE")
        if row.get("scan_path") != case["scan_path"] or str(row.get("part_id")) != case["part_key"]:
            raise ValueError("GOLD11_REFERENCE_SOURCE_IDENTITY_MISMATCH")
        if not isinstance(row.get("bends"), list) or len(row["bends"]) > 64:
            raise ValueError("GOLD11_REFERENCE_SCHEMA_OR_BUDGET")
        selected[token] = row
    if set(selected) != {row["case_token"] for row in cases}:
        raise ValueError("GOLD11_REFERENCE_CASE_DENOMINATOR")
    if sum(len(row["bends"]) for row in selected.values()) != 71:
        raise ValueError("GOLD11_REFERENCE_LINE_DENOMINATOR")
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != SELECTED_REFERENCE_SHA256:
        raise ValueError("GOLD11_SELECTED_REFERENCE_HASH_MISMATCH")
    return selected


def _pose_evidence(token: str) -> tuple[dict[str, Any], Path, str]:
    path = ATTEMPT_ROOT / token / "control" / "RESULT.json"
    blob = path.read_bytes()
    return json.loads(blob), path, hashlib.sha256(blob).hexdigest()


def _gold_records(reference: dict[str, Any]) -> list[dict[str, Any]]:
    # This helper is called only after all raw scan/CAD inference has completed.
    from scripts.gold11_admission_geometry import parse_reference_endpoints

    records = []
    for bend in reference["bends"]:
        endpoints = parse_reference_endpoints(bend)["endpoints"]
        records.append({"bend_id": str(bend["bend_id"]),
                        "line_start": np.asarray(endpoints[0], dtype=np.float64).tolist(),
                        "line_end": np.asarray(endpoints[1], dtype=np.float64).tolist()})
    return records


def _empty_recovery(reason: str) -> dict[str, Any]:
    return {"schema": "CAD_GUIDED_V11_TOPOLOGY_RECOVERY_V2",
            "implementation": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING",
            "status": "UNAVAILABLE", "research_only": True,
            "gold_or_expected_count_used": False, "predictions": [], "records": [],
            "accepted_count": 0, "native_axis_count": None, "reason": reason,
            "observed_bend_count": None,
            "observed_bend_count_status": "MANUFACTURING_EVENT_IDENTITY_UNKNOWN",
            "prediction_geometry_frame": "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"}


def run(output: Path) -> dict[str, Any]:
    # Source-only bindings are checked before any raw or reference bytes are read.
    cases = _read_cases()
    from backend.cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
    from backend.cad_guided_v11_topology_recovery_v2 import recover_scan_supported_axes_v2

    case_results: list[dict[str, Any]] = []
    inference_complete = True
    for case in cases:
        token = case["case_token"]
        retained, retained_path, retained_sha = _pose_evidence(token)
        if token == "CG4BDD1371B6" and retained_sha != V11_REVIEW_RESULT_SHA256:
            raise ValueError("RETAINED_V11_CONTROL_RESULT_HASH_MISMATCH")
        registration = retained.get("registration") or {}
        evidence = registration.get("evidence") or {}
        input_bindings = {
            "scan_path": case["scan_path"], "scan_sha256": case["scan_sha256"],
            "cad_path": case["cad_path"], "cad_sha256": case["cad_sha256"],
            "gold_used_for_inference": False,
        }
        try:
            # No Gold/reference object is loaded in this section.
            # A missing pose is a legitimate abstention and does not justify
            # reading the raw bytes or manufacturing an identity.
            from backend.cad_guided_v11_topology_recovery_v2 import select_pose_v2
            pose = select_pose_v2(evidence)
            if pose.get("status") != "AVAILABLE":
                recovery = _empty_recovery(pose.get("reason", "NO_V11_POSE"))
                recovery["pose_selection"] = {key: value for key, value in pose.items()
                                               if key not in {"rotation", "translation"}}
            else:
                scan_bytes = Path(case["scan_path"]).read_bytes()
                scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes, max_engine_rows=50_000)
                if scan.get("status") != "AVAILABLE":
                    raise ValueError("V11_SCAN_PREPARATION_UNAVAILABLE")
                recovery = recover_scan_supported_axes_v2(
                    scan["raw_points"], Path(case["cad_path"]), evidence)
            inference_complete = inference_complete and recovery.get("status") in {"PARTIAL", "UNAVAILABLE"}
        except Exception as error:
            recovery = _empty_recovery("INFERENCE_EXCEPTION:" + type(error).__name__)
            inference_complete = False
        case_results.append({
            "case_token": token,
            "part_key": case["part_key"],
            "reference_id": case["reference_id"],
            "input_bindings": input_bindings,
            "retained_v11_control": {
                "path": str(retained_path.relative_to(ROOT)),
                "sha256": retained_sha,
                "expected_sha256": V11_REVIEW_RESULT_SHA256 if token == "CG4BDD1371B6" else None,
                "status": retained.get("status"),
                "phase": retained.get("phase"),
                "registration_status": registration.get("status"),
                "measurement_context_indices": evidence.get("measurement_context_indices", []),
            },
            "recovery": recovery,
        })

    # Post-inference scoring seam: Gold enters only here.
    references = _selected_references(cases)
    from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
    totals = {key: 0 for key in
              ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}
    for row in case_results:
        score = match_lines(references[row["case_token"]]["bends"],
                            row["recovery"].get("predictions", []), Thresholds())
        row["score"] = {**score, "matcher": "UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST"}
        for key in totals:
            totals[key] += int(score[key])
    denominator = 2 * totals["true_positive"] + totals["false_positive"] + totals["false_negative"]
    precision = totals["true_positive"] / totals["prediction_count"] if totals["prediction_count"] else None
    recall = totals["true_positive"] / totals["gold_count"] if totals["gold_count"] else None
    f1 = 2.0 * totals["true_positive"] / denominator if denominator else None
    result = {
        "schema": "GOLD11_V11_MULTI_SCAN_LUNA_MAX_RESULT_V1",
        "candidate_implementation": "V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING",
        "scope": {"case_count": len(cases), "family_count": len({row["part_key"] for row in cases}),
                  "reference_line_count": 71, "review_source_sha256": REVIEW_SHA256,
                  "catalog_sha256": CATALOG_SHA256,
                  "selected_reference_sha256": SELECTED_REFERENCE_SHA256},
        "inference": {"complete": inference_complete,
                       "gold_used_for_inference": False,
                       "pose_policy": {"max_provisional_p95_upper_mm": 2.0,
                                       "min_provisional_coverage_lower": 0.55},
                       "terminal_v11_reopened": False},
        "score": {**totals, "precision": precision, "recall": recall, "f1": f1,
                  "bend_count_capture": recall,
                  "target_80_percent_met": bool(recall is not None and recall >= 0.80),
                  "historical_score": 0.4580152672,
                  "historical_comparability": "UNESTABLISHED",
                  "historical_score_beaten": False,
                  "matcher": "UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST"},
        "cases": case_results,
        "production_use_allowed": False,
        "floor_release_allowed": False,
        "uncertainty_status": "UNCALIBRATED_PROVISIONAL_RETAINED_POSE_AND_SCAN_SUPPORT_DISTANCES",
        "limitations": [
            "Rejected V11 context candidates are only a provisional pose source; no verified stable datum is claimed.",
            "Native STEP axes are CAD-targeted scan-supported contact-line proposals, not measured radii or manufacturing events.",
            "Four cases without a gated finite pose remain explicit abstentions in the 71-line denominator.",
            "Consumed-development feedback is not unseen-family generalization and does not prove historical comparability.",
        ],
        "source_bindings": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "recovery_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v2.py"),
            "recovery_base_module_sha256": _sha256(ROOT / "backend/cad_guided_v11_topology_recovery_v1.py"),
            "review_source_sha256": _sha256(REVIEW_PATH),
            "catalog_sha256": _sha256(CATALOG_PATH),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checksums = {"schema": "GOLD11_V11_MULTI_SCAN_LUNA_MAX_CHECKSUMS_V1",
                 "result_sha256": _sha256(output / "RESULT.json"),
                 "source_bindings": result["source_bindings"]}
    (output / "CHECKSUMS.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({"status": "COMPUTED_RESEARCH_ONLY",
                      **{key: result["score"][key] for key in
                         ("gold_count", "prediction_count", "true_positive", "false_positive",
                          "false_negative", "precision", "recall", "f1", "target_80_percent_met")},
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
