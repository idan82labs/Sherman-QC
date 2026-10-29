#!/usr/bin/env python3
"""Seal the V56 retain/terminal-retire decision after manual visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
TERMINAL_TOKEN = "V56_CTMLR_TERMINAL_RETIRE"
VISUAL_KIND = "sherman_qc_f1_v56_ctmlr_source_only_visual_manifest"
MANUAL_KIND = "sherman_qc_f1_v56_ctmlr_signed_manual_visual_audit"
REQUIRED_EXACT = {"S005", "S006", "S007", "S013", "S026", "S027"}
REQUIRED_CONTROL = {"S011", "S016", "S024", "S029"}
EXPECTED_DIMENSIONS = (4096, 2304)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = json.loads(args.score.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
    visual = json.loads(args.visual_manifest.read_text(encoding="utf-8"))
    manual = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    if any(payload.get("token") != TOKEN for payload in (score, execution, visual, manual)):
        raise ValueError("token mismatch")
    if visual.get("kind") != VISUAL_KIND or visual.get("truth_opened") is not False:
        raise ValueError("invalid source-only visual manifest")
    if manual.get("kind") != MANUAL_KIND or manual.get("signed") is not True:
        raise ValueError("manual visual audit is not signed")

    changed = set(str(value) for value in visual.get("changed_cases") or ())
    rendered = set(str(value) for value in visual.get("rendered_cases") or ())
    required = changed | REQUIRED_EXACT | REQUIRED_CONTROL
    automatic_failures: list[str] = []
    manual_failures: list[str] = []
    if rendered != required:
        automatic_failures.append("RENDERED_CASE_SET_MISMATCH")
    if visual.get("source_point_serialization") is not False:
        automatic_failures.append("SOURCE_POINT_SERIALIZATION")
    if tuple(visual.get("pixel_dimensions") or ()) != EXPECTED_DIMENSIONS:
        automatic_failures.append("MANIFEST_DIMENSIONS")

    record_by_case = {str(row["case_id"]): row for row in visual.get("records") or ()}
    manual_by_case = {str(row["case_id"]): row for row in manual.get("cases") or ()}
    case_reports = []
    for case_id in sorted(required):
        record = record_by_case.get(case_id)
        if record is None:
            automatic_failures.append(f"{case_id}:MISSING_RECORD")
            continue
        overview = Path(record["overview"])
        if not overview.is_file() or _sha(overview) != record.get("overview_sha256"):
            automatic_failures.append(f"{case_id}:OVERVIEW_HASH_OR_PATH")
        else:
            with Image.open(overview) as image:
                if image.size != EXPECTED_DIMENSIONS:
                    automatic_failures.append(f"{case_id}:OVERVIEW_DIMENSIONS")
        if record.get("source_point_serialization") is not False:
            automatic_failures.append(f"{case_id}:SOURCE_POINT_SERIALIZATION")
        lines = list(record.get("lines") or ())
        if int(record.get("diagnostic_line_count", -1)) != len(lines):
            automatic_failures.append(f"{case_id}:DIAGNOSTIC_COUNT")
        line_keys: list[str] = []
        for line in lines:
            key = str(line["key"])
            line_keys.append(key)
            path = Path(line["render"])
            if not path.is_file() or _sha(path) != line.get("render_sha256"):
                automatic_failures.append(f"{case_id}:{key}:RENDER_HASH_OR_PATH")
                continue
            with Image.open(path) as image:
                if image.size != EXPECTED_DIMENSIONS:
                    automatic_failures.append(f"{case_id}:{key}:DIMENSIONS")
            stations = list(line.get("station_records") or ())
            if len(stations) != 7 or [row.get("station_index") for row in stations] != list(range(7)):
                automatic_failures.append(f"{case_id}:{key}:SEVEN_STATIONS")

        human = manual_by_case.get(case_id)
        if human is None:
            manual_failures.append(f"{case_id}:MISSING_MANUAL_REVIEW")
        else:
            if human.get("inspected_native_resolution") is not True:
                manual_failures.append(f"{case_id}:NOT_NATIVE_INSPECTION")
            if human.get("overview_alignment") not in ("COHERENT", "NO_LINES_COHERENT"):
                manual_failures.append(f"{case_id}:OVERVIEW_ALIGNMENT")
            for failure in human.get("failures") or ():
                manual_failures.append(f"{case_id}:{failure}")
            human_lines = {str(row["key"]): row for row in human.get("lines") or ()}
            if set(human_lines) != set(line_keys):
                manual_failures.append(f"{case_id}:MANUAL_LINE_SET")
            for key in line_keys:
                row = human_lines.get(key)
                if row is None:
                    continue
                if row.get("section_geometry") != "COHERENT":
                    manual_failures.append(f"{case_id}:{key}:SECTION_GEOMETRY")
                if row.get("station_markers_aligned") is not True:
                    manual_failures.append(f"{case_id}:{key}:STATION_MARKERS")
                if row.get("decision_visually_supported") is not True:
                    manual_failures.append(f"{case_id}:{key}:DECISION_SUPPORT")
                for failure in row.get("failures") or ():
                    manual_failures.append(f"{case_id}:{key}:{failure}")
        case_reports.append(
            {
                "case_id": case_id,
                "changed": case_id in changed,
                "diagnostic_line_count": len(lines),
                "automatic_visual_pass": not any(value.startswith(case_id + ":") for value in automatic_failures),
                "manual_visual_pass": not any(value.startswith(case_id + ":") for value in manual_failures),
            }
        )

    if set(manual_by_case) != required:
        manual_failures.append("MANUAL_CASE_SET_MISMATCH")
    execution_pass = bool(
        execution.get("status") == "READY_FOR_SINGLE_SCORE"
        and execution.get("completed_execution_count") == 60
        and all((execution.get("semantic_replay_identity") or {}).values())
        and float(execution.get("p95_case_wall_seconds", float("inf"))) <= 300.0
        and int(execution.get("peak_process_rss_bytes", 16 * 1024**3)) < 16 * 1024**3
    )
    numerical_pass = score.get("retention_gates", {}).get("numerical_and_operational_pass") is True
    visual_pass = not automatic_failures and not manual_failures
    retained = bool(execution_pass and numerical_pass and visual_pass)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_ctmlr_final_decision",
        "token": TOKEN,
        "status": "RETAINED_RESEARCH_BASELINE" if retained else "TERMINAL_RETIRED",
        "terminal_token": None if retained else TERMINAL_TOKEN,
        "execution_gate_passed": execution_pass,
        "numerical_retention_gate_passed": numerical_pass,
        "visual_gate_passed": visual_pass,
        "required_case_ids": sorted(required),
        "automatic_visual_failures": automatic_failures,
        "manual_visual_failures": manual_failures,
        "visual_manifest_sha256": _sha(args.visual_manifest),
        "manual_audit_sha256": _sha(args.manual_audit),
        "score_sha256": _sha(args.score),
        "cases": case_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    print(json.dumps({"status": report["status"], "required_cases": len(required), "automatic_failures": len(automatic_failures), "manual_failures": len(manual_failures)}, sort_keys=True))
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())
