#!/usr/bin/env python3
"""Finalize the frozen V54 retain/terminal-retire decision after visual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
TERMINAL_FAILURE_TOKEN = "V54_ENTC_TERMINAL_RETIRE"
MANDATORY_DIFFICULT = {"S011", "S016", "S024", "S029"}


def _bin_hash(additions: list[dict]) -> str:
    payload = [
        {
            "bin_turn_mass": row["bin_turn_mass"],
            "bin_compatible_mass": row["bin_compatible_mass"],
            "bin_marginal_mass": row["bin_marginal_mass"],
            "coherent_bin_mask": row["coherent_bin_mask"],
            "novel_bin_mask": row["novel_bin_mask"],
        }
        for row in additions
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = json.loads(args.score.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
    manual = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    if score.get("token") != TOKEN or execution.get("token") != TOKEN or manual.get("token") != TOKEN:
        raise ValueError("token mismatch")
    changed = set(str(value) for value in score.get("changed_scans") or ())
    exact = set(str(value) for value in score.get("v52_exact_case_ids") or ())
    required = sorted(changed | exact | MANDATORY_DIFFICULT)
    manual_by_case = {str(row["case_id"]): row for row in manual.get("cases") or ()}
    automatic_failures: list[str] = []
    manual_failures: list[str] = []
    cases = []
    for case_id in required:
        semantic = json.loads(
            (args.prediction_root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8")
        )
        additions = list(
            semantic["exclusive_normal_turn_coverage_audit"].get("recovered_candidates") or ()
        )
        metadata_path = args.visual_root / case_id / "visual_artifacts.json"
        if not metadata_path.is_file():
            automatic_failures.append(f"{case_id}:MISSING_VISUAL_METADATA")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = [
            metadata.get("raw_source_overview"),
            metadata.get("v52_only_overlay"),
            metadata.get("entc_addition_layer"),
            metadata.get("v54_composite_overlay"),
            metadata.get("entc_dashboard"),
            *(metadata.get("axis_normal_closeups") or ()),
        ]
        missing = [str(path) for path in paths if not path or not Path(path).is_file()]
        if missing:
            automatic_failures.append(f"{case_id}:MISSING_RENDER")
        dimensions = {}
        for raw_path in paths:
            if raw_path and Path(raw_path).is_file():
                with Image.open(raw_path) as image:
                    dimensions[Path(raw_path).name] = list(image.size)
                    if image.width < 3000 or image.height < 1800:
                        automatic_failures.append(f"{case_id}:NOT_NATIVE_RESOLUTION:{Path(raw_path).name}")
        if metadata.get("native_resolution") is not True:
            automatic_failures.append(f"{case_id}:NATIVE_FLAG_FALSE")
        if metadata.get("source_identifier_serialization") is not False:
            automatic_failures.append(f"{case_id}:SOURCE_IDENTIFIER_SERIALIZED")
        if metadata.get("v52_pixels_unchanged_outside_entc_layer") is not True:
            automatic_failures.append(f"{case_id}:V52_PIXEL_MUTATION")
        if int(metadata.get("entc_addition_count") or 0) != len(additions):
            automatic_failures.append(f"{case_id}:ADDITION_RENDER_COUNT")
        if len(metadata.get("axis_normal_closeups") or ()) != len(additions):
            automatic_failures.append(f"{case_id}:CLOSEUP_COUNT")
        if metadata.get("displayed_addition_bin_summary_sha256") != _bin_hash(additions):
            automatic_failures.append(f"{case_id}:DISPLAYED_BIN_MISMATCH")
        for index, addition in enumerate(additions, start=1):
            if float(addition.get("marginal_span_fraction") or 0.0) < 0.75:
                automatic_failures.append(f"{case_id}:ADDITION_{index}:MARGINAL_SPAN")
        if case_id in exact and additions:
            automatic_failures.append(f"{case_id}:ADDITION_ON_V52_EXACT")
        human = manual_by_case.get(case_id)
        if human is None or human.get("inspected_native_resolution") is not True:
            manual_failures.append(f"{case_id}:NOT_MANUALLY_INSPECTED")
        else:
            for failure in human.get("failures") or ():
                manual_failures.append(f"{case_id}:{failure}")
            human_additions = list(human.get("additions") or ())
            if len(human_additions) != len(additions):
                manual_failures.append(f"{case_id}:MANUAL_ADDITION_COUNT")
            for index, row in enumerate(human_additions, start=1):
                if row.get("positive_signature") is not True:
                    manual_failures.append(f"{case_id}:ADDITION_{index}:POSITIVE_SIGNATURE")
                for failure in row.get("failures") or ():
                    manual_failures.append(f"{case_id}:ADDITION_{index}:{failure}")
        cases.append(
            {
                "case_id": case_id,
                "addition_count": len(additions),
                "dimensions": dimensions,
                "automatic_visual_pass": not any(value.startswith(case_id + ":") for value in automatic_failures),
                "manual_visual_pass": not any(value.startswith(case_id + ":") for value in manual_failures),
            }
        )
    score_pass = score.get("status") == "SCORE_GATE_PASS_PENDING_VISUAL"
    execution_pass = (
        execution.get("status") == "READY_FOR_SINGLE_SCORE"
        and all((execution.get("semantic_replay_identity") or {}).values())
    )
    visual_pass = not automatic_failures and not manual_failures
    retained = bool(score_pass and execution_pass and visual_pass)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v54_entc_final_decision",
        "token": TOKEN,
        "status": "RETAINED" if retained else "TERMINAL_RETIRED",
        "terminal_failure_token": None if retained else TERMINAL_FAILURE_TOKEN,
        "score_gate_passed": score_pass,
        "execution_gate_passed": execution_pass,
        "visual_gate_passed": visual_pass,
        "required_case_ids": required,
        "automatic_visual_failures": automatic_failures,
        "manual_visual_failures": manual_failures,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "required_case_count": len(required), "automatic_failure_count": len(automatic_failures), "manual_failure_count": len(manual_failures)}, sort_keys=True))
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())

