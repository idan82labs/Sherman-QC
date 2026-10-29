#!/usr/bin/env python3
"""Render all 30 post-score V51 artifacts and emit a manual-audit template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.render_f1_v51_axis_normal_stack import render as render_axis_stack
from scripts.render_f1_v51_new_corpus_scored_overlay import render as render_scored_overlay


TOKEN = "V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opaque-manifest", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-template", type=Path, required=True)
    args = parser.parse_args()
    opaque = json.loads(args.opaque_manifest.read_text(encoding="utf-8"))
    if opaque.get("token") != TOKEN:
        raise ValueError("opaque manifest token mismatch")
    cases = list(opaque.get("cases") or [])
    if [row.get("case_id") for row in cases] != list(CASE_IDS):
        raise ValueError("opaque manifest must contain O001..O030 in exact order")
    truth = args.truth.resolve()
    if _sha(truth) != args.truth_sha256:
        raise ValueError("sealed truth hash mismatch")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    execution_root = args.execution_root.resolve()
    rows = []
    for case in cases:
        case_id = str(case["case_id"])
        scan = Path(str(case["scan_path"])).resolve()
        case_root = execution_root / "primary" / case_id
        prediction = case_root / "semantic_result.json"
        pool = case_root / "pool" / f"{case_id}.json"
        prediction_only = case_root / "prediction_only.png"
        for required in (scan, prediction, pool, prediction_only):
            if not required.is_file():
                raise FileNotFoundError(required)
        scored_overlay = output_root / case_id / "scored_overlay.png"
        axis_stack = output_root / case_id / "axis_normal_stack.png"
        render_scored_overlay(scan, prediction, truth, scored_overlay, case_id)
        render_axis_stack(scan, prediction, pool, axis_stack, case_id)
        prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
        rows.append(
            {
                "case_id": case_id,
                "prediction_only_path": str(prediction_only),
                "prediction_only_sha256": _sha(prediction_only),
                "scored_overlay_path": str(scored_overlay),
                "scored_overlay_sha256": _sha(scored_overlay),
                "axis_stack_path": str(axis_stack),
                "axis_stack_sha256": _sha(axis_stack),
                "render_admissible_marker_count": len(prediction_payload.get("predictions") or []),
                "reviewer": None,
                "ruling": None,
                "visual_notes": None,
            }
        )
    template = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_new_untouched_corpus_visual_audit_DRAFT",
        "token": TOKEN,
        "truth_sha256": args.truth_sha256,
        "instructions": "Inspect all three native-resolution artifacts. Fill reviewer, ruling (coherent/geometric_failure/render_failure), and visual_notes; then change kind by removing _DRAFT. Do not alter artifact paths, hashes, marker counts, truth, or predictions.",
        "cases": rows,
    }
    target = args.audit_template.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(template, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": 30, "audit_template": str(target), "output_root": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
