#!/usr/bin/env python3
"""Materialize identity-free zero-shot artifacts from isolated F1 decompositions.

External inference batches already carry the zero-shot result inside each
patch-graph decomposition.  The v14-v18 research stages consume the same data
through the historical zero-summary/overlay interface.  This bridge preserves
that interface without rerunning the detector or exposing intake metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover(prediction_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for case_dir in sorted((prediction_root / "isolated_cases").glob("[0-9][0-9]_*")):
        token = case_dir.name.split("_", 1)[1]
        matches = sorted(case_dir.glob("result/**/decomposition_result.json"))
        if len(matches) != 1:
            raise ValueError(f"{case_dir.name}: expected one decomposition result, found {len(matches)}")
        rows.append((token, matches[0]))
    if not rows:
        raise ValueError(f"no isolated decomposition results under {prediction_root}")
    return rows


def _overlay_manifest(zero: dict[str, Any]) -> dict[str, Any]:
    bends = []
    for index, bend in enumerate(zero.get("renderable_bend_objects") or (), start=1):
        detected = dict(bend.get("detected_line") or bend.get("cad_line") or {})
        if detected.get("display_start") is None or detected.get("display_end") is None:
            continue
        bends.append(
            {
                "bend_id": str(bend.get("bend_id") or f"ZS{index}"),
                "detected_line": detected,
                "cad_line": dict(bend.get("cad_line") or detected),
            }
        )
    return {"schema_version": 1, "bends": bends}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sanitized-prediction-root", required=True, type=Path)
    args = parser.parse_args()

    result_map: dict[str, dict[str, str]] = {}
    for index, (token, decomposition_path) in enumerate(_discover(args.prediction_root.resolve()), start=1):
        decomposition = json.loads(decomposition_path.read_text(encoding="utf-8"))
        exposed_part_id = decomposition.get("part_id")
        if exposed_part_id not in {None, token}:
            raise ValueError(f"{token}: decomposition exposed non-opaque part identity")
        zero = dict(decomposition.get("current_zero_shot_result") or {})
        if not zero:
            raise ValueError(f"{token}: decomposition has no current_zero_shot_result")
        zero["part_id"] = None
        decomposition["part_id"] = None
        decomposition["current_zero_shot_result"] = zero
        sanitized_path = (
            args.sanitized_prediction_root.resolve()
            / "isolated_cases"
            / f"{index:02d}_{token}"
            / "result"
            / token
            / "decomposition_result.json"
        )
        _write_json(sanitized_path, decomposition)
        case_dir = args.output_dir.resolve() / token
        zero_path = case_dir / "zero_shot_result.json"
        overlay_path = case_dir / "zero_shot_bend_overlay_manifest.json"
        _write_json(zero_path, zero)
        _write_json(overlay_path, _overlay_manifest(zero))
        result_map[token] = {
            "zero_shot_result": str(zero_path),
            "overlay_manifest": str(overlay_path),
        }

    summary = {
        "schema_version": 1,
        "inference_contract": {
            "inference_receives_truth": False,
            "inference_receives_expected_count": False,
            "inference_receives_part_identifier": False,
        },
        "result_map": result_map,
    }
    summary_path = args.output_dir.resolve() / "inference_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({"case_count": len(result_map), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
