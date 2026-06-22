#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_singleton_range(count_range: Sequence[Any]) -> bool:
    if len(count_range) < 2:
        return False
    try:
        return int(count_range[0]) == int(count_range[1])
    except (TypeError, ValueError):
        return False


def _is_final_exact(algorithm: Mapping[str, Any]) -> bool:
    exact = algorithm.get("accepted_exact_bend_count")
    count_range = list(algorithm.get("accepted_bend_count_range") or ())
    if exact is None or not _is_singleton_range(count_range):
        return False
    try:
        return int(exact) == int(count_range[0])
    except (TypeError, ValueError):
        return False


def _path_exists(value: Any) -> bool:
    return bool(value) and Path(str(value)).exists()


def _image_size(path: Path) -> Optional[List[int]]:
    try:
        from PIL import Image
    except Exception:
        return None
    with Image.open(path) as image:
        return [int(image.size[0]), int(image.size[1])]


def _manifest_bends(manifest_path: Path) -> List[Mapping[str, Any]]:
    manifest = _load_json(manifest_path)
    bends = manifest.get("bends")
    if isinstance(bends, list):
        return [bend for bend in bends if isinstance(bend, Mapping)]
    legacy = manifest.get("owned_bend_regions") or manifest.get("bend_regions") or []
    return [bend for bend in legacy if isinstance(bend, Mapping)]


def _audit_manifest_anchors(manifest_path: Path, *, tolerance_mm: float) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for bend in _manifest_bends(manifest_path):
        metadata = dict(bend.get("metadata") or {})
        bend_id = str(bend.get("bend_id") or bend.get("label") or "unknown")
        anchor = bend.get("anchor")
        support_points = metadata.get("owned_support_points") or []
        source = metadata.get("render_anchor_source")
        if source != "owned_atom_medoid":
            issues.append(
                {
                    "issue": "anchor_source_not_owned_atom_medoid",
                    "manifest_path": str(manifest_path),
                    "bend_id": bend_id,
                    "source": source,
                }
            )
        if not isinstance(anchor, list) or len(anchor) != 3 or not support_points:
            issues.append(
                {
                    "issue": "missing_anchor_or_support_points",
                    "manifest_path": str(manifest_path),
                    "bend_id": bend_id,
                }
            )
            continue
        try:
            nearest = min(math.dist(anchor, point) for point in support_points if isinstance(point, list) and len(point) == 3)
        except ValueError:
            issues.append(
                {
                    "issue": "invalid_support_points",
                    "manifest_path": str(manifest_path),
                    "bend_id": bend_id,
                }
            )
            continue
        if nearest > tolerance_mm:
            issues.append(
                {
                    "issue": "anchor_not_on_owned_support_point",
                    "manifest_path": str(manifest_path),
                    "bend_id": bend_id,
                    "nearest_support_distance_mm": nearest,
                    "tolerance_mm": tolerance_mm,
                }
            )
    return issues


def _audit_result(
    result: Mapping[str, Any],
    *,
    require_1080p: bool,
    audit_debug_manifests: bool,
    anchor_tolerance_mm: float,
) -> Dict[str, Any]:
    algorithm = dict(result.get("algorithm") or {})
    render = dict(result.get("render_artifacts") or {})
    accepted_range = list(algorithm.get("accepted_bend_count_range") or ())
    accepted_exact = algorithm.get("accepted_exact_bend_count")
    final_exact = _is_final_exact(algorithm)
    contact_sheet = render.get("contact_sheet_1080p_path")
    manifest_path = render.get("manifest_path")
    scan_context_path = render.get("scan_context_path")

    issues: List[Dict[str, Any]] = []
    if not _path_exists(contact_sheet):
        issues.append({"issue": "missing_contact_sheet_1080p", "path": contact_sheet})
    elif require_1080p:
        size = _image_size(Path(str(contact_sheet)))
        if size is not None and size != [1920, 1080]:
            issues.append({"issue": "contact_sheet_not_1080p", "path": contact_sheet, "size": size})

    if final_exact:
        if not _path_exists(manifest_path):
            issues.append({"issue": "missing_final_manifest", "path": manifest_path})
        else:
            bends = _manifest_bends(Path(str(manifest_path)))
            if len(bends) != int(accepted_exact):
                issues.append(
                    {
                        "issue": "final_manifest_bend_count_mismatch",
                        "path": manifest_path,
                        "expected": int(accepted_exact),
                        "actual": len(bends),
                    }
                )
            issues.extend(_audit_manifest_anchors(Path(str(manifest_path)), tolerance_mm=anchor_tolerance_mm))
    else:
        if not _path_exists(scan_context_path):
            issues.append({"issue": "missing_scan_context_for_nonfinal_range", "path": scan_context_path})
        if audit_debug_manifests and _path_exists(manifest_path):
            issues.extend(_audit_manifest_anchors(Path(str(manifest_path)), tolerance_mm=anchor_tolerance_mm))

    return {
        "track": result.get("track"),
        "part_key": result.get("part_key"),
        "scan_name": Path(str(result.get("scan_path") or "")).name,
        "accepted_range": accepted_range,
        "accepted_exact": accepted_exact,
        "final_exact": final_exact,
        "contact_sheet_1080p_path": contact_sheet,
        "manifest_path": manifest_path,
        "scan_context_path": scan_context_path,
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def audit_visual_review_artifacts(
    *,
    summary_paths: Sequence[Path],
    require_1080p: bool = True,
    audit_debug_manifests: bool = True,
    anchor_tolerance_mm: float = 1e-6,
) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        for result in list(summary.get("results") or ()):
            case = _audit_result(
                result,
                require_1080p=require_1080p,
                audit_debug_manifests=audit_debug_manifests,
                anchor_tolerance_mm=anchor_tolerance_mm,
            )
            case["source_summary_json"] = str(summary_path)
            cases.append(case)
    issues = [
        {**issue, "part_key": case.get("part_key"), "scan_name": case.get("scan_name")}
        for case in cases
        for issue in case.get("issues", [])
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_paths": [str(path) for path in summary_paths],
        "case_count": len(cases),
        "issue_count": len(issues),
        "final_exact_case_count": sum(1 for case in cases if case.get("final_exact")),
        "range_or_review_case_count": sum(1 for case in cases if not case.get("final_exact")),
        "cases": cases,
        "issues": issues,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit F1 visual review render artifacts.")
    parser.add_argument("--validation-summary-json", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--skip-1080p-check", action="store_true")
    parser.add_argument("--skip-debug-manifest-anchor-check", action="store_true")
    parser.add_argument("--anchor-tolerance-mm", type=float, default=1e-6)
    args = parser.parse_args(argv)

    report = audit_visual_review_artifacts(
        summary_paths=args.validation_summary_json,
        require_1080p=not args.skip_1080p_check,
        audit_debug_manifests=not args.skip_debug_manifest_anchor_check,
        anchor_tolerance_mm=args.anchor_tolerance_mm,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "case_count": report["case_count"],
                "issue_count": report["issue_count"],
                "final_exact_case_count": report["final_exact_case_count"],
            },
            indent=2,
        )
    )
    return 0 if int(report["issue_count"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
