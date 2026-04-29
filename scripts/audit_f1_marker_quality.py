#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_point(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _distance(lhs: Tuple[float, float, float], rhs: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((lhs[index] - rhs[index]) ** 2 for index in range(3)))


def _canonical_pair(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return tuple()
    return tuple(sorted(str(item) for item in value if item is not None))


def _manifest_path_for_decomposition(path: Path) -> Optional[Path]:
    candidate = path.parent / "render" / "owned_region_manifest.json"
    return candidate if candidate.exists() else None


def _manifest_bends(manifest_path: Path) -> List[Mapping[str, Any]]:
    manifest = _load_json(manifest_path)
    bends = manifest.get("bends")
    if isinstance(bends, list):
        return [bend for bend in bends if isinstance(bend, Mapping)]
    legacy = manifest.get("owned_bend_regions") or manifest.get("bend_regions") or []
    return [bend for bend in legacy if isinstance(bend, Mapping)]


def _owned_region_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for region in decomposition.get("owned_bend_regions") or []:
        if not isinstance(region, Mapping):
            continue
        bend_id = str(region.get("bend_id") or "")
        if bend_id:
            lookup[bend_id] = region
    return lookup


def _connected_atom_count(atom_ids: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> int:
    remaining = set(str(atom_id) for atom_id in atom_ids)
    if not remaining:
        return 0
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            atom_id = stack.pop()
            for neighbor in adjacency.get(atom_id, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _build_marker_records(
    *,
    manifest_path: Path,
    decomposition: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    region_lookup = _owned_region_lookup(decomposition)
    adjacency = dict((decomposition.get("atom_graph") or {}).get("adjacency") or {})

    for bend in _manifest_bends(manifest_path):
        bend_id = str(bend.get("bend_id") or bend.get("label") or "unknown")
        metadata = dict(bend.get("metadata") or {})
        anchor = _as_point(bend.get("anchor"))
        support_points = [
            point
            for point in (_as_point(item) for item in (metadata.get("owned_support_points") or ()))
            if point is not None
        ]
        source = metadata.get("render_anchor_source")
        if source != "owned_atom_medoid":
            issues.append(
                {
                    "issue": "anchor_source_not_owned_atom_medoid",
                    "bend_id": bend_id,
                    "source": source,
                }
            )
        if anchor is None:
            issues.append({"issue": "missing_marker_anchor", "bend_id": bend_id})
            continue
        if not support_points:
            issues.append({"issue": "missing_owned_support_points", "bend_id": bend_id})
        else:
            nearest = min(_distance(anchor, point) for point in support_points)
            if nearest > 1e-6:
                issues.append(
                    {
                        "issue": "marker_anchor_not_on_owned_support",
                        "bend_id": bend_id,
                        "nearest_support_distance_mm": round(float(nearest), 6),
                    }
                )

        region = region_lookup.get(bend_id, {})
        owned_atom_ids = tuple(str(atom_id) for atom_id in (region.get("owned_atom_ids") or metadata.get("owned_atom_ids") or ()))
        if owned_atom_ids and adjacency:
            component_count = _connected_atom_count(owned_atom_ids, adjacency)
            if component_count > 1:
                issues.append(
                    {
                        "issue": "owned_region_atoms_disconnected",
                        "bend_id": bend_id,
                        "connected_component_count": int(component_count),
                    }
                )

        pair = _canonical_pair(region.get("incident_flange_ids") or metadata.get("incident_flange_ids"))
        records.append(
            {
                "bend_id": bend_id,
                "anchor": anchor,
                "pair": pair,
                "atom_count": int(len(owned_atom_ids)),
            }
        )
    return records, issues


def _cluster_issues(
    records: Sequence[Mapping[str, Any]],
    *,
    local_spacing_mm: float,
    alias_atom_count_max: int,
    tiny_atom_count_max: int,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    same_pair_close_mm = max(35.0, 6.0 * local_spacing_mm)
    tiny_cross_pair_close_mm = max(35.0, 10.0 * local_spacing_mm)
    for index, lhs in enumerate(records):
        lhs_anchor = lhs.get("anchor")
        if not isinstance(lhs_anchor, tuple):
            continue
        for rhs in records[index + 1 :]:
            rhs_anchor = rhs.get("anchor")
            if not isinstance(rhs_anchor, tuple):
                continue
            distance = _distance(lhs_anchor, rhs_anchor)
            lhs_pair = tuple(lhs.get("pair") or ())
            rhs_pair = tuple(rhs.get("pair") or ())
            min_atom_count = min(int(lhs.get("atom_count") or 0), int(rhs.get("atom_count") or 0))
            same_pair = len(lhs_pair) == 2 and lhs_pair == rhs_pair
            if same_pair and min_atom_count <= alias_atom_count_max and distance <= same_pair_close_mm:
                issues.append(
                    {
                        "issue": "selected_pair_duplicate_marker_cluster",
                        "lhs_bend_id": lhs.get("bend_id"),
                        "rhs_bend_id": rhs.get("bend_id"),
                        "pair": list(lhs_pair),
                        "centroid_distance_mm": round(float(distance), 4),
                        "min_atom_count": int(min_atom_count),
                        "threshold_mm": round(float(same_pair_close_mm), 4),
                    }
                )
            elif (
                not same_pair
                and min_atom_count <= tiny_atom_count_max
                and distance <= tiny_cross_pair_close_mm
            ):
                issues.append(
                    {
                        "issue": "tiny_cross_pair_marker_cluster",
                        "lhs_bend_id": lhs.get("bend_id"),
                        "rhs_bend_id": rhs.get("bend_id"),
                        "lhs_pair": list(lhs_pair),
                        "rhs_pair": list(rhs_pair),
                        "centroid_distance_mm": round(float(distance), 4),
                        "min_atom_count": int(min_atom_count),
                        "threshold_mm": round(float(tiny_cross_pair_close_mm), 4),
                    }
                )
    return issues


def _audit_decomposition_result(
    path: Path,
    *,
    alias_atom_count_max: int,
    tiny_atom_count_max: int,
) -> Dict[str, Any]:
    decomposition = _load_json(path)
    manifest_path = _manifest_path_for_decomposition(path)
    issues: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    if manifest_path is None:
        issues.append({"issue": "missing_owned_region_manifest", "path": str(path.parent / "render" / "owned_region_manifest.json")})
    else:
        records, issues = _build_marker_records(manifest_path=manifest_path, decomposition=decomposition)
        local_spacing_mm = float((decomposition.get("atom_graph") or {}).get("local_spacing_mm") or 1.0)
        issues.extend(
            _cluster_issues(
                records,
                local_spacing_mm=max(local_spacing_mm, 1.0),
                alias_atom_count_max=alias_atom_count_max,
                tiny_atom_count_max=tiny_atom_count_max,
            )
        )
    summary_path = path.parent / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        candidate = dict(summary.get("candidate") or {})
        candidate_exact = candidate.get("exact_bend_count")
        candidate_range = list(candidate.get("bend_count_range") or ())
        candidate_source = str(candidate.get("candidate_source") or "")
        render_count = candidate.get("render_owned_bend_region_count")
        accepted_render_marker_count = candidate.get("accepted_render_marker_count")
        accepted_render_semantics = str(candidate.get("accepted_render_semantics") or "")
        singleton_candidate = (
            candidate_exact is not None
            and len(candidate_range) >= 2
            and int(candidate_range[0]) == int(candidate_range[1]) == int(candidate_exact)
        )
        scan_context_only = accepted_render_semantics.startswith("scan_context_only") and int(accepted_render_marker_count or 0) == 0
        if singleton_candidate and render_count is not None and int(render_count) != int(candidate_exact) and not scan_context_only:
            issues.append(
                {
                    "issue": "accepted_exact_not_rendered_by_owned_regions",
                    "candidate_source": candidate_source,
                    "accepted_exact_bend_count": int(candidate_exact),
                    "render_owned_bend_region_count": int(render_count),
                    "summary_json": str(summary_path),
                }
            )

    return {
        "part_id": decomposition.get("part_id") or path.parent.name,
        "decomposition_result_json": str(path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "owned_region_count": len(decomposition.get("owned_bend_regions") or []),
        "marker_count": len(records),
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def _discover_decomposition_results(paths: Sequence[Path]) -> List[Path]:
    discovered: List[Path] = []
    for path in paths:
        if path.is_file() and path.name == "decomposition_result.json":
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(path.rglob("decomposition_result.json")))
    return sorted(dict.fromkeys(discovered))


def audit_marker_quality(
    *,
    paths: Sequence[Path],
    alias_atom_count_max: int = 15,
    tiny_atom_count_max: int = 5,
) -> Dict[str, Any]:
    decomposition_paths = _discover_decomposition_results(paths)
    cases = [
        _audit_decomposition_result(
            path,
            alias_atom_count_max=alias_atom_count_max,
            tiny_atom_count_max=tiny_atom_count_max,
        )
        for path in decomposition_paths
    ]
    issues = [
        {**issue, "part_id": case.get("part_id"), "decomposition_result_json": case.get("decomposition_result_json")}
        for case in cases
        for issue in case.get("issues", ())
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decomposition_result_count": len(decomposition_paths),
        "case_count": len(cases),
        "issue_count": len(issues),
        "pass_count": sum(1 for case in cases if case.get("status") == "pass"),
        "fail_count": sum(1 for case in cases if case.get("status") == "fail"),
        "cases": cases,
        "issues": issues,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit F1 render marker quality from prototype decomposition outputs.")
    parser.add_argument("--prototype-output-dir", action="append", type=Path, default=[])
    parser.add_argument("--decomposition-result-json", action="append", type=Path, default=[])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--alias-atom-count-max", type=int, default=15)
    parser.add_argument("--tiny-atom-count-max", type=int, default=5)
    args = parser.parse_args(argv)
    paths = list(args.prototype_output_dir or ()) + list(args.decomposition_result_json or ())
    if not paths:
        parser.error("provide at least one --prototype-output-dir or --decomposition-result-json")

    report = audit_marker_quality(
        paths=paths,
        alias_atom_count_max=args.alias_atom_count_max,
        tiny_atom_count_max=args.tiny_atom_count_max,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "case_count": report["case_count"],
                "issue_count": report["issue_count"],
                "pass_count": report["pass_count"],
            },
            indent=2,
        )
    )
    return 0 if int(report["issue_count"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
