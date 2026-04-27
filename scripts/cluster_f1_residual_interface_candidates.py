#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _vec(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
    }


def _candidate_atoms(diagnostics: Mapping[str, Any]) -> List[str]:
    atoms: set[str] = set()
    for candidate in diagnostics.get("candidates") or ():
        if not candidate.get("admissible"):
            continue
        atoms.update(str(value) for value in candidate.get("atom_ids") or ())
    return sorted(atoms)


def _pca_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if len(points) < 2:
        return centroid, np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    return centroid, axis


def _connected_components(atom_ids: Iterable[str], adjacency: Mapping[str, Sequence[str]]) -> List[List[str]]:
    remaining = set(str(value) for value in atom_ids)
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()) or ():
                neighbor_id = str(neighbor)
                if neighbor_id not in remaining:
                    continue
                remaining.remove(neighbor_id)
                stack.append(neighbor_id)
                component.append(neighbor_id)
        components.append(sorted(component))
    return components


def _projection_gap_split(atom_ids: Sequence[str], atoms: Mapping[str, Mapping[str, Any]]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    centroid, axis = _pca_axis(points)
    projections = [(atom_id, float(np.dot(_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) - centroid, axis))) for atom_id in atom_ids]
    projections.sort(key=lambda item: item[1])
    gaps = [projections[index + 1][1] - projections[index][1] for index in range(len(projections) - 1)]
    if not gaps:
        return list(atom_ids), [], {"split": False, "reason": "no_projection_gaps"}
    max_index = max(range(len(gaps)), key=lambda index: gaps[index])
    max_gap = float(gaps[max_index])
    median_gap = float(np.median(gaps)) if gaps else 0.0
    left = [atom_id for atom_id, _ in projections[: max_index + 1]]
    right = [atom_id for atom_id, _ in projections[max_index + 1 :]]
    span = float(projections[-1][1] - projections[0][1]) if projections else 0.0
    split = bool(min(len(left), len(right)) >= 4 and max_gap >= 8.0 and max_gap / max(median_gap, 1e-6) >= 2.5)
    return left, right, {
        "split": split,
        "max_gap_mm": round(max_gap, 6),
        "median_gap_mm": round(median_gap, 6),
        "gap_ratio": round(max_gap / max(median_gap, 1e-6), 6),
        "span_mm": round(span, 6),
        "left_count": len(left),
        "right_count": len(right),
        "axis": [round(float(value), 6) for value in axis.tolist()],
    }


def _recursive_split(atom_ids: Sequence[str], atoms: Mapping[str, Mapping[str, Any]], depth: int = 0, max_depth: int = 3) -> List[Dict[str, Any]]:
    if depth >= max_depth or len(atom_ids) < 10:
        return [{"atom_ids": sorted(atom_ids), "split_depth": depth, "split_trace": []}]
    left, right, decision = _projection_gap_split(atom_ids, atoms)
    if not decision.get("split") or not right:
        return [{"atom_ids": sorted(atom_ids), "split_depth": depth, "split_trace": [decision]}]
    clusters: List[Dict[str, Any]] = []
    for child in (left, right):
        for row in _recursive_split(child, atoms, depth + 1, max_depth=max_depth):
            row["split_trace"] = [decision] + list(row.get("split_trace") or [])
            clusters.append(row)
    return clusters


def _cluster_summary(cluster_id: str, atom_ids: Sequence[str], atoms: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    centroid, axis = _pca_axis(points)
    projections = [float(np.dot(point - centroid, axis)) for point in points]
    span = max(projections) - min(projections) if projections else 0.0
    weights = [float((atoms.get(atom_id) or {}).get("weight") or 0.0) for atom_id in atom_ids]
    bendness_values = [min(1.0, float((atoms.get(atom_id) or {}).get("curvature_proxy") or 0.0) * 20.0) for atom_id in atom_ids]
    return {
        "cluster_id": cluster_id,
        "atom_ids": sorted(atom_ids),
        "atom_count": len(atom_ids),
        "support_mass": round(float(sum(weights)), 6),
        "support_centroid": [round(float(value), 6) for value in centroid.tolist()],
        "axis_direction": [round(float(value), 6) for value in axis.tolist()],
        "span_mm": round(float(span), 6),
        "support_bendness": round(float(sum(bendness_values) / max(1, len(bendness_values))), 6),
    }


def cluster_residual_interface_candidates(
    *,
    decomposition: Mapping[str, Any],
    residual_interface_diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    atom_ids = [atom_id for atom_id in _candidate_atoms(residual_interface_diagnostics) if atom_id in atoms]
    adjacency = {
        str(key): [str(value) for value in values or ()]
        for key, values in ((decomposition.get("atom_graph") or {}).get("adjacency") or {}).items()
    }
    components = _connected_components(atom_ids, adjacency)
    clusters: List[Dict[str, Any]] = []
    for component_index, component in enumerate(sorted(components, key=lambda values: (-len(values), values[0])), start=1):
        for sub_index, row in enumerate(_recursive_split(component, atoms), start=1):
            summary = _cluster_summary(f"RICC{component_index}_{sub_index}", row["atom_ids"], atoms)
            summary["parent_component_id"] = f"RICC{component_index}"
            summary["split_depth"] = row.get("split_depth")
            summary["split_trace"] = row.get("split_trace")
            clusters.append(summary)
    clusters.sort(key=lambda item: (-int(item["atom_count"]), str(item["cluster_id"])))
    conflict_count = int(residual_interface_diagnostics.get("admissible_conflict_count") or 0)
    recovery_deficit = int(residual_interface_diagnostics.get("recovery_deficit") or 0)
    plausible = [
        item
        for item in clusters
        if int(item.get("atom_count") or 0) >= 8 and float(item.get("span_mm") or 0.0) >= 12.0
    ]
    status = "no_candidate_atoms"
    if plausible and len(plausible) >= recovery_deficit > 0 and conflict_count == 0:
        status = "subclusters_cover_deficit"
    elif plausible and len(plausible) >= recovery_deficit > 0:
        status = "subclusters_exist_but_parent_candidates_conflict"
    elif plausible:
        status = "partial_subcluster_signal"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only sub-clustering of ambiguous residual/interface support candidates.",
        "source_status": residual_interface_diagnostics.get("status"),
        "recovery_deficit": recovery_deficit,
        "source_admissible_candidate_count": residual_interface_diagnostics.get("admissible_candidate_count"),
        "source_admissible_conflict_count": conflict_count,
        "candidate_atom_count": len(atom_ids),
        "connected_component_count": len(components),
        "cluster_count": len(clusters),
        "plausible_cluster_count": len(plausible),
        "status": status,
        "clusters": clusters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sub-cluster ambiguous residual/interface F1 candidate support.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--residual-interface-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    report = cluster_residual_interface_candidates(
        decomposition=_load_json(args.decomposition_json),
        residual_interface_diagnostics=_load_json(args.residual_interface_json),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "candidate_atom_count": report["candidate_atom_count"],
                "connected_component_count": report["connected_component_count"],
                "cluster_count": report["cluster_count"],
                "plausible_cluster_count": report["plausible_cluster_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
