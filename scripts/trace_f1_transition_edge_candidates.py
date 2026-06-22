#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _vec(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _unit(values: Sequence[Any]) -> np.ndarray:
    vector = _vec(values)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _axis_angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    dot = abs(float(np.dot(_unit(lhs), _unit(rhs))))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
    }


def _adjacency(decomposition: Mapping[str, Any]) -> Dict[str, List[str]]:
    return {
        str(atom_id): [str(value) for value in neighbors or ()]
        for atom_id, neighbors in ((decomposition.get("atom_graph") or {}).get("adjacency") or {}).items()
    }


def _assignment(decomposition: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    assignment = decomposition.get("assignment") or {}
    return (
        {str(key): str(value) for key, value in (assignment.get("atom_labels") or {}).items()},
        {str(key): str(value) for key, value in (assignment.get("label_types") or {}).items()},
    )


def _excluded_atoms(
    decomposition: Mapping[str, Any],
    *,
    exclude_owned_bend_atoms: bool,
    excluded_atom_ids: Optional[Iterable[str]],
) -> set[str]:
    excluded = {str(value) for value in excluded_atom_ids or ()}
    if exclude_owned_bend_atoms:
        excluded.update(
            str(atom_id)
            for region in decomposition.get("owned_bend_regions") or ()
            for atom_id in region.get("owned_atom_ids") or ()
        )
    return excluded


def _connected_edge_components(edges: Sequence[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    atom_to_edges: Dict[str, List[int]] = {}
    for index, edge in enumerate(edges):
        for atom_id in edge:
            atom_to_edges.setdefault(atom_id, []).append(index)

    seen: set[int] = set()
    components: List[List[Tuple[str, str]]] = []
    for start_index in range(len(edges)):
        if start_index in seen:
            continue
        stack = [start_index]
        seen.add(start_index)
        component_indexes: List[int] = []
        while stack:
            edge_index = stack.pop()
            component_indexes.append(edge_index)
            for atom_id in edges[edge_index]:
                for neighbor_edge_index in atom_to_edges.get(atom_id, ()):
                    if neighbor_edge_index not in seen:
                        seen.add(neighbor_edge_index)
                        stack.append(neighbor_edge_index)
        components.append([edges[index] for index in sorted(component_indexes)])
    components.sort(key=lambda component: (-len(component), component[0]))
    return components


def _line_stats(points: np.ndarray) -> Dict[str, float]:
    if len(points) == 0:
        return {"span_mm": 0.0, "thickness_mm": 0.0, "linearity": 0.0}
    if len(points) == 1:
        return {"span_mm": 0.0, "thickness_mm": 0.0, "linearity": 1.0}
    centered = points - np.mean(points, axis=0)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axis = eigenvectors[:, order[0]]
    projections = centered @ axis
    span = float(np.max(projections) - np.min(projections))
    distances = np.linalg.norm(centered - np.outer(projections, axis), axis=1)
    total = float(np.sum(np.maximum(eigenvalues, 0.0)))
    linearity = float(max(eigenvalues[0], 0.0) / total) if total > 1e-9 else 0.0
    return {
        "span_mm": span,
        "thickness_mm": float(np.percentile(distances, 80)) if len(distances) else 0.0,
        "linearity": linearity,
    }


def _edge_key(left: str, right: str) -> str:
    return "::".join(sorted((str(left), str(right))))


def trace_transition_edge_candidates(
    *,
    decomposition: Mapping[str, Any],
    normal_jump_deg: float = 28.0,
    min_edges: int = 4,
    min_span_mm: float = 18.0,
    max_thickness_mm: float = 28.0,
    exclude_owned_bend_atoms: bool = False,
    excluded_atom_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    adjacency = _adjacency(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    excluded = _excluded_atoms(
        decomposition,
        exclude_owned_bend_atoms=exclude_owned_bend_atoms,
        excluded_atom_ids=excluded_atom_ids,
    )

    selected_edges: List[Tuple[str, str]] = []
    edge_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for left, neighbors in adjacency.items():
        if left not in atoms or left in excluded:
            continue
        for right in neighbors:
            if right not in atoms or right in excluded or left >= right:
                continue
            left_atom = atoms[left]
            right_atom = atoms[right]
            normal_jump = _axis_angle_deg(left_atom.get("normal") or (1, 0, 0), right_atom.get("normal") or (1, 0, 0))
            if normal_jump < normal_jump_deg:
                continue
            edge = (left, right)
            selected_edges.append(edge)
            midpoint = (_vec(left_atom.get("centroid") or (0, 0, 0)) + _vec(right_atom.get("centroid") or (0, 0, 0))) * 0.5
            edge_rows[edge] = {
                "edge_id": _edge_key(left, right),
                "atom_ids": [left, right],
                "midpoint": [float(value) for value in midpoint.tolist()],
                "normal_jump_deg": float(normal_jump),
                "left_label": atom_labels.get(left),
                "right_label": atom_labels.get(right),
                "left_label_type": label_types.get(atom_labels.get(left, "")),
                "right_label_type": label_types.get(atom_labels.get(right, "")),
            }

    components = _connected_edge_components(selected_edges)
    candidates: List[Dict[str, Any]] = []
    for index, component_edges in enumerate(components, start=1):
        component_atoms = sorted({atom_id for edge in component_edges for atom_id in edge})
        points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in component_atoms], dtype=np.float64)
        stats = _line_stats(points)
        jumps = [float(edge_rows[edge]["normal_jump_deg"]) for edge in component_edges]
        label_mix: Dict[str, int] = {}
        for atom_id in component_atoms:
            label = atom_labels.get(atom_id, "unassigned")
            label_mix[label] = label_mix.get(label, 0) + 1
        admissible = (
            len(component_edges) >= min_edges
            and stats["span_mm"] >= min_span_mm
            and stats["thickness_mm"] <= max_thickness_mm
        )
        candidates.append(
            {
                "candidate_id": f"TEC{index}",
                "atom_ids": component_atoms,
                "edge_ids": [_edge_key(*edge) for edge in component_edges],
                "edge_count": len(component_edges),
                "atom_count": len(component_atoms),
                "mean_normal_jump_deg": float(np.mean(jumps)) if jumps else 0.0,
                "max_normal_jump_deg": float(np.max(jumps)) if jumps else 0.0,
                "span_mm": stats["span_mm"],
                "thickness_mm": stats["thickness_mm"],
                "linearity": stats["linearity"],
                "label_mix": label_mix,
                "admissible": bool(admissible),
                "admissibility_reasons": []
                if admissible
                else [
                    reason
                    for reason, failed in (
                        ("too_few_edges", len(component_edges) < min_edges),
                        ("span_too_short", stats["span_mm"] < min_span_mm),
                        ("too_thick_for_line_candidate", stats["thickness_mm"] > max_thickness_mm),
                    )
                    if failed
                ],
            }
        )

    candidates.sort(
        key=lambda row: (
            not bool(row["admissible"]),
            -float(row["span_mm"]),
            -float(row["mean_normal_jump_deg"]),
            str(row["candidate_id"]),
        )
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only thin atom-edge transition tracing for missing F1 bend support.",
        "normal_jump_deg": float(normal_jump_deg),
        "min_edges": int(min_edges),
        "min_span_mm": float(min_span_mm),
        "max_thickness_mm": float(max_thickness_mm),
        "exclude_owned_bend_atoms": bool(exclude_owned_bend_atoms),
        "explicitly_excluded_atom_count": len({str(value) for value in excluded_atom_ids or ()}),
        "selected_edge_count": len(selected_edges),
        "candidate_count": len(candidates),
        "admissible_candidate_count": sum(1 for candidate in candidates if candidate["admissible"]),
        "status": "edge_candidates_found" if any(candidate["admissible"] for candidate in candidates) else "no_edge_candidate",
        "candidates": candidates,
    }


def _excluded_ids_from_json(path: Optional[Path]) -> set[str]:
    if not path:
        return set()
    payload = _load_json(path)
    excluded: set[str] = set()
    for key in ("atom_ids", "excluded_atom_ids"):
        excluded.update(str(value) for value in payload.get(key) or ())
    for key in ("ridges", "candidates"):
        for row in payload.get(key) or ():
            excluded.update(str(value) for value in row.get("atom_ids") or ())
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace thin high-normal-jump atom-edge candidates for F1 support-discovery diagnostics.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--normal-jump-deg", type=float, default=28.0)
    parser.add_argument("--min-edges", type=int, default=4)
    parser.add_argument("--min-span-mm", type=float, default=18.0)
    parser.add_argument("--max-thickness-mm", type=float, default=28.0)
    parser.add_argument("--exclude-owned-bend-atoms", action="store_true")
    parser.add_argument("--exclude-atom-ids-json", type=Path, default=None)
    args = parser.parse_args()
    report = trace_transition_edge_candidates(
        decomposition=_load_json(args.decomposition_json),
        normal_jump_deg=float(args.normal_jump_deg),
        min_edges=int(args.min_edges),
        min_span_mm=float(args.min_span_mm),
        max_thickness_mm=float(args.max_thickness_mm),
        exclude_owned_bend_atoms=bool(args.exclude_owned_bend_atoms),
        excluded_atom_ids=_excluded_ids_from_json(args.exclude_atom_ids_json),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_edge_count": report["selected_edge_count"],
                "candidate_count": report["candidate_count"],
                "admissible_candidate_count": report["admissible_candidate_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
