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


def _assignment(decomposition: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    assignment = decomposition.get("assignment") or {}
    return (
        {str(key): str(value) for key, value in (assignment.get("atom_labels") or {}).items()},
        {str(key): str(value) for key, value in (assignment.get("label_types") or {}).items()},
    )


def _owned_region_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(region.get("bend_id")): region
        for region in decomposition.get("owned_bend_regions") or ()
        if region.get("bend_id")
    }


def _manual_atom_sets(decomposition: Mapping[str, Any], feature_labels: Optional[Mapping[str, Any]]) -> Dict[str, set[str]]:
    regions = _owned_region_lookup(decomposition)
    accepted: set[str] = set()
    rejected: set[str] = set()
    if not feature_labels:
        return {"accepted": accepted, "rejected": rejected}
    for row in feature_labels.get("region_labels") or ():
        region = regions.get(str(row.get("bend_id")))
        if not region:
            continue
        atoms = {str(value) for value in region.get("owned_atom_ids") or ()}
        if row.get("human_feature_label") == "conventional_bend":
            accepted.update(atoms)
        elif row.get("human_feature_label") in {"fragment_or_duplicate", "false_positive", "non_bend"}:
            rejected.update(atoms)
    return {"accepted": accepted, "rejected": rejected}


def _domain_atom_ids(
    *,
    decomposition: Mapping[str, Any],
    residual_interface_diagnostics: Optional[Mapping[str, Any]],
    accepted_atoms: set[str],
) -> List[str]:
    if residual_interface_diagnostics:
        ids: set[str] = set()
        for candidate in residual_interface_diagnostics.get("candidates") or ():
            if candidate.get("admissible"):
                ids.update(str(value) for value in candidate.get("atom_ids") or ())
        return sorted(atom_id for atom_id in ids if atom_id not in accepted_atoms)
    atoms = _atom_lookup(decomposition)
    return sorted(atom_id for atom_id in atoms if atom_id not in accepted_atoms)


def _normal_contrast(atom_id: str, atoms: Mapping[str, Mapping[str, Any]], adjacency: Mapping[str, Sequence[str]]) -> float:
    atom = atoms.get(atom_id)
    if not atom:
        return 0.0
    normal = atom.get("normal") or (1, 0, 0)
    angles = [
        _axis_angle_deg(normal, (atoms.get(str(neighbor)) or {}).get("normal") or normal)
        for neighbor in adjacency.get(atom_id, ()) or ()
        if str(neighbor) in atoms
    ]
    if not angles:
        return 0.0
    return min(1.0, float(np.percentile(angles, 80)) / 70.0)


def _ridge_scores(
    *,
    atoms: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[str]],
    domain_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for atom_id in domain_ids:
        atom = atoms.get(atom_id)
        if not atom:
            continue
        curvature_signal = min(1.0, float(atom.get("curvature_proxy") or 0.0) * 20.0)
        contrast = _normal_contrast(atom_id, atoms, adjacency)
        axis_neighbors = [
            _axis_angle_deg(atom.get("axis_direction") or (1, 0, 0), (atoms.get(str(neighbor)) or {}).get("axis_direction") or (1, 0, 0))
            for neighbor in adjacency.get(atom_id, ()) or ()
            if str(neighbor) in atoms
        ]
        axis_coherence = 0.0
        if axis_neighbors:
            axis_coherence = max(0.0, 1.0 - float(np.percentile(axis_neighbors, 60)) / 75.0)
        ridge_score = 0.52 * curvature_signal + 0.38 * contrast + 0.10 * axis_coherence
        rows[atom_id] = {
            "atom_id": atom_id,
            "curvature_signal": round(float(curvature_signal), 6),
            "normal_contrast": round(float(contrast), 6),
            "axis_coherence": round(float(axis_coherence), 6),
            "ridge_score": round(float(ridge_score), 6),
        }
    return rows


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


def _pca_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    if len(points) < 2:
        return centroid, np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vh[0]
    norm = float(np.linalg.norm(axis))
    return centroid, axis / norm if norm > 1e-9 else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def _component_summary(
    *,
    component_id: str,
    atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    centroid, axis = _pca_axis(points)
    projections = [float(np.dot(point - centroid, axis)) for point in points]
    span = max(projections) - min(projections) if projections else 0.0
    values = [float(score_rows[atom_id]["ridge_score"]) for atom_id in atom_ids if atom_id in score_rows]
    curvature = [float(score_rows[atom_id]["curvature_signal"]) for atom_id in atom_ids if atom_id in score_rows]
    contrast = [float(score_rows[atom_id]["normal_contrast"]) for atom_id in atom_ids if atom_id in score_rows]
    weights = [float((atoms.get(atom_id) or {}).get("weight") or 0.0) for atom_id in atom_ids]
    reasons: List[str] = []
    if len(atom_ids) < 4:
        reasons.append("ridge_atom_count_below_floor")
    if span < 8.0:
        reasons.append("ridge_span_below_floor")
    if (sum(values) / max(1, len(values))) < 0.58:
        reasons.append("ridge_score_below_floor")
    if not reasons:
        reasons.append("local_ridge_candidate")
    return {
        "ridge_id": component_id,
        "atom_ids": sorted(atom_ids),
        "atom_count": len(atom_ids),
        "support_mass": round(float(sum(weights)), 6),
        "support_centroid": [round(float(value), 6) for value in centroid.tolist()],
        "axis_direction": [round(float(value), 6) for value in axis.tolist()],
        "span_mm": round(float(span), 6),
        "mean_ridge_score": round(float(sum(values) / max(1, len(values))), 6),
        "mean_curvature_signal": round(float(sum(curvature) / max(1, len(curvature))), 6),
        "mean_normal_contrast": round(float(sum(contrast) / max(1, len(contrast))), 6),
        "admissible": reasons == ["local_ridge_candidate"],
        "reason_codes": reasons,
    }


def trace_local_ridge_candidates(
    *,
    decomposition: Mapping[str, Any],
    feature_labels: Optional[Mapping[str, Any]] = None,
    feature_label_summary: Optional[Mapping[str, Any]] = None,
    residual_interface_diagnostics: Optional[Mapping[str, Any]] = None,
    score_threshold: float = 0.58,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    atom_graph = decomposition.get("atom_graph") or {}
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in (atom_graph.get("adjacency") or {}).items()}
    manual_sets = _manual_atom_sets(decomposition, feature_labels)
    domain_ids = _domain_atom_ids(
        decomposition=decomposition,
        residual_interface_diagnostics=residual_interface_diagnostics,
        accepted_atoms=manual_sets["accepted"],
    )
    scores = _ridge_scores(atoms=atoms, adjacency=adjacency, domain_ids=domain_ids)
    ridge_ids = [
        atom_id
        for atom_id, row in scores.items()
        if float(row["ridge_score"]) >= score_threshold
        and (float(row["curvature_signal"]) >= 0.35 or float(row["normal_contrast"]) >= 0.50)
    ]
    components = _connected_components(ridge_ids, adjacency)
    ridges = [
        _component_summary(component_id=f"RIDGE{index}", atom_ids=component, atoms=atoms, score_rows=scores)
        for index, component in enumerate(sorted(components, key=lambda values: (-len(values), values[0])), start=1)
    ]
    ridges.sort(key=lambda item: (not item["admissible"], -int(item["atom_count"]), -float(item["mean_ridge_score"]), str(item["ridge_id"])))
    capacity = dict((feature_label_summary or {}).get("label_capacity") or {})
    recovery_deficit = max(
        int(capacity.get("valid_conventional_region_deficit") or 0),
        int(capacity.get("valid_counted_feature_deficit") or 0),
        int(capacity.get("owned_region_capacity_deficit") or 0),
    )
    admissible = [item for item in ridges if item.get("admissible")]
    status = "no_recovery_deficit" if recovery_deficit <= 0 else "no_local_ridge_candidate"
    if recovery_deficit > 0 and len(admissible) >= recovery_deficit:
        status = "ridge_candidates_cover_deficit"
    elif recovery_deficit > 0 and admissible:
        status = "partial_ridge_signal"
    label_mix: Dict[str, int] = {}
    for atom_id in ridge_ids:
        label = atom_labels.get(atom_id, "unassigned")
        label_mix[label] = label_mix.get(label, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only local curvature/normal ridge tracing for missing F1 bend support.",
        "score_threshold": score_threshold,
        "recovery_deficit": recovery_deficit,
        "domain_atom_count": len(domain_ids),
        "ridge_atom_count": len(ridge_ids),
        "ridge_label_mix": dict(sorted(label_mix.items())),
        "candidate_count": len(ridges),
        "admissible_candidate_count": len(admissible),
        "status": status,
        "accepted_manual_atom_count": len(manual_sets["accepted"]),
        "rejected_manual_atom_count": len(manual_sets["rejected"]),
        "ridges": ridges,
        "atom_scores": [scores[atom_id] for atom_id in sorted(scores)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace diagnostic local curvature/normal ridge candidates for F1 missing support.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--feature-labels-json", type=Path, default=None)
    parser.add_argument("--feature-label-summary-json", type=Path, default=None)
    parser.add_argument("--residual-interface-json", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.58)
    args = parser.parse_args()

    report = trace_local_ridge_candidates(
        decomposition=_load_json(args.decomposition_json),
        feature_labels=_load_json(args.feature_labels_json) if args.feature_labels_json else None,
        feature_label_summary=_load_json(args.feature_label_summary_json) if args.feature_label_summary_json else None,
        residual_interface_diagnostics=_load_json(args.residual_interface_json) if args.residual_interface_json else None,
        score_threshold=float(args.score_threshold),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "domain_atom_count": report["domain_atom_count"],
                "ridge_atom_count": report["ridge_atom_count"],
                "candidate_count": report["candidate_count"],
                "admissible_candidate_count": report["admissible_candidate_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
