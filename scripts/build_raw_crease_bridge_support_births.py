#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


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


def _flange_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(flange.get("flange_id")): flange
        for flange in decomposition.get("flange_hypotheses") or ()
        if flange.get("flange_id")
    }


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
    }


def _nearest_atom_id(points: np.ndarray, atom_ids: Sequence[str], atoms: Mapping[str, Mapping[str, Any]]) -> str | None:
    atom_points = []
    kept_ids = []
    for atom_id in atom_ids:
        atom = atoms.get(str(atom_id))
        if not atom:
            continue
        atom_points.append(_vec(atom.get("centroid") or (0.0, 0.0, 0.0)))
        kept_ids.append(str(atom_id))
    if len(atom_points) == 0 or len(points) == 0:
        return None
    tree = cKDTree(np.asarray(atom_points, dtype=np.float64))
    distances, indexes = tree.query(points, k=1)
    best_point_index = int(np.argmin(distances))
    return kept_ids[int(indexes[best_point_index])]


def _synthetic_normal_for_pair(pair: Sequence[str], flanges: Mapping[str, Mapping[str, Any]]) -> List[float]:
    left = flanges.get(str(pair[0]))
    right = flanges.get(str(pair[1]))
    if not left or not right:
        return [1.0, 0.0, 0.0]
    normal = _unit(_unit(left.get("normal") or (1.0, 0.0, 0.0)) + _unit(right.get("normal") or (0.0, 1.0, 0.0)))
    return [float(value) for value in normal.tolist()]


def _bridge_safe_candidates(bridge: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [candidate for candidate in bridge.get("candidates") or () if candidate.get("bridge_safe")]


def build_raw_crease_bridge_support_births(
    *,
    decomposition: Mapping[str, Any],
    bridge_validation: Mapping[str, Any],
    max_points_per_birth: int = 90,
) -> Dict[str, Any]:
    augmented = json.loads(json.dumps(decomposition))
    graph = augmented.setdefault("atom_graph", {})
    graph.setdefault("atoms", [])
    graph.setdefault("adjacency", {})
    graph.setdefault("edge_weights", {})
    assignment = augmented.setdefault("assignment", {})
    atom_labels = assignment.setdefault("atom_labels", {})
    label_types = assignment.setdefault("label_types", {})
    label_types.setdefault("residual", "residual")

    atoms = _atom_lookup(augmented)
    flanges = _flange_lookup(augmented)
    local_spacing = max(float(graph.get("local_spacing_mm") or 1.0), 1.0)
    interior_candidates: List[Dict[str, Any]] = []
    validated_pairs: List[Dict[str, Any]] = []
    birth_summaries: List[Dict[str, Any]] = []

    for candidate in _bridge_safe_candidates(bridge_validation):
        raw_id = str(candidate.get("raw_candidate_id"))
        pair = [str(value) for value in candidate.get("best_flange_pair") or ()]
        if len(pair) != 2 or pair[0] not in flanges or pair[1] not in flanges:
            continue
        points = np.asarray(candidate.get("sampled_points") or (), dtype=np.float64)
        if len(points) == 0:
            continue
        axis = _unit(candidate.get("axis_direction") or (1.0, 0.0, 0.0))
        center = np.mean(points, axis=0)
        projections = (points - center) @ axis
        order = np.argsort(projections)
        if len(order) > max_points_per_birth:
            keep_indexes = np.linspace(0, len(order) - 1, max_points_per_birth).round().astype(np.int64)
            order = order[keep_indexes]
        selected_points = points[order]
        normal = _synthetic_normal_for_pair(pair, flanges)
        synthetic_ids: List[str] = []
        for index, point in enumerate(selected_points):
            atom_id = f"RAW_{raw_id}_{index:03d}"
            synthetic_ids.append(atom_id)
            graph["atoms"].append(
                {
                    "atom_id": atom_id,
                    "centroid": [float(value) for value in point.tolist()],
                    "normal": normal,
                    "local_spacing_mm": local_spacing,
                    "curvature_proxy": 0.08,
                    "axis_direction": [float(value) for value in axis.tolist()],
                    "weight": 1.0,
                    "area_proxy_mm2": max(1.0, local_spacing * local_spacing),
                    "source_kind": "raw_point_crease_bridge_birth",
                    "source_raw_candidate_id": raw_id,
                }
            )
            atom_labels[atom_id] = "residual"
            graph["adjacency"].setdefault(atom_id, [])
        for left, right in zip(synthetic_ids, synthetic_ids[1:]):
            graph["adjacency"].setdefault(left, []).append(right)
            graph["adjacency"].setdefault(right, []).append(left)
            graph["edge_weights"]["::".join(sorted((left, right)))] = 1.0
        atom_lookup = _atom_lookup(augmented)
        for flange_id in pair:
            nearest = _nearest_atom_id(selected_points, flanges[flange_id].get("atom_ids") or (), atom_lookup)
            if not nearest:
                continue
            anchor_id = synthetic_ids[0]
            graph["adjacency"].setdefault(anchor_id, []).append(nearest)
            graph["adjacency"].setdefault(nearest, []).append(anchor_id)
            graph["edge_weights"]["::".join(sorted((anchor_id, nearest)))] = 0.75
        interior_candidates.append(
            {
                "candidate_id": raw_id,
                "source_kind": "raw_point_crease_bridge_birth",
                "admissible": True,
                "atom_ids": synthetic_ids,
                "raw_candidate": {
                    "span_mm": candidate.get("span_mm"),
                    "thickness_mm": candidate.get("thickness_mm"),
                    "linearity": candidate.get("linearity"),
                    "best_pair_score": candidate.get("best_pair_score"),
                },
            }
        )
        validated_pairs.append(
            {
                "source_id": raw_id,
                "source_kind": "raw_point_crease_bridge_birth",
                "flange_pair": pair,
                "status": "recovered_contact_validated",
                "score": candidate.get("best_pair_score"),
                "axis_delta_deg": (candidate.get("top_pair_scores") or [{}])[0].get("axis_delta_deg"),
                "median_line_distance_mm": (candidate.get("top_pair_scores") or [{}])[0].get("median_line_distance_mm"),
                "reason_codes": ["raw_point_crease_bridge_validated"],
            }
        )
        birth_summaries.append(
            {
                "raw_candidate_id": raw_id,
                "flange_pair": pair,
                "synthetic_atom_count": len(synthetic_ids),
                "bridge_score": candidate.get("best_pair_score"),
                "span_mm": candidate.get("span_mm"),
                "thickness_mm": candidate.get("thickness_mm"),
            }
        )

    interior_payload = {
        "purpose": "Synthetic F1 support atoms generated from bridge-safe raw-point crease candidates.",
        "candidates": interior_candidates,
        "merged_hypotheses": [],
    }
    contact_payload = {
        "purpose": "Recovered contact payload for bridge-safe raw-point crease births.",
        "validated_pairs": validated_pairs,
    }
    summary = {
        "part_id": augmented.get("part_id"),
        "scan_path": augmented.get("scan_path"),
        "bridge_safe_input_count": len(_bridge_safe_candidates(bridge_validation)),
        "synthetic_birth_count": len(birth_summaries),
        "current_owned_bend_region_count": len(augmented.get("owned_bend_regions") or []),
        "projected_count_if_all_births_count": len(augmented.get("owned_bend_regions") or []) + len(birth_summaries),
        "births": birth_summaries,
    }
    return {
        "augmented_decomposition": augmented,
        "interior_gaps_payload": interior_payload,
        "contact_recovery_payload": contact_payload,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diagnostic F1 support-birth payloads from bridge-safe raw crease candidates.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--bridge-validation-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-points-per-birth", type=int, default=90)
    args = parser.parse_args()
    result = build_raw_crease_bridge_support_births(
        decomposition=_load_json(args.decomposition_json),
        bridge_validation=_load_json(args.bridge_validation_json),
        max_points_per_birth=int(args.max_points_per_birth),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "augmented_decomposition.json", result["augmented_decomposition"])
    _write_json(args.output_dir / "raw_bridge_interior_gaps.json", result["interior_gaps_payload"])
    _write_json(args.output_dir / "raw_bridge_contact_recovery.json", result["contact_recovery_payload"])
    _write_json(args.output_dir / "raw_bridge_support_birth_summary.json", result["summary"])
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
