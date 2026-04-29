#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _load_bridge_module():
    path = Path(__file__).resolve().parent / "validate_raw_point_creases_against_f1.py"
    spec = importlib.util.spec_from_file_location("validate_raw_point_creases_against_f1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge_module()


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


def _aligned_normal_and_d(flange: Mapping[str, Any], reference: np.ndarray) -> Tuple[np.ndarray, float]:
    normal = _unit(flange.get("normal") or (1.0, 0.0, 0.0))
    d = float(flange.get("d") or 0.0)
    if float(np.dot(normal, reference)) < 0.0:
        return -normal, -d
    return normal, d


def _angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    return bridge._angle_deg(lhs, rhs)


def _flange_groups(
    flanges: Mapping[str, Mapping[str, Any]],
    *,
    normal_tol_deg: float,
    offset_tol_mm: float,
) -> List[List[str]]:
    rows = sorted(
        flanges.values(),
        key=lambda flange: (
            -len(flange.get("atom_ids") or ()),
            str(flange.get("flange_id")),
        ),
    )
    groups: List[List[str]] = []
    group_normals: List[np.ndarray] = []
    group_offsets: List[float] = []
    for flange in rows:
        flange_id = str(flange.get("flange_id"))
        normal = _unit(flange.get("normal") or (1.0, 0.0, 0.0))
        matched = False
        for index, group in enumerate(groups):
            aligned_normal, aligned_d = _aligned_normal_and_d(flange, group_normals[index])
            if _angle_deg(aligned_normal, group_normals[index]) <= normal_tol_deg and abs(aligned_d - group_offsets[index]) <= offset_tol_mm:
                group.append(flange_id)
                normals = []
                offsets = []
                for member_id in group:
                    member_normal, member_d = _aligned_normal_and_d(flanges[member_id], group_normals[index])
                    weight = max(1, len(flanges[member_id].get("atom_ids") or ()))
                    normals.extend([member_normal] * min(weight, 30))
                    offsets.extend([member_d] * min(weight, 30))
                group_normals[index] = _unit(np.mean(np.asarray(normals, dtype=np.float64), axis=0))
                group_offsets[index] = float(np.median(offsets))
                matched = True
                break
        if not matched:
            groups.append([flange_id])
            group_normals.append(normal)
            group_offsets.append(float(flange.get("d") or 0.0))
    for group in groups:
        group.sort()
    groups.sort(key=lambda group: (-sum(len(flanges[item].get("atom_ids") or ()) for item in group), group))
    return groups


def _family_flange(family_id: str, member_ids: Sequence[str], flanges: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    reference = _unit(flanges[member_ids[0]].get("normal") or (1.0, 0.0, 0.0))
    normals: List[np.ndarray] = []
    offsets: List[float] = []
    atom_ids: List[str] = []
    confidences: List[float] = []
    centroids: List[np.ndarray] = []
    centroid_weights: List[float] = []
    for member_id in member_ids:
        flange = flanges[member_id]
        normal, d = _aligned_normal_and_d(flange, reference)
        weight = max(1, len(flange.get("atom_ids") or ()))
        normals.extend([normal] * min(weight, 50))
        offsets.extend([d] * min(weight, 50))
        atom_ids.extend(str(value) for value in flange.get("atom_ids") or ())
        confidences.append(float(flange.get("confidence") or 0.0))
        if flange.get("centroid"):
            centroids.append(_vec(flange.get("centroid") or (0.0, 0.0, 0.0)))
            centroid_weights.append(float(weight))
    centroid = np.average(np.asarray(centroids, dtype=np.float64), axis=0, weights=np.asarray(centroid_weights, dtype=np.float64)) if centroids else np.zeros(3)
    return {
        "flange_id": family_id,
        "family_id": family_id,
        "member_flange_ids": list(member_ids),
        "normal": [float(value) for value in _unit(np.mean(np.asarray(normals, dtype=np.float64), axis=0)).tolist()],
        "d": float(np.median(offsets)) if offsets else 0.0,
        "centroid": [float(value) for value in centroid.tolist()],
        "atom_ids": sorted(set(atom_ids)),
        "confidence": max(confidences) if confidences else 0.0,
    }


def _build_family_decomposition(
    decomposition: Mapping[str, Any],
    *,
    normal_tol_deg: float,
    offset_tol_mm: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    flanges = bridge._flange_lookup(decomposition)
    groups = _flange_groups(flanges, normal_tol_deg=normal_tol_deg, offset_tol_mm=offset_tol_mm)
    family_flanges = {
        f"FG{index}": _family_flange(f"FG{index}", group, flanges)
        for index, group in enumerate(groups, start=1)
    }
    family_decomposition = json.loads(json.dumps(decomposition))
    family_decomposition["flange_hypotheses"] = list(family_flanges.values())
    family_payload = {
        "normal_tol_deg": normal_tol_deg,
        "offset_tol_mm": offset_tol_mm,
        "family_count": len(family_flanges),
        "families": [
            {
                "family_id": family_id,
                "member_flange_ids": family["member_flange_ids"],
                "atom_count": len(family.get("atom_ids") or ()),
                "normal": family["normal"],
                "d": family["d"],
            }
            for family_id, family in family_flanges.items()
        ],
    }
    return family_decomposition, family_payload


def validate_raw_point_creases_against_flange_families(
    *,
    decomposition: Mapping[str, Any],
    raw_creases: Mapping[str, Any],
    normal_tol_deg: float = 6.0,
    offset_tol_mm: float = 4.0,
    max_pairs_per_candidate: int = 8,
) -> Dict[str, Any]:
    family_decomposition, family_payload = _build_family_decomposition(
        decomposition,
        normal_tol_deg=normal_tol_deg,
        offset_tol_mm=offset_tol_mm,
    )
    family_report = bridge.validate_raw_point_creases_against_f1(
        decomposition=family_decomposition,
        raw_creases=raw_creases,
        max_pairs_per_candidate=max_pairs_per_candidate,
    )
    family_by_id = {row["family_id"]: row for row in family_payload["families"]}
    for candidate in family_report.get("candidates") or ():
        candidate["best_flange_family_pair_members"] = [
            family_by_id.get(str(family_id), {}).get("member_flange_ids", [])
            for family_id in candidate.get("best_flange_pair") or ()
        ]
        for score in candidate.get("top_pair_scores") or ():
            score["flange_family_pair_members"] = [
                family_by_id.get(str(family_id), {}).get("member_flange_ids", [])
                for family_id in score.get("flange_pair") or ()
            ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only raw crease validation after merging fragmented F1 flanges into coplanar families.",
        "flange_family_summary": family_payload,
        "raw_admissible_candidate_count": family_report.get("raw_admissible_candidate_count"),
        "family_bridge_candidate_count": family_report.get("bridge_candidate_count"),
        "family_bridge_safe_candidate_count": family_report.get("bridge_safe_candidate_count"),
        "status": family_report.get("status"),
        "candidates": family_report.get("candidates") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw-point creases against merged F1 flange families.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--raw-creases-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--normal-tol-deg", type=float, default=6.0)
    parser.add_argument("--offset-tol-mm", type=float, default=4.0)
    parser.add_argument("--max-pairs-per-candidate", type=int, default=8)
    args = parser.parse_args()
    report = validate_raw_point_creases_against_flange_families(
        decomposition=_load_json(args.decomposition_json),
        raw_creases=_load_json(args.raw_creases_json),
        normal_tol_deg=float(args.normal_tol_deg),
        offset_tol_mm=float(args.offset_tol_mm),
        max_pairs_per_candidate=int(args.max_pairs_per_candidate),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "family_count": report["flange_family_summary"]["family_count"],
                "family_bridge_candidate_count": report["family_bridge_candidate_count"],
                "family_bridge_safe_candidate_count": report["family_bridge_safe_candidate_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
