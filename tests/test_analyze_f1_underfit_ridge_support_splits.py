from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_underfit_ridge_support_splits.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_underfit_ridge_support_splits", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id: str, x: float) -> dict:
    return {
        "atom_id": atom_id,
        "centroid": [x, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "curvature_proxy": 0.05,
        "weight": 10.0,
    }


def _payload(xs: list[float]) -> dict:
    atoms = [_atom(f"A{index}", x) for index, x in enumerate(xs)]
    adjacency = {
        atom["atom_id"]: [
            other["atom_id"]
            for other in atoms
            if abs(float(other["centroid"][0]) - float(atom["centroid"][0])) <= 2.1
            and other["atom_id"] != atom["atom_id"]
        ] + ["F1_NEIGHBOR"]
        for atom in atoms
    }
    labels = {atom["atom_id"]: "residual" for atom in atoms}
    labels["F1_NEIGHBOR"] = "F1"
    label_types = {"residual": "residual", "F1": "flange"}
    return {
        "scan_path": "synthetic.ply",
        "part_id": "synthetic",
        "atom_graph": {
            "atoms": atoms,
            "adjacency": adjacency,
            "local_spacing_mm": 1.0,
        },
        "assignment": {
            "atom_labels": labels,
            "label_types": label_types,
        },
    }


def _ridge(xs: list[float]) -> dict:
    return {
        "ridges": [
            {
                "ridge_id": "RIDGE1",
                "atom_ids": [f"A{index}" for index, _ in enumerate(xs)],
            }
        ],
        "atom_scores": [
            {
                "atom_id": f"A{index}",
                "ridge_score": 0.8,
            }
            for index, _ in enumerate(xs)
        ],
    }


def test_continuous_ridge_does_not_cover_underfit_deficit():
    xs = [float(index) for index in range(12)]
    report = analyzer.analyze_underfit_ridge_support_splits(
        decomposition=_payload(xs),
        local_ridge_payload=_ridge(xs),
        feature_label_summary={"label_capacity": {"owned_region_capacity_deficit": 4}},
    )

    assert report["status"] == "underfit_not_splittable_single_ridge_family"
    assert report["candidate_segment_count"] == 1
    assert report["admissible_segment_count"] == 1


def test_projection_gaps_can_create_multiple_support_segments():
    xs = [*range(0, 10), *range(25, 35), *range(50, 60)]
    report = analyzer.analyze_underfit_ridge_support_splits(
        decomposition=_payload([float(value) for value in xs]),
        local_ridge_payload=_ridge([float(value) for value in xs]),
        feature_label_summary={"label_capacity": {"owned_region_capacity_deficit": 3}},
        min_gap_mm=8.0,
    )

    assert report["status"] == "split_segments_cover_recovery_deficit"
    assert report["candidate_segment_count"] == 3
    assert report["admissible_segment_count"] == 3
