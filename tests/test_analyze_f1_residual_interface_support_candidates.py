import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_residual_interface_support_candidates.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_residual_interface_support_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x, z=0.0, label="residual"):
    return {
        "atom_id": atom_id,
        "centroid": [0.0, float(z), float(x)],
        "normal": [0.7071, 0.7071, 0.0],
        "local_spacing_mm": 1.0,
        "curvature_proxy": 0.05,
        "axis_direction": [1.0, 0.0, 0.0],
        "weight": 10.0,
        "_label": label,
    }


def _payload(atoms):
    atom_labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    atom_ids = [atom["atom_id"] for atom in atoms]
    adjacency = {
        atom_id: [
            other
            for other in atom_ids
            if other != atom_id and abs(int(other[1:]) - int(atom_id[1:])) == 1
        ]
        for atom_id in atom_ids
    }
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {
            "atoms": atoms,
            "adjacency": adjacency,
            "edge_weights": {
                "::".join(sorted((left, right))): 1.0
                for left, neighbors in adjacency.items()
                for right in neighbors
            },
            "local_spacing_mm": 1.0,
        },
        "flange_hypotheses": [
            {
                "flange_id": "F1",
                "normal": [1.0, 0.0, 0.0],
                "d": 0.0,
                "centroid": [0.0, 0.0, 0.0],
            },
            {
                "flange_id": "F2",
                "normal": [0.0, 1.0, 0.0],
                "d": 0.0,
                "centroid": [0.0, 0.0, 0.0],
            },
        ],
        "owned_bend_regions": [
            {
                "bend_id": "OB1",
                "owned_atom_ids": ["A1", "A2", "A3", "A4"],
            }
        ],
        "assignment": {
            "atom_labels": atom_labels,
            "label_types": {
                "F1": "flange",
                "F2": "flange",
                "residual": "residual",
                "B1": "bend",
            },
        },
    }


def test_residual_interface_candidate_covers_deficit():
    atoms = [_atom(f"A{i}", i * 3.0) for i in range(1, 7)]
    atoms[0]["_label"] = "F1"
    atoms[-1]["_label"] = "F2"
    report = analyzer.analyze_residual_interface_support_candidates(
        decomposition=_payload(atoms),
        feature_label_summary={
            "label_capacity": {
                "valid_conventional_region_deficit": 1,
            }
        },
    )

    assert report["status"] == "candidate_pool_covers_deficit"
    assert report["admissible_candidate_count"] >= 1
    assert report["candidates"][0]["admissible"] is True
    assert report["candidates"][0]["flange_pair"] == ["F1", "F2"]


def test_manual_accepted_atoms_are_excluded_from_candidate_pool():
    atoms = [_atom(f"A{i}", i * 3.0, label="B1") for i in range(1, 7)]
    report = analyzer.analyze_residual_interface_support_candidates(
        decomposition=_payload(atoms),
        feature_labels={
            "region_labels": [
                {
                    "bend_id": "OB1",
                    "human_feature_label": "conventional_bend",
                }
            ]
        },
        feature_label_summary={
            "label_capacity": {
                "valid_conventional_region_deficit": 1,
            }
        },
    )

    assert report["accepted_manual_atom_count"] == 4
    assert report["candidate_count"] == 0
    assert report["status"] == "no_residual_interface_candidate"


def test_far_corridor_geometry_is_rejected():
    atoms = [_atom(f"A{i}", i * 3.0, z=40.0) for i in range(1, 7)]
    report = analyzer.analyze_residual_interface_support_candidates(
        decomposition=_payload(atoms),
        feature_label_summary={
            "label_capacity": {
                "valid_conventional_region_deficit": 1,
            }
        },
    )

    assert report["admissible_candidate_count"] == 0
    assert report["status"] == "no_residual_interface_candidate"
