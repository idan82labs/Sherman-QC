import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_interior_flange_contact_recovery.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_interior_flange_contact_recovery", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x, y=0.0, z=0.0, *, label="residual", normal=(0.707, 0.707, 0.0)):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": list(normal),
        "axis_direction": [0.0, 0.0, 1.0],
        "curvature_proxy": 0.04,
        "weight": 10.0,
        "_label": label,
    }


def _payload(atoms):
    labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": {}},
        "flange_hypotheses": [
            {"flange_id": "F1", "normal": [1.0, 0.0, 0.0], "d": 0.0},
            {"flange_id": "F2", "normal": [0.0, 1.0, 0.0], "d": 0.0},
            {"flange_id": "F3", "normal": [0.0, 0.0, 1.0], "d": 0.0},
        ],
        "assignment": {
            "atom_labels": labels,
            "label_types": {"residual": "residual", "F1": "flange", "F2": "flange", "F3": "flange"},
        },
    }


def _interior(axis=(0.0, 0.0, 1.0)):
    return {
        "candidates": [
            {
                "candidate_id": "IBG1",
                "admissible": True,
                "atom_ids": [f"B{i}" for i in range(8)],
                "atom_count": 8,
                "centroid": [0.0, 0.0, 14.0],
                "axis_direction": list(axis),
                "span_mm": 28.0,
                "median_line_residual_mm": 1.0,
            }
        ],
        "merged_hypotheses": [],
    }


def test_contact_recovery_validates_two_sided_nearby_flanges():
    bend_atoms = [_atom(f"B{i}", 0.0, 0.0, i * 4.0, normal=(0.707, 0.707, 0.0)) for i in range(8)]
    flange_a = [_atom(f"F1A{i}", -3.0, 0.0, i * 4.0, label="F1", normal=(1.0, 0.0, 0.0)) for i in range(5)]
    flange_b = [_atom(f"F2A{i}", 0.0, 3.0, i * 4.0, label="F2", normal=(0.0, 1.0, 0.0)) for i in range(5)]
    payload = _payload(bend_atoms + flange_a + flange_b)

    report = analyzer.analyze_interior_flange_contact_recovery(
        decomposition=payload,
        interior_gaps=_interior(),
        line_radius_mm=8.0,
        min_side_atoms=3,
    )

    assert report["status"] == "fragment_contact_recovered"
    assert report["validated_pairs"][0]["flange_pair"] == ["F1", "F2"]


def test_contact_recovery_rejects_one_sided_contact():
    bend_atoms = [_atom(f"B{i}", 0.0, 0.0, i * 4.0, normal=(0.707, 0.707, 0.0)) for i in range(8)]
    flange_a = [_atom(f"F1A{i}", -3.0, 0.0, i * 4.0, label="F1", normal=(1.0, 0.0, 0.0)) for i in range(5)]
    payload = _payload(bend_atoms + flange_a)

    report = analyzer.analyze_interior_flange_contact_recovery(
        decomposition=payload,
        interior_gaps=_interior(),
        line_radius_mm=8.0,
        min_side_atoms=3,
    )

    assert report["status"] == "no_validated_interior_contact"
    assert report["validated_pair_count"] == 0
