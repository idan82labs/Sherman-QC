import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_family_excluded_interior_bend_gaps.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_family_excluded_interior_bend_gaps", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x, y=0.0, z=0.0, *, label="residual", curvature=0.08):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": [0.0, 0.707, 0.707],
        "axis_direction": [1.0, 0.0, 0.0],
        "curvature_proxy": float(curvature),
        "weight": 10.0,
        "_label": label,
    }


def _payload(raw_atoms, candidate_atoms):
    atoms = list(raw_atoms) + list(candidate_atoms)
    labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    atom_ids = [atom["atom_id"] for atom in atoms]
    adjacency = {atom_id: [] for atom_id in atom_ids}
    for group in (raw_atoms, candidate_atoms):
        ids = [atom["atom_id"] for atom in group]
        for left, right in zip(ids, ids[1:]):
            adjacency[left].append(right)
            adjacency[right].append(left)
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": adjacency, "local_spacing_mm": 1.0},
        "owned_bend_regions": [
            {
                "bend_id": "OB1",
                "incident_flange_ids": ["F1", "F2"],
                "owned_atom_ids": [atom["atom_id"] for atom in raw_atoms],
            }
        ],
        "assignment": {
            "atom_labels": labels,
            "label_types": {"residual": "residual", "F1": "flange", "F2": "flange"},
        },
    }


def test_family_excluded_search_masks_alias_atoms_near_raw_bend_line():
    raw = [_atom(f"R{i}", 20.0 + i * 4.0, y=50.0, label="F1") for i in range(8)]
    alias = [_atom(f"A{i}", 20.0 + i * 4.0, y=52.0) for i in range(8)]
    payload = _payload(raw, alias)

    report = analyzer.analyze_family_excluded_interior_bend_gaps(
        decomposition=payload,
        score_threshold=0.5,
        interior_percentile_low=0.0,
        interior_percentile_high=100.0,
        raw_family_tube_radius_mm=8.0,
        raw_family_span_pad_mm=12.0,
    )

    assert report["candidate_count"] == 0
    assert report["raw_family_excluded_atom_count"] == 8


def test_family_excluded_search_keeps_separate_interior_candidate():
    raw = [_atom(f"R{i}", 20.0 + i * 4.0, y=50.0, label="F1") for i in range(8)]
    separate = [_atom(f"S{i}", 20.0 + i * 4.0, y=85.0) for i in range(8)]
    payload = _payload(raw, separate)

    report = analyzer.analyze_family_excluded_interior_bend_gaps(
        decomposition=payload,
        score_threshold=0.5,
        interior_percentile_low=0.0,
        interior_percentile_high=100.0,
        raw_family_tube_radius_mm=8.0,
        raw_family_span_pad_mm=12.0,
    )

    assert report["status"] == "family_excluded_interior_candidates_found"
    assert report["admissible_candidate_count"] == 1
    assert report["candidates"][0]["candidate_id"] == "FEIG1"
