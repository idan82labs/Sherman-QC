import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_interior_bend_gaps.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_interior_bend_gaps", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x, y=50.0, z=50.0, *, label="residual", curvature=0.08):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": [0.0, 0.707, 0.707],
        "axis_direction": [1.0, 0.0, 0.0],
        "curvature_proxy": float(curvature),
        "weight": 10.0,
        "_label": label,
    }


def _payload(atoms):
    labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    atom_ids = [atom["atom_id"] for atom in atoms]
    adjacency = {atom_id: [] for atom_id in atom_ids}
    for left, right in zip(atom_ids, atom_ids[1:]):
        adjacency[left].append(right)
        adjacency[right].append(left)
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": adjacency, "local_spacing_mm": 1.0},
        "owned_bend_regions": [],
        "assignment": {
            "atom_labels": labels,
            "label_types": {"residual": "residual", "flange": "flange", "outlier": "outlier"},
        },
    }


def test_interior_gap_finds_simple_component():
    atoms = [_atom(f"A{i}", 20.0 + i * 4.0) for i in range(10)]
    payload = _payload(atoms)

    report = analyzer.analyze_interior_bend_gaps(
        decomposition=payload,
        score_threshold=0.5,
        interior_percentile_low=0.0,
        interior_percentile_high=100.0,
    )

    assert report["status"] == "interior_gap_candidates_found"
    assert report["admissible_candidate_count"] >= 1
    assert report["merged_hypothesis_count"] >= 1


def test_interior_gap_excludes_owned_bend_atoms():
    atoms = [_atom(f"A{i}", 20.0 + i * 4.0) for i in range(10)]
    payload = _payload(atoms)
    payload["owned_bend_regions"] = [{"bend_id": "OB1", "owned_atom_ids": [atom["atom_id"] for atom in atoms]}]

    report = analyzer.analyze_interior_bend_gaps(
        decomposition=payload,
        score_threshold=0.5,
        interior_percentile_low=0.0,
        interior_percentile_high=100.0,
    )

    assert report["candidate_count"] == 0
    assert report["merged_hypothesis_count"] == 0


def test_interior_gap_merges_nearby_parallel_fragments():
    first = [_atom(f"A{i}", 20.0 + i * 4.0, y=50.0) for i in range(5)]
    second = [_atom(f"B{i}", 40.0 + i * 4.0, y=64.0) for i in range(5)]
    atoms = first + second
    payload = _payload(atoms)
    # Split graph connectivity while keeping fragments geometrically close.
    payload["atom_graph"]["adjacency"]["A4"] = []
    payload["atom_graph"]["adjacency"]["B0"] = []

    report = analyzer.analyze_interior_bend_gaps(
        decomposition=payload,
        score_threshold=0.5,
        interior_percentile_low=0.0,
        interior_percentile_high=100.0,
    )

    assert report["admissible_candidate_count"] == 2
    assert report["merged_hypothesis_count"] == 1
    assert report["merged_hypotheses"][0]["reason_codes"] == ["interior_gap_fragments_merged"]
