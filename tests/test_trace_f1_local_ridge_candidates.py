import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "trace_f1_local_ridge_candidates.py"
    spec = importlib.util.spec_from_file_location("trace_f1_local_ridge_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tracer = _load_module()


def _atom(atom_id, x, curvature=0.05, normal=(1.0, 0.0, 0.0), label="residual"):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), 0.0, 0.0],
        "normal": list(normal),
        "curvature_proxy": float(curvature),
        "axis_direction": [1.0, 0.0, 0.0],
        "weight": 10.0,
        "_label": label,
    }


def _payload(atoms):
    atom_labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    atom_ids = [atom["atom_id"] for atom in atoms]
    adjacency = {atom_id: [] for atom_id in atom_ids}
    for left, right in zip(atom_ids, atom_ids[1:]):
        adjacency[left].append(right)
        adjacency[right].append(left)
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": adjacency},
        "owned_bend_regions": [{"bend_id": "OB1", "owned_atom_ids": atom_ids[:4]}],
        "assignment": {
            "atom_labels": atom_labels,
            "label_types": {"residual": "residual", "B1": "bend"},
        },
    }


def test_trace_local_ridge_candidates_covers_deficit():
    atoms = [_atom(f"A{i}", i * 3.0, curvature=0.06) for i in range(8)]
    report = tracer.trace_local_ridge_candidates(
        decomposition=_payload(atoms),
        feature_label_summary={"label_capacity": {"valid_conventional_region_deficit": 1}},
        score_threshold=0.55,
    )

    assert report["status"] == "ridge_candidates_cover_deficit"
    assert report["admissible_candidate_count"] == 1
    assert report["ridges"][0]["atom_count"] == 8


def test_trace_local_ridge_candidates_excludes_accepted_manual_atoms():
    atoms = [_atom(f"A{i}", i * 3.0, curvature=0.06, label="B1") for i in range(8)]
    report = tracer.trace_local_ridge_candidates(
        decomposition=_payload(atoms),
        feature_labels={"region_labels": [{"bend_id": "OB1", "human_feature_label": "conventional_bend"}]},
        feature_label_summary={"label_capacity": {"valid_conventional_region_deficit": 1}},
        score_threshold=0.55,
    )

    assert report["accepted_manual_atom_count"] == 4
    assert all("A0" not in ridge["atom_ids"] for ridge in report["ridges"])
