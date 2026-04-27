import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cluster_f1_residual_interface_candidates.py"
    spec = importlib.util.spec_from_file_location("cluster_f1_residual_interface_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


clusterer = _load_module()


def _atom(atom_id, x):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), 0.0, 0.0],
        "curvature_proxy": 0.05,
        "weight": 10.0,
    }


def _payload(atoms):
    atom_ids = [atom["atom_id"] for atom in atoms]
    adjacency = {atom_id: [] for atom_id in atom_ids}
    for left, right in zip(atom_ids, atom_ids[1:]):
        adjacency[left].append(right)
        adjacency[right].append(left)
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {
            "atoms": atoms,
            "adjacency": adjacency,
        },
    }


def test_subclusters_split_large_projection_gap():
    left = [_atom(f"L{i}", i * 2.0) for i in range(8)]
    right = [_atom(f"R{i}", 60.0 + i * 2.0) for i in range(8)]
    atoms = left + right
    payload = _payload(atoms)
    payload["atom_graph"]["adjacency"]["L7"] = []
    payload["atom_graph"]["adjacency"]["R0"] = []
    diagnostics = {
        "status": "ambiguous_overlapping_candidate_pool",
        "recovery_deficit": 2,
        "admissible_candidate_count": 2,
        "admissible_conflict_count": 0,
        "candidates": [
            {
                "candidate_id": "RIC1",
                "admissible": True,
                "atom_ids": [atom["atom_id"] for atom in atoms],
            }
        ],
    }

    report = clusterer.cluster_residual_interface_candidates(
        decomposition=payload,
        residual_interface_diagnostics=diagnostics,
    )

    assert report["status"] == "subclusters_cover_deficit"
    assert report["plausible_cluster_count"] == 2


def test_conflicting_parent_candidates_block_clean_cover_status():
    atoms = [_atom(f"A{i}", i * 2.0) for i in range(10)]
    diagnostics = {
        "status": "ambiguous_overlapping_candidate_pool",
        "recovery_deficit": 1,
        "admissible_candidate_count": 2,
        "admissible_conflict_count": 1,
        "candidates": [
            {
                "candidate_id": "RIC1",
                "admissible": True,
                "atom_ids": [atom["atom_id"] for atom in atoms],
            }
        ],
    }

    report = clusterer.cluster_residual_interface_candidates(
        decomposition=_payload(atoms),
        residual_interface_diagnostics=diagnostics,
    )

    assert report["status"] == "subclusters_exist_but_parent_candidates_conflict"
