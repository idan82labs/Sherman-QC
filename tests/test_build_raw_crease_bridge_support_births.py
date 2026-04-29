import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_raw_crease_bridge_support_births.py"
    spec = importlib.util.spec_from_file_location("build_raw_crease_bridge_support_births", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def _atom(atom_id, x, y, z, label):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": [1.0, 0.0, 0.0],
        "local_spacing_mm": 1.0,
        "curvature_proxy": 0.0,
        "axis_direction": [0.0, 0.0, 1.0],
        "weight": 1.0,
        "_label": label,
    }


def _payload():
    atoms = [_atom(f"L{i}", 0.0, -2.0, i * 4.0, "F1") for i in range(4)] + [
        _atom(f"R{i}", -2.0, 0.0, i * 4.0, "F2") for i in range(4)
    ]
    labels = {atom["atom_id"]: atom.pop("_label") for atom in atoms}
    return {
        "scan_path": "/tmp/demo.ply",
        "part_id": "demo",
        "atom_graph": {"atoms": atoms, "adjacency": {atom["atom_id"]: [] for atom in atoms}, "edge_weights": {}, "local_spacing_mm": 1.0},
        "flange_hypotheses": [
            {"flange_id": "F1", "normal": [1.0, 0.0, 0.0], "d": 0.0, "atom_ids": [f"L{i}" for i in range(4)]},
            {"flange_id": "F2", "normal": [0.0, 1.0, 0.0], "d": 0.0, "atom_ids": [f"R{i}" for i in range(4)]},
        ],
        "owned_bend_regions": [{"bend_id": "OB1", "owned_atom_ids": ["L0"]}],
        "assignment": {"atom_labels": labels, "label_types": {"F1": "flange", "F2": "flange", "residual": "residual"}},
    }


def test_build_raw_crease_bridge_support_births_creates_synthetic_atoms_and_payloads():
    bridge = {
        "candidates": [
            {
                "raw_candidate_id": "RPC1",
                "bridge_safe": True,
                "best_flange_pair": ["F1", "F2"],
                "best_pair_score": 3.1,
                "span_mm": 20.0,
                "thickness_mm": 1.0,
                "linearity": 0.9,
                "axis_direction": [0.0, 0.0, 1.0],
                "sampled_points": [[0.0, 0.0, i * 4.0] for i in range(6)],
                "top_pair_scores": [{"axis_delta_deg": 0.0, "median_line_distance_mm": 0.0}],
            },
            {
                "raw_candidate_id": "RPC2",
                "bridge_safe": False,
                "best_flange_pair": ["F1", "F2"],
                "sampled_points": [[0.0, 0.0, 0.0]],
            },
        ]
    }

    result = builder.build_raw_crease_bridge_support_births(decomposition=_payload(), bridge_validation=bridge)

    summary = result["summary"]
    assert summary["bridge_safe_input_count"] == 1
    assert summary["synthetic_birth_count"] == 1
    assert summary["projected_count_if_all_births_count"] == 2
    interior = result["interior_gaps_payload"]
    contact = result["contact_recovery_payload"]
    assert interior["candidates"][0]["candidate_id"] == "RPC1"
    assert all(atom_id.startswith("RAW_RPC1_") for atom_id in interior["candidates"][0]["atom_ids"])
    assert contact["validated_pairs"][0]["source_id"] == "RPC1"
