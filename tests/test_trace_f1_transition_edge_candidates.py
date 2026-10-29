import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "trace_f1_transition_edge_candidates.py"
    spec = importlib.util.spec_from_file_location("trace_f1_transition_edge_candidates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tracer = _load_module()


def _atom(atom_id, x, normal=(1.0, 0.0, 0.0), label="residual"):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), 0.0, 0.0],
        "normal": list(normal),
        "curvature_proxy": 0.02,
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
        "owned_bend_regions": [{"bend_id": "OB1", "owned_atom_ids": ["A0", "A1"]}],
        "assignment": {
            "atom_labels": atom_labels,
            "label_types": {"residual": "residual", "F1": "flange", "F2": "flange"},
        },
    }


def test_trace_transition_edge_candidates_extracts_high_normal_jump_chain():
    atoms = [
        _atom("A0", 0, normal=(1, 0, 0), label="F1"),
        _atom("A1", 10, normal=(0, 1, 0), label="residual"),
        _atom("A2", 20, normal=(1, 0, 0), label="residual"),
        _atom("A3", 30, normal=(0, 1, 0), label="F2"),
    ]

    report = tracer.trace_transition_edge_candidates(
        decomposition=_payload(atoms),
        normal_jump_deg=45.0,
        min_edges=3,
        min_span_mm=20.0,
        max_thickness_mm=2.0,
    )

    assert report["status"] == "edge_candidates_found"
    assert report["admissible_candidate_count"] == 1
    assert report["candidates"][0]["edge_count"] == 3


def test_trace_transition_edge_candidates_can_exclude_owned_atoms():
    atoms = [
        _atom("A0", 0, normal=(1, 0, 0), label="F1"),
        _atom("A1", 10, normal=(0, 1, 0), label="residual"),
        _atom("A2", 20, normal=(1, 0, 0), label="residual"),
        _atom("A3", 30, normal=(0, 1, 0), label="F2"),
    ]

    report = tracer.trace_transition_edge_candidates(
        decomposition=_payload(atoms),
        normal_jump_deg=45.0,
        min_edges=1,
        min_span_mm=1.0,
        exclude_owned_bend_atoms=True,
    )

    assert report["exclude_owned_bend_atoms"] is True
    remaining_atoms = {atom_id for candidate in report["candidates"] for atom_id in candidate["atom_ids"]}
    assert not (remaining_atoms & {"A0", "A1"})
