import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_f1_uncovered_bend_corridors.py"
    spec = importlib.util.spec_from_file_location("analyze_f1_uncovered_bend_corridors", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _atom(atom_id, x, y=0.0, z=0.0, normal=(0.707, 0.707, 0.0), label="residual"):
    return {
        "atom_id": atom_id,
        "centroid": [float(x), float(y), float(z)],
        "normal": list(normal),
        "axis_direction": [0.0, 0.0, 1.0],
        "curvature_proxy": 0.04,
        "weight": 100.0,
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
        "flange_hypotheses": [
            {"flange_id": "F1", "normal": [1.0, 0.0, 0.0], "d": 0.0, "atom_ids": []},
            {"flange_id": "F2", "normal": [0.0, 1.0, 0.0], "d": 0.0, "atom_ids": []},
        ],
        "owned_bend_regions": [],
        "assignment": {
            "atom_labels": labels,
            "label_types": {"residual": "residual", "F1": "flange", "F2": "flange"},
        },
    }


def test_uncovered_corridor_finds_simple_candidate():
    atoms = [_atom(f"A{i}", 0.0, y=0.0, z=i * 4.0) for i in range(8)]
    payload = _payload(atoms)
    payload["flange_hypotheses"][0]["atom_ids"] = ["FA"]
    payload["flange_hypotheses"][1]["atom_ids"] = ["FB"]
    payload["atom_graph"]["atoms"].extend(
        [
            _atom("FA", 0.0, y=-2.0, z=12.0, normal=(1.0, 0.0, 0.0), label="F1"),
            _atom("FB", 0.0, y=2.0, z=12.0, normal=(0.0, 1.0, 0.0), label="F2"),
        ]
    )
    payload["assignment"]["atom_labels"]["FA"] = "F1"
    payload["assignment"]["atom_labels"]["FB"] = "F2"
    payload["atom_graph"]["adjacency"]["A3"].extend(["FA", "FB"])
    payload["atom_graph"]["adjacency"]["FA"] = ["A3"]
    payload["atom_graph"]["adjacency"]["FB"] = ["A3"]

    report = analyzer.analyze_uncovered_bend_corridors(decomposition=payload)

    assert report["candidate_count"] >= 1
    assert report["admissible_candidate_count"] >= 1
    assert report["status"] == "candidate_pool_has_uncovered_corridors"


def test_uncovered_corridor_can_exclude_existing_owned_atoms():
    atoms = [_atom(f"A{i}", 0.0, y=0.0, z=i * 4.0) for i in range(8)]
    payload = _payload(atoms)
    payload["owned_bend_regions"] = [{"bend_id": "OB1", "owned_atom_ids": [atom["atom_id"] for atom in atoms]}]

    report = analyzer.analyze_uncovered_bend_corridors(
        decomposition=payload,
        exclude_owned_bend_atoms=True,
    )

    assert report["candidate_count"] == 0
    assert report["status"] == "no_uncovered_corridor_candidate"
