from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as edge


ZERO_SHA = "0" * 64


def _parent(parent_id: str, key: tuple[int, int, int], mode_ids: list[str]) -> dict:
    return {
        "parent_atom_id": parent_id, "centroid": [float(key[0]), float(key[1]), float(key[2])],
        "point_count": 10, "normalized_support_mass": 1, "mode_child_ids": mode_ids,
        "mode_overflow": False, "transition_certificate_allowed": True,
        "voxel_key": list(key), "voxel_level": 0, "voxel_size_mm": 1.0,
    }


def _mode(mode_id: str, parent_id: str, centroid, normal) -> dict:
    return {
        "mode_atom_id": mode_id, "parent_atom_id": parent_id,
        "centroid": list(centroid), "normal_projective": list(normal),
        "point_count": 10, "surface_fit_p90_mm": 0.0,
    }


def _pool(parents, modes, edges=(), *, spacing=1.0) -> dict:
    parents, modes, edges = list(parents), list(modes), list(edges)
    return {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "support_parent_table": parents, "support_mode_table": modes,
        "support_mode_edges": edges, "plane_table": [],
        "transition_parent_table_sha256": lpf._table_sha256(parents, "parent_atom_id"),
        "transition_mode_table_sha256": lpf._table_sha256(modes, "mode_atom_id"),
        "transition_mode_edge_table_sha256": lpf._table_sha256(edges, "mode_edge_id"),
        "transition_local_spacing_mm": spacing,
        "transition_atom_caps": dict(lpf.TRANSITION_ATOM_CAPS),
        "working_point_budget": 200_000, "transient_sampled_point_count": 20,
    }


def _two_mode_pool(normal=(1.0, 0.0, 0.0), right_centroid=(0.0, 1.0, 0.0)):
    parents = [_parent("P_a", (0, 0, 0), ["M_a"]), _parent("P_b", (0, 1, 0), ["M_b"])]
    modes = [_mode("M_a", "P_a", (0, 0, 0), (1, 0, 0)),
             _mode("M_b", "P_b", right_centroid, normal)]
    return _pool(parents, modes)


def test_angle_equal_to_15_is_eligible() -> None:
    radians = math.radians(15.0)
    pool = _two_mode_pool((math.cos(radians), math.sin(radians), 0.0))
    rows = edge.build_complete_same_surface_edges(pool)
    assert len(rows) == 1
    assert rows[0]["normal_residual"] <= 1.0


def test_angle_nextafter_above_15_is_ineligible() -> None:
    angle = math.nextafter(15.0, math.inf)
    while True:
        radians = math.radians(angle)
        normal = (math.cos(radians), math.sin(radians), 0.0)
        if lpf.projective_angle_deg((1, 0, 0), normal) > 15.0:
            break
        angle = math.nextafter(angle, math.inf)
    assert edge.build_complete_same_surface_edges(_two_mode_pool(normal)) == []


def test_gap_equal_to_tau_is_eligible_and_above_is_not() -> None:
    equal = _two_mode_pool((0, 0, 1), (0, 1, 1.5))
    equal["support_mode_table"][0]["normal_projective"] = [0, 0, 1]
    equal["transition_mode_table_sha256"] = lpf._table_sha256(equal["support_mode_table"], "mode_atom_id")
    assert len(edge.build_complete_same_surface_edges(equal)) == 1
    above = deepcopy(equal)
    above["support_mode_table"][1]["centroid"][2] = math.nextafter(1.5, math.inf)
    above["transition_mode_table_sha256"] = lpf._table_sha256(above["support_mode_table"], "mode_atom_id")
    assert edge.build_complete_same_surface_edges(above) == []


def test_complete_builder_has_no_degree_truncation() -> None:
    parents = [_parent("P_00", (0, 0, 0), ["M_00"])]
    modes = [_mode("M_00", "P_00", (0, 0, 0), (0, 0, 1))]
    neighbor_keys = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                     for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)][:9]
    for index, key in enumerate(neighbor_keys, 1):
        pid, mid = f"P_{index:02d}", f"M_{index:02d}"
        parents.append(_parent(pid, key, [mid]))
        modes.append(_mode(mid, pid, key, (0, 0, 1)))
    rows = edge.build_complete_same_surface_edges(_pool(parents, modes))
    central_degree = sum("M_00" in row["mode_ids"] for row in rows)
    assert central_degree == 9


def test_inherited_subset_and_cross_mode_are_byte_identical() -> None:
    pool = _two_mode_pool()
    same = edge.build_complete_same_surface_edges(pool)[0]
    # A same-parent rung is added on an otherwise independent third parent.
    pool["support_parent_table"].append(_parent("P_c", (4, 4, 4), ["M_c1", "M_c2"]))
    pool["support_mode_table"].extend([
        _mode("M_c1", "P_c", (4, 4, 4), (1, 0, 0)),
        _mode("M_c2", "P_c", (4, 4, 4), (0, 1, 0)),
    ])
    cross = {"mode_edge_id": "E_cross", "mode_ids": ["M_c1", "M_c2"],
             "edge_kind": "CROSS_MODE_LOCAL", "spatial_residual": 0.0,
             "normal_residual": 0.0, "plane_residual": 0.0}
    pool["support_mode_edges"] = [deepcopy(same), deepcopy(cross)]
    pool["transition_parent_table_sha256"] = lpf._table_sha256(pool["support_parent_table"], "parent_atom_id")
    pool["transition_mode_table_sha256"] = lpf._table_sha256(pool["support_mode_table"], "mode_atom_id")
    pool["transition_mode_edge_table_sha256"] = lpf._table_sha256(pool["support_mode_edges"], "mode_edge_id")
    validated = edge._validate_projected_input(pool)
    _complete, inherited_same, inherited_cross, _added, effective, *_ = edge._edge_tables(pool, validated)
    assert edge.canonical_json_bytes(inherited_same[0]) == edge.canonical_json_bytes(same)
    assert edge.canonical_json_bytes(inherited_cross[0]) == edge.canonical_json_bytes(cross)
    assert next(row for row in effective if row["mode_edge_id"] == "E_cross") == cross


def test_order_and_projective_sign_are_output_table_invariant() -> None:
    pool = _two_mode_pool()
    reference = edge.build_complete_same_surface_edges(pool)
    reordered = deepcopy(pool)
    reordered["support_parent_table"].reverse(); reordered["support_mode_table"].reverse()
    assert edge.build_complete_same_surface_edges(reordered) == reference
    flipped = deepcopy(pool)
    for mode in flipped["support_mode_table"]:
        mode["normal_projective"] = [-value for value in mode["normal_projective"]]
    flipped["transition_mode_table_sha256"] = lpf._table_sha256(flipped["support_mode_table"], "mode_atom_id")
    assert edge.build_complete_same_surface_edges(flipped) == reference


def test_projection_rejects_missing_and_core_rejects_extra() -> None:
    pool = _two_mode_pool()
    missing = deepcopy(pool); missing.pop("plane_table")
    with pytest.raises(ValueError, match="missing fields"):
        edge.project_edge_probe_input(missing)
    extra = deepcopy(pool); extra["truth"] = 1
    with pytest.raises(ValueError, match="exact allowlisted"):
        edge.build_complete_same_surface_edges(extra)


def test_real_c001_determinism_control_and_one_parent_invariant() -> None:
    root = Path("output/research/f1-gold27-20260803")
    manifest = json.loads((root / "local_plane_fragment_probe_v2/manifest.json").read_text())
    row = manifest["cases"][0]
    projected = edge.project_edge_probe_input(json.loads(Path(row["candidate_pool_path"]).read_text()))
    kwargs = dict(scan_sha256=row["scan_sha256"], candidate_pool_sha256=row["candidate_pool_sha256"],
                  a0_prediction_sha256=row["a0_prediction_sha256"],
                  frozen_control_probe_sha256=row["probe_sha256"])
    first = edge.build_v2_complete_same_surface_edge_probe(projected, **kwargs)
    second = edge.build_v2_complete_same_surface_edge_probe(projected, **kwargs)
    assert edge.canonical_json_bytes(first) == edge.canonical_json_bytes(second)
    assert first["case_counters"]["complete_same_surface_edge_count"] == 7632
    assert first["case_counters"]["added_same_surface_edge_count"] == 2535
    assert first["case_counters"]["one_parent_finite_span_count"] == 0
    evidence = edge.verify_edge_deletion_counterfactual(
        projected, first, scan_sha256=row["scan_sha256"],
        candidate_pool_sha256=row["candidate_pool_sha256"],
        a0_prediction_sha256=row["a0_prediction_sha256"],
        expected_control_probe_sha256=row["probe_sha256"])
    assert evidence["valid"] is True

