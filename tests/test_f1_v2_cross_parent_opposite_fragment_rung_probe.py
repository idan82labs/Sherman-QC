from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import json

import pytest

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as complete
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as rung


def _fixture():
    parents = []
    modes = []
    edges = []
    for index, z_value in enumerate((0.0, 1.0)):
        parent_id = f"P_{index}"
        left_mode = f"M_{index}L"
        right_mode = f"M_{index}R"
        parents.append({
            "parent_atom_id": parent_id, "centroid": [0.0, 0.0, z_value],
            "point_count": 20, "normalized_support_mass": 1,
            "mode_child_ids": [left_mode, right_mode], "mode_overflow": False,
            "transition_certificate_allowed": True, "voxel_key": [0, 0, index],
            "voxel_level": 0, "voxel_size_mm": 1.0,
        })
        modes.extend([
            {"mode_atom_id": left_mode, "parent_atom_id": parent_id,
             "centroid": [0.0, 0.0, z_value], "normal_projective": [1.0, 0.0, 0.0],
             "point_count": 10, "surface_fit_p90_mm": 0.0},
            {"mode_atom_id": right_mode, "parent_atom_id": parent_id,
             "centroid": [0.0, 0.0, z_value], "normal_projective": [0.0, 1.0, 0.0],
             "point_count": 10, "surface_fit_p90_mm": 0.0},
        ])
        edges.append({
            "mode_edge_id": f"E_cross{index}", "mode_ids": [left_mode, right_mode],
            "edge_kind": "CROSS_MODE_LOCAL", "spatial_residual": 0.0,
            "normal_residual": 0.0, "plane_residual": 0.0,
        })
    projected = {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "support_parent_table": parents, "support_mode_table": modes,
        "support_mode_edges": edges, "plane_table": [],
        "transition_parent_table_sha256": lpf._table_sha256(parents, "parent_atom_id"),
        "transition_mode_table_sha256": lpf._table_sha256(modes, "mode_atom_id"),
        "transition_mode_edge_table_sha256": lpf._table_sha256(edges, "mode_edge_id"),
        "transition_local_spacing_mm": 1.0,
        "transition_atom_caps": dict(lpf.TRANSITION_ATOM_CAPS),
        "working_point_budget": 200_000, "transient_sampled_point_count": 40,
    }
    hashes = {"scan_sha256": "a" * 64, "candidate_pool_sha256": "b" * 64,
              "a0_prediction_sha256": "c" * 64}
    control = lpf.build_local_plane_fragment_probe(projected, **hashes)
    control_sha = hashlib.sha256(lpf.canonical_json_bytes(control)).hexdigest()
    baseline = complete.build_v2_complete_same_surface_edge_probe(
        projected, **hashes, frozen_control_probe_sha256=control_sha)
    baseline_sha = hashlib.sha256(complete.canonical_json_bytes(baseline)).hexdigest()
    build_kwargs = {
        **hashes, "case_token": "C999", "frozen_control_probe_sha256": control_sha,
        "complete_edge_baseline_probe_sha256": baseline_sha,
    }
    return projected, control, baseline, build_kwargs


def test_full_fixed_pair_probe_adds_only_authorized_rungs() -> None:
    projected, _control, baseline, kwargs = _fixture()
    payload = rung.build_v2_cross_parent_opposite_fragment_rung_probe(projected, **kwargs)
    rung.validate_v2_cross_parent_opposite_fragment_rung_probe_payload(
        payload, projected_input=projected)
    counters = payload["case_counters"]
    assert counters["frozen_target_count"] == 2
    assert counters["mapped_frozen_target_count"] == 2
    assert counters["target_eligible_pair_count"] == 1
    assert counters["added_cross_parent_rung_count"] == 2
    assert counters["rung_candidate_evaluation_count"] == 2
    assert counters["one_parent_finite_span_count"] == 0
    assert payload["baseline_local_plane_fragment_table"] == baseline["local_plane_fragment_table"]
    assert payload["baseline_local_plane_orphan_table"] == baseline["local_plane_orphan_table"]
    assert [row["local_plane_pair_id"] for row in payload["local_plane_pair_records"]] == [
        row["local_plane_pair_id"] for row in baseline["local_plane_pair_records"]
    ]
    for row in payload["added_cross_parent_rung_table"]:
        assert set(row) == rung.CROSS_PARENT_RUNG_FIELDS
        assert row["mode_edge_id"].startswith("XR_")
        assert row["edge_kind"] == "CROSS_MODE_ADJACENT_PARENT"
        assert row["parent_atom_ids"][0] != row["parent_atom_ids"][1]
        assert row["parent_voxel_chebyshev_distance"] == 1
        assert row["baseline_fragment_ids"] == payload["local_plane_pair_records"][0]["fragment_ids"]


def test_axial_binary64_equality_and_nextafter_semantics() -> None:
    width = 3.0
    assert rung.axial_separation_is_eligible(width, width) is True
    assert rung.axial_separation_is_eligible(math.nextafter(width, math.inf), width) is False


def test_cross_parent_rung_contributes_exactly_one_midpoint_bin() -> None:
    assert rung.cross_parent_midpoint_bin(0.0, 4.0, 0.0, 3.0) == 0
    # The endpoints occupy two different bins, but the rung contributes only
    # the midpoint bin rather than both endpoint bins.
    assert {int(math.floor(value / 3.0 + 1e-9)) for value in (0.0, 4.0)} == {0, 1}


def test_deletion_counterfactual_restores_complete_edge_baseline() -> None:
    projected, _control, baseline, kwargs = _fixture()
    payload = rung.build_v2_cross_parent_opposite_fragment_rung_probe(projected, **kwargs)
    deletion = payload["global_deletion_counterfactual"]
    assert deletion["global_deletion_counterfactual_valid"] is True
    assert deletion["deleted_cross_parent_rung_count"] == 2
    assert deletion["restored_complete_edge_table_sha256"] == (
        baseline["replacement_edge_table_hashes"]["effective_mode_edge_table_sha256"]
    )
    assert deletion["fragment_table_restored"] is True
    assert deletion["orphan_table_restored"] is True
    assert deletion["pair_component_records_restored"] is True
    assert deletion["component_certificate_census_restored"] is True


def test_input_table_order_and_projective_sign_flip_are_output_invariant() -> None:
    projected, _control, _baseline, kwargs = _fixture()
    reference = rung.build_v2_cross_parent_opposite_fragment_rung_probe(projected, **kwargs)
    reordered = deepcopy(projected)
    reordered["support_parent_table"].reverse()
    reordered["support_mode_table"].reverse()
    reordered["support_mode_edges"].reverse()
    assert rung.build_v2_cross_parent_opposite_fragment_rung_probe(reordered, **kwargs) == reference

    flipped = deepcopy(projected)
    for mode in flipped["support_mode_table"]:
        mode["normal_projective"] = [-value for value in mode["normal_projective"]]
    flipped["transition_mode_table_sha256"] = lpf._table_sha256(
        flipped["support_mode_table"], "mode_atom_id")
    control = lpf.build_local_plane_fragment_probe(
        flipped, scan_sha256=kwargs["scan_sha256"],
        candidate_pool_sha256=kwargs["candidate_pool_sha256"],
        a0_prediction_sha256=kwargs["a0_prediction_sha256"])
    control_sha = hashlib.sha256(lpf.canonical_json_bytes(control)).hexdigest()
    baseline = complete.build_v2_complete_same_surface_edge_probe(
        flipped, scan_sha256=kwargs["scan_sha256"],
        candidate_pool_sha256=kwargs["candidate_pool_sha256"],
        a0_prediction_sha256=kwargs["a0_prediction_sha256"],
        frozen_control_probe_sha256=control_sha)
    baseline_sha = hashlib.sha256(complete.canonical_json_bytes(baseline)).hexdigest()
    flipped_payload = rung.build_v2_cross_parent_opposite_fragment_rung_probe(
        flipped, **{**kwargs, "frozen_control_probe_sha256": control_sha,
                    "complete_edge_baseline_probe_sha256": baseline_sha})
    assert flipped_payload["output_table_hashes"] == reference["output_table_hashes"]


def test_caps_fail_closed_without_structural_truncation(monkeypatch) -> None:
    projected, control, baseline, _kwargs = _fixture()
    context = rung._baseline_context(projected, baseline)
    pair_contexts = {
        pair_id: rung._pair_context(pair, context)
        for pair_id, pair in context["pair_by_id"].items()
    }
    targets = rung._frozen_targets(control, "C999")
    mappings = rung._map_targets(targets, pair_contexts)
    eligible = {pair_id for pair_id, _component in mappings.values()}
    monkeypatch.setattr(rung, "MAX_CROSS_PARENT_RUNG_EVALUATIONS", 0)
    with pytest.raises(ValueError, match="evaluation cap"):
        rung._build_rungs(context, pair_contexts, eligible)


def test_real_c001_exact_zero_mapping_fails_closed() -> None:
    root = Path("output/research/f1-gold27-20260803")
    control_manifest = json.loads((root / "local_plane_fragment_probe_v2/manifest.json").read_text())
    row = control_manifest["cases"][0]
    projected = rung.project_cross_parent_rung_input(json.loads(Path(row["candidate_pool_path"]).read_text()))
    control = json.loads((root / "local_plane_fragment_probe_v2/C001.json").read_text())
    baseline = json.loads((root / "v2_complete_same_surface_edge_probe_v1/C001.json").read_text())
    context = rung._baseline_context(projected, baseline)
    pair_contexts = {pair_id: rung._pair_context(pair, context)
                     for pair_id, pair in context["pair_by_id"].items()}
    targets = rung._frozen_targets(control, "C001")
    assert len(targets) == 260
    with pytest.raises(ValueError, match="exactly one"):
        rung._map_targets(targets, pair_contexts)
    blocked = next(target for target in targets
                   if target["frozen_target_component_id"] == "LTC_32d3609a79a9c314")
    assert blocked["frozen_target_mode_ids"] == [
        "M_73e7f7dd53e69d0f", "M_d5d9b9a524d05ccc"]

