"""Invented positions only; no application, native CAD, or physical inputs."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_instances.py"


def _module():
    name = "cad_guided_instances_isolated_test"
    module = ModuleType(name)
    module.__file__ = str(SOURCE)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), module.__dict__)
    finally:
        if prior is None:
            del sys.modules[name]
        else:
            sys.modules[name] = prior
    return module


def _fixture(stations=(0., 2., 4., 6., 8.), angles=None):
    angles = np.deg2rad(np.linspace(10., 80., 15) if angles is None else angles)
    arc = np.array([[6 * np.cos(t), 6 * np.sin(t), z] for z in stations for t in angles])
    first = np.array([[6., -d, z] for z in stations for d in (3., 4., 5., 6., 7.)])
    second = np.array([[-d, 6., z] for z in stations for d in (3., 4., 5., 6., 7.)])
    points = np.vstack((arc, first, second))
    ids = [np.arange(len(arc)), np.arange(len(arc), len(arc) + len(first)),
           np.arange(len(arc) + len(first), len(points))]
    measured = {"measured_axis_point": [0., 0., 0.], "measured_axis_direction": [0., 0., 1.],
                "measured_radius_mm": 6., "circle_max_residual_mm": 1e-10}
    return points, measured, ids


def _build(module, points, measured, ids, policy=None):
    return module.build_finite_support_instance(
        points, measured, ids[0], ids[1:], feature_id="scan-supported-one",
        policy=policy or module.SupportPolicy(2.1, 6., 30., 3))


def test_endpoints_follow_scan_support_not_declared_nominal_or_measurement_span():
    module = _module()
    points, measured, ids = _fixture()
    measured["sampled_axial_interval_mm"] = [-100., 200.]
    measured["nominal_radius_mm"] = 5.
    result = _build(module, points, measured, ids)
    assert result["status"] == "SUPPORTED", result
    instance = result["instance"]
    np.testing.assert_allclose([instance["line_start"], instance["line_end"]], [[0, 0, 0], [0, 0, 8]])
    assert instance["line_convention"] == "CYLINDER_CENTER_AXIS_SUPPORT"
    assert instance["endpoint_semantics"] == "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM"
    assert instance["physical_trim_endpoints"] is None
    assert instance["manufacturing_event_identity"] == "UNKNOWN"
    assert instance["continuous_surface_coverage"] == "NOT_ESTABLISHED"
    assert len(instance["local_coverage"]) == 4
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("offset", [-100., 7., 1000.])
def test_axis_point_gauge_does_not_change_finite_endpoints(offset):
    module = _module()
    points, measured, ids = _fixture()
    measured["measured_axis_point"] = [0., 0., offset]
    result = _build(module, points, measured, ids)
    assert result["status"] == "SUPPORTED", result
    np.testing.assert_allclose([result["instance"]["line_start"], result["instance"]["line_end"]],
                               [[0, 0, 0], [0, 0, 8]], atol=1e-12)


def test_single_partial_interval_is_not_padded_to_a_whole_cad_span():
    module = _module()
    result = _build(module, *_fixture(stations=(2., 4., 6.)))
    assert result["status"] == "SUPPORTED", result
    assert result["instance"]["line_start"] == [0., 0., 2.]
    assert result["instance"]["line_end"] == [0., 0., 6.]


def test_disconnected_axial_patches_do_not_become_one_overlong_line_or_two_bends():
    module = _module()
    result = _build(module, *_fixture(stations=(0., 2., 6., 8.)))
    assert result == {"status": "UNAVAILABLE", "instance": None,
                      "reasons": ["DISCONNECTED_AXIAL_SUPPORT"]}


def test_global_angular_extrema_cannot_hide_missing_local_sweep():
    module = _module()
    points, measured, ids = _fixture()
    arc = ids[0]
    # First station covers full arc, later stations only a 15-degree patch.
    theta = np.rad2deg(np.arctan2(points[arc, 1], points[arc, 0]))
    keep = (points[arc, 2] == 0) | (theta <= 25.1)
    ids[0] = arc[keep]
    result = _build(module, points, measured, ids)
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["INSUFFICIENT_LOCAL_ANGULAR_COVERAGE"]


def test_angular_void_between_two_arcs_is_not_continuous_support():
    module = _module()
    result = _build(module, *_fixture(angles=[10., 15., 20., 70., 75., 80.]))
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["DISCONNECTED_LOCAL_ANGULAR_SUPPORT"]


@pytest.mark.parametrize("role", [1, 2])
def test_each_local_slab_requires_both_flange_contexts(role):
    module = _module()
    points, measured, ids = _fixture()
    ids[role] = ids[role][points[ids[role], 2] <= 2.]
    result = _build(module, points, measured, ids)
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["INSUFFICIENT_LOCAL_ARC_OR_FLANGE_SUPPORT"]


def test_repeated_coordinates_do_not_inflate_sampling_evidence():
    module = _module()
    points, measured, ids = _fixture(angles=[10., 80.])
    expanded, expanded_ids = [], []
    offset = 0
    for rows in ids:
        block = np.repeat(points[rows], 30, axis=0)
        expanded.append(block)
        expanded_ids.append(np.arange(offset, offset + len(block)))
        offset += len(block)
    result = _build(module, np.vstack(expanded), measured, expanded_ids)
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT"]


def test_duplicate_rows_within_each_role_leave_valid_evidence_unchanged():
    module = _module()
    points, measured, ids = _fixture()
    baseline = _build(module, points, measured, ids)
    expanded = np.vstack([points[rows].repeat(4, axis=0) for rows in ids])
    cuts = np.cumsum([0, *[len(rows) * 4 for rows in ids]])
    result = _build(module, expanded, measured, [np.arange(cuts[i], cuts[i + 1]) for i in range(3)])
    assert result == baseline


def test_support_row_role_reuse_is_rejected():
    module = _module()
    points, measured, ids = _fixture()
    ids[1] = np.r_[ids[1], ids[0][:3]]
    result = _build(module, points, measured, ids)
    assert result["reasons"] == ["OVERLAPPING_SUPPORT_ROLES"]


def test_support_position_role_reuse_with_new_ids_is_rejected():
    module = _module()
    points, measured, ids = _fixture()
    extra = points[ids[0][:3]].copy()
    ids[1] = np.r_[ids[1], np.arange(len(points), len(points) + len(extra))]
    result = _build(module, np.vstack((points, extra)), measured, ids)
    assert result["reasons"] == ["OVERLAPPING_SUPPORT_POSITIONS"]


@pytest.mark.parametrize("change", ["nonunit", "wrong_radius", "nan", "all_one_station"])
def test_invalid_geometry_cannot_become_an_instance(change):
    module = _module()
    points, measured, ids = _fixture()
    if change == "nonunit":
        measured["measured_axis_direction"] = [0., 0., 2.]
    elif change == "wrong_radius":
        measured["measured_radius_mm"] = 5.
    elif change == "nan":
        points[0, 0] = np.nan
    else:
        points[:, 2] = 0.
    result = _build(module, points, measured, ids)
    assert result["status"] == "UNAVAILABLE"
    assert result["instance"] is None


@pytest.mark.parametrize("angle", [0., 1e-11, 13., 57., 89., 120., 179., 270.])
def test_rigid_pose_and_unwrap_invariance_including_axial_boundary_roundoff(angle):
    module = _module()
    points, measured, ids = _fixture()
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), 0, np.sin(radians)], [0, 1, 0],
                         [-np.sin(radians), 0, np.cos(radians)]])
    translation = np.array([345., -721., 233.])
    points = points @ rotation.T + translation
    measured["measured_axis_point"] = translation.tolist()
    measured["measured_axis_direction"] = rotation[:, 2].tolist()
    result = _build(module, points, measured, ids)
    assert result["status"] == "SUPPORTED", result
    expected = np.array([[0., 0., 0.], [0., 0., 8.]]) @ rotation.T + translation
    np.testing.assert_allclose([result["instance"]["line_start"], result["instance"]["line_end"]],
                               expected, atol=1e-10)


@pytest.mark.parametrize("name,value", [("max_axial_gap_mm", 0.), ("max_axial_gap_mm", True),
                                      ("max_angular_gap_deg", float("nan")),
                                      ("min_arc_sweep_deg", 181.), ("min_local_points", 2),
                                      ("min_local_points", 3.)])
def test_policy_is_explicit_and_bounded(name, value):
    module = _module()
    with pytest.raises(ValueError):
        replace(module.SupportPolicy(2.1, 6., 30., 3), **{name: value})


def test_point_and_slab_caps_abstain_without_prefix_selection():
    module = _module()
    points, measured, ids = _fixture()
    result = _build(module, np.zeros((200_001, 3)), measured, ids)
    assert result["reasons"] == ["INVALID_POINT_ARRAY_OR_BUDGET"]
    dense, measured, ids = _fixture(stations=np.linspace(0., 8., 2050))
    result = _build(module, dense, measured, ids, module.SupportPolicy(.003905, 6., 30., 3))
    assert result["reasons"] == ["AXIAL_SLAB_BUDGET"]


def test_numeric_guard_cannot_hide_geometry_at_unresolvable_radius_scale():
    module = _module()
    points, measured, ids = _fixture()
    points[:, :2] *= 1e-8 / 6.
    points[:, 0] += 999_999.
    measured.update(measured_axis_point=[999_999., 0., 0.], measured_radius_mm=1e-8)
    result = _build(module, points, measured, ids)
    assert result["reasons"] == ["NUMERIC_RESOLUTION_TOO_COARSE"]


def test_ragged_support_ids_are_unavailable_instead_of_partial_success():
    module = _module()
    points, measured, ids = _fixture()
    ids[0] = [[0], [1, 2]]
    result = _build(module, points, measured, ids)
    assert result["status"] == "UNAVAILABLE" and result["instance"] is None
