"""Invented-only audit of the full-row local-window measurement adapter.

Loads only the named pure backend modules through a private package: no backend
initialization, native CAD, files/data fixtures, scorer, registration, or physical
authority. Real analytic fits check the mathematical seam; monkeypatched kernels
check bounded ownership/denominator behavior, not measurement accuracy. Root owns
execution and must bind this file before running its focused harness.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


_BACKEND = Path(__file__).resolve().parents[1] / "backend"
_PACKAGE = "_cad_guided_local_measurement_v1_tests"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(_BACKEND)]
sys.modules[_PACKAGE] = _package
_spec = importlib.util.spec_from_file_location(
    _PACKAGE + ".cad_guided_local_measurement_v1",
    _BACKEND / "cad_guided_local_measurement_v1.py",
)
assert _spec is not None and _spec.loader is not None
adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = adapter
_spec.loader.exec_module(adapter)


SCAN_SHA = "a" * 64


def pose(oblique=False):
    if not oblique:
        return np.eye(3), np.zeros(3)
    direction = np.array([2., -1., 3.]) / math.sqrt(14.)
    x, y, z = direction
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = .73
    rotation = (math.cos(angle) * np.eye(3)
                + (1. - math.cos(angle)) * np.outer(direction, direction)
                + math.sin(angle) * cross)
    return rotation, np.array([155., -235., 95.])


def feature(identity="invented:0", *, sweep=90., oblique=False):
    rotation, origin = pose(oblique)
    return adapter.CADGuidedFeature(
        feature_id=identity, nominal_radius_mm=5., nominal_sweep_deg=sweep,
        origin=tuple(origin), axis=tuple(rotation[:, 2]),
        start_radial=tuple(rotation[:, 0]), span_mm=(0., 30.),
        flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
        max_fit_residual_mm=.05, plane_exclusion_mm=.075,
        max_tangency_error_mm=.1, nominal_source="invented declaration only",
    )


def policy():
    return adapter.SupportPolicy(max_axial_gap_mm=4., max_angular_gap_deg=5.,
                                 min_arc_sweep_deg=20., min_local_points=3)


def scan(*, nominal_sweep=90., measured_radius=6., sweep_change=5.,
         oblique=False, center_shift=(.4, -.3, 7.)):
    """Independent circular extrusion plus its true tangent planes, 1,035 rows.

    Nominal R=5; actual R=6 and sweep changes by signed5 degrees. The axial
    center shift deliberately makes the circle's arbitrary axis point differ
    from the first observed station. No source fit supplies fixture truth.
    """
    sign = math.copysign(1., nominal_sweep)
    actual_sweep = sign * (abs(nominal_sweep) + sweep_change)
    phi = math.radians(actual_sweep)
    axial = np.linspace(0., 30., 9)
    arc = np.array([
        [measured_radius * math.cos(t), measured_radius * math.sin(t), z]
        for t in np.linspace(0., phi, 81) for z in axial
    ])
    first = np.array([measured_radius, 0., 0.])
    second = measured_radius * np.array([math.cos(phi), math.sin(phi), 0.])
    rays = (np.array([0., -sign, 0.]),
            sign * np.array([-math.sin(phi), math.cos(phi), 0.]))
    flanges = [np.array([contact + distance * ray + [0., 0., z]
                        for distance in np.linspace(3., 12., 17) for z in axial])
               for contact, ray in zip((first, second), rays)]
    rotation, translation = pose(oblique)
    shift = np.asarray(center_shift)
    points = (np.vstack((arc, *flanges)) + shift) @ rotation.T + translation
    center = shift @ rotation.T + translation
    offset = measured_radius * np.array([math.cos(phi / 2), math.sin(phi / 2), 0.])
    return points, {"radius": measured_radius, "sweep": actual_sweep,
                    "axis": rotation[:, 2], "center": center,
                    "surface_offset": offset @ rotation.T}


def inspect(points, features=None, **kwargs):
    return adapter.inspect_full_raw_windows_v1(
        points, [feature()] if features is None else features,
        instance_policy=kwargs.pop("instance_policy", policy()),
        scan_sha256=kwargs.pop("scan_sha256", SCAN_SHA),
        registration_status=kwargs.pop("registration_status", "CONDITIONAL_BEST_FIT"),
        **kwargs,
    )


def assert_no_authority(result):
    assert result["observed_bend_count"] is None
    assert result["observed_bend_count_status"] == "MANUFACTURING_EVENT_IDENTITY_UNKNOWN"
    assert result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE"
    assert result["uncertainty_status"] == "UNCALIBRATED"
    assert result["reference_carrier_compatibility"] == "UNVERIFIED_NO_GOLD_IN_CARRIER_SELECTION"
    assert result["surface_side_identity"] == "CALLER_ASSUMED_NOT_VERIFIED"
    assert result["datum_relative_deviation_status"] == "PROVISIONAL_NOT_STABLE_DATUM_METROLOGY"


@pytest.mark.parametrize("sweep", [-135., -90., 90., 135.])
@pytest.mark.parametrize("oblique", [False, True])
def test_actual_fit_and_surface_carrier_use_measured_radius_and_signed_sweep(sweep, oblique):
    declared = feature(sweep=sweep, oblique=oblique)
    points, truth = scan(nominal_sweep=sweep, oblique=oblique)
    before = points.copy()
    result = inspect(points, [declared])
    assert_no_authority(result)
    assert result["expected_window_count"] == 1
    assert result["measured_conditional_window_count"] == 1
    assert result["observed_support_instance_count"] == 1
    row = result["features"][0]
    measured = row["measurements"]
    assert measured["measured_radius_mm"] == pytest.approx(truth["radius"], abs=1e-7)
    assert measured["measured_sweep_deg"] == pytest.approx(truth["sweep"], abs=1e-8)
    assert abs(measured["radius_deviation_mm"]) > .99
    assert abs(measured["sweep_deviation_deg"]) == pytest.approx(5., abs=1e-8)
    assert "support_row_indices" not in measured
    assert "_instance_support" not in measured
    instance = result["finite_support_instances"][0]
    axis = np.asarray(measured["measured_axis_direction"])
    np.testing.assert_allclose(axis, truth["axis"], atol=1e-10, rtol=0)
    for endpoint in ("line_start", "line_end"):
        center_axis_endpoint = np.asarray(instance["axis_support_" + endpoint])
        np.testing.assert_allclose(
            np.asarray(instance[endpoint]) - center_axis_endpoint,
            truth["surface_offset"], atol=1e-7, rtol=0,
        )
        # The center's arbitrary along-axis coordinate must not shift the line.
        np.testing.assert_allclose(np.cross(center_axis_endpoint - truth["center"], axis),
                                   np.zeros(3), atol=1e-7, rtol=0)
    arc_points = points[row["raw_support"]["arc_rows"]]
    arc_stations = (arc_points - truth["center"]) @ axis
    axis_endpoints = np.array([instance["axis_support_line_start"],
                              instance["axis_support_line_end"]])
    np.testing.assert_allclose((axis_endpoints - truth["center"]) @ axis,
                               [arc_stations.min(), arc_stations.max()], atol=1e-7, rtol=0)
    assert instance["manufacturing_event_identity"] == "UNKNOWN"
    assert instance["physical_trim_endpoints"] is None
    assert instance["line_convention"] == "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR"
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("sweep", [-135., -90., 90., 135.])
@pytest.mark.parametrize("oblique", [False, True])
def test_crop_is_superset_of_actual_kernel_support(sweep, oblique):
    declared = feature(sweep=sweep, oblique=oblique)
    points, _ = scan(nominal_sweep=sweep, oblique=oblique)
    direct = adapter._measure(points, declared)
    selected = adapter._crop_rows(points, declared)
    cropped = adapter._measure(points[selected], declared)
    support = direct["_instance_support"]
    assert set(direct["support_row_indices"]).issubset(set(selected))
    np.testing.assert_array_equal(selected[np.asarray(cropped["_instance_support"]["arc_row_indices"])],
                                  support["arc_row_indices"])
    for role in range(2):
        np.testing.assert_array_equal(
            selected[np.asarray(cropped["_instance_support"]["flange_row_indices"][role])],
            support["flange_row_indices"][role],
        )
    assert cropped["measured_radius_mm"] == pytest.approx(direct["measured_radius_mm"], abs=1e-10)
    assert cropped["measured_sweep_deg"] == pytest.approx(direct["measured_sweep_deg"], abs=1e-10)


@pytest.mark.parametrize("sweep", [-135., -90., 90., 135.])
def test_crop_preserves_flange_strip_corners_and_whole_radial_disk(sweep):
    declared = feature(sweep=sweep)
    phi = math.radians(sweep)
    sign = math.copysign(1., sweep)
    radial0 = np.array([1., 0., 0.])
    radial1 = np.array([math.cos(phi), math.sin(phi), 0.])
    axis = np.array([0., 0., 1.])
    rays = (-sign * np.cross(axis, radial0), sign * np.cross(axis, radial1))
    corners = [5. * radial + distance * ray + normal_offset * radial + z * axis
               for radial, ray in zip((radial0, radial1), rays)
               for distance in (2.5, 12.5) for normal_offset in (-3., 3.) for z in (0., 30.)]
    # These disk points intentionally need not pass the kernel's angular sector.
    disk = [[0., 0., 0.], [-8., 0., 30.], [0., -8., 15.], [0., 8., 15.]]
    points = np.vstack((corners, disk, [[0., 0., -1.], [0., 0., 31.], [100., 100., 15.]]))
    selected = adapter._crop_rows(points, declared)
    np.testing.assert_array_equal(selected, np.arange(len(corners) + len(disk)))


def test_crop_keeps_last_row_across_chunk_boundary():
    points = np.tile([100., 100., 15.], (65_537, 1))
    points[0] = [0., 0., 0.]
    points[65_535] = [5., -12.5, 15.]
    points[65_536] = [0., 0., 30.]
    np.testing.assert_array_equal(adapter._crop_rows(points, feature()), [0, 65_535, 65_536])


def test_declared_raw_subsequence_maps_support_to_original_ids_and_is_immutable():
    points, _ = scan()
    raw_ids = np.arange(len(points), dtype=np.int64)[::-1] * 3 + 17
    direct = inspect(points)
    rebound = inspect(points, source_raw_row_indices=raw_ids)
    assert rebound["raw_row_identity_sha256"] == hashlib.sha256(raw_ids.astype("<i8").tobytes()).hexdigest()
    assert rebound["scan_sha256"] == SCAN_SHA
    left = direct["features"][0]["raw_support"]
    right = rebound["features"][0]["raw_support"]
    np.testing.assert_array_equal(right["arc_rows"], raw_ids[left["arc_rows"]])
    for direct_rows, rebound_rows in zip(left["flange_rows"], right["flange_rows"]):
        np.testing.assert_array_equal(rebound_rows, raw_ids[direct_rows])
    for rows in (right["arc_rows"], *right["flange_rows"]):
        assert rows.flags.writeable is False
        with pytest.raises(ValueError):
            rows.setflags(write=True)


@pytest.mark.parametrize("axis_shift", [-1000., 0., 1000.])
def test_carrier_ignores_axis_point_gauge_and_signed_axis_reversal(axis_shift):
    flange = np.array([[6., y, z] for y in (-12., -7., -3.) for z in (0., 15., 30.)])
    support = {"flange_row_indices": [list(range(len(flange))), []]}
    measured = {"measured_axis_point": [0., 0., axis_shift],
                "measured_axis_direction": [0., 0., 1.],
                "measured_radius_mm": 6., "measured_sweep_deg": 95.}
    offset = adapter._surface_carrier(measured, flange, support)
    expected = 6. * np.array([math.cos(math.radians(47.5)), math.sin(math.radians(47.5)), 0.])
    np.testing.assert_allclose(offset, expected, atol=1e-12, rtol=0)
    reversed_axis = {**measured, "measured_axis_direction": [0., 0., -1.],
                     "measured_sweep_deg": -95.}
    np.testing.assert_allclose(adapter._surface_carrier(reversed_axis, flange, support),
                               expected, atol=1e-12, rtol=0)


def install_boundary_stubs(monkeypatch, selections, *, unavailable_ids=()):
    """Small deterministic support claims, not an alternate accuracy fixture."""
    measured_calls = []
    monkeypatch.setattr(adapter, "_crop_rows", lambda _points, f: np.asarray(selections[f.feature_id]))

    def measured(_points, f):
        measured_calls.append(f.feature_id)
        if f.feature_id in unavailable_ids:
            raise adapter._Unavailable("INVENTED_KERNEL_REFUSAL")
        return {"measured_axis_point": [0., 0., 0.], "measured_axis_direction": [0., 0., 1.],
                "measured_radius_mm": 6., "measured_sweep_deg": 95.,
                "support_row_indices": list(range(9)),
                "_instance_support": {"arc_row_indices": [0, 1, 2],
                                      "flange_row_indices": [[3, 4, 5], [6, 7, 8]]}}

    def finite(_points, _measured, **kwargs):
        return {"status": "SUPPORTED", "reasons": [],
                "instance": {"bend_id": kwargs["feature_id"],
                             "line_start": [0., 0., 0.], "line_end": [0., 0., 1.]}}

    monkeypatch.setattr(adapter, "_measure", measured)
    monkeypatch.setattr(adapter, "build_finite_support_instance", finite)
    monkeypatch.setattr(adapter, "_surface_carrier", lambda *_: np.array([1., 1., 0.]))
    return measured_calls


def test_shared_curved_rows_reject_both_windows_beyond_old_sixteen_limit(monkeypatch):
    features = [feature(f"window:{index}") for index in range(18)]
    selections = {f.feature_id: np.arange(9 * index, 9 * (index + 1))
                  for index, f in enumerate(features)}
    selections["window:16"][0] = 0  # Only window0 and window16 share a curved row.
    install_boundary_stubs(monkeypatch, selections)
    result = inspect(np.zeros((162, 3)), features,
                     source_raw_row_indices=np.arange(162) * 5 + 7)
    assert result["expected_window_count"] == 18
    assert result["measured_conditional_window_count"] == 16
    assert result["observed_support_instance_count"] == 16
    assert result["raw_input_rows"] == 162
    for index in (0, 16):
        row = result["features"][index]
        assert row["status"] == "UNAVAILABLE"
        assert row["measurements"] is None
        assert row["finite_support"]["instance"] is None
        assert row["reasons"] == ["AMBIGUOUS_SHARED_RAW_FEATURE_SUPPORT"]
    assert_no_authority(result)


@pytest.mark.parametrize("shared_role,expected_measured", [("flange", 2), ("arc", 0)])
def test_shared_planar_context_is_distinct_from_arc_to_flange_reuse(monkeypatch, shared_role,
                                                                  expected_measured):
    features = [feature("first"), feature("second")]
    first, second = np.arange(9), np.arange(9, 18)
    second[3] = first[3 if shared_role == "flange" else 0]
    install_boundary_stubs(monkeypatch, {"first": first, "second": second})
    result = inspect(np.zeros((18, 3)), features)
    assert result["measured_conditional_window_count"] == expected_measured
    assert result["observed_support_instance_count"] == expected_measured
    assert result["expected_window_count"] == 2


def test_refused_kernel_window_remains_in_denominator(monkeypatch):
    features = [feature(f"w:{i}") for i in range(3)]
    selections = {f.feature_id: np.arange(9 * i, 9 * (i + 1)) for i, f in enumerate(features)}
    install_boundary_stubs(monkeypatch, selections, unavailable_ids=("w:1",))
    result = inspect(np.zeros((27, 3)), features)
    assert result["expected_window_count"] == 3
    assert len(result["features"]) == 3
    assert result["measured_conditional_window_count"] == 2
    assert result["features"][1]["reasons"] == ["INVENTED_KERNEL_REFUSAL"]


def test_local_budget_refuses_without_thinning_or_kernel_call(monkeypatch):
    monkeypatch.setattr(adapter, "MAX_LOCAL_ROWS", 8)
    calls = install_boundary_stubs(monkeypatch, {"invented:0": np.arange(9)})
    result = inspect(np.zeros((9, 3)))
    assert calls == []
    assert result["expected_window_count"] == 1
    assert result["total_local_input_rows"] == 9
    assert result["measured_conditional_window_count"] == 0
    assert result["features"][0]["local_input_rows"] == 9
    assert result["features"][0]["reasons"] == ["LOCAL_OR_TOTAL_SUPPORT_BUDGET_NO_THINNING"]


def test_total_budget_preserves_later_dispositions_not_silent_prefix(monkeypatch):
    features = [feature(f"w:{i}") for i in range(3)]
    selections = {f.feature_id: np.arange(9 * i, 9 * (i + 1)) for i, f in enumerate(features)}
    calls = install_boundary_stubs(monkeypatch, selections)
    monkeypatch.setattr(adapter, "MAX_TOTAL_LOCAL_ROWS", 17)
    result = inspect(np.zeros((27, 3)), features)
    assert calls == ["w:0"]
    assert result["expected_window_count"] == 3
    assert len(result["features"]) == 3
    assert result["total_local_input_rows"] == 27
    assert result["measured_conditional_window_count"] == 1
    for row in result["features"][1:]:
        assert row["measurements"] is None
        assert row["reasons"] == ["LOCAL_OR_TOTAL_SUPPORT_BUDGET_NO_THINNING"]


@pytest.mark.parametrize("points", [
    None, [], [1., 2., 3.], [[1., 2.], [3., 4.]], [[1., 2., 3.], [4.]],
    [[True, False, True]], [[1 + 2j, 0., 0.]], [["1", "2", "3"]],
    [[float("nan"), 0., 0.]], [[float("inf"), 0., 0.]], [[1_000_001., 0., 0.]],
])
def test_malformed_scan_is_explicit_unavailable(points):
    result = inspect(points)
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["INVALID_FULL_RAW_SCAN"]
    assert_no_authority(result)


@pytest.mark.parametrize("raw_ids", [
    [], [0], [0, 0, 2], [-1, 0, 1], [0, 1, 1_000_000],
    [0., 1., 2.], [False, True, False], [[0], [1], [2]], [[0], [1, 2]],
    [0, 1, "2"], np.array([0, 1, 2], dtype=object),
])
def test_malformed_or_reused_raw_identities_are_rejected(raw_ids):
    result = inspect(np.zeros((3, 3)), source_raw_row_indices=raw_ids)
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["INVALID_OR_REUSED_RAW_ROW_IDENTITIES"]


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "A" * 64, "g" * 64, 7, True])
def test_scan_digest_is_strict_declaration_not_fabricated(digest):
    result = inspect(np.zeros((3, 3)), scan_sha256=digest)
    assert result["reasons"] == ["BYTE_BOUND_SCAN_DIGEST_REQUIRED"]
    assert "scan_sha256" not in result


@pytest.mark.parametrize("status", [None, True, "DECLARED_ALIGNED", "UNTRUSTED", ["CONDITIONAL_BEST_FIT"]])
def test_registration_requires_explicit_conditional_best_fit(status):
    result = inspect(np.zeros((3, 3)), registration_status=status)
    assert result["reasons"] == ["EXPLICIT_PROVISIONAL_REGISTRATION_REQUIRED"]
    assert_no_authority(result)


@pytest.mark.parametrize("features", [[], (), [None], [object()], [feature(), feature()]])
def test_invalid_or_duplicate_features_rejected(features):
    result = inspect(np.zeros((3, 3)), features)
    assert result["reasons"] == ["INVALID_OR_OVER_BUDGET_FEATURE_POLICY"]


def test_window_cap_and_raw_row_cap_are_fail_closed(monkeypatch):
    monkeypatch.setattr(adapter, "MAX_WINDOWS", 1)
    result = inspect(np.zeros((3, 3)), [feature("a"), feature("b")])
    assert result["reasons"] == ["INVALID_OR_OVER_BUDGET_FEATURE_POLICY"]
    monkeypatch.setattr(adapter, "MAX_RAW_ROWS", 2)
    result = inspect(np.zeros((3, 3)))
    assert result["reasons"] == ["INVALID_FULL_RAW_SCAN"]


def test_policy_must_be_validated_type():
    result = inspect(np.zeros((3, 3)), instance_policy={"max_axial_gap_mm": 4.})
    assert result["reasons"] == ["INVALID_OR_OVER_BUDGET_FEATURE_POLICY"]


def test_unavailable_surface_carrier_cannot_publish_instance(monkeypatch):
    install_boundary_stubs(monkeypatch, {"invented:0": np.arange(9)})
    monkeypatch.setattr(adapter, "_surface_carrier", lambda *_: None)
    result = inspect(np.zeros((9, 3)))
    assert result["measured_conditional_window_count"] == 1
    assert result["observed_support_instance_count"] == 0
    assert result["finite_support_instances"] == []
    assert result["features"][0]["finite_support"]["reasons"] == ["MEASURED_CARRIER_UNAVAILABLE"]
    assert_no_authority(result)
