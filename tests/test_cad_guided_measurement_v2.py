"""Measurement kernel V2: declared material-side layer selection, both-flange diagnostics,
unresolved single-layer identity, sign conventions for both sweep signs, and the derived
policy feasibility for the 0.8 mm / 90 deg geometry. Synthetic analytic fixtures only."""
from __future__ import annotations

import importlib.util
import dataclasses
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "backend" / f"{name}.py"
    if "backend" not in sys.modules:
        package = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("backend", None, is_package=True))
        package.__path__ = [str(ROOT / "backend")]
        sys.modules["backend"] = package
    spec = importlib.util.spec_from_file_location(f"backend.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"backend.{name}"] = module
    spec.loader.exec_module(module)
    return module


v1 = _load("cad_guided_measurement")
v2 = _load("cad_guided_measurement_v2")


def feature(mod, radius=0.8, sweep=90.0, delta=0.03, excl=0.04, margin=3.0, span=(0.0, 12.0)):
    return mod.CADGuidedFeature(feature_id="f", nominal_radius_mm=radius, nominal_sweep_deg=sweep,
        origin=(0, 0, 0), axis=(0, 0, 1), start_radial=(1, 0, 0), span_mm=span,
        flange_window_mm=(2.5, 12.5), search_margin_mm=margin, max_fit_residual_mm=delta,
        plane_exclusion_mm=excl, max_tangency_error_mm=0.1, nominal_source="SYNTHETIC")


def bend(radius=0.8, sweep=90.0, noise=0.0, seed=3, spacing=0.3, thickness=None, outer_only=False, axial=12.0):
    """Analytic inner surface (arc radius r, tangent flanges) plus an optional outer skin at r+t."""
    rng = np.random.default_rng(seed)
    sign = math.copysign(1.0, sweep)
    phi = math.radians(sweep)
    zs = np.arange(0.0, axial + 1e-9, spacing)

    def layer(r):
        out = []
        n = max(3, int(round(r * abs(phi) / spacing)) + 1)
        for th in np.linspace(0.0, sweep, n):
            for z in zs:
                out.append((r * math.cos(math.radians(th)), r * math.sin(math.radians(th)), z))
        c1 = np.array([r, 0., 0.])
        c2 = r * np.array([math.cos(phi), math.sin(phi), 0.])
        ray1 = np.array([0., -sign, 0.])
        ray2 = sign * np.array([-math.sin(phi), math.cos(phi), 0.])
        for c, ray in ((c1, ray1), (c2, ray2)):
            for d in np.arange(0.3, 14.0, spacing):
                for z in zs:
                    out.append(tuple(c + d * ray + np.array([0., 0., z])))
        return out

    points = [] if outer_only else layer(radius)
    if thickness:
        points += layer(radius + thickness)
    points = np.array(points, float)
    if noise:
        points = points + rng.normal(0, noise, points.shape)
    return points


def measure(mod, points, feat, policy=None):
    kwargs = {} if mod is v1 else dict(layer_policy=policy)
    return mod._measure(points, feat, **kwargs)


@pytest.mark.parametrize("sweep", [90.0, -90.0, 120.0, -120.0])
def test_single_skin_v2_reproduces_v1_and_reports_unresolved_identity(sweep):
    points = bend(sweep=sweep, noise=0.004, seed=11)
    old = measure(v1, points, feature(v1, sweep=sweep))
    new = measure(v2, points, feature(v2, sweep=sweep), v2.LayerSelectionPolicyV1())
    for key in ("measured_radius_mm", "measured_sweep_deg", "measured_axis_direction", "measured_axis_point",
                "flange_max_residual_mm", "circle_max_residual_mm", "arc_support_count", "flange_support_counts"):
        assert new[key] == old[key], key
    assert new["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED"
    flanges = new["diagnostics"]["flanges"]
    assert [row["layer_selection"]["identity"] for row in flanges] == ["SINGLE_LAYER_IDENTITY_UNRESOLVED"] * 2
    assert [row["layer_selection"]["qualifying_groups_detected"] for row in flanges] == [1, 1]
    assert all(row["before_selection"]["status"] == "PASS" and row["after_selection"]["status"] == "PASS"
               for row in flanges)
    assert new["diagnostics"]["arc_material_side_filter"]["material_side_dropped_rows"] == 0
    heuristic = new["diagnostics"]["arc_material_side_filter"]["outer_arc_nominal_heuristic"]
    assert heuristic["status"] == "NOMINAL_HEURISTIC_NOT_EVALUATED_NO_TWO_LAYER_GAP" and heuristic["certificate"] is False
    assert new["inner_face_claim_eligible"] is False and new["carrier_role"] == v2.GENERIC_CARRIER_ROLE
    assert new["radius_deviation_mm"] is None and new["radius_identity"] == "OBSERVED_SURFACE_RADIUS_NOT_INNER_RADIUS"


def test_disabled_policy_reproduces_v1_exactly_including_material_status():
    points = bend(noise=0.004, seed=12)
    old = measure(v1, points, feature(v1))
    new = measure(v2, points, feature(v2), v2.LayerSelectionPolicyV1(enabled=False))
    assert new["measured_radius_mm"] == old["measured_radius_mm"]
    assert new["support_row_indices"] == old["support_row_indices"]
    assert new["_instance_support"] == old["_instance_support"]
    assert new["material_side_status"] == "CONTEXT_ASSUMED_NOT_VERIFIED"
    # Disabling the selection supplies no material-side evidence: the geometry reproduces V1,
    # but no inner-face claim (role or nominal deviation) is granted.
    assert new["inner_face_claim_eligible"] is False and new["carrier_role"] == v2.GENERIC_CARRIER_ROLE
    assert new["radius_deviation_mm"] is None and old["radius_deviation_mm"] is not None
    assert new["diagnostics"]["arc_material_side_filter"]["rule"] == "V1_ABSOLUTE_BAND_EXCLUSION"
    assert all(row["layer_selection"]["identity"] == "LAYER_SELECTION_DISABLED" for row in new["diagnostics"]["flanges"])


def test_disabled_policy_keeps_material_side_arc_rows_exactly_like_v1_on_two_skins():
    # The V1 control must not silently drop material-side arc rows: on a two-skin fixture V1 and
    # disabled-V2 fail with the identical first reason and identical diagnostics-free semantics.
    points = bend(thickness=1.5, noise=0.004, seed=21)
    with pytest.raises(v1._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"):
        measure(v1, points, feature(v1))
    with pytest.raises(v2._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"):
        measure(v2, points, feature(v2), v2.LayerSelectionPolicyV1(enabled=False))


@pytest.mark.parametrize("sweep", [90.0, -90.0, 120.0])
def test_two_skins_refused_by_v1_are_measured_after_inner_layer_selection(sweep):
    points = bend(sweep=sweep, thickness=1.5, noise=0.004, seed=13)
    with pytest.raises(v1._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"):
        measure(v1, points, feature(v1, sweep=sweep))
    new = measure(v2, points, feature(v2, sweep=sweep), v2.LayerSelectionPolicyV1())
    assert abs(new["measured_radius_mm"] - 0.8) < 0.03
    assert abs(abs(new["measured_sweep_deg"]) - abs(sweep)) < 0.2
    assert new["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    for row in new["diagnostics"]["flanges"]:
        selection = row["layer_selection"]
        assert selection["qualifying_groups_detected"] == 2
        assert selection["kept_layer"] == "SMALLER_MATERIAL_SIDE_COORDINATE"
        assert abs(selection["observed_layer_gap_mm"] - 1.5) < 0.05
        assert row["before_selection"]["status"] == "WEAK" and row["before_selection"]["thickness_ratio_gate_passed"] is False
        assert row["after_selection"]["status"] == "PASS"
    arc = new["diagnostics"]["arc_material_side_filter"]
    assert arc["material_side_dropped_rows"] > 0 and arc["rule"] == "SIGNED_GREATER_THAN_PLANE_EXCLUSION_BOTH_PLANES"
    heuristic = arc["outer_arc_nominal_heuristic"]
    assert heuristic["certificate"] is False and heuristic["status"] == "NOMINAL_HEURISTIC_SWEEP_BELOW_BOUND"
    assert heuristic["bound_deg"] > abs(sweep) and len(heuristic["observed_flange_layer_gaps_mm"]) == 2
    assert arc["outer_arc_isolation_evidence"].startswith("UNCHANGED_CIRCLE_RESIDUAL_AND_TANGENCY_GATES")
    # The two-layer selection is a retained side HYPOTHESIS, not verified side evidence: no
    # present status is inner-face eligible, so the fit is a generic contact line without a
    # nominal inner-radius deviation even here.
    assert v2.INNER_FACE_ELIGIBLE_STATUSES == () and new["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert new["inner_face_claim_eligible"] is False and new["carrier_role"] == v2.GENERIC_CARRIER_ROLE
    assert new["radius_identity"] == "OBSERVED_SURFACE_RADIUS_NOT_INNER_RADIUS" and new["radius_deviation_mm"] is None
    assert new["diagnostics"]["flanges"][0]["layer_selection"]["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    for row in new["diagnostics"]["flanges"]:
        compatibility = row["layer_selection"]["pair_compatibility"]
        assert compatibility["status"] == "COMPATIBLE_PARALLEL_LAYER_PAIR_UNVERIFIED_MATERIAL"
        assert compatibility["normal_angle_deg"] < 1.0 and min(compatibility["projected_bounding_interval_overlap"]) > 0.9
        assert row["layer_selection"]["qualifying_split_count"] == 1


def test_layer_split_keeps_smaller_material_side_coordinate_not_fitted_normal_side():
    """Material lies on +radial; the fitted normal points to -radial. The kept layer must be the
    axis side (smaller material coordinate) regardless of which normal is used for reporting."""
    points = bend(thickness=1.5, noise=0.0, seed=1)
    feat = feature(v2)
    new = measure(v2, points, feat, v2.LayerSelectionPolicyV1())
    origin, axis, radial = map(np.asarray, (feat.origin, feat.axis, feat.start_radial))
    kept = points[np.asarray(new["_instance_support"]["flange_row_indices"][0])]
    material = (kept - origin - feat.nominal_radius_mm * radial) @ radial
    assert np.max(np.abs(material)) < 0.05, "kept first flange rows are the inner skin (material coordinate ~0)"
    assert new["diagnostics"]["flanges"][0]["layer_selection"]["layer_means_mm"][1] - \
        new["diagnostics"]["flanges"][0]["layer_selection"]["layer_means_mm"][0] > 1.0


def test_outer_only_scan_is_single_layer_with_unresolved_identity_not_inner():
    points = bend(thickness=1.5, outer_only=True, noise=0.004, seed=14)
    new = measure(v2, points, feature(v2), v2.LayerSelectionPolicyV1())
    assert abs(new["measured_radius_mm"] - 2.3) < 0.05
    assert new["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED"
    assert "INNER" not in new["material_side_status"]
    # Connected eligibility: the observed surface radius (2.3 mm) is never an inner radius claim.
    assert new["inner_face_claim_eligible"] is False and new["carrier_role"] == v2.GENERIC_CARRIER_ROLE
    assert new["radius_deviation_mm"] is None and new["radius_identity"] == "OBSERVED_SURFACE_RADIUS_NOT_INNER_RADIUS"


def test_three_layers_are_ambiguous_not_selected_and_tilted_single_plane_bands_are_incompatible():
    feat = feature(v2)
    policy = v2.LayerSelectionPolicyV1()
    rng = np.random.default_rng(31)
    def strip(offset, x0=0.0, x1=10.0):
        x = rng.uniform(x0, x1, 400); y = rng.uniform(0., 12., 400)
        return np.column_stack((x, y, np.full(400, offset)+rng.normal(0, .004, 400)))
    radial = np.array([0., 0., 1.])
    three = np.vstack((strip(0.), strip(1.5), strip(3.0)))
    keep, record = v2._select_flange_layer(three, radial, feat, policy, 0)
    assert record["identity"] == "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED" and record["qualifying_groups_detected"] == 3
    assert keep.all() and record["qualifying_split_count"] == 2
    # One tilted plane sampled in two disjoint bands: a large coordinate gap but no footprint overlap.
    tilt = np.array([0., 0., 1.]); band_a = strip(0., 0., 4.); band_b = strip(0., 12., 16.)
    tilted = np.vstack((band_a, band_b)); tilted[:, 2] += 0.3*tilted[:, 0]
    keep, record = v2._select_flange_layer(tilted, tilt, feat, policy, 0)
    assert record["identity"] in ("TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED", "SINGLE_LAYER_IDENTITY_UNRESOLVED")
    assert keep.all()
    if record["identity"] == "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED":
        assert "GROUP_PROJECTED_BOUNDING_INTERVALS_DO_NOT_OVERLAP" in record["pair_compatibility"]["reasons"]


def test_failure_carries_diagnostics_for_both_flanges_and_both_branches():
    points = bend(thickness=1.5, noise=0.004, seed=15)
    with pytest.raises(v2._Unavailable) as info:
        measure(v2, points, feature(v2), v2.LayerSelectionPolicyV1(enabled=False))
    diagnostics = info.value.diagnostics
    assert len(diagnostics["flanges"]) == 2
    for row in diagnostics["flanges"]:
        before = row["before_selection"]
        assert before["rows"] >= 12 and len(before["singular_over_sqrt_n"]) == 3
        assert before["spread_gate_passed"] is True and before["thickness_ratio_gate_passed"] is False
        assert before["first_failure"] == "FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"
        assert before["max_residual_mm"] > 0.5 and before["axial_extent_mm"] > 10.0


def test_narrow_crop_is_reported_as_spread_failure_not_thickness_failure():
    feat = feature(v2, delta=0.05, excl=0.075)
    strip = np.array([(x, y, 0.0) for x in np.linspace(0.0, 1.2, 8) for y in np.linspace(0.0, 12.0, 41)])
    row = v2._flange_diagnostics(strip, np.array([0., 0., 1.]), feat, role=0)
    assert row["status"] == "WEAK" and row["spread_gate_passed"] is False and row["thickness_ratio_gate_passed"] is True
    assert row["spread_gate_minor_rms_mm"] < row["spread_gate_threshold_mm"] == 0.5
    assert "ray_extent_mm" not in row and row["second_principal_extent_mm"] > 0 and row["flange_ray_extent_mm"] is None


def test_frozen_policy_is_infeasible_for_0p8_mm_90_deg_even_on_perfect_data_and_derived_policy_measures():
    points = bend(noise=0.0)
    # 0.3 mm sampling leaves too few arc rows beyond the 0.075 mm exclusion band;
    # dense sampling exposes the flat-witness refusal itself. Both are refusals.
    with pytest.raises(v1._Unavailable, match="ARC_SUPPORT_RANK_DEFICIENT|ARC_COMPATIBLE_WITH_FLAT"):
        measure(v1, points, feature(v1, delta=0.05, excl=0.075))
    dense = bend(noise=0.0, spacing=0.05)
    with pytest.raises(v1._Unavailable, match="ARC_COMPATIBLE_WITH_FLAT"):
        measure(v1, dense, feature(v1, delta=0.05, excl=0.075))
    with pytest.raises(v1._Unavailable, match="ARC_COMPATIBLE_WITH_FLAT"):
        measure(v1, dense, feature(v1, delta=0.05, excl=0.055))
    with pytest.raises(v1._Unavailable, match="ARC_COMPATIBLE_WITH_FLAT"):
        measure(v1, points, feature(v1, delta=0.05, excl=0.055))
    measured = measure(v1, points, feature(v1, delta=0.03, excl=0.04))
    assert abs(measure(v1, dense, feature(v1, delta=0.03, excl=0.04))["measured_radius_mm"] - 0.8) < 1e-6
    assert abs(measured["measured_radius_mm"] - 0.8) < 1e-6
    usable = 90.0 - 2 * math.degrees(math.acos(1 - 0.075 / 0.8))
    sagitta = 0.8 * (1 - math.cos(math.radians(usable / 2)))
    assert usable < 41.0 and sagitta / 2 < 0.05


@pytest.mark.parametrize("policy", [dict(layer_split_gap_mm=0.0), dict(min_layer_rows=5), dict(enabled="yes"),
                                    dict(max_layers_reported=1), dict(max_layer_gap_mm=0.2), dict(min_projected_interval_overlap=0.)])
def test_invalid_layer_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        v2.LayerSelectionPolicyV1(**policy)


def test_measure_rejects_foreign_policy_type():
    with pytest.raises(ValueError):
        v2._measure(bend(), feature(v2), layer_policy=dict(enabled=True))


def test_selected_partition_is_reconstructible_from_published_sort_normal_center_and_threshold():
    feat = feature(v2)
    policy = v2.LayerSelectionPolicyV1()
    rng = np.random.default_rng(41)
    def strip(offset):
        x = rng.uniform(0., 10., 300); y = rng.uniform(0., 12., 300)
        return np.column_stack((x, y, np.full(300, offset)+rng.normal(0, .004, 300)))
    points = np.vstack((strip(0.), strip(1.5)))
    keep, record = v2._select_flange_layer(points, np.array([0., 0., 1.]), feat, policy, 0)
    assert record["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert record["summary_scope"].startswith("BOUNDED_DIAGNOSTIC_SUMMARY")
    assert record["kept_rule"] == "ROW_KEPT_IFF_(ROW-SORT_CENTER)DOT_SORT_NORMAL_BELOW_SPLIT_THRESHOLD"
    coordinate = (points-np.asarray(record["sort_center_mm"])) @ np.asarray(record["sort_normal"])
    assert np.array_equal(coordinate < record["split_threshold_mm"], keep)
    assert record["qualifying_splits"][0]["split_threshold_mm"] == record["split_threshold_mm"]
    assert record["reported_split_count"] == 1 and record["omitted_split_count"] == 0
    assert record["qualifying_groups_detected"] == 2 and record["qualifying_groups_scope"].endswith("NOT_A_PHYSICAL_OR_GEOMETRIC_SKIN_COUNT")
    assert record["unqualified_large_gap_count"] == 0 and record["unqualified_large_gaps"] == []


def test_sparse_extra_population_is_published_as_an_unqualified_large_gap_not_hidden():
    feat = feature(v2)
    policy = v2.LayerSelectionPolicyV1()
    rng = np.random.default_rng(43)
    def strip(offset, n=100):
        x = rng.uniform(0., 10., n); y = rng.uniform(0., 12., n)
        return np.column_stack((x, y, np.full(n, offset)+rng.normal(0, .004, n)))
    singleton = np.array([[5., 6., 2.0]])
    points = np.vstack((strip(0.), strip(1.5), singleton))
    keep, record = v2._select_flange_layer(points, np.array([0., 0., 1.]), feat, policy, 0)
    assert record["qualifying_split_count"] == 1 and record["unqualified_large_gap_count"] == 1
    gap = record["unqualified_large_gaps"][0]
    assert gap["upper_rows"] == 1 and gap["gap_mm"] > .3
    # The reviewer's algebra: the singleton contributes ~0.5*sqrt(100)/101 mm to the upper RMS; the
    # pair may still pass, and the published count says qualifying groups, not physical skins.
    assert record["qualifying_groups_detected"] == 2
    if record["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED":
        assert record["layer_rows"] == [100, 101] and keep.sum() == 100


def test_island_inside_a_hole_scores_full_interval_overlap_and_is_labelled_a_proxy():
    feat = feature(v2)
    policy = v2.LayerSelectionPolicyV1()
    rng = np.random.default_rng(47)
    lower = []
    while len(lower) < 400:
        x, y = rng.uniform(-10., 10.), rng.uniform(-10., 10.)
        if 6. <= math.hypot(x, y) <= 10.:
            lower.append((x, y, rng.normal(0, .004)))
    upper = [(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), 1.5+rng.normal(0, .004)) for _ in range(150)]
    points = np.array(lower+upper)
    mask = np.zeros(len(points), dtype=bool); mask[:400] = True
    record = v2._layer_pair_compatibility(points, mask, np.array([0., 0., 1.]), feat, policy)
    assert min(record["projected_bounding_interval_overlap"]) == 1.0
    assert "GROUP_PROJECTED_BOUNDING_INTERVALS_DO_NOT_OVERLAP" not in record["reasons"]
    assert record["overlap_scope"].startswith("BOUNDING_INTERVAL_OVERLAP_FRACTION_OF_SHORTER_INTERVAL")
    assert "NOT_AREA_OR_MATERIAL_FOOTPRINT" in record["overlap_scope"] and len(record["projection_basis"]) == 2
    assert "footprint_overlap" not in record


def test_numerical_failure_inside_layer_selection_retains_the_diagnostic_prefix_and_phase(monkeypatch):
    points = bend(thickness=1.5, noise=0.004, seed=15)
    original = v2._select_flange_layer
    def failing(crop, radial, feat, policy, role):
        if role == 1:
            raise np.linalg.LinAlgError("SVD did not converge")
        return original(crop, radial, feat, policy, role)
    monkeypatch.setattr(v2, "_select_flange_layer", failing)
    with pytest.raises(np.linalg.LinAlgError) as info:
        measure(v2, points, feature(v2), v2.LayerSelectionPolicyV1())
    diagnostics = info.value.diagnostics
    assert diagnostics["failure_phase"] == "FLANGE_DIAGNOSTICS_OR_LAYER_SELECTION_ROLE_1"
    assert diagnostics["phases_completed"] == [] and len(diagnostics["flanges"]) == 1
    assert diagnostics["flanges"][0]["layer_selection"]["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert diagnostics["arc_material_side_filter"] is None
    # Public entrypoint: the same numerical failure is an ordinary refused row that retains the prefix.
    monkeypatch.setattr(v2, "_select_flange_layer", failing)
    row = v2.inspect_cad_guided_scan(points, [feature(v2)], registration_status="DECLARED_ALIGNED", layer_policy=v2.LayerSelectionPolicyV1())["features"][0]
    assert row["status"] == "UNAVAILABLE" and row["reasons"] == ["SVD did not converge"]
    assert row["diagnostics"]["failure_phase"] == "FLANGE_DIAGNOSTICS_OR_LAYER_SELECTION_ROLE_1"


def test_successful_measurement_lists_every_phase_and_refusals_name_the_failing_phase():
    good = measure(v2, bend(thickness=1.5, noise=0.004, seed=15), feature(v2), v2.LayerSelectionPolicyV1())
    assert good["diagnostics"]["phases_completed"] == ["FLANGE_DIAGNOSTICS_AND_LAYER_SELECTION", "PLANE_FITS",
        "AXIS_AND_SWEEP", "ARC_MATERIAL_SIDE_FILTER", "ARC_FIT_AND_TANGENCY"]
    assert good["diagnostics"]["failure_phase"] is None
    with pytest.raises(v2._Unavailable) as info:
        measure(v2, bend(thickness=1.5, noise=0.004, seed=15), feature(v2), v2.LayerSelectionPolicyV1(enabled=False))
    assert info.value.diagnostics["failure_phase"] == "PLANE_FIT_ROLE_0"
    assert info.value.diagnostics["phases_completed"] == ["FLANGE_DIAGNOSTICS_AND_LAYER_SELECTION"]


def test_shared_support_invalidation_preserves_attempted_diagnostics_without_a_surviving_measurement():
    points = bend(thickness=1.5, noise=0.004, seed=15)
    twin = v2.CADGuidedFeature(**dict(dataclasses.asdict(feature(v2)), feature_id="g"))
    payload = v2.inspect_cad_guided_scan(points, [feature(v2), twin], registration_status="DECLARED_ALIGNED", layer_policy=v2.LayerSelectionPolicyV1())
    rows = payload["features"]
    assert [row["status"] for row in rows] == ["UNAVAILABLE", "UNAVAILABLE"]
    for row in rows:
        assert row["reasons"] == ["AMBIGUOUS_SHARED_FEATURE_SUPPORT"] and row["measurements"] is None
        assert row["attempted_diagnostics"]["phases_completed"][-1] == "ARC_FIT_AND_TANGENCY"
        assert row["attempted_diagnostics_scope"].endswith("NOT_A_SURVIVING_MEASUREMENT")
