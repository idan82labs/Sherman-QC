"""Measurement kernel V3: the V2 gates unchanged, ONE changed mechanism (two-mode band-membership layer
separation). Modeled fixtures built from the declared V10 observation HYPOTHESIS (two parallel planar populations
about 1.3-1.8 mm apart, per-skin noise, a minority of between-skin rows continuous in the material-side
coordinate, published as UNASSIGNED rows of unverified cut-edge identity); synthetic algorithm checks, never
physical truth, sheet identity or thickness."""
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location("isolated_measurement_kernel_v3."+name, ROOT/"backend"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module            # dataclasses under `from __future__ import annotations` need the module registered
    spec.loader.exec_module(module)
    return module


v2 = _load("cad_guided_measurement_v2")
v3 = _load("cad_guided_measurement_v3")


def feature(mod, radius=0.8, sweep=90.0, delta=0.03, excl=0.04, margin=3.0, span=(0.0, 12.0)):
    return mod.CADGuidedFeature(feature_id="f", nominal_radius_mm=radius, nominal_sweep_deg=sweep,
        origin=(0, 0, 0), axis=(0, 0, 1), start_radial=(1, 0, 0), span_mm=span,
        flange_window_mm=(2.5, 12.5), search_margin_mm=margin, max_fit_residual_mm=delta,
        plane_exclusion_mm=excl, max_tangency_error_mm=0.1, nominal_source="SYNTHETIC")


def bend(radius=0.8, sweep=90.0, noise=0.0, seed=3, spacing=0.3, thickness=None, outer_only=False, axial=12.0,
         edge_spacing=None, contaminant=None):
    """Analytic inner surface (arc radius r, tangent flanges) plus an optional outer skin at r+t.

    `edge_spacing` adds the sheet's CUT-EDGE rows at both axial ends of each flange: a strip perpendicular to
    the skins spanning the thickness (the physical bridging rows that defeat an empty-gap partition).
    `contaminant=(offset_mm, fraction)` adds rows displaced from the inner skin by `offset_mm` along the
    material side on a fraction of the flange positions (dents / arc-start contamination model)."""
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
    if thickness and edge_spacing:
        c1 = np.array([radius, 0., 0.]); c2 = radius * np.array([math.cos(phi), math.sin(phi), 0.])
        ray1 = np.array([0., -sign, 0.]); ray2 = sign * np.array([-math.sin(phi), math.cos(phi), 0.])
        radial1 = np.array([1., 0., 0.]); radial2 = np.array([math.cos(phi), math.sin(phi), 0.])
        for c, ray, radial in ((c1, ray1, radial1), (c2, ray2, radial2)):
            for d in np.arange(0.3, 14.0, spacing):
                for s in np.arange(edge_spacing, thickness, edge_spacing):
                    for z in (0.0, axial):
                        points.append(tuple(c + d * ray + s * radial + np.array([0., 0., z])))
    if contaminant:
        offset, fraction = contaminant
        c1 = np.array([radius, 0., 0.]); ray1 = np.array([0., -sign, 0.]); radial1 = np.array([1., 0., 0.])
        for d in np.arange(0.3, 14.0, spacing):
            for z in zs:
                if rng.uniform() < fraction:
                    points.append(tuple(c1 + d * ray1 + offset * radial1 + np.array([0., 0., z])))
    points = np.array(points, float)
    if noise:
        points = points + rng.normal(0, noise, points.shape)
    return points


def measure(mod, points, feat, policy=None):
    return mod._measure(points, feat, layer_policy=policy)


def _selections(new):
    return [row["layer_selection"] for row in new["diagnostics"]["flanges"]]


# ---- V2 behaviours preserved --------------------------------------------------------------------------

@pytest.mark.parametrize("sweep", [90.0, -90.0, 120.0])
def test_single_skin_reproduces_v2_with_unresolved_identity(sweep):
    points = bend(sweep=sweep, noise=0.004, seed=5)
    old = measure(v2, points, feature(v2, sweep=sweep), v2.LayerSelectionPolicyV1())
    new = measure(v3, points, feature(v3, sweep=sweep), v3.LayerSelectionPolicyV2())
    for key in ("measured_radius_mm", "measured_sweep_deg", "material_side_status", "carrier_role", "inner_face_claim_eligible"):
        assert new[key] == old[key], key
    assert new["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED"
    for selection in _selections(new):
        assert selection["qualifying_groups_detected"] == 1 and selection["modes"]["secondary"] is None
        assert selection["implementation"] == v3.LAYER_SELECTION_IMPLEMENTATION and selection["kept_rule"] == "ALL_ROWS"


def test_disabled_policy_reproduces_v1_crops_exactly():
    points = bend(thickness=1.5, noise=0.004, seed=7)
    with pytest.raises(v3._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"):
        measure(v3, points, feature(v3), v3.LayerSelectionPolicyV2(enabled=False))
    with pytest.raises(v2._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"):
        measure(v2, points, feature(v2), v2.LayerSelectionPolicyV1(enabled=False))


@pytest.mark.parametrize("sweep", [90.0, -90.0, 120.0])
def test_clean_two_skins_are_measured_as_in_v2_with_the_smaller_coordinate_kept(sweep):
    points = bend(sweep=sweep, thickness=1.5, noise=0.004, seed=13)
    old = measure(v2, points, feature(v2, sweep=sweep), v2.LayerSelectionPolicyV1())
    new = measure(v3, points, feature(v3, sweep=sweep), v3.LayerSelectionPolicyV2())
    assert abs(new["measured_radius_mm"] - 0.8) < 0.03 and abs(abs(new["measured_sweep_deg"]) - abs(sweep)) < 0.2
    assert abs(new["measured_radius_mm"] - old["measured_radius_mm"]) < 0.01
    assert new["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert new["inner_face_claim_eligible"] is False and new["carrier_role"] == v3.GENERIC_CARRIER_ROLE
    assert v3.INNER_FACE_ELIGIBLE_STATUSES == () and new["radius_deviation_mm"] is None
    for selection, row in zip(_selections(new), new["diagnostics"]["flanges"]):
        assert selection["qualifying_groups_detected"] == 2 and selection["kept_layer"] == "SMALLER_MATERIAL_SIDE_COORDINATE"
        assert abs(selection["observed_layer_gap_mm"] - 1.5) < 0.05 and selection["bridging_fraction"] < 0.02
        assert selection["pair_compatibility"]["status"] == "COMPATIBLE_PARALLEL_LAYER_PAIR_UNVERIFIED_MATERIAL"
        assert selection["band_exceeds_declared_residual"] is True and selection["refit_iterations_used"] >= 1
        assert row["before_selection"]["status"] == "WEAK" and row["after_selection"]["status"] == "PASS"
        assert selection["dropped_other_skin_rows"] > 0 and selection["kept_rows"] + selection["dropped_rows"] == selection["input_rows"]


# ---- the V10 mechanism: bridging rows ------------------------------------------------------------------

@pytest.mark.parametrize("sweep", [90.0, -90.0])
def test_two_skins_with_cut_edge_bridging_rows_defeat_v2_and_are_measured_by_v3(sweep):
    points = bend(sweep=sweep, thickness=1.5, noise=0.004, seed=21, edge_spacing=0.3)   # cut edges sampled at the scan spacing
    # V2 (empty-gap partition): the edge rows fill the coordinate between the skins -> no qualifying split
    with pytest.raises(v2._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY") as failure:
        measure(v2, points, feature(v2, sweep=sweep), v2.LayerSelectionPolicyV1())
    old = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert old["identity"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED" and old["largest_gap_mm"] < 0.3
    new = measure(v3, points, feature(v3, sweep=sweep), v3.LayerSelectionPolicyV2())
    assert abs(new["measured_radius_mm"] - 0.8) < 0.03 and abs(abs(new["measured_sweep_deg"]) - abs(sweep)) < 0.2
    assert new["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    for selection, row in zip(_selections(new), new["diagnostics"]["flanges"]):
        assert selection["qualifying_groups_detected"] == 2 and selection["bridging_rows"] > 0
        assert 0.02 < selection["bridging_fraction"] <= selection["max_bridging_row_fraction"]
        assert selection["dropped_bridging_rows"] == selection["bridging_rows"]
        assert row["after_selection"]["status"] == "PASS" and row["after_selection"]["max_residual_mm"] <= 0.03
        assert abs(selection["observed_layer_gap_mm"] - 1.5) < 0.05
    # Reconstructibility: the published group planes and band reproduce the kept rows exactly.
    origin, axis, radial = map(np.asarray, (feature(v3).origin, feature(v3).axis, feature(v3).start_radial))
    selection = _selections(new)[0]
    kept_rows = np.asarray(new["_instance_support"]["flange_row_indices"][0])
    planes = [(np.asarray(p["center_mm"]), np.asarray(p["normal"])) for p in selection["group_planes"]]
    lower = min(range(2), key=lambda k: selection["group_planes"][k]["coordinate_mm"])
    distances = np.stack([np.abs((points - c) @ n) for c, n in planes], axis=1)
    nearest = np.argmin(distances, axis=1)
    member = (nearest == lower) & (distances[np.arange(len(points)), lower] <= selection["membership_band_mm"])
    assert set(kept_rows) <= set(np.flatnonzero(member))


def test_bridging_fraction_above_the_declared_bound_refuses_the_separation_with_all_rows_kept():
    points = bend(thickness=1.5, noise=0.004, seed=22, edge_spacing=0.3)
    with pytest.raises(v3._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY") as failure:
        measure(v3, points, feature(v3), v3.LayerSelectionPolicyV2(max_bridging_row_fraction=0.02))
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["identity"] == "TWO_MODES_MEMBERSHIP_UNRESOLVED" and selection["kept_rule"] == "ALL_ROWS"
    assert selection["bridging_fraction"] > 0.02 and selection["kept_rows"] == selection["input_rows"]
    assert failure.value.diagnostics["failure_phase"] == "PLANE_FIT_ROLE_0"
    # Dense edge sampling (2.5 cut-edge rows per skin row, uniform between the skins): no edge local maximum reaches the
    # exact-band population threshold (about 330 rows against 0.25 * 1456), so it is not a third population; the
    # unassigned fraction (0.41) exceeds the declared bound -> refused with the EXACT reason, all rows kept.
    dense = bend(thickness=1.5, noise=0.004, seed=22, edge_spacing=0.02)
    with pytest.raises(v3._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY") as failure:
        measure(v3, dense, feature(v3), v3.LayerSelectionPolicyV2())
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["identity"] == "TWO_MODES_MEMBERSHIP_UNRESOLVED" and selection["kept_rows"] == selection["input_rows"]
    assert selection["unresolved_reason"] == "BRIDGING_FRACTION_ABOVE_DECLARED_MAXIMUM" and selection["bridging_fraction"] > .35
    assert selection["modes"]["ambiguous_third"] is None and selection["modes"]["local_maximum_count"] > 2


def test_contaminant_rows_outside_the_band_are_bridging_but_inside_the_band_still_fail_the_unchanged_gate():
    feat = feature(v3)
    outside = bend(thickness=1.5, noise=0.004, seed=23, edge_spacing=0.3, contaminant=(0.2, 0.15))
    new = measure(v3, outside, feat, v3.LayerSelectionPolicyV2())
    selection = _selections(new)[0]
    assert new["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert selection["bridging_rows"] > 0 and new["diagnostics"]["flanges"][0]["after_selection"]["max_residual_mm"] <= 0.03
    inside = bend(thickness=1.5, noise=0.004, seed=24, edge_spacing=0.3, contaminant=(0.06, 0.15))   # inside the 0.1 band, beyond delta
    with pytest.raises(v3._Unavailable, match="FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL") as failure:
        measure(v3, inside, feat, v3.LayerSelectionPolicyV2())
    after = failure.value.diagnostics["flanges"][0]["after_selection"]
    assert after["status"] == "NOT_PLANAR" and after["max_residual_mm"] > 0.03      # reported by the gate, never trimmed
    assert failure.value.diagnostics["flanges"][0]["layer_selection"]["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"


def test_three_populations_are_ambiguous_and_a_tilted_single_plane_is_not_two_skins():
    feat = feature(v3)
    policy = v3.LayerSelectionPolicyV2()
    rng = np.random.default_rng(31)
    def strip(offset, x0=0.0, x1=10.0):
        x = rng.uniform(x0, x1, 400); y = rng.uniform(0., 12., 400)
        return np.column_stack((x, y, np.full(400, offset)+rng.normal(0, .004, 400)))
    radial = np.array([0., 0., 1.])
    three = np.vstack((strip(0.), strip(1.5), strip(3.0)))
    keep, record = v3._select_flange_layer(three, radial, feat, policy, 0)
    assert record["identity"] == "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED" and record["qualifying_groups_detected"] == 3
    assert keep.all() and record["modes"]["ambiguous_third"] is not None
    band_a = strip(0., 0., 4.); band_b = strip(0., 12., 16.)
    tilted = np.vstack((band_a, band_b)); tilted[:, 2] += 0.3*tilted[:, 0]
    keep, record = v3._select_flange_layer(tilted, radial, feat, policy, 0)
    assert keep.all() and record["identity"] in ("SINGLE_LAYER_IDENTITY_UNRESOLVED", "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED")


def test_incompatible_pair_is_unresolved_and_a_small_second_group_is_membership_unresolved():
    feat = feature(v3)
    policy = v3.LayerSelectionPolicyV2()
    rng = np.random.default_rng(41)
    def strip(offset, n=400, x0=0.0, x1=10.0, tilt=0.0):
        x = rng.uniform(x0, x1, n); y = rng.uniform(0., 12., n)
        return np.column_stack((x, y, np.full(n, offset)+tilt*x+rng.normal(0, .004, n)))
    radial = np.array([0., 0., 1.])
    tilted_pair = np.vstack((strip(0.), strip(1.5, tilt=0.12)))          # normals ~6.8 deg apart > 3 deg
    keep, record = v3._select_flange_layer(tilted_pair, radial, feat, policy, 0)
    assert keep.all() and record["identity"] in ("TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED", "TWO_MODES_MEMBERSHIP_UNRESOLVED",
                                                 "SINGLE_LAYER_IDENTITY_UNRESOLVED", "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED")
    small = np.vstack((strip(0.), strip(1.5, n=8)))                       # second mode below min_layer_rows
    keep, record = v3._select_flange_layer(small, radial, feat, policy, 0)
    assert keep.all() and record["identity"] in ("SINGLE_LAYER_IDENTITY_UNRESOLVED", "TWO_MODES_MEMBERSHIP_UNRESOLVED")


def test_mode_detection_is_deterministic_bounded_and_published():
    policy = v3.LayerSelectionPolicyV2()
    coordinate = np.concatenate((np.full(500, 0.), np.full(300, 1.5), np.linspace(0.05, 1.45, 60)))
    primary, secondary, record = v3._two_modes(coordinate, policy)
    assert abs(primary - 0.025) < 0.05 and abs(secondary - 1.525) < 0.05 and record["ambiguous_third"] is None
    assert record["bins_widened"] is False and record["local_maximum_count"] >= 2 and len(record["local_maxima"]) <= policy.max_layers_reported
    again = v3._two_modes(coordinate, policy)
    assert again[:2] == (primary, secondary)
    wide = np.concatenate((np.zeros(100), np.full(100, 500.)))            # span would need > MAX_MODE_BINS bins
    _, _, widened = v3._two_modes(wide, policy)
    assert widened["bins_widened"] is True and widened["bin_count"] == v3.MAX_MODE_BINS
    assert v3._two_modes(np.full(50, 0.), policy)[1] is None


@pytest.mark.parametrize("policy", [dict(layer_mode_bin_mm=0.2), dict(layer_membership_band_mm=0.2), dict(max_bridging_row_fraction=0.6),
                                    dict(layer_mode_ambiguity_fraction=0.), dict(layer_mode_ambiguity_fraction=1.5),
                                    dict(layer_refit_iterations=0), dict(layer_refit_iterations=9), dict(layer_refit_iterations=2.0), dict(max_bridging_row_fraction=0.),
                                    dict(layer_split_gap_mm=0.0), dict(min_layer_rows=5), dict(enabled="yes")])
def test_invalid_layer_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        v3.LayerSelectionPolicyV2(**policy)


def test_measure_rejects_the_v1_policy_type_and_foreign_types():
    points = bend(thickness=1.5, noise=0.004, seed=8)
    with pytest.raises(ValueError, match="LayerSelectionPolicyV2"):
        measure(v3, points, feature(v3), v2.LayerSelectionPolicyV1())
    with pytest.raises(ValueError):
        measure(v3, points, feature(v3), dict(enabled=True))


def test_numerical_failure_inside_layer_selection_retains_the_diagnostic_prefix_and_phase(monkeypatch):
    points = bend(thickness=1.5, noise=0.004, seed=9, edge_spacing=0.3)
    calls = []
    original = v3._group_plane
    def failing(subset):
        calls.append(1)
        if len(calls) == 3:
            raise np.linalg.LinAlgError("INJECTED")
        return original(subset)
    monkeypatch.setattr(v3, "_group_plane", failing)
    with pytest.raises(np.linalg.LinAlgError) as failure:
        measure(v3, points, feature(v3), v3.LayerSelectionPolicyV2())
    assert failure.value.diagnostics["failure_phase"].startswith("FLANGE_DIAGNOSTICS_OR_LAYER_SELECTION_ROLE_")


def test_successful_measurement_lists_every_phase_and_the_selection_summary_is_bounded():
    points = bend(thickness=1.5, noise=0.004, seed=10, edge_spacing=0.3)
    new = measure(v3, points, feature(v3), v3.LayerSelectionPolicyV2(max_layers_reported=2))
    assert new["diagnostics"]["phases_completed"] == ["FLANGE_DIAGNOSTICS_AND_LAYER_SELECTION", "PLANE_FITS", "AXIS_AND_SWEEP",
                                                       "ARC_MATERIAL_SIDE_FILTER", "ARC_FIT_AND_TANGENCY"]
    for selection in _selections(new):
        assert len(selection["modes"]["local_maxima"]) <= 2 and selection["modes"]["local_maximum_count"] >= 2


def test_public_entry_and_instances_carry_the_generic_identity_fields():
    points = bend(thickness=1.5, noise=0.004, seed=11, edge_spacing=0.3)
    instances = _load("cad_guided_instances")
    answer = v3.inspect_cad_guided_scan(points, [feature(v3)], registration_status="DECLARED_ALIGNED",
                                        instance_policy=instances.SupportPolicy(4., 30., 20., 3), layer_policy=v3.LayerSelectionPolicyV2())
    row = answer["features"][0]
    assert row["status"] == "MEASURED_CONDITIONAL" and row["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert row["inner_face_claim_eligible"] is False and row["carrier_role"] == v3.GENERIC_CARRIER_ROLE
    assert answer["observed_support_instance_count"] == 1 and answer["production_use_allowed"] is False
    assert "TWO_MODES_MEMBERSHIP_UNRESOLVED" in v3.MATERIAL_SIDE_STATUSES


def test_projection_premise_is_validated_after_refits_and_published():
    feat = feature(v3)
    policy = v3.LayerSelectionPolicyV2()
    points = bend(thickness=1.5, noise=0.004, seed=51, edge_spacing=0.3)
    new = measure(v3, points, feat, policy)
    for selection in _selections(new):
        assert max(selection["crop_normal_to_group_normal_deg"]) <= policy.max_layer_normal_angle_deg
    rng = np.random.default_rng(52)
    # Population A: a flat 10 x 12 mm skin (600 rows). Population B: a NARROW 0.3 mm strip 1.5 mm above it whose own
    # plane is tilted 12 deg about y. B projects onto the crop TLS normal (dominated by A, tilt < 0.2 deg) as one clean
    # mode (smear 0.3*sin 12 deg = 0.06 mm), so two modes are found and membership reaches a fixed point; B's refitted
    # normal then departs 12 deg from the crop normal -> the projection-premise rejection is the EXACT outcome (checked
    # before pair compatibility), all rows kept, both angles published. No other disposition is possible here.
    a = np.column_stack((rng.uniform(0., 10., 600), rng.uniform(0., 12., 600), rng.normal(0, .004, 600)))
    bx = rng.uniform(4.85, 5.15, 300)
    b = np.column_stack((bx, rng.uniform(0., 12., 300), 1.5 + math.tan(math.radians(12.)) * (bx - 5.) + rng.normal(0, .004, 300)))
    keep, record = v3._select_flange_layer(np.vstack((a, b)), np.array([0., 0., 1.]), feat, policy, 0)
    assert keep.all() and record["identity"] == "PROJECTION_PREMISE_UNRESOLVED"
    assert record["unresolved_reason"] == "REFITTED_SKIN_NORMAL_DEPARTS_FROM_CROP_TLS_NORMAL_BEYOND_DECLARED_ANGLE"
    tilt = sorted(record["crop_normal_to_group_normal_deg"])
    assert tilt[0] < 1.0 and 11. < tilt[1] < 13. and record["refit_converged"] is True   # B pulls the crop normal < 1 deg
    assert record["qualifying_groups_detected"] == 2 and record["modes"]["ambiguous_third"] is None
    assert record["bridging_rows"] == 0 and record["group_rows"] in ([600, 300], [300, 600])
    assert record["pair_compatibility"] is None                       # never reached
    assert record["kept_rows"] == record["input_rows"] == 900 and record["dropped_rows"] == 0
    assert "PROJECTION_PREMISE_UNRESOLVED" in v3.MATERIAL_SIDE_STATUSES


def test_membership_band_must_exceed_the_declared_residual_policy_when_selection_is_enabled():
    points = bend(thickness=1.5, noise=0.004, seed=71, edge_spacing=0.3)
    for band in (0.03, 0.02):                                              # equal and below delta = 0.03
        with pytest.raises(ValueError, match="LAYER_MEMBERSHIP_BAND_MUST_EXCEED_DECLARED_RESIDUAL_POLICY"):
            measure(v3, points, feature(v3, delta=0.03), v3.LayerSelectionPolicyV2(layer_membership_band_mm=band))
    measure(v3, points, feature(v3, delta=0.03), v3.LayerSelectionPolicyV2(layer_membership_band_mm=0.031))   # strictly above
    measure(v3, points, feature(v3, delta=0.03), v3.LayerSelectionPolicyV2(enabled=False, layer_membership_band_mm=0.02)) if False else None
    with pytest.raises(v3._Unavailable):                                   # disabled selection: contract not applied, V1 crops refuse
        measure(v3, points, feature(v3, delta=0.03), v3.LayerSelectionPolicyV2(enabled=False, layer_membership_band_mm=0.02))


def _tilt_outer_skin(points, degrees, seed=0):
    """Rotate the ORIGINAL outer-skin rows of the first flange (x = r + t plane) by `degrees` about the flange's
    far end so the outer skin is a genuine plane that is not parallel to the crop TLS normal: the initial band
    membership (mode centres with the crop normal) misses its far rows, the first refit recovers them."""
    pts = points.copy()
    outer = (np.abs(pts[:, 0] - 2.3) < 0.05) & (pts[:, 1] < -0.3)
    pts[outer, 0] += math.tan(math.radians(degrees)) * (-pts[outer, 1] - 0.3)
    return pts


def test_membership_must_reach_a_fixed_point_within_the_declared_refits_or_abstain():
    points = bend(thickness=1.5, noise=0.004, seed=72, edge_spacing=0.3)
    new = measure(v3, points, feature(v3), v3.LayerSelectionPolicyV2())
    for selection in _selections(new):
        assert selection["refit_converged"] is True and 1 <= selection["refit_iterations_used"] <= 6
        assert selection["refit_loop_entries"] == selection["refit_iterations_used"]      # no starved entry
        assert selection["refit_accounting"].startswith("REFIT_ITERATIONS_USED_COUNTS_COMPLETED")
        assert selection["kept_rows"] + selection["dropped_other_skin_rows"] + selection["dropped_bridging_rows"] == selection["input_rows"]
    # Genuine membership change: the outer skin of flange 0 tilted 2.5 deg (its departure from the crop normal stays
    # within the 3 deg premise). Membership from the mode centres along the crop normal leaves the near rows of the
    # tilted skin unassigned (between the planes, beyond the 0.1 mm band), the refit recovers them, the second pass is
    # the fixed point. With ONE declared refit the separation must abstain with the EXACT reason and all rows kept;
    # with six it measures, using exactly two completed refits (no unbudgeted iteration).
    tilted = _tilt_outer_skin(points, 2.5)
    with pytest.raises(v3._Unavailable) as failure:
        measure(v3, tilted, feature(v3), v3.LayerSelectionPolicyV2(layer_refit_iterations=1))
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["identity"] == "TWO_MODES_MEMBERSHIP_UNRESOLVED" and selection["refit_converged"] is False
    assert selection["unresolved_reason"] == "MEMBERSHIP_NOT_A_FIXED_POINT_WITHIN_DECLARED_REFITS"
    assert selection["kept_rows"] == selection["input_rows"] and selection["dropped_rows"] == 0
    assert selection["refit_iterations_used"] == 1 and selection["refit_loop_entries"] == 1
    assert failure.value.diagnostics["failure_phase"] == "PLANE_FIT_ROLE_0"
    new = measure(v3, tilted, feature(v3), v3.LayerSelectionPolicyV2())
    selection = _selections(new)[0]
    assert selection["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert selection["refit_converged"] is True and selection["refit_iterations_used"] == 2 == selection["refit_loop_entries"]
    assert 1.0 < max(selection["crop_normal_to_group_normal_deg"]) < 1.5 and selection["outside_sandwich_member_rows"] == 0


def test_third_population_admission_is_band_population_support_and_band_mass_is_exact():
    policy = v3.LayerSelectionPolicyV2()
    # Codex bin-boundary probe: 40 at 0, 40 at 1.5, 6 at 2.98 and 6 at 3.02 (two bins). The third band holds 12 rows
    # (>= min_layer_rows 12, >= 0.25 * 40) although each of its peak bins holds only 6 -> ambiguous third population.
    coordinate = np.concatenate((np.zeros(40), np.full(40, 1.5), np.full(6, 2.98), np.full(6, 3.02)))
    primary, secondary, record = v3._two_modes(coordinate, policy)
    assert abs(primary - 0.025) < 1e-9 and abs(secondary - 1.525) < 1e-9
    assert record["ambiguous_third"] is not None and record["ambiguous_third"]["band_mass"] == 12 and record["ambiguous_third"]["rows"] == 6
    assert record["third_population_support"] == "BAND_MASS_AT_LEAST_MIN_LAYER_ROWS_NOT_SINGLE_BIN_PEAK"
    # Just below population support (11 rows in the band): not a third population.
    below = np.concatenate((np.zeros(40), np.full(40, 1.5), np.full(6, 2.98), np.full(5, 3.02)))
    assert v3._two_modes(below, policy)[2]["ambiguous_third"] is None
    # Exact inclusive band mass around the published bin centre: rows 0.0999 mm from it count, rows 0.1001 mm from it
    # do not (a quantized five-bin window would have counted everything within +-0.125 mm: 18 instead of 12).
    centre = 3.025
    exact = np.concatenate((np.zeros(40), np.full(40, 1.5), np.full(6, centre), np.full(3, centre + .0999), np.full(3, centre - .0999),
                            np.full(3, centre + .1001), np.full(3, centre - .1001)))
    record = v3._two_modes(exact, policy)[2]
    third = [row for row in record["local_maxima"] if abs(row["centre_mm"] - centre) < 1e-9][0]
    assert third["band_mass"] == 12 and record["mass_semantics"].startswith("EXACT_ROW_COUNT_WITHIN_INCLUSIVE_MEMBERSHIP_BAND")
    assert record["mass_band_mm"] == policy.layer_membership_band_mm and "mass_half_bins" not in record
    assert record["ambiguous_third"] is not None and record["ambiguous_third"]["band_mass"] == 12
    # Widened bins (span needing more than MAX_MODE_BINS bins): masses stay exact coordinate-band counts around the
    # published (coarser) bin centres; the widening itself is published.
    wide = np.concatenate((np.zeros(40), np.full(40, 1.5), np.array([1000.])))
    record = v3._two_modes(wide, policy)[2]
    assert record["bins_widened"] is True and record["bin_width_mm"] > policy.layer_mode_bin_mm
    band = policy.layer_membership_band_mm
    interval = lambda values, centre: int(np.sum((values >= centre - band) & (values <= centre + band)))   # published bound form
    assert all(row["band_mass"] == interval(wide, row["centre_mm"]) for row in record["local_maxima"])
    assert all(row["band_mass"] == interval(exact, row["centre_mm"]) for row in v3._two_modes(exact, policy)[2]["local_maxima"])
    assert v3._two_modes(exact, policy)[2]["mass_semantics"].endswith("EVALUATED_IN_BINARY64")


def _crop_index(points, feat, role):
    """Row indices of the kernel's flange crop for `role`, rebuilt from the feature geometry (kernel conventions)."""
    origin, a0, n0 = (np.asarray(v, float) for v in (feat.origin, feat.axis, feat.start_radial))
    k0 = np.cross(a0, n0); phi = math.radians(feat.nominal_sweep_deg); sign = math.copysign(1., phi)
    n1 = math.cos(phi) * n0 + math.sin(phi) * k0
    radial, ray = ((n0, -sign * k0), (n1, sign * np.cross(a0, n1)))[role]
    local = points - origin; axial = local @ a0
    guard = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(points))), float(np.max(np.abs(origin))))
    in_span = (axial >= feat.span_mm[0] - guard) & (axial <= feat.span_mm[1] + guard)
    from_contact = local - feat.nominal_radius_mm * radial
    along = from_contact @ ray
    return np.flatnonzero(in_span & (along >= feat.flange_window_mm[0]) & (along <= feat.flange_window_mm[1])
                          & (np.abs(from_contact @ radial) <= feat.search_margin_mm))


def test_membership_rule_is_complete_nearest_plane_ties_sandwich_and_exactly_reconstructible():
    z = np.array([0., 0., 1.])
    planes = [(np.zeros(3), z), (np.array([0., 0., .3]), z)]
    # Equal distances (band = gap/2 = 0.15, row exactly mid-way): plane index 0.
    labels, distances, between, within = v3._band_membership(np.array([[1., 2., .15]]), planes, .15)
    assert labels.tolist() == [0] and distances[0, 0] == distances[0, 1] and between.tolist() == [True] and within.tolist() == [True]
    # Strictly between and outside both bands: unassigned (-1); air side / far side outside both bands: nearest plane
    # (kept, published as outside-sandwich members); within a band: that plane, whether or not between.
    rows = np.array([[0., 0., .15], [0., 0., -.5], [0., 0., .8], [0., 0., .02], [0., 0., .29]])
    labels, _, between, within = v3._band_membership(rows, planes, .1)
    assert labels.tolist() == [-1, 0, 1, 0, 1] and between.tolist() == [True, False, False, True, True]
    assert within.tolist() == [False, False, False, True, True] and int((~within & ~between).sum()) == 2
    # Non-parallel overlapping bands: a row within both bands belongs to the NEAREST plane, not the first.
    tilted = np.array([-math.sin(math.radians(2.)), 0., math.cos(math.radians(2.))])
    planes2 = [(np.zeros(3), z), (np.array([0., 0., .3]), tilted)]
    labels, distances, _, _ = v3._band_membership(np.array([[-3., 0., .18]]), planes2, .2)
    assert labels.tolist() == [1] and distances[0, 1] < distances[0, 0] <= .2
    assert v3.MEMBERSHIP_RULE["tie"] == "EQUAL_DISTANCES_ASSIGN_PLANE_INDEX_0_THE_PRIMARY_MODE_PLANE"
    # Exact reconstruction of the published partition on the actual flange crops from the public entry: the crop is
    # rebuilt from the feature geometry, the labels from the published group planes + band + rule, and every count and
    # the kept row set must equal the kernel's own (both labels, unassigned rows, outside-sandwich members).
    feat, policy = feature(v3), v3.LayerSelectionPolicyV2()
    fixtures = dict(tilted=_tilt_outer_skin(bend(thickness=1.5, noise=0.004, seed=73, edge_spacing=0.3), 2.5),
                    far_rows=bend(thickness=1.5, noise=0.004, seed=74, edge_spacing=0.3, contaminant=(2.2, .01)))
    for name, points in fixtures.items():
        new = measure(v3, points, feat, policy)
        for role in (0, 1):
            crop_index = _crop_index(points, feat, role)
            selection = _selections(new)[role]
            assert selection["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED" and selection["input_rows"] == len(crop_index)
            planes = [(np.asarray(p["center_mm"]), np.asarray(p["normal"])) for p in selection["group_planes"]]
            labels, _, between, within = v3._band_membership(points[crop_index], planes, policy.layer_membership_band_mm)
            assert [int((labels == k).sum()) for k in (0, 1)] == selection["group_rows"]
            assert int((labels == -1).sum()) == selection["bridging_rows"] == selection["dropped_bridging_rows"]
            assert int((~within & ~between).sum()) == selection["outside_sandwich_member_rows"]
            lower = int(np.argmin([p["coordinate_mm"] for p in selection["group_planes"]]))
            kept = set(crop_index[labels == lower].tolist())
            assert kept == set(np.asarray(new["_instance_support"]["flange_row_indices"][role]).tolist())
            assert len(kept) == selection["kept_rows"] and selection["membership_rule"] == v3.MEMBERSHIP_RULE
            assert selection["kept_rows"] + selection["dropped_other_skin_rows"] + selection["dropped_bridging_rows"] == len(crop_index)
            assert new["diagnostics"]["flange_support"]["per_role"][role]["outside_sandwich_member_rows"] == selection["outside_sandwich_member_rows"]
        # the 1 % rows 0.7 mm beyond the outer skin (far side, outside both bands, not between) are outer-group members
        assert (_selections(new)[0]["outside_sandwich_member_rows"] > 0) is (name == "far_rows")


def _dent_inner_skin(points, fraction, offset, seed):
    """Displace ORIGINAL inner-skin flange rows (not extra rows) toward the material side by `offset`."""
    rng = np.random.default_rng(seed)
    pts = points.copy()
    on_flange1 = (np.abs(pts[:, 0] - 0.8) < 0.05) & (pts[:, 1] < -0.3)          # inner skin, first flange (x = r)
    idx = np.flatnonzero(on_flange1)
    chosen = rng.choice(idx, size=int(round(fraction * len(idx))), replace=False)
    pts[chosen, 0] += offset
    return pts, chosen


def test_a_dented_flange_is_refused_not_trimmed_and_a_warped_span_is_not_certified():
    feat = feature(v3)
    policy = v3.LayerSelectionPolicyV2()
    # (1) single skin with 12 % of its ORIGINAL rows displaced 0.2 mm: no second population (mass below the floor),
    #     all rows kept, the unchanged gate refuses NOT_PLANAR (the span is never certified by dropping the dent).
    single = bend(noise=0.004, seed=61)
    dented, chosen = _dent_inner_skin(single, .12, .2, 62)
    with pytest.raises(v3._Unavailable, match="FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL") as failure:
        measure(v3, dented, feat, policy)
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["identity"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED" and selection["kept_rows"] == selection["input_rows"]
    assert selection["modes"]["secondary"] is None and selection["modes"]["local_maximum_count"] >= 2
    assert failure.value.diagnostics["flange_support"]["per_role"][0]["excluded_bridging_rows"] == 0
    # (2) two skins with the same dent in the inner skin: the dent is neither skin; with the dent fraction inside the
    #     bridging bound the skin is measured and the excluded rows are PUBLISHED, with it above the bound it abstains.
    two = bend(thickness=1.5, noise=0.004, seed=63, edge_spacing=0.3)
    dented_two, chosen = _dent_inner_skin(two, .12, .2, 64)
    new = measure(v3, dented_two, feat, policy)
    role0 = new["diagnostics"]["flange_support"]["per_role"][0]
    assert role0["excluded_bridging_rows"] >= len(chosen) and role0["excluded_opposite_skin_rows"] > 0
    assert new["diagnostics"]["flange_support"]["semantics"].startswith("SKIN_MEMBERSHIP_SUBSET")
    kept = set(np.asarray(new["_instance_support"]["flange_row_indices"][0]).tolist())
    assert not (kept & set(chosen.tolist()))                                         # displaced rows are never measured rows
    assert role0["excluded_bridging_rows"] + role0["excluded_opposite_skin_rows"] + role0["kept_rows"] == role0["crop_rows"]
    assert new["diagnostics"]["flange_support"]["semantics"].endswith("COVERAGE_NOT_ESTABLISHED")
    assert new["continuous_surface_coverage"] == "NOT_ESTABLISHED" and new["physical_trim_endpoints"] is None
    with pytest.raises(v3._Unavailable, match="FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY") as failure:
        measure(v3, dented_two, feat, v3.LayerSelectionPolicyV2(max_bridging_row_fraction=.05))
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["identity"] == "TWO_MODES_MEMBERSHIP_UNRESOLVED" and selection["unresolved_reason"] == "BRIDGING_FRACTION_ABOVE_DECLARED_MAXIMUM"
    # (2b) the same dent toward the AIR side (outside the sandwich): the displaced rows are NOT between the skins, stay
    #      in their nearest (inner) group by the membership rule, and the unchanged gate refuses the dented flange.
    dented_air, chosen_air = _dent_inner_skin(two, .12, -.2, 65)
    with pytest.raises(v3._Unavailable, match="FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL") as failure:
        measure(v3, dented_air, feat, policy)
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    in_crop = set(_crop_index(dented_air, feat, 0).tolist()) & set(chosen_air.tolist())
    assert len(in_crop) > 100 and selection["outside_sandwich_member_rows"] == len(in_crop) and selection["refit_converged"] is True
    assert selection["kept_rows"] == max(selection["group_rows"]) and failure.value.diagnostics["failure_phase"] == "PLANE_FIT_ROLE_0"
    assert failure.value.diagnostics["flange_support"]["per_role"][0]["outside_sandwich_member_rows"] == len(in_crop)
    # (3) a contiguous warped span (35 % of the first flange displaced 0.5 mm along the flange's far end) is a second
    #     mode but not a compatible parallel pair of the same footprint: unresolved, all rows kept, refused.
    warped = single.copy()
    far = (np.abs(warped[:, 0] - 0.8) < 0.05) & (warped[:, 1] < -9.0)
    warped[far, 0] += 0.5
    with pytest.raises(v3._Unavailable) as failure:
        measure(v3, warped, feat, policy)
    selection = failure.value.diagnostics["flanges"][0]["layer_selection"]
    assert selection["kept_rows"] == selection["input_rows"] and selection["identity"] in (
        "TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED", "SINGLE_LAYER_IDENTITY_UNRESOLVED", "PROJECTION_PREMISE_UNRESOLVED",
        "TWO_MODES_MEMBERSHIP_UNRESOLVED", "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED")
    assert str(failure.value) in ("FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL", "FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY")
