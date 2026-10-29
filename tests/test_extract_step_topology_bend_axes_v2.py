"""Pure native-row mathematics; no STEP input or native geometry job."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from scripts import extract_step_topology_bend_axes_v2 as candidate


@pytest.mark.parametrize("sweep,accepted", [
    (15., True), (15.-1e-7, True), (15.+1e-7, True),
    (15.-1e-6, False), (15.+1e-6, True),
    (14., False), (0., False), (90., True), (200., True),
    (200.+1e-7, False), (200.+1e-6, False),
    (math.inf, False), (-math.inf, False), (math.nan, False),
])
def test_lower_minimum_uses_published_six_decimal_representation_only(sweep, accepted):
    assert candidate._eligible_native_cylinder(sweep, 40.) is accepted


@pytest.mark.parametrize("span,accepted", [(8., True), (8.-1e-7, False),
    (8.+1e-7, True), (0., False), (-10., False), (math.nan, False), (math.inf, False)])
def test_raw_axial_span_threshold_is_unchanged(span, accepted):
    assert candidate._eligible_native_cylinder(15., span) is accepted


def native_row(index, radius, sweep=15., offset=0.):
    u = math.radians(sweep)/2.
    return {"face_index": index, "radius_mm": radius, "angular_sweep_deg": sweep,
        "axial_span_mm": 40., "line_start": [10.+offset, 20., 0.],
        "line_end": [10.+offset, 20., 40.],
        "turn_midpoint_center": [10.+offset+radius*math.cos(u),
                                 20.+radius*math.sin(u), 20.]}


def eligible_rows(rows):
    return [r for r in rows if candidate._eligible_native_cylinder(r["angular_sweep_deg"], r["axial_span_mm"])]


def test_boundary_pair_retains_raw_sweeps_and_recovers_complete_axis():
    rows = [native_row(2, 1.2, 15.-1e-7), native_row(5, 2.7, 15.+1e-7)]
    prior = copy.deepcopy(rows)
    axes, rejected = candidate._group_cylinder_rows(eligible_rows(rows))
    assert len(axes) == 1 and rejected == []
    axis = axes[0]
    assert axis["face_indices"] == [2, 5]
    assert axis["radii_mm"] == [1.2, 2.7]
    assert axis["angular_sweeps_deg"] == [15., 15.]
    assert axis["angular_sweeps_unrounded_deg"] == [15.-1e-7, 15.+1e-7]
    assert axis["native_sweep_rows"] == [
        {"face_index": 2, "unrounded_deg": 15.-1e-7, "published_deg": 15.},
        {"face_index": 5, "unrounded_deg": 15.+1e-7, "published_deg": 15.}]
    midpoint = np.mean([r["turn_midpoint_center"] for r in rows], axis=0)
    assert axis["canonical_line_start"] == pytest.approx(midpoint-[0., 0., 20.])
    assert axis["canonical_line_end"] == pytest.approx(midpoint+[0., 0., 20.])
    assert axis["canonical_line_rule"] == "NEUTRAL_RADIUS_HALF_ANGULAR_TURN"
    assert rows == prior


def test_real_below_boundary_inner_face_still_leaves_rejected_singleton_group():
    rows = [native_row(2, 1.2, 15.-1e-6), native_row(5, 2.7, 15.+1e-7)]
    axes, rejected = candidate._group_cylinder_rows(eligible_rows(rows))
    assert axes == [] and len(rejected) == 1
    assert rejected[0]["face_indices"] == [5]
    assert rejected[0]["rejection_reason"] == "missing_plausible_inner_outer_cylinder_pair"


def test_coaxial_offset_and_missing_radius_partner_are_unchanged_refusals():
    rows = [native_row(2, 1.2), native_row(5, 2.7, offset=1.)]
    axes, rejected = candidate._group_cylinder_rows(rows)
    assert axes == [] and len(rejected) == 2
    axes, rejected = candidate._group_cylinder_rows([native_row(2, 1.2), native_row(5, 1.2)])
    assert axes == [] and len(rejected) == 1


def test_pair_requires_existing_radius_separation_domain():
    axes, rejected = candidate._group_cylinder_rows([native_row(2, 1.2), native_row(5, 1.3)])
    assert axes == [] and len(rejected) == 1


def test_native_signed_v_coordinates_are_preserved_before_axis_ordering():
    ends = candidate._cylinder_axis_endpoints(np.array([10., 0., 0.]),
        np.array([-1., 0., 0.]), 2., 5.)
    assert ends == pytest.approx(np.array([[5., 0., 0.], [8., 0., 0.]]))


def test_representation_policy_is_explicit_and_not_an_added_cutoff_parameter():
    assert candidate.ANGULAR_PUBLICATION_DECIMALS == 6
    assert candidate.MINIMUM_SWEEP_REPRESENTATION_POLICY == "SIX_DECIMAL_DEGREES_AT_EXISTING_MINIMUM"
    assert candidate.MIN_ANGULAR_SWEEP_DEG == 15.
    assert candidate.MAX_ANGULAR_SWEEP_DEG == 200.
