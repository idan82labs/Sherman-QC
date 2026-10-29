"""Frozen all-support scale challenge, not automatic dense bend discovery.

Before first execution: N=2,000,000, chunk=8,192, R=2 mm, partial arc
25--110 degrees, ideal axial range [-5,5] mm, epsilon=1e-6 mm, and exact
stipulated z-axis through the origin. Every valid row is cylinder-support
evidence, not background. Targets are 120 s and 512 MiB per complete check.
Theta and z are correlated: this is dense sampling along a 1-D partial helical
trace, not 2-D surface-area coverage. The fixed distribution is not changed.

No RNG, physical input, native runtime, scan-sized array or retained row set.
Only the fixed negative changes one late row; it does not tune the prior.
"""
import json
import math
import resource
import sys
import time

import numpy as np
import pytest

from research.geometry.dense_axis_tube_radius import analyze_dense_axis_tube_radius


ROW_COUNT = 2_000_000
CHUNK_ROWS = 8192
TRUE_RADIUS = 2.0
CONTRADICTORY_RADIUS = 2.01
EPSILON = 1e-6
RADIUS_PRIOR = (1.99, 2.02)
WALL_TARGET_SECONDS = 120.0
RSS_TARGET_BYTES = 512 * 1024**2


def points_for_ids(start, stop, *, contradictory_row=None):
    """Reconstruct any bounded row range from IDs, without an N-row allocation."""
    assert 0 <= start <= stop <= ROW_COUNT
    assert stop-start <= CHUNK_ROWS
    ids = np.arange(start, stop, dtype=np.int64)
    fraction = ids / float(ROW_COUNT-1)
    theta = np.deg2rad(25.0 + 85.0*fraction)
    radial_error = 0.5*EPSILON*((ids % 17)-8)/8.0
    axial_error = 0.25*EPSILON*(2*(ids % 2)-1)
    radius = TRUE_RADIUS + radial_error
    if contradictory_row is not None and start <= contradictory_row < stop:
        radius[contradictory_row-start] = CONTRADICTORY_RADIUS
    ideal_z = -5.0 + 10.0*fraction
    points = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), ideal_z+axial_error))
    return points


def support_stream(stats, *, contradiction=False, late_nan=False):
    """Each valid row contributes; NaN gets its own chunk after a contradiction."""
    valid_count = ROW_COUNT-1 if late_nan else ROW_COUNT
    contradiction_id = valid_count-1 if contradiction else None
    stats.update(generated_valid_rows=0, generated_valid_chunks=0,
                 peak_generated_chunk_rows=0, complete_source_eof=False,
                 nan_chunk_delivered=False, all_valid_rows_have_distinct_z=True)
    previous_z = -math.inf
    for start in range(0, valid_count, CHUNK_ROWS):
        stop = min(start+CHUNK_ROWS, valid_count)
        points = points_for_ids(start, stop, contradictory_row=contradiction_id)
        # Ideal pitch 10/(N-1) exceeds the largest possible 0.5*epsilon
        # inter-row axial-error jump. This checks all rows, not a sample.
        assert points[0, 2] > previous_z
        assert np.all(np.diff(points[:, 2]) > 0)
        previous_z = float(points[-1, 2])
        stats["generated_valid_rows"] += len(points)
        stats["generated_valid_chunks"] += 1
        stats["peak_generated_chunk_rows"] = max(stats["peak_generated_chunk_rows"], len(points))
        yield points
    if late_nan:
        stats["nan_chunk_delivered"] = True
        yield np.asarray([[math.nan, 0.0, 0.0]])
    stats["complete_source_eof"] = True


def analyze(chunks):
    return analyze_dense_axis_tube_radius(
        chunks, unit="mm", axis=[0.0, 0.0, 1.0], axis_point=[0.0, 0.0, 0.0],
        axis_angle_bound=0.0, axis_point_error_bound=0.0,
        position_error_bound=EPSILON, radius_bounds=RADIUS_PRIOR,
        max_total_rows=ROW_COUNT, max_chunk_rows=CHUNK_ROWS, max_chunks=4096,
    )


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_bend_count"] is None
    assert result["shared_model_feasibility_is_established"] is False
    assert result["hypothesis"]["axis_and_point_are_verified"] is False
    assert result["hypothesis"]["support_is_verified"] is False
    assert result["hypothesis"]["axis_point_bound_is_inferred_from_measurement_epsilon"] is False
    assert result["policy"]["all_supplied_rows_are_in_the_single_support_hypothesis"] is True
    assert result["policy"]["sampling_or_residual_trimming"] is False
    assert result["policy"]["eof_required_for_any_geometric_conclusion"] is True
    assert result["diagnostics"]["retained_raw_rows"] == 0
    assert "NO_COMPRESSION_OR_PROPOSAL_DISCOVERY" in result["limitations"]
    assert "STREAM_ROWS_AUDITED_NOT_DURABLY_RETAINED_MEASUREMENT_ARCHIVE" in result["limitations"]


def complete_all_row_receipt(result, stats):
    scope(result)
    assert stats["complete_source_eof"]
    assert stats["generated_valid_rows"] == ROW_COUNT
    assert stats["generated_valid_chunks"] == math.ceil(ROW_COUNT/CHUNK_ROWS)
    assert stats["peak_generated_chunk_rows"] == CHUNK_ROWS
    assert result["stream_fully_consumed"]
    assert result["search_complete_within_declared_policy"]
    assert result["input_row_count"] == ROW_COUNT
    assert result["diagnostics"]["constrained_raw_rows"] == ROW_COUNT
    assert result["diagnostics"]["processed_chunks"] == stats["generated_valid_chunks"]
    assert result["diagnostics"]["received_chunks"] == stats["generated_valid_chunks"]
    assert result["diagnostics"]["peak_chunk_rows"] == CHUNK_ROWS
    assert result["diagnostics"]["max_propagated_error_bound"] == EPSILON


def row_constraint_oracle(row_id, *, contradictory_row=None):
    """Independent scalar formula checks the reported global extremum IDs."""
    p = points_for_ids(row_id, row_id+1, contradictory_row=contradictory_row)[0]
    radial = math.hypot(float(p[0]), float(p[1]))
    footprint = math.hypot(float(p[0]), float(p[1]), float(p[2]))
    guard = 512*float(np.finfo(float).eps)*(max(map(abs, p))+footprint+radial+EPSILON)
    return max(0.0, radial-EPSILON-guard), radial+EPSILON+guard


def extremum_row_receipts(result, *, contradictory_row=None):
    extrema = result["constraint_extrema"]
    for bound, row_key, side in [
        ("max_sample_lower", "max_sample_lower_row", 0),
        ("min_sample_upper", "min_sample_upper_row", 1),
    ]:
        row_id = extrema[row_key]
        assert isinstance(row_id, int) and 0 <= row_id < ROW_COUNT
        reconstructed = row_constraint_oracle(row_id, contradictory_row=contradictory_row)[side]
        assert extrema[bound] == pytest.approx(reconstructed, rel=0.0, abs=1e-14)
    assert extrema["prior_intersection_lower"] == max(RADIUS_PRIOR[0], extrema["max_sample_lower"])
    assert extrema["prior_intersection_upper"] == min(RADIUS_PRIOR[1], extrema["min_sample_upper"])


def resource_receipt(start, result, name):
    # ru_maxrss is a process-lifetime high-water mark, not this call's delta.
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(raw_rss if sys.platform == "darwin" else raw_rss*1024)
    elapsed = time.perf_counter()-start
    receipt = dict(
        name=name, supplied_valid_rows=result["input_row_count"],
        constrained_raw_rows=result["diagnostics"]["constrained_raw_rows"],
        processed_chunks=result["diagnostics"]["processed_chunks"],
        status=result["status"], radius_interval=result["radius_interval"],
        constraint_extrema=result["constraint_extrema"],
        analysis_and_verification_wall_seconds=elapsed,
        process_lifetime_peak_rss_bytes=peak_bytes,
        wall_target_seconds=WALL_TARGET_SECONDS, rss_target_bytes=RSS_TARGET_BYTES,
        scope="ALL_DENSE_ROWS_AUDITED_UNDER_SUPPLIED_AXIS_AND_SUPPORT_NOT_DISCOVERY_OR_PHYSICAL_QUALIFICATION",
    )
    assert elapsed <= WALL_TARGET_SECONDS
    assert peak_bytes <= RSS_TARGET_BYTES
    print("DENSE_AXIS_TUBE_SCALE_RECEIPT " + json.dumps(receipt, sort_keys=True, allow_nan=False))


def test_two_million_distinct_dense_support_rows_preserve_true_radius():
    start = time.perf_counter()
    # Orthogonal radial/axial errors have norm <=sqrt(5)/4*epsilon.
    # Their strict slack also accommodates ordinary trig rounding in this
    # invented fixture; epsilon is never estimated from the returned fit.
    assert math.hypot(0.5*EPSILON, 0.25*EPSILON) < EPSILON
    assert 10.0/(ROW_COUNT-1) > 0.5*EPSILON
    stats = {}
    result = analyze(support_stream(stats))
    complete_all_row_receipt(result, stats)
    assert result["status"] == "NECESSARY_RADIUS_CONSTRAINTS"
    assert result["radius_interval"][0] <= TRUE_RADIUS <= result["radius_interval"][1]
    assert result["radius_interval"][1]-result["radius_interval"][0] < 2.0*EPSILON
    assert result["reasons"] == []
    extremum_row_receipts(result)
    resource_receipt(start, result, "all_support_partial_arc_positive")


def test_last_dense_row_excludes_model_that_a_first_256_sample_would_not_exclude():
    start = time.perf_counter()
    prefix = analyze([points_for_ids(0, 256)])
    scope(prefix)
    assert prefix["status"] == "NECESSARY_RADIUS_CONSTRAINTS"
    assert prefix["input_row_count"] == 256
    assert prefix["radius_interval"][0] <= TRUE_RADIUS <= prefix["radius_interval"][1]
    # The prior admits BOTH radii. Exclusion must come from auditing their
    # incompatible joint support, not from a prior that rules out 2.01 alone.
    assert RADIUS_PRIOR[0] < TRUE_RADIUS < CONTRADICTORY_RADIUS < RADIUS_PRIOR[1]
    stats = {}
    result = analyze(support_stream(stats, contradiction=True))
    complete_all_row_receipt(result, stats)
    assert result["status"] == "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
    assert result["radius_interval"] is None
    assert result["reasons"] == ["EMPTY_NECESSARY_RADIUS_INTERSECTION"]
    extrema = result["constraint_extrema"]
    assert extrema["max_sample_lower_row"] == ROW_COUNT-1
    assert extrema["min_sample_upper_row"] < ROW_COUNT-1
    assert extrema["max_sample_lower"] > extrema["min_sample_upper"] + 0.009
    extremum_row_receipts(result, contradictory_row=ROW_COUNT-1)
    resource_receipt(start, result, "last_row_contradiction_not_visible_in_256_prefix")


def test_late_nan_invalidates_an_already_audited_dense_contradiction():
    start = time.perf_counter()
    stats = {}
    result = analyze(support_stream(stats, contradiction=True, late_nan=True))
    scope(result)
    assert stats["generated_valid_rows"] == ROW_COUNT-1
    assert stats["nan_chunk_delivered"]
    assert not stats["complete_source_eof"]
    assert result["input_row_count"] == ROW_COUNT-1
    assert result["diagnostics"]["constrained_raw_rows"] == ROW_COUNT-1
    assert result["diagnostics"]["processed_chunks"] == math.ceil((ROW_COUNT-1)/CHUNK_ROWS)
    assert result["diagnostics"]["received_chunks"] == result["diagnostics"]["processed_chunks"]+1
    # This diagnostic proves the separate preceding valid chunk reached the
    # radial arithmetic, not merely that a contradictory row was generated.
    assert result["diagnostics"]["max_nominal_axis_distance"] == pytest.approx(CONTRADICTORY_RADIUS, abs=1e-14)
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["NONFINITE_CHUNK_COORDINATE"]
    assert result["radius_interval"] is None
    assert result["constraint_extrema"] is None
    assert result["stream_fully_consumed"] is False
    assert result["search_complete_within_declared_policy"] is False
    resource_receipt(start, result, "late_nan_clears_dense_contradiction")
