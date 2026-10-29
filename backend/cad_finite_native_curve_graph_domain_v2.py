"""Nominal finite-cylinder graph core from original active rational curves.

Detection only. Positive-weight normalized B-spline bases retain total positivity;
ordered scalar controls preserve monotonicity. Every active span must be strict
to exclude vertical plateaus. No Bezier decomposition or coefficient snapping.
Native vertex uncertainty is retained separately, not claimed calibrated away.

Primary mathematical background: Carnicer/Peña, "Spline functions and total
positivity", https://www.mat.ucm.es/serv/revmat/vol9-e/vol9-ee.pdf; rational
normalization preserves positive minors by positive row/column scaling.
"""
import math


def _basis(curve, coordinate):
    poles, weights, basis = curve["poles"], curve["weights"], curve["basis"]
    n = len(poles)
    if not 2 <= n <= 4096 or len(weights) != n:
        raise ValueError("COEFFICIENT_BUDGET_OR_WEIGHT_COUNT")
    if any(len(p) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in p) for p in poles):
        raise ValueError("NONFINITE_COEFFICIENT")
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("NONPOSITIVE_WEIGHT")
    kind, degree = basis["kind"], basis["degree"]
    if type(degree) is not int or not 1 <= degree < n or basis.get("actual_interval_proven") is not True:
        raise ValueError("BASIS_OR_ACTIVE_INTERVAL_UNPROVEN")
    actual, restricted = basis["actual_interval"], basis["restricted_interval"]
    if len(actual) != 2 or actual != restricted or not all(type(x) in (int, float) and math.isfinite(x) for x in actual) or actual[1] <= actual[0]:
        raise ValueError("EXACT_ACTIVE_INTERVAL_REQUIRED")
    values = [p[coordinate] for p in poles]
    sign = 1 if values[-1] > values[0] else -1
    ordered = [sign * v for v in values]
    if ordered[-1] <= ordered[0] or any(a > b for a, b in zip(ordered, ordered[1:])):
        raise ValueError("CONTROL_MONOTONICITY_UNPROVEN")
    if kind in ("LINE", "BEZIER"):
        if degree != n - 1 or kind == "LINE" and n != 2:
            raise ValueError("INVALID_BERNSTEIN_BASIS")
        return {"basis": kind, "strict_graph": True, "active_span_count": 1}
    if kind != "BSPLINE" or basis.get("periodic") is not False:
        raise ValueError("UNSUPPORTED_OR_PERIODIC_BASIS")
    knots, mults = basis["knots"], basis["multiplicities"]
    if len(knots) < 2 or len(knots) != len(mults) or any(type(x) not in (int, float) or not math.isfinite(x) for x in knots) or any(a >= b for a,b in zip(knots, knots[1:])):
        raise ValueError("INVALID_KNOT_VECTOR")
    if any(type(m) is not int or m < 1 for m in mults) or mults[0] != degree+1 or mults[-1] != degree+1 or any(m > degree for m in mults[1:-1]) or sum(mults) != n+degree+1:
        raise ValueError("UNCLAMPED_OR_DISCONTINUOUS_BASIS")
    if actual != [knots[0], knots[-1]]:
        raise ValueError("ACTIVE_INTERVAL_NOT_EXACT_CLAMPED_BASIS")
    expanded = [k for k,m in zip(knots,mults) for _ in range(m)]
    span_count = 0
    for j in range(degree, n):
        if expanded[j] < expanded[j+1]:
            span_count += 1
            if ordered[j] <= ordered[j-degree]:
                raise ValueError("CONSTANT_COORDINATE_ACTIVE_SPAN")
    return {"basis": kind, "strict_graph": True, "active_span_count": span_count,
            "proof": "POSITIVE_NORMALIZED_TOTALLY_POSITIVE_BASIS_AND_ORDERED_CONTROLS"}


def certify_native_curve_graph_domain(curves, *, face_tolerance_mm,
                                     native_brep_valid, complete_outer_wire,
                                     chart_branch_proven):
    base = {"schema": "finite-native-curve-graph-v2", "status": "REFUSED", "type_proven": False,
            "detection_only": True, "unfold_certificate": False,
            "full_boundary_geometry_calibrated": False,
            "units": "native chart units; mm assumed not independently verified"}
    def refusal(reason):
        return {**base, "reason": reason}
    if native_brep_valid is not True or complete_outer_wire is not True or chart_branch_proven is not True:
        return refusal("NATIVE_VALIDITY_CENSUS_AND_CHART_BRANCH_REQUIRED")
    if type(face_tolerance_mm) not in (float, int) or not math.isfinite(face_tolerance_mm) or not 0 < face_tolerance_mm <= 1e-5:
        return refusal("SOURCE_FACE_TOLERANCE_INVALID")
    if not isinstance(curves, list) or len(curves) != 4:
        return refusal("FOUR_NATIVE_EDGES_SINGLE_OUTER_WIRE_REQUIRED")
    try:
        if len({c["native_edge_id"] for c in curves}) != 4 or any(type(c["native_edge_id"]) is not int or c["native_edge_id"] <= 0 for c in curves):
            return refusal("UNIQUE_NATIVE_EDGES_REQUIRED")
        if sum(len(c["poles"]) for c in curves) > 4096:
            return refusal("TOTAL_NATIVE_COEFFICIENT_BUDGET")
        # Validate coefficients first using either coordinate; classification is
        # based on full coefficient enclosure, never endpoint samples alone.
        rail_indices = []
        proofs = []
        for i,c in enumerate(curves):
            for p in c["poles"]:
                if len(p) != 2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p):
                    raise ValueError("NONFINITE_COEFFICIENT")
            xs = [p[0] for p in c["poles"]]
            rail = max(xs)-min(xs) <= face_tolerance_mm
            proofs.append(_basis(c, 1 if rail else 0))
            if rail: rail_indices.append(i)
        if len(rail_indices) != 2 or (rail_indices[1]-rail_indices[0]) % 2 != 0:
            return refusal("TWO_OPPOSITE_SOURCE_BOUNDED_RAILS_REQUIRED")
        gaps = []
        for i,c in enumerate(curves):
            nxt = curves[(i+1)%4]
            vid = c["native_end_vertex_id"]
            if type(vid) is not int or vid <= 0 or nxt["native_start_vertex_id"] != vid:
                return refusal("SHARED_NATIVE_VERTEX_IDENTITY_REQUIRED")
            gap = math.dist(c["poles"][-1], nxt["poles"][0]); gaps.append(gap)
            if gap > face_tolerance_mm:
                return refusal("MEASURED_CHART_JOIN_EXCEEDS_FACE_TOLERANCE")
        ends = [i for i in range(4) if i not in rail_indices]
        z_ranges = [(min(p[1] for p in curves[i]["poles"]), max(p[1] for p in curves[i]["poles"])) for i in ends]
        low, high = sorted(range(2), key=lambda j:z_ranges[j][0])
        rail_ranges = sorted([(min(p[0] for p in curves[i]["poles"]),max(p[0] for p in curves[i]["poles"])) for i in rail_indices])
        end_ranges = [(min(c["poles"][0][0],c["poles"][-1][0]),max(c["poles"][0][0],c["poles"][-1][0])) for c in (curves[i] for i in ends)]
        scale = max(1., *(abs(x) for c in curves for p in c["poles"] for x in p))
        guard = 64*math.ulp(scale)
        exclusion = face_tolerance_mm + guard
        x0 = math.nextafter(max(rail_ranges[0][1], *(r[0] for r in end_ranges))+exclusion, math.inf)
        x1 = math.nextafter(min(rail_ranges[1][0], *(r[1] for r in end_ranges))-exclusion, -math.inf)
        y0 = math.nextafter(z_ranges[low][1]+guard, math.inf)
        y1 = math.nextafter(z_ranges[high][0]-guard, -math.inf)
        if x1 <= x0 or y1 <= y0:
            return refusal("NO_CERTIFIED_POSITIVE_INTERIOR_CORE")
        return {**base, "status": "PROVED", "type_proven": True,
                "proof_kind": "NOMINAL_FACE_TOLERANCE_INTERIOR_GRAPH_CORE",
                "safe_rectangular_interior_core_mm": [x0,x1,y0,y1],
                "angular_support_metric_mm": [x0,x1],
                "core_semantics": "STRICT_INTERIOR_SUBSET_NOT_FULL_FILLED_BOUNDING_RECTANGLE",
                "ordered_original_native_curves": curves, "basis_proofs": proofs,
                "ordered_original_boundary": curves,
                "maximum_measured_chart_join_gap_mm": max(gaps),
                "source_face_tolerance_mm": float(face_tolerance_mm),
                "face_tolerance_mm": float(face_tolerance_mm),
                "kernel_vertex_uncertainty_retained": [c.get("kernel_vertex_tolerance_mm") for c in curves],
                "native_rail_edge_ids": [curves[i]["native_edge_id"] for i in rail_indices],
                "isoangular_rail_native_edge_ids": [curves[i]["native_edge_id"] for i in rail_indices],
                "native_isoangular_rail_centers_metric_mm": [(min(p[0] for p in curves[i]["poles"])+max(p[0] for p in curves[i]["poles"]))/2 for i in rail_indices],
                "pairing_material_ownership": "REQUIRED_SEPARATELY"}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return refusal(str(exc))
