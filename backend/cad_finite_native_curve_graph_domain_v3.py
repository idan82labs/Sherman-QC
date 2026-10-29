"""Detection-only nominal native rational boundary-chain interior certificate.

Original bases are validated by V2's positive-weight monotonicity proof. This
does not certify full trim accuracy, manufacturing units, or an unfolding.
"""
import math

from backend.cad_finite_native_curve_graph_domain_v2 import _basis


def certify_native_curve_graph_domain(curves, *, face_tolerance_mm,
                                     native_brep_valid, complete_outer_wire,
                                     chart_branch_proven, holes_present=False):
    base = dict(schema="finite-native-curve-graph-v3", status="REFUSED", type_proven=False,
                detection_only=True, unfold_certificate=False, full_boundary_geometry_calibrated=False,
                units="native chart units; mm assumed not independently verified",
                geometry_sampled=False)
    def refuse(reason):
        return {**base, "reason": reason}
    if native_brep_valid is not True or complete_outer_wire is not True or chart_branch_proven is not True:
        return refuse("NATIVE_VALIDITY_CENSUS_AND_CHART_BRANCH_REQUIRED")
    if holes_present is not False:
        return refuse("HOLES_UNSUPPORTED")
    tol = face_tolerance_mm
    if type(tol) not in (int, float) or not math.isfinite(tol) or not 0 < tol <= 1e-5:
        return refuse("SOURCE_FACE_TOLERANCE_INVALID")
    if not isinstance(curves, list) or not 4 <= len(curves) <= 64:
        return refuse("NATIVE_EDGE_COUNT_BUDGET")
    try:
        ids = [c["native_edge_id"] for c in curves]
        if any(type(i) is not int or i <= 0 for i in ids) or len(set(ids)) != len(ids):
            return refuse("UNIQUE_NATIVE_EDGES_REQUIRED")
        if sum(len(c["poles"]) for c in curves) > 4096:
            return refuse("TOTAL_NATIVE_COEFFICIENT_BUDGET")
        for c in curves:
            if any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in c["poles"]):
                return refuse("NONFINITE_COEFFICIENT")
        ranges = [(min(p[0] for p in c["poles"]), max(p[0] for p in c["poles"])) for c in curves]
        xmin, xmax = min(a for a, b in ranges), max(b for a, b in ranges)
        if xmax - xmin <= 4 * tol:
            return refuse("NO_DISTINCT_RAIL_BANDS")
        kinds = []
        for low, high in ranges:
            if high - low <= tol and high <= xmin + tol:
                kinds.append("L")
            elif high - low <= tol and low >= xmax - tol:
                kinds.append("R")
            else:
                kinds.append("E")
        # Rotate only traversal metadata, never curves or coordinates.
        transitions = [i for i in range(len(curves)) if kinds[i] != kinds[i-1]]
        if not transitions:
            return refuse("TWO_RAIL_CHAINS_TWO_END_CHAINS_REQUIRED")
        rotation = transitions[0]
        order = list(range(rotation, len(curves))) + list(range(rotation))
        blocks = []
        for i in order:
            if not blocks or blocks[-1][0] != kinds[i]:
                blocks.append((kinds[i], [i]))
            else:
                blocks[-1][1].append(i)
        if len(blocks) != 4 or sorted(k for k, _ in blocks) != ["E", "E", "L", "R"]:
            return refuse("TWO_RAIL_CHAINS_TWO_END_CHAINS_REQUIRED")
        if any(blocks[i][0] == "E" and blocks[(i+1)%4][0] == "E" for i in range(4)):
            return refuse("RAIL_END_CHAIN_ALTERNATION_REQUIRED")
        join_gaps = []
        for i, c in enumerate(curves):
            nxt = curves[(i+1)%len(curves)]
            vid = c["native_end_vertex_id"]
            if type(vid) is not int or vid <= 0 or nxt["native_start_vertex_id"] != vid:
                return refuse("SHARED_NATIVE_VERTEX_IDENTITY_REQUIRED")
            gap = math.dist(c["poles"][-1], nxt["poles"][0])
            if gap > tol:
                return refuse("MEASURED_CHART_JOIN_EXCEEDS_FACE_TOLERANCE")
            join_gaps.append(gap)
        scale = max(1., *(abs(v) for c in curves for p in c["poles"] for v in p))
        guard = 64 * math.ulp(scale)
        exclusion = tol + guard
        proofs, forbidden, end_bounds, end_coverage, end_edge_coverage, rail_blocks = [], [], [], [], [], {}
        for kind, indices in blocks:
            if kind != "E":
                rail_blocks[kind] = indices
                directions = []
                for i in indices:
                    proofs.append({"native_edge_id": ids[i], **_basis(curves[i], 1), "role": "RAIL"})
                    directions.append(1 if curves[i]["poles"][-1][1] > curves[i]["poles"][0][1] else -1)
                if len(set(directions)) != 1:
                    return refuse("RAIL_CHAIN_BACKTRACK")
                for i, j in zip(indices, indices[1:]):
                    if directions[0] * (curves[j]["poles"][0][1] - curves[i]["poles"][-1][1]) < 0:
                        return refuse("RAIL_CHAIN_JOIN_BACKTRACK")
                continue
            directions = []
            covered_intervals = []
            for i in indices:
                c = curves[i]
                vertical = ranges[i][1] - ranges[i][0] <= tol
                proofs.append({"native_edge_id": ids[i], **_basis(c, 1 if vertical else 0),
                               "role": "EXCLUDED_VERTICAL_CONNECTOR" if vertical else "END_GRAPH"})
                if vertical:
                    forbidden.append((ranges[i][0] - exclusion, ranges[i][1] + exclusion))
                else:
                    directions.append(1 if c["poles"][-1][0] > c["poles"][0][0] else -1)
                    covered_intervals.append(sorted([c["poles"][0][0], c["poles"][-1][0]]))
            if not directions or len(set(directions)) != 1:
                return refuse("END_CHAIN_ANGULAR_BACKTRACK")
            # All internal joins are excluded, including exact joins: the safe
            # components never fill an unobserved chart gap or vertical fiber.
            for i, j in zip(indices, indices[1:]):
                a, b = curves[i]["poles"][-1][0], curves[j]["poles"][0][0]
                forbidden.append((min(a, b)-exclusion, max(a, b)+exclusion))
                if directions[0] * (b-a) < -tol:
                    return refuse("END_CHAIN_JOIN_BACKTRACK")
            end_bounds.append((min(p[1] for i in indices for p in curves[i]["poles"]),
                               max(p[1] for i in indices for p in curves[i]["poles"])))
            # Explicit endpoint coverage of the complete chain; rails alone
            # never authorize extending an end graph over a missing interval.
            endpoints = [curves[indices[0]]["poles"][0][0], curves[indices[-1]]["poles"][-1][0]]
            end_coverage.append((min(endpoints), max(endpoints)))
            end_edge_coverage.append(covered_intervals)
        end_bounds.sort()
        y0 = math.nextafter(end_bounds[0][1] + exclusion, math.inf)
        y1 = math.nextafter(end_bounds[1][0] - exclusion, -math.inf)
        left = max(ranges[i][1] for i in rail_blocks["L"])
        right = min(ranges[i][0] for i in rail_blocks["R"])
        x0 = math.nextafter(max(left, *(a for a, b in end_coverage))+exclusion, math.inf)
        x1 = math.nextafter(min(right, *(b for a, b in end_coverage))-exclusion, -math.inf)
        if x1 <= x0 or y1 <= y0:
            return refuse("NO_CERTIFIED_POSITIVE_INTERIOR_CORE")
        components = [(x0, x1)]
        for low, high in sorted(forbidden):
            pieces = []
            for a, b in components:
                if high <= a or low >= b:
                    pieces.append((a, b))
                else:
                    if a < low:
                        pieces.append((a, math.nextafter(low, -math.inf)))
                    if high < b:
                        pieces.append((math.nextafter(high, math.inf), b))
            components = [(a, b) for a, b in pieces if b > a]
        components = [(a, b) for a, b in components
                      if all(any(low <= a and b <= high for low, high in coverage)
                             for coverage in end_edge_coverage)]
        if not components:
            return refuse("NO_SAFE_ANGULAR_COMPONENT")
        chosen = min(components, key=lambda p: (-(p[1]-p[0]), p[0]))
        rails = rail_blocks["L"] + rail_blocks["R"]
        return {**base, "status": "PROVED", "type_proven": True,
                "proof_kind": "NOMINAL_NATIVE_RATIONAL_BOUNDARY_CHAIN_INTERIOR_CORE",
                "safe_rectangular_interior_core_mm": [*chosen, y0, y1],
                "safe_angular_components_metric_mm": [list(c) for c in components],
                "component_selection": "WIDEST_THEN_LOWEST_SOURCE_ANGULAR_COORDINATE",
                "excluded_angular_bands_metric_mm": [list(b) for b in sorted(forbidden)],
                "complete_end_chain_angular_coverage_metric_mm": [list(c) for c in end_coverage],
                "original_nonvertical_end_edge_coverage_metric_mm": end_edge_coverage,
                "each_component_covered_by_both_end_chains": True,
                "angular_support_metric_mm": list(chosen),
                "core_semantics": "STRICT_INTERIOR_SUBSET_NOT_FULL_FILLED_BOUNDING_RECTANGLE",
                "ordered_original_native_curves": curves, "ordered_original_boundary": curves,
                "basis_proofs": proofs, "boundary_chain_edge_ids": [{"role": k, "native_edge_ids": [ids[i] for i in ix]} for k, ix in blocks],
                "maximum_measured_chart_join_gap_mm": max(join_gaps),
                "source_face_tolerance_mm": float(tol), "face_tolerance_mm": float(tol),
                "native_rail_edge_ids": [ids[i] for i in rails],
                "isoangular_rail_native_edge_ids": [ids[i] for i in rails],
                "native_isoangular_rail_centers_metric_mm": [(xmin+left)/2, (right+xmax)/2],
                "kernel_vertex_uncertainty_retained": [c.get("kernel_vertex_tolerance_mm") for c in curves],
                "pairing_material_ownership": "REQUIRED_SEPARATELY",
                "topology_assumptions": "Valid complete single native outer wire, shared native vertices, two monotone rails, separated nonbacktracking end chains; internal vertical fibers and join uncertainty excluded from core."}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return refuse(str(exc))
