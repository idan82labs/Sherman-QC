"""Detection-only certificate for finite nonrectangular cylinder graph strips.

Input is the complete ordered outer wire in physical cylinder chart (r*theta,z),
decomposed into exact active-domain rational Bezier spans. No native imports,
sampling, reference labels, bend counts, or shape repair. Positive rational
weights and ordered u controls prove each end chain single-valued; separated
z coefficient hulls prove positive strip width. Holes deliberately refuse.
"""
import math


def certify_graph_strip(wires, *, native_brep_valid, exact_active_domains,
                       maximum_spans=256, maximum_poles=4096):
    return _certify(wires, native_brep_valid=native_brep_valid, exact_active_domains=exact_active_domains,
                    maximum_spans=maximum_spans, maximum_poles=maximum_poles, join_receipts=None)


def certify_single_edge_end_join_core(wires, *, native_brep_valid, exact_active_domains,
                                      join_receipts, maximum_spans=256, maximum_poles=4096):
    """Kernel-topology-qualified interior only; never fill endpoint gaps.

Native adapter must establish chart-gap budgets from source kernel tolerance,
not an arbitrary fitting tolerance. Receipts are aligned to span end joins.
Internal same-native-edge Bezier joins remain exactly closed.
"""
    return _certify(wires, native_brep_valid=native_brep_valid, exact_active_domains=exact_active_domains,
                    maximum_spans=maximum_spans, maximum_poles=maximum_poles, join_receipts=join_receipts)


def _certify(wires, *, native_brep_valid, exact_active_domains, maximum_spans,
             maximum_poles, join_receipts):
    receipt = {"schema": "finite-cylinder-graph-strip-v1", "status": "REFUSED",
               "type_proven": False, "detection_only": True,
               "unfold_certificate": False,
               "units": "native chart coordinate units; mm assumed by caller, not verified here",
               "locus_semantics": "ORIGINAL_CURVED_FINITE_BOUNDARY_NOT_RECTANGLE"}
    def refuse(reason):
        return {**receipt, "reason": reason}
    if native_brep_valid is not True or exact_active_domains is not True:
        return refuse("NATIVE_VALIDITY_AND_EXACT_ACTIVE_DOMAIN_REQUIRED")
    if not isinstance(wires, list) or len(wires) != 1:
        return refuse("HOLES_OR_MULTIPLE_WIRES_UNSUPPORTED")
    spans = wires[0]
    if not isinstance(spans, list) or not 4 <= len(spans) <= maximum_spans:
        return refuse("SPAN_BUDGET_OR_INSUFFICIENT_BOUNDARY")
    total = 0
    normalized = []
    for span in spans:
        poles, weights = span.get("poles"), span.get("weights")
        if not isinstance(poles, list) or not isinstance(weights, list) or len(poles) != len(weights) or len(poles) < 2:
            return refuse("INVALID_RATIONAL_BEZIER")
        total += len(poles)
        if total > maximum_poles:
            return refuse("POLE_BUDGET")
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 or any(type(x) not in (float, int) or not math.isfinite(x) for x in p) for p in poles):
            return refuse("NONFINITE_CHART_COEFFICIENT")
        if any(type(w) not in (float, int) or not math.isfinite(w) or w <= 0 for w in weights):
            return refuse("NONPOSITIVE_RATIONAL_WEIGHT")
        if type(span.get("native_edge_id")) is not int or span["native_edge_id"] <= 0 or span.get("actual_interval_proven") is not True:
            return refuse("SPAN_NATIVE_PROVENANCE_REQUIRED")
        normalized.append({**span, "poles": [list(map(float, p)) for p in poles]})
    qualified = join_receipts is not None
    join_bound = 0.0
    if qualified and (not isinstance(join_receipts, list) or len(join_receipts) != len(normalized)):
        return refuse("COMPLETE_NATIVE_JOIN_RECEIPTS_REQUIRED")
    # Strict mode tolerates no unproved snapping. Qualified mode validates native
    # identity but keeps all gap coordinates untouched and excludes side strips.
    for i, span in enumerate(normalized):
        next_span = normalized[(i + 1) % len(normalized)]
        if not qualified or span["native_edge_id"] == next_span["native_edge_id"]:
            if span["poles"][-1] != next_span["poles"][0]:
                return refuse("ORDERED_WIRE_NOT_EXACTLY_CLOSED")
            continue
        join_receipt = join_receipts[i]
        if not isinstance(join_receipt, dict):
            return refuse("COMPLETE_NATIVE_JOIN_RECEIPTS_REQUIRED")
        vid = join_receipt.get("shared_vertex_id")
        source = join_receipt.get("source_tolerance_mm")
        chart = join_receipt.get("chart_gap_budget_mm")
        if type(vid) is not int or vid <= 0 or span.get("native_end_vertex_id") != vid or next_span.get("native_start_vertex_id") != vid:
            return refuse("NATIVE_JOIN_VERTEX_IDENTITY_REQUIRED")
        if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 < x <= 1e-5 for x in (source, chart)):
            return refuse("NATIVE_SOURCE_JOIN_BUDGET_INVALID")
        gap = math.dist(span["poles"][-1], next_span["poles"][0])
        if gap > chart:
            return refuse("ACTUAL_JOIN_GAP_EXCEEDS_NATIVE_BUDGET")
        join_bound = max(join_bound, chart)
    rails = [i for i, s in enumerate(normalized) if all(p[0] == s["poles"][0][0] for p in s["poles"])]
    if len(rails) != 2:
        return refuse("EXACTLY_TWO_COMPLETE_ISOANGULAR_RAILS_REQUIRED")
    for i in rails:
        zs = [p[1] for p in normalized[i]["poles"]]
        if zs[-1] == zs[0] or not (all(a <= b for a, b in zip(zs, zs[1:])) or all(a >= b for a, b in zip(zs, zs[1:]))):
            return refuse("RAIL_INJECTIVITY_UNPROVEN")
    left, right = sorted(rails, key=lambda i: normalized[i]["poles"][0][0])
    u0, u1 = normalized[left]["poles"][0][0], normalized[right]["poles"][0][0]
    if not u1 > u0:
        return refuse("NONPOSITIVE_ANGULAR_SUPPORT")
    chains = []
    for start, end in ((left, right), (right, left)):
        chain = []
        i = (start + 1) % len(normalized)
        while i != end:
            chain.append(normalized[i]); i = (i + 1) % len(normalized)
        if not chain:
            return refuse("MISSING_CURVED_END_CHAIN")
        direction = 1 if chain[-1]["poles"][-1][0] > chain[0]["poles"][0][0] else -1
        for span in chain:
            us = [direction * p[0] for p in span["poles"]]
            if not us[-1] > us[0] or any(a > b for a, b in zip(us, us[1:])):
                return refuse("END_CHAIN_SINGLE_VALUEDNESS_UNPROVEN")
        chains.append(chain)
    if qualified:
        if any(len({s["native_edge_id"] for s in chain}) != 1 for chain in chains):
            return refuse("QUALIFIED_END_CHAIN_MUST_BE_ONE_NATIVE_EDGE")
        # Original endpoint positions and rails bound the side exclusions. Each
        # monotone native end edge must cover every retained angular coordinate.
        starts = [min(chain[0]["poles"][0][0], chain[-1]["poles"][-1][0]) for chain in chains]
        ends = [max(chain[0]["poles"][0][0], chain[-1]["poles"][-1][0]) for chain in chains]
        u0 = max(u0, *starts) + join_bound
        u1 = min(u1, *ends) - join_bound
        if not u1 > u0:
            return refuse("NO_ANGULAR_CORE_AFTER_NATIVE_JOIN_EXCLUSION")
    ranges = [(min(p[1] for s in c for p in s["poles"]), max(p[1] for s in c for p in s["poles"])) for c in chains]
    low, high = sorted(range(2), key=lambda j: ranges[j][0])
    scale = max(1.0, *(abs(v) for pair in ranges for v in pair), abs(u0), abs(u1))
    arithmetic_guard = 64 * math.ulp(scale)
    if qualified:
        u0 = math.nextafter(u0 + arithmetic_guard, math.inf)
        u1 = math.nextafter(u1 - arithmetic_guard, -math.inf)
        if not u1 > u0:
            return refuse("NO_ANGULAR_CORE_AFTER_NUMERICAL_GUARD")
    gap = ranges[high][0] - ranges[low][1]
    if not math.isfinite(gap) or gap <= 4 * arithmetic_guard:
        return refuse("END_CHAINS_NOT_CERTIFIABLY_SEPARATED")
    return {**receipt, "status": "PROVED", "type_proven": True,
            "proof_kind": "NATIVE_TOLERANCE_QUALIFIED_INTERIOR_CORE" if qualified else "EXACT_CONNECTED_GRAPH_STRIP",
            "full_boundary_geometry_proven": not qualified,
            "native_join_chart_exclusion_mm": join_bound,
            "angular_support_metric_mm": [u0, u1],
            "axial_outer_enclosure_mm_not_filled_rectangle": [ranges[low][0], ranges[high][1]],
            "safe_rectangular_interior_core_mm": [u0, u1, math.nextafter(ranges[low][1] + arithmetic_guard, math.inf),
                                                   math.nextafter(ranges[high][0] - arithmetic_guard, -math.inf)],
            "core_semantics": "SUBSET_OF_CURVED_MATERIAL_NOT_COMPLETE_FOOTPRINT",
            "positive_strip_width_lower_bound_mm": math.nextafter(gap - 2 * arithmetic_guard, -math.inf),
            "numerical_guard_mm": arithmetic_guard,
            "ordered_original_boundary": normalized,
            "isoangular_rail_native_edge_ids": [normalized[left]["native_edge_id"], normalized[right]["native_edge_id"]],
            "span_count": len(normalized), "pole_count": total,
            "connected_material": "NATIVE_VALID_FACE_WITH_CERTIFIED_INTERIOR_NO_GAP_FILL" if qualified else "SINGLE_CONTINUOUS_GRAPH_STRIP_NO_HOLES",
            "pairing_and_tangent_flange_proof": "REQUIRED_SEPARATELY"}
