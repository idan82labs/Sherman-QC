"""Explicit mechanism5 common mesh/source/marker admission.

Exact incumbent common validator body is copied, except its supported-version
set adds 5. All numeric geometry checks and helper implementations are retained
from source-pinned V2. No diagnostic view is relabelled and no finite-strip or
native oracle proof is supplied here. Those mandatory gates are composed by the
public transport4 contract and the controlled worker.
"""
import math
import struct
from backend.vendor_unfold_contract_v2 import PLANAR_TOL
import numpy as np
from backend.vendor_unfold_contract_v2 import (ANGULAR_TOL_RADIANS, MAX_CANDIDATES, MAX_TRIANGLES, MAX_VERTICES, MODEL, REQUIRED_PROOFS, _mesh, _number, _point, _require, canonical_candidates, re)

def _validate_preview(payload, expected_hash, candidates, *, mechanism_version):
    """Common geometry boundary, not standalone V4 admission.

    The public V2 entry point below stays strictly V3. The V4 transport must
    additionally compose the separately audited finite-width proof contract;
    this private seam avoids relabelling a receipt or duplicating mesh checks.
    """
    _require(type(mechanism_version) is int and mechanism_version in (3, 4, 5),
             "UNSUPPORTED_COMMON_PREVIEW_MECHANISM")
    _require(type(expected_hash) is str and re.fullmatch(r"[0-9a-f]{64}", expected_hash), "INVALID_SOURCE_HASH")
    rows = canonical_candidates(candidates)
    envelope_keys = {"status", "reason", "model", "units", "units_verified", "manufacturing_ready",
                     "source_sha256", "vertices", "triangles", "bend_lines", "flange_surfaces", "diagnostics"}
    _require(type(payload) is dict and set(payload) == envelope_keys
             and type(payload.get("status")) is str and payload.get("status") in {"AVAILABLE", "UNAVAILABLE"}
             and payload.get("model") == MODEL and payload.get("units") == "assumed_mm"
             and payload.get("units_verified") is False and payload.get("manufacturing_ready") is False,
             "INVALID_PREVIEW_ENVELOPE")
    # Extra nested approval fields may not contradict the envelope.
    def claims(value, depth=0):
        _require(depth <= 16, "RECEIPT_DEPTH_BUDGET_EXCEEDED")
        if type(value) is dict:
            for key, item in value.items():
                if key in {"manufacturing_ready", "release_allowed", "cut_release_allowed", "physical_calibration_verified", "units_verified"}:
                    _require(item is False, "CONTRADICTORY_RELEASE_CLAIM")
                claims(item, depth + 1)
        elif type(value) is list:
            # Geometry is numeric and already bounded separately. Avoid recursion
            # through millions of scalars while still checking nested envelopes.
            for item in value:
                if type(item) in (dict, list):
                    claims(item, depth + 1)
    claims(payload)
    _require(type(payload.get("diagnostics")) is dict, "INVALID_PREVIEW_DIAGNOSTICS")
    if payload["status"] == "UNAVAILABLE":
        _require(type(payload.get("reason")) is str and 0 < len(payload["reason"]) <= 200
                 and payload.get("source_sha256") in (None, expected_hash)
                 and all(payload.get(key) == [] for key in ("vertices", "triangles", "bend_lines", "flange_surfaces")),
                 "INVALID_UNAVAILABLE_PREVIEW")
        return payload
    _require(payload.get("reason") is None and payload.get("source_sha256") == expected_hash,
             "PREVIEW_SOURCE_MISMATCH")
    diag = payload["diagnostics"]
    if "candidate_binding" in diag:
        _require(canonical_candidates(diag["candidate_binding"]) == rows, "PREVIEW_CANDIDATE_BINDING_MISMATCH")
    _require(all(diag.get(key) is True for key in REQUIRED_PROOFS)
             and type(diag.get("mechanism_version")) is int and diag.get("mechanism_version") == mechanism_version
             and type(diag.get("candidate_count")) is int and diag.get("candidate_count") == len(rows)
             and diag.get("selected_skin_rule") == "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN"
             and diag.get("direction_status") == "UNKNOWN_NATIVE_NORMAL_SIGN_NOT_AUTHORITY"
             and type(diag.get("limitations")) is list
             and all(type(item) is str for item in diag["limitations"])
             and {"VISUAL_ONLY_SELECTED_CAD_SKIN", "NOT_NEUTRAL_AXIS_OR_K_FACTOR_BLANK",
                  "UNITS_ASSUMED_NOT_VERIFIED"}.issubset(diag["limitations"]),
             "MISSING_NATIVE_COVERAGE_PROOF")
    count = diag.get("native_face_count")
    selected, cylinders, root = (diag.get("selected_source_face_indices"),
        diag.get("selected_cylinder_face_indices"), diag.get("root_source_face_index"))
    _require(type(count) is int and 2*(2*len(rows)+1) <= count <= 1024 and type(selected) is list
             and all(type(i) is int and 0 <= i < count for i in selected)
             and len(selected) == len(set(selected)) and len(selected) == 2 * len(rows) + 1
             and type(cylinders) is list and len(cylinders) == len(rows)
             and all(type(i) is int and i in selected for i in cylinders)
             and len(cylinders) == len(set(cylinders))
             and type(root) is int and root in selected and root not in cylinders,
             "INVALID_NATIVE_FACE_RECEIPT")
    body_count = _mesh(payload.get("vertices"), payload.get("triangles"))
    _require(body_count >= len(selected), "INCOMPLETE_NATIVE_MESH_RECEIPT")
    lines, surfaces = payload.get("bend_lines"), payload.get("flange_surfaces")
    ids = {row["feature_id"] for row in rows}
    _require(type(lines) is list and len(lines) == len(ids) and type(surfaces) is list
             and len(surfaces) == len(ids), "INCOMPLETE_BEND_PREVIEW")
    for collection in (lines, surfaces):
        _require(all(type(row) is dict and type(row.get("feature_id")) is str for row in collection)
                 and {row["feature_id"] for row in collection} == ids, "INVALID_PREVIEW_BEND_IDS")
    for row in lines:
        a, b, sweep = row.get("line_start"), row.get("line_end"), row.get("sweep_deg")
        _require(_point(a, True) and _point(b, True) and math.dist(a, b) > 1e-9
                 and _number(sweep) and 0 < sweep <= 175 + math.degrees(ANGULAR_TOL_RADIANS)
                 and row.get("bend_direction") == "unknown", "INVALID_FLAT_BEND_LINE")
    triangles = sum(_mesh(row.get("vertices"), row.get("triangles")) for row in surfaces)
    _require(triangles <= MAX_TRIANGLES, "FLANGE_MESH_BUDGET_EXCEEDED")
    _require(len(payload["vertices"]) + sum(len(row["vertices"]) for row in surfaces) <= MAX_VERTICES * 6,
             "TOTAL_VERTEX_BUDGET_EXCEEDED")
    # A flange highlight is a subset of the actual developed triangulation,
    # never a independently filled polygon (which could cover a retained hole).
    def triangle_keys(vertices, indices):
        return {tuple(sorted((vertices[3*i], vertices[3*i+1]) for i in indices[j:j+3]))
                for j in range(0, len(indices), 3)}
    body = triangle_keys(payload["vertices"], payload["triangles"])
    _require(len(body) == body_count, "DUPLICATE_BODY_TRIANGLE")
    seen = set()
    for surface in surfaces:
        keys = triangle_keys(surface["vertices"], surface["triangles"])
        _require(keys.issubset(body) and not seen.intersection(keys)
                 and len(keys) == len(surface["triangles"])//3 and len(keys) < len(body),
                 "FLANGE_NOT_BODY_TRIANGLE_SUBSET")
        seen.update(keys)
    _require(len(body)-len(seen) >= len(cylinders)+1, "FLANGE_OVERLAY_OMITS_ROOT_AND_BEND_BODY")
    remaining = 2 * MAX_CANDIDATES * MAX_TRIANGLES
    def supported(start, end, chart):
        nonlocal remaining
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = math.hypot(dx, dy)
        _require(length > 1e-9, "INVALID_FLAT_BEND_LINE")
        intervals = []
        for triangle in chart:
            remaining -= 1
            _require(remaining >= 0, "MARKER_SUPPORT_WORK_BUDGET_EXCEEDED")
            a, b, c = triangle
            if (max(start[0],end[0]) < min(p[0] for p in triangle)
                    or min(start[0],end[0]) > max(p[0] for p in triangle)
                    or max(start[1],end[1]) < min(p[1] for p in triangle)
                    or min(start[1],end[1]) > max(p[1] for p in triangle)):
                continue
            area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            sign = math.copysign(1., area)
            low, high = 0., 1.
            for p, q in ((a,b),(b,c),(c,a)):
                ex, ey = q[0]-p[0], q[1]-p[1]
                at_start = sign*(ex*(start[1]-p[1])-ey*(start[0]-p[0]))
                slope = sign*(ex*dy-ey*dx)
                if slope == 0:
                    if at_start < 0:
                        low, high = 1., 0.
                elif slope > 0:
                    low = max(low, -at_start/slope)
                else:
                    high = min(high, -at_start/slope)
                if low > high:
                    break
            if low <= high:
                intervals.append((low, high))
        # A numerical gap allowance cannot turn no material (or an isolated
        # point contact) into a supported short marker. Match the rendering
        # boundary's positive-interval requirement before spending that budget.
        if not any(high > low for low, high in intervals):
            return False
        covered_end = uncovered = 0.
        for low, high in sorted(intervals):
            uncovered += max(0., low-covered_end)
            covered_end = max(covered_end, high)
        uncovered += max(0., 1.-covered_end)
        # One physical-length budget for the union, not a tolerance per gap.
        return math.isfinite(uncovered) and uncovered*length <= PLANAR_TOL
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]
    float_body = {tuple(sorted((f32(x), f32(y)) for x,y in triangle)) for triangle in body}
    _require(len(float_body) == len(body), "FLOAT32_DUPLICATE_BODY_TRIANGLE")
    _require(all(supported(line["line_start"], line["line_end"], body)
                 and supported([f32(v) for v in line["line_start"]],
                               [f32(v) for v in line["line_end"]], float_body) for line in lines),
             "FLAT_BEND_LINE_OUTSIDE_BODY")
    return payload
