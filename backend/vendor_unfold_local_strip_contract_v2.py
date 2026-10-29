"""Pure V5 finite-strip diagnostic add-on, never standalone mesh admission.

Checks the new local-strip certificate without relabelling an incumbent V4
certificate. Common body/marker/source validation, native ownership, oracle,
runtime source pin verification and worker supervision remain external gates.
No CAD, files, native modules or catalog are accessed here.
"""
from __future__ import annotations

from fractions import Fraction
import math

from backend.vendor_unfold_contract_v2 import PreviewContractError, canonical_candidates

MODEL = "NATIVE_TESSELLATED_FLANGE_FINITE_DEVELOPED_STRIP_WITH_LOCAL_SEAM_SIDE"
GEOM_TOL = 1e-5
LOCAL_STRIP_SOURCE_SHA256 = "7268fede9a8817b86a18f749d6fb682088a709a5aae4fc0c31b6c00066c14fad"
V4_COMPONENT_SOURCE_SHA256 = "01ff5236499273d28e36a75ebeb6e34c8172de2a0ee1d6b99cfb416de27d59dc"
WORK_CAPS = {"triangles": 1_000_000, "projected_vertices": 3_000_000,
             "clip_edge_tests": 20_000_000, "seam_edge_tests": 3_000_000}
OBLIGATIONS = ("finite_width_clipping_checked", "finite_width_native_seam_span_checked",
               "whole_unchanged_mesh_overlap_checked", "seam_closure_checked",
               "native_surface_ownership_checked", "cross_face_overlap_checked",
               "native_surface_coverage_checked", "excluded_caps_and_hole_walls_checked")
LIMITATIONS = frozenset({"LOCAL_FINITE_STRIP_GUARD_IS_NATIVE_TESSELLATION_ONLY",
    "LOCAL_STRIP_WIDTH_IS_SELECTED_NATIVE_RADIUS_TIMES_SWEEP",
    "LOCAL_NATIVE_SEAM_BOUNDARY_AND_MATERIAL_SIDE_REQUIRED",
    "WHOLE_UNCHANGED_FACE_OVERLAP_REMAINS_REQUIRED",
    "TANGENT_SEAM_MUST_COVER_NATIVE_FINITE_AXIAL_SPAN"})
EXTERNAL_GATES = frozenset({"native_seam_identity_and_same_radius_sweep_strip_width",
    "native_ownership_and_complete_skin_material_coverage",
    "proper_chart_transform_and_native_seam_closure",
    "whole_unchanged_developed_mesh_overlap"})
ROW_KEYS = frozenset({"feature_id", "cylinder_source_face_index", "parent_source_face_index",
                      "child_source_face_index", "parent", "child"})
CERTIFICATE_KEYS = frozenset({"status", "mechanism_version", "model", "side", "geometry_mutated",
    "manufacturing_ready", "exact_binary_clipping", "tolerance_mm", "axial_bounds",
    "developed_strip_width_mm", "certified_strip_intrusion_bounds", "charged_uncertainty_mm",
    "intersecting_axial_triangles", "retained_remote_side_triangles",
    "native_seam_boundary_interval_count", "native_seam_local_material_side_checked",
    "whole_supplied_mesh_finite_strip_checked", "work", "shared_work", "required_external_gates",
    "radius_mm", "sweep_radians"})
_TOL = Fraction.from_float(GEOM_TOL)
_FRAME_EPS = Fraction.from_float(64 * math.ulp(1.0))
_MAX_UNCERTAINTY = math.nextafter(float(9 * _TOL / 64), math.inf)


class LocalStripContractError(PreviewContractError):
    """Inconsistent diagnostic receipt; not a CAD/manufacturing verdict."""


def _require(condition, reason):
    if not condition:
        raise LocalStripContractError(reason)


def _finite(value):
    _require(type(value) in (int, float), "INVALID_LOCAL_STRIP_NUMBER")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise LocalStripContractError("NONFINITE_LOCAL_STRIP_NUMBER") from exc
    _require(math.isfinite(number), "NONFINITE_LOCAL_STRIP_NUMBER")
    return number


def _f(value):
    return Fraction.from_float(value)


def _one_ulp(value, expected):
    return abs(_f(value) - _f(expected)) <= _f(math.ulp(expected))


def _indices(value, count, native_count):
    _require(type(value) is list and len(value) == count
             and all(type(i) is int and 0 <= i < native_count for i in value)
             and len(set(value)) == count, "INVALID_LOCAL_STRIP_SOURCE_INDICES")
    return set(value)


def _work(value, local=False):
    _require(type(value) is dict and set(value) == set(WORK_CAPS)
             and all(type(value[key]) is int and 0 <= value[key] <= cap
                     for key, cap in WORK_CAPS.items()), "INVALID_LOCAL_STRIP_WORK")
    if local:
        t, v, e, s = (value[key] for key in WORK_CAPS)
        # Four sequential clipping planes visit at most 3+4+5+6 polygon
        # edges per input triangle. The first 3 and all seam 3 are mandatory.
        _require(1 <= t <= 200_000 and 3 <= v <= 600_000
                 and 3*t <= e <= 18*t and s == 3*t,
                 "INVALID_LOCAL_STRIP_LOCAL_WORK")
    return value


def _certificate(value, side):
    _require(type(value) is dict and set(value) == CERTIFICATE_KEYS,
             "INVALID_LOCAL_STRIP_CERTIFICATE_SCHEMA")
    _require(value["status"] == "CERTIFIED" and value["model"] == MODEL
             and type(value["mechanism_version"]) is int and value["mechanism_version"] == 1
             and value["side"] == side and value["geometry_mutated"] is False
             and value["manufacturing_ready"] is False and value["exact_binary_clipping"] is True
             and value["native_seam_local_material_side_checked"] is True
             and value["whole_supplied_mesh_finite_strip_checked"] is True,
             "INVALID_LOCAL_STRIP_CERTIFICATE_FLAGS")
    _require(_finite(value["tolerance_mm"]) == GEOM_TOL, "CHANGED_LOCAL_STRIP_TOLERANCE")
    bounds = value["axial_bounds"]
    _require(type(bounds) is list and len(bounds) == 2, "INVALID_LOCAL_STRIP_AXIAL_BOUNDS")
    low, high = map(_finite, bounds)
    _require(_f(high)-_f(low) > 4*_TOL
             and max(math.ulp(low), math.ulp(high)) <= GEOM_TOL/64,
             "INVALID_LOCAL_STRIP_AXIAL_BOUNDS")
    radius, sweep, width = map(_finite, (value["radius_mm"], value["sweep_radians"],
                                       value["developed_strip_width_mm"]))
    _require(GEOM_TOL < radius <= 1_000_000 and GEOM_TOL < sweep <= math.radians(175)+GEOM_TOL
             and _f(width) > 4*_TOL and math.ulp(width) <= GEOM_TOL/64
             and _one_ulp(width, radius*sweep), "LOCAL_STRIP_NATIVE_WIDTH_MISMATCH")
    uncertainty = _finite(value["charged_uncertainty_mm"])
    _require(0 <= uncertainty <= _MAX_UNCERTAINTY, "INVALID_LOCAL_STRIP_UNCERTAINTY")
    intrusion = value["certified_strip_intrusion_bounds"]
    _require(type(intrusion) is list and len(intrusion) == 2,
             "INVALID_LOCAL_STRIP_INTRUSION_BOUNDS")
    lower, upper = map(_finite, intrusion)
    # Helper reports a subset of the exact checked strip, rounding lower up
    # and upper down. Uncertainty itself is rounded up by at most one ULP.
    u = _f(uncertainty)
    _require(0 < lower < upper
             and _f(lower) >= _TOL*(1-_FRAME_EPS)-u
             and _f(lower) <= _TOL-u+_f(math.ulp(uncertainty))+_f(math.ulp(lower))
             and _f(upper) <= _f(width)+_TOL+u
             and _f(upper) >= _f(width)+_TOL+u-_f(math.ulp(uncertainty))-_f(math.ulp(upper)),
             "INVALID_LOCAL_STRIP_INTRUSION_BOUNDS")
    local, cumulative = _work(value["work"], True), _work(value["shared_work"])
    intersecting, remote, intervals = (value[key] for key in ("intersecting_axial_triangles",
        "retained_remote_side_triangles", "native_seam_boundary_interval_count"))
    _require(type(intersecting) is int and 1 <= intersecting <= local["triangles"]
             and type(remote) is int and 0 <= remote <= intersecting
             and type(intervals) is int and 1 <= intervals <= local["seam_edge_tests"],
             "INVALID_LOCAL_STRIP_WITNESS_COUNTS")
    gates = value["required_external_gates"]
    _require(type(gates) is list and len(gates) == len(EXTERNAL_GATES)
             and all(type(gate) is str for gate in gates) and set(gates) == EXTERNAL_GATES,
             "MISSING_LOCAL_STRIP_EXTERNAL_GATES")
    return local, cumulative


def validate_local_strip_diagnostics(payload, candidates):
    """Validate only the local-strip receipt; return None without mutations."""
    try:
        canonical = canonical_candidates(candidates)
    except PreviewContractError as exc:
        raise LocalStripContractError("INVALID_LOCAL_STRIP_CANDIDATES") from exc
    ids = {row["feature_id"] for row in canonical}
    n = len(ids)
    _require(type(payload) is dict and type(payload.get("diagnostics")) is dict,
             "INVALID_LOCAL_STRIP_DIAGNOSTICS")
    diag = payload["diagnostics"]
    _require(type(diag.get("mechanism_version")) is int and diag["mechanism_version"] == 6
             and diag.get("finite_width_model") == MODEL
             and diag.get("local_strip_source_sha256") == LOCAL_STRIP_SOURCE_SHA256
             and diag.get("v4_component_source_sha256") == V4_COMPONENT_SOURCE_SHA256,
             "UNREVIEWED_LOCAL_STRIP_MECHANISM")
    _require(all(diag.get(key) is True for key in OBLIGATIONS), "MISSING_LOCAL_STRIP_OBLIGATION")
    limitations = diag.get("limitations")
    _require(type(limitations) is list and len(limitations) <= 64
             and all(type(item) is str and 0 < len(item) <= 256 for item in limitations)
             and LIMITATIONS.issubset(limitations), "MISSING_LOCAL_STRIP_LIMITATIONS")
    _require(type(diag.get("candidate_count")) is int and diag["candidate_count"] == n,
             "LOCAL_STRIP_CANDIDATE_COUNT_MISMATCH")
    count = diag.get("native_face_count")
    _require(type(count) is int and 2*(2*n+1) <= count <= 1024, "INVALID_LOCAL_STRIP_FACE_COUNT")
    selected = _indices(diag.get("selected_source_face_indices"), 2*n+1, count)
    cylinders = _indices(diag.get("selected_cylinder_face_indices"), n, count)
    planes, root = selected-cylinders, diag.get("root_source_face_index")
    _require(cylinders.issubset(selected) and type(root) is int and root in planes
             and diag.get("selected_skin_rule") == "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
             "INVALID_LOCAL_STRIP_ROOTED_SKIN")
    lines = payload.get("bend_lines")
    _require(type(lines) is list and len(lines) == n
             and all(type(row) is dict and type(row.get("feature_id")) is str for row in lines)
             and {row["feature_id"] for row in lines} == ids, "INVALID_LOCAL_STRIP_BEND_LINES")
    line_sweeps = {row["feature_id"]: _finite(row.get("sweep_deg")) for row in lines}
    proofs = diag.get("finite_width_proofs")
    _require(type(proofs) is list and len(proofs) == n, "INCOMPLETE_LOCAL_STRIP_PROOFS")
    features, used_cylinders, children, events = set(), set(), set(), []
    for row in proofs:
        _require(type(row) is dict and set(row) == ROW_KEYS, "INVALID_LOCAL_STRIP_PROOF_ROW")
        ident = row["feature_id"]
        cylinder, parent, child = (row[key] for key in ("cylinder_source_face_index",
            "parent_source_face_index", "child_source_face_index"))
        _require(type(ident) is str and ident in ids and ident not in features
                 and type(cylinder) is int and cylinder in cylinders and cylinder not in used_cylinders,
                 "INVALID_LOCAL_STRIP_FEATURE_CYLINDER_BIJECTION")
        _require(type(parent) is int and type(child) is int and parent in planes and child in planes
                 and parent != child and child != root and child not in children,
                 "INVALID_LOCAL_STRIP_TREE_EDGE")
        features.add(ident); used_cylinders.add(cylinder); children.add(child)
        a, b = row["parent"], row["child"]
        aw, ashared = _certificate(a, "parent")
        bw, bshared = _certificate(b, "child")
        _require(all(a[key] == b[key] for key in ("axial_bounds", "radius_mm", "sweep_radians",
                                                 "developed_strip_width_mm")),
                 "LOCAL_STRIP_PARENT_CHILD_BINDING_MISMATCH")
        _require(_one_ulp(line_sweeps[ident], math.degrees(a["sweep_radians"])),
                 "LOCAL_STRIP_EXPORTED_SWEEP_MISMATCH")
        events.extend([(ashared["triangles"], ident, "parent", aw, ashared, parent, child),
                       (bshared["triangles"], ident, "child", bw, bshared, parent, child)])
    _require(features == ids and used_cylinders == cylinders and children == planes-{root},
             "INCOMPLETE_LOCAL_STRIP_BIJECTION_OR_TREE")
    global_work = _work(diag.get("finite_width_work"))
    events.sort(key=lambda event: event[0])
    previous = {key: 0 for key in WORK_CAPS}
    developed = {root}
    for index, (_, ident, side, local, shared, parent, child) in enumerate(events):
        _require(all(previous[key]+local[key] == shared[key] for key in WORK_CAPS),
                 "LOCAL_STRIP_SHARED_WORK_CHAIN_MISMATCH")
        if index % 2 == 0:
            _require(side == "parent" and parent in developed and child not in developed,
                     "LOCAL_STRIP_TREE_CHRONOLOGY_MISMATCH")
        else:
            prior = events[index-1]
            _require(side == "child" and ident == prior[1] and parent == prior[5] and child == prior[6],
                     "LOCAL_STRIP_PAIR_CHRONOLOGY_MISMATCH")
            developed.add(child)
        previous = shared
    _require(previous == global_work and developed == planes, "LOCAL_STRIP_FINAL_WORK_OR_TREE_MISMATCH")
    return None
