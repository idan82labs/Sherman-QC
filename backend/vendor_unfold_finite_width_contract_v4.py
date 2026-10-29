"""Typed diagnostic ADD-ON for the reviewed V4 finite-width candidate.

Success is None, NOT AVAILABLE admission. This module deliberately does not
validate body meshes, marker support, common V3 surface/coverage receipts,
source bytes, process isolation, or runtime pins. A controlled worker must also
validate that common contract and verify the actual pinned source bundle.

The legacy/helper diagnostic hashes are declarations to check, not proof of
runtime identity. V4_ENGINE_SHA256 is the reviewed deployment pin; an optional
transport V4 hash must match it, but the frozen pure core does not self-hash.
The future trusted transport must add its independently verified V4 runtime
pin. No native modules, files, CAD, or catalog access occurs here.
"""
from __future__ import annotations

import math
from fractions import Fraction

from backend.vendor_unfold_contract_v2 import canonical_candidates, PreviewContractError


V4_ENGINE_SHA256 = "01ff5236499273d28e36a75ebeb6e34c8172de2a0ee1d6b99cfb416de27d59dc"
LEGACY_ENGINE_SHA256 = "8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07"
FINITE_WIDTH_SHA256 = "11d8cc1b912797a9a14b15b0c64d900566f07eaa536bacd28428142ca55615a2"
MODEL = "NATIVE_TESSELLATION_FINITE_AXIAL_SLAB"
GEOM_TOL = 1e-5
MAX_FACES = 1024
MAX_LOCAL_TRIANGLES = 200_000
MAX_LOCAL_VERTICES = 600_000
WORK_CAPS = {"triangles": 1_000_000, "projected_vertices": 3_000_000,
             "clip_edge_tests": 12_000_000}
EXTERNAL_GATES = frozenset({"native_ownership_and_complete_skin_coverage",
                           "native_seam_identity_and_chart_closure",
                           "whole_unchanged_developed_mesh_overlap"})
LIMITATIONS = frozenset({"FINITE_WIDTH_GUARD_IS_NATIVE_TESSELLATION_ONLY",
                        "WHOLE_UNCHANGED_FACE_OVERLAP_REMAINS_REQUIRED",
                        "TANGENT_SEAM_MUST_COVER_NATIVE_FINITE_AXIAL_SPAN"})
OBLIGATIONS = ("finite_width_clipping_checked", "finite_width_native_seam_span_checked",
               "whole_unchanged_mesh_overlap_checked", "seam_closure_checked",
               "native_surface_ownership_checked", "cross_face_overlap_checked",
               "native_surface_coverage_checked", "excluded_caps_and_hole_walls_checked")
CERTIFICATE_KEYS = frozenset({"status", "model", "side", "geometry_mutated",
    "manufacturing_ready", "exact_binary_clipping", "tolerance_mm", "axial_bounds",
    "expanded_axial_bounds", "intersecting_triangles", "max_oriented_side_distance_mm",
    "charged_uncertainty_mm", "work", "shared_work", "required_external_gates"})
ROW_KEYS = frozenset({"feature_id", "cylinder_source_face_index", "parent_source_face_index",
                      "child_source_face_index", "parent", "child"})
_TOL = Fraction.from_float(GEOM_TOL)
_FRAME_EPS = 64*math.ulp(1.0)
# The helper charges at most tol/8 seam-anchor + tol/64 frame uncertainty.
_MAX_REPORTED_UNCERTAINTY = math.nextafter(
    (GEOM_TOL/8+GEOM_TOL/64)/(1-_FRAME_EPS), math.inf)


class FiniteWidthContractError(PreviewContractError):
    """Malformed or inconsistent V4 diagnostic ledger, not a CAD diagnosis."""


def _require(value, reason):
    if not value:
        raise FiniteWidthContractError(reason)


def _finite(value):
    _require(type(value) in (int, float), "INVALID_FINITE_WIDTH_DIAGNOSTIC_NUMBER")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise FiniteWidthContractError("NONFINITE_FINITE_WIDTH_DIAGNOSTIC_NUMBER") from exc
    _require(math.isfinite(number), "NONFINITE_FINITE_WIDTH_DIAGNOSTIC_NUMBER")
    return number


def _indices(value, count, native_count):
    _require(type(value) is list and len(value) == count
             and all(type(i) is int and 0 <= i < native_count for i in value),
             "INVALID_FINITE_WIDTH_SOURCE_INDICES")
    _require(len(set(value)) == count, "DUPLICATE_FINITE_WIDTH_SOURCE_INDEX")
    return set(value)


def _work(value, local=False):
    _require(type(value) is dict and set(value) == set(WORK_CAPS)
             and all(type(value[key]) is int and 0 <= value[key] <= limit
                     for key, limit in WORK_CAPS.items()), "INVALID_FINITE_WIDTH_WORK")
    if local:
        t, v, e = (value[key] for key in ("triangles", "projected_vertices", "clip_edge_tests"))
        # Each source triangle tests 3 edges at the first plane and at most 4
        # edges after that clipping. No sampling/optional skipped work exists.
        _require(1 <= t <= MAX_LOCAL_TRIANGLES and 3 <= v <= MAX_LOCAL_VERTICES
                 and 3*t <= e <= 7*t, "INVALID_FINITE_WIDTH_LOCAL_WORK")
    return value


def _certificate(value, side):
    _require(type(value) is dict and set(value) == CERTIFICATE_KEYS,
             "INVALID_FINITE_WIDTH_CERTIFICATE_SCHEMA")
    _require(type(value["status"]) is str and value["status"] == "CERTIFIED"
             and type(value["model"]) is str and value["model"] == MODEL
             and type(value["side"]) is str and value["side"] == side and value["geometry_mutated"] is False
             and value["manufacturing_ready"] is False and value["exact_binary_clipping"] is True,
             "INVALID_FINITE_WIDTH_CERTIFICATE_FLAGS")
    _require(_finite(value["tolerance_mm"]) == GEOM_TOL, "CHANGED_FINITE_WIDTH_TOLERANCE")
    bounds, expanded = value["axial_bounds"], value["expanded_axial_bounds"]
    _require(type(bounds) is list and len(bounds) == 2 and type(expanded) is list
             and len(expanded) == 2, "INVALID_FINITE_WIDTH_AXIAL_SPAN")
    low, high = map(_finite, bounds)
    outer_low, outer_high = map(_finite, expanded)
    _require(Fraction.from_float(high)-Fraction.from_float(low) > 4*_TOL
             and max(math.ulp(low), math.ulp(high)) <= GEOM_TOL/64
             and outer_low == low-GEOM_TOL and outer_high == high+GEOM_TOL,
             "INVALID_FINITE_WIDTH_AXIAL_SPAN")
    maximum = _finite(value["max_oriented_side_distance_mm"])
    uncertainty = _finite(value["charged_uncertainty_mm"])
    _require(0 <= uncertainty <= _MAX_REPORTED_UNCERTAINTY and maximum <= GEOM_TOL,
             "INVALID_FINITE_WIDTH_SIDE_BOUND")
    # Helper report fields are each conservatively rounded UP from Fractions.
    # Their sum can exceed the unchanged exact tol by those two reporting ULPs.
    # This is a diagnostic serialization allowance, NEVER an input-geometry
    # tolerance or permission to bypass the helper's exact rational decision.
    reported = Fraction.from_float(maximum)+Fraction.from_float(uncertainty)
    reporting_ulps = Fraction.from_float(math.ulp(maximum))+Fraction.from_float(math.ulp(uncertainty))
    _require(reported <= _TOL+reporting_ulps, "INVALID_FINITE_WIDTH_SIDE_BOUND")
    local, cumulative = _work(value["work"], True), _work(value["shared_work"])
    intersecting = value["intersecting_triangles"]
    _require(type(intersecting) is int and 1 <= intersecting <= local["triangles"],
             "INVALID_FINITE_WIDTH_INTERSECTION_COUNT")
    gates = value["required_external_gates"]
    _require(type(gates) is list and len(gates) == len(EXTERNAL_GATES)
             and all(type(gate) is str for gate in gates) and set(gates) == EXTERNAL_GATES,
             "MISSING_FINITE_WIDTH_EXTERNAL_GATES")
    return local, cumulative


def validate_finite_width_diagnostics(payload, candidates):
    """Check only the V4 diagnostic add-on; return None without mutating input.

    Must be composed with common preview/mesh/marker/source/transport admission.
    Passing this function alone does not make ANY geometry AVAILABLE or safe.
    """
    try:
        rows = canonical_candidates(candidates)
    except PreviewContractError as exc:
        raise FiniteWidthContractError("INVALID_FINITE_WIDTH_CANDIDATES") from exc
    expected_ids = {row["feature_id"] for row in rows}
    n = len(rows)
    _require(type(payload) is dict and type(payload.get("diagnostics")) is dict,
             "INVALID_FINITE_WIDTH_DIAGNOSTICS")
    diag = payload["diagnostics"]
    _require(type(diag.get("mechanism_version")) is int and diag["mechanism_version"] == 4
             and type(diag.get("finite_width_model")) is str and diag["finite_width_model"] == MODEL
             and type(diag.get("legacy_engine_source_sha256")) is str
             and diag["legacy_engine_source_sha256"] == LEGACY_ENGINE_SHA256
             and type(diag.get("finite_width_source_sha256")) is str
             and diag["finite_width_source_sha256"] == FINITE_WIDTH_SHA256,
             "UNREVIEWED_FINITE_WIDTH_MECHANISM")
    if "v4_engine_source_sha256" in diag:
        _require(type(diag["v4_engine_source_sha256"]) is str
                 and diag["v4_engine_source_sha256"] == V4_ENGINE_SHA256,
                 "UNREVIEWED_V4_ENGINE_DECLARATION")
    _require(all(diag.get(key) is True for key in OBLIGATIONS), "MISSING_FINITE_WIDTH_OBLIGATION")
    limitations = diag.get("limitations")
    _require(type(limitations) is list and len(limitations) <= 64
             and all(type(item) is str and 0 < len(item) <= 256 for item in limitations)
             and LIMITATIONS.issubset(limitations), "MISSING_FINITE_WIDTH_LIMITATIONS")
    _require(type(diag.get("candidate_count")) is int and diag["candidate_count"] == n,
             "FINITE_WIDTH_CANDIDATE_COUNT_MISMATCH")
    native_count = diag.get("native_face_count")
    _require(type(native_count) is int and 2*(2*n+1) <= native_count <= MAX_FACES,
             "INVALID_FINITE_WIDTH_NATIVE_FACE_COUNT")
    selected = _indices(diag.get("selected_source_face_indices"), 2*n+1, native_count)
    cylinders = _indices(diag.get("selected_cylinder_face_indices"), n, native_count)
    planes = selected-cylinders
    root = diag.get("root_source_face_index")
    _require(cylinders.issubset(selected) and type(root) is int and root in planes
             and type(diag.get("selected_skin_rule")) is str
             and diag["selected_skin_rule"] == "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
             "INVALID_FINITE_WIDTH_ROOTED_SKIN")
    proofs = diag.get("finite_width_proofs")
    _require(type(proofs) is list and len(proofs) == n, "INCOMPLETE_FINITE_WIDTH_PROOFS")
    features, used_cylinders, used_children = set(), set(), set()
    events = []
    for proof in proofs:
        _require(type(proof) is dict and set(proof) == ROW_KEYS, "INVALID_FINITE_WIDTH_PROOF_ROW")
        ident = proof["feature_id"]
        cylinder, parent, child = (proof[key] for key in
            ("cylinder_source_face_index", "parent_source_face_index", "child_source_face_index"))
        _require(type(ident) is str and ident in expected_ids and ident not in features
                 and type(cylinder) is int and cylinder in cylinders and cylinder not in used_cylinders,
                 "INVALID_FINITE_WIDTH_FEATURE_CYLINDER_BIJECTION")
        _require(type(parent) is int and type(child) is int and parent in planes and child in planes
                 and parent != child and child != root and child not in used_children,
                 "INVALID_FINITE_WIDTH_TREE_EDGE")
        features.add(ident); used_cylinders.add(cylinder); used_children.add(child)
        parent_work, parent_shared = _certificate(proof["parent"], "parent")
        child_work, child_shared = _certificate(proof["child"], "child")
        _require(proof["parent"]["axial_bounds"] == proof["child"]["axial_bounds"],
                 "FINITE_WIDTH_SIDE_SPAN_MISMATCH")
        events.extend([(parent_shared["triangles"], ident, "parent", parent_work, parent_shared, parent, child),
                       (child_shared["triangles"], ident, "child", child_work, child_shared, parent, child)])
    _require(features == expected_ids and used_cylinders == cylinders and used_children == planes-{root},
             "INCOMPLETE_FINITE_WIDTH_BIJECTION_OR_TREE")
    global_work = _work(diag.get("finite_width_work"))
    # Source rows are feature-sorted; only these cumulative positive triangle
    # counters retain chronological order. Reconstruct without resetting budget.
    events.sort(key=lambda event: event[0])
    previous = {key: 0 for key in WORK_CAPS}
    developed = {root}
    for index, (_, ident, side, local, cumulative, parent, child) in enumerate(events):
        _require(all(previous[key]+local[key] == cumulative[key] for key in WORK_CAPS),
                 "FINITE_WIDTH_SHARED_WORK_CHAIN_MISMATCH")
        if index % 2 == 0:
            _require(side == "parent" and parent in developed and child not in developed,
                     "FINITE_WIDTH_TREE_CHRONOLOGY_MISMATCH")
        else:
            prior = events[index-1]
            _require(side == "child" and ident == prior[1] and parent == prior[5] and child == prior[6],
                     "FINITE_WIDTH_PAIR_CHRONOLOGY_MISMATCH")
            developed.add(child)
        previous = cumulative
    _require(previous == global_work and developed == planes,
             "FINITE_WIDTH_FINAL_WORK_OR_TREE_MISMATCH")
    return None
