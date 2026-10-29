"""Bounded plane-only starts with finite assigned-face screening, additive V4.

Both public entrypoints derive patches and hypotheses from owned array snapshots.
No supplied plane/support dictionary, pose, bend value or identity enters them.
Screening is a NONEXHAUSTIVE initialization heuristic: a rejected start could
have refined successfully. Assigned faces are initialization provenance, never
final matching, acquisition-side identity, observed count or conformance.

The registration arm invokes frozen V2 coarse/fine refinement and diagnostics.
It intentionally does not add PCA fallback, V3 updates, approximate start
clustering, threshold changes or a best-prefix escape from a resource limit.
External process-tree time/RSS supervision remains required.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json

import numpy as np

from . import cad_guided_global_registration_v1 as _v1
from . import cad_guided_global_registration_v2 as _v2
from . import cad_guided_mesh_plane_proposals_v4 as _patch
from . import cad_guided_plane_hypotheses_v4 as _planes
from . import cad_guided_triangle_surface_v2 as _surface


_LIMITS = dict(max_hypothesis_records=4096, max_cached_cad_faces=24,
    max_indexed_face_triangles=500_000, max_prefilter_query_calls=32768,
    max_prefilter_query_rows=32_000_000, chunk_rows=65536,
    max_node_visits=512_000_000, max_triangle_pairs=256_000_000,
    max_normal_pairs=32_000_000, max_work_units=4_000_000_000,
    max_face_build_nodes=131071, max_face_build_triangle_visits=10_000_000,
    max_surviving_start_groups=24, max_report_bytes=16_777_216)


@dataclass(frozen=True)
class PlaneInitializationBudgetsV4:
    max_hypothesis_records: int = 4096
    max_cached_cad_faces: int = 24
    max_indexed_face_triangles: int = 500_000
    max_prefilter_query_calls: int = 32768
    max_prefilter_query_rows: int = 32_000_000
    chunk_rows: int = 65536
    max_node_visits: int = 512_000_000
    max_triangle_pairs: int = 256_000_000
    max_normal_pairs: int = 32_000_000
    max_work_units: int = 4_000_000_000
    max_face_build_nodes: int = 131071
    max_face_build_triangle_visits: int = 10_000_000
    max_surviving_start_groups: int = 24
    max_report_bytes: int = 16_777_216

    def __post_init__(self):
        _budget_snapshot(self)


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unavailable(reason)


def _budget_snapshot(value):
    _require(type(value) is PlaneInitializationBudgetsV4, "INVALID_INITIALIZATION_BUDGET_TYPE")
    result = {key: getattr(value, key) for key in _LIMITS}
    _require(all(type(item) is int and 0 <= item <= _LIMITS[key]
                 for key, item in result.items()) and result["chunk_rows"] > 0,
             "INVALID_INITIALIZATION_BUDGET")
    return result


def _registration_policy(value):
    _require(type(value) is _v1.WholePartRegistrationPolicyV1,
             "INVALID_WHOLE_PART_REGISTRATION_POLICY")
    # Reconstruct, not merely isinstance: frozen instances can be bypass-mutated.
    values = {field.name: getattr(value, field.name) for field in fields(value)}
    _require(all(type(item) in (int, float) for item in values.values()),
             "INVALID_WHOLE_PART_REGISTRATION_POLICY")
    return _v1.WholePartRegistrationPolicyV1(**values)


def _configuration(registration_policy, patch_policy, patch_budgets,
                   hypothesis_policy, hypothesis_budgets, budgets, surface_budgets):
    cap = _budget_snapshot(budgets)
    policy = _registration_policy(registration_policy)
    patch_gate, patch_cap = _patch._configuration(patch_policy, patch_budgets)
    plane_gate = _planes._policy_snapshot(hypothesis_policy)
    plane_cap = _planes._budget_snapshot(hypothesis_budgets)
    surface = _surface._budgets(surface_budgets)
    # Own every policy/budget snapshot before any geometry/extraction/query work.
    return dict(policy=policy, budgets=PlaneInitializationBudgetsV4(**cap),
        patch_policy=_patch.CoplanarPatchPolicyV4(**patch_gate),
        patch_budgets=_patch.CoplanarPatchBudgetsV4(**patch_cap),
        hypothesis_policy=_planes.PlaneHypothesisPolicyV4(**plane_gate),
        hypothesis_budgets=_planes.PlaneHypothesisBudgetsV4(**plane_cap),
        surface_budgets=_surface.TriangleSurfaceBudgets(**asdict(surface)))


def _array(value, tail, kinds, maximum, dtype, name):
    _require(type(value) is np.ndarray and value.ndim == len(tail)+1
             and value.shape[1:] == tail and value.dtype.kind in kinds
             and value.dtype.itemsize <= 8 and 1 <= len(value) <= maximum,
             "INVALID_OR_OVER_CAP_"+name)
    if dtype == "<i8":
        _require(np.all(value >= 0) and (value.dtype.kind == "i"
                 or np.all(value <= np.uint64(2**63-1))), "INVALID_"+name)
    else:
        _require(np.isfinite(value).all() and np.all(value >= -1e6)
                 and np.all(value <= 1e6), "INVALID_"+name)
    snapshot = _patch._immutable(value, dtype)
    return snapshot


def _digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _base():
    return dict(schema_version="plane_initialization_prefilter_v4", status="UNAVAILABLE",
        reasons=[], search_complete=False, accepted_registration=False,
        registration_status="NOT_EVALUATED", production_use_allowed=False,
        observed_bend_count=None, conformance="INDETERMINATE",
        unit_status="PROVISIONAL_ARRAY_MM_UNVERIFIED",
        search_scope="PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_NONEXHAUSTIVE",
        rejected_start_can_be_recoverable=True, noisy_plane_completeness=False,
        assignment_scope="INITIALIZATION_ONLY_NOT_FINAL_FACE_MATCH",
        support_basis="ALL_ORIGINAL_UNIQUE_RAW_IDS_PER_SCAN_PATCH_NO_COORDINATE_DEDUP_OR_TRIM",
        lineage_scope="RECOMPUTED_FROM_OWNED_ARRAYS_NOT_NATIVE_OR_FILE_AUTHENTICATION",
        patch_evidence={}, hypothesis_evidence={}, input_digests={},
        hypotheses=[], start_groups=[], face_index_evidence=[],
        statistics=dict(hypothesis_count=0, hypotheses_evaluated=0,
            passing_hypothesis_count=0, rejected_hypothesis_count=0,
            surviving_start_group_count=0, planned_query_rows=0, planned_query_calls=0,
            indexed_face_count=0, indexed_face_triangles=0,
            face_build_nodes=0, face_build_triangle_visits=0,
            query_calls=0, query_rows=0, node_visits=0, triangle_pairs=0,
            normal_pairs=0, work_units=0))


def _compact_patches(payload):
    # Complete component/patch metadata is retained. Potentially millions of raw
    # indices remain bound by their little-endian digests and exact row counts;
    # they are NOT claimed to be embedded or externally authenticated here.
    result = {key: value for key, value in payload.items()
              if key not in ("patches", "triangle_component_labels")}
    result["row_map_representation"] = "INTERNAL_EXACT_ARRAYS_OUTPUT_COUNTS_AND_SHA256_ONLY"
    result["patches"] = []
    for patch in payload["patches"]:
        row = {}
        for key, value in patch.items():
            if isinstance(value, np.ndarray):
                row[key+"_count"] = len(value)
            else:
                row[key] = value
        result["patches"].append(row)
    return result


def _prepare(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
             cad_source_triangle_ids, configuration, result):
    patch_cap = configuration["patch_budgets"]
    budget = configuration["budgets"]
    # Shape/count/index admission precedes copies, triangle expansion and fitting.
    scan = _array(scan_points, (3,), "fiu", patch_cap.max_vertices, "<f8", "SCAN_POINTS")
    raw = _array(scan_triangle_raw_rows, (3,), "iu", patch_cap.max_triangles, "<i8", "SCAN_TRIANGLES")
    _require(np.all(raw < len(scan)), "SCAN_TRIANGLE_RAW_ROW_OUT_OF_RANGE")
    _patch._input_caps(len(raw), asdict(patch_cap), scan=True)
    cad = _array(cad_triangles, (3, 3), "fiu",
        min(patch_cap.max_triangles, budget.max_indexed_face_triangles), "<f8", "CAD_TRIANGLES")
    faces = _array(cad_face_ids, (), "iu", len(cad), "<i8", "CAD_FACE_IDS")
    local = _array(cad_source_triangle_ids, (), "iu", len(cad), "<i8", "CAD_SOURCE_TRIANGLE_IDS")
    _require(len(faces) == len(cad) == len(local) and np.all(faces <= 2**53), "INVALID_CAD_SOURCE_BINDING")
    _patch._input_caps(len(cad), asdict(patch_cap), scan=False)
    result["input_digests"] = dict(scan_points_sha256=_digest(scan),
        scan_triangle_raw_rows_sha256=_digest(raw), cad_triangles_sha256=_digest(cad),
        cad_face_ids_sha256=_digest(faces), cad_source_triangle_ids_sha256=_digest(local))
    result["input_counts"] = dict(scan_raw_rows=len(scan), scan_triangles=len(raw), cad_triangles=len(cad))
    scan_patches = _patch.extract_coplanar_scan_patches_v4(scan, raw,
        policy=configuration["patch_policy"], budgets=patch_cap)
    result["patch_evidence"]["scan"] = _compact_patches(scan_patches)
    _require(scan_patches["status"] == "AVAILABLE", "SCAN_PLANE_PROPOSALS_UNAVAILABLE")
    cad_patches = _patch.fit_cad_face_planes_v4(cad, source_face_indices=faces,
        source_triangle_indices=local, policy=configuration["patch_policy"], budgets=patch_cap)
    result["patch_evidence"]["cad"] = _compact_patches(cad_patches)
    _require(cad_patches["status"] == "AVAILABLE", "CAD_PLANE_PROPOSALS_UNAVAILABLE")
    observed = tuple(_planes.ObservedPlaneV4(
        _planes.PlaneEquationV4(row["plane_id"], tuple(row["normal"]), row["offset_mm"]),
        row["support_count"], row["support_rms_mm"], tuple(row["support_extent_mm"]))
        for row in scan_patches["patches"])
    nominal = tuple(_planes.PlaneEquationV4(row["plane_id"], tuple(row["normal"]), row["offset_mm"])
                    for row in cad_patches["patches"])
    hypotheses = _planes.propose_plane_hypotheses_v4(observed, nominal,
        policy=configuration["hypothesis_policy"], budgets=configuration["hypothesis_budgets"])
    result["hypothesis_evidence"] = {key: value for key, value in hypotheses.items() if key != "hypotheses"}
    _require(hypotheses["status"] == "COMPLETE", "PLANE_HYPOTHESES_UNAVAILABLE")
    # Private context is constructed only here. Public APIs accept no dictionary
    # claiming patch provenance; mock contexts prove only the narrower seam.
    return dict(scan=scan, raw_triangles=raw, cad=cad, faces=faces, local=local,
                scan_patches=scan_patches, cad_patches=cad_patches,
                hypotheses=hypotheses["hypotheses"])


def _pose(hypothesis):
    rotation = np.asarray(hypothesis["rotation"], dtype="<f8")
    translation = np.asarray(hypothesis["translation_mm"], dtype="<f8")
    _require(rotation.shape == (3, 3) and translation.shape == (3,)
        and np.isfinite(rotation).all() and np.isfinite(translation).all()
        and np.max(np.abs(rotation.T@rotation-np.eye(3))) <= 1e-8
        and abs(float(np.linalg.det(rotation))-1.) <= 1e-8, "INVALID_INITIALIZATION_POSE")
    return rotation, translation


def _bound_support(context, scan_patch=None, cad_patch=None):
    # Defense in depth at the private seam: array digests alone do not prove
    # ownership. Recompute complete scan raw union and complete CAD face union.
    if scan_patch is not None:
        rows = scan_patch["triangle_rows"]
        raw = scan_patch["raw_vertex_rows"]
        _require(type(rows) is np.ndarray and rows.dtype.kind == "i" and rows.ndim == 1
            and len(rows) > 0 and np.all(rows >= 0) and np.all(rows < len(context["raw_triangles"])),
            "INVALID_SCAN_PATCH_TRIANGLE_BINDING")
        component = scan_patch["component_id"]
        expected_triangles = np.flatnonzero(context["scan_patches"]["triangle_component_labels"] == component)
        _require(scan_patch["plane_id"] == component and np.array_equal(np.sort(rows), expected_triangles)
            and np.array_equal(raw, np.unique(context["raw_triangles"][rows]))
            and scan_patch["support_count"] == len(raw)
            and _digest(rows) == scan_patch["triangle_rows_sha256"]
            and _digest(raw) == scan_patch["raw_vertex_rows_sha256"], "STALE_OR_INCOMPLETE_SCAN_PATCH_SUPPORT")
    if cad_patch is None:
        return
    face = cad_patch["plane_id"]
    cad_rows = cad_patch["triangle_rows"]
    expected_cad = np.flatnonzero(context["faces"] == face)
    _require(type(cad_rows) is np.ndarray and cad_rows.dtype.kind == "i" and cad_rows.ndim == 1
        and np.all(cad_rows >= 0) and np.all(cad_rows < len(context["cad"])), "INVALID_CAD_PATCH_TRIANGLE_BINDING")
    _require(cad_patch["source_face_id"] == face and np.array_equal(np.sort(cad_rows), expected_cad)
        and len(cad_rows) > 0 and _digest(cad_rows) == cad_patch["triangle_rows_sha256"]
        and np.array_equal(cad_patch["source_triangle_ids"], context["local"][cad_rows])
        and _digest(cad_patch["source_triangle_ids"]) == cad_patch["source_triangle_ids_sha256"],
        "STALE_OR_INCOMPLETE_CAD_FACE_SUPPORT")


class _FaceQueries:
    def __init__(self, context, face_map, budget, surface_budget, result):
        self.indices = {}
        self.budget, self.surface_budget = budget, surface_budget
        self.statistics = result["statistics"]
        _require(len(face_map) <= budget.max_cached_cad_faces, "CAD_FACE_CACHE_BUDGET_EXCEEDED")
        planned = sum(len(patch["triangle_rows"]) for patch in face_map.values())
        _require(planned <= budget.max_indexed_face_triangles, "CAD_FACE_TRIANGLE_BUDGET_EXCEEDED")
        for face, patch in sorted(face_map.items()):
            values = asdict(surface_budget)
            for name in ("nodes", "triangle_visits"):
                key = "build_"+name
                values["max_"+key] = min(values["max_"+key],
                    getattr(budget, "max_face_"+key)-self.statistics["face_"+key])
            rows = patch["triangle_rows"]
            built = _surface.build_triangle_surface(context["cad"][rows], context["faces"][rows],
                budgets=_surface.TriangleSurfaceBudgets(**values))
            result["face_index_evidence"].append(dict(cad_plane_id=face,
                triangle_rows_sha256=patch["triangle_rows_sha256"], triangle_count=len(rows),
                result={key: value for key, value in built.items() if key != "index"}))
            for name in ("build_nodes", "build_triangle_visits"):
                self.statistics["face_"+name] += built["diagnostics"].get(name, 0)
            _require(built["status"] == "COMPLETE", "ASSIGNED_FACE_SURFACE_UNAVAILABLE")
            self.indices[face] = built["index"]
            self.statistics["indexed_face_count"] += 1
            self.statistics["indexed_face_triangles"] += len(rows)

    def __call__(self, face, points):
        stats, budget = self.statistics, self.budget
        _require(stats["query_calls"] < budget.max_prefilter_query_calls
            and stats["query_rows"]+len(points) <= budget.max_prefilter_query_rows,
            "PREFILTER_QUERY_BUDGET_EXCEEDED")
        values = asdict(self.surface_budget)
        for name in ("node_visits", "triangle_pairs", "normal_pairs", "work_units"):
            values["max_"+name] = min(values["max_"+name], getattr(budget, "max_"+name)-stats[name])
        stats["query_calls"] += 1
        stats["query_rows"] += len(points)
        query = _surface.query_closest_surface(self.indices[face], points,
            budgets=_surface.TriangleSurfaceBudgets(**values))
        for name in ("node_visits", "triangle_pairs", "normal_pairs", "work_units"):
            stats[name] += query["diagnostics"].get(name, 0)
        _require(query["status"] == "COMPLETE", "ASSIGNED_FACE_QUERY_UNAVAILABLE:"+str(query["reason"]))
        return query


def _prefilter_prepared(context, result, policy, budgets, surface_budgets):
    """Trusted internal context seam; mock use is NOT public provenance proof."""
    hypotheses, stats = context["hypotheses"], result["statistics"]
    _require(len(hypotheses) <= budgets.max_hypothesis_records, "HYPOTHESIS_RECORD_BUDGET_EXCEEDED")
    scans = {row["plane_id"]: row for row in context["scan_patches"]["patches"]}
    faces = {row["plane_id"]: row for row in context["cad_patches"]["patches"]}
    chunk = min(budgets.chunk_rows, surface_budgets.max_query_rows)
    _require(chunk > 0 or not hypotheses, "PREFILTER_QUERY_ROW_CAP_ZERO")
    used_faces, checked_scans, checked_faces = {}, set(), set()
    result["hypotheses"] = [dict(hypothesis, prefilter_status="NOT_ATTEMPTED",
        prefilter_reasons=[], assigned_support=[]) for hypothesis in hypotheses]
    stats["hypothesis_count"] = len(hypotheses)
    # Entire row/call obligation is frozen before first BVH or query. All three
    # patches will be evaluated even when the first is a geometric rejection.
    for expected_index, row in enumerate(result["hypotheses"]):
        _require(row["hypothesis_index"] == expected_index, "INVALID_HYPOTHESIS_INDEX")
        _pose(row)
        matches = row["correspondences"]
        _require(len(matches) == 3 and len({m["scan_plane_id"] for m in matches}) == 3
            and len({m["cad_plane_id"] for m in matches}) == 3, "INVALID_TRIPLET_BINDING")
        for match in matches:
            scan_id, face = match["scan_plane_id"], match["cad_plane_id"]
            _require(scan_id in scans and face in faces and match["normal_sign"] in (-1, 1),
                     "UNKNOWN_PLANE_SUPPORT_BINDING")
            if scan_id not in checked_scans:
                _bound_support(context, scan_patch=scans[scan_id])
                checked_scans.add(scan_id)
            if face not in checked_faces:
                _bound_support(context, cad_patch=faces[face])
                checked_faces.add(face)
            patch = scans[scan_id]
            count = len(patch["raw_vertex_rows"])
            stats["planned_query_rows"] += count
            stats["planned_query_calls"] += (count+chunk-1)//chunk
            used_faces[face] = faces[face]
            row["assigned_support"].append(dict(match, support_count=count,
                raw_vertex_rows_sha256=patch["raw_vertex_rows_sha256"],
                scan_triangle_rows_sha256=patch["triangle_rows_sha256"],
                cad_triangle_rows_sha256=faces[face]["triangle_rows_sha256"],
                query_rows=0, attempted_query_rows=0, finite_unambiguous_inlier_count=0, coverage=None,
                ambiguous_rows=0, boundary_rows=0, status="NOT_ATTEMPTED", reasons=[]))
    _require(stats["planned_query_rows"] <= budgets.max_prefilter_query_rows
        and stats["planned_query_calls"] <= budgets.max_prefilter_query_calls,
        "PREFILTER_PREFLIGHT_BUDGET_EXCEEDED")
    query = _FaceQueries(context, used_faces, budgets, surface_budgets, result)
    exact_groups = {}
    for row in result["hypotheses"]:
        r, t = _pose(row)
        row["prefilter_status"] = "INCOMPLETE"
        for support in row["assigned_support"]:
            raw = scans[support["scan_plane_id"]]["raw_vertex_rows"]
            support["status"] = "INCOMPLETE"
            for start in range(0, len(raw), chunk):
                points = context["scan"][raw[start:start+chunk]]@r.T+t
                support["attempted_query_rows"] += len(points)
                answer = query(support["cad_plane_id"], points)
                unambiguous = _v2._unambiguous(answer)
                support["query_rows"] += len(points)
                support["finite_unambiguous_inlier_count"] += int(np.count_nonzero(
                    unambiguous & (answer["distances"] <= policy.max_correspondence_distance_mm)))
                support["ambiguous_rows"] += int(np.count_nonzero(~unambiguous))
                support["boundary_rows"] += int(np.count_nonzero(answer["closest_regions"] != "FACE"))
            support["coverage"] = support["finite_unambiguous_inlier_count"]/support["support_count"]
            passed = support["coverage"] >= policy.min_scan_coverage
            support.update(status="PASSED" if passed else "REJECTED",
                reasons=[] if passed else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"])
        row["prefilter_reasons"] = ([] if all(s["status"] == "PASSED" for s in row["assigned_support"])
                                     else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
        passed = not row["prefilter_reasons"]
        row["prefilter_status"] = "PASSED" if passed else "REJECTED"
        stats["hypotheses_evaluated"] += 1
        stats["passing_hypothesis_count" if passed else "rejected_hypothesis_count"] += 1
        if passed:
            # Byte identity only, including signed zero. Nearby poses are NOT
            # merged. Every assignment was independently screened beforehand.
            key = r.tobytes(order="C")+t.tobytes(order="C")
            if key not in exact_groups:
                exact_groups[key] = len(result["start_groups"])
                result["start_groups"].append(dict(index=exact_groups[key],
                    initialized_rotation=r.tolist(), initialized_translation_mm=t.tolist(), hypothesis_indices=[]))
            group = exact_groups[key]
            row["start_group_index"] = group
            result["start_groups"][group]["hypothesis_indices"].append(row["hypothesis_index"])
    stats["surviving_start_group_count"] = len(result["start_groups"])
    # Deliberately finish and retain all dispositions before the survivor cap;
    # selecting the first/best24 would silently change the declared search.
    _require(len(result["start_groups"]) <= budgets.max_surviving_start_groups,
             "SURVIVING_START_GROUP_BUDGET_EXCEEDED")
    result.update(status="COMPLETE", search_complete=True,
        reasons=[] if result["start_groups"] else ["NO_FINITE_PREFILTERED_PLANE_STARTS"])


def _fail_prefilter(result, error):
    discarded_groups = len(result["start_groups"])
    result.update(status="UNAVAILABLE", reasons=[_reason(error)], search_complete=False,
                  start_groups=[], usable_hypotheses=False)
    result["statistics"]["surviving_start_groups_before_fatal_clear"] = discarded_groups
    result["statistics"]["surviving_start_group_count"] = 0
    result["evidence_scope"] = "ATTEMPTED_WORK_ONLY_ALL_STARTS_INVALIDATED"
    for row in result["hypotheses"]:
        row.pop("start_group_index", None)
        row["usable_initialization"] = False
    return result


def _reason(error):
    value = str(error)
    return value if value and value.isupper() and len(value) <= 180 else "INVALID_OR_NUMERICAL_INITIALIZATION_INPUT"


_ERRORS = (_Unavailable, _surface._Refusal, ValueError, TypeError, AttributeError, KeyError,
           IndexError, OverflowError, FloatingPointError, MemoryError, np.linalg.LinAlgError)


def _bounded_report(result, maximum):
    # A cap refusal publishes only a small refusal envelope, never a successful
    # prefix. The fixed refusal envelope is outside the detailed-report budget.
    size = 0
    try:
        for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(result):
            size += len(chunk.encode("utf-8"))
            if size > maximum:
                raise _Unavailable("INITIALIZATION_REPORT_BUDGET_EXCEEDED")
    except (ValueError, TypeError, OverflowError, MemoryError) as error:
        return dict(schema_version=result.get("schema_version", "plane_initialized_finite_registration_v4"),
            status="UNAVAILABLE", transform_scan_to_cad=None, reasons=[_reason(error)],
            search_complete=False, accepted_registration=False, production_use_allowed=False,
            observed_bend_count=None, conformance="INDETERMINATE", hypotheses=[], start_groups=[],
            detailed_report_discarded_whole=True, detailed_report_limit_bytes=maximum,
            encoded_bytes_examined=size,
            discarded_hypothesis_records=len(result.get("hypotheses", result.get("evidence", {}).get(
                "initialization", {}).get("hypotheses", []))),
            discarded_candidate_records=len(result.get("evidence", {}).get("candidates", [])),
            evidence_scope="NO_PARTIAL_REPORT_OR_POSE_PUBLISHED")
    return result


def _initialize(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                cad_source_triangle_ids, **configuration_args):
    result, context, configuration = _base(), None, None
    try:
        configuration = _configuration(**configuration_args)
        result.update(budgets=asdict(configuration["budgets"]),
            registration_policy=asdict(configuration["policy"]),
            surface_budgets=asdict(configuration["surface_budgets"]))
        context = _prepare(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                           cad_source_triangle_ids, configuration, result)
        _prefilter_prepared(context, result, configuration["policy"],
                            configuration["budgets"], configuration["surface_budgets"])
    except _ERRORS as error:
        _fail_prefilter(result, error)
        context = None
    maximum = configuration["budgets"].max_report_bytes if configuration else _LIMITS["max_report_bytes"]
    result = _bounded_report(result, maximum)
    return result, context if result["status"] == "COMPLETE" else None, configuration


def prefilter_plane_initializations_v4(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, registration_policy,
        patch_policy=_patch.CoplanarPatchPolicyV4(), patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        hypothesis_policy=_planes.PlaneHypothesisPolicyV4(), hypothesis_budgets=_planes.PlaneHypothesisBudgetsV4(),
        budgets=PlaneInitializationBudgetsV4(), surface_budgets=None):
    """Pure-array proposals; internally derived support, no accepted registration."""
    result, _, _ = _initialize(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, registration_policy=registration_policy,
        patch_policy=patch_policy, patch_budgets=patch_budgets,
        hypothesis_policy=hypothesis_policy, hypothesis_budgets=hypothesis_budgets,
        budgets=budgets, surface_budgets=surface_budgets)
    return result


def _registration_rows(context, supplied, evidence):
    if supplied is None:
        points = context["scan"]
        evidence["coverage_basis"] = "ALL_SUPPLIED_UNIQUE_SCAN_ROWS_TO_TRIANGLES_AND_CAD_SAMPLE_ROWS_TO_SCAN_ROWS"
        evidence["registration_raw_rows"] = dict(selection="ALL_RAW_ROWS", count=len(points), sha256=None)
    else:
        rows = _array(supplied, (), "iu", 200000, "<i8", "REGISTRATION_RAW_ROWS")
        _require(len(rows) >= 36 and np.all(rows < len(context["scan"]))
            and len(np.unique(rows)) == len(rows), "INVALID_REGISTRATION_RAW_ROW_SELECTION")
        points = context["scan"][rows]
        evidence["coverage_basis"] = "EXPLICIT_SELECTED_UNIQUE_SCAN_ROWS_TO_TRIANGLES_AND_CAD_SAMPLE_ROWS_TO_SELECTED_SCAN_ROWS"
        evidence["registration_raw_rows"] = dict(selection="EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING",
            count=len(rows), sha256=_digest(rows))
    evidence["prefilter_full_scan_raw_rows"] = len(context["scan"])
    evidence["registration_input_rows"] = len(points)
    evidence["registration_input_points_sha256"] = _digest(_patch._immutable(points, "<f8"))
    return _v1._canonical_scan(points)


def _registration_result(unit_assumption):
    evidence = dict(schema="PLANE_INITIALIZED_FINITE_TRIANGLE_WHOLE_PART_REGISTRATION_V4",
        registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT", datum_status="NO_VERIFIED_STABLE_DATUM",
        transform_model="SE3_RIGID_NO_FITTED_SCALE", uncertainty_status="UNCALIBRATED",
        unit_status=unit_assumption if type(unit_assumption) is str else "INVALID",
        candidate_search_status="PLANE_ONLY_FINITE_PREFILTERED_EXACT_STARTS_BOUNDED_NOT_GLOBAL",
        candidates=[], candidate_count=0, attempted_candidate_count=0, candidate_search_complete=False,
        candidate_numerical_failure_count=0, candidate_evaluation_complete=False,
        acquisition_side_identity="NOT_VERIFIED_BY_GLOBAL_ALIGNMENT",
        deformation_absorption_status="BEST_FIT_MAY_ABSORB_REAL_DEVIATION",
        observability_basis="UNAMBIGUOUS_FACE_INTERIORS_WITH_ACTUAL_TRIM_AND_HUBER_WEIGHTS_NO_REFILL",
        finite_boundary_status="USED_IN_DISTANCE_FIT_EXCLUDED_FROM_CONSERVATIVE_NORMAL_RANK",
        surface_error_status="APPROXIMATE_TRIANGULATION_NOT_ANALYTIC_CAD_OR_CALIBRATION",
        residual_gate="FINITE_EUCLIDEAN_P95_USING_UNCHANGED_LEGACY_PLANE_P95_NUMERIC_THRESHOLD",
        initial_assignments_are_final_face_matches=False, production_use_allowed=False)
    return dict(schema_version="plane_initialized_finite_registration_v4", status="UNAVAILABLE",
        transform_scan_to_cad=None, reasons=[], observed_bend_count=None,
        conformance="INDETERMINATE", production_use_allowed=False, evidence=evidence)


def _v2_select(result, refine, policy):
    """Exact V2 eligibility, ordering, final pose classes and ambiguity criteria."""
    evidence, candidates = result["evidence"], result["evidence"]["candidates"]
    eligible = [row for row in candidates if not row["reasons"]]
    evidence["eligible_candidate_count"] = len(eligible)
    if not eligible:
        result["reasons"] = ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
        return
    eligible.sort(key=lambda row: (-row["scan_coverage"], row["inlier_surface_rms_mm"],
                                   -row["cad_coverage"], row["index"]))
    classes = []
    for row in eligible:
        p = refine@np.asarray(row["rotation"]).T+row["translation"]
        if not any(float(np.max(_v2._norm(p-prior[1]))) <= policy.pose_class_distance_mm for prior in classes):
            classes.append((row, p))
    best = classes[0][0]
    competing = [row for row, _ in classes[1:]
        if row["scan_coverage"] >= best["scan_coverage"]-policy.ambiguity_coverage_margin
        and row["cad_coverage"] >= best["cad_coverage"]-policy.ambiguity_coverage_margin
        and row["inlier_surface_rms_mm"] <= best["inlier_surface_rms_mm"]+policy.ambiguity_plane_rms_margin_mm]
    evidence.update(distinct_eligible_pose_class_count=len(classes), comparable_distinct_pose_count=len(competing),
                    best_generated_candidate_index=best["index"])
    if competing:
        result["reasons"] = ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
        return
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = best["rotation"], best["translation"]
    result.update(status="CONDITIONAL_BEST_FIT", transform_scan_to_cad=transform.tolist())
    evidence["selected_candidate_index"] = best["index"]


def register_whole_part_planes_v4(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, cad_initialization_points,
        registration_policy, unit_assumption, registration_raw_rows=None,
        patch_policy=_patch.CoplanarPatchPolicyV4(), patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        hypothesis_policy=_planes.PlaneHypothesisPolicyV4(), hypothesis_budgets=_planes.PlaneHypothesisBudgetsV4(),
        budgets=PlaneInitializationBudgetsV4(), surface_budgets=None, registration_budgets=None):
    """Opt-in plane-only V2 arm; no native/file IO or default product activation.

    None registration_raw_rows preserves V2's <=200k supplied scan-row contract.
    Explicit unique raw IDs change ONLY global registration input, not prefilter
    support. The caller must separately bind CAD initialization samples to the
    supplied tessellation; this array API does not authenticate native/file origin.
    """
    result = _registration_result(unit_assumption)
    evidence, query = result["evidence"], None
    maximum = _LIMITS["max_report_bytes"]
    try:
        # All policy/budget/unit admission, including mutation revalidation,
        # precedes patch extraction or finite queries.
        configuration = _configuration(registration_policy, patch_policy, patch_budgets,
            hypothesis_policy, hypothesis_budgets, budgets, surface_budgets)
        maximum = configuration["budgets"].max_report_bytes
        execution = _v2._validated_execution_budgets(
            _v2.SurfaceRegistrationBudgetsV2() if registration_budgets is None else registration_budgets)
        execution = _v2.SurfaceRegistrationBudgetsV2(**asdict(execution))
        _require(type(unit_assumption) is str and unit_assumption == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                 "EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED")
        # Reject malformed/over-cap registration selection and CAD samples before
        # extracting patches or issuing any finite query. The owned full snapshot
        # is reused by initialization, so selected rows cannot come from another
        # mutable version of the input array.
        full_scan = _array(scan_points, (3,), "fiu", configuration["patch_budgets"].max_vertices,
                           "<f8", "SCAN_POINTS")
        scan = _registration_rows(dict(scan=full_scan), registration_raw_rows, evidence)
        cad = _v1._canonical_scan(cad_initialization_points)
        evidence["cad_initialization_points_sha256"] = _digest(_patch._immutable(cad, "<f8"))
        evidence["cad_sample_provenance"] = "CALLER_MUST_BIND_TO_TRIANGLES_NOT_AUTHENTICATED_BY_ARRAY_API"
        initialization, context, configuration = _initialize(full_scan, scan_triangle_raw_rows,
            cad_triangles, cad_face_ids, cad_source_triangle_ids,
            registration_policy=configuration["policy"], patch_policy=configuration["patch_policy"],
            patch_budgets=configuration["patch_budgets"], hypothesis_policy=configuration["hypothesis_policy"],
            hypothesis_budgets=configuration["hypothesis_budgets"], budgets=configuration["budgets"],
            surface_budgets=configuration["surface_budgets"])
        evidence["initialization"] = initialization
        _require(initialization["status"] == "COMPLETE", "PLANE_INITIALIZATION_UNAVAILABLE")
        groups = initialization["start_groups"]
        _require(bool(groups), "NO_FINITE_PREFILTERED_PLANE_STARTS")
        # Every admitted start is part of this execution denominator before
        # global sampling/index work, which can itself fail. Never hide known
        # starts merely because their shared whole-surface BVH was unavailable.
        candidates = evidence["candidates"]
        for group in groups:
            candidates.append(dict(group, execution_status="NOT_ATTEMPTED",
                                   reasons=["NOT_ATTEMPTED_YET"]))
        policy, surface_budget = configuration["policy"], configuration["surface_budgets"]
        evidence.update(policy=asdict(policy), surface_query_budgets=asdict(surface_budget),
            registration_budgets=asdict(execution), input_digests=dict(initialization["input_digests"]))
        coarse = _v1._spatial_sample(scan, policy.coarse_points)
        refine = _v1._spatial_sample(scan, policy.refine_points)
        built = _surface.build_triangle_surface(context["cad"], context["faces"], budgets=surface_budget)
        evidence["surface_index"] = {key: value for key, value in built.items() if key != "index"}
        _require(built["status"] == "COMPLETE", "FINITE_TRIANGLE_SURFACE_UNAVAILABLE")
        query = _v2._Queries(built["index"], execution, surface_budget)
        diagonal = float(_v2._norm(np.ptp(built["index"].triangles.reshape(-1, 3), axis=0)))
        evidence.update(scan_unique_rows=len(scan), cad_unique_rows=len(cad),
            coarse_scan_rows=len(coarse), refined_scan_rows=len(refine))
        for row in candidates:
            row.update(execution_status="ATTEMPTED", reasons=[])
            r, t = np.asarray(row["initialized_rotation"]), np.asarray(row["initialized_translation_mm"])
            try:
                r, t, nc, cr, cd = _v2._refine(coarse, query, r, t, diagonal=diagonal, policy=policy, coarse=True)
                r, t, nf, fr, fd = _v2._refine(refine, query, r, t, diagonal=diagonal, policy=policy, coarse=False)
                _pose(dict(rotation=r, translation_mm=t))
                row.update(_v2._diagnose(scan, cad, refine, query, r, t, policy))
                row.update(coarse_iterations=nc, refine_iterations=nf, coarse_convergence_state=cr,
                    refine_convergence_state=fr, coarse_last_step=cd, refine_last_step=fd,
                    rotation=r.tolist(), translation=t.tolist())
            except _v2._SurfaceFailure as error:
                row["reasons"] = [str(error)]
                raise
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                # Preserve V2's per-candidate numerical refusal, not a pass.
                row["reasons"] = ["CANDIDATE_NUMERICAL_FAILURE"]
        evidence["candidate_search_complete"] = True
        _v2_select(result, refine, policy)
    except _ERRORS as error:
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=[_reason(error)])
        evidence.pop("selected_candidate_index", None)
        evidence["candidate_search_complete"] = False
        evidence["evidence_scope"] = "ATTEMPTED_WORK_ONLY_NO_ACCEPTED_REGISTRATION"
        for row in evidence["candidates"]:
            if row["execution_status"] == "NOT_ATTEMPTED":
                row["reasons"] = ["NOT_ATTEMPTED_AFTER_FATAL_FAILURE"]
    finally:
        evidence["candidate_count"] = len(evidence["candidates"])
        evidence["attempted_candidate_count"] = sum(row["execution_status"] == "ATTEMPTED"
                                                    for row in evidence["candidates"])
        evidence["candidate_numerical_failure_count"] = sum("CANDIDATE_NUMERICAL_FAILURE" in row["reasons"]
                                                           for row in evidence["candidates"])
        evidence["candidate_evaluation_complete"] = (evidence["candidate_search_complete"]
            and evidence["attempted_candidate_count"] == evidence["candidate_count"]
            and evidence["candidate_numerical_failure_count"] == 0)
        if query is not None:
            evidence["query_resources"] = dict(query.statistics)
    return _bounded_report(result, maximum)
