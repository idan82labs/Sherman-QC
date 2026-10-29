"""Conditional scan-patch roles for measurement row selection, source-only V1.

No extraction, CAD face, registration, measurement, IO or triangle-normal fit.
V7 JSON contains ARRAY DESCRIPTORS, so the caller must also supply the exact
already-owned extraction result. Every proposed support is bound to those
descriptors and the original raw triangle/vertex union, before any selection.

An AXIAL_CAP_LIKE role means only that the WHOLE declared unoriented normal
cone lies inside a stipulated 15 degree axis cone. The source cone assumes a
common plane and stipulated error balls; neither that physical premise nor the
axis is certified here. Normalization/trigonometry and the 1e-10 radian inward
reserve are engineering arithmetic, not an interval certificate. No cap or
surface absence, scanner calibration, material orientation or conformance claim.

Cap eligibility requires a cap-contained proposal AND NO covering qualified
noncap/unresolved proposal. Conflicting triangles stay unknown. Only vertices
with >=1 incident raw triangle and ALL incident triangles cap-eligible are
excluded. Unknown/noncap incident triangles keep a vertex;
unreferenced vertices also stay. Coordinates are never welded or deleted.
Prepared contexts are process-local immutable implementation objects, not an
authentication mechanism against code able to alter Python module internals.
"""

from dataclasses import dataclass, fields
import hashlib
import json
import math

import numpy as np

from . import cad_guided_noisy_mesh_planes_v7 as _codec


AXIS_CONE_RADIANS = math.pi / 12.
ANGULAR_INWARD_RESERVE_RADIANS = 1e-10
UNIT_NORM_TOLERANCE = 1e-8
_TOKEN = object()


@dataclass(frozen=True)
class ContextSurfaceRoleBudgetsV1:
    max_raw_rows: int = 1_000_000
    max_triangles: int = 2_000_000
    max_proposals: int = 65_552
    max_triangle_memberships: int = 20_000_000
    max_raw_memberships: int = 32_000_000
    max_encoded_metadata_bytes: int = 67_108_864
    max_decoded_metadata_bytes: int = 268_435_456
    max_incidence_corner_visits: int = 6_000_000
    chunk_triangles: int = 65_536


class ContextSurfaceRolesUnavailableV1(ValueError):
    """No usable context/row filter is returned on failure."""

    def __init__(self, reason, *, statistics=None):
        super().__init__(reason)
        self.statistics = {} if statistics is None else dict(statistics)


@dataclass(frozen=True)
class _Support:
    kind: str
    record_id: int
    triangle_rows: np.ndarray
    triangle_rows_sha256: str
    raw_vertex_rows_sha256: str
    raw_support_count: int
    normal: tuple
    angle_bound_radians: float


@dataclass(frozen=True)
class PreparedContextSurfaceRolesV1:
    points_sha256: str
    triangle_raw_rows_sha256: str
    position_error_bounds_mm_sha256: str
    metadata_sha256: str
    raw_row_count: int
    triangle_count: int
    triangle_membership_count: int
    raw_membership_count: int
    encoded_metadata_bytes: int
    decoded_metadata_bytes: int
    unscheduled_qualified_core_count: int
    _triangles: np.ndarray
    _referenced: np.ndarray
    _supports: tuple
    _budgets: ContextSurfaceRoleBudgetsV1
    _token: object


def _require(condition, reason):
    if not condition:
        raise ContextSurfaceRolesUnavailableV1(reason)


def _sha(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _configuration(budgets):
    _require(type(budgets) is ContextSurfaceRoleBudgetsV1, "INVALID_ROLE_BUDGET_TYPE")
    maximum = ContextSurfaceRoleBudgetsV1()
    values = {f.name: getattr(budgets, f.name) for f in fields(maximum)}
    _require(all(type(v) is int and 0 <= v <= getattr(maximum, k) for k, v in values.items())
             and values["chunk_triangles"] > 0, "INVALID_ROLE_BUDGET")
    return ContextSurfaceRoleBudgetsV1(**values)


def _owned(value, shape, dtype, name):
    _require(type(value) is np.ndarray and value.shape == shape and value.dtype.kind in
             ("iu" if dtype == "<i8" else "fiu"), "INVALID_" + name)
    # Shape is captured by the caller before either snapshot. Recheck the owned
    # bytes before any indexing; caller mutation cannot introduce -1 aliases.
    copied = np.asarray(value, dtype=dtype).tobytes(order="C")
    _require(len(copied) == math.prod(shape) * 8, "OWNED_" + name + "_SHAPE_CHANGED")
    return np.frombuffer(copied, dtype=dtype).reshape(shape)


def _unit(value, name):
    _require(type(value) in (list, tuple) and len(value) == 3
             and all(type(v) in (int, float) and math.isfinite(v) for v in value),
             "INVALID_" + name)
    a = np.asarray(tuple(value), dtype="<f8")
    norm = float(np.linalg.norm(a))
    _require(math.isfinite(norm) and abs(norm - 1.) <= UNIT_NORM_TOLERANCE,
             "NONUNIT_" + name)
    return tuple(float(x) for x in a / norm)


def _encoded_snapshot(envelope, budget):
    _require(type(envelope) is dict, "MISSING_ENCODED_PATCH_EVIDENCE")
    # Do not let one huge scalar allocate an unbounded encoder fragment. This
    # is a structural input preflight, not a decoder or scientific validator.
    visits = 0
    def preflight(value, depth=0):
        nonlocal visits
        visits += 1
        _require(depth <= 192 and visits <= 8_000_000, "ROLE_METADATA_STRUCTURE_BUDGET_EXCEEDED")
        if type(value) is dict:
            _require(len(value) <= 128 and all(type(k) is str and len(k) <= 128 for k in value),
                     "INVALID_ROLE_METADATA_KEYS")
            for item in value.values():
                preflight(item, depth+1)
        elif type(value) is list:
            _require(len(value) <= 8_000_000, "ROLE_METADATA_LIST_BUDGET_EXCEEDED")
            for item in value:
                preflight(item, depth+1)
        else:
            _require(value is None or type(value) in (str, bool, int, float), "INVALID_ROLE_METADATA_SCALAR")
            if type(value) is str:
                _require(len(value) <= 65_536, "ROLE_METADATA_STRING_BUDGET_EXCEEDED")
            elif type(value) is int:
                _require(value.bit_length() <= 4096, "ROLE_METADATA_INTEGER_BUDGET_EXCEEDED")
            elif type(value) is float:
                _require(math.isfinite(value), "NONFINITE_ROLE_METADATA_SCALAR")
    preflight(envelope)
    # Bound before joining the encoded buffer or asking the decoder to allocate.
    buffer, size = bytearray(), 0
    encoder = json.JSONEncoder(allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    for fragment in encoder.iterencode(envelope):
        part = fragment.encode("utf-8")
        size += len(part)
        _require(size <= budget.max_encoded_metadata_bytes, "ROLE_ENCODED_METADATA_BUDGET_EXCEEDED")
        buffer.extend(part)
    blob = bytes(buffer)
    del buffer
    detached = json.loads(blob)
    # Charge the expanded wire tree BEFORE its construction, including keys
    # repeated by schema expansion. The strict decoder validates the grammar.
    expanded = 0
    def add(size):
        nonlocal expanded
        expanded += size
        _require(expanded <= budget.max_decoded_metadata_bytes, "ROLE_DECODED_METADATA_BUDGET_EXCEEDED")
    def expanded_size(node, depth=0):
        _require(depth <= 64, "ROLE_METADATA_DEPTH_BUDGET_EXCEEDED")
        if type(node) is not list:
            add(len(encoder.encode(node).encode("utf-8")))
        elif len(node) == 2 and node[0] == "L" and type(node[1]) is list:
            add(2 + max(0, len(node[1])-1))
            for item in node[1]:
                expanded_size(item, depth+1)
        else:
            _require(len(node) == 3 and node[0] == "D" and type(node[1]) is int
                     and type(node[2]) is list, "INVALID_ROLE_METADATA_NODE")
            schemas = detached.get("schemas")
            _require(type(schemas) is list and 0 <= node[1] < len(schemas), "INVALID_ROLE_METADATA_SCHEMA")
            keys, values = schemas[node[1]], node[2]
            _require(type(keys) is list and len(keys) == len(values), "INVALID_ROLE_METADATA_ARITY")
            add(2 + max(0, len(keys)-1) + len(keys))
            for key in keys:
                _require(type(key) is str, "INVALID_ROLE_METADATA_KEY")
                add(len(encoder.encode(key).encode("utf-8")))
            for item in values:
                expanded_size(item, depth+1)
    _require("root" in detached, "MISSING_ROLE_METADATA_ROOT")
    expanded_size(detached["root"])
    decoded = _codec.decode_metadata_v7(detached, max_bytes=budget.max_encoded_metadata_bytes)
    return decoded, hashlib.sha256(blob).hexdigest(), size, expanded


def _descriptor(array, descriptor, digest, name):
    _require(type(descriptor) is dict and set(descriptor) == {"shape", "dtype", "sha256"}
             and type(descriptor["shape"]) is list and all(type(v) is int for v in descriptor["shape"])
             and descriptor == dict(shape=list(array.shape), dtype="<i8", sha256=digest)
             and _sha(array) == digest, "STALE_" + name + "_DESCRIPTOR")


def _analysis(value):
    _require(type(value) is dict and value.get("status") == "PROPOSED"
             and value.get("witness_status") == "FOUND" and value.get("cone_status") == "AVAILABLE"
             and value.get("same_physical_plane_verified") is False
             and value.get("uncertainty_status") == "STIPULATED_UNCALIBRATED"
             and value.get("numerical_bound_status") == "ENGINEERING_ENCLOSURE_NOT_DIRECTED_ROUNDING_CERTIFICATE",
             "UNQUALIFIED_OR_UNSUPPORTED_PATCH_PREMISES")
    normal = _unit(value.get("normal"), "PATCH_NORMAL")
    alpha, count = value.get("normal_angle_bound_radians"), value.get("raw_support_count")
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 <= alpha <= math.pi / 4.
             and type(count) is int and count >= 3, "INVALID_PATCH_CONE_OR_COUNT")
    return normal, float(alpha), count


def prepare_context_surface_roles_v1(raw_points, raw_triangle_rows, scan_patches,
        encoded_patch_evidence, epsilon_digest, *, budgets=ContextSurfaceRoleBudgetsV1()):
    """Bind all qualified proposals once; failure raises, never an empty filter.

    epsilon_digest must come from the caller's owned error-array binding. This
    function compares it; it does not independently possess/calibrate epsilon.
    It cannot authenticate an arbitrary forged extraction analysis. The adapter
    must supply the already-completed V12 extraction object and bind the returned
    source hashes to that same initialization, not an unrelated caller report.
    """
    try:
        cap = _configuration(budgets)
        _require(type(raw_points) is np.ndarray and raw_points.ndim == 2 and raw_points.shape[1:] == (3,)
                 and type(raw_triangle_rows) is np.ndarray and raw_triangle_rows.ndim == 2
                 and raw_triangle_rows.shape[1:] == (3,), "INVALID_RAW_ARRAY_SHAPES")
        n, m = len(raw_points), len(raw_triangle_rows)
        _require(3 <= n <= cap.max_raw_rows and 1 <= m <= cap.max_triangles
                 and 3*m <= cap.max_incidence_corner_visits, "ROLE_INPUT_OR_INCIDENCE_BUDGET_EXCEEDED")
        p = _owned(raw_points, (n, 3), "<f8", "RAW_POINTS")
        t = _owned(raw_triangle_rows, (m, 3), "<i8", "RAW_TRIANGLES")
        _require(np.isfinite(p).all() and np.max(np.abs(p)) <= 1e6, "INVALID_OWNED_RAW_POINTS")
        _require(np.all(t >= 0) and np.all(t < n) and np.all(t[:, 0] != t[:, 1])
                 and np.all(t[:, 0] != t[:, 2]) and np.all(t[:, 1] != t[:, 2]), "INVALID_OWNED_RAW_TRIANGLE_IDS")
        _require(_hex(epsilon_digest), "INVALID_EPSILON_DIGEST")
        meta, meta_sha, encoded_bytes, decoded_bytes = _encoded_snapshot(encoded_patch_evidence, cap)
        _require(type(meta) is dict and type(scan_patches) is dict, "MISSING_NATIVE_PATCH_EVIDENCE")
        binding = dict(points_sha256=_sha(p), triangle_raw_rows_sha256=_sha(t),
                       position_error_bounds_mm_sha256=epsilon_digest)
        for value in (meta, scan_patches):
            _require(value.get("schema_version") == "noisy_mesh_planes_v7" and value.get("status") == "COMPLETE"
                     and value.get("input_digests") == binding and value.get("unit") == "mm"
                     and value.get("coordinate_frame") == "RAW_SCAN_UNREGISTERED"
                     and value.get("raw_coordinate_welding") is False and value.get("residual_trimming") is False,
                     "UNAVAILABLE_OR_STALE_PATCH_EVIDENCE")
            stats = value.get("statistics")
            _require(type(stats) is dict and stats.get("core_construction_complete") is True
                     and stats.get("core_evaluation_complete") is True
                     and stats.get("scheduled_closure_evaluation_complete") is True
                     and stats.get("input_raw_rows") == n and stats.get("input_triangles") == m,
                     "INCOMPLETE_PATCH_EVALUATION")
        refs = meta.get("proposals")
        _require(type(refs) is list and len(refs) <= cap.max_proposals
                 and scan_patches.get("proposals") == refs, "INVALID_OR_STALE_PROPOSAL_TABLE")
        qualified = []
        for kind, table in (("CORE", "cores"), ("CLOSURE", "closures")):
            rows, originals = meta.get(table), scan_patches.get(table)
            _require(type(rows) is list and type(originals) is list and len(rows) == len(originals)
                     and len(rows) <= cap.max_proposals, "INVALID_PATCH_RECORD_TABLE")
            for i, row in enumerate(rows):
                field = "core_id" if kind == "CORE" else "closure_id"
                _require(type(row) is dict and type(row.get(field)) is int and row[field] == i
                         and type(row.get("analysis")) is dict, "INVALID_PATCH_RECORD_ID")
                _require(type(originals[i]) is dict and type(originals[i].get(field)) is int
                         and originals[i][field] == i and type(originals[i].get("analysis")) is dict
                         and originals[i]["analysis"].get("status") == row["analysis"].get("status"),
                         "STALE_NATIVE_PATCH_RECORD")
                if row["analysis"].get("status") == "PROPOSED":
                    qualified.append((kind, i))
        for value in (meta, scan_patches):
            stats = value["statistics"]
            expected = dict(core_count=len(meta["cores"]), cores_evaluated=len(meta["cores"]),
                unique_closure_count=len(meta["closures"]),
                qualified_core_count=sum(kind == "CORE" for kind, _ in qualified),
                qualified_closure_count=sum(kind == "CLOSURE" for kind, _ in qualified))
            _require(all(type(stats.get(k)) is int and stats[k] == v for k, v in expected.items())
                     and type(stats.get("closures_scheduled")) is int
                     and type(stats.get("closures_evaluated")) is int
                     and stats["closures_scheduled"] == stats.get("closures_evaluated"),
                     "INCOMPLETE_PATCH_RECORD_DENOMINATORS")
        identities = []
        for ref in refs:
            _require(type(ref) is dict and set(ref) == {"kind", "record_id"}
                     and type(ref["kind"]) is str and ref["kind"] in ("CORE", "CLOSURE")
                     and type(ref["record_id"]) is int and ref["record_id"] >= 0, "INVALID_PROPOSAL_REFERENCE")
            identities.append((ref["kind"], ref["record_id"]))
        _require(len(set(identities)) == len(identities) and set(identities) == set(qualified),
                 "INCOMPLETE_OR_DUPLICATE_QUALIFIED_PROPOSAL_TABLE")
        supports, triangle_memberships, raw_memberships = [], 0, 0
        for kind, identifier in identities:
            table = "cores" if kind == "CORE" else "closures"
            descriptor, original = meta[table][identifier], scan_patches[table][identifier]
            _require(type(original) is dict and original.get("core_id" if kind == "CORE" else "closure_id") == identifier,
                     "STALE_NATIVE_PATCH_RECORD_ID")
            normal, alpha, count = _analysis(descriptor["analysis"])
            _require(_analysis(original.get("analysis")) == (normal, alpha, count)
                     and tuple(original["analysis"]["normal"]) == tuple(descriptor["analysis"]["normal"]),
                     "STALE_NATIVE_PATCH_ANALYSIS")
            rows, vertices = original.get("triangle_rows"), original.get("raw_vertex_rows")
            _require(type(rows) is np.ndarray and rows.ndim == 1 and type(vertices) is np.ndarray
                     and vertices.ndim == 1 and 0 < len(rows) <= m and 3 <= len(vertices) <= n,
                     "INVALID_NATIVE_PATCH_SUPPORT")
            triangle_memberships += len(rows)
            raw_memberships += len(vertices)
            _require(triangle_memberships <= cap.max_triangle_memberships
                     and raw_memberships <= cap.max_raw_memberships, "ROLE_SUPPORT_MEMBERSHIP_BUDGET_EXCEEDED")
            r = _owned(rows, (len(rows),), "<i8", "PATCH_TRIANGLE_ROWS")
            v = _owned(vertices, (len(vertices),), "<i8", "PATCH_RAW_VERTEX_ROWS")
            _require(np.all(r >= 0) and np.all(r < m) and np.all(r[1:] > r[:-1])
                     and np.all(v >= 0) and np.all(v < n) and np.all(v[1:] > v[:-1]), "INVALID_OWNED_PATCH_SUPPORT_IDS")
            rd, vd = descriptor.get("triangle_rows_sha256"), descriptor.get("raw_vertex_rows_sha256")
            _require(_hex(rd) and _hex(vd) and original.get("triangle_rows_sha256") == rd
                     and original.get("raw_vertex_rows_sha256") == vd, "STALE_PATCH_SUPPORT_DIGEST")
            _descriptor(r, descriptor.get("triangle_rows"), rd, "TRIANGLE_SUPPORT")
            _descriptor(v, descriptor.get("raw_vertex_rows"), vd, "RAW_SUPPORT")
            _require(count == len(v) and np.array_equal(v, np.unique(t[r].ravel())),
                     "INCOMPLETE_OR_WRONG_RAW_SUPPORT_UNION")
            supports.append(_Support(kind, identifier, r, rd, vd, count, normal, alpha))
        unscheduled = meta["statistics"].get("unscheduled_qualified_core_count")
        _require(type(unscheduled) is int and 0 <= unscheduled <= len(meta["cores"]), "INVALID_CLOSURE_SCHEDULING_EVIDENCE")
        referenced = np.zeros(n, dtype=bool)
        for start in range(0, m, cap.chunk_triangles):
            referenced[t[start:start+cap.chunk_triangles].ravel()] = True
        owned_referenced = np.frombuffer(referenced.tobytes(), dtype=bool)
        return PreparedContextSurfaceRolesV1(binding["points_sha256"], binding["triangle_raw_rows_sha256"],
            epsilon_digest, meta_sha, n, m, triangle_memberships, raw_memberships, encoded_bytes,
            decoded_bytes, unscheduled, t, owned_referenced, tuple(supports), cap, _TOKEN)
    except ContextSurfaceRolesUnavailableV1:
        raise
    except Exception as error:
        raise ContextSurfaceRolesUnavailableV1("ROLE_PREPARATION_NUMERIC_OR_METADATA_FAILURE") from error


def select_context_measurement_rows_v1(prepared, axis, *, max_triangle_membership_visits=20_000_000,
        max_incidence_corner_visits=6_000_000):
    """One axis, complete preflight: caller must allocate aggregate work/cache.

    All proposal memberships are charged, including noncap proposals; complete
    raw incidence is charged regardless of exclusion. No positive prefix exists
    when the requested allowance is insufficient. The axis must be raw-frame.
    """
    work = dict(planned_triangle_membership_visits=None, planned_incidence_corner_visits=None,
        attempted_triangle_membership_visits=0, completed_triangle_membership_visits=0,
        attempted_incidence_corner_visits=0, completed_incidence_corner_visits=0,
        whole_selection_preflight_complete=False)
    try:
        _require(type(prepared) is PreparedContextSurfaceRolesV1 and prepared._token is _TOKEN,
                 "INVALID_PREPARED_ROLE_CONTEXT")
        for value, maximum in ((max_triangle_membership_visits, 20_000_000),
                               (max_incidence_corner_visits, 6_000_000)):
            _require(type(value) is int and 0 <= value <= maximum, "INVALID_ROLE_SELECTION_BUDGET")
        _require(type(axis) is np.ndarray and axis.shape == (3,) and axis.dtype.kind in "fiu", "INVALID_RAW_FRAME_AXIS")
        owned_axis = _owned(axis, (3,), "<f8", "RAW_FRAME_AXIS")
        direction = _unit(owned_axis.tolist(), "RAW_FRAME_AXIS")
        planned_memberships, planned_corners = prepared.triangle_membership_count, 3*prepared.triangle_count
        work.update(planned_triangle_membership_visits=planned_memberships,
                    planned_incidence_corner_visits=planned_corners)
        _require(planned_memberships <= max_triangle_membership_visits
                 and planned_corners <= max_incidence_corner_visits, "ROLE_SELECTION_WORK_BUDGET_EXCEEDED_BEFORE_SELECTION")
        work["whole_selection_preflight_complete"] = True
        cap_union = np.zeros(prepared.triangle_count, dtype=bool)
        competing_union = np.zeros(prepared.triangle_count, dtype=bool)
        dispositions = []
        for support in prepared._supports:
            work["attempted_triangle_membership_visits"] += len(support.triangle_rows)
            cosine = min(1., max(0., abs(float(np.dot(support.normal, direction)))))
            theta = math.acos(cosine)
            contained = theta + support.angle_bound_radians + ANGULAR_INWARD_RESERVE_RADIANS <= AXIS_CONE_RADIANS
            if contained:
                cap_union[support.triangle_rows] = True
            else:
                competing_union[support.triangle_rows] = True
            dispositions.append(dict(kind=support.kind, record_id=support.record_id,
                role="AXIAL_CAP_LIKE" if contained else "UNKNOWN_NONCAP_OR_UNRESOLVED",
                nominal_axis_angle_radians=theta, normal_angle_bound_radians=support.angle_bound_radians,
                triangle_rows_sha256=support.triangle_rows_sha256,
                raw_vertex_rows_sha256=support.raw_vertex_rows_sha256,
                triangle_count=len(support.triangle_rows), raw_support_count=support.raw_support_count))
            work["completed_triangle_membership_visits"] += len(support.triangle_rows)
        conflict_count = int(np.count_nonzero(cap_union & competing_union))
        cap_proposed_count = int(np.count_nonzero(cap_union))
        cap_union &= ~competing_union
        keep = ~prepared._referenced.copy()
        for start in range(0, prepared.triangle_count, prepared._budgets.chunk_triangles):
            rows = prepared._triangles[start:start+prepared._budgets.chunk_triangles]
            work["attempted_incidence_corner_visits"] += 3*len(rows)
            keep[rows[~cap_union[start:start+len(rows)]].ravel()] = True
            work["completed_incidence_corner_visits"] += 3*len(rows)
        retained = np.frombuffer(np.asarray(np.flatnonzero(keep), dtype="<i8").tobytes(), dtype="<i8")
        excluded = np.frombuffer(np.asarray(np.flatnonzero(~keep), dtype="<i8").tobytes(), dtype="<i8")
        return dict(status="COMPLETE", keep_raw_rows=retained, excluded_raw_rows=excluded,
            metadata=dict(schema_version="context_surface_roles_v1", coordinate_frame="RAW_SCAN_UNREGISTERED",
                scope="CONDITIONAL_PATCH_NORMAL_ROLE_NOT_PHYSICAL_CAP_CERTIFICATE",
                role_policy="CAP_AND_NO_COMPETING_NONCAP_OR_UNRESOLVED_ALL_INCIDENT_TRIANGLES",
                normal_premise="STIPULATED_COMMON_PLANE_ERROR_BALL_CONE_NOT_VERIFIED_PHYSICAL_PLANE",
                numeric_status="ENGINEERING_NORMALIZATION_AND_ANGULAR_INWARD_RESERVE_NOT_CERTIFIED",
                axis=direction, axis_cone_radians=AXIS_CONE_RADIANS,
                angular_inward_reserve_radians=ANGULAR_INWARD_RESERVE_RADIANS,
                points_sha256=prepared.points_sha256, triangle_raw_rows_sha256=prepared.triangle_raw_rows_sha256,
                position_error_bounds_mm_sha256=prepared.position_error_bounds_mm_sha256,
                encoded_patch_metadata_sha256=prepared.metadata_sha256,
                keep_raw_rows_sha256=_sha(retained), excluded_raw_rows_sha256=_sha(excluded),
                raw_row_count=prepared.raw_row_count, kept_row_count=len(retained), excluded_row_count=len(excluded),
                cap_like_triangle_count=int(np.count_nonzero(cap_union)), unknown_triangle_count=int(np.count_nonzero(~cap_union)),
                cap_proposed_triangle_count=cap_proposed_count, conflicting_triangle_count=conflict_count,
                unreferenced_rows_kept=int(np.count_nonzero(~prepared._referenced)),
                proposal_dispositions=dispositions, proposal_count=len(dispositions),
                unscheduled_qualified_core_closures=prepared.unscheduled_qualified_core_count,
                selected_proposal_evaluation_complete=True, extraction_search_exhaustive=False,
                **work,
                work_semantics="ALL_PROPOSAL_MEMBERSHIPS_AND_COMPLETE_RAW_INCIDENCE_CHARGED_NO_REFILL",
                production_use_allowed=False, observed_bend_count=None, conformance="INDETERMINATE"))
    except ContextSurfaceRolesUnavailableV1 as error:
        error.statistics = dict(work)
        raise
    except Exception as error:
        raise ContextSurfaceRolesUnavailableV1("ROLE_SELECTION_NUMERIC_FAILURE", statistics=work) from error
