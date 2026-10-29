"""Bounded caller-scheduled two-plane/context seeds, never registration approval.

Pure NumPy/stdlib; no native, file, scoring, case-name or application imports.
Input plane qualification and mm/frame labels are DECLARED premises, not an
authentication/calibration claim. Both normal systems are independently checked
for nominal rank and a positive conditional cone margin. Unequal central angles
give only a proper-rotation proposal. Axial extent votes are heuristic, not pose
bounds, support, a measurement, a count or global uniqueness.

No scheduler is implemented: callers supply <=64 distinct scan pairs and <=4
ordered CAD matches per pair. Four target signs times eight explicit slots gives
<=8192 slots. A future scheduler/union with the triplet lane requires independent
qualification and joint quota accounting. Frozen V10 is not modified/invoked.

The 256 MiB budget accounts owned numeric arrays, conservative temporary numeric
payload, cache keys/records and serialized proposal payload. It is NOT total
Python-object/allocator/process RSS. Input caller memory is outside that ledger.
One 64 KiB fixed scratch reservation covers bounded rank/pose and JSON fragments;
large buffer validation, unions, sorting and projection have separate charges.
Refusal envelopes have a separate fixed 4096-byte ceiling. No execution budget or
physical configuration is changed by these source-only helper limits.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np


@dataclass(frozen=True)
class ContextPlaneV1:
    plane_id: int
    normal: tuple
    anchor_mm: tuple
    offset_from_anchor_mm: float
    normal_angle_bound_radians: float
    normal_qualified: bool
    triangle_rows: np.ndarray
    triangle_rows_sha256: str
    raw_vertex_rows: object = None
    raw_vertex_rows_sha256: object = None


@dataclass(frozen=True)
class OrderedPlanePairV1:
    scan_plane_ids: tuple
    cad_plane_ids: tuple


@dataclass(frozen=True)
class TwoPlaneContextBudgetsV1:
    max_scan_rows: int = 1_000_000
    max_scan_triangles: int = 2_000_000
    max_cad_triangles: int = 500_000
    max_scan_planes: int = 128
    max_cad_planes: int = 2048
    max_scan_pairs: int = 64
    max_ordered_cad_matches_per_scan_pair: int = 4
    max_signed_families: int = 1024
    max_candidate_slots: int = 8192
    max_scan_row_visits: int = 64_000_000
    max_cad_vertex_visits: int = 128_000_000
    max_index_visits: int = 64_000_000
    max_index_sort_work: int = 256_000_000
    max_hash_bytes: int = 1_073_741_824
    max_projection_cache_entries: int = 6144
    max_working_bytes: int = 268_435_456
    max_report_bytes: int = 33_554_432
    chunk_rows: int = 65_536


# Immutable literals independent of the mutable public default instance.
_MAXIMA = (("max_scan_rows", 1_000_000), ("max_scan_triangles", 2_000_000),
    ("max_cad_triangles", 500_000), ("max_scan_planes", 128), ("max_cad_planes", 2048),
    ("max_scan_pairs", 64), ("max_ordered_cad_matches_per_scan_pair", 4),
    ("max_signed_families", 1024), ("max_candidate_slots", 8192),
    ("max_scan_row_visits", 64_000_000), ("max_cad_vertex_visits", 128_000_000),
    ("max_index_visits", 64_000_000), ("max_index_sort_work", 256_000_000),
    ("max_hash_bytes", 1_073_741_824), ("max_projection_cache_entries", 6144),
    ("max_working_bytes", 268_435_456), ("max_report_bytes", 33_554_432),
    ("chunk_rows", 65_536))
_GATES = (("min_normal_condition_ratio", .02), ("max_normal_alignment_extra_radians", .02),
    ("max_anchor_backward_residual_mm", 1e-8), ("pair_angle_numeric_guard_radians", 1e-10),
    ("max_normal_angle_radians", math.pi/18), ("normal_unit_guard", 1e-8),
    ("proper_rotation_guard", 1e-10))
_SLOTS = ("WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
          "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT")


class _Unavailable(ValueError):
    pass


def _require(condition, code):
    if not condition:
        raise _Unavailable(code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _encoded_size(value, maximum):
    count = 0
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        count += len(part.encode())
        _require(count <= maximum, "ENCODED_PAYLOAD_BYTE_BUDGET")
    return count


def _identifier(value):
    _require(type(value) is int and 0 <= value < 2**63, "INVALID_IMPORT_LOCAL_PLANE_ID")
    return value


def _vector(value):
    _require(type(value) is tuple and len(value) == 3, "EXPLICIT_THREE_VECTOR_REQUIRED")
    _require(all(type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1e6 for x in value),
             "UNSUPPORTED_VECTOR_RANGE")
    return tuple(float(x) for x in value)


class _Work:
    def __init__(self, cap):
        self.cap, self.cache = cap, {}
        self.stats = dict(scan_row_visits=0, cad_vertex_visits=0, index_visits=0,
            index_sort_work=0, hash_bytes=0, owned_array_bytes=0, temporary_bytes=0,
            fixed_scratch_bytes=65_536,
            cache_payload_bytes=0, proposal_payload_bytes=0, accounted_working_peak_bytes=0,
            projection_cache_hits=0, projection_cache_misses=0, projection_cache_entries=0,
            signed_families_attempted=0, signed_families_completed=0,
            slots_generation_attempted=0, slot_records_completed=0, proposed_slot_count=0,
            absent_midpoint_slots=0, rejected_slot_count=0, exact_pose_duplicates=0)

    def charge(self, name, amount):
        _require(type(amount) is int and amount >= 0, "INVALID_WORK_CHARGE")
        # Increment BEFORE refusal: attempted work is not called completed.
        self.stats[name] += amount
        _require(self.stats[name] <= self.cap["max_"+name], "BUDGET_"+name.upper())

    def memory(self, extra=0):
        size = sum(self.stats[k] for k in ("owned_array_bytes", "temporary_bytes", "fixed_scratch_bytes",
                   "cache_payload_bytes", "proposal_payload_bytes"))+extra
        self.stats["accounted_working_peak_bytes"] = max(self.stats["accounted_working_peak_bytes"], size)
        _require(size <= self.cap["max_working_bytes"], "ACCOUNTED_WORKING_PAYLOAD_BUDGET")

    @contextmanager
    def temporary(self, size):
        self.stats["temporary_bytes"] += size
        try:
            self.memory()
            yield
        finally:
            self.stats["temporary_bytes"] -= size

    def record(self, value):
        # Record trees are bounded fixed schemas, never arbitrary caller objects.
        # Stream size first; do not allocate an uncharged whole encoded record.
        size = _encoded_size(value, self.cap["max_report_bytes"])
        self.memory(size)
        self.stats["proposal_payload_bytes"] += size
        _require(self.stats["proposal_payload_bytes"] <= self.cap["max_report_bytes"], "PROPOSAL_BYTE_BUDGET")
        self.memory()

    def digest(self, array):
        self.charge("hash_bytes", int(array.nbytes))
        return hashlib.sha256(memoryview(array).cast("B")).hexdigest()

    def own(self, value, tail, dtype, maximum, name):
        _require(type(value) is np.ndarray, "INVALID_"+name+"_ARRAY")
        shape, actual_dtype = tuple(value.shape), value.dtype
        size = int(value.nbytes)
        expected_dtype = np.dtype(dtype)
        _require(actual_dtype == expected_dtype and len(shape) == 1+len(tail)
            and shape[1:] == tail and shape[0] <= maximum
            and size == math.prod(shape)*expected_dtype.itemsize, "INVALID_"+name+"_ARRAY")
        self.memory(size)
        # A caller may reshape after byte capture. Never reread its shape to
        # interpret the admitted bytes: row identity uses this captured shape.
        blob = value.tobytes(order="C")
        _require(len(blob) == size, "CHANGED_"+name+"_SNAPSHOT_BYTE_COUNT")
        answer = np.frombuffer(blob, dtype=expected_dtype).reshape(shape)
        _require(type(answer) is np.ndarray and answer.dtype == expected_dtype
            and answer.shape == shape and answer.nbytes == size and len(answer) <= maximum,
            "INVALID_"+name+"_OWNED_SNAPSHOT")
        self.stats["owned_array_bytes"] += size
        return answer

    def sort_charge(self, count):
        self.charge("index_sort_work", count*max(1, (max(2, count)-1).bit_length()))


def _base():
    return dict(schema_version="two_plane_context_v1", status="UNAVAILABLE", reasons=[],
        scope="CALLER_SCHEDULED_ANCHORED_TWO_PLANE_EXTREMA_SEEDS_ONLY",
        accepted_registration=False, finite_support_match_verified=False, observed_bend_count=None,
        production_use_allowed=False, global_pose_uniqueness="NOT_ESTABLISHED",
        axial_context_status="AXIAL_CONTEXT_NOT_CERTIFIED", search_exhaustive=False,
        scheduler_qualification="NOT_IMPLEMENTED_CALLER_SCHEDULE_NOT_AUTHENTICATED",
        selected_policy_evaluation_complete=False, usable_proposals=False,
        normal_qualification="CALLER_DECLARED_QUALIFIED_CONES_NOT_RECOMPUTED_OR_AUTHENTICATED",
        translation_uncertainty="UNAVAILABLE_NO_OFFSET_ERROR_OR_POSE_ENCLOSURE_MODEL",
        downstream_requirement="UNCHANGED_FINITE_SUPPORT_AMBIGUITY_AND_ALL_RANK_GATES",
        lineage_scope="INPUT_BUFFER_AND_SUPPORT_ROW_HASHES_PLUS_EXPLICIT_EXTREMA_ROW_IDENTITIES",
        numeric_scope="ENGINEERING_BINARY64_PROPOSALS_NOT_INTERVAL_CERTIFICATES",
        correspondence_order="CANONICAL_SOURCE_ID_ORDER_CALLER_ORDERED_TARGETS_NOT_NOISY_ID_RENUMBERING_INVARIANT",
        memory_scope="ACCOUNTED_NUMERIC_AND_ENCODED_PAYLOAD_NOT_TOTAL_PYTHON_RSS",
        input_digests={}, plane_bindings=[], scheduled_pairs=[], families=[], candidates=[],
        context_projections=[], statistics={})


def _configuration(budgets):
    _require(type(budgets) is TwoPlaneContextBudgetsV1, "INVALID_BUDGET_TYPE")
    snapshot = {key: getattr(budgets, key) for key, _ in _MAXIMA}
    _require(all(type(snapshot[key]) is int and 0 <= snapshot[key] <= value for key, value in _MAXIMA)
             and snapshot["chunk_rows"] >= 3, "INVALID_BUDGET_VALUES")
    return snapshot


def _owned_indices(value, expected, maximum, work, name):
    rows = work.own(value, (), "int64", maximum, name)
    work.charge("index_visits", len(rows))
    with work.temporary(4*len(rows)):
        _require(len(rows) > 0 and np.all(rows >= 0) and np.all(rows < maximum)
            and np.all(rows[1:] > rows[:-1]), "INVALID_"+name+"_ROWS")
    _require(type(expected) is str and len(expected) == 64 and work.digest(rows) == expected,
             "STALE_"+name+"_DIGEST")
    return rows


def _planes(values, role, context, work, result):
    cap, gate = work.cap, dict(_GATES)
    _require(type(values) is tuple and len(values) <= cap["max_"+role+"_planes"], "INVALID_PLANE_TABLE")
    table = {}
    for value in values:
        _require(type(value) is ContextPlaneV1, "INVALID_CONTEXT_PLANE_TYPE")
        # Snapshot all caller fields before validation; arrays get their own copy.
        fields = {name: getattr(value, name) for name in ContextPlaneV1.__dataclass_fields__}
        identifier = _identifier(fields["plane_id"])
        _require(identifier not in table and fields["normal_qualified"] is True, "DUPLICATE_OR_UNQUALIFIED_PLANE")
        normal, anchor = _vector(fields["normal"]), _vector(fields["anchor_mm"])
        length = math.hypot(*normal)
        _require(abs(length-1.) <= gate["normal_unit_guard"], "PLANE_NORMAL_NOT_UNIT")
        offset, alpha = fields["offset_from_anchor_mm"], fields["normal_angle_bound_radians"]
        _require(type(offset) in (int, float) and math.isfinite(offset) and abs(offset) <= 1e6
            and type(alpha) in (int, float) and math.isfinite(alpha)
            and 0 <= alpha <= gate["max_normal_angle_radians"], "INVALID_ANCHORED_PLANE_OR_CONE")
        normal = tuple(x/length for x in normal)
        offset = float(offset)/length  # Normalize BOTH equation coefficients.
        triangles = _owned_indices(fields["triangle_rows"], fields["triangle_rows_sha256"],
            len(context[role+"_triangles"]), work, role.upper()+"_SUPPORT_TRIANGLE")
        raw = None
        if role == "scan":
            raw = _owned_indices(fields["raw_vertex_rows"], fields["raw_vertex_rows_sha256"],
                len(context["scan_points"]), work, "SCAN_SUPPORT_RAW")
            _require(len(raw) >= 3, "SCAN_SUPPORT_TOO_SMALL")
            count = 3*len(triangles)
            work.charge("index_visits", count+len(raw))
            work.sort_charge(count)
            with work.temporary(48*count+8*len(raw)):
                union = np.unique(context["scan_triangles"][triangles])
                _require(np.array_equal(union, raw), "INCOMPLETE_SCAN_SUPPORT_RAW_UNION")
                del union
        else:
            _require(fields["raw_vertex_rows"] is None and fields["raw_vertex_rows_sha256"] is None,
                     "CAD_CONTEXT_MUST_USE_TRIANGLE_VERTEX_IDENTITIES")
            order, sorted_faces = context["cad_order"], context["cad_sorted_faces"]
            start, end = np.searchsorted(sorted_faces, identifier, side="left"), np.searchsorted(sorted_faces, identifier, side="right")
            work.charge("index_visits", int(end-start))
            work.sort_charge(int(end-start))
            with work.temporary(32*int(end-start)):
                _require(np.array_equal(np.sort(order[start:end]), triangles), "INCOMPLETE_ASSIGNED_CAD_FACE")
        record = dict(role=role, plane_id=identifier, normal=list(normal), anchor_mm=list(anchor),
            offset_from_anchor_mm=offset, normal_angle_bound_radians=float(alpha),
            triangle_rows_sha256=fields["triangle_rows_sha256"], triangle_count=len(triangles),
            raw_vertex_rows_sha256=fields["raw_vertex_rows_sha256"], raw_row_count=None if raw is None else len(raw))
        work.record(record)
        result["plane_bindings"].append(record)
        table[identifier] = dict(record, triangle_rows=triangles, raw_vertex_rows=raw)
    return table


def _inputs(scan_points, scan_triangles, cad_triangles, cad_face_ids, cad_source_triangle_ids, work, result):
    cap, context = work.cap, {}
    for name, value, tail, dtype, maximum in (
        ("scan_points", scan_points, (3,), "float64", cap["max_scan_rows"]),
        ("scan_triangles", scan_triangles, (3,), "int64", cap["max_scan_triangles"]),
        ("cad_triangles", cad_triangles, (3, 3), "float64", cap["max_cad_triangles"]),
        ("cad_face_ids", cad_face_ids, (), "int64", cap["max_cad_triangles"]),
        ("cad_source_triangle_ids", cad_source_triangle_ids, (), "int64", cap["max_cad_triangles"])):
        context[name] = work.own(value, tail, dtype, maximum, name.upper())
    n, t = len(context["scan_points"]), len(context["cad_triangles"])
    _require(n >= 3 and len(context["scan_triangles"]) > 0 and t > 0
        and len(context["cad_face_ids"]) == len(context["cad_source_triangle_ids"]) == t,
        "INCOMPLETE_CONTEXT_BUFFERS")
    work.charge("scan_row_visits", n)
    work.charge("cad_vertex_visits", 3*t)
    work.charge("index_visits", 3*len(context["scan_triangles"])+2*t)
    chunk = cap["chunk_rows"]
    for name in ("scan_points", "cad_triangles"):
        width = chunk if name == "scan_points" else max(1, chunk//3)
        for start in range(0, len(context[name]), width):
            value = context[name][start:start+width]
            with work.temporary(int(value.nbytes)*2):
                _require(np.isfinite(value).all() and np.all(np.abs(value) <= 1e6), "UNSUPPORTED_CONTEXT_COORDINATE_RANGE")
    raw, faces, local = context["scan_triangles"], context["cad_face_ids"], context["cad_source_triangle_ids"]
    with work.temporary(max(int(raw.size), t)):
        _require(np.all(raw >= 0) and np.all(raw < n) and np.all(faces >= 0) and np.all(local >= 0),
                 "INVALID_CONTEXT_TRIANGLE_IDENTITY")
    for name, array in context.items():
        result["input_digests"][name+"_sha256"] = work.digest(array)
    work.sort_charge(t)
    with work.temporary(64*t):
        order = np.lexsort((local, faces))
        ordered_faces, ordered_local = faces[order], local[order]
        _require(not np.any((ordered_faces[1:] == ordered_faces[:-1]) &
                            (ordered_local[1:] == ordered_local[:-1])), "DUPLICATE_CAD_SOURCE_TRIANGLE_IDENTITY")
        context["cad_order"] = work.own(order, (), "int64", t, "CAD_SORT_ORDER")
        context["cad_sorted_faces"] = work.own(ordered_faces, (), "int64", t, "CAD_SORTED_FACES")
        del order, ordered_faces, ordered_local
    result["input_counts"] = dict(scan_raw_rows=n, scan_triangles=len(raw), cad_triangles=t)
    return context


def _schedule(matches, scans, cads, cap):
    _require(type(matches) is tuple and len(matches) <= cap["max_scan_pairs"]*cap["max_ordered_cad_matches_per_scan_pair"],
             "ORDERED_MATCH_TABLE_BUDGET")
    pairs, seen, records = {}, set(), []
    for item in matches:
        _require(type(item) is OrderedPlanePairV1, "INVALID_ORDERED_PAIR_TYPE")
        source, target = item.scan_plane_ids, item.cad_plane_ids
        _require(type(source) is tuple and type(target) is tuple and len(source) == len(target) == 2,
                 "EXACTLY_TWO_DISTINCT_ASSIGNMENTS_REQUIRED")
        source, target = tuple(_identifier(x) for x in source), tuple(_identifier(x) for x in target)
        _require(len(set(source)) == len(set(target)) == 2 and all(x in scans for x in source)
            and all(x in cads for x in target), "UNKNOWN_OR_REPEATED_PLANE_ASSIGNMENT")
        # Canonical source order; reversing both sides is the SAME assignment.
        if source[0] > source[1]:
            source, target = source[::-1], target[::-1]
        _require((source, target) not in seen, "DUPLICATE_ORDERED_PAIR_ASSIGNMENT")
        seen.add((source, target))
        pairs.setdefault(source, []).append(target)
        _require(len(pairs) <= cap["max_scan_pairs"] and len(pairs[source]) <= cap["max_ordered_cad_matches_per_scan_pair"],
                 "PAIR_ASSIGNMENT_QUOTA_EXCEEDED")
        records.append(dict(scan_plane_ids=list(source), cad_plane_ids=list(target)))
    _require(4*len(records) <= cap["max_signed_families"] and 32*len(records) <= cap["max_candidate_slots"],
             "COMPLETE_SIGN_AND_SLOT_PREFLIGHT_BUDGET")
    return records


def _rank(planes):
    normals = np.asarray([p["normal"] for p in planes])
    singular = np.linalg.svd(normals, compute_uv=False)
    _require(np.isfinite(singular).all() and singular[0] > 0, "NONFINITE_TWO_ROW_SYSTEM")
    eta = math.sqrt(math.fsum((2*math.sin(p["normal_angle_bound_radians"]/2))**2 for p in planes))
    ratio, margin = float(singular[-1]/singular[0]), float(singular[-1])-eta
    reasons = []
    if ratio < dict(_GATES)["min_normal_condition_ratio"]:
        reasons.append("NOMINAL_TWO_ROW_SYSTEM_ILL_CONDITIONED")
    if margin <= 0:
        reasons.append("TWO_ROW_INDEPENDENCE_UNRESOLVED_WITH_CONES")
    return dict(status="REJECTED" if reasons else "AVAILABLE", reasons=reasons,
        normal_condition_ratio=ratio, declared_normal_perturbation_frobenius_bound=eta,
        normal_independence_margin=margin)


def _angle(a, b, unoriented=False):
    dot = float(np.dot(a, b))
    return math.atan2(math.hypot(*np.cross(a, b)), abs(dot) if unoriented else dot)


def _frame(normals):
    first = normals[0]
    second = normals[1]-np.dot(first, normals[1])*first
    length = math.hypot(*second)
    _require(math.isfinite(length) and length > 0, "DEGENERATE_CENTRAL_NORMAL_FRAME")
    second = second/length
    return np.column_stack((first, second, np.cross(first, second)))


def _family(source, target, signs, origin_s, origin_c):
    gate = dict(_GATES)
    rank_s, rank_c = _rank(source), _rank(target)
    out = dict(source_rank=rank_s, target_rank=rank_c, status="REJECTED", reasons=[], rotation=None,
               translation_family=None)
    if rank_s["status"] != "AVAILABLE" or rank_c["status"] != "AVAILABLE":
        out["reasons"] = ["TWO_ROW_SYSTEM_UNAVAILABLE"]
        return out
    ns, nc = np.asarray([p["normal"] for p in source]), np.asarray([p["normal"] for p in target])
    angle_gap = abs(_angle(*ns, unoriented=True)-_angle(*nc, unoriented=True))
    allowance = sum(p["normal_angle_bound_radians"] for p in source+target)+2*gate["pair_angle_numeric_guard_radians"]
    out.update(unoriented_included_angle_gap_radians=angle_gap, included_angle_allowance_radians=allowance)
    if angle_gap > allowance:
        out["reasons"] = ["CENTRAL_PAIR_ANGLE_INTERVALS_DISJOINT"]
        return out
    m = np.asarray(signs)[:, None]*nc
    rotation = _frame(m)@_frame(ns).T
    _require(np.isfinite(rotation).all() and np.allclose(rotation.T@rotation, np.eye(3), rtol=0.,
        atol=gate["proper_rotation_guard"]) and abs(float(np.linalg.det(rotation))-1.) <= gate["proper_rotation_guard"],
        "INVALID_PROPER_CENTRAL_ROTATION")
    residual = [_angle(rotation@ns[i], m[i]) for i in range(2)]
    allowances = [a["normal_angle_bound_radians"]+b["normal_angle_bound_radians"]+
                   gate["max_normal_alignment_extra_radians"] for a, b in zip(source, target)]
    out.update(normal_alignment_residuals_radians=residual, normal_alignment_allowances_radians=allowances)
    if any(a > b for a, b in zip(residual, allowances)):
        out["reasons"] = ["CENTRAL_NORMAL_WITNESS_OUTSIDE_DECLARED_GATE"]
        return out
    source_witness = [np.asarray(p["anchor_mm"])+p["offset_from_anchor_mm"]*ns[i] for i, p in enumerate(source)]
    b = np.asarray([signs[i]*(np.dot(nc[i], np.asarray(target[i]["anchor_mm"])-origin_c)+
            target[i]["offset_from_anchor_mm"])-np.dot(m[i], rotation@(source_witness[i]-origin_s)) for i in range(2)])
    t0 = m.T@np.linalg.solve(m@m.T, b)
    u = np.cross(m[0], m[1])
    u /= math.hypot(*u)
    backward = float(np.max(np.abs(m@t0-b)))
    _require(np.isfinite(t0).all() and np.isfinite(u).all() and math.isfinite(backward), "NONFINITE_TRANSLATION_FAMILY")
    out["anchor_backward_residual_mm"] = backward
    if backward > gate["max_anchor_backward_residual_mm"]:
        out["reasons"] = ["TWO_PLANE_ANCHOR_BACKWARD_RESIDUAL_EXCEEDS_GATE"]
        return out
    out.update(status="AVAILABLE", rotation=rotation.tolist(), translation_family=dict(
        minimum_norm_local_translation_mm=t0.tolist(), target_axial_direction=u.tolist(),
        rule="Y_EQUALS_R_X_MINUS_SOURCE_ORIGIN_PLUS_TARGET_ORIGIN_PLUS_T0_PLUS_LAMBDA_U",
        exact_plane_compatibility="ONLY_FOR_EXACT_EQUAL_ANGLE_CORRESPONDENCES",
        axial_translation_resolved=False))
    return out


def _projection(role, plane, direction, origin, constant, frame, context, work, result):
    whole = plane is None
    support = None if whole else plane["raw_vertex_rows" if role == "scan" else "triangle_rows"]
    binding = "WHOLE" if whole else plane["raw_vertex_rows_sha256" if role == "scan" else "triangle_rows_sha256"]
    buffer_name = "scan_points" if role == "scan" else "cad_triangles"
    key = (role, frame, result["input_digests"][buffer_name+"_sha256"], binding,
           tuple(float(x) for x in direction), tuple(float(x) for x in origin), float(constant))
    if key in work.cache:
        work.stats["projection_cache_hits"] += 1
        return work.cache[key]
    work.stats["projection_cache_misses"] += 1
    _require(work.stats["projection_cache_entries"] < work.cap["max_projection_cache_entries"], "PROJECTION_CACHE_BUDGET")
    count = len(context[buffer_name]) if whole else len(support)
    width = work.cap["chunk_rows"] if role == "scan" else max(1, work.cap["chunk_rows"]//3)
    low, high, low_witness, high_witness = math.inf, -math.inf, None, None
    for start in range(0, count, width):
        length = min(width, count-start)
        work.charge("scan_row_visits" if role == "scan" else "cad_vertex_visits", length if role == "scan" else 3*length)
        work.charge("index_visits", length)
        with work.temporary(length*(128 if role == "scan" else 320)):
            ids = np.arange(start, start+length, dtype=np.int64) if whole else support[start:start+length]
            points = context[buffer_name][ids].reshape((-1, 3))
            values = (points-origin)@direction+constant
            _require(np.isfinite(values).all(), "NONFINITE_CONTEXT_PROJECTION")
            left, right = int(np.argmin(values)), int(np.argmax(values))
            def witness(position):
                if role == "scan":
                    return dict(raw_vertex_row=int(ids[position]))
                triangle = int(ids[position//3])
                return dict(triangle_row=triangle, vertex_slot=position % 3,
                    face_id=int(context["cad_face_ids"][triangle]),
                    source_triangle_id=int(context["cad_source_triangle_ids"][triangle]))
            if float(values[left]) < low:
                low, low_witness = float(values[left]), witness(left)
            if float(values[right]) > high:
                high, high_witness = float(values[right]), witness(right)
            del ids, points, values
    _require(low_witness is not None and high_witness is not None, "EMPTY_CONTEXT_PROJECTION")
    record = dict(projection_id=len(result["context_projections"]), role=role,
        scope="WHOLE_SUPPLIED_BUFFER" if whole else "COMPLETE_ASSIGNED_SUPPORT", frame=frame,
        buffer_sha256=key[2], support_rows_sha256=None if whole else binding,
        projected_coordinate_origin_mm=origin.tolist(), projection_direction=direction.tolist(),
        projected_constant_mm=float(constant), interval_mm=[low, high],
        minimum_witness=low_witness, maximum_witness=high_witness,
        row_visits=count if role == "scan" else 3*count, context_not_correspondence_truth=True)
    payload = _encoded_size(key, work.cap["max_report_bytes"])+_encoded_size(record, work.cap["max_report_bytes"])
    work.memory(payload)
    work.stats["cache_payload_bytes"] += payload
    work.stats["projection_cache_entries"] += 1
    work.cache[key] = record["projection_id"]
    result["context_projections"].append(record)
    return record["projection_id"]


def _slots(family, source, target, origins, frames, context, work, result):
    rotation = np.asarray(family["rotation"])
    t0 = np.asarray(family["translation_family"]["minimum_norm_local_translation_mm"])
    u = np.asarray(family["translation_family"]["target_axial_direction"])
    direction, constant = rotation.T@u, float(np.dot(u, t0))
    intervals, votes = [], []
    for scan, cad in ((None, None), *zip(source, target)):
        ai = _projection("scan", scan, direction, origins[0], constant, frames[0], context, work, result)
        bi = _projection("cad", cad, u, origins[1], 0., frames[1], context, work, result)
        a, b = result["context_projections"][ai]["interval_mm"], result["context_projections"][bi]["interval_mm"]
        lo, hi = b[0]-a[0], b[1]-a[1]
        _require(math.isfinite(lo) and math.isfinite(hi), "NONFINITE_AXIAL_VOTE")
        intervals.append(dict(scan_projection_id=ai, cad_projection_id=bi,
            heuristic_lambda_interval_mm=[lo, hi], interval_nonempty=lo <= hi))
        votes.extend((lo, hi))
    lo, hi = intervals[0]["heuristic_lambda_interval_mm"]
    votes.append(lo+(hi-lo)/2 if lo <= hi else None)
    lo = max(p["heuristic_lambda_interval_mm"][0] for p in intervals[1:])
    hi = min(p["heuristic_lambda_interval_mm"][1] for p in intervals[1:])
    votes.append(lo+(hi-lo)/2 if lo <= hi else None)
    family["context_intervals"] = intervals
    return votes, rotation, t0, u


def _final_report(result, work):
    result["statistics"] = dict(work.stats)
    # Streaming size check: no second full JSON payload allocation. The bounded
    # fixed-schema trees cannot provide arbitrary __iter__/encoder callbacks.
    _encoded_size(result, work.cap["max_report_bytes"])
    return result


def propose_two_plane_context_v1(scan_points, scan_triangles, cad_triangles, cad_face_ids,
        cad_source_triangle_ids, scan_planes, cad_planes, ordered_matches, *,
        scan_origin_mm, cad_origin_mm, unit_assumption, source_frame, target_frame,
        budgets=TwoPlaneContextBudgetsV1()):
    """Return complete caller-schedule evidence, or NO usable proposal prefix.

    Arrays must be ordinary native float64/int64 NumPy arrays with listed shapes;
    plane/match tables must be exact immutable typed tuples. Scan support rows
    are rederived from input triangle rows; CAD supports must equal full face-ID
    membership. Plane qualification is caller-declared, not recomputed here.
    The caller owns fair scheduling and omitted-bank identities; this API cannot
    attest them. Both CAD orders count against four matches per scan pair.
    """
    result, work = _base(), None
    try:
        cap = _configuration(budgets)
        work = _Work(cap)
        work.memory()
        _require(all(type(x) is str and 0 < len(x) <= 128 and x.isascii() and x.strip()
            for x in (unit_assumption, source_frame, target_frame)), "EXPLICIT_MM_AND_FRAME_DECLARATIONS_REQUIRED")
        origins = tuple(np.asarray(_vector(x)) for x in (scan_origin_mm, cad_origin_mm))
        result.update(unit="mm", unit_assumption=unit_assumption, source_frame=source_frame,
            target_frame=target_frame, source_origin_mm=origins[0].tolist(), target_origin_mm=origins[1].tolist(),
            policy=dict(_GATES), budgets=cap,
            counter_semantics=dict(
                scan_and_cad_visits="CHARGED_BEFORE_FULL_BUFFER_VALIDATION_OR_CACHE_MISS_PROJECTION",
                index_visits="CHARGED_BINDING_AND_PROJECTION_INDEX_ELEMENTS",
                sort_work="N_TIMES_CEIL_LOG2_N_DECLARED_ABSTRACT_COMPARISON_WORK_NOT_PROFILED_OPS",
                hash_bytes="OWNED_ARRAY_SHA256_INPUT_BYTES_CHARGED_BEFORE_HASH",
                working_bytes="OWNED_ARRAY_AND_TEMP_NUMERIC_PLUS_ENCODED_CACHE_PROPOSAL_PAYLOAD_NOT_PYTHON_RSS",
                attempted_overflow="ATTEMPTED_COUNTER_CAN_EXCEED_CAP_WITH_WHOLE_REFUSAL_BEFORE_OPERATION",
                slots="PLANNED32_PER_ORDERED_MATCH_WITH_EXPLICIT_REJECTED_OR_ABSENT_OUTCOMES"))
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            context = _inputs(scan_points, scan_triangles, cad_triangles, cad_face_ids,
                cad_source_triangle_ids, work, result)
            scans, cads = _planes(scan_planes, "scan", context, work, result), _planes(cad_planes, "cad", context, work, result)
            schedule = _schedule(ordered_matches, scans, cads, cap)
            result["scheduled_pairs"] = schedule
            work.record(schedule)
            work.stats.update(signed_families_planned=4*len(schedule), candidate_slots_planned=32*len(schedule))
            poses = {}
            for assignment_id, assignment in enumerate(schedule):
                source = [scans[x] for x in assignment["scan_plane_ids"]]
                target = [cads[x] for x in assignment["cad_plane_ids"]]
                for signs in itertools.product((-1, 1), repeat=2):
                    work.stats["signed_families_attempted"] += 1
                    family = dict(signed_family_id=len(result["families"]), ordered_assignment_id=assignment_id,
                        normal_signs=list(signs), correspondences=[dict(scan_plane_id=a["plane_id"],
                            cad_plane_id=b["plane_id"], normal_sign=s) for a, b, s in zip(source, target, signs)], slots=[])
                    result["families"].append(family)
                    family.update(_family(source, target, signs, *origins))
                    available = family["status"] == "AVAILABLE"
                    if available:
                        votes, rotation, t0, u = _slots(family, source, target, origins,
                            (source_frame, target_frame), context, work, result)
                    else:
                        votes = [None]*8
                    for slot, vote in zip(_SLOTS, votes):
                        row = dict(slot=slot, status="PAIR_REJECTED", lambda_mm=None, candidate_index=None)
                        if available:
                            work.stats["slots_generation_attempted"] += 1
                            if vote is None:
                                row["status"] = "ABSENT_EMPTY_HEURISTIC_INTERVAL"
                                work.stats["absent_midpoint_slots"] += 1
                            else:
                                local = t0+vote*u
                                translation = origins[1]+local-rotation@origins[0]
                                _require(np.isfinite(local).all() and np.isfinite(translation).all(), "NONFINITE_AXIAL_POSE")
                                values = np.concatenate((rotation.ravel(), translation))
                                # Numerical equality treats signed zeros equally;
                                # no tolerance-based merge of distinct axial votes.
                                key = np.where(values == 0., 0., values).tobytes()
                                witness = dict(signed_family_id=family["signed_family_id"], slot=slot)
                                if key not in poses:
                                    index = len(result["candidates"])
                                    poses[key] = index
                                    candidate = dict(candidate_index=index, rotation=rotation.tolist(),
                                        translation_mm=translation.tolist(), anchored_local_translation_mm=local.tolist(),
                                        contributing_slots=[witness], accepted_registration=False)
                                    work.record(candidate)
                                    work.memory(len(key))
                                    work.stats["cache_payload_bytes"] += len(key)
                                    result["candidates"].append(candidate)
                                else:
                                    index = poses[key]
                                    work.record(witness)
                                    result["candidates"][index]["contributing_slots"].append(witness)
                                    work.stats["exact_pose_duplicates"] += 1
                                row.update(status="PROPOSED_SEED_ONLY", lambda_mm=float(vote), candidate_index=index)
                                work.stats["proposed_slot_count"] += 1
                        else:
                            work.stats["rejected_slot_count"] += 1
                        family["slots"].append(row)
                        work.stats["slot_records_completed"] += 1
                    work.record(family)
                    work.stats["signed_families_completed"] += 1
            _require(work.stats["signed_families_completed"] == work.stats["signed_families_planned"]
                and work.stats["slot_records_completed"] == work.stats["candidate_slots_planned"], "INCOMPLETE_SCHEDULED_SLOT_LEDGER")
            result.update(status="COMPLETE", selected_policy_evaluation_complete=True,
                usable_proposals=bool(result["candidates"]), reasons=[])
        return _final_report(result, work)
    except Exception as error:
        # No central rotation/translation family, lambda or candidate survives a
        # late malformed/budget/numerical failure. Retain bounded attempted work;
        # no stringification of arbitrary caller exceptions enters the report.
        refusal = _base()
        refusal.update(reasons=[str(error) if type(error) is _Unavailable else "CONTEXT_HELPER_EXCEPTION"],
            evidence_scope="ATTEMPTED_COUNTERS_ONLY_NO_PARTIAL_POSE_OR_GEOMETRY_REPORT",
            discarded_signed_families=len(result["families"]), discarded_candidates=len(result["candidates"]),
            statistics={} if work is None else dict(work.stats))
        _require(len(_json(refusal)) <= 4096, "INTERNAL_REFUSAL_ENVELOPE_BUDGET")
        return refusal
