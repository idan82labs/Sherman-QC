"""Strict owned admission of V7 plus seeded proposals to the existing selector.

The V7 codec is only a structural envelope: neither lane is relabelled as V7.
Every qualified origin survives, including cross-lane duplicate supports. One
combined context lets the unchanged selector apply its noncap/unknown conflict
veto and complete raw-triangle incidence rule across BOTH lanes.

This binds already-completed in-process extraction evidence, not a signature or
independent proof of the reported fits, seed graph or physical-plane premise.
There is no extraction, refitting, registration, measurement, CAD, IO or count
inference here. Epsilon is caller-owned digest evidence, not recalibration.
Metadata/hash/array work is bounded; Python allocation and wall time still need
an outer supervisor. No context or favorable prefix is returned after failure.
"""

import hashlib
import json
import math

import numpy as np

from . import cad_guided_context_surface_roles_v1 as _roles
from . import cad_guided_noisy_mesh_planes_v5 as _v5
from . import cad_guided_noisy_mesh_planes_v7 as _codec
from . import cad_guided_seeded_plane_proposals_v1 as _seeded


_COMPOSITE_MAXIMUM = dict(max_qualified_planes=65_552,
    max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
    max_full_support_row_tests=64_000_000, max_raw_predicate_visits=48_000_000,
    max_triangle_predicate_visits=160_000_000, max_edge_predicate_visits=192_000_000,
    max_fit_evaluations=65_584, max_retained_array_bytes=536_870_912,
    max_metadata_bytes=67_108_864)
_ORIGIN_FIELDS = {"plane_id", "lane", "kind", "record_id",
                  "triangle_rows_sha256", "raw_vertex_rows_sha256"}
_require = _roles._require


def _same_metadata(native, decoded):
    """Strict full-tree binding, including descriptors of rejected supports.

    Mirrors only the codec's tuple/list and exact np.float64 normalization.
    No array descriptor is inverted. Array hashing is charged before reading
    its buffer; custom scalar/equality hooks and bool/int aliases are refused.
    """
    visits = hashed = 0

    def compare(value, wire, depth=0):
        nonlocal visits, hashed
        visits += 1
        _require(depth <= 64 and visits <= 8_000_000,
                 "COMPOSITE_NATIVE_METADATA_STRUCTURE_EXCEEDED")
        if type(value) is np.float64:
            value = float(value)
        if type(value) is np.ndarray:
            _require(value.dtype.kind in "biuf" and value.flags.c_contiguous,
                     "INVALID_COMPOSITE_NATIVE_ARRAY")
            hashed += value.nbytes
            _require(hashed <= 4 * 1024**3, "COMPOSITE_ARRAY_HASH_WORK_EXCEEDED")
            expected = dict(shape=list(value.shape), dtype=value.dtype.str,
                            sha256=_roles._sha(value))
            compare(expected, wire, depth)
        elif type(value) is dict:
            _require(type(wire) is dict and len(value) <= 128
                     and all(type(k) is str and len(k) <= 128 for k in value)
                     and tuple(value) == tuple(wire), "STALE_COMPOSITE_METADATA_KEYS")
            for key in value:
                compare(value[key], wire[key], depth + 1)
        elif type(value) in (tuple, list):
            _require(type(wire) is list and len(value) == len(wire)
                     and len(value) <= 8_000_000, "STALE_COMPOSITE_METADATA_LIST")
            for a, b in zip(value, wire):
                compare(a, b, depth + 1)
        else:
            _require(value is None or type(value) in (str, bool, int, float),
                     "INVALID_COMPOSITE_NATIVE_SCALAR")
            _require(type(value) is type(wire), "STALE_COMPOSITE_SCALAR_TYPE")
            if type(value) is float:
                _require(math.isfinite(value) and value.hex() == wire.hex(),
                         "STALE_COMPOSITE_FLOAT")
            else:
                _require((type(value) is not str or len(value) <= 65_536)
                         and (type(value) is not int or value.bit_length() <= 4096)
                         and value == wire, "STALE_COMPOSITE_SCALAR")

    compare(native, decoded)


def _counts(stats, expected, reason):
    _require(type(stats) is dict and all(type(stats.get(k)) is int and stats[k] == v
             for k, v in expected.items()), reason)


def _complete(value, schema, binding):
    _require(type(value) is dict and value.get("schema_version") == schema
             and value.get("status") == "COMPLETE" and value.get("input_digests") == binding
             and value.get("unit") == "mm" and value.get("coordinate_frame") == "RAW_SCAN_UNREGISTERED"
             and value.get("raw_coordinate_welding") is False and value.get("residual_trimming") is False
             and value.get("production_use_allowed") is False and value.get("accepted_registration") is False
             and value.get("observed_bend_count", False) is None
             and value.get("physical_plane_count", False) is None and value.get("search_exhaustive") is False,
             "UNAVAILABLE_OR_UNSUPPORTED_COMPOSITE_LANE")


def _lane_envelope(value, cap, work):
    # Re-encoding the detached descriptor tree exactly preserves the original
    # lane codec representation, without a second read/hash of native arrays.
    envelope, _, expanded = _codec._envelope(value, work, cap.max_encoded_metadata_bytes)
    _require(expanded <= cap.max_decoded_metadata_bytes, "COMPOSITE_LANE_EXPANSION_EXCEEDED")
    digest, size = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False,
                                      separators=(",", ":")).iterencode(envelope):
        blob = fragment.encode("utf-8")
        size += len(blob)
        _require(size <= cap.max_encoded_metadata_bytes, "COMPOSITE_LANE_ENCODING_EXCEEDED")
        digest.update(blob)
    return envelope, digest.hexdigest()


def _policies_and_work(meta):
    """Validate declarations and all-work sums, distinct from role memberships."""
    first, second = meta["lanes"]["v7"], meta["lanes"]["seeded_v1"]
    policy = first["policy"]
    _require(type(policy) is dict, "INVALID_COMPOSITE_QUALIFICATION_POLICY")
    _same_metadata(policy, second.get("qualification_policy"))
    copied = dict(policy)
    _require(type(copied.get("tile_triangles")) is list, "INVALID_COMPOSITE_TILING_POLICY")
    copied["tile_triangles"] = tuple(copied["tile_triangles"])
    gate, a_cap = _v5._configuration(_v5.NoisyMeshPlanePolicyV5(**copied),
        _v5.NoisyMeshPlaneBudgetsV5(**first["budgets"]))
    b_cap = _seeded._configuration(_seeded.SeededPlaneBudgetsV1(**second["budgets"]))
    _same_metadata(dict(v7=a_cap, seeded_v1=b_cap), meta.get("effective_lane_budgets"))
    cap = meta.get("budgets")
    _require(type(cap) is dict and set(cap) == set(_COMPOSITE_MAXIMUM)
             and all(type(v) is int and 0 <= v <= _COMPOSITE_MAXIMUM[k] for k, v in cap.items()),
             "INVALID_COMPOSITE_WORK_BUDGETS")
    a, b = first["statistics"], second["statistics"]
    for stats, limits in ((a, a_cap), (b, b_cap)):
        for name in ("triangle_memberships", "raw_memberships", "full_support_row_tests",
                     "fit_evaluations", "retained_array_bytes", "metadata_bytes"):
            _require(type(stats.get(name)) is int and 0 <= stats[name] <= limits["max_" + name],
                     "INVALID_COMPOSITE_LANE_WORK")
    totals = {k: a[k] + b[k] for k in ("triangle_memberships", "raw_memberships",
              "full_support_row_tests", "fit_evaluations", "retained_array_bytes")}
    for field, items in (
            ("raw_predicate_visits", ((a, a_cap, "cone_row_tests"), (b, b_cap, "proposal_raw_row_visits"))),
            ("triangle_predicate_visits", ((a, a_cap, "closure_triangle_tests"),
                (b, b_cap, "proposal_triangle_visits"), (b, b_cap, "seed_selection_triangle_visits"))),
            ("edge_predicate_visits", ((a, a_cap, "closure_edge_tests"), (b, b_cap, "proposal_edge_visits")))):
        total = 0
        for stats, limits, name in items:
            _require(type(stats.get(name)) is int and 0 <= stats[name] <= limits["max_" + name],
                     "INVALID_COMPOSITE_PREDICATE_WORK")
            total += stats[name]
        totals[field] = total
    _counts(meta["statistics"], totals, "STALE_COMPOSITE_ALL_WORK_SUMS")
    _require(all(v <= cap["max_" + k] for k, v in totals.items()), "COMPOSITE_ALL_WORK_CAP_EXCEEDED")
    return gate, b_cap, cap


def _support(original, wire, triangles, n, m):
    rows, vertices = original["triangle_rows"], original["raw_vertex_rows"]
    _require(type(rows) is np.ndarray and rows.ndim == 1 and 0 < len(rows) <= m
             and type(vertices) is np.ndarray and vertices.ndim == 1 and 3 <= len(vertices) <= n,
             "INVALID_SEEDED_NATIVE_SUPPORT")
    r = _roles._owned(rows, (len(rows),), "<i8", "COMPOSITE_TRIANGLE_ROWS")
    v = _roles._owned(vertices, (len(vertices),), "<i8", "COMPOSITE_RAW_ROWS")
    _require(np.all(r >= 0) and np.all(r < m) and np.all(r[1:] > r[:-1])
             and np.all(v >= 0) and np.all(v < n) and np.all(v[1:] > v[:-1]),
             "INVALID_COMPOSITE_SUPPORT_IDS")
    rd, vd = wire["triangle_rows_sha256"], wire["raw_vertex_rows_sha256"]
    _require(_roles._hex(rd) and _roles._hex(vd), "INVALID_COMPOSITE_SUPPORT_DIGEST")
    _roles._descriptor(r, wire["triangle_rows"], rd, "COMPOSITE_TRIANGLE")
    _roles._descriptor(v, wire["raw_vertex_rows"], vd, "COMPOSITE_RAW")
    _require(np.array_equal(v, np.unique(triangles[r].ravel())), "INCOMPLETE_COMPOSITE_RAW_UNION")
    _counts(wire["analysis"], dict(raw_support_count=len(v)), "STALE_COMPOSITE_ANALYSIS_SUPPORT_COUNT")
    return r, v, rd, vd


def _retained_support_preflight(native, n, m, cap):
    """Bound validation of all retained records, not just published proposals.

    The original producer's logical work counters can include construction and
    repeated tests. These are separate literal retained-array membership sums.
    Qualified V7 supports are checked by its existing admission; rejected V7
    and all seeded supports also receive exact original raw-union validation.
    """
    triangles = vertices = 0
    for lane, tables in (("v7", ("cores", "closures")), ("seeded_v1", ("regions",))):
        for table in tables:
            records = native["lanes"][lane][table]
            _require(type(records) is list and len(records) <= 65_552,
                     "INVALID_COMPOSITE_RETAINED_TABLE")
            for record in records:
                _require(type(record) is dict, "INVALID_COMPOSITE_RETAINED_RECORD")
                rows, raw = record["triangle_rows"], record["raw_vertex_rows"]
                _require(type(rows) is np.ndarray and rows.ndim == 1 and 0 < len(rows) <= m
                    and type(raw) is np.ndarray and raw.ndim == 1 and 3 <= len(raw) <= n,
                    "INVALID_COMPOSITE_RETAINED_SUPPORT")
                triangles += len(rows)
                vertices += len(raw)
                _require(triangles <= cap.max_triangle_memberships and vertices <= cap.max_raw_memberships,
                         "COMPOSITE_ALL_RETAINED_VALIDATION_CAP_EXCEEDED")


def _validate_rejected_v7(native, wire, prepared):
    for table, id_key in (("cores", "core_id"), ("closures", "closure_id")):
        for index, row in enumerate(wire[table]):
            _require(type(row.get(id_key)) is int and row[id_key] == index
                and type(row.get("analysis")) is dict
                and row["analysis"].get("status") in ("PROPOSED", "REJECTED"),
                "INVALID_COMPOSITE_RETAINED_V7_RECORD")
            if row["analysis"]["status"] == "REJECTED":
                _support(native[table][index], row, prepared._triangles,
                         prepared.raw_row_count, prepared.triangle_count)


def _seeded_supports(native, wire, prepared, gate, seed_cap, cap):
    n, m = prepared.raw_row_count, prepared.triangle_count
    schedule, seeds, regions = wire["scheduled_seed_triangle_rows"], wire["seeds"], wire["regions"]
    _require(type(schedule) is list and len(schedule) == min(seed_cap["max_seeds"], m)
             and all(type(i) is int and 0 <= i < m for i in schedule)
             and len(set(schedule)) == len(schedule) and type(seeds) is list and len(seeds) == len(schedule)
             and type(regions) is list and len(regions) <= len(schedule), "INCOMPLETE_SEEDED_SCHEDULE")
    _counts(wire, dict(unscheduled_seed_triangle_count=m - len(schedule)), "STALE_UNSCHEDULED_SEED_COUNT")
    _require(wire.get("qualification") == "UNCHANGED_V5_ALL_RAW_ID_FIXED_NORMAL_WITNESS_AND_CONDITIONAL_CONE"
             and wire.get("uncertainty_status") == "STIPULATED_UNCALIBRATED"
             and wire.get("finite_support_representation") == "ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON"
             and wire["statistics"].get("selected_policy_evaluation_complete") is True,
             "INCOMPLETE_SEEDED_QUALIFICATION")
    all_triangles = all_raw = 0
    for index, row in enumerate(regions):
        _require(type(row) is dict and type(row.get("region_id")) is int and row["region_id"] == index
                 and type(row.get("analysis")) is dict
                 and row["analysis"].get("status") in ("PROPOSED", "REJECTED"), "INVALID_SEEDED_REGION")
        original = native["regions"][index]
        for name, maximum in (("triangle_rows", m), ("raw_vertex_rows", n)):
            array = original[name]
            _require(type(array) is np.ndarray and array.ndim == 1 and 0 < len(array) <= maximum,
                     "INVALID_SEEDED_SUPPORT_PREFLIGHT")
        all_triangles += len(original["triangle_rows"])
        all_raw += len(original["raw_vertex_rows"])
    # Validate the complete retained seeded denominator before any support copy.
    _require(all_triangles <= cap.max_triangle_memberships and all_raw <= cap.max_raw_memberships,
             "SEEDED_ALL_RECORD_VALIDATION_CAP_EXCEEDED")
    expected_refs = [dict(kind="SEEDED_REGION", record_id=i) for i, row in enumerate(regions)
                     if row["analysis"]["status"] == "PROPOSED"]
    _require(wire["proposals"] == expected_refs, "INCOMPLETE_SEEDED_QUALIFIED_TABLE")
    _counts(wire["statistics"], dict(input_raw_rows=n, input_triangles=m,
        completed_seed_count=len(seeds), unique_region_count=len(regions), fit_evaluations=len(regions),
        qualified_region_count=len(expected_refs), triangle_memberships=all_triangles,
        raw_memberships=all_raw, full_support_row_tests=all_raw), "STALE_SEEDED_DENOMINATORS")
    seen = set()
    for seed_id, seed in zip(schedule, seeds):
        _require(type(seed) is dict and type(seed.get("seed_triangle_row")) is int
                 and seed["seed_triangle_row"] == seed_id and type(seed.get("region_id")) is int
                 and 0 <= seed["region_id"] < len(regions)
                 and seed.get("normal_is_uncertainty_bound") is False
                 and seed.get("threshold_scope") == "FIXED_SEED_PROPOSAL_ONLY_NOT_POSITION_UNCERTAINTY_ENCLOSURE",
                 "INVALID_SEEDED_COMPLETION_RECORD")
        identifier = seed["region_id"]
        row = regions[identifier]
        _require(seed.get("disposition") == ("EXACT_SUPPORT_DUPLICATE" if identifier in seen
                 else row["analysis"]["status"]), "STALE_SEEDED_DISPOSITION")
        if identifier not in seen:
            _require(identifier == len(seen) and type(row.get("first_seed_triangle_row")) is int
                     and row["first_seed_triangle_row"] == seed_id, "STALE_SEEDED_FIRST_ORIGIN")
        seen.add(identifier)
    _require(len(seen) == len(regions), "UNREFERENCED_SEEDED_REGION")
    supports = {}
    for i, row in enumerate(regions):
        r, v, rd, vd = _support(native["regions"][i], row, prepared._triangles, n, m)
        for seed in seeds:
            if seed["region_id"] == i:
                index = int(np.searchsorted(r, seed["seed_triangle_row"]))
                _require(index < len(r) and int(r[index]) == seed["seed_triangle_row"],
                         "SEED_OUTSIDE_BOUND_SUPPORT")
                _counts(seed, dict(seed_connected_triangle_count=len(r)), "STALE_SEED_CONNECTED_COUNT")
                _require(type(seed.get("compatible_triangle_count")) is int
                         and len(r) <= seed["compatible_triangle_count"] <= m,
                         "INVALID_SEED_COMPATIBLE_COUNT")
        if row["analysis"]["status"] == "PROPOSED":
            normal, alpha, count = _roles._analysis(row["analysis"])
            _require(count == len(v) and count >= gate["min_support_rows"]
                     and alpha <= gate["max_normal_angle_radians"], "SEEDED_QUALIFICATION_GATE_MISMATCH")
            supports[i] = _roles._Support("seeded_v1:SEEDED_REGION", i, r, rd, vd, count, normal, alpha)
    accounting = wire["raw_input_accounting"]
    original = native["raw_input_accounting"]["unreferenced_raw_vertex_rows"]
    expected = np.flatnonzero(~prepared._referenced)
    _require(type(original) is np.ndarray and original.shape == expected.shape,
             "STALE_SEEDED_UNREFERENCED_ROWS")
    owned = _roles._owned(original, expected.shape, "<i8", "UNREFERENCED_RAW_ROWS")
    _require(np.array_equal(owned, expected)
             and accounting.get("unreferenced_disposition") == "RETAINED_NOT_TRIANGLE_SUPPORT",
             "INCOMPLETE_SEEDED_INPUT_ACCOUNTING")
    _counts(accounting, dict(referenced_raw_row_count=n - len(expected)), "STALE_SEEDED_INPUT_COUNT")
    return supports


def prepare_composite_surface_roles_v1(raw_points, raw_triangle_rows, composite_patches,
        encoded_composite_evidence, epsilon_digest, *, budgets=_roles.ContextSurfaceRoleBudgetsV1()):
    """Bind a complete native composite to one immutable existing role context.

    Support.kind is ``lane:original_kind`` and Support.record_id is the combined
    plane_id. The full metadata retains original kind/id, lane and lane digests.
    The existing unscheduled-core field remains V7-only, not a seeded coverage
    assertion. Unknown competing supports from either lane remain in selection.
    """
    try:
        cap = _roles._configuration(budgets)
        _require(type(raw_points) is np.ndarray and raw_points.ndim == 2 and raw_points.shape[1:] == (3,)
                 and type(raw_triangle_rows) is np.ndarray and raw_triangle_rows.ndim == 2
                 and raw_triangle_rows.shape[1:] == (3,), "INVALID_COMPOSITE_INPUT_SHAPES")
        n, m = len(raw_points), len(raw_triangle_rows)
        _require(3 <= n <= cap.max_raw_rows and 1 <= m <= cap.max_triangles
                 and 3 * m <= cap.max_incidence_corner_visits, "COMPOSITE_INPUT_CAP_EXCEEDED")
        p = _roles._owned(raw_points, (n, 3), "<f8", "COMPOSITE_POINTS")
        t = _roles._owned(raw_triangle_rows, (m, 3), "<i8", "COMPOSITE_TRIANGLES")
        _require(_roles._hex(epsilon_digest), "INVALID_COMPOSITE_EPSILON_DIGEST")
        meta, meta_sha, encoded_size, expanded_size = _roles._encoded_snapshot(encoded_composite_evidence, cap)
        _same_metadata(composite_patches, meta)
        binding = dict(points_sha256=_roles._sha(p), triangle_raw_rows_sha256=_roles._sha(t),
                       position_error_bounds_mm_sha256=epsilon_digest)
        _complete(meta, "composite_scan_planes_v1", binding)
        _require(type(meta.get("lanes")) is dict and tuple(meta["lanes"]) == ("v7", "seeded_v1")
                 and meta.get("support_id_namespace") == "COMPOSITE_PLANE_ID_WITH_EXPLICIT_LANE_KIND_RECORD_ORIGIN"
                 and type(meta.get("statistics")) is dict
                 and meta["statistics"].get("selected_policy_evaluation_complete") is True,
                 "INCOMPLETE_COMPOSITE_SOURCE")
        for lane, schema in (("v7", "noisy_mesh_planes_v7"), ("seeded_v1", "seeded_plane_proposals_v1")):
            _complete(meta["lanes"][lane], schema, binding)
        gate, seed_cap, source_cap = _policies_and_work(meta)
        _retained_support_preflight(composite_patches, n, m, cap)
        _require(encoded_size <= source_cap["max_metadata_bytes"], "COMPOSITE_DECLARED_METADATA_CAP_EXCEEDED")
        _counts(meta["statistics"], dict(metadata_bytes=encoded_size, metadata_expanded_bytes=expanded_size),
                "STALE_COMPOSITE_METADATA_BYTE_COUNTS")
        _counts(meta["lanes"]["v7"]["statistics"], dict(input_raw_rows=n, input_triangles=m),
                "STALE_COMPOSITE_V7_INPUT_COUNTS")
        origins, triangle_count, raw_count = [], 0, 0
        for lane in ("v7", "seeded_v1"):
            refs = meta["lanes"][lane]["proposals"]
            _require(type(refs) is list and len(refs) <= cap.max_proposals, "INVALID_COMPOSITE_PROPOSALS")
            for ref in refs:
                kinds = ("CORE", "CLOSURE") if lane == "v7" else ("SEEDED_REGION",)
                _require(type(ref) is dict and set(ref) == {"kind", "record_id"}
                         and type(ref["kind"]) is str and ref["kind"] in kinds
                         and type(ref["record_id"]) is int and ref["record_id"] >= 0,
                         "INVALID_COMPOSITE_ORIGIN_REFERENCE")
                table = ("cores" if ref["kind"] == "CORE" else "closures") if lane == "v7" else "regions"
                record = composite_patches["lanes"][lane][table][ref["record_id"]]
                for field in ("triangle_rows", "raw_vertex_rows"):
                    _require(type(record[field]) is np.ndarray and record[field].ndim == 1,
                             "INVALID_COMPOSITE_QUALIFIED_ARRAY")
                triangle_count += len(record["triangle_rows"])
                raw_count += len(record["raw_vertex_rows"])
                origins.append(dict(plane_id=len(origins), lane=lane, kind=ref["kind"], record_id=ref["record_id"],
                    triangle_rows_sha256=record["triangle_rows_sha256"],
                    raw_vertex_rows_sha256=record["raw_vertex_rows_sha256"]))
        _require(len(origins) <= min(cap.max_proposals, source_cap["max_qualified_planes"])
                 and triangle_count <= cap.max_triangle_memberships and raw_count <= cap.max_raw_memberships,
                 "COMPOSITE_QUALIFIED_MEMBERSHIP_CAP_EXCEEDED")
        planes = meta.get("planes")
        _require(type(planes) is list and len(planes) == len(origins)
                 and all(type(row) is dict and set(row) == _ORIGIN_FIELDS
                         and type(row["plane_id"]) is int and type(row["record_id"]) is int
                         and row == expected for row, expected in zip(planes, origins)),
                 "INCOMPLETE_OR_STALE_COMPOSITE_ORIGIN_TABLE")
        _counts(meta["statistics"], dict(input_raw_rows=n, input_triangles=m, qualified_plane_count=len(origins),
            qualified_triangle_memberships=triangle_count, qualified_raw_memberships=raw_count),
            "STALE_COMPOSITE_QUALIFIED_COUNTS")
        digests = meta.get("lane_metadata_sha256")
        _require(type(digests) is dict and set(digests) == {"v7", "seeded_v1"}
                 and all(_roles._hex(x) for x in digests.values()), "INVALID_COMPOSITE_LANE_DIGESTS")
        work, v7_envelope = _codec._Work(), None
        for lane in ("v7", "seeded_v1"):
            envelope, digest = _lane_envelope(meta["lanes"][lane], cap, work)
            _require(digest == digests[lane], "STALE_COMPOSITE_LANE_DIGEST")
            if lane == "v7":
                v7_envelope = envelope
        prepared = _roles.prepare_context_surface_roles_v1(p, t, composite_patches["lanes"]["v7"],
            v7_envelope, epsilon_digest, budgets=cap)
        _validate_rejected_v7(composite_patches["lanes"]["v7"], meta["lanes"]["v7"], prepared)
        seeded = _seeded_supports(composite_patches["lanes"]["seeded_v1"], meta["lanes"]["seeded_v1"],
                                  prepared, gate, seed_cap, cap)
        old = {(row.kind, row.record_id): row for row in prepared._supports}
        supports = []
        for row in planes:
            source = old[(row["kind"], row["record_id"])] if row["lane"] == "v7" else seeded[row["record_id"]]
            _require(source.angle_bound_radians <= gate["max_normal_angle_radians"]
                     and source.raw_support_count >= gate["min_support_rows"], "COMPOSITE_CONE_POLICY_MISMATCH")
            supports.append(_roles._Support(row["lane"] + ":" + row["kind"], row["plane_id"],
                source.triangle_rows, source.triangle_rows_sha256, source.raw_vertex_rows_sha256,
                source.raw_support_count, source.normal, source.angle_bound_radians))
        return _roles.PreparedContextSurfaceRolesV1(binding["points_sha256"], binding["triangle_raw_rows_sha256"],
            epsilon_digest, meta_sha, n, m, triangle_count, raw_count, encoded_size, expanded_size,
            prepared.unscheduled_qualified_core_count, prepared._triangles, prepared._referenced,
            tuple(supports), cap, _roles._TOKEN)
    except _roles.ContextSurfaceRolesUnavailableV1:
        raise
    except Exception as error:
        raise _roles.ContextSurfaceRolesUnavailableV1("COMPOSITE_ROLE_PREPARATION_FAILURE") from error
