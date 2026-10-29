"""Opt-in nominal V2 proposals with real V3 final queries; no workflow entrypoint.

This module does not choose starts, accept poses or measure bends. A caller must
complete the unchanged V3 diagnostics and connected-context gates for EVERY
declared candidate before publication. V2 soup-wide conditioning and changed
support/trajectory are intentional limitations; no fallback/recentering occurs.

Payload reservations below are conservative source-level numeric/logical buffer
envelopes, not an allocator/RSS guarantee. External CPU/wall/RSS supervision is
still required. Returned original arrays are weak-reference tracked; arbitrary
caller copies and unrelated process allocations are outside this accounting.
All frozen V2/V3 functions and module globals remain unchanged.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import weakref

import numpy as np

from . import cad_guided_global_registration_v1 as _v1
from . import cad_guided_global_registration_v2 as _v2
from . import cad_guided_global_registration_v3 as _v3
from . import cad_guided_triangle_surface_v2 as _s2
from . import cad_guided_triangle_surface_v3 as _s3


SCHEMA = "nominal_proposal_optimizer_v4"
PROPOSAL_MODEL = "V2_NOMINAL_FINITE_TRIANGLE_ENGINEERING_NOT_INTERVAL_EVIDENCE"
PROPOSAL_SUPPORT = "V2_NOMINAL_DISTANCE_WITH_ENGINEERING_AMBIGUITY_EXCLUSION"
FINAL_RANK_SCOPE = "V3_ENDPOINT_RECOMPUTED_NOT_LAST_V2_PROPOSAL_WEIGHTS"
_TOKEN = object()
_COMMON = ("node_visits", "triangle_pairs", "certificate_pairs", "normal_pairs", "work_units")
_V2_COUNTS = ("node_visits", "triangle_pairs", "normal_pairs", "work_units", "passes_completed",
    "maximum_pair_batch", "tie_batches_completed", "maximum_tie_storage_rows")
_SURFACE_MAXIMA = (
    ("max_triangles", 500000), ("max_query_rows", 200000), ("leaf_size", 16),
    ("max_build_nodes", 131071), ("max_build_triangle_visits", 10000000),
    ("max_node_visits", 32000000), ("max_triangle_pairs", 32000000),
    ("max_normal_pairs", 32000000), ("max_work_units", 512000000),
    ("max_pair_batch", 65536), ("max_ties_per_query", 32),
)
_RESOURCE_MAXIMA = (
    ("max_build_nodes", 131071), ("max_build_triangle_visits", 10000000),
    ("max_owned_input_bytes", 268435456), ("max_index_array_bytes", 536870912),
    ("max_live_payload_bytes", 2147483648), ("max_binding_bytes", 536870912),
    ("max_export_reference_tests", 8000000), ("max_adapter_scalar_tests", 4000000000),
    ("max_proposal_calls", 96), ("max_proposal_input_rows", 1920000),
    ("max_report_bytes", 1048576),
)


@dataclass(frozen=True)
class ProposalOptimizerResourceBudgetsV4:
    max_build_nodes: int = 131071
    max_build_triangle_visits: int = 10000000
    max_owned_input_bytes: int = 268435456
    max_index_array_bytes: int = 536870912
    max_live_payload_bytes: int = 2147483648
    max_binding_bytes: int = 536870912
    max_export_reference_tests: int = 8000000
    max_adapter_scalar_tests: int = 4000000000
    max_proposal_calls: int = 96
    max_proposal_input_rows: int = 1920000
    max_report_bytes: int = 1048576


class _OptimizerFailureV4(_v3._SurfaceFailure):
    """Fatal session failure; the caller's V3 fatal-query path must invalidate all candidates."""


def _require(condition, reason):
    if not condition:
        raise _OptimizerFailureV4(reason)


def _surface_configuration(value, version):
    kind = _s2.TriangleSurfaceBudgets if version == 2 else _s3.TriangleSurfaceBudgets
    maxima = _SURFACE_MAXIMA + (("max_certificate_pairs", 128000000),) if version == 3 else _SURFACE_MAXIMA
    _require(type(value) is kind, "INVALID_TYPED_SURFACE_BUDGET")
    copied = {key: getattr(value, key) for key, _ in maxima}
    _require(all(type(copied[key]) is int and 0 <= copied[key] <= cap for key, cap in maxima)
        and copied["leaf_size"] >= 8 and copied["max_pair_batch"] >= 1 and copied["max_ties_per_query"] >= 1,
        "INVALID_SURFACE_BUDGET_VALUES")
    return kind(**copied)


def _resource_configuration(value):
    _require(type(value) is ProposalOptimizerResourceBudgetsV4, "INVALID_TYPED_RESOURCE_BUDGET")
    copied = {key: getattr(value, key) for key, _ in _RESOURCE_MAXIMA}
    _require(all(type(copied[key]) is int and 0 <= copied[key] <= cap for key, cap in _RESOURCE_MAXIMA)
        and copied["max_report_bytes"] >= 4096, "INVALID_RESOURCE_BUDGET_VALUES")
    return ProposalOptimizerResourceBudgetsV4(**copied)


def _counts():
    return dict(attempted_dispatches=0, attempted_rows=0, query_calls=0, query_rows=0,
        completed_query_calls=0, completed_query_rows=0, node_visits=0, triangle_pairs=0,
        certificate_pairs=0, normal_pairs=0, triangle_box_pairs=0, work_units=0)


def _encode(value, maximum):
    blocks, size = [], 0
    for fragment in json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":")).iterencode(value):
        block = fragment.encode("utf-8")
        size += len(block)
        _require(size <= maximum, "OPTIMIZER_EVIDENCE_BYTE_BUDGET")
        blocks.append(block)
    return b"".join(blocks)


class _Work:
    def __init__(self, resources, combined):
        self.resources, self.combined = resources, combined
        self.phase = {name: _counts() for name in ("proposal", "final", "combined")}
        self.stats = dict(build_attempts=0, completed_builds=0, build_nodes=0, build_triangle_visits=0,
            owned_input_bytes=0, index_array_bytes=0, retained_identity_bytes=0, binding_bytes=0,
            export_reference_tests=0, adapter_scalar_tests=0, proposal_calls=0, proposal_input_rows=0,
            proposal_numerical_failures=0, temporary_bytes=0, exported_nonweakref_bytes=0,
            live_export_array_bytes=0, working_peak_bytes=0)
        self.exports = []
        self.unknown_work = False
        self.unknown_payload = False

    def charge(self, key, amount):
        _require(type(amount) is int and amount >= 0, "INVALID_ADAPTER_WORK_CHARGE")
        self.stats[key] += amount
        _require(self.stats[key] <= getattr(self.resources, "max_"+key), "OPTIMIZER_"+key.upper()+"_BUDGET")

    def live_exports(self):
        retained, live = [], 0
        for refs, size in self.exports:
            self.charge("export_reference_tests", len(refs))
            # Keep the full allocation until EVERY original array is dead. A
            # retained slice/base chain keeps its original allocation alive.
            if any(reference() is not None for reference in refs):
                retained.append((refs, size))
                live += size
        self.exports = retained
        self.stats["live_export_array_bytes"] = live
        return live

    def memory(self, extra=0):
        live = (self.stats["owned_input_bytes"]+self.stats["index_array_bytes"]
            + self.stats["retained_identity_bytes"]+self.stats["temporary_bytes"]
            + self.stats["exported_nonweakref_bytes"]+self.live_exports()+1048576+extra)
        self.stats["working_peak_bytes"] = max(self.stats["working_peak_bytes"], live)
        _require(live <= self.resources.max_live_payload_bytes, "OPTIMIZER_LIVE_PAYLOAD_BUDGET")

    @contextmanager
    def temporary(self, amount):
        self.stats["temporary_bytes"] += amount
        try:
            self.memory()
            yield
        finally:
            self.stats["temporary_bytes"] -= amount

    def exported(self, result, row_count):
        arrays = {id(value): value for value in result.values() if type(value) is np.ndarray}
        refs = tuple(weakref.ref(value) for value in arrays.values())
        size = sum(value.nbytes for value in arrays.values())
        # Tuples/dicts cannot be weakly referenced. Charge their conservative
        # payload cumulatively, even after a caller may have released them.
        self.stats["exported_nonweakref_bytes"] += 8*row_count+16384+256*len(refs)
        self.exports.append((refs, size))
        self.memory()

    def digest_array(self, value):
        self.charge("binding_bytes", value.nbytes)
        return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _snapshot_array(value, trailing, maximum, magnitude, work, name):
    _require(type(value) is np.ndarray and value.dtype == np.dtype(np.float64)
        and value.ndim == len(trailing)+1 and value.shape[1:] == trailing and len(value) <= maximum,
        "INVALID_"+name)
    shape, expected = tuple(value.shape), int(value.size)*8
    work.charge("adapter_scalar_tests", int(value.size))
    blob = value.tobytes(order="C")
    _require(len(blob) == expected, "CHANGED_"+name+"_SNAPSHOT")
    owned = np.frombuffer(blob, dtype=np.float64).reshape(shape)
    _require(owned.shape == shape and owned.nbytes == expected and len(owned) <= maximum,
        "INVALID_OWNED_"+name)
    absolute = np.abs(owned)
    _require(np.isfinite(owned).all() and np.all(absolute <= magnitude)
        and not np.any((absolute > 0) & (absolute < np.finfo(float).tiny)), "UNSUPPORTED_"+name)
    return owned


def _snapshot_faces(values, count, work):
    if type(values) is np.ndarray:
        _require(values.ndim == 1 and values.shape == (count,) and values.dtype.kind in "iu"
            and values.dtype.itemsize <= 8, "INVALID_SOURCE_FACE_ARRAY")
        dtype, expected = values.dtype, int(values.nbytes)
        blob = values.tobytes(order="C")
        _require(len(blob) == expected, "CHANGED_SOURCE_FACE_SNAPSHOT")
        copied = np.frombuffer(blob, dtype=dtype)
        _require(len(copied) == count, "CHANGED_SOURCE_FACE_COUNT")
        rows = tuple(int(value) for value in copied)
    else:
        _require(type(values) in (list, tuple) and len(values) == count, "INVALID_SOURCE_FACE_IDENTITIES")
        rows = tuple(values)
        _require(len(rows) == count, "CHANGED_SOURCE_FACE_COUNT")
    result, logical_bytes, digest = [], 0, hashlib.sha256()
    for value in rows:
        _require((type(value) is int and 0 <= value <= 2**53)
            or (type(value) is str and 0 < len(value) <= 128), "INVALID_SOURCE_FACE_IDENTITY")
        work.charge("adapter_scalar_tests", len(value) if type(value) is str else 1)
        if type(value) is str:
            _require(all(32 <= ord(character) <= 126 for character in value), "INVALID_SOURCE_FACE_STRING")
            block = b"s:"+str(len(value)).encode("ascii")+b":"+value.encode("ascii")+b";"
        else:
            block = b"i:"+str(value).encode("ascii")+b";"
        work.charge("binding_bytes", len(block))
        logical_bytes += 16+len(block)
        digest.update(block)
        result.append(value)
    return tuple(result), logical_bytes, digest.hexdigest()


def _index_arrays(version):
    common = dict(triangles=("f", (3, 3)), normals=("f", (3,)), edge_vectors=("f", (3, 3)),
        edge_units=("f", (3, 3)), edge_lengths=("f", (3,)), basis_u=("f", (3,)), basis_v=("f", (3,)),
        c_u=("f", ()), c_v=("f", ()), permutation=("i", ()))
    if version == 3:
        common.update(local_edges=("f", (3, 3)), scales=("f", ()), minimum_edges=("f", ()),
            qualities=("f", ()), normal_guards=("f", ()), geometric_guards=("f", ()),
            altitudes=("f", (3,)), triangle_lower=("f", (3,)), triangle_upper=("f", (3,)))
    return common


def _validate_index(index, version, triangles, identities, work):
    kind = _s2._TriangleSurfaceIndex if version == 2 else _s3._TriangleSurfaceIndex
    _require(type(index) is kind, "WRONG_VERSION_OR_MALFORMED_INDEX")
    _require(type(index.lower) is np.ndarray and index.lower.ndim == 2 and index.lower.shape[1:] == (3,),
        "MALFORMED_INDEX_NODE_ARRAY")
    nodes = len(index.lower)
    _require(1 <= nodes <= 131071, "INVALID_INDEX_NODE_COUNT")
    specifications = {name: (tag, (len(triangles),)+shape) for name, (tag, shape) in _index_arrays(version).items()}
    specifications.update({name: ("f", (nodes, 3)) for name in ("lower", "upper")})
    specifications.update({name: ("i", (nodes,)) for name in ("left", "right", "starts", "stops")})
    size = 0
    for name, (tag, shape) in specifications.items():
        value = getattr(index, name)
        _require(type(value) is np.ndarray and value.shape == shape and value.dtype == np.dtype(
            np.float64 if tag == "f" else np.int64) and value.flags.c_contiguous and not value.flags.writeable,
            "MALFORMED_OR_MUTABLE_INDEX_ARRAY")
        work.charge("adapter_scalar_tests", int(value.size))
        _require(np.isfinite(value).all(), "NONFINITE_INDEX_ARRAY")
        size += value.nbytes
    work.charge("adapter_scalar_tests", int(triangles.size)+len(identities))
    _require(np.array_equal(index.triangles, triangles) and type(index.source_face_ids) is tuple
        and all(type(a) is type(b) and a == b for a, b in zip(index.source_face_ids, identities))
        and len(index.source_face_ids) == len(identities), "INDEX_GEOMETRY_OR_FACE_BINDING_MISMATCH")
    _require(size == (336 if version == 2 else 520)*len(triangles)+80*nodes,
        "INDEX_NUMERIC_PAYLOAD_MODEL_MISMATCH")
    return size, nodes


def _v2_telemetry(answer, allocated, work):
    _require(type(answer) is dict and type(answer.get("diagnostics")) is dict, "MALFORMED_V2_QUERY_TELEMETRY")
    raw = answer["diagnostics"]
    _require(set(raw) in (set(_V2_COUNTS), set(_V2_COUNTS)|{"normal_dot_guard"})
        and all(type(key) is str for key in raw), "MALFORMED_V2_QUERY_TELEMETRY")
    values = {key: raw[key] for key in _V2_COUNTS}
    _require(all(type(value) is int and 0 <= value <= 2**53 for value in values.values()),
        "MALFORMED_V2_QUERY_COUNTER")
    _require(values["work_units"] == values["node_visits"]+8*values["triangle_pairs"]+3*values["normal_pairs"]
        and values["passes_completed"] <= 2 and values["maximum_pair_batch"] <= allocated.max_pair_batch
        and values["maximum_tie_storage_rows"] <= allocated.max_pair_batch,
        "INCONSISTENT_V2_QUERY_TELEMETRY")
    if "normal_dot_guard" in raw:
        _require(type(raw["normal_dot_guard"]) in (float, np.float64) and math.isfinite(raw["normal_dot_guard"])
            and raw["normal_dot_guard"] > 0, "INVALID_V2_NORMAL_DOT_GUARD")
    if answer.get("status") == "COMPLETE":
        _require(values["passes_completed"] == 2 and "normal_dot_guard" in raw, "INCOMPLETE_V2_QUERY_PASSES")
    return dict(values, certificate_pairs=0, triangle_box_pairs=0)


def _validate_v2_query(answer, rows, index, allocated, work):
    expected = set(_s2.RESULT_ARRAY_KEYS)|{"status", "reason", "diagnostics"}
    _require(type(answer) is dict and len(answer) == len(expected) and set(answer) == expected
        and all(type(key) is str for key in answer), "MALFORMED_V2_QUERY_SCHEMA")
    _require(answer["status"] == "COMPLETE" and answer["reason"] == "FINITE_TRIANGLE_CLOSEST_QUERY_COMPLETE",
        "INCOMPLETE_V2_QUERY_RESULT")
    for key in ("distances", "numerical_distance_guard", "closest_points", "oriented_normals"):
        value, shape = answer[key], (rows, 3) if key in ("closest_points", "oriented_normals") else (rows,)
        _require(type(value) is np.ndarray and value.dtype == np.dtype(np.float64) and value.shape == shape,
            "MALFORMED_V2_QUERY_FLOAT_ARRAY")
        work.charge("adapter_scalar_tests", int(value.size))
        _require(np.isfinite(value).all(), "NONFINITE_V2_QUERY_ARRAY")
        if key in ("distances", "numerical_distance_guard"):
            _require(np.all(value >= 0), "NEGATIVE_V2_QUERY_DISTANCE")
    for key in ("normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous"):
        value = answer[key]
        _require(type(value) is np.ndarray and value.dtype == np.dtype(bool) and value.shape == (rows,),
            "MALFORMED_V2_QUERY_BOOLEAN_ARRAY")
    for key in ("triangle_indices", "near_tie_counts"):
        value = answer[key]
        _require(type(value) is np.ndarray and value.dtype == np.dtype(np.int64) and value.shape == (rows,),
            "MALFORMED_V2_QUERY_INDEX_ARRAY")
    ids, ties = answer["triangle_indices"], answer["near_tie_counts"]
    _require(np.all(ids >= 0) and np.all(ids < len(index.triangles))
        and np.all(ties >= 1) and np.all(ties <= allocated.max_ties_per_query), "INVALID_V2_QUERY_IDENTITIES_OR_TIES")
    regions = answer["closest_regions"]
    _require(type(regions) is np.ndarray and regions.dtype == np.dtype("U6") and regions.shape == (rows,)
        and np.isin(regions, ("FACE", "EDGE", "VERTEX")).all(), "MALFORMED_V2_QUERY_REGIONS")
    source = answer["source_face_ids"]
    _require(type(source) is tuple and len(source) == rows, "MALFORMED_V2_QUERY_FACE_IDENTITIES")
    work.charge("adapter_scalar_tests", 12*rows)
    _require(all(type(value) is type(index.source_face_ids[int(identifier)])
        and value == index.source_face_ids[int(identifier)] for value, identifier in zip(source, ids))
        and np.array_equal(answer["oriented_normals"], index.normals[ids]), "V2_QUERY_INDEX_BINDING_MISMATCH")
    stats = answer["diagnostics"]
    _require((rows == 0 and stats["triangle_pairs"] == stats["tie_batches_completed"] == 0)
        or (rows > 0 and stats["triangle_pairs"] >= 2*rows and stats["tie_batches_completed"] > 0),
        "INCOMPLETE_V2_QUERY_ROW_WORK")
    return answer


class _ProposalOptimizerSessionV4:
    def __init__(self, token, work, proposal_budget, final_budget):
        _require(token is _TOKEN, "PRIVATE_OPTIMIZER_SESSION_CONSTRUCTOR")
        self._token, self._work = token, work
        self._surface_budgets = {"proposal": proposal_budget, "final": final_budget}
        self._indices = {"proposal": None, "final": None}
        self._binding = {}
        self._state, self._failure = "PREPARING", None

    def _live(self):
        _require(self._token is _TOKEN and self._state == "READY", "OPTIMIZER_SESSION_NOT_USABLE")

    def _fatal(self, phase, error):
        reason = str(error) if isinstance(error, (_OptimizerFailureV4, _v2._SurfaceFailure, _v3._SurfaceFailure)) else type(error).__name__
        reason = reason[:180]
        self._state = "FAILED"
        self._failure = dict(phase=phase, reason=reason)
        self._indices = {"proposal": None, "final": None}
        return _OptimizerFailureV4("PROPOSAL_OPTIMIZER_V4_FATAL:"+phase+":"+reason)

    def _build(self, phase, triangles, identities, identity_bytes):
        work, version = self._work, 2 if phase == "proposal" else 3
        budget = self._surface_budgets[phase]
        available = asdict(budget)
        for key in ("build_nodes", "build_triangle_visits"):
            available["max_"+key] = min(available["max_"+key], getattr(work.resources, "max_"+key)-work.stats[key])
        kind, module = (_s2.TriangleSurfaceBudgets, _s2) if version == 2 else (_s3.TriangleSurfaceBudgets, _s3)
        allocated = kind(**available)
        # >2x immutable+mutable output arrays plus local geometry, sorting and
        # per-node numeric/list payload. Source-model envelope, not RSS proof.
        scratch = 2048*len(triangles)+512*allocated.max_build_nodes+1048576
        work.stats["build_attempts"] += 1
        with work.temporary(scratch):
            try:
                answer = module.build_triangle_surface(triangles, identities, budgets=allocated)
            except Exception:
                work.unknown_work = work.unknown_payload = True
                raise
            _require(type(answer) is dict and type(answer.get("diagnostics")) is dict, "MALFORMED_BUILD_RESULT")
            raw = answer["diagnostics"]
            _require(set(raw) == {"build_nodes", "build_triangle_visits"}
                and all(type(key) is str for key in raw)
                and all(type(value) is int and value >= 0 for value in raw.values()), "MALFORMED_BUILD_TELEMETRY")
            for key in ("build_nodes", "build_triangle_visits"):
                _require(raw[key] <= available["max_"+key], "BUILD_EXCEEDED_ALLOCATED_BUDGET")
                work.charge(key, raw[key])
            self._binding[phase] = dict(version=version, build_diagnostics=dict(raw), build_status=answer.get("status"))
            if answer.get("status") != "COMPLETE":
                _require(answer.get("status") == "UNAVAILABLE" and answer.get("index") is None
                    and type(answer.get("reason")) is str, "MALFORMED_BUILD_REFUSAL")
                raise _OptimizerFailureV4("INDEX_UNAVAILABLE:"+answer["reason"][:160])
            size, nodes = _validate_index(answer["index"], version, triangles, identities, work)
            _require(raw["build_nodes"] == nodes, "BUILD_NODE_COUNT_MISMATCH")
            work.charge("index_array_bytes", size)
            work.stats["retained_identity_bytes"] += identity_bytes
            work.memory()
            self._binding[phase].update(triangles_sha256=work.digest_array(answer["index"].triangles),
                numeric_array_bytes=size, source_face_id_encoding="ORDERED_TYPED_ASCII_FACE_IDS_V1")
            self._indices[phase] = answer["index"]
            work.stats["completed_builds"] += 1

    def _query(self, phase, points):
        self._live()
        work, budget = self._work, self._surface_budgets[phase]
        version, index = (2 if phase == "proposal" else 3), self._indices[phase]
        telemetry_known = False
        dispatched = False
        try:
            _require(type(points) is np.ndarray and points.dtype == np.dtype(np.float64)
                and points.ndim == 2 and points.shape[1:] == (3,) and len(points) <= budget.max_query_rows,
                "INVALID_OPTIMIZER_QUERY_ROWS")
            rows = len(points)
            for name in (phase, "combined"):
                work.phase[name]["attempted_dispatches"] += 1
                work.phase[name]["attempted_rows"] += rows
            total = work.phase["combined"]
            _require(total["query_calls"] < work.combined.max_query_calls
                and total["query_rows"]+rows <= work.combined.max_query_rows, "COMBINED_QUERY_CALL_OR_ROW_BUDGET")
            available = asdict(budget)
            for key in _COMMON:
                if key == "certificate_pairs" and version == 2:
                    continue
                available["max_"+key] = min(available["max_"+key], getattr(work.combined, "max_"+key)-total[key])
            kind = _s2.TriangleSurfaceBudgets if version == 2 else _s3.TriangleSurfaceBudgets
            allocated = kind(**available)
            # Balanced median BVH depth is below bit_length(T). Reserve both
            # pending paths plus full outputs/validation and certificate scratch.
            scratch = (4096 if version == 2 else 8192)*allocated.max_pair_batch
            scratch += (2048+16*(len(index.triangles).bit_length()+4))*rows+1048576
            with work.temporary(scratch):
                owned = _snapshot_array(points, (3,), allocated.max_query_rows, 1e100, work, "QUERY_POINTS")
                for name in (phase, "combined"):
                    work.phase[name]["query_calls"] += 1
                    work.phase[name]["query_rows"] += rows
                dispatched = True
                answer = (_s2 if version == 2 else _s3).query_closest_surface(index, owned, budgets=allocated)
                _require(type(answer) is dict, "MALFORMED_QUERY_RESULT")
                stats = (_v2_telemetry(answer, allocated, work) if version == 2 else
                    _v3._diagnostic_snapshot(answer, complete=answer.get("status") == "COMPLETE"))
                for key in _COMMON:
                    _require(stats[key] <= (0 if key == "certificate_pairs" and version == 2 else available["max_"+key]),
                        "QUERY_EXCEEDED_ALLOCATED_BUDGET")
                for name in (phase, "combined"):
                    for key in _COMMON+("triangle_box_pairs",):
                        work.phase[name][key] += stats[key]
                telemetry_known = True
                if answer.get("status") != "COMPLETE":
                    keys = _s2.RESULT_ARRAY_KEYS if version == 2 else _s3.RESULT_ARRAY_KEYS
                    _require(answer.get("status") == "UNAVAILABLE" and type(answer.get("reason")) is str
                        and 0 < len(answer["reason"]) <= 180 and all(answer.get(key, object()) is None for key in keys),
                        "MALFORMED_WHOLE_QUERY_REFUSAL")
                    raise _OptimizerFailureV4("FINITE_SURFACE_QUERY_UNAVAILABLE:"+answer["reason"])
                if version == 2:
                    answer = _validate_v2_query(answer, rows, index, allocated, work)
                else:
                    work.charge("adapter_scalar_tests", 64*rows)
                    answer = _v3._validate_query_result(answer, rows, index=index)
                work.exported(answer, rows)
                for name in (phase, "combined"):
                    work.phase[name]["completed_query_calls"] += 1
                    work.phase[name]["completed_query_rows"] += rows
                return answer
        except Exception as error:
            if dispatched and not telemetry_known:
                work.unknown_work = True
            raise self._fatal(phase+"_QUERY", error) from error

    def final_query(self, represented_points):
        """Return only the real, complete, index-bound V3 result; no nominal fallback."""
        return self._query("final", represented_points)

    def propose(self, scan_subset, rotation, translation, *, diagonal, policy, coarse):
        """Unchanged V2 math after strict admission. Returned pose is NEVER acceptance."""
        self._live()
        work = self._work
        try:
            policy = _v3._policy_snapshot(policy)
            _require(type(coarse) is bool and type(diagonal) in (int, float) and math.isfinite(diagonal)
                and 0 <= diagonal <= 1e100, "INVALID_PROPOSAL_CONFIGURATION")
            limit = policy.coarse_points if coarse else policy.refine_points
            _require(type(scan_subset) is np.ndarray and 36 <= len(scan_subset) <= limit,
                "PROPOSAL_SUBSET_OUTSIDE_DECLARED_POLICY")
            work.charge("proposal_calls", 1)
            work.charge("proposal_input_rows", len(scan_subset))
            with work.temporary(4096*len(scan_subset)+1048576):
                scan = _snapshot_array(scan_subset, (3,), limit, 1e6, work, "PROPOSAL_SCAN")
                _require(type(rotation) is np.ndarray and rotation.dtype == np.dtype(np.float64)
                    and rotation.shape == (3, 3) and type(translation) is np.ndarray
                    and translation.dtype == np.dtype(np.float64) and translation.shape == (3,), "INVALID_TYPED_PROPOSAL_POSE")
                r, t = np.array(rotation, copy=True), np.array(translation, copy=True)
                _v3._candidate_pose(dict(rotation=r, translation=t))
                try:
                    answer = _v2._refine(scan, lambda points: self._query("proposal", points), r, t,
                        diagonal=diagonal, policy=policy, coarse=coarse)
                except _v2._SurfaceFailure as error:
                    raise self._fatal("PROPOSAL_QUERY", error) from error
                except _OptimizerFailureV4:
                    raise
                except (ValueError, OverflowError, np.linalg.LinAlgError):
                    work.stats["proposal_numerical_failures"] += 1
                    raise
                nr, nt, iterations, state, last = answer
                _v3._candidate_pose(dict(rotation=nr, translation=nt))
                _require(type(iterations) is int and 0 <= iterations <= (
                    policy.coarse_iterations if coarse else policy.refine_iterations)
                    and type(state) is str and state in ("INSUFFICIENT_REFINEMENT_SUPPORT", "STATIONARY",
                        "ITERATION_LIMIT", "NO_IMPROVING_FIXED_CORRESPONDENCE_STEP") and type(last) is dict,
                    "INVALID_PROPOSAL_RETURN")
                last = dict(last, proposal_distance_model=PROPOSAL_MODEL, proposal_support_policy=PROPOSAL_SUPPORT,
                    proposal_optimizer_objective=_v3.SURROGATE, final_rank_weight_scope=FINAL_RANK_SCOPE,
                    accepted_registration=False, final_v3_diagnostics_completed_by_this_method=False)
                _encode(last, 16384)
                nr = np.frombuffer(np.ascontiguousarray(nr, dtype=np.float64).tobytes(), dtype=np.float64).reshape(3, 3)
                nt = np.frombuffer(np.ascontiguousarray(nt, dtype=np.float64).tobytes(), dtype=np.float64)
                work.exported(dict(rotation=nr, translation=nt), 0)
                return nr, nt, iterations, state, last
        except _OptimizerFailureV4 as error:
            if self._state != "FAILED":
                raise self._fatal("PROPOSAL", error) from error
            raise
        except (ValueError, OverflowError, np.linalg.LinAlgError):
            # Ordinary candidate math failures remain distinct. The caller must
            # refuse whole publication if ANY selected candidate lacks diagnostics.
            raise
        except Exception as error:
            raise self._fatal("PROPOSAL", error) from error

    def snapshot_evidence(self):
        work = self._work
        try:
            work.memory()
            evidence = dict(schema=SCHEMA, session_status=self._state, usable=self._state == "READY",
                failure=self._failure, proposal_mode="NOMINAL_V2_ONLY", proposal_distance_model=PROPOSAL_MODEL,
                proposal_support_policy=PROPOSAL_SUPPORT, final_distance_model=_v3.DISTANCE_MODEL,
                final_rank_weight_scope=FINAL_RANK_SCOPE, trajectory_equivalence_with_v3=False,
                v2_soup_wide_conditioning_retained=True, fallback_policy="NONE_NO_RECENTERING_OR_GEOMETRY_CHANGE",
                accepted_registration=False, observed_bend_count=None, global_pose_uniqueness="NOT_ESTABLISHED",
                continuous_pose_family_coverage="NOT_ESTABLISHED", joint_configuration_status="NOT_ESTABLISHED",
                production_use_allowed=False, final_diagnostics_and_context_publication_owned_by_caller=True,
                previous_results_invalid_for_publication=self._state == "FAILED",
                unknown_work=work.unknown_work, unknown_payload=work.unknown_payload,
                statistics=dict(work.stats), query_resources={key: dict(value) for key, value in work.phase.items()},
                surface_budgets={key: asdict(value) for key, value in self._surface_budgets.items()},
                combined_query_budgets=asdict(work.combined), resource_budgets=asdict(work.resources),
                index_bindings=dict(self._binding), payload_scope="SOURCE_BUFFER_ENVELOPE_NOT_RSS_OR_CALLER_COPIES",
                input_custody_scope="REPRESENTED_ARRAY_IDENTITY_NOT_NATIVE_OR_PHYSICAL_AUTHENTICATION")
            _encode(evidence, work.resources.max_report_bytes)
            return evidence
        except Exception as error:
            self._fatal("EVIDENCE", error)
            # Fixed-schema terminal fallback keeps all scalar work; no prefix is usable.
            fallback = dict(schema=SCHEMA, session_status="FAILED", usable=False, failure=self._failure,
                unknown_work=work.unknown_work, unknown_payload=work.unknown_payload,
                statistics=dict(work.stats), query_resources={key: dict(value) for key, value in work.phase.items()},
                accepted_registration=False, observed_bend_count=None, production_use_allowed=False,
                previous_results_invalid_for_publication=True, full_evidence_discarded=True)
            _encode(fallback, 4096)
            return fallback


def prepare_proposal_optimizer_v4(owned_cad_triangles, owned_source_face_ids, *,
        proposal_surface_budgets, final_surface_budgets, combined_query_budgets,
        build_and_payload_budgets):
    """Prepare exactly two owned indexes and a shared ledger; no geometry queries.

All budget arguments are required exact dataclass instances. This returns no
accepted transform and does not schedule/refill candidates or invoke PCA.
"""
    session = None
    try:
        resources = _resource_configuration(build_and_payload_budgets)
        combined = _v3._validated_execution_budgets(combined_query_budgets)
        proposal = _surface_configuration(proposal_surface_budgets, 2)
        final = _surface_configuration(final_surface_budgets, 3)
        work = _Work(resources, combined)
        session = _ProposalOptimizerSessionV4(_TOKEN, work, proposal, final)
        maximum = min(proposal.max_triangles, final.max_triangles)
        _require(type(owned_cad_triangles) is np.ndarray and owned_cad_triangles.ndim == 3
            and owned_cad_triangles.shape[1:] == (3, 3) and 1 <= len(owned_cad_triangles) <= maximum,
            "INVALID_CAD_TRIANGLE_SHAPE_OR_COUNT")
        count = len(owned_cad_triangles)
        # Numeric triangles plus maximum typed face-ID bytes and immutable-copy
        # overlap are admitted BEFORE copying caller buffers or iterating IDs.
        owned_reservation = count*(72+176)
        work.charge("owned_input_bytes", owned_reservation)
        with work.temporary(owned_reservation):
            triangles = _snapshot_array(owned_cad_triangles, (3, 3), maximum, 1e100, work, "CAD_TRIANGLES")
            identities, identity_bytes, face_sha = _snapshot_faces(owned_source_face_ids, count, work)
            _require(triangles.nbytes+identity_bytes <= owned_reservation, "SOURCE_ID_PAYLOAD_MODEL_MISMATCH")
            source_sha = work.digest_array(triangles)
        session._binding["input"] = dict(triangle_count=count, triangles_sha256=source_sha,
            source_face_ids_sha256=face_sha, source_face_id_encoding="ORDERED_TYPED_ASCII_FACE_IDS_V1",
            unit_frame_scope="CALLER_DECLARED_COMMON_FRAME_UNITS_NOT_CALIBRATED")
        session._build("final", triangles, identities, identity_bytes)
        session._build("proposal", triangles, identities, identity_bytes)
        _require(all(session._binding[key]["triangles_sha256"] == source_sha for key in ("proposal", "final")),
            "DUAL_INDEX_SOURCE_BINDING_MISMATCH")
        for key in ("proposal", "final"):
            session._binding[key]["source_face_ids_sha256"] = face_sha
        session._state = "READY"
    except Exception as error:
        if session is None:
            return dict(status="UNAVAILABLE", session_or_none=None,
                reasons=[str(error)[:180] if isinstance(error, _OptimizerFailureV4) else "INVALID_OPTIMIZER_CONFIGURATION"],
                evidence=dict(schema=SCHEMA, usable=False, preparation_dispatched=False,
                    accepted_registration=False, production_use_allowed=False))
        # A malformed returned build cannot be interpreted as measured zero work.
        if session._work.stats["build_attempts"] > session._work.stats["completed_builds"] and not isinstance(error, _OptimizerFailureV4):
            session._work.unknown_work = session._work.unknown_payload = True
        session._fatal("PREPARE", error)
    evidence = session.snapshot_evidence()
    complete = evidence["usable"] is True
    return dict(status="COMPLETE" if complete else "UNAVAILABLE", session_or_none=session if complete else None,
        evidence=evidence, reasons=[] if complete else ["PROPOSAL_OPTIMIZER_PREPARATION_UNAVAILABLE"])
