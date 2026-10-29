"""V5 registration consumer: the frozen V4 text reused by attribute, plus four bounded additions.

1. Exact reuse of the duplicated final full-scan surface query: V4 `_diagnose` queries
   `scan @ r.T + t` and the V15 `_context_support` queried the same rows at the same pose again.
   `_diagnose_with_reusable_query` runs the UNCHANGED V4 diagnosis through a transparent recorder
   and returns an owned, immutable, bound query result; `_context_support_v5` reuses it only when
   every binding matches exactly (same query object and index, same row count, same
   transformed-point bytes, same pose bytes, same distance gate, COMPLETE status) and otherwise
   performs its own query. Performed and reused work are accounted separately; a partial or
   unavailable query is never reused. Geometry, gates and uncertainty outputs are unchanged.
2. Cooperative intra-query stop points: `_QueriesV5` splits any query larger than the declared
   `cooperative_chunk_rows` into row chunks, validates and charges every chunk through the
   unchanged V4 ledger, and merges the chunk results exactly (row arrays concatenated in order,
   count statistics summed, batch maxima maximised, passes = 2 only if every chunk completed).
   Between chunks the allowance is checked, so the largest uninterruptible operation is one chunk
   query bounded in abstract work by the surface budgets, not in seconds. A partial chunk set is
   discarded, never reused or published.
3. A declared `RefinementAllowanceV1`: DETERMINISTIC per-lane group and surface-query row quotas
   (a function of the inputs and the fixed order only) plus an absolute process-CPU safety guard
   (timing dependent, declared as such, its realized trace bound by the caller). Exhaustion raises
   `_AllowanceExhausted`, a distinct typed exception (NOT a `_SurfaceFailure`): the caller turns
   it into an explicit EXPIRED/OMITTED disposition (unknown, never rejected). Unclassified errors,
   corrupted results and surface failures remain fatal exactly as in V4.
4. Per-call CPU/wall metering with an injectable clock, so lane/candidate cost receipts can be
   published instead of inferred. Process CPU is absolute (imports and preparation included).
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import resource
import time

import numpy as np

from . import cad_guided_global_registration_v4 as _v4
from . import cad_guided_stage_policies_v2 as _stages
from . import cad_guided_triangle_surface_v4 as _surface

DISTANCE_MODEL = _v4.DISTANCE_MODEL
SUPPORT_POLICY = _v4.SUPPORT_POLICY
SURROGATE = _v4.SURROGATE
CONSUMER_COMPLETE = _v4.CONSUMER_COMPLETE
REGISTRATION_IMPLEMENTATION = "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1"
QUERY_REUSE_IMPLEMENTATION = "EXACT_BOUND_REUSE_OF_DIAGNOSIS_FULL_SCAN_QUERY_FOR_CONTEXT_SUPPORT_V1"
CHUNKING_IMPLEMENTATION = "ROW_CHUNKED_SURFACE_QUERIES_WITH_EXACT_MERGE_AND_COOPERATIVE_ALLOWANCE_STOPS_V1"
_EXECUTION_MAXIMA = _v4._EXECUTION_MAXIMA
_RESOURCE_KEYS = _v4._RESOURCE_KEYS
_STAT_KEYS = _v4._STAT_KEYS
_QUERY_METADATA = _v4._QUERY_METADATA
SurfaceRegistrationBudgetsV4 = _v4.SurfaceRegistrationBudgetsV4
_SurfaceFailure = _v4._SurfaceFailure
_require = _v4._require
_validated_execution_budgets = _v4._validated_execution_budgets
_policy_snapshot = _v4._policy_snapshot
_diagnostic_snapshot = _v4._diagnostic_snapshot
_validate_query_result = _v4._validate_query_result
_support_masks, _unambiguous, _chosen, _possible_mask = _v4._support_masks, _v4._unambiguous, _v4._chosen, _v4._possible_mask
_metric_bounds, _fraction_bound, _error_summary = _v4._metric_bounds, _v4._fraction_bound, _v4._error_summary
_refine, _diagnose, _candidate_pose, _select = _v4._refine, _v4._diagnose, _v4._candidate_pose, _v4._select
_v1, _v2 = _v4._v1, _v4._v2
LANES = ("main", "completion")
STAGE = "stage"   # metered for receipts only: no lane accumulation, no quota
_SUM_STATS = ("node_visits", "triangle_box_pairs", "triangle_pairs", "certificate_pairs", "normal_pairs", "work_units",
              "tie_batches_completed", "interval_width_uncertain_pairs", "certificate_failures",
              "proof_pruned_triangle_pairs", "vertex_lookup_rows")
_MAX_STATS = ("maximum_pair_batch", "maximum_certificate_batch", "maximum_tie_storage_rows")


class _AllowanceExhausted(RuntimeError):
    """A declared allowance (lane group/row quota or the CPU guard) is exhausted: the current
    candidate's evidence is UNKNOWN, not rejected. Deliberately not a `_SurfaceFailure`."""


def _process_clock():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime+usage.ru_stime, time.monotonic()


def _bytes_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def _pose_digest(rotation, translation):
    return _bytes_digest(np.concatenate((np.asarray(rotation, dtype=np.float64).ravel(),
                                         np.asarray(translation, dtype=np.float64).ravel())))


@dataclass(frozen=True)
class RefinementAllowanceV1:
    """Declared per-lane work allocation inside the unchanged outer leaf cap.

    Deterministic quotas (functions of the inputs and the fixed candidate order only):
    `main_refined_groups` / `completion_refined_groups` bound how many start groups each lane may
    refine (the frozen 48-group schedule is the outer bound); `main_query_rows` bounds the surface
    rows queried by main-lane refinement; `completion_query_rows` is ONE allocation for the whole
    completion lane's QUERY rows (prefilter box screens, finite face queries and refinement
    queries; enumeration, support binding and face-index builds charge no rows and are covered
    only by the CPU-guard phase checks). Rows beyond a quota make the current candidate
    EXPIRED_BY_LANE_WORK_QUOTA and the remaining lane candidates OMITTED_BY_LANE_WORK_QUOTA
    (reproducible).

    Safety guard (timing dependent, declared as such): once absolute process CPU (imports and
    preparation included) exceeds `cpu_guard_seconds`, metered work stops cooperatively at the next
    stop point (EXPIRED_BY_CPU_GUARD / OMITTED_BY_CPU_GUARD): before each candidate, before each
    query, between the chunks of a chunked full-scan query, and in the completion lane before and
    during enumeration, before support binding, before and between eager face-index builds and
    before each screen/query. The difference `cpu_budget_seconds - cpu_guard_seconds` is headroom
    for context publication, measurement, publication validation and serialization; it is NOT a
    guarantee. Atomic (uninterruptible) work that the allowance never stops: everything before the
    first stop point (imports, scan/CAD preparation, plane proposals, the main-lane prefilter and
    its face-index builds, surface-index construction — all counted by the absolute guard but not
    interrupted by it), one chunk query of at most `cooperative_chunk_rows` rows under the surface
    budgets' per-call abstract work caps, and one face-index build under the face build caps. Each
    is bounded in declared work, not in seconds; a kill inside one is a retained negative outcome.
    """
    enabled: bool = True
    main_refined_groups: int = 12
    completion_refined_groups: int = 4
    main_query_rows: int = 6_000_000
    completion_query_rows: int = 3_000_000
    cpu_budget_seconds: float = 1860.
    cpu_guard_seconds: float = 1500.
    cooperative_chunk_rows: int = 32_768

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        for name, lower, upper in (("main_refined_groups", 1, 48), ("completion_refined_groups", 0, 48),
                                   ("main_query_rows", 1, 64_000_000), ("completion_query_rows", 0, 64_000_000),
                                   ("cooperative_chunk_rows", 1024, 200_000)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid "+name)
        for name in ("cpu_budget_seconds", "cpu_guard_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.:
                raise ValueError("invalid "+name)
        if self.cpu_guard_seconds >= self.cpu_budget_seconds:
            raise ValueError("the CPU guard must leave headroom below the declared CPU budget")

    def group_quota(self, lane):
        return dict(main=self.main_refined_groups, completion=self.completion_refined_groups)[lane]

    def row_quota(self, lane):
        return dict(main=self.main_query_rows, completion=self.completion_query_rows)[lane]

    def headroom_seconds(self):
        return self.cpu_budget_seconds-self.cpu_guard_seconds


class _CpuMeter:
    """Lane/candidate CPU, wall and row accounting with an injectable clock; the quota checks
    are deterministic (rows), the CPU guard is a declared timing-dependent safety stop."""

    def __init__(self, allowance=None, clock=None):
        _require(allowance is None or type(allowance) is RefinementAllowanceV1, "INVALID_REFINEMENT_ALLOWANCE")
        self.allowance = allowance
        self.clock = _process_clock if clock is None else clock
        self.lane_cpu = {lane: 0. for lane in LANES}
        self.lane_wall = {lane: 0. for lane in LANES}
        self.lane_rows = {lane: 0 for lane in LANES}            # ATTEMPTED rows (charged before the work, as V4 charges)
        self.lane_rows_completed = {lane: 0 for lane in LANES}  # rows whose query returned COMPLETE
        self.lane_calls = {lane: 0 for lane in LANES}
        self.lane_calls_completed = {lane: 0 for lane in LANES}
        self.active = None
        self.receipts = []
        self.guard_tripped = False
        self.lane_stopped = {}

    @property
    def current_lane(self):
        return None if self.active is None or self.active[0] == STAGE else self.active[0]

    def begin(self, lane, label):
        _require((lane in LANES or lane == STAGE) and self.active is None, "INVALID_METER_SECTION")
        cpu, wall = self.clock()
        self.active = (lane, label, cpu, wall)

    def end(self, **fields):
        _require(self.active is not None, "NO_ACTIVE_METER_SECTION")
        lane, label, cpu0, wall0 = self.active
        cpu, wall = self.clock()
        record = dict(lane=lane, label=label, cpu_seconds=max(0., cpu-cpu0), wall_seconds=max(0., wall-wall0),
                      process_cpu_seconds_at_end=cpu, **fields)
        if lane != STAGE:
            self.lane_cpu[lane] += record["cpu_seconds"]
            self.lane_wall[lane] += record["wall_seconds"]
        self.active = None
        self.receipts.append(record)
        return record

    def charge_rows(self, lane, rows):
        """Attempted work, charged BEFORE the primitive runs (a fatal attempted query stays charged)."""
        if lane in LANES:
            self.lane_rows[lane] += int(rows)
            self.lane_calls[lane] += 1

    def complete_rows(self, lane, rows):
        if lane in LANES:
            self.lane_rows_completed[lane] += int(rows)
            self.lane_calls_completed[lane] += 1

    def mark_stopped(self, lane, reason):
        """A typed expiry in `lane` stops every later candidate of that lane (deterministic omission)."""
        if lane in LANES:
            self.lane_stopped.setdefault(lane, str(reason))

    def exhaustion(self, lane=None, *, rows_about_to_query=0):
        """Return the typed exhaustion reason for `lane` at this instant, else None.

        Row quotas are checked deterministically BEFORE the work; the CPU guard is checked on the
        absolute process clock and is timing dependent by declaration."""
        if self.allowance is None or not self.allowance.enabled:
            return None
        lane = lane if lane is not None else self.current_lane
        if lane is not None and lane in self.lane_stopped:
            return self.lane_stopped[lane]
        if lane is not None and self.lane_rows[lane]+int(rows_about_to_query) > self.allowance.row_quota(lane):
            return "LANE_WORK_QUOTA:"+lane
        cpu, _ = self.clock()
        if cpu > self.allowance.cpu_guard_seconds:
            self.guard_tripped = True
            return "CPU_GUARD"
        return None

    def summary(self):
        cpu, wall = self.clock()
        return dict(allowance=None if self.allowance is None else asdict(self.allowance),
                    lane_cpu_seconds=dict(self.lane_cpu), lane_wall_seconds=dict(self.lane_wall),
                    lane_query_rows_attempted=dict(self.lane_rows), lane_query_calls_attempted=dict(self.lane_calls),
                    lane_query_rows_completed=dict(self.lane_rows_completed),
                    lane_query_calls_completed=dict(self.lane_calls_completed),
                    process_cpu_seconds_now=cpu, cpu_guard_tripped=self.guard_tripped,
                    lane_stopped=dict(self.lane_stopped),
                    receipts=list(self.receipts),
                    scope="PROCESS_CPU_AND_MONOTONIC_WALL_DELTAS_OF_METERED_SECTIONS_NOT_A_PROFILE",
                    timing_dependence="CPU_GUARD_STOPS_ARE_TIMING_DEPENDENT_ROW_QUOTA_STOPS_ARE_DETERMINISTIC")


def _merge_chunks(chunks, rows):
    """Exact merge of validated chunk results into one V4-shaped result for `rows` points."""
    first = chunks[0]
    merged = {key: first[key] for key in _QUERY_METADATA}
    for key in _surface.RESULT_ARRAY_KEYS:
        if key == "source_face_ids":
            merged[key] = tuple(item for chunk in chunks for item in chunk[key])
        else:
            merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    diagnostics = {}
    for key in _STAT_KEYS:
        values = [int(chunk["diagnostics"][key]) for chunk in chunks]
        if key in _SUM_STATS:
            diagnostics[key] = sum(values)
        elif key in _MAX_STATS:
            diagnostics[key] = max(values)
        elif key == "passes_completed":
            diagnostics[key] = 2 if all(value == 2 for value in values) else min(values)
        else:
            raise _SurfaceFailure("UNMERGED_V4_QUERY_STATISTIC:"+key)
    merged["diagnostics"] = diagnostics
    _require(len(merged["distances"]) == rows, "CHUNK_MERGE_ROW_COUNT_MISMATCH")
    return merged


class _QueriesV5(_v4._Queries):
    """V4 query ledger plus chunked cooperative queries, reuse accounting, metering and quotas."""

    def __init__(self, index, budgets, surface_budgets, *, meter=None, chunk_rows=None):
        super().__init__(index, budgets, surface_budgets)
        _require(meter is None or type(meter) is _CpuMeter, "INVALID_QUERY_METER")
        _require(chunk_rows is None or (type(chunk_rows) is int and 1024 <= chunk_rows <= 200_000), "INVALID_CHUNK_ROWS")
        self.meter = meter
        self.chunk_rows = chunk_rows
        self.statistics.update(reused_query_calls=0, reused_query_rows=0, chunked_query_calls=0, chunk_calls=0,
                               cpu_seconds=0., wall_seconds=0., allowance_checks=0, allowance_stops=0)

    def _stop_if_exhausted(self, rows):
        if self.meter is None:
            return
        self.statistics["allowance_checks"] += 1
        reason = self.meter.exhaustion(rows_about_to_query=rows)
        if reason is not None:
            self.statistics["allowance_stops"] += 1
            raise _AllowanceExhausted(reason)

    def _one(self, points):
        clock = self.meter.clock if self.meter is not None else _process_clock
        if self.meter is not None:
            self.meter.charge_rows(self.meter.current_lane, len(points))   # attempted, before any work
        cpu0, wall0 = clock()
        try:
            answer = super().__call__(points)
        finally:
            cpu, wall = clock()
            self.statistics["cpu_seconds"] += max(0., cpu-cpu0)
            self.statistics["wall_seconds"] += max(0., wall-wall0)
        if self.meter is not None:
            self.meter.complete_rows(self.meter.current_lane, len(points))
        return answer

    def __call__(self, points):
        _require(type(points) is np.ndarray and points.ndim == 2 and points.shape[1:] == (3,),
                 "INVALID_REGISTRATION_QUERY_ROWS")
        rows = len(points)
        if self.chunk_rows is None or rows <= self.chunk_rows:
            self._stop_if_exhausted(rows)
            return self._one(points)
        self.statistics["chunked_query_calls"] += 1
        chunks = []
        for start in range(0, rows, self.chunk_rows):
            block = points[start:start+self.chunk_rows]
            self._stop_if_exhausted(len(block))          # cooperative stop point between chunks
            chunks.append(self._one(block))
            self.statistics["chunk_calls"] += 1
        merged = _merge_chunks(chunks, rows)
        return _validate_query_result(merged, rows, index=self.index)

    def note_reuse(self, rows):
        self.statistics["reused_query_calls"] += 1
        self.statistics["reused_query_rows"] += int(rows)


def _query_policy_digest(query_surface, policy):
    """Exact immutable snapshot of everything that determines a query's answer besides the rows and the
    index: the live surface budgets and execution budgets of the query object (read at call time, so a
    mutation without replacing the object changes the digest), the chunking, the registration policy
    snapshot and the implementation contract strings of this consumer and the surface primitive."""
    snapshot = dict(
        surface_budgets=asdict(getattr(query_surface, "surface_budgets", None)) if getattr(query_surface, "surface_budgets", None) is not None else None,
        execution_budgets=asdict(getattr(query_surface, "budgets", None)) if getattr(query_surface, "budgets", None) is not None else None,
        chunk_rows=getattr(query_surface, "chunk_rows", None),
        policy=asdict(_policy_snapshot(policy)),
        registration_implementation=REGISTRATION_IMPLEMENTATION, chunking=CHUNKING_IMPLEMENTATION,
        reuse=QUERY_REUSE_IMPLEMENTATION, surface_initial_upper_bound=_surface.INITIAL_UPPER_BOUND,
        surface_interval_width_policy=_surface.INTERVAL_WIDTH_POLICY, surface_max_ties=_surface.MAX_TIES_PER_QUERY,
        query_metadata=dict(_QUERY_METADATA))
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _FrozenQueryPayload:
    """Tamper-evident owned snapshot of one COMPLETE query result: every row array is a read-only view
    over private bytes (NumPy refuses to make such a view writable again), face ids are a tuple, metadata
    and diagnostics are canonical JSON text. Consumers never receive the payload itself: `_thaw_query`
    revalidates the bound digest (key, dtype, SHAPE and bytes of every array plus both texts) and
    reconstructs an owned consumer dictionary validated against the expected rows and the index."""
    arrays: tuple          # ((key, read-only bytes-backed ndarray | tuple of face ids), ...) in RESULT_ARRAY_KEYS order
    metadata_json: str
    diagnostics_json: str
    rows: int


def _immutable_array(value):
    """A read-only view over a private bytes copy: `setflags(write=True)` raises on such a view."""
    value = np.ascontiguousarray(value)
    _require(value.dtype.kind in "biufU", "UNFREEZABLE_QUERY_ARRAY_DTYPE")
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _bytes_backed_read_only(value):
    if type(value) is not np.ndarray or value.flags.writeable:
        return False
    base = value
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _freeze_query(answer, rows):
    """Owned immutable snapshot of a COMPLETE query result plus its bound digest."""
    arrays = []
    for key in _surface.RESULT_ARRAY_KEYS:
        arrays.append((key, tuple(answer[key]) if key == "source_face_ids" else _immutable_array(answer[key])))
    payload = _FrozenQueryPayload(arrays=tuple(arrays),
        metadata_json=json.dumps({key: answer[key] for key in _QUERY_METADATA}, sort_keys=True, separators=(",", ":")),
        diagnostics_json=json.dumps(answer["diagnostics"], sort_keys=True, separators=(",", ":")), rows=int(rows))
    return payload, _query_result_digest(payload)


def _query_result_digest(payload):
    digest = hashlib.sha256()
    digest.update(str(int(payload.rows)).encode())
    for key, value in payload.arrays:
        digest.update(key.encode())
        if key == "source_face_ids":
            digest.update(json.dumps(list(value), separators=(",", ":")).encode())
        else:
            digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(payload.metadata_json.encode()); digest.update(payload.diagnostics_json.encode())
    return digest.hexdigest()


def _thaw_query(reusable, index=None):
    """INTRINSIC integrity of the retained record, checked BEFORE any lookup-key comparison: schema,
    immutable bytes-backed arrays, bound digest (with shapes), COMPLETE status and the full V4 result
    grammar against the retained row count. Any failure is the fatal `_SurfaceFailure` (corrupted
    retained evidence is never silently recomputed), never a typed allowance expiry. The index-bound
    part of the grammar (row identities against a triangle index) is checked only when the caller
    passes the index the record was bound to, i.e. after every lookup key matched: a legitimately
    replaced index is a cache miss, not a corruption."""
    try:
        _require(type(reusable) is _ReusableQuery, "REUSABLE_QUERY_RECORD_INVALID")
        payload = reusable.payload
        _require(type(payload) is _FrozenQueryPayload and type(payload.arrays) is tuple
            and [key for key, _ in payload.arrays] == list(_surface.RESULT_ARRAY_KEYS)
            and type(payload.metadata_json) is str and type(payload.diagnostics_json) is str
            and type(payload.rows) is int and payload.rows == reusable.rows
            and type(reusable.result_sha256) is str, "REUSABLE_QUERY_PAYLOAD_SCHEMA")
        for key, value in payload.arrays:
            if key == "source_face_ids":
                _require(type(value) is tuple and len(value) == payload.rows, "REUSABLE_QUERY_FACE_IDS_NOT_IMMUTABLE")
            else:
                _require(_bytes_backed_read_only(value) and value.shape[:1] == (payload.rows,),
                         "REUSABLE_QUERY_ARRAY_NOT_IMMUTABLE")
        _require(_query_result_digest(payload) == reusable.result_sha256, "REUSABLE_QUERY_DIGEST_MISMATCH")
        metadata, diagnostics = json.loads(payload.metadata_json), json.loads(payload.diagnostics_json)
        _require(type(metadata) is dict and set(metadata) == set(_QUERY_METADATA)
            and metadata.get("status") == "COMPLETE" and type(diagnostics) is dict, "REUSABLE_QUERY_METADATA_INVALID")
        answer = dict(payload.arrays)
        answer.update(metadata)
        answer["diagnostics"] = diagnostics
        return _validate_query_result(answer, payload.rows, index=index)
    except _SurfaceFailure:
        raise
    except Exception as error:
        raise _SurfaceFailure("REUSABLE_QUERY_INTEGRITY_FAILURE:"+str(error)[:120]) from error


@dataclass(frozen=True)
class _ReusableQuery:
    """Owned immutable result payload with a bound result digest and the exact bindings that made it:
    rows, transformed-point bytes, pose bytes, query object and index identities and the exact
    policy/implementation snapshot. Reuse revalidates integrity first (fatal), then every binding."""
    payload: _FrozenQueryPayload
    result_sha256: str
    points_sha256: str
    rows: int
    query_surface_id: int
    index_id: int
    pose_sha256: str
    gate_mm: float
    policy_sha256: str


def _diagnose_with_reusable_query(scan, cad, refine, query_surface, r, t, policy):
    """Run the UNCHANGED V4 diagnosis; additionally return its first (full-scan) query as an owned,
    bound, immutable result when that query was exactly `scan @ r.T + t` and COMPLETE."""
    captured = []

    class _Recorder:
        def __call__(self, points):
            answer = query_surface(points)
            if not captured:
                captured.append((np.array(points, dtype=np.float64, copy=True), answer))
            return answer
    result = _v4._diagnose(scan, cad, refine, _Recorder(), r, t, policy)
    positions = _v1._positions(scan, 200000)
    expected = positions@np.asarray(r, dtype=np.float64).T+np.asarray(t, dtype=np.float64)
    reusable = None
    if captured:
        points, answer = captured[0]
        if (type(answer) is dict and answer.get("status") == "COMPLETE" and points.shape == expected.shape
                and np.array_equal(points, expected)):
            frozen, digest = _freeze_query(answer, len(expected))
            reusable = _ReusableQuery(payload=frozen, result_sha256=digest, points_sha256=_bytes_digest(expected),
                                      rows=int(len(expected)), query_surface_id=id(query_surface),
                                      index_id=id(getattr(query_surface, "index", None)), pose_sha256=_pose_digest(r, t),
                                      gate_mm=float(_policy_snapshot(policy).max_correspondence_distance_mm),
                                      policy_sha256=_query_policy_digest(query_surface, policy))
    return result, reusable


def _context_support_v5(scan, query_surface, rotation, translation, policy, reverse_coverage, reusable=None):
    """V15 `_context_support` text with exact bound reuse of the diagnosis full-scan query.

    All selected registration rows are retained; no ambiguous row is erased. These are
    represented-coordinate intervals, not latent scanner-error bounds. Reuse changes no value:
    the reused answer is the same immutable query result the diagnosis consumed.
    """
    points = scan@rotation.T+translation
    binding = dict(pose_sha256=_pose_digest(rotation, translation), points_sha256=_bytes_digest(points),
                   rows=int(len(points)), gate_mm=float(policy.max_correspondence_distance_mm),
                   query_surface_id=id(query_surface), index_id=id(getattr(query_surface, "index", None)),
                   policy_sha256=_query_policy_digest(query_surface, policy))
    reuse = "PERFORMED_NEW_QUERY"
    if reusable is not None:
        # Intrinsic integrity first and fatal on failure; only then is a legitimate binding mismatch a
        # cache miss (a genuinely replaced index requeries); with every key matched, the record is finally
        # validated against the index it was bound to (the current one), still fatal on failure.
        retained = _thaw_query(reusable)
        if all(getattr(reusable, key) == binding[key] for key in ("query_surface_id", "index_id", "rows", "gate_mm",
                                                                  "pose_sha256", "points_sha256", "policy_sha256")):
            answer = _thaw_query(reusable, index=getattr(query_surface, "index", None))
            if hasattr(query_surface, "note_reuse"):
                query_surface.note_reuse(len(points))
            reuse = "REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING"
        else:
            answer = query_surface(points)
    else:
        answer = query_surface(points)
    masks = _stages.classify_stage_rows_v1(answer, policy.max_correspondence_distance_mm)
    inside = masks["definite_proximity"]
    count = int(np.count_nonzero(inside))
    coverage = _fraction_bound(count, len(scan), upper=False)
    metrics = _metric_bounds(answer["distance_lower"][inside], answer["distance_upper"][inside])
    reasons = []
    if count < 36 or coverage < policy.min_scan_coverage or reverse_coverage < policy.min_cad_coverage:
        reasons.append("INSUFFICIENT_CONDITIONAL_CONTEXT_PROXIMITY")
    if metrics["p95_upper"] is None or metrics["p95_upper"] > policy.max_inlier_plane_p95_mm:
        reasons.append("CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE")
    return dict(status="SUPPORTED_CONTEXT" if not reasons else "REJECTED_CONTEXT",
        pose_sha256=binding["pose_sha256"],
        reasons=reasons, geometric_scan_coverage_lower=coverage,
        geometric_inlier_count=count, input_row_count=len(scan),
        row_counts=_stages.stage_row_counts_v1(answer, policy.max_correspondence_distance_mm),
        finite_distance_metrics=metrics, reverse_sample_coverage=reverse_coverage,
        proximity_policy=_stages.PROXIMITY, accepted_registration=False,
        query_reuse=reuse, query_reuse_implementation=QUERY_REUSE_IMPLEMENTATION,
        reuse_binding={key: binding[key] for key in ("pose_sha256", "points_sha256", "rows", "gate_mm", "policy_sha256")},
        scope="REPRESENTED_COORDINATE_PROXIMITY_NOT_POSE_OR_CORRESPONDENCE_CERTIFICATE")
