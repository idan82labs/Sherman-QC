"""Prospective two-plane contexts with shared, bounded raw extrema indexes.

V1/V2 and the standalone extrema helper remain unchanged. Only scan projections
use the explicitly staged binary64 program; CAD projections and all central
normal/pose, schedule, sign, slot and downstream admission rules are retained.
This is not bitwise legacy projection equivalence or a measured speedup.

One private registry shares the bytes-backed scan snapshot from V1 _inputs.
All index construction/query work is cumulative with the parent context work;
every parent memory admission includes every live index. Four million indexed
memberships is a NEW engineering guard, not an old support-coverage invariant.
chunk_rows still controls input validation and CAD projection chunks; indexed
scan projections use the frozen helper's fixed 256-row leaves instead.

No target import, test, integration, native or physical qualification is implied
by this source. No caller table is culled and no failed partial context survives.
"""
from dataclasses import dataclass
import itertools
import math

import numpy as np

from . import cad_guided_two_plane_context_v1 as _v1
from . import cad_guided_two_plane_context_v2 as _v2
from . import cad_guided_raw_extrema_index_v1 as _extrema
from .cad_guided_two_plane_context_v1 import (
    ContextPlaneV1, OrderedPlanePairV1, _Unavailable, _require, _vector,
    _inputs, _planes, _schedule, _family, _SLOTS, _GATES, _final_report, _json,
)


@dataclass(frozen=True)
class TwoPlaneContextBudgetsV3(_v2.TwoPlaneContextBudgetsV2):
    max_extrema_indexes: int = 129
    max_extrema_indexed_rows: int = 4_000_000
    max_extrema_node_visits: int = 16_000_000
    max_extrema_coordinate_visits: int = 256_000_000
    max_extrema_queries: int = 3072


_EXTRA_MAXIMA = (("max_extrema_indexes", 129), ("max_extrema_indexed_rows", 4_000_000),
    ("max_extrema_node_visits", 16_000_000), ("max_extrema_coordinate_visits", 256_000_000),
    ("max_extrema_queries", 3072))
_MAXIMA = _v2._MAXIMA + _EXTRA_MAXIMA
_INDEX_SCRATCH_BYTES = 65_536
_INDEX_METADATA_BYTES = 1024
_ALIAS_METADATA_BYTES = 256
_CHARGE_MAP = {"row_visits": "scan_row_visits", "index_visits": "index_visits",
    "sort_work": "index_sort_work", "hash_bytes": "hash_bytes",
    "node_visits": "extrema_node_visits", "coordinate_visits": "extrema_coordinate_visits",
    "queries": "extrema_queries"}


def _configuration(budgets):
    _require(type(budgets) is TwoPlaneContextBudgetsV3, "INVALID_BUDGET_TYPE")
    snapshot = {key: getattr(budgets, key) for key, _ in _MAXIMA}
    _require(all(type(snapshot[key]) is int and 0 <= snapshot[key] <= maximum
        for key, maximum in _MAXIMA) and snapshot["chunk_rows"] >= 3
        and snapshot["max_extrema_queries"] <= 3*snapshot["max_signed_families"],
        "INVALID_BUDGET_VALUES")
    return snapshot


def _base():
    result = _v2._base()
    result.update(schema_version="two_plane_context_v3", scan_extrema_indexes=[],
        scan_projection_arithmetic=_extrema.ARITHMETIC,
        scan_projection_bound_scope="ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
        cad_projection_arithmetic="UNCHANGED_V1_NUMPY_MATMUL_PROJECTION",
        scan_projection_chunk_scope="FIXED_EXTREMA_LEAF256_NOT_CHUNK_ROWS",
        chunk_rows_scope="INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY",
        extrema_guard_status="NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_INVARIANTS",
        extrema_membership_scope="SUM_OF_DISTINCT_INDEX_SUPPORT_SIZES_INCLUDING_WHOLE_SCAN",
        extrema_memory_scope="SHARED_RAW_OWNER_PLUS_ALL_LIVE_INDEX_PAYLOAD_UNDER_PARENT_CAP_NOT_RSS")
    return result


def _bytes_backed(array):
    """Check the bounded base chain produced by V1 _Work.own, not a flag alone."""
    current = array
    for _ in range(8):
        if type(current) is bytes:
            return True
        if type(current) is not np.ndarray or current.flags.writeable:
            return False
        current = current.base
    return False


def _digest_label(value):
    return (type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))


class _ContextWork(_v1._Work):
    """All old memory calls dynamically include the persistent index registry."""
    def __init__(self, cap):
        super().__init__(cap)
        self.registry = None
        self.registry_started = False
        self.index_stats = dict(queries=0, index_visits=0, row_visits=0,
            coordinate_visits=0, sort_work=0, node_visits=0, hash_bytes=0,
            query_rows_evaluated=0, nodes_pruned=0, leaves_evaluated=0,
            report_bytes=0, owned_bytes=0, accounted_working_peak_bytes=0)
        self.stats.update({name[4:]: 0 for name, _ in _EXTRA_MAXIMA})
        self.stats.update(extrema_completed_indexes=0, extrema_registry_hits=0,
            extrema_support_aliases=0, extrema_alias_comparison_rows=0,
            extrema_registry_metadata_bytes=0, extrema_work=self.index_stats)

    def memory(self, extra=0):
        _require(type(extra) is int and extra >= 0, "INVALID_WORKING_PAYLOAD_EXTRA")
        registry = (self.index_stats["owned_bytes"] + self.stats["extrema_registry_metadata_bytes"]
            + (_INDEX_SCRATCH_BYTES if self.registry_started else 0))
        # Called by inherited CAD/record/cache/input helpers as well as indexes.
        super().memory(registry + extra)

    def release_registry(self):
        """Break private ownership cycles after the final report decision."""
        registry, self.registry = self.registry, None
        if registry is None:
            return
        registry.failed = True
        indices = getattr(registry, "indices", None)
        if type(indices) is dict:
            for index in indices.values():
                index.valid = False
            indices.clear()
        aliases = getattr(registry, "aliases", None)
        if type(aliases) is dict:
            aliases.clear()
        registry.points = registry.work = registry.result = None


class _ContextIndex(_extrema._Index):
    """Private support-only constructor; inherited geometry/traversal is frozen."""
    def __init__(self, registry, rows, expected_sha256, index_id):
        self.valid = False
        self.registry = registry
        self.stats = registry.work.index_stats  # NEVER reset for a later index.
        self.points = registry.points
        self.point_sha256 = registry.point_sha256
        self.index_id = index_id
        self.source_rows = rows  # Already immutable and owned by the context.
        if rows is None:
            count = len(self.points)
            self._memory(8*count)
            owned = np.arange(count, dtype=np.int64)
            self.scope = "WHOLE_RAW_BUFFER"
        else:
            shape, dtype, size = tuple(rows.shape), rows.dtype, int(rows.nbytes)
            _require(dtype == np.dtype("int64") and len(shape) == 1
                and 0 < shape[0] <= len(self.points) and size == shape[0]*8,
                "INVALID_EXTREMA_SUPPORT_CAPTURE")
            count = shape[0]
            self._memory(size)
            blob = rows.tobytes(order="C")
            _require(len(blob) == size, "CHANGED_EXTREMA_SUPPORT_SNAPSHOT_BYTES")
            owned = np.frombuffer(blob, dtype=np.int64)
            _require(owned.shape == shape and owned.nbytes == size
                and 0 < len(owned) <= len(self.points), "INVALID_OWNED_EXTREMA_SUPPORT")
            self.scope = "COMPLETE_SUPPLIED_RAW_SUPPORT"
        self.stats["owned_bytes"] += int(owned.nbytes)
        self._charge("index_visits", count)
        self._memory(4*count)
        _require(np.all(owned >= 0) and np.all(owned < len(self.points))
            and np.all(owned[1:] > owned[:-1]), "INVALID_EXTREMA_SUPPORT_IDENTITIES")
        self.support_sha256 = self._digest(owned)
        _require(rows is None or self.support_sha256 == expected_sha256,
            "STALE_EXTREMA_SUPPORT_DIGEST")
        self.count = count
        self._build(owned)
        self.valid = True

    def _charge(self, name, amount):
        _require(type(amount) is int and amount >= 0 and name in _CHARGE_MAP,
            "INVALID_EXTREMA_WORK_CHARGE")
        self.stats[name] += amount
        self.registry.work.charge(_CHARGE_MAP[name], amount)

    def _memory(self, temporary=0):
        _require(type(temporary) is int and temporary >= 0, "INVALID_EXTREMA_MEMORY_CHARGE")
        local = (self.stats["owned_bytes"] + self.registry.work.stats["extrema_registry_metadata_bytes"]
            + _INDEX_SCRATCH_BYTES + temporary)
        self.stats["accounted_working_peak_bytes"] = max(
            self.stats["accounted_working_peak_bytes"], local)
        self.registry.work.memory(temporary)


class _ExtremaRegistry:
    """Private to one context: immutable raw owner, exact supports, no eviction."""
    def __init__(self, context, work, result):
        self.failed = True
        self.work, self.result = work, result
        self.indices, self.aliases = {}, {}
        points = context["scan_points"]
        _require(type(points) is np.ndarray and points.dtype == np.dtype("float64")
            and points.ndim == 2 and points.shape[1:] == (3,)
            and 3 <= len(points) <= work.cap["max_scan_rows"]
            and points.flags.c_contiguous and _bytes_backed(points),
            "EXTREMA_REQUIRES_CONTEXT_OWNED_RAW_SNAPSHOT")
        self.points = points
        self.point_sha256 = result["input_digests"]["scan_points_sha256"]
        _require(_digest_label(self.point_sha256), "INVALID_OWNED_EXTREMA_POINT_DIGEST")
        work.memory()
        _extrema._arithmetic_sentinels()
        self.failed = False

    def _metadata(self, size):
        self.work.memory(size)
        self.work.stats["extrema_registry_metadata_bytes"] += size
        self.work.memory()

    def _alias(self, rows, index):
        _require(len(self.aliases) < self.work.cap["max_scan_planes"], "EXTREMA_SUPPORT_ALIAS_BUDGET")
        self._metadata(_ALIAS_METADATA_BYTES)
        self.aliases[id(rows)] = (rows, index)  # Strong reference prevents ID reuse.
        self.work.stats["extrema_support_aliases"] += 1

    def _index(self, plane):
        rows = None if plane is None else plane["raw_vertex_rows"]
        binding = None if plane is None else plane["raw_vertex_rows_sha256"]
        if rows is not None:
            _require(type(rows) is np.ndarray and rows.dtype == np.dtype("int64")
                and rows.ndim == 1 and 0 < len(rows) <= len(self.points)
                and _bytes_backed(rows) and _digest_label(binding),
                "EXTREMA_REQUIRES_CONTEXT_OWNED_SUPPORT")
            alias = self.aliases.get(id(rows))
            if alias is not None:
                _require(alias[0] is rows and alias[1].support_sha256 == binding,
                    "CHANGED_EXTREMA_SUPPORT_ALIAS")
                self.work.stats["extrema_registry_hits"] += 1
                return alias[1]
        count = len(self.points) if rows is None else len(rows)
        key = (self.point_sha256, "WHOLE" if rows is None else "ROWS", binding, count)
        if key in self.indices:
            index = self.indices[key]
            if rows is not None:
                # Verify equal bytes once for each distinct owned alias, rather
                # than treating a matching digest as a complete set proof.
                self.work.charge("index_visits", count)
                self.work.stats["extrema_alias_comparison_rows"] += count
                with self.work.temporary(4*count):
                    _require(np.array_equal(rows, index.source_rows), "EXTREMA_SUPPORT_DIGEST_COLLISION")
                self._alias(rows, index)
            self.work.stats["extrema_registry_hits"] += 1
            return index
        self.work.charge("extrema_indexes", 1)
        self.work.charge("extrema_indexed_rows", count)
        self._metadata(_INDEX_METADATA_BYTES)
        index = _ContextIndex(self, rows, binding, len(self.indices))
        record = dict(index_id=index.index_id, point_buffer_sha256=index.point_sha256,
            support_rows_sha256=index.support_sha256, support_scope=index.scope,
            support_row_count=index.count, sort_axis=index.sort_axis,
            leaf_rows=_extrema.LEAF_ROWS, node_slots=index.node_count,
            raw_point_snapshot="SHARED_WITH_CONTEXT_INPUTS_NOT_RECOPIED_OR_REHASHED")
        self.work.record(record)
        self.indices[key] = index
        self.result["scan_extrema_indexes"].append(record)
        self.work.stats["extrema_completed_indexes"] += 1
        if rows is not None:
            self._alias(rows, index)
        return index

    def resolve(self, plane):
        try:
            _require(not self.failed, "EXTREMA_REGISTRY_PREVIOUSLY_FAILED")
            return self._index(plane)
        except BaseException:
            self.failed = True
            raise

    def project(self, index, direction, origin, constant):
        try:
            _require(not self.failed and index.registry is self,
                "EXTREMA_REGISTRY_PREVIOUSLY_FAILED_OR_FOREIGN_INDEX")
            value = index.query(tuple(float(x) for x in direction),
                tuple(float(x) for x in origin), float(constant))
            return index.index_id, value
        except BaseException:
            self.failed = True
            raise


def _projection(role, plane, direction, origin, constant, frame, context, work, result):
    if role == "cad":
        return _v1._projection(role, plane, direction, origin, constant, frame, context, work, result)
    _require(role == "scan", "INVALID_EXTREMA_PROJECTION_ROLE")
    whole = plane is None
    binding = "WHOLE" if whole else plane["raw_vertex_rows_sha256"]
    if work.registry is None:
        work.registry_started = True
        work.memory()
        # Retain a partially initialized registry for whole-context failure
        # diagnostics; all counters/owned payload already belong to work.
        registry = _ExtremaRegistry.__new__(_ExtremaRegistry)
        work.registry = registry
        registry.__init__(context, work, result)
    # Resolve every complete support BEFORE any projection-cache hit: a new
    # owned alias must pass exact row equality, not just a matching digest.
    index = work.registry.resolve(plane)
    key = (role, frame, result["input_digests"]["scan_points_sha256"], binding,
        tuple(float(x) for x in direction), tuple(float(x) for x in origin), float(constant),
        _extrema.ARITHMETIC)
    if key in work.cache:
        work.stats["projection_cache_hits"] += 1
        return work.cache[key]
    work.stats["projection_cache_misses"] += 1
    _require(work.stats["projection_cache_entries"] < work.cap["max_projection_cache_entries"],
        "PROJECTION_CACHE_BUDGET")
    index_id, extrema = work.registry.project(index, direction, origin, constant)
    record = dict(projection_id=len(result["context_projections"]), role=role,
        scope="WHOLE_SUPPLIED_BUFFER" if whole else "COMPLETE_ASSIGNED_SUPPORT", frame=frame,
        buffer_sha256=key[2], support_rows_sha256=None if whole else binding,
        projected_coordinate_origin_mm=origin.tolist(), projection_direction=direction.tolist(),
        projected_constant_mm=float(constant), interval_mm=extrema["interval"],
        minimum_witness=dict(raw_vertex_row=extrema["minimum_raw_row"]),
        maximum_witness=dict(raw_vertex_row=extrema["maximum_raw_row"]),
        support_row_count=extrema["support_row_count"], row_visits=extrema["work"]["row_visits"],
        row_visits_scope="ACTUAL_QUERY_ROW_EVALUATION_ATTEMPTS_INCLUDING_SEEDS_NOT_SUPPORT_POPULATION",
        scan_extrema_index_id=index_id, arithmetic=_extrema.ARITHMETIC,
        extrema_query_work=extrema["work"], context_not_correspondence_truth=True)
    payload = (_v1._encoded_size(key, work.cap["max_report_bytes"])
        + _v1._encoded_size(record, work.cap["max_report_bytes"]))
    work.memory(payload)
    work.stats["cache_payload_bytes"] += payload
    work.stats["projection_cache_entries"] += 1
    work.cache[key] = record["projection_id"]
    result["context_projections"].append(record)
    return record["projection_id"]


def _slots(family, source, target, origins, frames, context, work, result):
    # V1 slot arithmetic/order; only _projection dispatch is separately versioned.
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


def propose_two_plane_context_v3(scan_points, scan_triangles, cad_triangles, cad_face_ids,
        cad_source_triangle_ids, scan_planes, cad_planes, ordered_matches, *,
        scan_origin_mm, cad_origin_mm, unit_assumption, source_frame, target_frame,
        budgets=TwoPlaneContextBudgetsV3()):
    """Preserve the complete V2 schedule; all index/caller failures clear output."""
    result, work = _base(), None
    try:
        cap = _configuration(budgets)
        work = _ContextWork(cap)
        work.memory()
        _v2._global_schedule_preflight(ordered_matches, cap)
        _require(all(type(x) is str and 0 < len(x) <= 128 and x.isascii() and x.strip()
            for x in (unit_assumption, source_frame, target_frame)), "EXPLICIT_MM_AND_FRAME_DECLARATIONS_REQUIRED")
        origins = tuple(np.asarray(_vector(x)) for x in (scan_origin_mm, cad_origin_mm))
        result.update(unit="mm", unit_assumption=unit_assumption, source_frame=source_frame,
            target_frame=target_frame, source_origin_mm=origins[0].tolist(), target_origin_mm=origins[1].tolist(),
            policy=dict(_GATES), budgets=cap,
            counter_semantics=dict(
                scan_and_cad_visits="ORIGINAL_INPUT_VALIDATION_PLUS_ACTUAL_PROJECTION_AND_INDEX_BUILD_CHARGES",
                index_visits="ORIGINAL_BINDING_PLUS_INDEX_BUILD_QUERY_AND_EXACT_ALIAS_COMPARISON_WORK",
                sort_work="ONE_STABLE_SORT_PER_DISTINCT_INDEX_PLUS_UNCHANGED_CONTEXT_SORT_WORK_ABSTRACT_N_CEIL_LOG2_N",
                hash_bytes="CONTEXT_RAW_SNAPSHOT_HASHED_ONCE_PLUS_ORIGINAL_SUPPORT_AND_EXTRA_INDEX_SUPPORT_HASH_BYTES",
                working_bytes="PARENT_OWNERSHIP_TEMP_CACHE_REPORT_PLUS_ALL_LIVE_INDEX_BYTES_METADATA_AND_ONE_EXTRA_SCRATCH_NOT_RSS",
                attempted_overflow="ATTEMPTED_COUNTER_CAN_EXCEED_CAP_WITH_WHOLE_REFUSAL_BEFORE_OPERATION",
                slots="PLANNED32_PER_ORDERED_MATCH_WITH_EXPLICIT_REJECTED_OR_ABSENT_OUTCOMES",
                extrema_coordinates="NEW_INDEX_WORK_ONLY_19M_BUILD_PLUS_9E_QUERY_ABSTRACT_UNITS_NOT_ALL_SCALAR_OPERATIONS",
                extrema_memberships="DISTINCT_INDEX_ROW_POPULATIONS_INCLUDING_WHOLE_NO_EVICTION_OR_REFUNDS",
                extrema_completed="QUERY_ROWS_EVALUATED_AND_LEAVES_EVALUATED_INCREMENT_ONLY_AFTER_SUCCESS",
                extrema_payload_peak="INDEX_REGISTRY_ONLY_PARENT_COMBINED_PEAK_IS_ACCOUNTED_WORKING_PEAK_BYTES",
                extrema_report_bytes="ZERO_IN_HELPER_STATS_CONTEXT_RECORD_AND_CACHE_LEDGERS_OWN_REPORT_ACCOUNTING"))
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
        refusal = _base()
        reason = (str(error) if type(error) is _Unavailable else
            "RAW_EXTREMA_"+str(error) if type(error) is _extrema._Unavailable else "CONTEXT_HELPER_EXCEPTION")
        refusal.update(reasons=[reason],
            evidence_scope="ATTEMPTED_COUNTERS_ONLY_NO_PARTIAL_POSE_OR_GEOMETRY_REPORT",
            discarded_signed_families=len(result["families"]), discarded_candidates=len(result["candidates"]),
            statistics={} if work is None else dict(work.stats))
        _require(len(_json(refusal)) <= 8192, "INTERNAL_V3_REFUSAL_ENVELOPE_BUDGET")
        return refusal
    finally:
        if work is not None:
            work.release_registry()
