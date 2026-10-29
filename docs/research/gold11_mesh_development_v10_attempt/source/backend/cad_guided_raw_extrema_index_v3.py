"""Bounded extrema V3: V1's staged binary64 program on a rank-partition kd-tree.

SOURCE CANDIDATE: not qualified, integrated, or evidence of physical improvement.
The rounded row program, the sign-selected corner enclosure, the query traversal,
the raw-ID tie rules, positive-zero canonicalization and the all-or-nothing
failure boundary are copied from cad_guided_raw_extrema_index_v1 unchanged, so
every returned interval/witness must be byte-identical to V1 for the same rows.

Only the internal permutation changes. V1 sorts the whole support once along its
longest root axis, so each fixed 256-row leaf is a thin slab spanning the other
two axes; oblique queries then straddle nearly every leaf. V3 partitions
recursively: at every depth the axis with the largest remaining halved root span
is chosen (deterministic global schedule, no fitted basis, no per-node box
computation), and each node's contiguous segment is stably sorted by that axis
so its left child receives exactly the rows of its leaf slots. Leaves hold at
most 64 original rows (V1: 256), a declared V3 constant chosen on invented
physical-scale measurements; leaf boxes are still reduced from the ORIGINAL points and
parents are unions, so the V1 enclosure argument is unchanged. Correctness never
depends on the partition; only cost does. Worst-case inputs (exactly planar
normal queries, plateaus, cancellation) can still evaluate every leaf.

Construction is charged in the same abstract ledgers as V1. One index-visit
unit is one raw-ID element touched by a binding, gather, permutation write,
copy or evaluation step; one coordinate-visit unit is one scalar coordinate
read or arithmetic operand in the declared program. V1 charges its single sort
as 2 index visits per row (key gather + permutation write). V3 charges: S index
visits for the working permutation copy, then for every partitioned node
segment 2 index visits per row (key gather + permutation write), one coordinate
visit per row (the sort key) and n*ceil(log2 n) sort work, then the unchanged
6S leaf boxes. Bottom-up unions charge one node visit per internal node, as in
V1. Every counter is attempted abstract work, not hardware operations or
process RSS; a CPU observation is a separate necessary measurement. The declared
coordinate maximum is 512M (V1 helper: 256M) because partition gathers are new
charged construction work and the whole-buffer query load of a 320-family case
is estimated near 190M under this model; a caller still freezes its own cap.
No pose,
CAD, Gold, measurements, counts, file access or physical uncertainty here.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class RawExtremaBudgetsV3:
    max_raw_rows: int = 1_000_000
    max_support_rows: int = 1_000_000
    max_queries: int = 6144
    max_index_visits: int = 64_000_000
    max_row_visits: int = 64_000_000
    max_coordinate_visits: int = 512_000_000
    max_sort_work: int = 256_000_000
    max_node_visits: int = 16_000_000
    max_hash_bytes: int = 1_073_741_824
    max_report_bytes: int = 8_388_608
    max_working_bytes: int = 134_217_728


_MAXIMA = (("max_raw_rows", 1_000_000), ("max_support_rows", 1_000_000),
    ("max_queries", 6144), ("max_index_visits", 64_000_000),
    ("max_row_visits", 64_000_000), ("max_coordinate_visits", 512_000_000),
    ("max_sort_work", 256_000_000), ("max_node_visits", 16_000_000),
    ("max_hash_bytes", 1_073_741_824), ("max_report_bytes", 8_388_608),
    ("max_working_bytes", 134_217_728))
LEAF_ROWS = 64
ARITHMETIC = "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1"
PARTITION = "RANK_PARTITION_KD_TREE_GLOBAL_HALVED_ROOT_SPAN_AXIS_SCHEDULE_STABLE_SORT_LEAF64_V3"


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unavailable(reason)


def _vector(value):
    _require(type(value) is tuple and len(value) == 3
        and all(type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1e6 for x in value),
        "INVALID_PROJECTION_VECTOR")
    return np.asarray(value, dtype=np.float64)


def _staged(points, direction, origin, constant):
    # Separate ufunc results intentionally prohibit BLAS/FMA reassociation.
    delta = np.subtract(points, origin)
    terms = np.multiply(delta, direction)
    first = np.add(terms[..., 0], terms[..., 1])
    second = np.add(first, terms[..., 2])
    answer = np.add(second, constant)
    return np.where(answer == 0., 0., answer)


def _arithmetic_sentinels():
    # Necessary sentinels, not full platform/binary arithmetic attestation.
    small = np.asarray([float.fromhex("0x0.0000000000001p-1022")])
    half = np.asarray([float.fromhex("0x1p-53")])
    normal = np.asarray([float.fromhex("0x1p-1022")])
    _require(float(np.add(small, small)[0]) == float.fromhex("0x0.0000000000002p-1022")
        and float(np.multiply(normal, .5)[0]) == float.fromhex("0x0.8p-1022")
        and float(np.add(np.asarray([1.]), half)[0]) == 1.,
        "ARITHMETIC_SENTINEL_REFUSAL")


def _axis_schedule(spans, depth):
    """Deterministic split axes: largest remaining halved root span, lowest axis on ties.

    Spans are the finite root extents (may be zero). The schedule depends only on
    these three represented numbers and the depth, never on node contents, CAD,
    queries or outcomes. It is an ordering heuristic; correctness ignores it.
    """
    _require(len(spans) == 3 and all(math.isfinite(x) and x >= 0. for x in spans), "INVALID_ROOT_SPANS")
    uses, schedule = [0, 0, 0], []
    for _ in range(depth):
        best, best_value = 0, None
        for axis in range(3):
            value = float(spans[axis])*(.5**uses[axis])
            if best_value is None or value > best_value:
                best, best_value = axis, value
        uses[best] += 1
        schedule.append(best)
    return tuple(schedule)


class _Index:
    """Internal owned index; callers must clear their whole output on failure.

    The public batch wrapper below provides that failure boundary. A future
    context adapter must do so too, and account these costs alongside its own.
    This class does not authenticate arbitrary externally mutated internals.
    """
    def __init__(self, points, support_rows, budgets):
        self.valid = False
        self.stats = dict(queries=0, index_visits=0, row_visits=0,
            coordinate_visits=0, sort_work=0, node_visits=0, hash_bytes=0,
            query_rows_evaluated=0, nodes_pruned=0, leaves_evaluated=0,
            report_bytes=0, owned_bytes=0, accounted_working_peak_bytes=0)
        _require(type(budgets) is RawExtremaBudgetsV3, "INVALID_BUDGET_TYPE")
        self.cap = {name: getattr(budgets, name) for name, _ in _MAXIMA}
        _require(all(type(self.cap[name]) is int and 0 <= self.cap[name] <= maximum
            for name, maximum in _MAXIMA), "INVALID_BUDGET_VALUES")
        self._memory()
        _arithmetic_sentinels()
        _require(type(points) is np.ndarray, "INVALID_RAW_POINTS")
        shape, dtype = tuple(points.shape), points.dtype
        size = int(points.nbytes)
        _require(dtype == np.dtype("float64") and len(shape) == 2 and shape[1:] == (3,)
            and 0 < shape[0] <= self.cap["max_raw_rows"] and size == math.prod(shape)*8,
            "INVALID_RAW_POINTS")
        self._memory(size)
        blob = points.tobytes(order="C")
        _require(len(blob) == size, "CHANGED_RAW_SNAPSHOT_BYTE_COUNT")
        self.points = np.frombuffer(blob, dtype=np.float64).reshape(shape)
        _require(type(self.points) is np.ndarray and self.points.dtype == np.dtype("float64")
            and self.points.shape == shape and self.points.nbytes == size
            and 0 < len(self.points) <= self.cap["max_raw_rows"], "INVALID_OWNED_RAW_POINTS")
        self.stats["owned_bytes"] += size
        self._charge("row_visits", len(self.points))
        self._charge("coordinate_visits", 6*len(self.points))
        self._memory(32*len(self.points))
        _require(np.isfinite(self.points).all() and np.all(np.abs(self.points) <= 1e6),
            "UNSUPPORTED_RAW_COORDINATE")
        self.point_sha256 = self._digest(self.points)
        if support_rows is None:
            _require(len(self.points) <= self.cap["max_support_rows"], "SUPPORT_ROW_BUDGET")
            self._memory(8*len(self.points))
            rows = np.arange(len(self.points), dtype=np.int64)
            self.scope = "WHOLE_RAW_BUFFER"
        else:
            _require(type(support_rows) is np.ndarray, "INVALID_SUPPORT_ROWS")
            support_shape, support_dtype = tuple(support_rows.shape), support_rows.dtype
            support_size = int(support_rows.nbytes)
            _require(support_dtype == np.dtype("int64") and len(support_shape) == 1
                and 0 < support_shape[0] <= self.cap["max_support_rows"]
                and support_size == support_shape[0]*8, "INVALID_SUPPORT_ROWS")
            count = support_shape[0]
            self._memory(8*count)
            raw = support_rows.tobytes(order="C")
            _require(len(raw) == 8*count, "CHANGED_SUPPORT_SNAPSHOT_BYTE_COUNT")
            rows = np.frombuffer(raw, dtype=np.int64)
            _require(type(rows) is np.ndarray and rows.dtype == np.dtype("int64")
                and rows.shape == support_shape and rows.nbytes == support_size
                and 0 < len(rows) <= self.cap["max_support_rows"], "INVALID_OWNED_SUPPORT_ROWS")
            self.scope = "COMPLETE_SUPPLIED_RAW_SUPPORT"
        self.stats["owned_bytes"] += int(rows.nbytes)
        self._charge("index_visits", len(rows))
        self._memory(4*len(rows))
        _require(np.all(rows >= 0) and np.all(rows < len(self.points))
            and np.all(rows[1:] > rows[:-1]), "INVALID_SUPPORT_RAW_IDENTITIES")
        self.support_sha256 = self._digest(rows)
        self.count = len(rows)
        self._build(rows)
        self.valid = True

    def _charge(self, name, amount):
        _require(type(amount) is int and amount >= 0, "INVALID_WORK_CHARGE")
        self.stats[name] += amount
        _require(self.stats[name] <= self.cap["max_"+name], "BUDGET_"+name.upper())

    def _memory(self, temporary=0):
        # Fixed 64KiB covers <=32 DFS stack entries, seed tuples, small vectors
        # and scalar Python bookkeeping. This is payload accounting, not RSS.
        amount = self.stats["owned_bytes"] + self.stats["report_bytes"] + temporary + 65_536
        self.stats["accounted_working_peak_bytes"] = max(
            self.stats["accounted_working_peak_bytes"], amount)
        _require(amount <= self.cap["max_working_bytes"], "WORKING_PAYLOAD_BUDGET")

    def _digest(self, array):
        self._charge("hash_bytes", int(array.nbytes))
        return hashlib.sha256(memoryview(array).cast("B")).hexdigest()

    def _slot_range(self, node):
        """Leaf-slot interval of an implicit complete-tree node (exclusive end)."""
        depth = node.bit_length()-1
        width = self.base >> depth
        first = (node-(1 << depth))*width
        return first, first+width

    def _row_range(self, node):
        lo, hi = self._slot_range(node)
        return min(self.count, lo*LEAF_ROWS), min(self.count, hi*LEAF_ROWS)

    def _build(self, rows):
        count = len(rows)
        self._charge("index_visits", count)
        self._charge("row_visits", count)
        self._charge("coordinate_visits", 12*count)
        self._memory(64*count)
        points = self.points[rows]
        lo, hi = np.min(points, axis=0), np.max(points, axis=0)
        # The supplied row sequence is strictly increasing, so argmin/argmax's
        # first occurrence is the minimum original raw ID for each axis tie.
        seeds = [int(rows[0])]
        for axis in range(3):
            seeds.extend((int(rows[int(np.argmin(points[:, axis]))]),
                int(rows[int(np.argmax(points[:, axis]))])))
        self.seed_rows = tuple(sorted(set(seeds)))
        spans = tuple(float(x) for x in (hi-lo))
        _require(all(math.isfinite(x) and x >= 0. for x in spans), "INVALID_ROOT_SPANS")
        self.sort_axis = int(np.argmax(hi-lo))
        del points
        leaf_count = (count+LEAF_ROWS-1)//LEAF_ROWS
        base = 1 << (leaf_count-1).bit_length()
        node_count = 2*base
        depth = base.bit_length()-1
        self.base, self.node_count, self.depth = base, node_count, depth
        self.axis_schedule = _axis_schedule(spans, depth)
        # Mutable working permutation of ORIGINAL raw IDs; every partition is a
        # stable sort of one node's contiguous segment by one raw coordinate.
        # Peak temporaries per node: segment copy, keys, argsort output and the
        # permuted copy, each 8 bytes per row; the working array is 8 per row.
        self._memory(8*count)
        self._charge("index_visits", count)  # working permutation copy
        perm = rows.copy()
        for level in range(depth):
            axis = self.axis_schedule[level]
            for node in range(1 << level, 1 << (level+1)):
                start, end = self._row_range(node)
                right_start, _ = self._row_range(2*node+1)
                if end-start <= 0 or right_start >= end:
                    continue  # Empty node, or every row already belongs to the left child.
                segment_count = end-start
                self._charge("index_visits", 2*segment_count)  # key gather + permutation write
                self._charge("coordinate_visits", segment_count)
                self._charge("sort_work", segment_count*max(1, (max(2, segment_count)-1).bit_length()))
                self._memory(8*count+32*segment_count)
                segment = perm[start:end]
                order = np.argsort(self.points[segment, axis], kind="stable")
                perm[start:end] = segment[order]
                del segment, order
        self.rows = np.frombuffer(perm.tobytes(order="C"), dtype=np.int64)
        _require(len(self.rows) == count, "CHANGED_PERMUTATION_BYTE_COUNT")
        del perm
        size = node_count*(6*8+3*8)
        # The immutable permutation replaces the same-size support buffer in
        # the owned ledger; original rows remain live until _build returns.
        self._memory(size+8*count)
        self.low = np.full((node_count, 3), np.inf, dtype=np.float64)
        self.high = np.full((node_count, 3), -np.inf, dtype=np.float64)
        self.start = np.zeros(node_count, dtype=np.int64)
        self.end = np.zeros(node_count, dtype=np.int64)
        self.minimum_raw = np.full(node_count, len(self.points), dtype=np.int64)
        self.stats["owned_bytes"] += size
        for leaf in range(leaf_count):
            start, end = leaf*LEAF_ROWS, min((leaf+1)*LEAF_ROWS, count)
            index = base+leaf
            self._charge("index_visits", end-start)
            self._charge("row_visits", end-start)
            self._charge("coordinate_visits", 6*(end-start))
            self._memory(8*count+64*(end-start))
            ids = self.rows[start:end]
            points = self.points[ids]
            self.low[index], self.high[index] = np.min(points, axis=0), np.max(points, axis=0)
            self.start[index], self.end[index] = start, end
            self.minimum_raw[index] = np.min(ids)
            del ids, points
        for node in range(base-1, 0, -1):
            left, right = 2*node, 2*node+1
            self._charge("node_visits", 1)
            self.low[node] = np.minimum(self.low[left], self.low[right])
            self.high[node] = np.maximum(self.high[left], self.high[right])
            self.start[node] = self.start[left]
            self.end[node] = max(self.end[left], self.end[right])
            self.minimum_raw[node] = min(self.minimum_raw[left], self.minimum_raw[right])
        self._memory(8*count)

    def _bounds(self, node, direction, origin, constant):
        self._charge("node_visits", 1)
        lower_corner = np.where(direction >= 0., self.low[node], self.high[node])
        upper_corner = np.where(direction >= 0., self.high[node], self.low[node])
        lower = float(_staged(lower_corner, direction, origin, constant))
        upper = float(_staged(upper_corner, direction, origin, constant))
        _require(math.isfinite(lower) and math.isfinite(upper) and lower <= upper,
            "INVALID_ROUNDED_NODE_BOUND")
        return lower, upper

    def _evaluate(self, ids, direction, origin, constant):
        count = len(ids)
        self._charge("index_visits", count)
        self._charge("row_visits", count)
        self._charge("coordinate_visits", 9*count)
        self._memory(128*count)
        values = _staged(self.points[ids], direction, origin, constant)
        _require(np.isfinite(values).all(), "NONFINITE_ROUNDED_PROJECTION")
        low, high = float(np.min(values)), float(np.max(values))
        low_id = int(np.min(ids[values == low]))
        high_id = int(np.min(ids[values == high]))
        self.stats["query_rows_evaluated"] += count
        return low, high, low_id, high_id

    def query(self, direction, origin, constant):
        try:
            _require(self.valid, "INDEX_INVALID_OR_PREVIOUSLY_FAILED")
            direction, origin = _vector(direction), _vector(origin)
            _require(type(constant) in (int, float) and math.isfinite(constant) and abs(constant) <= 1e6,
                "INVALID_PROJECTION_CONSTANT")
            constant = float(constant)
            self._charge("queries", 1)
            before = dict(self.stats)
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                seeds = np.asarray(self.seed_rows, dtype=np.int64)
                low, high, low_id, high_id = self._evaluate(seeds, direction, origin, constant)
                stack = [(1, *self._bounds(1, direction, origin, constant))]
                while stack:
                    node, lower, upper = stack.pop()
                    first = int(self.minimum_raw[node])
                    no_minimum = lower > low or lower == low and first >= low_id
                    no_maximum = upper < high or upper == high and first >= high_id
                    if no_minimum and no_maximum:
                        self.stats["nodes_pruned"] += 1
                        continue
                    if node >= self.base:
                        ids = self.rows[int(self.start[node]):int(self.end[node])]
                        a, b, ai, bi = self._evaluate(ids, direction, origin, constant)
                        self.stats["leaves_evaluated"] += 1
                        if a < low or a == low and ai < low_id:
                            low, low_id = a, ai
                        if b > high or b == high and bi < high_id:
                            high, high_id = b, bi
                    else:
                        children = []
                        for child in (2*node, 2*node+1):
                            if self.end[child] > self.start[child]:
                                a, b = self._bounds(child, direction, origin, constant)
                                children.append((max(low-a, b-high), child, a, b))
                        # Most promising child first; this affects cost only.
                        children.sort(key=lambda item: (item[0], -item[1]))
                        stack.extend((child, a, b) for _, child, a, b in children)
                        _require(len(stack) <= 32, "INTERNAL_TRAVERSAL_STACK_BUDGET")
                return dict(interval=[low, high], minimum_raw_row=low_id, maximum_raw_row=high_id,
                    support_row_count=self.count,
                    work={name: self.stats[name]-before[name] for name in
                        ("row_visits", "index_visits", "coordinate_visits", "node_visits",
                         "query_rows_evaluated", "nodes_pruned", "leaves_evaluated")})
        except BaseException:
            self.valid = False
            raise


def project_raw_extrema_v3(points, projections, *, support_rows=None,
        budgets=RawExtremaBudgetsV3()):
    """All-or-nothing batch of (direction_tuple, origin_tuple, constant).

    This helper's costs do not replace a caller's aggregate workflow budgets.
    Output work is charged before operations and survives late whole refusal.
    Never describes a pruned row as absent or the extrema as a measured bend.
    """
    index, completed = None, []
    result = dict(schema_version="raw_extrema_index_v3", status="UNAVAILABLE", reasons=[],
        arithmetic=ARITHMETIC, partition=PARTITION,
        bound_scope="ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
        geometric_scope="SUPPLIED_RAW_ROW_LINEAR_EXTREMA_ONLY", accepted_registration=False,
        observed_bend_count=None, production_use_allowed=False, projections=[], statistics={})
    try:
        _require(type(budgets) is RawExtremaBudgetsV3, "INVALID_BUDGET_TYPE")
        _require(type(projections) is tuple and 0 < len(projections) <= budgets.max_queries,
            "INVALID_OR_OVER_BUDGET_PROJECTIONS")
        _require(all(type(item) is tuple and len(item) == 3 for item in projections),
            "INVALID_PROJECTION_RECORD")
        # Retain the attempted constructor ledger on a late validation/build
        # refusal; ordinary assignment after __init__ would lose that evidence.
        index = _Index.__new__(_Index)
        index.__init__(points, support_rows, budgets)
        for direction, origin, constant in projections:
            record = index.query(direction, origin, constant)
            size = sum(len(part.encode()) for part in json.JSONEncoder(
                sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(record))
            index._charge("report_bytes", size)
            _require(index.stats["report_bytes"]+8192 <= index.cap["max_report_bytes"],
                "BUDGET_REPORT_ENVELOPE_BYTES")
            index._memory()
            completed.append(record)
        result.update(status="COMPLETE", projections=completed, statistics=dict(index.stats),
            point_buffer_sha256=index.point_sha256, support_rows_sha256=index.support_sha256,
            support_scope=index.scope, support_row_count=index.count, sort_axis=index.sort_axis,
            axis_schedule=list(index.axis_schedule), tree_depth=index.depth,
            leaf_rows=LEAF_ROWS, node_slots=index.node_count,
            counter_scope="ROW_INDEX_HASH_SORT_NODE_AND_COORDINATE_CHARGES_ARE_ATTEMPTED_ABSTRACT_WORK",
            coordinate_work_model="6N_RAW_VALIDATION_12S_ROOT_EXTREMA_S_PER_PARTITION_DEPTH_6S_LEAF_BOXES_9E_QUERY_ARITHMETIC",
            index_work_model="S_BIND_S_ROOT_S_PERMUTATION_COPY_2N_PER_PARTITIONED_SEGMENT_S_LEAVES_E_QUERY_ROWS",
            sort_work_model="N_CEIL_LOG2_N_PER_PARTITIONED_NODE_SEGMENT_STABLE_SORT",
            completed_counter_scope="QUERY_ROWS_EVALUATED_AND_LEAVES_EVALUATED_INCREMENT_AFTER_SUCCESS",
            memory_scope="OWNED_NUMERIC_AND_FIXED_SCRATCH_ACCOUNTING_NOT_TOTAL_PROCESS_RSS")
        encoded = 0
        for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                allow_nan=False).iterencode(result):
            encoded += len(part.encode())
            _require(encoded <= index.cap["max_report_bytes"], "BUDGET_FINAL_REPORT_BYTES")
    except Exception as error:
        # Rebuild the envelope so a late report failure cannot retain COMPLETE,
        # raw bindings or any previously populated successful projection prefix.
        result = dict(schema_version="raw_extrema_index_v3", status="UNAVAILABLE",
            arithmetic=ARITHMETIC, partition=PARTITION, accepted_registration=False,
            observed_bend_count=None, production_use_allowed=False,
            reasons=[str(error) if type(error) is _Unavailable else "EXTREMA_HELPER_EXCEPTION"],
            discarded_projection_count=len(completed), projections=[],
            statistics={} if index is None else dict(index.stats))
    return result
