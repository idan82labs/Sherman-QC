"""Bounded extrema of a separately versioned binary64 projection program.

SOURCE CANDIDATE: not qualified, integrated, or evidence of physical improvement.
All raw rows remain eligible. A spatial hierarchy only avoids evaluating rows
whose rounded-program bounds cannot improve either extremum or its raw-ID tie.
The arithmetic is NOT legacy BLAS matmul and NOT an exact-real interval proof:
for each row, subtract origin, multiply three coordinates, then add ((0+1)+2)
and constant in separate binary64 NumPy ufunc calls. Final zero is positive zero.
Monotonicity of these same operations at sign-selected AABB corners encloses
the rounded row outputs. This statement assumes consistent IEEE binary64
round-to-nearest, gradual underflow and no reassociation/fused operations.

One stable longest-root-axis sort and fixed leaves256 bound construction cost;
correctness is independent of spatial ordering/pruning quality. Worst-case
inputs can still consume the complete query budget and refuse. Raw-row identity
is never changed by the internal permutation. No pose, CAD, Gold, measurements,
counts, file access or physical uncertainty is provided here.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class RawExtremaBudgetsV1:
    max_raw_rows: int = 1_000_000
    max_support_rows: int = 1_000_000
    max_queries: int = 6144
    max_index_visits: int = 64_000_000
    max_row_visits: int = 64_000_000
    max_coordinate_visits: int = 256_000_000
    max_sort_work: int = 64_000_000
    max_node_visits: int = 16_000_000
    max_hash_bytes: int = 1_073_741_824
    max_report_bytes: int = 8_388_608
    max_working_bytes: int = 134_217_728


_MAXIMA = (("max_raw_rows", 1_000_000), ("max_support_rows", 1_000_000),
    ("max_queries", 6144), ("max_index_visits", 64_000_000),
    ("max_row_visits", 64_000_000), ("max_coordinate_visits", 256_000_000),
    ("max_sort_work", 64_000_000), ("max_node_visits", 16_000_000),
    ("max_hash_bytes", 1_073_741_824), ("max_report_bytes", 8_388_608),
    ("max_working_bytes", 134_217_728))
LEAF_ROWS = 256
ARITHMETIC = "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1"


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
        _require(type(budgets) is RawExtremaBudgetsV1, "INVALID_BUDGET_TYPE")
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
        axis = int(np.argmax(hi-lo))
        self.sort_axis = axis
        self._charge("sort_work", count*max(1, (max(2, count)-1).bit_length()))
        self._charge("index_visits", 2*count)
        self._charge("coordinate_visits", count)
        # Exactly one stable coordinate sort; equal keys retain original raw IDs.
        order = np.argsort(points[:, axis], kind="stable")
        ordered = rows[order]
        self.rows = np.frombuffer(ordered.tobytes(order="C"), dtype=np.int64)
        # This replaces the same-size support buffer in the owned ledger. Peak
        # old/new rows, sort/permutation and numeric scratch were precharged.
        del points, order, ordered
        leaf_count = (count+LEAF_ROWS-1)//LEAF_ROWS
        base = 1 << (leaf_count-1).bit_length()
        node_count = 2*base
        self.base, self.node_count = base, node_count
        size = node_count*(6*8+3*8)
        # Original rows remain live in the constructor until _build returns;
        # the sorted immutable replacement is already live as self.rows.
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


def project_raw_extrema_v1(points, projections, *, support_rows=None,
        budgets=RawExtremaBudgetsV1()):
    """All-or-nothing batch of (direction_tuple, origin_tuple, constant).

    This helper's costs do not replace a caller's aggregate workflow budgets.
    Output work is charged before operations and survives late whole refusal.
    Never describes a pruned row as absent or the extrema as a measured bend.
    """
    index, completed = None, []
    result = dict(schema_version="raw_extrema_index_v1", status="UNAVAILABLE", reasons=[],
        arithmetic=ARITHMETIC, bound_scope="ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
        geometric_scope="SUPPLIED_RAW_ROW_LINEAR_EXTREMA_ONLY", accepted_registration=False,
        observed_bend_count=None, production_use_allowed=False, projections=[], statistics={})
    try:
        _require(type(budgets) is RawExtremaBudgetsV1, "INVALID_BUDGET_TYPE")
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
            leaf_rows=LEAF_ROWS, node_slots=index.node_count,
            counter_scope="ROW_INDEX_HASH_SORT_NODE_AND_COORDINATE_CHARGES_ARE_ATTEMPTED_ABSTRACT_WORK",
            coordinate_work_model="6N_RAW_VALIDATION_12S_ROOT_EXTREMA_S_SORT_6S_LEAF_BOXES_9E_QUERY_ARITHMETIC",
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
        result = dict(schema_version="raw_extrema_index_v1", status="UNAVAILABLE",
            arithmetic=ARITHMETIC, accepted_registration=False, observed_bend_count=None,
            production_use_allowed=False,
            reasons=[str(error) if type(error) is _Unavailable else "EXTREMA_HELPER_EXCEPTION"],
            discarded_projection_count=len(completed), projections=[],
            statistics={} if index is None else dict(index.stats))
    return result
